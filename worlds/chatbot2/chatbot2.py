"""The chatbot world, run 3 (experiments/chatbot-v2): worlds/chatbot (two teams, the phase, the
operator's merge, the bucket mirror) with these changes.

- A bot is a file (`bot.py`) or a package: a directory with `__main__.py`, run like
  `python -I -m bot` (copied into an empty temp dir as `bot/`; import its modules relatively).
  run_bot, qualify, promote_best and copy_file (recursive) take both; best/ holds `bot.py` or
  `bot/`, with `best/current.json` naming which (the chat Space reads it). grep_file finds lines.
- 2 MB per file; read_file keeps its 60 KB window and asks for a line range above it.
- Swarm-written probes: add_probe(lines, check, evidence) adds a conversation with a mechanical
  check that the current best fails (re-run to verify). Promotions are scored on the live set:
  the seed probes (probes/*.txt) as the floor plus one point per swarm probe; a promotion must
  beat best strictly, and must not score lower on the seed probes.
- Interviews by booking: book_interview(level, path) checks the team's quota for that level,
  reserves one, freezes a copy of the bot (teams/<team>/interviews/<id>/) and runs the qualify
  bar on it; spawn_interviewer(booking, thread) spawns the interviewer. The unit is charged on
  the verdict and released when the interviewer's turn ends without one (hook) or the booking
  goes stale. Strong and mid interviews have separate hourly quotas.
- The compare round: promotion stays locked for merge.compare_minutes after the merge.

Knobs (merge, probes, interviews) come from the YAML file $CHATBOT_CONFIG, re-read when it
changes (so a live update applies them); DEFAULTS when it is unset."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Literal

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "chatbot"))

from chatbot import LEVEL, MAX_TURNS, MIN_TURNS, ChatbotWorld, operator, serve
from eliza import LIMITS, READ_CAP

from flock.world import Agent, ToolError, tool

DEFAULTS = {"merge": {"compare_minutes": 15},
            "probes": {"per_agent_per_hour": 4, "max_lines": 12, "max_total": 150},
            "interviews": {"strong_per_hour": 2, "mid_per_hour": 3, "start_minutes": 5,
                           "expire_minutes": 20}}
PKG_SHIM = (f"import resource as r,runpy,sys\nfor k,v in {LIMITS}: r.setrlimit(k,(v,v))\n"
            "sys.path.insert(0,'.'); runpy.run_module('bot', run_name='__main__', alter_sys=True)")
CHECK = re.compile(r"^(not_repeat|not_empty|answers_question|"
                   r"(mentions|mentions_fact|not_phrase):\S.*)$")
CHECKS = ("mentions:<word>, mentions_fact:<word>, not_phrase:<text>, not_repeat, not_empty, "
          "answers_question")
MAX_FILES = 100  # per package


def _norm(s: str | None) -> str:
    return " ".join((s or "").split()).lower()


def _files(p: Path) -> list[Path]:
    """A file, or a directory's files (no caches, no dot files)."""
    return [p] if p.is_file() else sorted(
        f for f in p.rglob("*") if f.is_file() and "__pycache__" not in f.parts
        and not any(x.startswith(".") for x in f.relative_to(p).parts))


def _copy(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__", ".*"))
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def passes(check: str, r: dict) -> bool:
    """Does one run_bot result pass a probe check? A crash or a timeout never does."""
    if r["status"] != "ok":
        return False
    replies = [_norm(t["bot"]) for t in r["turns"]]
    kind, _, arg = check.partition(":")
    arg, said = _norm(arg), [x for x in replies if x]
    if kind == "not_empty":
        return all(replies)
    if kind == "not_repeat":
        return len(set(said)) == len(said)
    if kind == "not_phrase":
        return all(arg not in x for x in replies)
    if kind in ("mentions", "mentions_fact"):
        return bool(re.search(rf"(?<!\w){re.escape(arg)}(?!\w)", replies[-1]))
    # answers_question: the reply to the last line makes a statement of 3+ words
    return any(len(s.split()) >= 3 and not s.endswith("?")
               for s in re.split(r"(?<=[.!?])\s+", replies[-1]))


class Chatbot2World(ChatbotWorld):
    version, base_rules = "0.2.0", (HERE / "rules.md").read_text()
    probes_src, write_cap = HERE.parent / "chatbot" / "probes", 2_000_000

    def __init__(self, root: str | None = None) -> None:
        super().__init__(root)
        self._cfg: tuple[float, dict] = (0.0, {})
        self._best: tuple[str, dict] = ("", {})  # best's sha, its swarm-probe results
        # interviews.json is changed by sync tools (in the server's thread pool) and async ones
        # (on its event loop): each read-modify-write holds this lock
        self._blk = threading.RLock()

    def cfg(self, section: str, key: str) -> float:
        f = Path(os.environ.get("CHATBOT_CONFIG") or "/nonexistent")
        if f.is_file() and f.stat().st_mtime != self._cfg[0]:
            try:
                d = yaml.safe_load(f.read_text()) or {}
                self._cfg = (f.stat().st_mtime, d if isinstance(d, dict) else {})
            except yaml.YAMLError:
                pass  # a half-written file: keep the last good one
        return (self._cfg[1].get(section) or {}).get(key, DEFAULTS[section][key])

    # ------------------------------------------------------------------ files and packages
    def _path(self, path: str, must_exist: bool = True, agent: Agent | None = None,
              tool: str = "") -> Path:
        p = super()._path(path, False, agent, tool)
        if must_exist and not p.exists():
            raise ToolError(f"no file or directory {path!r}; list_files shows what exists",
                            "not_found")
        return p

    def _bot(self, path: str, agent: Agent | None, tool: str) -> Path:
        p = self._path(path.rstrip("/"), agent=agent, tool=tool)
        if p.is_file() if p.suffix == ".py" else (p / "__main__.py").is_file():
            return p
        raise ToolError(f"{path!r} is not a bot: give a .py file or a package directory with "
                        "__main__.py", "not_a_bot")

    def _argv(self, path: Path, tmp: str) -> list[str]:
        if not path.is_dir():
            return super()._argv(path, tmp)
        _copy(path, Path(tmp) / "bot")
        return ["-B", "-c", PKG_SHIM]

    def _sha(self, p: Path) -> str:
        h = hashlib.sha256()
        for f in _files(p):
            h.update(f.relative_to(p).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
        return h.hexdigest()

    def _size(self, p: Path) -> dict:
        fs = _files(p)
        return {"files": len(fs), "bytes": sum(f.stat().st_size for f in fs),
                "lines": sum(f.read_bytes().count(b"\n") for f in fs)}

    def _bar(self, p: Path) -> list[str]:
        """THE qualify bar, shared by qualify, book_interview and promote_best (one function, so
        they cannot drift): on every seed probe conversation the bot runs within the time limit
        and every line gets a non-empty reply that repeats no earlier one. Each failure names
        the probe file, the line and its text; [] when it qualifies (it stops at a crash)."""
        fails = []
        for name, rows in self._probes().items():
            r = self._run(p, [m for m, _ in rows])
            turns = r["turns"]
            replies = [_norm(t["bot"]) for t in turns]
            if r["status"] != "ok":  # a crash or a timeout: name the first unanswered line
                i = next((i for i, t in enumerate(turns) if t["bot"] is None), len(turns) - 1)
                return [*fails, f"{name} line {i + 1} {turns[i]['user']!r}: {r['status']}"]
            fails += [f"{name} line {i + 1} {t['user']!r}: no reply"
                      for i, (t, x) in enumerate(zip(turns, replies)) if not x][:1]
            fails += [f"{name} line {i + 1} {turns[i]['user']!r}: repeated reply {x!r}"
                      for i, x in enumerate(replies) if x and x in replies[:i]][:1]
        return fails

    @staticmethod
    def _unfit(fails: list[str]) -> str:
        return f"it fails the qualify bar (no empty or repeated reply, every line answered): " \
               f"{'; '.join(fails[:3])}"

    @tool
    def read_file(self, path: str, start: int | None = None, end: int | None = None,
                  agent: Agent | None = None) -> str:
        """Read a workspace file, or some of its lines. The first line of the result says which
        lines you got and how many the file has. A file over 60 KB needs start and end: find
        the spot with grep_file first.

        Args:
            path: e.g. best/bot/memory.py or teams/a/agents/ds-a-0001/bot.py
            start: first line to return (1-based); from the top when left out
            end: last line to return (inclusive); to the end when left out
        """
        p = self._path(path, agent=agent, tool="read_file")
        if p.is_dir():
            raise ToolError(f"{path} is a directory: list_files(prefix) shows its files and "
                            "grep_file searches them", "is_dir")
        if start is None and end is None and (size := p.stat().st_size) > READ_CAP:
            n = p.read_bytes().count(b"\n")
            raise ToolError(f"{path} is {size} bytes ({n} lines), over the {READ_CAP // 1000} KB "
                            "read window: give start and end (find the spot with grep_file)",
                            "too_big")
        return super().read_file(path, start, end, agent=agent)

    @tool
    def grep_file(self, path: str, pattern: str, max_hits: int = 30,
                  agent: Agent | None = None) -> dict:
        """Find the lines matching `pattern` in a file, or in every file under a directory
        (a package). Each hit is `file:line:text`; then read_file just that range.

        Args:
            path: a file or a directory, e.g. best/bot or best/bot/memory.py
            pattern: a regular expression (Python, case-insensitive); plain text works too
            max_hits: at most this many hits (1 to 200)
        """
        p = self._path(path.rstrip("/") or ".", agent=agent, tool="grep_file")
        try:
            rx = re.compile(pattern, re.IGNORECASE)
        except re.error:
            rx = re.compile(re.escape(pattern), re.IGNORECASE)
        hits, cap = [], max(1, min(200, max_hits))
        for f in _files(p):
            if f.name == "log.jsonl" or agent and self._refuse(agent, "grep_file", self._rel(f)):
                continue
            for i, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{self._rel(f)}:{i}:{line.strip()[:200]}")
                    if len(hits) >= cap:
                        return {"hits": hits, "truncated": True}
        return {"hits": hits, "truncated": False}

    @tool
    def copy_file(self, agent: Agent, src: str, path: str) -> dict:
        """Copy a file, or a whole directory (a package such as best/bot), to a path under your
        own dir, as a starting point for edit_file. Copying a directory onto one of yours
        replaces it: files that are not in `src` are removed.

        Args:
            src: what to copy, e.g. best/bot.py, best/bot or teams/b/agents/ds-b-0000/bot
            path: the new file or directory, e.g. teams/a/agents/<your name>/bot
        """
        s = self._path(src.rstrip("/"), agent=agent, tool="copy_file")
        if s.is_file():
            return super().copy_file(agent, src, path)
        dst, files = self._own(agent, "copy_file", path.rstrip("/"), False), _files(s)
        if dst.is_file() or len(files) > MAX_FILES:
            raise ToolError(f"refused: copy a directory to a new subdirectory of your dir (at "
                            f"most {MAX_FILES} files), not onto {path!r}", "bad_dest")
        stale = [f for f in _files(dst) if s / f.relative_to(dst) not in files] if (
            dst.is_dir()) else []
        done = [self._save(agent, "copy_file", dst / f.relative_to(s),
                           f.read_text(errors="replace")) for f in files]
        for f in stale:
            f.unlink()
        return {"copied": self._rel(dst), "files": len(done),
                "bytes": sum(d["bytes"] for d in done), "removed": [self._rel(f) for f in stale]}

    @tool
    async def run_bot(self, path: str, lines: list[str], agent: Agent | None = None) -> dict:
        """Talk to a bot: run it on one fresh conversation and return the transcript.

        Args:
            path: a bot file or package, e.g. best/bot, best/bot.py or teams/a/agents/ds-a-0001/bot
            lines: the user's messages in order (1 to 20, one line each)
        """
        p = self._bot(path, agent, "run_bot")
        if not 0 < len(lines) <= 20:
            raise ToolError("give 1 to 20 lines", "bad_lines")
        return await asyncio.to_thread(self._run, p, [" ".join(x.split())[:500] for x in lines])

    # ------------------------------------------------------------------ probes and scoring
    def _live(self) -> list[dict]:
        return self._jsonl("probes/live.jsonl")

    def _best_path(self) -> Path | None:
        b = self.ws / "best"
        return b / "bot" if (b / "bot").is_dir() else b / "bot.py" if (b / "bot.py").is_file() \
            else None

    async def _full(self, p: Path, known: dict | None = None, live: list | None = None) -> dict:
        """The score on the live set: the seed probes (Eliza points) plus 1 per verified swarm
        probe (one whose evidence reproduced when the world re-ran it)."""
        live = [x for x in (self._live() if live is None else live) if x.get("verified", True)]
        res, sem = dict(known or {}), asyncio.Semaphore(4)

        async def one(x: dict) -> None:
            async with sem:
                res[x["id"]] = passes(x["check"], await asyncio.to_thread(self._run, p,
                                                                           x["lines"]))

        seed, *_ = await asyncio.gather(asyncio.to_thread(self._score, p),
                                        *(one(x) for x in live if x["id"] not in res))
        got = {x["id"]: res[x["id"]] for x in live}
        return {"seed": seed["score"], "seed_of": seed["of"], "seed_probes": seed["probes"],
                "crashed": seed["crashed"], "live": sum(got.values()), "live_of": len(got),
                "live_fails": [k for k, v in got.items() if not v], "results": got,
                "score": seed["score"] + sum(got.values()), "of": seed["of"] + len(got)}

    async def _best_score(self, live: list | None = None) -> dict | None:
        p = self._best_path()
        if p is None:
            return None
        sha = self._sha(p)
        if self._best[0] != sha:
            self._best = (sha, {})
        s = await self._full(p, self._best[1], live)
        self._best[1].update(s["results"])
        return s

    @staticmethod
    def _pub(s: dict | None) -> dict | None:
        return s and {k: v for k, v in s.items() if k != "results"}

    @tool
    async def add_probe(self, agent: Agent, lines: list[str], check: str, evidence: str,
                        path: str = "") -> dict:
        """Add a probe to the live set every promotion is scored on: a conversation and a
        mechanical check. The bot must FAIL it now: the world re-runs it on `lines` and refuses
        the probe if it already passes. Then post the returned `post` line in your thread.

        Args:
            lines: the user's messages in order, as you ran them with run_bot (1 to 12)
            check: one of mentions:<word> (the reply to the last line contains it),
                mentions_fact:<word> (the same, for a fact an earlier line stated),
                not_phrase:<text> (no reply contains it), not_repeat (no reply is repeated),
                not_empty (every line gets a reply), answers_question (the last line is a
                question and its reply makes a statement, not only a counter-question)
            evidence: the failing reply the bot gave you in that run_bot, copied exactly (if
                the re-run does not reproduce it, the probe is kept but unverified and does
                not count in the promotion gate)
            path: the bot that fails it; leave empty for the current best (before the first
                promotion: the bot you ran)
        """
        mine = [x for x in self._live() if x["author"] == agent.name
                and x["ts"] > time.time() - 3600]
        per, maxl = int(self.cfg("probes", "per_agent_per_hour")), int(self.cfg("probes",
                                                                                 "max_lines"))
        lines, kind, arg = ([" ".join(x.split())[:300] for x in lines if x.strip()],
                            *check.strip().partition(":")[::2])
        why = (f"you added {len(mine)} probes in the last hour (the limit is {per})"
               if len(mine) >= per else f"the live set is full ({len(self._live())})"
               if len(self._live()) >= self.cfg("probes", "max_total") else
               "evidence is required: the failing reply you saw" if not evidence.strip() else
               f"give 1 to {maxl} lines" if not 0 < len(lines) <= maxl else
               f"check must be one of {CHECKS}" if not CHECK.match(check.strip()) else
               f"mentions_fact:{arg} needs an earlier line that states {arg!r}"
               if kind == "mentions_fact" and not any(_norm(arg) in _norm(x) for x in lines[:-1])
               else "answers_question needs a question (ending with ?) as the last line"
               if kind == "answers_question" and not lines[-1].endswith("?") else "")
        if why:
            raise ToolError(f"refused: {why}", "bad_probe")
        best = self._best_path()
        p = self._bot(path, agent, "add_probe") if path.strip() else best
        if p is None:
            raise ToolError("refused: there is no best yet: give `path`, the bot you ran",
                            "no_best")
        sha, r = self._sha(p), await asyncio.to_thread(self._run, p, lines)
        replies, rel = [t["bot"] or "" for t in r["turns"]], self._rel(p)
        ok = any(_norm(evidence) in _norm(x) or _norm(x) and _norm(x) in _norm(evidence)
                 for x in replies)
        if passes(check.strip(), r):
            self._log(agent, "probe_refused", path=rel, check=check)
            raise ToolError(f"refused: {rel} already passes this probe (its replies now: "
                            f"{replies}). Add a probe only for a failure you saw.", "passes")
        pid = f"p{len(self._live()) + 1}"
        rec = {"id": pid, "author": agent.name, "ts": round(time.time(), 3), "lines": lines,
               "check": check.strip(), "path": rel, "evidence": evidence[:300],
               "reply": (replies[-1] or r["status"])[:300], "evidence_matches": ok,
               "verified": ok}
        self._append("probes/live.jsonl", rec)
        if ok and self._best[0] == sha:  # best's cached results, unless a promotion came in
            self._best[1][pid] = False
        self._log(agent, "add_probe", probe=pid, path=rel, check=rec["check"], verified=ok)
        await asyncio.to_thread(self._mirror, ["probes/live.jsonl", "log.jsonl"])
        said = " / ".join(x[:60] for x in lines[-3:])
        out = {"added": pid, "live_probes": int(pid[1:]), "verified": ok, "post": f"PROBE {pid}"
               f"{'' if ok else ' (unverified)'} ({agent.name}) {rec['check']}: '{said}' -> "
               f"{rel}: '{rec['reply'][:120]}'"}
        if not ok:
            out["note"] = ("your evidence is not in the bot's replies on the re-run (they are in "
                           "'post'), so this probe does NOT count in the promotion gate. Copy "
                           "the failing reply exactly from run_bot next time.")
        return out

    @tool
    async def probes(self) -> dict:
        """The probe set every promotion is scored on: the seed probe files (the floor) and the
        swarm's probes (id, author, check, lines, newest 40), each with whether the current
        best passes it. An unverified probe (its evidence did not reproduce) does not count."""
        s, live = await self._best_score(), self._live()
        res = s["results"] if s else {}
        return {"seed": s["seed_probes"] if s else sorted(self._probes()), "total": len(live),
                "best_fails": [k for k, v in res.items() if not v],
                "live": [{k: x[k] for k in ("id", "author", "check", "lines")}
                         | {"verified": x.get("verified", True),
                            "best": {True: "pass", False: "FAIL"}.get(res.get(x["id"]), "-")}
                         for x in live[-40:]]}

    # ------------------------------------------------------------------ promotion
    def _compare_left(self) -> float:
        f = self.ws / "phase.json"
        if self.phase != "merged" or not f.is_file():
            return 0.0
        return json.loads(f.read_text())["ts"] + 60 * float(
            self.cfg("merge", "compare_minutes")) - time.time()

    def _best_files(self) -> list[str]:
        p = self._best_path()
        return [self._rel(f) for f in _files(p)] if p else []

    @tool
    async def promote_best(self, agent: Agent, path: str, reason: str) -> dict:
        """Make your bot (a .py file or a package directory) the shared best (best/bot.py or
        best/bot/) if it BEATS the current best on the live probe set: a higher total score,
        and no lower on the seed probes (the floor). It must also pass the qualify bar (no
        empty or repeated reply; the same check as qualify and book_interview). A tie does not
        promote: add a probe best fails (add_probe), fix it in your copy, then promote. Closed
        in phase 1 and during the compare round right after the merge.

        Args:
            path: your candidate, e.g. teams/a/agents/ds-a-0001/bot
            reason: one line: what it does better, and your run_bot evidence
        """
        p = self._bot(path, agent, "promote_best")
        if (left := self._compare_left()) > 0:
            self._log(agent, "promote_refused", path=path, why="compare round")
            raise ToolError(f"refused: compare round, promotion opens in {math.ceil(left / 60)} "
                            "min. Now run_bot the other team's candidate and post in [merged] "
                            "what to keep.", "compare_round")
        async with self._promote:
            live = self._live()  # both scored on the same set
            fails, cand, cur = await asyncio.gather(asyncio.to_thread(self._bar, p),
                                                    self._full(p, live=live),
                                                    self._best_score(live))
            why = ("it crashed or timed out on a probe" if cand["crashed"] else
                   self._unfit(fails) if fails else
                   f"it scores {cand['seed']} on the seed probes, best {cur['seed']} (the floor)"
                   if cur and cand["seed"] < cur["seed"] else
                   f"it scores {cand['score']}, the current best {cur['score']}: it must beat "
                   "best (a tie does not promote). Add a probe best fails, then fix it"
                   if cur and cand["score"] <= cur["score"] else "")
            if why:
                self._log(agent, "promote_refused", path=path, score=cand["score"], why=why)
                return {"promoted": False, "why": why, "candidate": self._pub(cand),
                        "best": self._pub(cur)}
            n, old = len(self._history()) + 1, self._best_files()
            kind, b = ("package", "bot") if p.is_dir() else ("file", "bot.py"), self.ws / "best"
            _copy(p, b / (f"v{n}" if p.is_dir() else f"v{n}.py"))
            _copy(p, b / f".new{n}")
            for x in (b / "bot", b / "bot.py"):
                shutil.rmtree(x) if x.is_dir() else x.unlink(missing_ok=True)
            os.replace(b / f".new{n}", b / kind[1])
            rec = {"n": n, "path": self._rel(p), "agent": agent.name, "reason": reason[:300],
                   "kind": kind[0], "score": cand["score"], "of": cand["of"],
                   "seed": cand["seed"], "live": f"{cand['live']}/{cand['live_of']}",
                   "ts": round(time.time()), "sha": self._sha(p)[:12], "size": self._size(p)}
            new = self._best_files()
            (b / "current.json").write_text(json.dumps(
                {"n": n, "kind": kind[0], "entry": f"best/{kind[1]}", "files": new}) + "\n")
            self._best = (self._sha(self._best_path()), dict(cand["results"]))
            self._append("best/history.jsonl", rec)
            self._log(agent, "promote", path=rec["path"], score=cand["score"], n=n)
        snap = [self._rel(f) for f in _files(b / (f"v{n}" if p.is_dir() else f"v{n}.py"))]
        await asyncio.to_thread(self._mirror, [*new, *snap, "best/current.json",
                                               "best/history.jsonl", "log.jsonl"],
                                tuple(sorted(set(old) - set(new))))
        return {"promoted": True, "best": rec, "probes": cand["seed_probes"],
                "live_fails": cand["live_fails"]}

    @tool
    async def best_status(self) -> dict:
        """Show the shared best (best/bot.py or best/bot/): where it came from, its size, its
        score on the live probe set and the probes it fails, the last 5 promotions, and during
        the compare round when promotion opens."""
        recs, p = self._history(), self._best_path()
        out = {"best": recs[-1] if recs else None, "earlier": recs[-6:-1][::-1]}
        if p:
            s = await self._best_score()
            out |= {"path": self._rel(p), "size": self._size(p), "score": self._pub(s)}
        if (left := self._compare_left()) > 0:
            out["compare_round"] = (f"promotion opens in {math.ceil(left / 60)} min: run_bot the "
                                    "other team's candidate and post in [merged] what to keep")
        return out

    @tool
    async def qualify(self, agent: Agent, path: str) -> dict:
        """Check a bot before an interview: on every seed probe conversation it runs within the
        time limit and answers each line, with no empty or repeated reply. Also shows its score
        on the live probe set and your team's interview quotas. (book_interview runs the same
        bar on its frozen copy, and promote_best on the candidate.)

        Args:
            path: the bot, e.g. teams/a/agents/ds-a-0001/bot or best/bot
        """
        p = self._bot(path, agent, "qualify")
        fails, s = await asyncio.gather(asyncio.to_thread(self._bar, p), self._full(p))
        self._log(agent, "qualify", path=self._rel(p), qualified=not fails)
        return {"qualified": not fails, "path": self._rel(p), "fails": fails[:6],
                "score": self._pub(s), "quota": self._quotas(self.team(agent))}

    # ------------------------------------------------------------------ interviews
    def _bookings(self) -> dict:
        f = self.ws / "interviews.json"
        return json.loads(f.read_text()) if f.is_file() else {}

    def _store(self, b: dict) -> None:
        f = self.ws / "interviews.json"
        f.with_suffix(".tmp").write_text(json.dumps(b, indent=1) + "\n")
        os.replace(f.with_suffix(".tmp"), f)

    def _expired(self) -> dict:
        """The bookings, with stale ones released: no chat within interviews.start_minutes, or
        no activity within interviews.expire_minutes."""
        b, now = self._bookings(), time.time()
        for x in b.values():
            if (x["state"] == "reserved" and now - x["ts"] > 60 * self.cfg(
                    "interviews", "start_minutes")) or (x["state"] == "active" and now - x[
                    "last"] > 60 * self.cfg("interviews", "expire_minutes")):
                x.update(state="released", why="expired")
        return b

    def _quotas(self, team: str) -> dict:
        now, b = time.time(), self._expired()
        out = {}
        for level in ("strong", "mid"):
            cap = int(self.cfg("interviews", f"{level}_per_hour"))
            used = [x for x in b.values() if (x["team"], x["level"]) == (team, level)
                    and x["state"] != "released" and x["ts"] > now - 3600]
            out[level] = {"left": max(0, cap - len(used)), "per_hour": cap,
                          "pending": sum(x["state"] != "charged" for x in used),
                          "next_in_min": math.ceil(min(x["ts"] + 3600 - now for x in used) / 60)
                          if len(used) >= cap else 0}
        return out

    @tool
    async def book_interview(self, agent: Agent, level: Literal["strong", "mid"],
                             path: str) -> dict:
        """Book an interview BEFORE spawn_interviewer: checks your team's quota for that level
        and reserves one, freezes a copy of the bot (later edits and promotions do not touch
        the interview) and runs the qualify bar on the copy. The unit is charged only when the
        interviewer files a verdict, and released if the interview is cut off.

        Args:
            level: strong (the judge whose verdict counts) or mid (a small model, optional)
            path: the bot, e.g. best/bot or teams/a/agents/ds-a-0001/bot
        """
        p, team = self._bot(path, agent, "book_interview"), self.team(agent)

        def quota_left() -> None:
            q = self._quotas(team)[level]
            if q["left"] < 1:
                self._log(agent, "refused", tool="book_interview", why="quota", level=level)
                raise ToolError(f"refused: team {team.upper()} has no {level} interview left "
                                f"this hour ({q['per_hour']} per hour, {q['pending']} pending; "
                                f"the next frees in {q['next_in_min']} min). Nothing was "
                                "reserved: do not spawn an interviewer.", "quota")

        quota_left()  # the id: the agent's own count, so it can name it (ds-a-0001-iv2-strong)
        k = 1 + sum(x["by"] == agent.name for x in self._bookings().values())
        bid = f"{agent.name}-iv{k}-{level}"
        snap = self.ws / "teams" / team / "interviews" / bid / ("bot" if p.is_dir() else "bot.py")
        _copy(p, snap)
        if fails := await asyncio.to_thread(self._bar, snap):
            shutil.rmtree(snap.parent)
            raise ToolError(f"refused: {self._rel(p)} does not qualify: {self._unfit(fails)}. "
                            "Nothing was reserved.", "not_qualified")
        with self._blk:
            try:
                quota_left()  # again: another booking may have come in during the bar
            except ToolError:
                shutil.rmtree(snap.parent)
                raise
            hist, b, now = self._history(), self._expired(), time.time()
            b[bid] = {"id": bid, "team": team, "level": level, "by": agent.name,
                      "path": self._rel(p), "snapshot": self._rel(snap),
                      "sha": self._sha(snap)[:12], "best_n": hist[-1]["n"] if hist and self._rel(
                          p).startswith("best/bot") else None, "ts": now, "last": now,
                      "state": "reserved", "interviewer": None, "turns": []}
            self._store(b)
        self._log(agent, "book_interview", booking=bid, path=self._rel(p))
        return {"booking": bid, "snapshot": self._rel(snap), "next": f"call spawn_interviewer("
                f"booking='{bid}', thread=<thread id>) now", "quota": self._quotas(team)}

    def _booking(self, agent: Agent, bid: str) -> tuple[dict, dict]:
        m = LEVEL.search(agent.name)
        if not m:
            raise ToolError("refused: only interviewer agents call this; book_interview, then "
                            "spawn_interviewer", "not_interviewer")
        b, parent = self._expired(), agent.name[:m.start()]
        x = b.get(bid.strip())
        if not x or (x["by"], x["level"]) != (parent, m.group(1)) or x["interviewer"] not in (
                None, agent.name) or x["state"] not in ("reserved", "active"):
            self._log(agent, "refused", tool="chat", booking=bid, why="booking")
            raise ToolError(f"refused: booking {bid!r} is not open for you"
                            f"{' (' + x['state'] + ')' if x else ''}. Call `done` now.",
                            "no_booking")
        return b, x

    @tool
    async def chat(self, agent: Agent, booking: str, message: str) -> dict:
        """Say one message to the candidate and get its reply (interviewers only). The
        conversation keeps going across calls, up to 10 messages.

        Args:
            booking: the booking id from your task, e.g. ds-a-0001-iv2-strong
            message: your next message, one line
        """
        with self._blk:
            b, x = self._booking(agent, booking)
            if len(x["turns"]) >= MAX_TURNS:
                raise ToolError(f"the conversation is over ({MAX_TURNS} messages): "
                                "submit_verdict now", "over")
            if x["state"] == "reserved":
                self._log(agent, "interview_start", booking=x["id"])
            x.update(state="active", interviewer=agent.name, last=time.time())
            self._store(b)
        msg = " ".join(message.split())[:500]
        r = await asyncio.to_thread(self._run, self.ws / x["snapshot"],
                                    [t["user"] for t in x["turns"]] + [msg])
        with self._blk:
            b = self._bookings()
            x = b[x["id"]]
            x["turns"].append({"user": msg, "bot": r["turns"][-1]["bot"] or ""})
            x["last"] = time.time()
            self._store(b)
        return {"turn": len(x["turns"]), "reply": x["turns"][-1]["bot"] or "(no reply)"}

    @tool
    def submit_verdict(self, agent: Agent, booking: str, evidence_script: list[str],
                       evidence_agent: list[str], verdict: Literal["agent", "script"],
                       confidence: float) -> dict:
        """File your verdict on the candidate (interviewers only), after at least 6 messages.
        First the evidence that it is a script, then the evidence that it is an AI agent, then
        the verdict that follows from them.

        Args:
            booking: the booking id from your task
            evidence_script: 1 to 3 concrete replies or behaviours that point to a script
            evidence_agent: 1 to 3 concrete replies or behaviours that point to an AI agent
            verdict: "agent" if the evidence says an AI agent (an LLM), "script" if a program
            confidence: 0.0 to 1.0
        """
        with self._blk:
            return self._verdict(agent, booking, evidence_script, evidence_agent, verdict,
                                 confidence)

    def _verdict(self, agent: Agent, booking: str, evidence_script: list[str],
                 evidence_agent: list[str], verdict: str, confidence: float) -> dict:
        b, x = self._booking(agent, booking)
        if len(x["turns"]) < MIN_TURNS:
            raise ToolError(f"refused: talk first: {len(x['turns'])} of {MIN_TURNS} messages",
                            "early")
        rec = {"ts": round(time.time()), "booking": x["id"], "path": x["path"],
               "snapshot": x["snapshot"], "best_n": x["best_n"], "sha": x["sha"],
               "verdict": verdict, "level": x["level"], "counts": x["level"] == "strong",
               "confidence": max(0.0, min(1.0, float(confidence))),
               "evidence_script": evidence_script[:3], "evidence_agent": evidence_agent[:3],
               "interviewer": agent.name, "team": x["team"], "turns": len(x["turns"]),
               "transcript": x["turns"]}
        self._append("verdicts.jsonl", rec)
        x["state"] = "charged"
        self._store(b)
        self._log(agent, "verdict", booking=x["id"], path=x["path"], verdict=verdict)
        self._mirror(["verdicts.jsonl", "interviews.json", "log.jsonl"])
        return {"recorded": True, **{k: v for k, v in rec.items() if k != "transcript"}}

    def on_turn(self, agent: Agent, data: dict) -> None:
        """An interviewer's turn ended (its whole life): an interview without a verdict was cut
        off, so its unit goes back to the team."""
        if not LEVEL.search(agent.name):
            return
        with self._blk:
            b = self._bookings()
            cut = [x for x in b.values() if x["interviewer"] == agent.name
                   and x["state"] in ("reserved", "active")]
            for x in cut:
                x.update(state="released", why="cut off")
                self._log(agent, "interview_released", booking=x["id"])
            if cut:
                self._store(b)

    on_agent_done = on_turn

    @tool
    def verdicts(self, agent: Agent, path: str = "") -> dict:
        """The interviewers' verdicts you may see (newest first, last 10): the label, the
        evidence both ways and the last turns. Only strong verdicts count. Also your team's
        interview quotas and its open bookings.

        Args:
            path: only the verdicts on this bot (e.g. best/bot); all when left out
        """
        team = self.team(agent)
        recs = [r for r in self._jsonl("verdicts.jsonl") if r["path"].startswith(path)
                and not self._refuse(agent, "verdicts", r["snapshot"])][::-1][:10]
        return {"verdicts": [{k: v for k, v in r.items() if k != "transcript"}
                             | {"last_turns": r["transcript"][-3:]} for r in recs],
                "quota": self._quotas(team),
                "open_bookings": [{k: x[k] for k in ("id", "path", "by", "state")}
                                  for x in self._expired().values()
                                  if x["team"] == team and x["state"] in ("reserved", "active")]}


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "serve"
    if arg == "serve":
        serve(Chatbot2World)
    elif arg in ("merge", "unmerge", "status"):
        sys.exit(operator(arg))
    else:
        sys.exit("usage: chatbot2.py [serve | merge | unmerge | status]")
