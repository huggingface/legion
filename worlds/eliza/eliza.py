"""The Eliza world (experiments/eliza): agents write ELIZA-style bots in a shared workspace
`$ELIZA_WS` (default ws/ here): agents/<agent>/ (writable only by that agent, by X-Flock-Agent),
best/ (bot.py, v<n>.py, history.jsonl), probes/ (seeded from probes/ here), log.jsonl. Writes and
edits are atomic, and a .py change that does not compile is refused. Bots run as
`python -I bot.py` in an empty temp dir under rlimits and a timeout: not a sandbox."""

import asyncio
import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from flock.world import Agent, ToolError, World, tool

HERE = Path(__file__).resolve().parent
TIMEOUT_S, MEM, OUT_CAP, READ_CAP, WRITE_CAP = 5, 256 << 20, 64 << 10, 60_000, 200_000
LIMITS = f"((r.RLIMIT_AS,{MEM}),(r.RLIMIT_FSIZE,{OUT_CAP}),(r.RLIMIT_CPU,{TIMEOUT_S}))"
COMPILE = ("import py_compile as c,sys\ntry: c.compile(sys.argv[1], sys.argv[2], sys.argv[3], True)"
           "\nexcept c.PyCompileError as e: print(e.msg); sys.exit(1)")
SHIM = (f"import resource as r,runpy,sys\nfor k,v in {LIMITS}: r.setrlimit(k,(v,v))\n"
        "sys.argv=sys.argv[1:]; runpy.run_path(sys.argv[0], run_name='__main__')")


class ElizaWorld(World):
    name, version, description = "eliza", "0.1.0", (HERE / "rules.md").read_text()
    dirs, probes_src = ("agents", "best", "probes"), HERE / "probes"
    write_cap = WRITE_CAP  # bytes per file

    def __init__(self, root: str | None = None) -> None:
        self.ws = Path(root or os.environ.get("ELIZA_WS", "ws")).resolve()
        for d in self.dirs:
            (self.ws / d).mkdir(parents=True, exist_ok=True)
        if not any((self.ws / "probes").glob("*.txt")):
            shutil.copytree(self.probes_src, self.ws / "probes", dirs_exist_ok=True)
        self._promote = asyncio.Lock()  # one promotion at a time in this process ...

    @contextlib.asynccontextmanager
    async def _promotion(self):
        """THE promotion critical section: one promotion at a time per workspace, across
        processes too (a world replaced on re-apply runs next to the old one for a moment), by
        an asyncio lock plus an exclusive file lock on .promote.lock in the workspace. Inside it the caller
        re-reads the current best, scores, compares and commits, so a promotion queued behind
        another is compared against the NEW best."""
        async with self._promote:
            f = open(self.ws / ".promote.lock", "a")  # noqa: SIM115, ASYNC230 (tiny, closed below)
            try:
                await asyncio.to_thread(fcntl.flock, f, fcntl.LOCK_EX)
                yield
            finally:
                fcntl.flock(f, fcntl.LOCK_UN)
                f.close()

    def _home(self, agent: Agent) -> Path:
        """The one directory `agent` may write under."""
        return self.ws / "agents" / agent.name

    def _refuse(self, agent: Agent, tool: str, rel: str) -> str | None:
        """Why `agent` may not use `tool` on the workspace path `rel`, or None (here: never)."""
        return None

    def _promoted(self, rec: dict) -> None:
        """Called after each promotion, outside the promotion lock (here: nothing)."""

    def _path(self, path: str, must_exist: bool = True, agent: Agent | None = None,
              tool: str = "") -> Path:
        p = (self.ws / path.strip().lstrip("/")).resolve()
        if p != self.ws and self.ws not in p.parents:
            raise ToolError(f"{path!r} is outside the workspace", "bad_path")
        if agent is not None and (why := self._refuse(agent, tool, p.relative_to(self.ws)
                                                      .as_posix())):
            self._log(agent, "refused", tool=tool, path=path)
            raise ToolError(why, "refused")
        if must_exist and not p.is_file():
            raise ToolError(f"no file {path!r}; list_files shows what exists", "not_found")
        return p

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.ws).as_posix()

    def _append(self, rel: str, rec: dict) -> None:
        with open(self.ws / rel, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _log(self, agent: Agent, action: str, **data: object) -> None:
        self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": agent.name, "turn":
                                   agent.turn, "req": agent.request_id, "action": action, **data})

    def _argv(self, path: Path, tmp: str) -> list[str]:
        """The interpreter's arguments after `-I` that run the bot at `path` in `tmp`."""
        return ["-c", SHIM, str(path)]

    def _run(self, path: Path, lines: list[str]) -> dict:
        """One conversation: each line with its reply line, the exit status, stderr's tail."""
        with tempfile.TemporaryDirectory() as tmp:
            out, err = Path(tmp) / ".out", Path(tmp) / ".err"
            with open(out, "wb") as fo, open(err, "wb") as fe:
                # TMPDIR: a bot's own temp files stay in its run dir (in a container /tmp is
                # RAM that counts against the job's memory)
                proc = subprocess.Popen([sys.executable, "-I", *self._argv(path, tmp)], cwd=tmp,
                                        env={"TMPDIR": tmp}, stdin=subprocess.PIPE, stdout=fo, stderr=fe,
                                        start_new_session=True)
                try:
                    proc.communicate("".join(x + "\n" for x in lines).encode(), TIMEOUT_S)
                    status = f"exit {proc.returncode}" if proc.returncode else "ok"
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, 9)
                    proc.wait()
                    status = f"timeout after {TIMEOUT_S} s"
            replies = out.read_text(errors="replace").splitlines()
            bots = replies + [None] * len(lines)
            return {"status": status, "stderr": err.read_text(errors="replace")[-400:],
                    "extra_lines": max(0, len(replies) - len(lines)),
                    "turns": [{"user": u, "bot": r} for u, r in zip(lines, bots)]}

    def _probes(self) -> dict[str, list[tuple[str, list[str]]]]:
        """probes/*.txt: a user line per line, optionally ` => kw1|kw2` (the reply holds one;
        `!kw`: the reply must not hold it)."""
        return {p.name: [(m.strip(), [k.strip().lower() for k in kws.split("|") if k.strip()])
                         for m, _, kws in (x.partition(" => ") for x in p.read_text().splitlines())
                         if m.strip() and not m.startswith("#")]
                for p in sorted((self.ws / "probes").glob("*.txt"))}

    def _score(self, path: Path) -> dict:
        """1 point per probe file run without a crash, per non-empty reply, per keyword hit."""
        got, total, report, crashed = 0, 0, {}, False
        for name, rows in self._probes().items():
            r = self._run(path, [m for m, _ in rows])
            crashed |= r["status"] != "ok"
            fails = [] if r["status"] == "ok" else [r["status"]]
            for (msg, kws), t in zip(rows, r["turns"]):
                reply = (t["bot"] or "").strip().lower()
                pos, neg = [k for k in kws if k[0] != "!"], [k[1:] for k in kws if k[0] == "!"]
                bad = [k for k in neg if k in reply]
                hit = bool(kws) and (not pos or any(k in reply for k in pos)) and not bad
                got += bool(reply) + hit
                if not reply or (kws and not hit):
                    fails.append(f"{msg!r}: no reply" if not reply else f"{msg!r}: says {bad[0]!r}"
                                 if bad else f"{msg!r}: none of {pos}")
            got += r["status"] == "ok"
            total += 1 + len(rows) + sum(1 for _, k in rows if k)
            report[name] = "GREEN" if not fails else "RED: " + "; ".join(fails[:3])
        return {"score": got, "of": total, "probes": report, "crashed": crashed}

    def _history(self) -> list[dict]:
        hist = self.ws / "best" / "history.jsonl"
        return [json.loads(x) for x in hist.read_text().splitlines()] if hist.is_file() else []

    def _own(self, agent: Agent, tool: str, path: str, must_exist: bool) -> Path:
        """`path` resolved, if it is under the agent's home (by flock's X-Flock-Agent)."""
        p, home = self._path(path, must_exist=False), self._home(agent)
        if home not in p.parents:
            self._log(agent, "refused", tool=tool, path=path)
            raise ToolError(f"refused: you ({agent.name}) may write only under {self._rel(home)}"
                            f"/, not {path!r}. best/bot.py changes by promote_best.",
                            "not_yours")
        if must_exist and not p.is_file():
            raise ToolError(f"no file {path!r}; create it with write_file or copy_file",
                            "not_found")
        return p

    def _save(self, agent: Agent, tool: str, p: Path, text: str) -> dict:
        """Write `text` to `p` atomically (temp file + rename). A .py file must compile first:
        on a syntax error the file keeps its previous content and the agent reads the error."""
        rel, data = self._rel(p), text.encode()
        before = p.stat().st_size if p.is_file() else 0
        if len(data) > self.write_cap:
            raise ToolError(f"refused: {len(data)} bytes, the limit is {self.write_cap}",
                            "too_big")
        p.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / p.name
            src.write_bytes(data)
            if p.suffix == ".py":
                try:
                    r = subprocess.run([sys.executable, "-I", "-c", COMPILE, str(src),
                                        str(src) + "c", rel], capture_output=True, text=True,
                                       timeout=10, env={}, check=False)
                    err = (r.stdout or r.stderr).strip() if r.returncode else ""
                except subprocess.TimeoutExpired:
                    err = "compile check timed out"
                if err:
                    self._log(agent, "syntax_error", tool=tool, path=rel, bytes_before=before,
                              bytes_after=len(data))
                    raise ToolError(f"{rel} NOT changed (it keeps its previous content): "
                                    f"syntax error\n{err[-1200:]}", "syntax_error")
            tmp_dst = p.with_name(f".{p.name}.{os.getpid()}.{threading.get_ident()}.tmp")
            shutil.copyfile(src, tmp_dst)
            os.replace(tmp_dst, p)
        self._log(agent, tool, path=rel, bytes_before=before, bytes_after=len(data))
        return {"written": rel, "bytes": len(data), "lines": text.count("\n") + 1}

    @staticmethod
    def _unique(text: str, needle: str, what: str, path: str) -> int:
        """The index of `needle` in `text` if it occurs exactly once, else a refusal."""
        n = text.count(needle) if needle else 0
        if n != 1:
            hint = ("copy it exactly from read_file" if n == 0 else
                    "add neighbouring lines to `" + what + "` so it matches one place")
            raise ToolError(f"refused: `{what}` occurs {n} times in {path}; it must occur "
                            f"exactly once ({hint}). The file is unchanged.", f"matches_{n}")
        return text.index(needle)

    @tool
    def read_file(self, path: str, start: int | None = None, end: int | None = None,
                  agent: Agent | None = None) -> str:
        """Read a workspace file (any agent's, best/bot.py, probes/*.txt), or some of its lines.
        The first line of the result says which lines you got and how many the file has.

        Args:
            path: e.g. agents/ds-0001/bot.py or best/bot.py
            start: first line to return (1-based); from the top when left out
            end: last line to return (inclusive); to the end when left out
        """
        p = self._path(path, agent=agent, tool="read_file")
        lines = p.read_text(errors="replace").splitlines(keepends=True)
        a, b = max(1, start or 1), min(len(lines), end or len(lines))
        text = "".join(lines[a - 1:b])
        if len(text) > READ_CAP:
            text = text[:READ_CAP] + "\n[... truncated: read the rest with start/end]\n"
        return f"# {path}: lines {a}-{b} of {len(lines)}\n{text}"

    @tool
    def write_file(self, agent: Agent, path: str, content: str) -> dict:
        """Create a file under agents/<your name>/ (replaces it if it exists). To change an
        existing file use edit_file or insert_after instead of writing it again.

        Args:
            path: e.g. agents/<your name>/bot.py or agents/<your name>/notes.md
            content: the complete file content
        """
        return self._save(agent, "write_file", self._own(agent, "write_file", path, False), content)

    @tool
    def copy_file(self, agent: Agent, src: str, path: str) -> dict:
        """Copy any workspace file (e.g. best/bot.py) to a file under agents/<your name>/, as a
        starting point for edit_file.

        Args:
            src: the file to copy, e.g. best/bot.py or agents/ds-0001/bot.py
            path: the new file, e.g. agents/<your name>/bot.py
        """
        text = self._path(src, agent=agent, tool="copy_file").read_text(errors="replace")
        return self._save(agent, "copy_file", self._own(agent, "copy_file", path, False), text)

    @tool
    def edit_file(self, agent: Agent, path: str, old: str, new: str) -> dict:
        """Change one of your files: replace the exact text `old` (it must occur exactly once)
        with `new`. An empty `new` deletes `old`.

        Args:
            path: a file under agents/<your name>/
            old: the exact current text to replace, copied from read_file (with enough lines
                to be unique)
            new: the replacement text (may be empty)
        """
        p = self._own(agent, "edit_file", path, True)
        text = p.read_text(errors="replace")
        i = self._unique(text, old, "old", path)
        return self._save(agent, "edit_file", p, text[:i] + new + text[i + len(old):])

    @tool
    def insert_after(self, agent: Agent, path: str, anchor: str, text: str) -> dict:
        """Insert new lines into one of your files, right after the line that holds `anchor`
        (which must occur exactly once).

        Args:
            path: a file under agents/<your name>/
            anchor: exact text from the line to insert after, e.g. "def reply(self, line):"
            text: the line(s) to insert
        """
        p = self._own(agent, "insert_after", path, True)
        old = p.read_text(errors="replace")
        i = self._unique(old, anchor, "anchor", path) + len(anchor)
        eol = old.find("\n", i)
        at = len(old) if eol < 0 else eol + 1
        head = old[:at] if eol >= 0 or not old else old + "\n"
        return self._save(agent, "insert_after", p, head + text.rstrip("\n") + "\n" + old[at:])

    @tool
    def list_files(self, prefix: str = "", agent: Agent | None = None) -> dict:
        """List workspace files, newest first, with size and age in seconds.

        Args:
            prefix: only paths starting with this, e.g. agents/glm-0000/ or best/
        """
        seen = (lambda r: True) if agent is None else (
            lambda r: not self._refuse(agent, "list_files", r))
        files = sorted((p for p in self.ws.rglob("*") if p.is_file()
                        and p.name not in ("log.jsonl", ".promote.lock")
                        and self._rel(p).startswith(prefix) and seen(self._rel(p))),
                       key=lambda p: -p.stat().st_mtime)
        return {"total": len(files), "files": [
            {"path": self._rel(p), "bytes": p.stat().st_size,
             "age_s": int(time.time() - p.stat().st_mtime)} for p in files[:200]]}

    @tool
    async def run_bot(self, path: str, lines: list[str], agent: Agent | None = None) -> dict:
        """Talk to a bot: run it on one fresh conversation and return the transcript.

        Args:
            path: the bot file, e.g. agents/ds-0002/bot.py or best/bot.py
            lines: the user's messages in order (1 to 20, one line each)
        """
        p = self._path(path, agent=agent, tool="run_bot")
        if not 0 < len(lines) <= 20:
            raise ToolError("give 1 to 20 lines", "bad_lines")
        return await asyncio.to_thread(self._run, p, [" ".join(x.split())[:500] for x in lines])

    @tool
    async def promote_best(self, agent: Agent, path: str, reason: str) -> dict:
        """Make a bot the team's best (best/bot.py) if it scores at least as well on the probes.

        Args:
            path: the candidate bot file, e.g. agents/ds-0001/bot.py
            reason: one line: what it does better, and your run_bot evidence
        """
        p = self._path(path, agent=agent, tool="promote_best")
        async with self._promotion():
            cand, hist = await asyncio.to_thread(self._score, p), self._history()
            best = self.ws / "best" / "bot.py"
            cur = await asyncio.to_thread(self._score, best) if hist else None
            why = ("it crashed or timed out on a probe" if cand["crashed"] else
                   f"it scores {cand['score']}, the current best {cur['score']}"
                   if cur and cand["score"] < cur["score"] else "")
            if why:
                self._log(agent, "promote_refused", path=path, score=cand["score"], why=why)
                return {"promoted": False, "why": why, "candidate": cand, "best": cur}
            text, n = p.read_text(errors="replace"), len(hist) + 1
            sha = hashlib.sha256(text.encode()).hexdigest()[:12]
            rec = {"n": n, "path": self._rel(p), "agent": agent.name, "reason": reason[:300],
                   "score": cand["score"], "of": cand["of"], "ts": round(time.time()),
                   "sha": sha, "best_before": hashlib.sha256(best.read_bytes()).hexdigest()[:12]
                   if hist and best.is_file() else None, "best_after": sha}
            for dst in (best, best.with_name(f"v{n}.py")):
                dst.write_text(text)
            self._append("best/history.jsonl", rec)
            self._log(agent, "promote", path=rec["path"], score=cand["score"], n=n)
        await asyncio.to_thread(self._promoted, rec)
        return {"promoted": True, "best": rec, "probes": cand["probes"]}

    @tool
    def best_status(self) -> dict:
        """Show best/bot.py: where it came from, its probe score, and the last 5 promotions."""
        recs = self._history()
        return {"best": recs[-1] if recs else None, "earlier": recs[-6:-1][::-1],
                "probes": sorted(self._probes())}
