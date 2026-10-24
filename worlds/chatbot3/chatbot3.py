"""The chatbot world, run 3 (experiments/run3): worlds/chatbot2 without teams, phases or the
compare round. One flat workspace `$CHATBOT_WS` (default ws/ here): agents/<name>/ (each agent
writes only there; everyone reads and runs everything), best/ (bot.py or bot/, v<n>
snapshots, current.json, history.jsonl), probes/ (seed *.txt, the swarm's live.jsonl),
interviews/<booking>/ (frozen copies), interviews.json, verdicts.jsonl, log.jsonl.

Kept from chatbot2: bots as files or packages, grep_file, the 2 MB write cap and the 60 KB
read window, swarm-written probes (add_probe, verified: best must fail it), the beat-best gate
(strictly better on the live probe set, ties refused, never lower on the seed probes), bookings
with a frozen copy, the evidence-first verdict, the turn/agent_done hooks, the log and the
bucket mirror of best/ ($CHATBOT_BUCKET).

Changed: promote_best is open from the start (it runs the same qualify bar as qualify and
book_interview: chatbot2's one `_bar`). Interviews are `strong` only and have no quota:
bookings are still reserved, charged on a verdict and released when cut off, for the records,
but never refused for quota. The interviewer is `<parent>-interviewer-NNNN` (type
`interviewer`). Bot runs share a pool of `runs.parallel` slots (the world's CPU), read at start.

Knobs: the YAML file $CHATBOT_CONFIG (probes, interviews, runs), re-read when it changes."""

from __future__ import annotations

import asyncio
import functools
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
sys.path.insert(0, str(HERE.parent / "chatbot2"))  # chatbot2 adds ../chatbot, which adds ../eliza

from chatbot2 import Chatbot2World, _copy
from eliza import ElizaWorld

from flock.world import Agent, ToolError, World, tool

DEFAULTS = {"probes": {"per_agent_per_hour": 4, "max_lines": 12, "max_total": 400},
            "interviews": {"start_minutes": 5, "expire_minutes": 20},
            "runs": {"parallel": 0}}  # 0: the usable CPUs (sched_getaffinity)
IV = re.compile(r"-interviewer-\d+$")  # agent-0007-interviewer-0012 -> parent agent-0007


def interviewer(agent: Agent) -> bool:
    """An interviewer: type `interviewer` (any naming; `naming: words` gives `iv-amber-otter`),
    or a numeric child name `<parent>-interviewer-NNNN`."""
    return agent.type == "interviewer" or bool(IV.search(agent.name))


def _doc(fn, doc: str):
    """The parent's tool under this world's description (the agents read it)."""
    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def w(self, *a, **k):
            return await fn(self, *a, **k)
    else:
        @functools.wraps(fn)
        def w(self, *a, **k):
            return fn(self, *a, **k)
    w.__doc__ = doc
    return tool(w)


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 4
    except (AttributeError, OSError):  # not on Linux
        return os.cpu_count() or 4


class Chatbot3World(Chatbot2World):
    name, version, base_rules = "chatbot3", "0.3.0", (HERE / "rules.md").read_text()
    dirs = ("agents", "best", "probes", "interviews")

    def __init__(self, root: str | None = None) -> None:
        super().__init__(root or os.environ.get("CHATBOT_WS") or str(HERE / "ws"))
        self.phase = "merged"  # no phases: nothing is ever refused by team
        # 0: the CPUs this process may use (a Slurm job's cpuset, not the whole node)
        n = int(self.cfg("runs", "parallel")) or _usable_cpus()
        self._slots = threading.BoundedSemaphore(n)

    @property
    def description(self) -> str:
        return self.base_rules

    def manifest(self) -> dict:
        return World.manifest(self)

    def cfg(self, section: str, key: str) -> float:
        f = Path(os.environ.get("CHATBOT_CONFIG") or "/nonexistent")
        if f.is_file() and f.stat().st_mtime != self._cfg[0]:
            try:
                d = yaml.safe_load(f.read_text()) or {}
                self._cfg = (f.stat().st_mtime, d if isinstance(d, dict) else {})
            except yaml.YAMLError:
                pass  # a half-written file: keep the last good one
        return (self._cfg[1].get(section) or {}).get(key, DEFAULTS[section][key])

    # ------------------------------------------------------------------ one flat workspace
    @staticmethod
    def team(agent: Agent) -> str:
        return ""

    def _home(self, agent: Agent) -> Path:
        return self.ws / "agents" / agent.name

    def _refuse(self, agent: Agent, tool: str, rel: str) -> str | None:
        return None

    def _compare_left(self) -> float:
        return 0.0

    def _run(self, path: Path, lines: list[str]) -> dict:
        with self._slots:  # every bot run (run_bot, probes, promotions, chats) shares the CPU
            return ElizaWorld._run(self, path, lines)

    # ------------------------------------------------------------------ promotion
    @tool
    async def qualify(self, agent: Agent, path: str) -> dict:
        """Check a bot before a promotion or an interview: on every seed probe conversation it
        runs within the time limit and answers each line, with no empty or repeated reply.
        Also shows its score on the live probe set. (promote_best and book_interview run the
        same bar.)

        Args:
            path: the bot, e.g. agents/<your name>/bot or best/bot
        """
        return await Chatbot2World.qualify(self, agent, path)

    # ------------------------------------------------------------------ interviews
    def _quotas(self, team: str) -> dict:
        """No quota: what was booked in the last hour, for the records."""
        now, b = time.time(), self._expired()
        recent = [x for x in b.values() if x["ts"] > now - 3600 and x["state"] != "released"]
        return {"strong": {"limit": None, "last_hour": len(recent),
                           "pending": sum(x["state"] != "charged" for x in recent)}}

    @tool
    async def book_interview(self, agent: Agent, path: str,
                             level: Literal["strong"] = "strong") -> dict:
        """Book a strong interview BEFORE spawn_interviewer: freezes a copy of the bot (later
        edits and promotions do not touch the interview) and runs the qualify bar on the copy.
        No quota, but each interview costs a strong model's time: book a bot you believe in
        (best/bot, or yours once it beats best on your own runs).

        Args:
            path: the bot, e.g. best/bot or agents/<your name>/bot
            level: strong (the only level)
        """
        p = self._bot(path, agent, "book_interview")
        with self._blk:  # the id: the agent's own count, so it can name it (agent-0001-iv2-strong)
            k = 1 + sum(x["by"] == agent.name for x in self._bookings().values())
            bid = f"{agent.name}-iv{k}-strong"
            self._store({**self._bookings(), bid: {
                "id": bid, "by": agent.name, "state": "freezing", "ts": time.time()}})
        snap = self.ws / "interviews" / bid / ("bot" if p.is_dir() else "bot.py")
        _copy(p, snap)
        if fails := await asyncio.to_thread(self._bar, snap):
            shutil.rmtree(snap.parent)
            with self._blk:
                b = self._bookings()
                b[bid].update(state="released", why="not qualified")
                self._store(b)
            raise ToolError(f"refused: {self._rel(p)} does not qualify: {self._unfit(fails)}. "
                            "Nothing was booked.", "not_qualified")
        with self._blk:
            hist, b, now = self._history(), self._expired(), time.time()
            b[bid] = {"id": bid, "team": "", "level": "strong", "by": agent.name,
                      "path": self._rel(p), "snapshot": self._rel(snap),
                      "sha": self._sha(snap)[:12], "best_n": hist[-1]["n"] if hist and self._rel(
                          p).startswith("best/bot") else None, "ts": now, "last": now,
                      "state": "reserved", "interviewer": None, "turns": []}
            self._store(b)
        self._log(agent, "book_interview", booking=bid, path=self._rel(p))
        return {"booking": bid, "snapshot": self._rel(snap), "next": f"call spawn_interviewer("
                f"booking='{bid}', thread=<thread id>) now"}

    def _expired(self) -> dict:
        b, now = self._bookings(), time.time()
        for x in b.values():
            if (x["state"] in ("reserved", "freezing") and now - x["ts"] > 60 * self.cfg(
                    "interviews", "start_minutes")) or (x["state"] == "active" and now - x[
                    "last"] > 60 * self.cfg("interviews", "expire_minutes")):
                x.update(state="released", why="expired")
        return b

    def _booking(self, agent: Agent, bid: str) -> tuple[dict, dict]:
        if not interviewer(agent):
            raise ToolError("refused: only interviewer agents call this; book_interview, then "
                            "spawn_interviewer", "not_interviewer")
        # a numeric name says its parent; a words name does not: the booking binds the first
        # interviewer that chats on it (its task names the booking)
        m = IV.search(agent.name)
        b, parent = self._expired(), agent.name[:m.start()] if m else None
        x = b.get(bid.strip())
        if not x or (parent is not None and x["by"] != parent) or x.get("interviewer") not in (
                None, agent.name) or x["state"] not in ("reserved", "active"):
            self._log(agent, "refused", tool="chat", booking=bid, why="booking")
            raise ToolError(f"refused: booking {bid!r} is not open for you"
                            f"{' (' + x['state'] + ')' if x else ''}. Call `done` now.",
                            "no_booking")
        return b, x

    def on_turn(self, agent: Agent, data: dict) -> None:
        """An interviewer's turn ended (its whole life): an interview without a verdict was cut
        off and is released (for the records; there is no quota)."""
        if not interviewer(agent):
            return
        with self._blk:
            b = self._bookings()
            cut = [x for x in b.values() if x.get("interviewer") == agent.name
                   and x["state"] in ("reserved", "active")]
            for x in cut:
                x.update(state="released", why="cut off")
                self._log(agent, "interview_released", booking=x["id"])
            if cut:
                self._store(b)

    on_agent_done = on_turn

    @tool
    def verdicts(self, agent: Agent, path: str = "") -> dict:
        """The interviewers' verdicts (newest first, last 10): the label, the evidence both
        ways and the last turns, plus the open bookings.

        Args:
            path: only the verdicts on this bot (e.g. best/bot); all when left out
        """
        recs = [r for r in self._jsonl("verdicts.jsonl") if r["path"].startswith(path)][::-1][:10]
        return {"verdicts": [{k: v for k, v in r.items() if k != "transcript"}
                             | {"last_turns": r["transcript"][-3:]} for r in recs],
                "open_bookings": [{k: x[k] for k in ("id", "path", "by", "state")}
                                  for x in self._expired().values()
                                  if x["state"] in ("reserved", "active")][-20:]}


DOCS = {
    "promote_best": """Make your bot (a .py file or a package directory) the shared best
        (best/bot.py or best/bot/). Open from the start. It must pass the qualify bar (every
        seed conversation runs, every line gets a reply, no empty or repeated reply) and BEAT
        the current best on the live probe set: a higher total score, never lower on the seed
        probes. A tie does not promote: add a probe best fails (add_probe), fix it in your
        copy, then promote.

        Args:
            path: your candidate, e.g. agents/<your name>/bot
            reason: one line: what it does better, and your run_bot evidence
        """,
    "read_file": """Read a workspace file, or some of its lines. The first line of the result
        says which lines you got and how many the file has. A file over 60 KB needs start and
        end: find the spot with grep_file first.

        Args:
            path: e.g. best/bot/memory.py or agents/agent-0001/bot/__main__.py
            start: first line to return (1-based); from the top when left out
            end: last line to return (inclusive); to the end when left out
        """,
    "grep_file": """Find the lines matching `pattern` in a file, or in every file under a
        directory (a package, or agents/ to see what everyone is doing). Each hit is
        `file:line:text`; then read_file just that range.

        Args:
            path: a file or a directory, e.g. best/bot, agents or best/bot/memory.py
            pattern: a regular expression (Python, case-insensitive); plain text works too
            max_hits: at most this many hits (1 to 200)
        """,
    "copy_file": """Copy a file, or a whole directory (a package such as best/bot), to a path
        under your own dir, as a starting point for edit_file. Copying a directory onto one of
        yours replaces it: files that are not in `src` are removed.

        Args:
            src: what to copy, e.g. best/bot, best/bot.py or agents/agent-0003/bot
            path: the new file or directory, e.g. agents/<your name>/bot
        """,
    "run_bot": """Talk to a bot: run it on one fresh conversation and return the transcript.

        Args:
            path: a bot file or package, e.g. best/bot, best/bot.py or agents/agent-0001/bot
            lines: the user's messages in order (1 to 20, one line each)
        """,
    "best_status": """Show the shared best (best/bot.py or best/bot/): where it came from, its
        size, its score on the live probe set and the probes it fails, and the last 5
        promotions.""",
    "chat": """Say one message to the candidate and get its reply (interviewers only). The
        conversation keeps going across calls, up to 10 messages.

        Args:
            booking: the booking id from your task, e.g. agent-0001-iv2-strong
            message: your next message, one line
        """,
}
for _name, _text in DOCS.items():
    setattr(Chatbot3World, _name, _doc(getattr(Chatbot2World, _name), _text))


def serve() -> None:
    """What the spec runs (`{python} chatbot3.py serve`); no operator routes. A world served
    elsewhere (a Space) can use `flock world serve chatbot3:Chatbot3World` instead."""
    from flock.world.server import serve as world_serve

    world_serve("chatbot3:Chatbot3World")


if __name__ == "__main__":
    if (sys.argv[1] if len(sys.argv) > 1 else "serve") != "serve":
        sys.exit("usage: chatbot3.py [serve]")
    serve()
