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
`interviewer`). Bot runs share a pool of `runs.parallel` slots (the world's CPU), re-read live.
Promotions are decided on a sampled gate (`probes.gate_sample`, the full set every
`probes.gate_full_every`-th promotion), and a promotion that takes longer than `promote.wait_s`
answers `queued` with a ticket (best_status shows `pending` and the result).

Retired probes (`probes.retire_after`, 0: off): the gate already scores best on its probes at
every promotion (the new best on the candidate's gate, the old best when the gate re-scores
it). probes/streaks.json keeps, per probe id, the consecutive versions of best that passed it
when evaluated (one count per version, by its sha; a fail resets it to 0). A probe whose
streak reaches retire_after is retired: it does not count toward `probes.max_total` and is
left out of the sampled gate, but a full-set gate (gate_sample 0, every gate_full_every-th)
still scores it, so a regression on it is still caught (and a fail there un-retires it).
With a sample of 60 out of 400-800 probes a probe lands in a given gate 8-15% of the time,
so 3 counted passes span 3 versions at the very least and usually many more: 3 is the run's
setting. No extra bot runs.

best_status runs no bots: it shows the score of the gate that promoted the current best and
the latest full-set score of best, which a background task computes once per version of best
(one at a time, for the newest version only, at most `runs.background` probe runs at once and
only on slots no other call is waiting for). Best's probe results are shared: concurrent
scorings of the same version (the gate re-scoring best, the background) run each probe once.

Calls nobody waits for are dropped: run_bot, qualify, best_status and probes give up
after `runs.call_deadline_s` (just under the tool timeout) or when the caller disconnects; their
bot runs not yet started are dropped and free their slots. promote_best keeps its queue: a
queued gate runs to the end and lands in its ticket.

Knobs: the YAML file $CHATBOT_CONFIG (probes, interviews, runs), re-read when it changes."""

import asyncio
import contextvars
import functools
import inspect
import json
import os
import random
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

from chatbot2 import Chatbot2World, _copy, passes
from eliza import ElizaWorld

from flock.world import Agent, ToolError, World, tool

DEFAULTS = {"probes": {"per_agent_per_hour": 4, "max_lines": 12, "max_total": 400,
                       "gate_sample": 60, "gate_full_every": 10,
                       "retire_after": 0},  # 0: no probe is ever retired
            "interviews": {"start_minutes": 5, "expire_minutes": 20},
            "runs": {"parallel": 0,  # 0: the usable CPUs (sched_getaffinity)
                     "call_deadline_s": 55,  # bot-running calls give up (0: never)
                     "background": 2},  # probe runs at once for best's background full score
            "promote": {"max_waiting": 0, "wait_s": 40}}
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


class CallDropped(Exception):
    """A bot run of a call that was given up (deadline, caller gone): not run."""


class _Prio:
    """A bot run's priority, shared with the thread waiting for its slot: a low run takes a
    slot only when no normal run waits; a normal caller joining a low run raises it."""

    def __init__(self, low: bool) -> None:
        self.low = low


# the current call's drop flag (set when it is given up) and the run's priority; copied into
# the bot-run threads by asyncio.to_thread
_CALL: contextvars.ContextVar[threading.Event | None] = contextvars.ContextVar("call",
                                                                              default=None)
_PRIO: contextvars.ContextVar[_Prio | None] = contextvars.ContextVar("prio", default=None)
SEED = "\0seed"  # the key of best's seed score among its probe results in flight


class _Pool:
    """Slots for bot runs whose size can change while the world runs (`runs.parallel`). A run
    of a dropped call leaves the queue (CallDropped); a low-priority run waits while a normal
    one does."""

    def __init__(self, limit) -> None:
        self.limit, self.active, self.cond = limit, 0, threading.Condition()
        self.waiting: list[_Prio | None] = []

    def acquire(self, drop: threading.Event | None = None, prio: _Prio | None = None) -> None:
        with self.cond:
            self.waiting.append(prio)
            try:
                while True:
                    if drop is not None and drop.is_set():
                        raise CallDropped
                    if self.active < self.limit() and not (prio is not None and prio.low and any(
                            x is None or not x.low for x in self.waiting)):
                        break
                    self.cond.wait(0.5)  # re-checks the limit (may grow), the drop, the prio
            finally:
                self.waiting.remove(prio)
            self.active += 1

    def release(self) -> None:
        with self.cond:
            self.active -= 1
            self.cond.notify_all()

    def wake(self) -> None:
        with self.cond:
            self.cond.notify_all()

    def __enter__(self) -> None:
        self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()


def _pid(k: str) -> tuple:
    return (len(k), k)  # p2 before p10


def _usable_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0)) or 4
    except (AttributeError, OSError):  # not on Linux
        return os.cpu_count() or 4


class Chatbot3World(Chatbot2World):
    name, version, base_rules = "chatbot3", "0.3.4", (HERE / "rules.md").read_text()
    dirs = ("agents", "best", "probes", "interviews")
    # tool calls and hooks run concurrently (flock.world.server): the bookings have their
    # lock, promotions their critical section, file writes are atomic, logs are appends
    concurrent = True
    # calls whose work is wasted once nobody waits: bounded by runs.call_deadline_s and
    # cancelled when the caller disconnects (promote_best is not: its gate lands in a ticket)
    bounded = ("run_bot", "qualify", "best_status", "probes")

    def __init__(self, root: str | None = None) -> None:
        super().__init__(root or os.environ.get("CHATBOT_WS") or str(HERE / "ws"))
        self.phase = "merged"  # no phases: nothing is ever refused by team
        self._slots = _Pool(self._parallel)
        self._flight: tuple[str, dict] = ("", {})  # best's sha, its probe runs in flight
        self._seed: tuple[str, dict | None] = ("", None)  # best's sha, its seed score
        self._gates: dict[str, dict] = {}  # best's sha[:12] -> the gate that promoted it
        self._fullrec: dict | None = None  # the latest background full score of best
        self._bg: asyncio.Task | None = None

    def _parallel(self) -> int:
        """`runs.parallel`, re-read live; 0: the CPUs this process may use (a Slurm job's
        cpuset, not the whole node)."""
        return max(1, int(self.cfg("runs", "parallel")) or _usable_cpus())

    def _gate_parallel(self) -> int:  # a promotion may use the whole pool
        return self._parallel()

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
        # every bot run (run_bot, probes, promotions, chats) shares the CPU; a run of a call
        # that was given up never starts
        drop = _CALL.get()
        self._slots.acquire(drop, _PRIO.get())
        try:
            if drop is not None and drop.is_set():
                raise CallDropped
            return ElizaWorld._run(self, path, lines)
        finally:
            self._slots.release()

    # ------------------------------------------------------------------ calls nobody waits for
    def cancellable(self, tool: str) -> bool:
        return tool in self.bounded

    def call(self, tool: str, args: dict, agent: Agent):
        out = super().call(tool, args, agent)
        return self._bounded(tool, agent, out) if tool in self.bounded and inspect.isawaitable(
            out) else out

    async def _bounded(self, tool: str, agent: Agent, aw):
        """Run one call's work with its drop flag: at `runs.call_deadline_s` (0: none), or when
        the server cancels it (the caller disconnected), the flag is set, so its bot runs still
        waiting for a slot are dropped (the ones running finish in at most the bot timeout)."""
        drop, deadline = threading.Event(), float(self.cfg("runs", "call_deadline_s"))
        tok = _CALL.set(drop)
        try:
            return await (asyncio.wait_for(aw, deadline) if deadline > 0 else aw)
        except TimeoutError:  # asyncio's, from 3.11
            self._log(agent, "call_dropped", tool=tool, why="deadline")
            raise ToolError(f"{tool}: the world is busy and gave up after {deadline:g} s (its bot "
                            "runs were dropped). Try again in a few minutes.", "busy") from None
        except asyncio.CancelledError:
            self._log(agent, "call_dropped", tool=tool, why="cancelled")
            raise
        finally:
            drop.set()
            _CALL.reset(tok)
            self._slots.wake()

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

    # ------------------------------------------------------------------ retired probes
    def _streaks(self) -> dict:
        """probes/streaks.json: {probe id: {"streak": consecutive versions of best that passed
        it when evaluated, "sha": the last version counted}}; {} before the first gate."""
        f = self.ws / "probes" / "streaks.json"
        try:
            return json.loads(f.read_text()) if f.is_file() else {}
        except (ValueError, OSError):
            return {}

    def _gate_scored(self, sha: str, results: dict) -> None:
        """Count the gate's results for best (called inside the promotion critical section, so
        one writer at a time, across processes too). One count per version: a version seen
        again (the gate re-scoring it) adds no pass, but a fail always resets."""
        if not results:
            return
        if sha[:12] not in self._gates and not any(  # the gate that promotes it (in memory;
                r.get("best_after") == sha[:12] for r in self._history()):  # before its record)
            self._gates[sha[:12]] = {"probes": sorted(results, key=_pid),
                                     "fails": sorted((k for k, v in results.items() if not v),
                                                     key=_pid)}
            while len(self._gates) > 20:
                self._gates.pop(next(iter(self._gates)))
        s, sha, n = self._streaks(), sha[:12], int(self.cfg("probes", "retire_after"))
        for pid, ok in results.items():
            r = s.setdefault(pid, {"streak": 0, "sha": ""})
            if not ok:
                r.update(streak=0, sha=sha)
            elif r["sha"] != sha:
                r.update(streak=r["streak"] + 1, sha=sha)
                if n > 0 and r["streak"] == n:
                    self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": "world",
                                               "action": "probe_retired", "probe": pid})
        f = self.ws / "probes" / "streaks.json"
        tmp = f.with_name(f".streaks.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(s, sort_keys=True) + "\n")
        os.replace(tmp, f)

    def _retired(self) -> set[str]:
        n = int(self.cfg("probes", "retire_after"))
        return {k for k, r in self._streaks().items() if r["streak"] >= n} if n > 0 else set()

    def _counts(self) -> dict:
        live, ret = self._live(), self._retired()
        k = sum(x["id"] in ret for x in live)
        return {"live": len(live) - k, "retired": k,
                "max_total": int(self.cfg("probes", "max_total")),
                "retire_after": int(self.cfg("probes", "retire_after"))}

    def _full_why(self) -> str:
        c = self._counts()
        if c["live"] < c["max_total"]:
            return ""
        return (f"the live set is full: {c['live']} live probes, the limit is {c['max_total']}"
                + (f" ({c['retired']} retired probes do not count: a probe retires once best "
                   f"has passed it {c['retire_after']} times in a row)" if c["retire_after"] > 0
                   else ""))

    def _gate(self, cand_sha: str) -> tuple[list[dict], str]:
        """chatbot2's gate, but the sample is drawn from the probes that are not retired; the
        full set (gate_sample 0, every gate_full_every-th promotion) includes the retired."""
        live = [x for x in self._live() if x.get("verified", True)]
        size, every = int(self.cfg("probes", "gate_sample")), int(self.cfg("probes",
                                                                           "gate_full_every"))
        if size <= 0 or (every > 0 and (len(self._history()) + 1) % every == 0):
            return live, f"full {len(live)}"
        ret = self._retired()
        pool = [x for x in live if x["id"] not in ret]
        tail = f" ({len(live) - len(pool)} retired)" if len(pool) < len(live) else ""
        if size >= len(pool):
            return pool, f"full {len(live)}" if not tail else f"active {len(pool)}{tail}"
        pick = sorted(random.Random(cand_sha).sample(range(len(pool)), size))
        return [pool[i] for i in pick], (f"sample {size}/{len(pool)} seed {cand_sha[:12]}"
                                         + tail)

    # ------------------------------------------------------------------ best's score, shared
    def _best_now(self) -> tuple[Path, str] | None:
        """The current best as an immutable path (best/v<n>, not best/bot that a promotion
        replaces) and its sha; None before the first promotion."""
        p = self._best_path()
        if p is None:
            return None
        p = self._stable(p)
        return p, self._sha(p)

    async def _best_score(self, live: list | None = None, low: bool = False,
                          parallel: int = 0) -> dict | None:
        """chatbot2's `_full` of the current best, shared: each probe of this version runs at
        most once (cached in `_best`), and a probe (or the seed score) already in flight for it
        is awaited, not run again, so concurrent scorings (the gate re-scoring best, the
        background full score) never duplicate a run. `low`: the runs take only slots no
        other call waits for (a normal caller joining one raises it); `parallel` > 0: at most
        that many of this call's runs at once."""
        now = self._best_now()
        return None if now is None else await self._score_of(*now, live, low, parallel)

    async def _score_of(self, p: Path, sha: str, live: list | None, low: bool, parallel: int,
                        still=lambda: True) -> dict | None:
        if self._best[0] != sha:
            self._best = (sha, {})
        if self._flight[0] != sha:
            self._flight = (sha, {})
        live = [x for x in (self._live() if live is None else live) if x.get("verified", True)]
        sem = asyncio.Semaphore(parallel) if parallel > 0 else None
        res: dict = {}

        async def one(key: str, x: dict | None) -> None:
            if self._cached(sha, key):
                res[key] = self._cached(sha, key)[0]
                return
            if sem is not None:
                async with sem:
                    if still():  # e.g. best changed: the background stops starting runs
                        res[key] = await self._shared(p, sha, key, x, low)
            else:
                res[key] = await self._shared(p, sha, key, x, low)

        await asyncio.gather(one(SEED, None), *(one(x["id"], x) for x in live))
        if not still() or any(k not in res for k in (SEED, *(x["id"] for x in live))):
            return None  # stopped part way (the background, when best changed)
        seed, got = res[SEED], {x["id"]: res[x["id"]] for x in live}
        return {"seed": seed["score"], "seed_of": seed["of"], "seed_probes": seed["probes"],
                "crashed": seed["crashed"], "live": sum(got.values()), "live_of": len(got),
                "live_fails": [k for k, v in got.items() if not v], "results": got,
                "score": seed["score"] + sum(got.values()), "of": seed["of"] + len(got)}

    def _cached(self, sha: str, key: str) -> tuple | None:
        """(best's cached result for this probe, or SEED: its seed score) or None."""
        if key == SEED:
            return (self._seed[1],) if self._seed[0] == sha else None
        return (self._best[1][key],) if self._best[0] == sha and key in self._best[1] else None

    async def _shared(self, p: Path, sha: str, key: str, x: dict | None, low: bool):
        """Best's result for one probe (or SEED: its seed score): the run in flight for this
        version, or a new one. The run is its own task (no caller's drop flag): a caller that
        gives up leaves it to the others; its result is cached when it ends."""
        if got := self._cached(sha, key):  # e.g. it ended while this caller waited its turn
            return got[0]
        loop, flight = asyncio.get_running_loop(), self._flight[1]
        t = flight.get(key)
        if t is None or t.get_loop() is not loop or t.cancelled():
            prio = _Prio(low)

            async def go():
                _CALL.set(None)
                _PRIO.set(prio)
                if key == SEED:
                    return await asyncio.to_thread(self._score, p)
                return passes(x["check"], await asyncio.to_thread(self._run, p, x["lines"]))

            t = flight[key] = loop.create_task(go())
            t.prio = prio

            def keep(t: asyncio.Task) -> None:
                if self._flight[0] == sha and self._flight[1].get(key) is t:
                    del self._flight[1][key]
                if t.cancelled() or t.exception() is not None:
                    return
                if key == SEED:
                    self._seed = (sha, t.result())
                elif self._best[0] == sha:
                    self._best[1][key] = t.result()

            t.add_done_callback(keep)
        elif not low:
            t.prio.low = False  # a normal caller waits on it: it runs at normal priority
        return await asyncio.shield(t)

    def _bg_kick(self) -> None:
        """Start the background full score of best unless it runs or best has one. One task:
        when it ends it looks again, so a best that changed meanwhile gets the next one (the
        newest version only)."""
        if self._bg is not None and not self._bg.done() and self._bg.get_loop() is (
                asyncio.get_running_loop()):
            return
        now = self._best_now()
        if now is None or (self._fullrec or {}).get("sha") == now[1][:12]:
            return
        self._bg = asyncio.get_running_loop().create_task(self._bg_full())

    async def _bg_full(self) -> None:
        _CALL.set(None)  # not the kicking call's: its deadline does not stop this
        while (now := self._best_now()) is not None and (self._fullrec or {}).get(
                "sha") != now[1][:12]:
            p, sha = now
            n = next((r["n"] for r in self._history()[::-1] if r.get("best_after") == sha[:12]),
                     None)
            t0 = time.time()
            try:
                s = await self._score_of(p, sha, None, True, max(1, int(self.cfg(
                    "runs", "background"))), still=lambda sha=sha: self._best[0] == sha)
            except Exception as e:  # noqa: BLE001 (logged; the next best_status tries again)
                self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": "world",
                                           "action": "best_full_error", "error": repr(e)[:300]})
                return
            if s is not None:  # None: best changed meanwhile, score the new one
                self._fullrec = {"sha": sha[:12], "n": n, "ts": time.time(), "s": s}
                self._append("log.jsonl", {"ts": round(time.time(), 3), "agent": "world",
                                           "action": "best_full", "n": n, "sha": sha[:12],
                                           "score": s["score"], "of": s["of"],
                                           "seconds": round(time.time() - t0, 1)})

    def _score_view(self, recs: list[dict], sha: str) -> dict:
        """best_status's score, without running a bot: the gate that promoted the current best
        and the latest full-set score of best (its version and age)."""
        rec = next((r for r in recs[::-1] if r.get("best_after") == sha[:12]), None)
        out: dict = {"gate": None, "full": None}
        if rec:
            out["gate"] = {k: rec.get(k) for k in ("n", "gate", "score", "of", "seed", "live")}
            if g := self._gates.get(sha[:12]):
                out["gate"] |= g
        if f := self._fullrec:
            out["full"] = {"n": f["n"], "sha": f["sha"], "current": f["sha"] == sha[:12],
                           "age_s": round(time.time() - f["ts"])} | self._pub(f["s"])
        if not out["full"] or not out["full"]["current"]:
            out["full_pending"] = ("the full-set score of the current best is computed in the "
                                   "background; call best_status again in a few minutes")
        return out

    @tool
    async def best_status(self, agent: Agent | None = None) -> dict:
        """Show the shared best (best/bot.py or best/bot/): where it came from, its size, its
        score (`gate`: the score on the promotion gate that made it best, with the probes it was
        scored on and failed; `full`: its latest score on the whole live probe set, computed
        in the background, with its version and age), the last 5 promotions, how many
        promotions are `pending`, the results of your queued ones, and the probe counts: live
        (count toward the limit) and retired (best passed them several times in a row). Runs
        no bot: it answers at once."""
        recs = self._history()
        out = {"best": recs[-1] if recs else None, "earlier": recs[-6:-1][::-1],
               "pending": getattr(self, "_waiting", 0)}
        if agent is not None and (mine := [t for t in self.__dict__.get("_tickets", {}).values()
                                           if t["agent"] == agent.name][-3:]):
            out["your_promotions"] = mine
        if now := self._best_now():
            out |= {"path": self._rel(self._best_path()), "size": self._size(now[0]),
                    "score": self._score_view(recs, now[1])}
            self._bg_kick()
        return out | {"probes": self._counts()}

    @tool
    async def probes(self) -> dict:
        """The probe set every promotion is scored on: the seed probe files (the floor) and the
        swarm's probes (id, author, check, lines, newest 40), each with whether the current
        best passes it (`-`: not scored yet for this version; the background full score fills
        it in). An unverified probe (its evidence did not reproduce) does not count. A retired
        probe (best passed it several times in a row) is only checked in full-set gates and
        does not count toward the probe limit."""
        live, ret, now = self._live(), self._retired(), self._best_now()
        res = self._best[1] if now and self._best[0] == now[1] else {}
        seed = self._seed[1] if now and self._seed[0] == now[1] else None
        if now:
            self._bg_kick()
        out = {"seed": seed["probes"] if seed else sorted(self._probes()), "total": len(live),
               "best_fails": [k for k, v in res.items() if not v],
               "live": [{k: x[k] for k in ("id", "author", "check", "lines")}
                        | {"verified": x.get("verified", True),
                           "best": {True: "pass", False: "FAIL"}.get(res.get(x["id"]), "-")}
                        | ({"retired": True} if x["id"] in ret else {}) for x in live[-40:]]}
        if now:  # which version of best the pass/FAIL marks are for, and how many it has
            n = next((r["n"] for r in self._history()[::-1] if r.get("best_after") == now[1][
                :12]), None)
            out["marks"] = {"best_n": n, "best_sha": now[1][:12], "scored": len(res),
                            "note": "best's results so far (its gate, then the background "
                                    "full score); `-`: not scored yet. No bot runs here."}
        return out | {"counts": self._counts()}

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
        if not interviewer(agent):  # most turns: nothing to do, answer at once
            return
        # the release is bookkeeping: off the request path, so the hook answers in ms
        threading.Thread(target=self._release_cut, args=(agent,), daemon=True).start()

    def _release_cut(self, agent: Agent) -> None:
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
        copy, then promote. Promotions run one at a time: one that waited behind another is
        compared with the NEW best (so the same score is then a tie). Both bots are scored on
        the same probes: a sample of the live set picked by your bot's hash (the full set every
        few promotions). If it is still queued after a while you get `queued` and a ticket:
        keep working, best_status shows the result.

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
