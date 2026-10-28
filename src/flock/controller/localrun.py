"""`flock run spec.yaml`: a whole swarm on one machine (docs/local.md).

One process runs the controller (the same FastAPI app, store and reconcile loop as a deployed
controller) on 127.0.0.1 with a generated token; the reconcile loop launches workers through
`LocalJobsClient` as subprocesses. The run lives in `<data-dir>/<swarm>-<UTC time>/`:

    controller/   the controller's SQLite store and raw event streams
    agents/       agent workdirs (WORK_ROOT)     state/   harness state (FLOCK_LLM_HOME)
    logs/         controller.log, <worker>.log, world-<swarm>.log (a started world)
    run.json      the bundle: spec (secret-looking env values redacted), totals, per agent,
                  each type's loop and the budget
    events.jsonl  every agent's events and the swarm-level ones (loop_*, final_*, budget_*),
                  merged by time;  events/<agent>.jsonl  per agent

`<data-dir>/current` (mode 0600) holds the URL and token while the run is up, so `flock status`
and friends work from a second terminal in the same directory.

Stops on Ctrl-C (a second one forces), `--until <duration>`, `--until idle`, or the spec's
`limits.budget_usd`. Stopping halts the reconcile loop (so nothing relaunches), sends each worker
a drain (in-flight turns finish within `grace` s, then the processes are killed), waits for the
final event flush, writes the bundle and exits.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import secrets
import shutil
import signal
import socket
import sys
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

import httpx

from flock.comms.managed import INVITE_ENV, SNAPSHOT_INTERVAL, ManagedBoard
from flock.controller.api import create_app, swarm_status
from flock.controller.health import SAMPLE_S, Health
from flock.controller.local import LocalJobsClient
from flock.controller.loop import loop_status, swarm_stream
from flock.controller.prices import cost_usd, type_price_table
from flock.controller.reconcile import Settings, agent_state, run_reconciler, worker_state
from flock.controller.store import Store, agent_tokens
from flock.current import CURRENT_FILE, clear_current, write_current
from flock.models import DrainCommand, SwarmSpec, SwarmStatus, TokenUsage
from flock.world.host import WorldHost

log = logging.getLogger("flock.run")

SECRET_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASS|AUTH|CREDENTIAL)", re.IGNORECASE)
RECENT_EVENTS = 8
RECENT_UPDATES = 6  # `--watch`: update lines kept in the view
# Event types too chatty for the "recent events" pane.
QUIET_EVENTS = frozenset({"sync"})


@dataclass
class RunOptions:
    spec: SwarmSpec
    data_dir: Path = Path(".flock")
    until: float | None = None  # seconds of wall time, None: no time limit
    until_idle: bool = False  # stop once every agent is idle with nothing pending (see `idle`)
    prompt: str | None = None  # sent once to every agent of every type at start
    preflight: bool = False
    strict: bool = False
    grace: float = 10.0  # s for in-flight turns at stop
    refresh: float = 2.0  # s between view frames and stop-condition checks
    reconcile_interval: float = 1.0
    host: str = "127.0.0.1"
    out: TextIO = field(default_factory=lambda: sys.stdout)
    tty: bool | None = None  # None: out.isatty()
    worker_command: list[str] | None = None  # tests
    tools: bytes | None = None  # the packed tools folder (docs/tools.md), uploaded before the PUT
    # `--watch` (docs/live.md): the spec file whose folder is polled; a settled change runs
    # `flock update` on the live swarm
    watch: Path | None = None
    watch_interval: float = 1.0  # s between polls of the folder's files
    watch_quiet: float = 1.0  # s without a further change before the update runs
    env_files: list[str] = field(default_factory=list)  # `--env` / `--overlay`, for --watch
    overlays: list[str] = field(default_factory=list)


@dataclass
class RunResult:
    run_dir: Path
    reason: str
    exit_code: int
    totals: dict[str, Any]


# --------------------------------------------------------------------------- helpers


def run_name(swarm: str, data_dir: Path, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    name, n = f"{swarm}-{stamp}", 1
    while (data_dir / name).exists():
        n += 1
        name = f"{swarm}-{stamp}-{n}"
    return name


def redacted_spec(spec: SwarmSpec) -> dict[str, Any]:
    """The spec as stored, with env values whose name looks like a secret replaced. A spec
    holds only secret *names* (`secrets:`, `api_key_env`), but `env:` is free-form."""
    d = spec.model_dump(mode="json")
    for ts in d.get("types", {}).values():
        ts["env"] = {k: ("<redacted>" if SECRET_NAME.search(k) else v)
                     for k, v in (ts.get("env") or {}).items()}
    return d


def active_spec(store: Store, last: SwarmSpec) -> SwarmSpec:
    """The swarm's spec as the controller stores it now (re-applied specs, `flock scale`), or
    the last one seen once the swarm is gone."""
    row = store.get_swarm(last.swarm)
    return row["spec"] if row is not None else last


def cli_hint(data_dir: Path) -> str:
    """How another terminal points the CLI at this run."""
    data_dir = Path(data_dir).resolve()
    if data_dir == (Path.cwd() / ".flock").resolve():
        return "`flock status`, `flock agents <swarm>`, ... from this directory"
    return f"FLOCK_CURRENT={data_dir / CURRENT_FILE} flock status"


def _bind(host: str) -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, 0))
    sock.listen(128)
    sock.set_inheritable(False)
    return sock


def loops_settled(spec: SwarmSpec, status: SwarmStatus) -> bool:
    """Every type with a `loop:` has a loop that is over (docs/loop.md): `done` after its final
    round, or stopped without one (max_rounds, `until`, budget, `flock loop --stop`). A loop not
    started yet, running (agents idle between rounds) or in its final round keeps the swarm busy.
    A dormant type (`replicas: 0` and no agents, e.g. one only spawned into) has no rounds to
    run and does not count until an agent of it exists."""
    for name, ts in spec.types.items():
        if ts.loop is None:
            continue
        t = status.types.get(name)
        if ts.replicas == 0 and (t is None or not sum(t.agents_by_state.values())):
            continue
        if t is None or t.loop is None or t.loop.state != "done":
            return False
    return True


def _clock(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _counts(d: dict[str, int]) -> str:
    return " · ".join(f"{k} {v}" for k, v in sorted(d.items()) if v) or "-"


# --------------------------------------------------------------------------- run state


class Tracker:
    """What the view and the stop conditions need, read from the store each tick: agent and
    worker states, turns from the event streams (incremental per agent), cost, tokens."""

    def __init__(self, store: Store, swarm: str) -> None:
        self.store, self.swarm = store, swarm
        self.cursor: dict[str, int] = {}
        self.open_turns: dict[str, str] = {}  # agent -> turn id started and not ended
        self.completed: Counter[str] = Counter()  # agent -> turns ended
        self.started: Counter[str] = Counter()
        self.recent: deque[str] = deque(maxlen=RECENT_EVENTS)
        self.new_lines: list[str] = []  # since the last frame, for non-tty output
        self.prompted: set[str] = set()  # agents `--prompt` was sent to

    def read_events(self) -> None:
        from flock.cli import pretty_event

        agents = self.store.list_agents(self.swarm)
        # a removed agent (an ephemeral child, docs/spawn.md) can leave a turn open here when
        # its last turn_ended was not read before its row went: it never blocks `idle`
        live = {a["name"] for a in agents}
        self.open_turns = {n: t for n, t in self.open_turns.items() if n in live}
        for a in agents:
            name = a["name"]
            after = self.cursor.get(name, 0)
            evs = self.store.read_events(name, after, limit=5000)
            self.cursor[name] = after + len(evs)
            for ev in evs:
                kind = ev.get("type")
                if kind == "turn_started":
                    self.open_turns[name] = ev.get("turn_id") or "?"
                    self.started[name] += 1
                elif kind in ("turn_ended", "turn_interrupted"):
                    self.open_turns.pop(name, None)
                    if kind == "turn_ended":
                        self.completed[name] += 1
                if kind in QUIET_EVENTS:
                    continue
                line = f"{name:<16} {pretty_event(ev)}"
                self.recent.append(line)
                self.new_lines.append(line)

    def snapshot(self, spec: SwarmSpec, now: float) -> dict[str, Any]:
        self.read_events()
        agents = self.store.list_agents(self.swarm)
        workers = {w["id"]: w for w in self.store.list_workers(self.swarm)}
        states: dict[str, str] = {}
        for a in agents:
            st = agent_state(a, workers.get(a["worker"]), now).value
            if a["name"] in self.open_turns and st in ("idle", "unknown"):
                st = "working"  # turn_started seen, heartbeat not yet
            states[a["name"]] = st
        pending = self.store.pending_counts(self.swarm)
        tokens: dict[str, TokenUsage] = {}
        for a in agents:
            for model, u in agent_tokens(a).items():
                tokens[model] = tokens.get(model, TokenUsage()) + u
        status = swarm_status(self.store, spec, now)
        return {
            "status": status,
            "loops_settled": loops_settled(spec, status),
            "agents": agents,
            "states": states,
            "workers": {wid: worker_state(w, now).value for wid, w in workers.items()},
            "pending": sum(pending.values()),
            "running": sum(1 for s in states.values() if s == "working"),
            "completed": sum(self.completed.values()),
            "cost": self.store.swarm_cost(self.swarm),
            "tokens": tokens,
            "grants": len(self.store.list_grants(self.swarm)),
        }

    def idle(self, snap: dict[str, Any]) -> bool:
        """Every agent idle, nothing pending, no turn open, at least one turn done overall, every
        agent `--prompt` went to has finished a turn, and every type's loop is over (agents are
        idle between loop rounds)."""
        if not snap["agents"] or snap["pending"] or self.open_turns or not snap["loops_settled"]:
            return False
        if any(s != "idle" for s in snap["states"].values()):
            return False
        if any(self.completed[n] < 1 for n in self.prompted):
            return False
        return snap["completed"] >= 1


def render(spec: SwarmSpec, snap: dict[str, Any], *, elapsed: float, url: str,
           current: Path, recent: list[str], note: str = "", width: int = 120,
           updates: list[str] | None = None, watching: Path | None = None) -> str:
    st = snap["status"]
    agent_states = Counter(snap["states"].values())
    worker_states = Counter(snap["workers"].values())
    toks = " ".join(f"{m} in={u.input} out={u.output}" for m, u in snap["tokens"].items())
    live = "".join([f" · tools {spec.tools_version[:8]}" if spec.tools_version else "",
                    f" · {snap['grants']} grant{'s' * (snap['grants'] != 1)}"
                    if snap.get("grants") else ""])
    lines = [
        f"flock run · {spec.swarm} · compute local · {_clock(elapsed)} · {url}{live}",
        "",
        f"agents   {len(snap['agents']):<4} {_counts(agent_states)}",
        f"workers  {len(snap['workers']):<4} {_counts(worker_states)}",
        (f"prompts  pending {snap['pending']} · running {snap['running']} · "
         f"completed {snap['completed']}"),
        f"cost     ${snap['cost']:.4f}   {toks or 'no tokens yet'}",
        "",
    ]
    rows = [["type", "health", "agents", "workers", "cost"]]
    for t in st.types.values():
        rows.append([t.name, t.health.value, _counts(t.agents_by_state) or "-",
                     _counts(t.workers_by_state) or "-", f"${t.cost_usd:.4f}"])
        if t.launch_error:
            rows.append(["", "", f"launch error: {t.launch_error[:80]}", "", ""])
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    lines += ["  ".join(v.ljust(w) for v, w in zip(r, widths, strict=True)).rstrip()
              for r in rows]
    cut = max(40, width - 3)
    lines += ["", "recent events"] + (
        [f"  {x if len(x) <= cut else x[: cut - 1] + '…'}" for x in recent] or ["  (none yet)"])
    if watching is not None:
        lines += ["", f"live updates (watching {watching.parent}/)"] + (
            [f"  {x if len(x) <= cut else x[: cut - 1] + '…'}" for x in updates or []]
            or ["  (no change yet)"])
    lines += ["", f"second terminal: {cli_hint(current.parent)}"]
    lines.append(note or "Ctrl-C stops the run (twice: force)")
    return "\n".join(lines)


def status_line(snap: dict[str, Any], elapsed: float) -> str:
    return (f"[{_clock(elapsed)}] agents {_counts(Counter(snap['states'].values()))} | "
            f"pending {snap['pending']} running {snap['running']} completed {snap['completed']} | "
            f"${snap['cost']:.4f}")


# --------------------------------------------------------------------------- bundle


def write_bundle(run_dir: Path, store: Store, spec: SwarmSpec, *, started: datetime,
                 ended: datetime, reason: str, url: str, tracker: Tracker) -> dict[str, Any]:
    """run.json, events.jsonl (agents' and swarm-level events merged by time) and
    events/<agent>.jsonl. Returns the totals."""
    ev_dir = run_dir / "events"
    ev_dir.mkdir(parents=True, exist_ok=True)
    merged: list[tuple[str, int, dict]] = []
    per_agent = []
    tracker.read_events()
    now = time.time()
    workers = {w["id"]: w for w in store.list_workers(spec.swarm)}
    tokens: dict[str, TokenUsage] = {}
    for a in store.list_agents(spec.swarm):
        evs = store.read_events(a["name"], 0, limit=10**9)
        with (ev_dir / f"{a['name']}.jsonl").open("w", encoding="utf-8") as f:
            for i, ev in enumerate(evs):
                f.write(json.dumps(ev) + "\n")
                merged.append((str(ev.get("ts", "")), i, ev))
        toks = agent_tokens(a)
        for model, u in toks.items():
            tokens[model] = tokens.get(model, TokenUsage()) + u
        per_agent.append({
            "name": a["name"], "type": a["type"], "parent": a["parent"], "worker": a["worker"],
            "state": agent_state(a, workers.get(a["worker"]), now).value,
            "turns_completed": tracker.completed[a["name"]], "turns_reported": a["turns"],
            "cost_usd": a["cost"], "tokens": {m: u.model_dump() for m, u in toks.items()},
            "events": len(evs), "events_file": f"events/{a['name']}.jsonl",
        })

    # Removed agents (spawned children, ephemeral ones: docs/spawn.md) keep their events; their
    # cost is in the swarm's total (removed_cost) and, per agent, priced from their turns' tokens.
    removed = []
    for name in sorted(set(store.event_agents(spec.swarm)) - {a["name"] for a in per_agent}
                       - {swarm_stream(spec.swarm)}):
        evs = store.read_events(name, 0, limit=10**9)
        with (ev_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for i, ev in enumerate(evs):
                f.write(json.dumps(ev) + "\n")
                merged.append((str(ev.get("ts", "")), i, ev))
        # its cost from its turns' tokens (the store keeps only the swarm's removed total)
        rtoks: dict[str, TokenUsage] = {}
        for ev in evs:
            data = (ev.get("data") or {}).get("data") if ev.get("type") in (
                "turn_ended", "turn_interrupted") else None  # a turn cut by the stop costs too
            for model, u in ((data or {}).get("tokens") or {}).items():
                with contextlib.suppress(ValueError, TypeError):
                    rtoks[model] = rtoks.get(model, TokenUsage()) + TokenUsage(**u)
        rtype = (store.reserved_name(name) or {}).get("type") or max(  # words, else numeric
            (t for t in spec.types if f"-{t}-" in name), key=len, default=None)
        removed.append({"name": name, "type": rtype, "events": len(evs),
                        "events_file": f"events/{name}.jsonl",
                        "cost_usd": cost_usd(rtoks, type_price_table(spec.types.get(rtype)))
                        if rtype else None,
                        "tokens": {m: u.model_dump() for m, u in rtoks.items()}})

    # The swarm-level stream (docs/loop.md "Events"). Its per-agent events (`final_round`) are
    # also on the agent's stream, so only the swarm's own (agent "") are added.
    for i, ev in enumerate(store.read_events(swarm_stream(spec.swarm), 0, limit=10**9)):
        if not ev.get("agent"):
            merged.append((str(ev.get("ts", "")), i, ev))

    def ts_key(item: tuple[str, int, dict]) -> tuple[float, str, int]:
        try:
            t = datetime.fromisoformat(item[0]).timestamp()
        except ValueError:
            t = 0.0
        return (t, item[2].get("agent", ""), item[1])

    with (run_dir / "events.jsonl").open("w", encoding="utf-8") as f:
        for _, _, ev in sorted(merged, key=ts_key):
            f.write(json.dumps(ev) + "\n")
    totals = {
        "agents": len(per_agent),
        "workers": len(workers),
        "turns_completed": sum(x["turns_completed"] for x in per_agent),
        "events": len(merged),
        "pending_prompts": sum(store.pending_counts(spec.swarm).values()),
        "cost_usd": round(store.swarm_cost(spec.swarm), 6),
        "tokens": {m: u.model_dump() for m, u in tokens.items()},
    }
    loops = {t: st.model_dump(mode="json", exclude={"per_agent"}) for t in sorted(spec.types)
             if (st := loop_status(store, spec.swarm, t)) is not None}
    budget = {
        "budget_usd": spec.limits.budget_usd,
        "final_reserve_usd": round(sum(ts.loop.final.reserve_usd for ts in spec.types.values()
                                       if ts.loop is not None and ts.loop.final is not None), 6),
        "cost_usd": totals["cost_usd"],
        "stop": store.swarm_flags(spec.swarm).get("budget_stop"),  # final_reserve, hard or None
    }
    doc = {
        "run": run_dir.name,
        "swarm": spec.swarm,
        "compute": "local",
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "duration_s": round((ended - started).total_seconds(), 1),
        "stop_reason": reason,
        "controller_url": url,
        "spec": redacted_spec(spec),
        "totals": totals,
        "loops": loops,
        "budget": budget,
        "agents": per_agent,
        "removed_agents": removed,
        "workers": [{"id": w["id"], "type": w["type"], "crashes": w["crashes"],
                     "last_error": w.get("last_error"), "log": f"logs/{w['id']}.log"}
                    for w in workers.values()],
    }
    (run_dir / "run.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return totals


# --------------------------------------------------------------------------- the run


class _Server:
    """uvicorn on a pre-bound socket, signals left to us."""

    def __init__(self, app, sock: socket.socket) -> None:
        import uvicorn

        class Server(uvicorn.Server):
            @contextlib.contextmanager
            def capture_signals(self):
                yield

        self.sock = sock
        self.server = Server(uvicorn.Config(app, log_config=None, log_level="warning",
                                            access_log=False, timeout_graceful_shutdown=2))
        self.task: asyncio.Task | None = None

    async def start(self) -> None:
        self.task = asyncio.create_task(self.server.serve(sockets=[self.sock]))
        while not self.server.started:
            if self.task.done():
                self.task.result()  # raises the bind / startup error
            await asyncio.sleep(0.02)

    async def stop(self) -> None:
        self.server.should_exit = True
        if self.task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.task, 10)


def _log_to(path: Path) -> logging.Handler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    if root.level > logging.INFO or root.level == logging.NOTSET:
        root.setLevel(logging.INFO)
    for name in ("httpx", "httpcore", "uvicorn.access"):
        logging.getLogger(name).setLevel(logging.WARNING)
    return handler


async def _apply(http: httpx.AsyncClient, opts: RunOptions, say) -> bool:
    spec = opts.spec
    body = spec.model_dump(mode="json")
    if opts.preflight:
        r = await http.post(f"/swarms/{spec.swarm}/preflight", json=body, timeout=300)
        if r.status_code >= 400:
            say(f"preflight error {r.status_code}: {r.text}")
            return False
        rep = r.json()
        for res in rep["results"]:
            verdict = "unverified" if res.get("unverified") else "ok" if res["ok"] else "FAIL"
            say(f"preflight {res['model']}: {verdict}" + (f" ({res['error']})" if res.get("error")
                                                          else ""))
        if not rep["ok"] or (opts.strict and rep.get("unverified")):
            say("preflight failed: not starting (fix the provider or pass --skip-preflight)")
            return False
    spec.tools_version = None
    if opts.tools is not None:  # as `flock apply`: upload the folder, name it in the spec by hash
        r = await http.post(f"/swarms/{spec.swarm}/tools", content=opts.tools,
                            headers={"content-type": "application/gzip"}, timeout=120)
        if r.status_code >= 400:
            say(f"tools upload refused ({r.status_code}): {r.text}")
            return False
        up = r.json()
        spec.tools_version = up["version"]
        say(f"tools: version {up['version']} ({len(opts.tools)} bytes): "
            f"{', '.join(up['tools']) or 'none'}")
    from flock.cli import source_header

    r = await http.put(f"/swarms/{spec.swarm}", params={"replicas": "spec"},
                       json=spec.model_dump(mode="json"), headers=source_header(opts.watch))
    if r.status_code >= 400:
        say(f"apply refused ({r.status_code}): {r.text}")
        return False
    return True


async def _prompt_all(http: httpx.AsyncClient, opts: RunOptions, store: Store,
                      tracker: Tracker, say) -> bool:
    for type_name in opts.spec.types:
        r = await http.post(f"/swarms/{opts.spec.swarm}/prompts",
                            json={"type": type_name, "text": opts.prompt})
        if r.status_code >= 400:
            say(f"prompt to type {type_name} refused ({r.status_code}): {r.text}")
            return False
    tracker.prompted = {a["name"] for a in store.list_agents(opts.spec.swarm)}
    return True


async def _watch(opts: RunOptions, url: str, token: str, post) -> None:
    """`--watch` (docs/live.md): poll the experiment folder; once a change settled, run
    `flock update` against this run's controller and post its lines (what changed, or the
    refusal: the running spec then stays as it was)."""
    from flock.cli import SpecError, read_spec, update_swarm
    from flock.live import FolderWatch

    assert opts.watch is not None
    spec_file = opts.watch
    watch = FolderWatch(spec_file, opts.spec, quiet=opts.watch_quiet)

    def update() -> tuple[list[str], SwarmSpec | None]:
        with httpx.Client(base_url=url, headers={"X-Flock-Token": token}, timeout=60) as c:
            lines = update_swarm(c, str(spec_file), strict=opts.strict,
                                 env_files=opts.env_files, overlays=opts.overlays)
        return lines, read_spec(str(spec_file), opts.env_files, opts.overlays)

    while True:
        await asyncio.sleep(opts.watch_interval)
        if not watch.poll(time.monotonic()):
            continue
        try:
            lines, spec = await asyncio.to_thread(update)
        except SpecError as e:
            lines, spec = [*e.lines, "not applied: the swarm keeps running as it was"], None
        except Exception as e:  # the run goes on whatever an update does
            log.exception("update failed")
            lines, spec = [f"update failed: {type(e).__name__}: {e}"], None
        if spec is not None:
            watch.retarget(spec)
        stamp = datetime.now().strftime("%H:%M:%S")
        for line in lines:
            post(f"{stamp} {line}")


async def run_local(opts: RunOptions) -> RunResult:
    """Run the swarm until a stop condition; always stops the workers and writes the bundle."""
    spec = opts.spec
    if spec.compute.kind != "local":
        raise ValueError("flock run needs `compute: local` in the spec")
    data_dir = Path(opts.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    while True:
        run_dir = data_dir / run_name(spec.swarm, data_dir)
        try:
            run_dir.mkdir()  # atomic allocation also when identical swarms start together
            break
        except FileExistsError:
            continue
    handler = _log_to(run_dir / "logs" / "controller.log")
    tty = opts.out.isatty() if opts.tty is None else opts.tty

    def say(msg: str) -> None:
        print(msg, file=opts.out, flush=True)

    token = secrets.token_urlsafe(32)
    sock = _bind(opts.host)
    port = sock.getsockname()[1]
    url = f"http://{opts.host}:{port}"
    store = Store(run_dir / "controller")
    jobs = LocalJobsClient(root=run_dir, **({"command": opts.worker_command}
                                            if opts.worker_command else {}))
    runtime_secrets = dict(os.environ)
    settings = Settings(public_url=url, token=token, secret_source=runtime_secrets)
    board = ManagedBoard(run_dir) if spec.comms is not None and spec.comms.managed else None
    board_task: asyncio.Task | None = None

    async def snapshot_board() -> None:
        while True:
            await asyncio.sleep(SNAPSHOT_INTERVAL)
            await asyncio.to_thread(board.snapshot)
    reconcile_task: list[asyncio.Task] = []

    health = Health(store, token=token, jobs=jobs, run=run_dir.name)
    health_every = float(os.environ.get("FLOCK_HEALTH_INTERVAL", SAMPLE_S))

    async def background(notifier) -> None:
        reconcile_task.append(asyncio.current_task())
        if health_every > 0:  # cancelled with the reconciler (_stop_workers)
            reconcile_task.append(asyncio.create_task(health.run(health_every)))
        await run_reconciler(store, jobs, settings, opts.reconcile_interval,
                             wake=lambda t: notifier.wait("reconcile:", t))

    worlds = WorldHost(log_dir=run_dir / "logs")  # a spec's `world.command` (docs/world.md)
    app = create_app(store, token, background=background, compute="local",
                     stats=lambda: {"mode": "local", "public_url": url}, worlds=worlds,
                     health=health, secret_source=runtime_secrets, managed_board=board)
    server = _Server(app, sock)
    started = datetime.now(UTC)
    t0 = time.monotonic()
    tracker = Tracker(store, spec.swarm)
    updates: deque[str] = deque(maxlen=RECENT_UPDATES)
    watcher: asyncio.Task | None = None
    stop = asyncio.Event()
    force = asyncio.Event()
    reason = "error"
    code = 0
    loop = asyncio.get_running_loop()

    def on_signal() -> None:
        if board is not None:
            board.cancel_start()
        if stop.is_set():
            force.set()
        stop.set()

    handled = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, RuntimeError, ValueError):
            loop.add_signal_handler(sig, on_signal)
            handled.append(sig)
    current = data_dir / CURRENT_FILE
    totals: dict[str, Any] = {}
    try:
        if board is not None:
            await asyncio.to_thread(board.start, spec)
            runtime_secrets[INVITE_ENV] = board.invite
            say(f"board: {board.url}; operator token: {board.archive / 'operator.token'}")
            board_task = asyncio.create_task(snapshot_board(), name="board-snapshots")
        await server.start()
        write_current(data_dir, {"url": url, "token": token, "run_dir": str(run_dir),
                                 "swarm": spec.swarm, "pid": os.getpid()})
        say(f"flock run: {spec.swarm} at {url}; run dir {run_dir}")
        say(f"second terminal: {cli_hint(data_dir)} (URL and token in {current})")
        headers = {"X-Flock-Token": token}
        async with httpx.AsyncClient(base_url=url, headers=headers, timeout=60) as http:
            if not await _apply(http, opts, say):
                reason, code = "apply_failed", 1
                stop.set()
            elif opts.prompt and not await _prompt_all(http, opts, store, tracker, say):
                reason, code = "prompt_failed", 1
                stop.set()
        if opts.watch is not None and not stop.is_set():
            def post(line: str) -> None:
                updates.append(line)
                if not tty:
                    say(f"update: {line}")

            watcher = asyncio.create_task(_watch(opts, url, token, post), name="watch")
        idle_streak = 0
        while not stop.is_set():
            if board is not None:
                board.check()
                if board_task is not None and board_task.done():
                    board_task.result()  # a failed snapshot must not silently lose history
            elapsed = time.monotonic() - t0
            spec = active_spec(store, spec)  # a live PUT (flock scale, apply) wins
            snap = tracker.snapshot(spec, time.time())
            if tty:
                frame = render(spec, snap, elapsed=elapsed, url=url, current=current,
                               recent=list(tracker.recent),
                               width=shutil.get_terminal_size((120, 24)).columns,
                               updates=list(updates), watching=opts.watch)
                opts.out.write("\x1b[H\x1b[J" + frame + "\n")
                opts.out.flush()
                tracker.new_lines.clear()
            else:
                for line in tracker.new_lines:
                    say(line)
                tracker.new_lines.clear()
                say(status_line(snap, elapsed))
            budget = spec.limits.budget_usd
            if opts.until is not None and elapsed >= opts.until:
                reason = "until"
                break
            if budget is not None and snap["cost"] >= budget:
                reason = "budget"
                break
            idle_streak = idle_streak + 1 if opts.until_idle and tracker.idle(snap) else 0
            if idle_streak >= 2:  # two frames in a row: a just-delivered prompt shows up
                reason = "idle"
                break
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), opts.refresh)
        else:
            if reason == "error":
                reason = "interrupted"
        if watcher is not None:
            watcher.cancel()
        if jobs.alive():
            say(f"stopping ({reason}): draining workers, up to {opts.grace:g} s for in-flight "
                "turns (Ctrl-C again to force)")
        await _stop_workers(store, jobs, spec, reconcile_task, opts.grace, force)
    except Exception as exc:  # always stop the workers and write the bundle
        log.exception("flock run failed")
        say(f"flock run failed: {type(exc).__name__}: {exc}")
        reason, code = "error", 1
        await _stop_workers(store, jobs, spec, reconcile_task, 0, force)
    finally:
        if watcher is not None:
            watcher.cancel()
        for sig in handled:
            loop.remove_signal_handler(sig)
        try:
            try:  # controller/run.json, metrics/ and rollups/ for the viewer (docs/health.md)
                await asyncio.to_thread(health.close, time.time())
            except Exception:
                log.exception("health close failed")
            spec = active_spec(store, spec)
            totals = write_bundle(run_dir, store, spec, started=started,
                                  ended=datetime.now(UTC), reason=reason, url=url,
                                  tracker=tracker)
        finally:
            if board_task is not None:
                board_task.cancel()
                await asyncio.gather(board_task, return_exceptions=True)
            try:
                await asyncio.to_thread(worlds.stop_all)  # after workers, before the board
            except Exception:
                log.exception("World shutdown failed")
                reason, code = "world_shutdown_failed", 1
            if board is not None:
                try:
                    await asyncio.to_thread(board.stop)
                except Exception:
                    log.exception("Managed board shutdown/snapshot failed")
                    reason, code = "board_snapshot_failed", 1
                bundle_path = run_dir / "run.json"
                try:
                    if bundle_path.is_file():
                        doc = json.loads(bundle_path.read_text())
                        doc["board"] = board.metadata()
                        doc["stop_reason"] = reason
                        bundle_path.write_text(json.dumps(doc, indent=2) + "\n")
                except (OSError, ValueError):
                    log.exception("Could not save board run metadata")
                    reason, code = "bundle_failed", 1
            # parked long-polls of exited workers are cancelled at shutdown: not an error
            uv_log = logging.getLogger("uvicorn.error")
            level = uv_log.level
            uv_log.setLevel(logging.CRITICAL)
            await server.stop()
            sock.close()
            uv_log.setLevel(level)
            clear_current(data_dir, os.getpid())
            store.close()
            logging.getLogger().removeHandler(handler)
            handler.close()
    say(f"run {run_dir.name}: {reason}; {totals.get('agents', 0)} agents, "
        f"{totals.get('turns_completed', 0)} turns, ${totals.get('cost_usd', 0):.4f}")
    say(f"bundle: {run_dir}/run.json, {run_dir}/events.jsonl, {run_dir}/events/")
    return RunResult(run_dir, reason, code, totals)


async def _stop_workers(store: Store, jobs: LocalJobsClient, spec: SwarmSpec,
                        reconcile_task: list[asyncio.Task], grace: float,
                        force: asyncio.Event) -> None:
    for t in reconcile_task:  # first: a drained worker must not be relaunched
        t.cancel()
    for w in store.list_workers(spec.swarm):
        if w["job_id"] is not None:
            store.enqueue_command(w["id"], DrainCommand().model_dump_json())
    end = time.monotonic() + grace
    while jobs.alive() and time.monotonic() < end and not force.is_set():
        await asyncio.sleep(0.1)
    if jobs.alive():
        await asyncio.to_thread(jobs.kill_all)
