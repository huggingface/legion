"""`flock run spec.yaml`: a whole swarm on one machine (docs/local.md).

One process runs the controller (the same FastAPI app, store and reconcile loop as a deployed
controller) on 127.0.0.1 with a generated token; the reconcile loop launches workers through
`LocalJobsClient` as subprocesses. The run lives in `<data-dir>/<swarm>-<UTC time>/`:

    controller/   the controller's SQLite store and raw event streams
    agents/       agent workdirs (WORK_ROOT)     state/   harness state (FLOCK_LLM_HOME)
    logs/         controller.log, <worker>.log
    run.json      the bundle: spec (secret-looking env values redacted), totals, per agent
    events.jsonl  every agent's events, merged by time;  events/<agent>.jsonl  per agent

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

from flock.controller.api import create_app, swarm_status
from flock.controller.local import LocalJobsClient
from flock.controller.reconcile import Settings, agent_state, run_reconciler, worker_state
from flock.controller.store import Store, agent_tokens
from flock.current import CURRENT_FILE, clear_current, write_current
from flock.models import DrainCommand, SwarmSpec, TokenUsage

log = logging.getLogger("flock.run")

SECRET_NAME = re.compile(r"(TOKEN|KEY|SECRET|PASSWORD|PASS|AUTH|CREDENTIAL)", re.IGNORECASE)
RECENT_EVENTS = 8
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

        for a in self.store.list_agents(self.swarm):
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
        return {
            "status": swarm_status(self.store, spec, now),
            "agents": agents,
            "states": states,
            "workers": {wid: worker_state(w, now).value for wid, w in workers.items()},
            "pending": sum(pending.values()),
            "running": sum(1 for s in states.values() if s == "working"),
            "completed": sum(self.completed.values()),
            "cost": self.store.swarm_cost(self.swarm),
            "tokens": tokens,
        }

    def idle(self, snap: dict[str, Any]) -> bool:
        """Every agent idle, nothing pending, no turn open, at least one turn done overall, and
        every agent `--prompt` went to has finished a turn."""
        if not snap["agents"] or snap["pending"] or self.open_turns:
            return False
        if any(s != "idle" for s in snap["states"].values()):
            return False
        if any(self.completed[n] < 1 for n in self.prompted):
            return False
        return snap["completed"] >= 1


def render(spec: SwarmSpec, snap: dict[str, Any], *, elapsed: float, url: str,
           current: Path, recent: list[str], note: str = "", width: int = 120) -> str:
    st = snap["status"]
    agent_states = Counter(snap["states"].values())
    worker_states = Counter(snap["workers"].values())
    toks = " ".join(f"{m} in={u.input} out={u.output}" for m, u in snap["tokens"].items())
    lines = [
        f"flock run · {spec.swarm} · compute local · {_clock(elapsed)} · {url}",
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
    """run.json, events.jsonl (merged by time) and events/<agent>.jsonl. Returns the totals."""
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
        "agents": per_agent,
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
    r = await http.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))
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


async def run_local(opts: RunOptions) -> RunResult:
    """Run the swarm until a stop condition; always stops the workers and writes the bundle."""
    spec = opts.spec
    if spec.compute.kind != "local":
        raise ValueError("flock run needs `compute: local` in the spec")
    data_dir = Path(opts.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    run_dir = data_dir / run_name(spec.swarm, data_dir)
    run_dir.mkdir(parents=True)
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
    settings = Settings(public_url=url, token=token)
    reconcile_task: list[asyncio.Task] = []

    async def background(notifier) -> None:
        reconcile_task.append(asyncio.current_task())
        await run_reconciler(store, jobs, settings, opts.reconcile_interval,
                             wake=lambda t: notifier.wait("reconcile:", t))

    app = create_app(store, token, background=background, compute="local",
                     stats=lambda: {"mode": "local", "public_url": url})
    server = _Server(app, sock)
    started = datetime.now(UTC)
    t0 = time.monotonic()
    tracker = Tracker(store, spec.swarm)
    stop = asyncio.Event()
    force = asyncio.Event()
    reason = "error"
    code = 0
    loop = asyncio.get_running_loop()

    def on_signal() -> None:
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
        idle_streak = 0
        while not stop.is_set():
            elapsed = time.monotonic() - t0
            spec = active_spec(store, spec)  # a live PUT (flock scale, apply) wins
            snap = tracker.snapshot(spec, time.time())
            if tty:
                frame = render(spec, snap, elapsed=elapsed, url=url, current=current,
                               recent=list(tracker.recent),
                               width=shutil.get_terminal_size((120, 24)).columns)
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
        for sig in handled:
            loop.remove_signal_handler(sig)
        try:
            spec = active_spec(store, spec)
            totals = write_bundle(run_dir, store, spec, started=started,
                                  ended=datetime.now(UTC), reason=reason, url=url,
                                  tracker=tracker)
        finally:
            # parked long-polls of exited workers are cancelled at shutdown: not an error
            uv_log = logging.getLogger("uvicorn.error")
            level = uv_log.level
            uv_log.setLevel(logging.CRITICAL)
            await server.stop()
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
