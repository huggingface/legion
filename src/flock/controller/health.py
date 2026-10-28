"""Controller health (docs/health.md): world and provider probes, per-swarm snapshots, token
usage per hour and hourly per-agent rollups for the run viewer.

Every SAMPLE_S the controller probes each swarm's world (`GET /manifest` timed, `GET /metrics`
when the manifest lists `metrics-v1`, the Slurm job state when `world.slurm_job_name` is set)
and the providers with `metrics: true` (vLLM Prometheus text), then builds one snapshot per
swarm from the store (heartbeats, telemetry), the swarm stream and in-memory counters fed by
`post_events`. Event files are never re-read.

For the run viewer it also writes `rollups/<swarm>/<YYYY-MM-DDTHH>.jsonl` (one line per agent
active that hour, appended when the hour closes; hours are by arrival time at the controller)
and `run.json` at the state root (at start, on each new generation, at the end).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from flock.controller.reconcile import agent_state, swarm_stream
from flock.controller.store import Row, Store
from flock.models import Event, SwarmSpec, hit_max_turns
from flock.world.host import world_token
from flock.world.protocol import TOKEN_HEADER

log = logging.getLogger("flock.health")

SAMPLE_S = 30.0  # FLOCK_HEALTH_INTERVAL overrides
PROBE_TIMEOUT_S = 5.0
KEEP_30S = 6 * 3600.0
ROLLUP_S = 300.0  # every 10th sample is also kept as a 5-min sample, 7 days
KEEP_5M = 7 * 86400.0
STUCK_S = 30 * 60.0  # working with no event for longer: listed as stuck (FLOCK_STUCK_MIN)
NEAR_LIMIT_S = 3600.0  # alert: a worker reaches its memory limit within this
TOOL_ALERT_PCT = 20.0  # alert: a tool's error or refusal share over the last sample
RECENT_EVENTS = 10
NOISY = ("loop_tick", "loop_wake")  # swarm-stream events that are not control-plane actions
TOKEN_FIELDS = (("input", "input"), ("output", "output"), ("cache_read", "cacheRead"),
                ("cache_write", "cacheWrite"), ("reasoning", "reasoning"))
DOWN_AFTER = 2  # consecutive failed manifest probes before `world_down`
VLLM_METRICS = {"num_requests_waiting": "waiting", "num_requests_running": "running",
                "gpu_cache_usage_perc": "kv", "kv_cache_usage_perc": "kv"}  # vLLM v0 / v1 names


_PROM = re.compile(r"^([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+(\S+)")


def parse_vllm(text: str) -> dict[str, float]:
    """`waiting`, `running` (summed over label sets) and `kv` (max) from vLLM's /metrics."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        m = _PROM.match(line)
        key = VLLM_METRICS.get(m.group(1).rsplit(":", 1)[-1]) if m else None
        if key is None:
            continue
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        out[key] = max(out.get(key, 0.0), v) if key == "kv" else out.get(key, 0.0) + v
    return out


def hour_of(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H")


def _tokens(u: Mapping[str, Any]) -> dict[str, int]:
    """TokenUsage fields from a TokenUsage dump or a pi `usage` object (camelCase)."""
    return {k: int(u.get(k) or u.get(alt) or 0) for k, alt in TOKEN_FIELDS}


def _add(acc: dict[str, int], d: Mapping[str, int]) -> None:
    for k, v in d.items():
        acc[k] = acc.get(k, 0) + v


def _write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Health:
    def __init__(self, store: Store, *, token: str = "", secrets: Mapping[str, str] = os.environ,
                 jobs: Any = None, transport: httpx.AsyncBaseTransport | None = None,
                 run: str | None = None) -> None:
        self.store, self.token, self.secrets, self.jobs = store, token, secrets, jobs
        self.run_name = run or store.data_dir.resolve().name
        self.transport = transport
        self.world: dict[str, dict[str, Any]] = {}  # swarm -> {up, manifest_ms, slurm_state, metrics}
        self.providers: dict[str, dict[str, dict[str, float]]] = {}  # swarm -> provider -> queue
        self._fails: dict[str, int] = {}  # swarm -> consecutive failed manifest probes
        self.stuck_s = float(secrets.get("FLOCK_STUCK_MIN", STUCK_S / 60)) * 60
        self._lock = threading.Lock()  # post_events (request threads) vs tick (a thread)
        self._since: float | None = None  # first tick: "no event since" baseline after a restart
        self._last_event: dict[str, float] = {}  # agent -> when its last event arrived
        self._open: dict[str, dict[str, dict[str, int]]] = {}  # agent -> model -> message usage
        self._usage: dict[tuple[str, str, str], dict[str, int]] = {}  # (swarm, hour, type) -> new
        self._cursor: dict[str, int] = {}  # swarm stream lines read
        self._relaunches: dict[str, Counter] = {}  # swarm -> worker id -> crash/planned_restart
        self._recent: dict[str, deque] = {}  # swarm -> last control-plane events
        self._tools: dict[str, dict[str, dict]] = {}  # swarm -> worker -> counters, last tick
        self._prev_tick: dict[str, float] = {}
        self._last_5m: dict[str, float] = {}
        self._hour: str | None = None  # the open rollup hour
        self._rollups: dict[str, dict[str, Any]] = {}  # agent -> its line for the open hour
        self._closed: list[dict[str, dict[str, Any]]] = []  # closed hours not yet written
        self._run_gens: dict[str, int] | None = None  # generations run.json was written for

    # ------------------------------------------------------------------ counters (post_events)
    def on_events(self, a: Row, evs: list[Event], now: float) -> None:
        """Token usage by type and hour from `turn_ended`, with assistant `message` usage
        counted only for turns that never got one (cut by a worker death or an interrupt): a
        turn's message usage is held until its turn_ended (then dropped: the turn total wins)
        or until the next turn_started / turn_interrupted / session_* event (then counted)."""
        with self._lock:
            self._last_event[a["name"]] = now
            for ev in evs:
                self._count(a, ev, now)

    def _count(self, a: Row, ev: Event, now: float) -> None:
        name, d = a["name"], ev.data
        r = self._rollup(a, now)
        r["events"][ev.type] = r["events"].get(ev.type, 0) + 1
        ts = ev.ts.isoformat()
        r["first_ts"] = r["first_ts"] or ts
        r["last_ts"] = ts
        if ev.type == "message":
            m = d.get("message") or {}
            if m.get("role") == "assistant" and isinstance(m.get("usage"), dict):
                held = self._open.setdefault(name, {}).setdefault(m.get("model") or "unknown", {})
                _add(held, _tokens(m["usage"]))
            elif m.get("role") == "toolResult":
                t = r["tools"].setdefault(m.get("toolName") or "?", {"n": 0, "err": 0})
                t["n"] += 1
                t["err"] += bool(m.get("isError"))
        elif ev.type == "error":
            r["errors"] += 1
        elif ev.type == "turn_ended":
            p = d.get("data") or {}
            r["turns"] += 1
            r["tool_calls"] += int(p.get("tool_calls") or 0)
            why = f"{p.get('reason') or ''} {p.get('subtype') or ''}"
            out = ("max_turns" if hit_max_turns(p) else "max_budget" if "budget" in why
                   else "error" if p.get("error") or p.get("status") not in (None, "completed")
                   else "done")
            r["out"][out] += 1
            self._open.pop(name, None)
            for model, u in (p.get("tokens") or {}).items():
                self._spend(a, model, _tokens(u), now)
        elif ev.type == "turn_started" or ev.type == "turn_interrupted" or ev.type.startswith(
                "session_"):
            for model, u in self._open.pop(name, {}).items():
                self._spend(a, model, u, now)

    def _spend(self, a: Row, model: str, u: dict[str, int], now: float) -> None:
        _add(self._usage.setdefault((a["swarm"], hour_of(now), a["type"]), {}), u)
        _add(self._rollup(a, now)["tokens"].setdefault(model, {}), u)

    # ------------------------------------------------------------------ hourly rollups
    def _roll(self, now: float) -> None:
        """Close the open hour when `now` is past it (under the lock)."""
        h = hour_of(now)
        if h != self._hour:
            if self._rollups:
                self._closed.append(self._rollups)
            self._hour, self._rollups = h, {}

    def _rollup(self, a: Row, now: float) -> dict[str, Any]:
        self._roll(now)
        r = self._rollups.get(a["name"])
        if r is None:
            r = self._rollups[a["name"]] = {
                "h": self._hour, "agent": a["name"], "swarm": a["swarm"], "type": a["type"],
                "parent": a.get("parent"), "turns": 0,
                "out": {"done": 0, "max_turns": 0, "max_budget": 0, "error": 0},
                "working_s": 0.0, "pending_s": 0.0, "idle_s": 0.0, "tool_calls": 0, "tools": {},
                "tokens": {}, "events": {}, "wakes": 0, "errors": 0, "first_ts": None,
                "last_ts": None}
        return r

    def _write_rollups(self, hours: list[dict[str, dict[str, Any]]]) -> None:
        specs = {sw["name"]: sw["spec"] for sw in self.store.list_swarms()}
        for lines in hours:
            files: dict[Path, list[str]] = {}
            for r in lines.values():
                if not (r["events"] or r["working_s"] or r["pending_s"]):
                    continue  # idle all hour, nothing happened: not active
                r = dict(r)
                swarm = r.pop("swarm")
                ts = specs[swarm].types.get(r["type"]) if swarm in specs else None
                r["ephemeral"] = bool(ts and ts.ephemeral)
                path = self.store.data_dir / "rollups" / swarm / f"{r['h']}.jsonl"
                files.setdefault(path, []).append(json.dumps(r, sort_keys=True))
            for path, out in files.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                # append: a controller restart mid-hour leaves two lines for an agent; sum them
                with path.open("a", encoding="utf-8") as f:
                    f.write("\n".join(out) + "\n")

    def _sample_states(self, swarm: str, agents: list[Row], states: dict[str, str],
                       dt: float, now: float) -> None:
        """Working / pending / idle seconds, sampled: each tick adds its interval to the state
        each agent is in (pending: idle with a prompt queued)."""
        pending = self.store.pending_counts(swarm)
        with self._lock:
            for a in agents:
                st = states[a["name"]]
                key = ("working_s" if st == "working" else
                       "pending_s" if pending.get(a["name"]) else "idle_s" if st == "idle"
                       else None)
                if key:
                    r = self._rollup(a, now)
                    r[key] = round(r[key] + dt, 1)

    # ------------------------------------------------------------------ run.json
    def write_run(self, now: float, ended: bool = False) -> None:
        """`run.json` at the state root, for the run viewer (viewer layout, schema 1)."""
        swarms = self.store.list_swarms()
        path = self.store.data_dir / "run.json"
        if not swarms and not path.exists():
            return  # nothing applied yet
        try:
            started = json.loads(path.read_text())["started_at"]
        except (OSError, ValueError, KeyError):
            started = datetime.fromtimestamp(now, UTC).isoformat()
        spec: SwarmSpec | None = swarms[0]["spec"] if swarms else None
        _write_json(path, {
            "schema": 1, "run": self.run_name, "swarm": spec.swarm if spec else None,
            "swarms": [sw["name"] for sw in swarms], "started_at": started,
            "ended_at": datetime.fromtimestamp(now, UTC).isoformat() if ended else None,
            "status": "ended" if ended else "running",
            "generation": swarms[0]["generation"] if swarms else None,
            "types": {t: {"harness": ts.harness.value, "model": ts.model, "replicas": ts.replicas}
                      for t, ts in spec.types.items()} if spec else {}})
        self._run_gens = {sw["name"]: sw["generation"] for sw in swarms}

    def close(self, now: float) -> None:
        """At shutdown: a last snapshot, the open hour's rollups, run.json as ended."""
        self.tick(now)
        with self._lock:
            hours, self._rollups = [self._rollups], {}
        self._write_rollups(hours)
        self.write_run(now, ended=True)

    # ------------------------------------------------------------------ snapshots
    def tick(self, now: float) -> None:
        """One snapshot per swarm: stored (30 s for 6 h, every 10th as 5-min for 7 d) and
        written as metrics/<swarm>/latest.json and metrics/<swarm>/<date>.jsonl (5-min)."""
        with self._lock:
            if self._since is None:
                self._since = now
            self._roll(now)
            usage, self._usage = self._usage, {}
            closed, self._closed = self._closed, []
        for (swarm, hour, type_name), u in usage.items():
            self.store.add_usage(swarm, hour, type_name, u)
        self._write_rollups(closed)
        swarms = self.store.list_swarms()
        if self._run_gens != {sw["name"]: sw["generation"] for sw in swarms}:
            self.write_run(now)
        for sw in swarms:
            spec: SwarmSpec = sw["spec"]
            self._tail(spec.swarm, now)
            body = self.snapshot(spec, now)
            self.store.add_health_sample(spec.swarm, now, "30s", body, KEEP_30S)
            sample = {"ts": now, "body": body}
            _write_json(self.store.data_dir / "metrics" / spec.swarm / "latest.json", sample)
            if now - self._last_5m.get(spec.swarm, 0.0) >= ROLLUP_S - 1:
                self._last_5m[spec.swarm] = now
                self.store.add_health_sample(spec.swarm, now, "5m", body, KEEP_5M)
                day = datetime.fromtimestamp(now, UTC).strftime("%Y-%m-%d")
                path = self.store.data_dir / "metrics" / spec.swarm / f"{day}.jsonl"
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(sample, sort_keys=True) + "\n")

    def _tail(self, swarm: str, now: float) -> None:
        """New swarm-stream events: relaunch counts and the recent control-plane actions."""
        key = swarm_stream(swarm)
        relaunches = self._relaunches.setdefault(swarm, Counter())
        recent = self._recent.setdefault(swarm, deque(maxlen=RECENT_EVENTS))
        live = swarm in self._cursor  # the first read is history: no wakes counted from it
        while evs := self.store.read_events(key, self._cursor.get(swarm, 0), limit=5000):
            self._cursor[swarm] = self._cursor.get(swarm, 0) + len(evs)
            for e in evs:
                data = e.get("data") or {}
                if live and e["type"] == "loop_wake" and (a := self.store.get_agent(e["agent"])):
                    with self._lock:
                        self._rollup(a, now)["wakes"] += 1
                if e["type"] in ("crash", "planned_restart"):
                    relaunches[data.get("target")] += 1
                if e["type"] not in NOISY:
                    with self._lock:
                        recent.append({"ts": e["ts"], "type": e["type"], "data": data})

    def recent_events(self, swarm: str) -> list[dict]:
        with self._lock:
            return list(self._recent.get(swarm, ()))

    def snapshot(self, spec: SwarmSpec, now: float) -> dict[str, Any]:
        swarm = spec.swarm
        workers = {w["id"]: w for w in self.store.list_workers(swarm)}
        relaunches = self._relaunches.get(swarm, Counter())
        out_workers: dict[str, Any] = {}
        tools: dict[str, dict[str, float]] = {}
        last, seen = self._tools.get(swarm, {}), {}
        prev_tick = self._prev_tick.get(swarm)
        self._prev_tick[swarm] = now
        for wid, w in workers.items():
            tel = json.loads(w.get("telemetry") or "{}")
            mem = tel.get("mem") or {}
            restore = tel.get("restore")
            if restore:
                restore = {**restore, "eta_s": _restore_eta(restore, now)}
            out_workers[wid] = {"mem": mem, "slope_mb_h": mem.get("slope_mb_h"),
                                "eta_limit_s": mem.get("eta_limit_s"), "restore": restore,
                                "relaunches": relaunches.get(wid, 0)}
            cur, prev = tel.get("tools") or {}, last.get(wid)
            seen[wid] = cur
            for name, c in cur.items():
                t = tools.setdefault(name, {"n": 0, "err": 0, "refused": 0, "max_s": 0.0})
                t["max_s"] = max(t["max_s"], c.get("max_s") or 0.0)
                if prev is None:
                    continue  # first sight of this worker: no interval to rate over
                p = prev.get(name) or {}
                if c.get("n", 0) < p.get("n", 0):
                    p = {}  # a new worker process: its counters restarted
                for k in ("n", "err", "refused"):
                    t[k] += max(0, c.get(k, 0) - p.get(k, 0))
        self._tools[swarm] = seen  # workers gone since are dropped
        minutes = (now - prev_tick) / 60 if prev_tick and now > prev_tick else None
        out_tools = {name: {
            "rate_per_min": round(t["n"] / minutes, 2) if minutes else None,
            "err_pct": round(100 * t["err"] / t["n"], 1) if t["n"] else None,
            "refused_pct": round(100 * t["refused"] / t["n"], 1) if t["n"] else None,
            "max_s": t["max_s"]} for name, t in tools.items()}

        types: dict[str, Any] = {}
        baseline = self._since or now
        agents = self.store.list_agents(swarm)
        states = {a["name"]: agent_state(a, workers.get(a["worker"]), now).value for a in agents}
        if prev_tick and now > prev_tick:
            self._sample_states(swarm, agents, states, min(now - prev_tick, 2 * SAMPLE_S), now)
        for a in agents:
            t = types.setdefault(a["type"], {"by_state": Counter(), "stuck": [], "gens": Counter()})
            state = states[a["name"]]
            t["by_state"][state] += 1
            t["gens"][a.get("rev") or "unknown"] += 1
            silent = now - self._last_event.get(a["name"], baseline)
            if state == "stuck" or (state == "working" and silent > self.stuck_s):
                t["stuck"].append({"agent": a["name"], "state": state, "silent_s": round(silent)})
        for t in types.values():
            t["by_state"], t["gens"] = dict(t["by_state"]), dict(t["gens"])
            t["stuck"].sort(key=lambda x: -x["silent_s"])

        hour = hour_of(now)
        by_type = {r["type"]: {k: r.get(k, 0) for k, _ in TOKEN_FIELDS}
                   for r in self.store.usage(swarm, hour)}
        body = {"world": self.world.get(swarm), "workers": out_workers, "types": types,
                "tools": out_tools, "providers": self.providers.get(swarm, {}),
                "tokens": {"hour": hour, "by_type": by_type}}
        body["alerts"] = alerts(body)
        return body

    async def run(self, interval: float = SAMPLE_S) -> None:
        """Probe, then snapshot, every `interval` s, until cancelled."""
        while True:
            now = time.time()
            await self.probe(now)
            try:
                await asyncio.to_thread(self.tick, now)
            except Exception:
                log.exception("health snapshot failed")
            await asyncio.sleep(interval)

    # ------------------------------------------------------------------ probes
    async def probe(self, now: float) -> None:
        async with httpx.AsyncClient(transport=self.transport, timeout=PROBE_TIMEOUT_S) as http:
            for sw in self.store.list_swarms():
                spec: SwarmSpec = sw["spec"]
                try:
                    await self._probe_world(http, spec, now)
                    await self._probe_providers(http, spec)
                except Exception:  # a probe never stops the loop
                    log.exception("health probe of %s failed", spec.swarm)

    async def _probe_world(self, http: httpx.AsyncClient, spec: SwarmSpec, now: float) -> None:
        w = spec.world
        if w is None or not w.endpoint:
            self.world.pop(spec.swarm, None)
            return
        token = world_token(spec, self.token, self.secrets)
        headers = {TOKEN_HEADER: token} if token else {}
        state: dict[str, Any] = {"up": False, "manifest_ms": None, "slurm_state": None,
                                 "metrics": None}
        t0 = time.monotonic()
        error = None
        try:
            r = await http.get(f"{w.endpoint}/manifest", headers=headers)
            state["manifest_ms"] = round((time.monotonic() - t0) * 1000, 1)
            state["up"] = r.status_code == 200
            if not state["up"]:
                error = f"manifest answered {r.status_code}"
            elif "metrics-v1" in (r.json().get("capabilities") or []):
                m = await http.get(f"{w.endpoint}/metrics", headers=headers)
                if m.status_code == 200:
                    state["metrics"] = m.json()  # stored as-is
        except (httpx.HTTPError, ValueError) as e:
            error = f"{type(e).__name__}: {e}"[:200]
        if w.slurm_job_name and hasattr(self.jobs, "state_by_name"):
            st = await asyncio.to_thread(self.jobs.state_by_name, w.slurm_job_name)
            state["slurm_state"] = st or "absent"
        self.world[spec.swarm] = state
        fails = 0 if state["up"] else self._fails.get(spec.swarm, 0) + 1
        prev, self._fails[spec.swarm] = self._fails.get(spec.swarm, 0), fails
        if fails == DOWN_AFTER:
            self._event(spec.swarm, now, "world_down", {
                "endpoint": w.endpoint, "error": error, "slurm_state": state["slurm_state"]})
        elif fails == 0 and prev >= DOWN_AFTER:
            self._event(spec.swarm, now, "world_up", {
                "endpoint": w.endpoint, "manifest_ms": state["manifest_ms"]})

    async def _probe_providers(self, http: httpx.AsyncClient, spec: SwarmSpec) -> None:
        out: dict[str, dict[str, float]] = {}
        for name, p in spec.providers.items():
            if not p.metrics:
                continue
            q: dict[str, float] = {}
            for base in p.base_urls or ([p.base_url] if p.base_url else []):
                root = base.rstrip("/").removesuffix("/v1")
                try:
                    r = await http.get(f"{root}/metrics")
                    r.raise_for_status()
                except httpx.HTTPError as e:
                    log.debug("provider %s metrics at %s: %s", name, root, e)
                    continue
                for k, v in parse_vllm(r.text).items():
                    q[k] = max(q.get(k, 0.0), v) if k == "kv" else q.get(k, 0.0) + v
            out[name] = q
        self.providers[spec.swarm] = out

    def _event(self, swarm: str, now: float, kind: str, data: dict[str, Any]) -> None:
        log.warning("%s: %s %s", swarm, kind, data)
        self.store.append_events(swarm, swarm_stream(swarm), [Event(
            type=kind, ts=datetime.fromtimestamp(now, UTC), agent="", data=data)])


def _restore_eta(r: dict[str, Any], now: float) -> float | None:
    """Seconds left at the rate so far: remaining files / (files done / time since started)."""
    done, total, started = r.get("files_done") or 0, r.get("files_total") or 0, r.get("started")
    if not done or not started or total <= done:
        return None
    elapsed = now - datetime.fromisoformat(started).timestamp()
    return round((total - done) * elapsed / done) if elapsed > 0 else None


def alerts(body: dict[str, Any]) -> list[dict[str, str]]:
    """What needs a look, from one snapshot body."""
    out = []
    w = body.get("world")
    if w and not w["up"]:
        slurm = w.get("slurm_state")
        why = (f" (Slurm job {slurm}: waiting for a node, not a crash)" if slurm == "PENDING"
               else f" (Slurm job {slurm})" if slurm else "")
        out.append({"kind": "world_down", "msg": f"world does not answer{why}"})
    for wid, wk in body["workers"].items():
        eta = wk.get("eta_limit_s")
        if eta is not None and eta < NEAR_LIMIT_S:
            out.append({"kind": "memory", "msg": f"{wid} reaches its memory limit in "
                        f"{eta / 60:.0f} min"})
    stuck = sum(len(t["stuck"]) for t in body["types"].values())
    if stuck:
        out.append({"kind": "stuck", "msg": f"{stuck} agents stuck or silent"})
    for name, t in body["tools"].items():
        for k in ("err_pct", "refused_pct"):
            if (t.get(k) or 0) >= TOOL_ALERT_PCT:
                out.append({"kind": "tool", "msg": f"{name}: {k.removesuffix('_pct')} {t[k]}%"})
    return out
