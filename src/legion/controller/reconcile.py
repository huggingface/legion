"""The reconcile loop: desired agents -> placement -> worker jobs.

`reconcile(store, jobs, now)` is one pass and returns the actions it took; the runner
calls it every few seconds in a thread (Hub calls block). A worker row is a stable *slot*
(`<swarm>-<type>-wNNNN`); its job is replaced on restart, its agents never move.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from legion.controller import naming
from legion.controller import spawn as spawning
from legion.controller.jobs import JobsClient, JobState
from legion.controller.prices import cost_usd, type_price_table
from legion.controller.store import Row, Store
from legion.models import (
    CRASHLOOP_THRESHOLD,
    LAUNCHES_PER_WINDOW,
    RATE_WINDOW,
    AgentState,
    AssignCommand,
    CommsSpec,
    DrainCommand,
    Event,
    Liveness,
    PromptCommand,
    RampHold,
    SwarmSpec,
    TypeHealth,
    TypeSpec,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerState,
    WorldSpec,
    max_workers,
    parse_duration,
    worker_type_spec,
)

log = logging.getLogger("legion.reconcile")

BACKOFF_CAP = 900.0  # s, 15 min
LAUNCH_BACKOFF_CAP = 300.0  # s, transient launch errors: crash_backoff, doubling up to 5 min
INSPECT_EVERY = 15.0  # s, min gap between job-status checks for one non-fresh worker
INSPECT_GRACE = 60.0  # s after launch before a worker with no heartbeat is checked at all
LIST_THRESHOLD = 10  # more non-fresh workers than this in a swarm: one list_jobs, not N inspects
LAUNCH_TIMEOUT = 900.0  # s, a RUNNING job that never heartbeats in this long is a crash
NUDGE_AFTER = 60.0  # s stuck before a stuck agent gets its type's resume_prompt
NUDGE_EVERY = 1800.0  # s, min gap between resume prompts to one stuck agent
NUDGE_MAX = 3  # resume prompts to one agent until it completes a turn
RETIRE_GRACE = 120.0  # s, an emptied worker gets this long to drain before it is cancelled
STABLE_AFTER = 600.0  # s, a heartbeat this long after launch resets the crash streak
# s, a planned end (drain, timeout) this soon after the job started backs off like a crash
# (without counting toward crashloop): 2x the default heartbeat, so the job did no real work
FAST_END = 60.0
# Not written to the swarm stream: per agent, or repeated every pass while the condition holds.
QUIET_ACTIONS = ("assign", "unassign", "budget_exhausted", "workers_full")


@dataclass
class Settings:
    public_url: str = "http://localhost:7860"
    token: str = ""
    secret_source: Mapping[str, str] = field(default_factory=lambda: os.environ)
    launches_per_window: int = LAUNCHES_PER_WINDOW
    rate_window: timedelta = RATE_WINDOW
    rng: random.Random = field(default_factory=random.Random)
    job_labels: dict[str, str] = field(default_factory=dict)  # deployment-wide extra job labels
    # LEGION_MAX_ROTATING: workers restoring or draining at once before a memory rotation is
    # sent (docs/slurm.md "Memory"); None: 10% of the swarm's worker slots, at least 1
    max_rotating: int | None = None


@dataclass
class Action:
    kind: str  # launch, planned_restart, crash, crashloop, cancel, assign, unassign, retire, ...
    target: str
    detail: str = ""


def launch_refused(exc: BaseException) -> bool:
    """Permanent launch failure: a Hub 4xx (not 408/429, which are worth retrying), or a
    client-side validation error (or one a backend marks `launch_refused`, e.g. a Slurm spec
    without its script). Anything else (5xx, network) is transient."""
    if getattr(exc, "launch_refused", False):
        return True
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return 400 <= status < 500 and status not in (408, 429)
    return type(exc).__name__ in ("HFValidationError", "BadRequestError")


def swarm_stream(swarm: str) -> str:
    """Event stream key for swarm-level events (agent names never contain a dot)."""
    return f"_swarm.{swarm}"


def worker_id_for(swarm: str, type_name: str, idx: int) -> str:
    return f"{swarm}-{type_name}-w{idx:04d}"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


# --------------------------------------------------------------------------- derived states


def last_seen(w: Row) -> float | None:
    """When the current job last reached us: its last heartbeat, or a later command poll or
    event post. Heartbeats can be throttled away (HTTP 429 at the Space edge)
    while other requests get through; only silence on every route makes a worker dead."""
    hb = w["last_heartbeat"]
    if hb is None:
        return None
    return max(hb, w.get("last_seen") or 0.0)


def worker_state(w: Row, now: float, lv: Liveness) -> WorkerState:
    """`lv`: the swarm's timings (`Store.liveness`)."""
    if w["job_id"] is None:
        return WorkerState.dead
    seen = last_seen(w)
    if seen is None:
        return WorkerState.launching
    age = now - seen
    if age > lv.dead_after:
        return WorkerState.dead
    if age > lv.unknown_after:
        return WorkerState.unknown
    return WorkerState.draining if w["draining"] or w["retire_at"] else WorkerState.running


def agent_state(a: Row, w: Row | None, now: float, lv: Liveness) -> AgentState:
    hb = a["last_heartbeat"]
    if w is None or w["job_id"] is None:
        return AgentState.dead if hb is not None else AgentState.unknown
    if worker_state(w, now, lv) == WorkerState.dead:
        return AgentState.dead
    if hb is None or (w["launched_at"] and hb < w["launched_at"]):
        return AgentState.unknown  # not yet reported by the current job
    age = now - hb
    if age > lv.dead_after:
        # dead if the worker's heartbeats arrive without it; unknown if they are not arriving
        # (the worker is alive by its polls, see last_seen, but its reports are stale)
        fresh = w["last_heartbeat"] is not None and now - w["last_heartbeat"] <= lv.unknown_after
        return AgentState.dead if fresh else AgentState.unknown
    if age > lv.unknown_after or not a["hb_state"]:
        return AgentState.unknown
    return AgentState(a["hb_state"])


# --------------------------------------------------------------------------- one pass


@dataclass
class _Pass:
    store: Store
    jobs: JobsClient
    now: float
    settings: Settings
    budget: int
    actions: list[Action] = field(default_factory=list)
    listed: dict[str, JobState] | None = None  # stages from one list; a job missing (ended) is inspected
    lv: Liveness = Liveness(0, 0, 0)  # noqa: RUF009 - a tuple; the swarm's, set per swarm
    local: bool = False  # the swarm's compute is local, set per swarm
    list_failed: bool = False  # the batch list raised: a missing job is unknown, not ended
    roster: dict[str, str | None] | None = None  # the swarm's, for its assigns (harbor_roster)

    def act(self, kind: str, target: str, detail: str = "") -> None:
        self.actions.append(Action(kind, target, detail))

    def job_state(self, job_id: str, namespace: str) -> JobState | None:
        if self.listed is not None and job_id in self.listed:
            return self.listed[job_id]
        if self.list_failed:  # no per-job inspect storm behind a broken listing; next pass
            return None
        try:
            return self.jobs.inspect(job_id, namespace=namespace)
        except Exception as exc:  # noqa: BLE001 - Hub hiccup: try again next pass
            log.warning("inspect %s failed: %s", job_id, type(exc).__name__)
            return None

    def cancel(self, job_id: str, namespace: str) -> None:
        try:
            self.jobs.cancel(job_id, namespace=namespace)
        except Exception as exc:  # noqa: BLE001 - the job may already be gone
            log.warning("cancel %s failed: %s", job_id, type(exc).__name__)


def _needs_check(w: Row, now: float, lv: Liveness, local: bool) -> bool:
    """Fresh, non-draining workers are trusted from heartbeats; others get a job-status check.
    A local worker is checked every pass (a process poll): its exit is a crash at once."""
    if w["job_id"] is None:
        return False
    if local or w.get("preempted") or w.get("drained"):  # synced and exiting: end it now
        return True
    seen = last_seen(w)
    if seen is None and now - (w["launched_at"] or now) < INSPECT_GRACE:
        return False
    if seen is not None and now - seen <= lv.unknown_after and not w["draining"] \
            and not w["retire_at"]:
        return False
    return w["last_inspect"] is None or now - w["last_inspect"] >= INSPECT_EVERY


def reconcile(
    store: Store, jobs: JobsClient, now: datetime, settings: Settings | None = None
) -> list[Action]:
    settings = settings or Settings()
    if settings.token and store.agent_token_key is None:
        spawning.set_token_key(store, settings.token)
    t = now.timestamp()
    used = store.launches_since(t - settings.rate_window.total_seconds())
    p = _Pass(store, jobs, t, settings, budget=settings.launches_per_window - used)
    for sw in store.list_swarms():
        spec: SwarmSpec = sw["spec"]
        if not sw["removing"]:
            spec = _ramp(p, spec)
        p.listed, p.lv, p.local = None, spec.compute.liveness(), spec.compute.kind == "local"
        p.list_failed = False
        first = len(p.actions)
        if sum(_needs_check(w, t, p.lv, p.local)
               for w in store.list_workers(spec.swarm)) > LIST_THRESHOLD:
            try:
                listing = jobs.list({"swarm": spec.swarm}, namespace=spec.namespace)
                p.listed = {j.job_id: JobState(j.stage, j.message) for j in listing}
            except Exception as exc:  # noqa: BLE001
                log.warning("list_jobs for %s failed: %s", spec.swarm, type(exc).__name__)
                p.listed, p.list_failed = {}, True
        names = sorted(set(spec.types) | set(store.type_names(spec.swarm)))
        # every type's agents first: an assign's roster lists the run's agents of all types
        for type_name in names:
            _sync_agents(p, spec, type_name, None if sw["removing"] else spec.types.get(type_name))
        p.roster = harbor_roster(store, spec)
        for type_name in names:
            ts = None if sw["removing"] else spec.types.get(type_name)
            _reconcile_type(p, spec, type_name, ts)
        if not sw["removing"]:
            _rotate(p, spec)
        _record(p, spec.swarm, p.actions[first:])
        gone = not store.list_workers(spec.swarm) and not store.list_agents(spec.swarm)
        if sw["removing"] and gone:
            store.delete_swarm(spec.swarm)
            p.act("remove_swarm", spec.swarm)
    return p.actions


def _record(p: _Pass, swarm: str, actions: list[Action]) -> None:
    """The pass's actions on the swarm event stream (docs/live.md): crash, planned_restart,
    crashloop, launch, cancel, retire, ..."""
    ts = datetime.fromtimestamp(p.now, UTC)
    evs = [Event(type=a.kind, ts=ts, agent="", data={"target": a.target, "detail": a.detail[:300]})
           for a in actions if a.kind not in QUIET_ACTIONS]
    if evs:
        p.store.append_events(swarm, swarm_stream(swarm), evs)


def rotation_room(store: Store, swarm: str, max_rotating: int | None, now: float) -> int:
    """Drains the rotation budget allows now: `max_rotating` (default 10% of the swarm's
    worker slots, at least 1) minus the workers restoring (launched, or a restore in flight)
    or draining."""
    workers = store.list_workers(swarm)
    budget = max_rotating or max(1, len(workers) // 10)
    busy = sum(1 for w in workers if w["job_id"] is not None and (
        w["draining"] or w["retire_at"] or w["last_heartbeat"] is None
        or json.loads(w.get("telemetry") or "{}").get("restore")))
    return budget - busy


def rotate_worker(store: Store, swarm: str, w: Row, now: float) -> None:
    """Drain one worker for a memory rotation (ends in a planned restart): `memory_rotation`."""
    mem = json.loads(w.get("telemetry") or "{}").get("mem") or {}
    store.enqueue_command(w["id"], DrainCommand().model_dump_json())
    store.update_worker(w["id"], draining=1)  # counted as draining from now on
    data = {"worker": w["id"], "nonreclaim_mb": mem.get("nonreclaim_mb"),
            "eta_limit_s": mem.get("eta_limit_s")}
    log.info("memory rotation: %s", data)
    store.append_events(swarm, swarm_stream(swarm), [Event(
        type="memory_rotation", ts=datetime.fromtimestamp(now, UTC), agent="", data=data)])


def inference_rev(ts: TypeSpec | None) -> str | None:
    """What a worker launched for `ts` holds of its inference block: the key (job secrets),
    and the endpoints, Pi's config and the context window its Trials get. A worker launched
    with another revision restarts (`_restart_stale`)."""
    r = ts.resolved_model if ts is not None else None
    if r is None:
        return None
    held = [r.inference.held().model_dump(mode="json"), r.model_id, r.context_window, r.headers]
    return hashlib.sha256(json.dumps(held, sort_keys=True).encode()).hexdigest()[:16]


def _restart_stale(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec | None) -> None:
    """Workers launched with another inference revision than the type's now, or an unknown one
    (docs/live.md): drained, each ends in a planned restart and its agents resume on a new job,
    launched with the current block and key. Checked every pass, so a launch that raced the
    update is caught by the next one. A job carries its revision (`INFERENCE_REV`), which a
    restored controller re-adopts with it. Each is a `restart` event on the swarm stream."""
    rev = inference_rev(ts)
    stale = [w for w in p.store.list_workers(spec.swarm, type_name)
             if rev is not None and w["job_id"] is not None and w["inference_rev"] != rev
             and not w["draining"] and w["retire_at"] is None]
    for w in stale:
        p.store.enqueue_command(w["id"], DrainCommand().model_dump_json())
        p.store.update_worker(w["id"], draining=1)
        p.act("restart", w["id"], "inference changed")


def _redrain(p: _Pass, w: Row) -> None:
    """A drain the controller sent (`draining`) that the current job has not reported (its last
    heartbeat says otherwise, or there is none yet), with none queued: its delivery was lost
    (the poll's response), so it goes again. The worker's drain is idempotent."""
    if (w["draining"] and w["job_id"] is not None
            and json.loads(w["telemetry"] or "{}").get("draining") is not True
            and not any(json.loads(c["body"]).get("kind") == "drain"
                        for c in p.store.commands_for(w["id"]))):
        p.store.enqueue_command(w["id"], DrainCommand().model_dump_json())
        p.act("redrain", w["id"])


def rotatable(w: Row, now: float, lv: Liveness) -> bool:
    return (w["job_id"] is not None and not w["draining"] and not w["retire_at"]
            and worker_state(w, now, lv) == WorkerState.running)


def _rotate(p: _Pass, spec: SwarmSpec) -> None:
    """Memory rotation (docs/slurm.md "Memory"): workers whose heartbeat says
    `wants_rotation` are drained, the smallest `eta_limit_s` first, while fewer than
    `max_rotating` workers are restoring (launched, or a restore in flight) or draining. The
    drain ends in a planned restart. A worker's own drain or preemption at its hard
    thresholds never waits for this budget (a preempted slot has already ended here)."""
    workers = p.store.list_workers(spec.swarm)
    tel = {w["id"]: json.loads(w.get("telemetry") or "{}") for w in workers}
    want = [w for w in workers if tel[w["id"]].get("wants_rotation") and rotatable(w, p.now, p.lv)]
    if not want:
        return
    eta = {w["id"]: (tel[w["id"]].get("mem") or {}).get("eta_limit_s") for w in want}
    room = rotation_room(p.store, spec.swarm, p.settings.max_rotating, p.now)
    for w in sorted(want, key=lambda w: (eta[w["id"]] is None, eta[w["id"]] or 0.0)
                    )[:max(0, room)]:
        rotate_worker(p.store, spec.swarm, w, p.now)


HOLD_RECHECK_S = 120.0  # a held step asks again this often (load_guard's 2-minute window)
HOLD_WARN_AFTER = 3  # consecutive holds before `ramp_held` carries a warning
VLLM_WINDOW_S = 150.0  # samples older than this are not looked at (2 min + one late probe)


def vllm_busy(servers: dict[str, list[list[float]]], h: RampHold, now: float) -> str | None:
    """Why a ramp step must wait (None: it may go), from the controller's vLLM samples
    (`[ts, waiting, running, completed]` per server, health.py). Was load_guard.capacity_ok."""
    if not servers:
        return "HOLD: no vLLM metrics (set `inference.metrics: true`)"
    details = []
    for i, url in enumerate(sorted(servers), 1):
        pts = [x for x in servers[url] if now - x[0] <= VLLM_WINDOW_S]
        if len(pts) < 3:
            return "HOLD: vLLM metrics unavailable or invalid"
        (t0, _, _, c0), (t1, w1, r1, c1) = pts[0], pts[-1]
        if c1 < c0:
            return "HOLD: vLLM counters reset during capacity check"
        rate = (c1 - c0) / max(t1 - t0, 0.001)
        tpr = r1 / rate if rate > 0 else (float("inf") if r1 else 0.0)
        details.append(f"server {i}: waiting={w1:g}, ~{tpr:.1f}s/request")
        if all(x[1] >= h.waiting for x in pts) or tpr > h.request_s:
            return "HOLD: " + "; ".join(details)
    return None


def _ramp(p: _Pass, spec: SwarmSpec) -> SwarmSpec:
    """`types.T.replicas_ramp` (docs/live.md "Ramp"): raise replicas along the ramp's schedule
    from the first pass that saw it, never above `max`. Replicas move only when a new step
    comes, and never down, so an operator's scale (up or down) between steps stays. The clock
    stops while the type is held or the swarm released (`legion stop`), so a resume does not jump the missed steps. A
    staggered raise adds one agent per `stagger`. `{t0, step, ...}` lives in the store
    (`ramp:<type>`): a restart keeps the schedule. Each raise rewrites the stored spec
    (re-read just before); each step writes `ramp_step`, each hold `ramp_held`."""
    flags = p.store.swarm_flags(spec.swarm)
    held = (flags.get("holds") or {}).get("types") or {}
    want: dict[str, tuple[int, dict | None]] = {}  # type -> (replicas now, ramp_step data)
    for name, ts in spec.types.items():
        if ts.replicas_ramp is None:
            continue
        key = f"ramp:{name}"
        old = p.store.get_control(spec.swarm, key)
        st = dict(old or {"t0": p.now, "step": -1})
        if name in held or flags.get("released"):
            st.setdefault("paused_at", p.now)
        else:
            if "paused_at" in st:
                st["t0"] += p.now - st.pop("paused_at")
            if (w := _ramp_type(p, spec, name, st)) is not None:
                want[name] = w
        if st != old:
            p.store.put_control(spec.swarm, key, st)
    if not want:
        return spec
    row = p.store.get_swarm(spec.swarm)
    fresh: SwarmSpec = row["spec"] if row else spec
    raised = {}
    for name, (n, data) in want.items():
        ts = fresh.types.get(name)
        if ts is None or ts.replicas_ramp is None:
            continue
        n = min(n, ts.replicas_ramp.max)  # a lowered max caps the raise, never lowers
        if n <= ts.replicas:
            continue
        raised[name] = ts.model_copy(update={"replicas": n})
        if data is not None:
            p.store.append_events(spec.swarm, swarm_stream(spec.swarm), [Event(
                type="ramp_step", ts=datetime.fromtimestamp(p.now, UTC), agent="", data=data)])
    if not raised:
        return fresh
    fresh = fresh.model_copy(update={"types": {**fresh.types, **raised}})
    p.store.update_swarm_spec(fresh)
    return fresh


def _ramp_type(p: _Pass, spec: SwarmSpec, name: str, st: dict) -> tuple[int, dict | None] | None:
    """The replicas one ramp type should get this pass (and its `ramp_step` data when a new
    step starts), updating its stored state `st`; None: nothing now."""
    r, live = spec.types[name].replicas_ramp, spec.types[name].replicas
    assert r is not None
    events = r.schedule()
    step = sum(1 for e in events if e[0] <= p.now - st["t0"]) - 1
    if step > st["step"]:
        _, n, gap = events[step]
        total = sum(t.replicas for t in spec.types.values()) - live + n
        if r.hold_if is not None and n > live and total > r.hold_if.above:
            if p.now - st.get("held_at", float("-inf")) < HOLD_RECHECK_S:
                return None
            why = vllm_busy(p.store.get_control(spec.swarm, "vllm") or {}, r.hold_if, p.now)
            if why is not None:
                st.update(held_at=p.now, holds=st.get("holds", 0) + 1)
                data = {"type": name, "replicas": live, "want": n, "step": step,
                        "holds": st["holds"], "reason": why}
                if st["holds"] >= HOLD_WARN_AFTER:
                    data["warning"] = (f"{st['holds']} consecutive capacity holds; replicas "
                                       f"remain {live}. Inspect the vLLM servers and calibrate "
                                       "hold_if.request_s; growth stays blocked.")
                    log.warning("%s: ramp %s held: %s", spec.swarm, name, data["warning"])
                p.store.append_events(spec.swarm, swarm_stream(spec.swarm), [Event(
                    type="ramp_held", ts=datetime.fromtimestamp(p.now, UTC), agent="",
                    data=data)])
                return None
        for k in ("held_at", "holds", "pending", "next_at", "gap"):
            st.pop(k, None)
        st["step"] = step
        data = {"type": name, "from": live, "replicas": n, "step": step, "max": r.max}
        if gap > 0 and n > live + 1:
            st.update(pending=n, gap=gap, next_at=p.now + gap)
            data["stagger_s"] = gap
            return live + 1, data
        return n, data
    if "pending" not in st or p.now < st["next_at"]:
        return None
    pending = st["pending"]
    if live + 1 >= pending:
        for k in ("pending", "next_at", "gap"):
            st.pop(k)
    else:
        st["next_at"] = p.now + st["gap"]
    return (live + 1, None) if live < pending else None


def _sync_agents(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec | None) -> None:
    """Desired agents; scale down removes the highest slots first. Spawned agents (parent set)
    are not replicas: they go with their parent, their type or the swarm (docs/spawn.md)."""
    store = p.store
    replicas = ts.replicas if ts else 0
    existing = {a["name"]: a for a in store.list_agents(spec.swarm, type_name)}
    reason = "type_removed" if type_name not in spec.types else "swarm_removed"
    for a in sorted(existing.values(), key=lambda a: a["slot"], reverse=True):
        if ts is None or a["removing"] or (a["parent"] is None and a["slot"] >= replicas):
            for r in spawning.remove_subtree(store, a["name"], reason, p.now):
                p.act("unassign", r["name"], r["worker"] or "")
    # words: a ramp's names are reserved up to its max, so rosters list them from the start
    most = max(replicas, ts.replicas_ramp.max if ts and ts.replicas_ramp else 0)
    names = naming.replica_names(store, spec, type_name, most, p.now)
    for slot, name in enumerate(names[:replicas]):
        if name not in existing:
            store.insert_agent(name, spec.swarm, type_name, slot, p.now)


def _reconcile_type(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec | None) -> None:
    store = p.store
    for w in store.list_workers(spec.swarm, type_name):
        _refresh_worker(p, spec, w, ts)
    _restart_stale(p, spec, type_name, ts)
    for w in store.list_workers(spec.swarm, type_name):
        _redrain(p, w)
    # `legion stop --release`: every worker drains (its agents sync) and exits; the rows stay
    # and nothing launches until `legion launch` clears the flag.
    released = ts is not None and bool(store.swarm_flags(spec.swarm).get("released"))
    if released:
        for w in store.list_workers(spec.swarm, type_name):
            if w["job_id"] is not None and not w["draining"] and w["retire_at"] is None:
                store.enqueue_command(w["id"], DrainCommand().model_dump_json())
                store.update_worker(w["id"], draining=1)
                p.act("release", w["id"], w["job_id"])

    # Placement: fill live slots in order, never move a placed agent, open new slots as needed.
    if ts is not None and not released:
        _place(p, spec, type_name, ts)
        _nudge_stuck(p, spec.swarm, type_name, ts.resume_prompt)

    # Empty workers drain and are cancelled after a grace period.
    counts = Counter(a["worker"] for a in store.list_agents(spec.swarm, type_name))
    for w in store.list_workers(spec.swarm, type_name):
        if counts[w["id"]]:
            continue
        if w["job_id"] is None:
            store.delete_worker(w["id"])
            p.act("remove_worker", w["id"])
        elif w["retire_at"] is None:
            store.enqueue_command(w["id"], DrainCommand().model_dump_json())
            store.update_worker(w["id"], retire_at=p.now + RETIRE_GRACE)
            p.act("retire", w["id"])
        elif p.now >= w["retire_at"]:
            p.cancel(w["job_id"], spec.namespace)
            store.delete_worker(w["id"])
            p.act("cancel", w["id"], w["job_id"])

    # Launch slots that have agents and no job.
    if ts is None or released or store.type_health(spec.swarm, type_name) != TypeHealth.ok:
        return  # released, crashlooping or launch_refused: wait for a launch / new apply
    for w in store.list_workers(spec.swarm, type_name):
        if w["job_id"] is not None or not counts[w["id"]]:
            continue
        if w["next_launch_at"] is not None and w["next_launch_at"] > p.now:
            continue
        if p.budget <= 0:
            p.act("budget_exhausted", w["id"])
            return
        _launch(p, spec, type_name, ts, w)
        if store.type_health(spec.swarm, type_name) != TypeHealth.ok:
            return  # refused: no more launches for this type


def _place(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec) -> None:
    store = p.store
    workers = [w for w in store.list_workers(spec.swarm, type_name) if w["retire_at"] is None]
    by_id = {w["id"]: w for w in workers}
    agents = store.list_agents(spec.swarm, type_name)
    counts = Counter(a["worker"] for a in agents if a["worker"] in by_id)
    unplaced = [a for a in agents if a["worker"] not in by_id]
    if not unplaced:
        return
    cap = max(1, ts.agents_per_worker)
    queue = iter(unplaced)
    pending = len(unplaced)

    def fill(w: Row) -> None:
        nonlocal pending
        while counts[w["id"]] < cap and pending:
            a = next(queue)
            pending -= 1
            counts[w["id"]] += 1
            store.update_agent(a["name"], worker=w["id"])
            if w["job_id"]:
                _send_assign(store, w["id"], a, type_name, ts, spec.tools_version, spec.comms,
                             spec.world, p.roster)
            p.act("assign", a["name"], w["id"])

    for w in workers:
        fill(w)
    # A type that declares `workers` never has more than initial + max_extra worker slots
    # (retiring ones included); agents that do not fit wait unplaced. Others: unbounded.
    slots = max_workers(ts) if ts.workers is not None else None
    while pending:
        if slots is not None and len(store.list_workers(spec.swarm, type_name)) >= slots:
            p.act("workers_full", f"{spec.swarm}/{type_name}", f"{pending} unplaced")
            return
        idx = store.next_worker_index(spec.swarm, type_name)
        wid = worker_id_for(spec.swarm, type_name, idx)
        store.insert_worker(wid, spec.swarm, type_name, idx, p.now)
        fill(store.get_worker(wid))


def can_resume(text: str | None, nudges: int, loop_st: dict | None) -> bool:
    """A stuck agent may still get its type's resume_prompt `text` (_nudge_stuck); False: it
    stays stuck until something else prompts it."""
    return bool(text) and nudges < NUDGE_MAX and (loop_st or {}).get("phase") != "done"


def _nudge_stuck(p: _Pass, swarm: str, type_name: str, text: str | None) -> None:
    """A newly stuck agent (its turn timed out) gets an `agent.stuck` event. One stuck for
    NUDGE_AFTER gets the type's resume_prompt (when it has one), at
    most every NUDGE_EVERY, NUDGE_MAX times until it completes a turn, and only with no prompt
    pending, no pause hold, no budget stop and its loop (if any) not done."""
    store, pending = p.store, None
    flags, loops = store.swarm_flags(swarm), store.loop_agents(swarm, type_name)
    for a in store.list_agents(swarm, type_name):
        stuck = a["hb_state"] == AgentState.stuck.value and not a["removing"] and agent_state(
            a, store.get_worker(a["worker"]) if a["worker"] else None, p.now, p.lv
        ) == AgentState.stuck
        since = (a["stuck_since"] or p.now) if stuck else None
        if stuck and a["stuck_since"] is None:
            p.act("stuck", a["name"], "turn timed out")  # `agent.stuck` on the swarm stream
        nudges = 0 if a["hb_state"] == AgentState.idle.value else a["nudges"]  # turn completed
        if since != a["stuck_since"] or nudges != a["nudges"]:
            store.update_agent(a["name"], stuck_since=since, nudges=nudges)
        if (not can_resume(text, nudges, loops.get(a["name"])) or since is None
                or p.now - since < NUDGE_AFTER
                or (a["nudged_at"] is not None and p.now - a["nudged_at"] < NUDGE_EVERY)
                or flags.get("budget_stop") or store.held(a, flags)):
            continue
        pending = store.pending_counts(swarm) if pending is None else pending
        if pending.get(a["name"]):
            continue
        prompt = PromptCommand(agent=a["name"], prompt_id=uuid.uuid4().hex, text=text)
        store.add_prompt(a["name"], prompt.model_dump_json(), p.now)
        store.update_agent(a["name"], nudged_at=p.now, nudges=nudges + 1)
        p.act("nudge", a["name"], f"stuck: resume_prompt queued ({nudges + 1}/{NUDGE_MAX})")


def harbor_roster(store: Store, spec: SwarmSpec) -> dict[str, str | None]:
    """With `naming: words`, each replica of the swarm's Harbor types (spawned children are not
    teammates) -> its role, the type's `harbor.role`: what the upstream launcher passes as the
    roster. A replica's name is reserved with its slot; a ramp's are, up to its max."""
    roster: dict[str, str | None] = {}
    for type_name, ts in spec.types.items():
        if ts.harbor and spec.naming == "words":
            most = max(ts.replicas, ts.replicas_ramp.max if ts.replicas_ramp else 0)
            slots = store.reserved_slot_names(spec.swarm, type_name)
            roster |= {slots[s]: ts.harbor.role for s in sorted(slots) if s < most}
    return roster


def _send_assign(store: Store, worker_id: str, a: Row, type_name: str, ts: TypeSpec,
                 tools_version: str | None = None, comms: CommsSpec | None = None,
                 world: WorldSpec | None = None, roster: dict | None = None) -> None:
    if a["removing"]:
        return  # mid-removal (docs/spawn.md): never re-assign it or re-issue its token
    resume = ts.resume_prompt if a["resume_pending"] else None
    # A provider with max_working admits a resume prompt like any other (api.py `admit`): it
    # goes to pending_prompts after the assign instead of riding inside it.
    r = ts.resolved_model
    # Held as a pending prompt (not inside the assign) when admission must see it: a provider
    # with max_working, or a swarm under the hard budget stop or a paused agent (docs/loop.md),
    # where pending prompts are not delivered until the budget is raised or the hold lifted.
    flags = store.swarm_flags(a["swarm"])
    hard = flags.get("budget_stop") == "hard" or store.held(a, flags)  # or paused
    hold = bool(resume) and (hard or bool(r and r.inference.max_working))
    held, resume = (resume, None) if hold else (None, resume)
    if ts.harbor and not ts.harbor.roster:  # words: the run's agents (harbor_roster), with roles
        if roster is None:
            sw = store.get_swarm(a["swarm"])
            roster = harbor_roster(store, sw["spec"]) if sw else {}
        ts = ts.model_copy(update={"harbor": ts.harbor.model_copy(update={"roster": roster})})
    cmd = AssignCommand(
        agent=a["name"], type=type_name, spec=worker_type_spec(ts), restore=True, resume_prompt=resume,
        parent=a["parent"], agent_token=spawning.agent_token(store, a["name"], a["swarm"], time.time()),
        tools_version=tools_version,
        comms=comms,
        world=world.model_copy(update={"down_pause_s": None}) if world else None,  # controller only
    )
    store.enqueue_command(worker_id, cmd.model_dump_json())
    if a["resume_pending"]:
        store.update_agent(a["name"], resume_pending=0)
    if held:
        prompt = PromptCommand(agent=a["name"], prompt_id=uuid.uuid4().hex, text=held)
        store.add_prompt(a["name"], prompt.model_dump_json(), time.time())


def _refresh_worker(p: _Pass, spec: SwarmSpec, w: Row, ts: TypeSpec | None) -> None:
    if not _needs_check(w, p.now, p.lv, p.local):
        if w["job_id"] is not None:
            p.store.update_worker(w["id"], state=worker_state(w, p.now, p.lv).value)
        return
    p.store.update_worker(w["id"], last_inspect=p.now)
    if w.get("preempted"):
        # docs/slurm.md: the worker's last heartbeat said it was preempted and has synced.
        # End the slot without waiting for the scheduler (squeue/sacct lag, a watcher that
        # cancels 30 s later): the relaunch registers a fresh incarnation and restores.
        p.cancel(w["job_id"], spec.namespace)
        _job_ended(p, spec, w, ts, JobState("ERROR", "preempted: worker reported it",
                                            preempted=True))
        return
    if w.get("drained") and not p.local:  # a local worker's exit is seen at once anyway
        # The last heartbeat after a finished drain: synced, exiting 0. End the slot as its job's
        # COMPLETED would, without waiting up to INSPECT_EVERY for the scheduler to show it; the
        # cancel only makes sure the job never lingers on (compute freed).
        p.cancel(w["job_id"], spec.namespace)
        _job_ended(p, spec, w, ts, JobState("COMPLETED", "drained: worker reported it"))
        return
    st = p.job_state(w["job_id"], spec.namespace)
    if st is None:
        return
    if st.terminal:
        _job_ended(p, spec, w, ts, st)
        return
    if st.stage == "SCHEDULING":  # queued (Slurm, HF Jobs): the launch timeout starts later
        p.store.update_worker(w["id"], started_at=p.now)
    hb, seen = w["last_heartbeat"], last_seen(w)
    silent = seen is not None and p.now - seen > p.lv.dead_after  # no heartbeat, poll or event post
    start = w["started_at"] or w["launched_at"]
    never = hb is None and st.stage != "SCHEDULING" and p.now - start > LAUNCH_TIMEOUT
    if silent or never:
        p.cancel(w["job_id"], spec.namespace)
        p.act("cancel", w["id"], w["job_id"])
        _job_ended(p, spec, w, ts, JobState("CANCELED", "unresponsive"))
        return
    p.store.update_worker(w["id"], state=worker_state(w, p.now, p.lv).value,
                          pending_reason=st.message if st.stage == "SCHEDULING" else None)


def _job_ended(p: _Pass, spec: SwarmSpec, w: Row, ts: TypeSpec | None, st: JobState) -> None:
    store, wid = p.store, w["id"]
    store.clear_commands(wid)
    if w["retire_at"] is not None:
        store.delete_worker(wid)
        p.act("remove_worker", wid, st.stage)
        return
    drain = parse_duration(ts.drain_window).total_seconds() if ts else 0.0
    near_deadline = w["deadline"] is not None and p.now >= w["deadline"] - drain
    planned = st.preempted or (st.stage == "ERROR" and st.message == "Job timeout") or (
        st.stage == "COMPLETED" and (bool(w["draining"]) or near_deadline)
    )
    common = {
        "job_id": None,
        "incarnation": None,
        "state": WorkerState.dead.value,
        "last_heartbeat": None,
        "last_seen": None,
        "draining": 0,
        "preempted": 0,
        "drained": 0,
    }
    released = bool(store.swarm_flags(w["swarm"]).get("released"))
    if not planned and released:
        # killed mid-drain (`stop --release`): no final heartbeat, so its agents' syncs are unknown
        why = f"worker {wid} ended before its final sync ({st.stage}: {st.message})"[:300]
        lost = {a["name"]: why for a in store.list_agents(worker=wid)}
        if lost:
            store.put_control(w["swarm"], "unsynced",
                              {**lost, **(store.get_control(w["swarm"], "unsynced") or {})})
    if planned:
        lived = p.now - (w["started_at"] or w["launched_at"] or p.now)
        # a preemption is the scheduler's doing: relaunched at once to restore
        fast = w["fast_ends"] + 1 if lived < FAST_END and not (released or st.preempted) else 0
        backoff = min(p.lv.crash_backoff * 2 ** (fast - 1), BACKOFF_CAP) if fast else 0.0
        store.update_worker(wid, **common, consecutive_crashes=0, fast_ends=fast,
                            next_launch_at=p.now + backoff)
        # released (`stop --release`): it ends for good, nothing relaunches it until a launch
        why = f"{st.stage}: {st.message}" if st.message else st.stage
        p.act("end" if released else "planned_restart", wid,
              f"{why}; ended {lived:.0f}s after start, backoff {backoff:.0f}s" if fast else why)
        if st.preempted:  # turns cut by the reclaim resume after the restore
            _interrupt_turns(p, w, st)
        return
    streak = w["consecutive_crashes"] + 1
    backoff = min(p.lv.crash_backoff * 2 ** (streak - 1), BACKOFF_CAP)
    store.update_worker(
        wid,
        **common,
        last_error=_crash_reason(p, spec, w, st),
        crashes=w["crashes"] + 1,
        consecutive_crashes=streak,
        next_launch_at=p.now + backoff,
    )
    p.act("crash", wid, f"{st.stage}: {st.message}; backoff {backoff:.0f}s")
    _interrupt_turns(p, w, st)
    if streak >= CRASHLOOP_THRESHOLD:
        store.set_type_health(w["swarm"], w["type"], TypeHealth.crashlooping.value)
        p.act("crashloop", f"{w['swarm']}/{w['type']}")


def _interrupt_turns(p: _Pass, w: Row, st: JobState) -> None:
    """Agents whose turn the ended job cut: a `turn_interrupted` event and a resume prompt
    after the restore (unless the worker already reported the turn finished)."""
    store, wid = p.store, w["id"]
    for a in store.list_agents(worker=wid):
        if a["turn_id"]:
            ev = Event(
                type="turn_interrupted",
                ts=datetime.fromtimestamp(p.now, UTC),
                agent=a["name"],
                turn_id=a["turn_id"],
                data={"worker": wid, "job_id": w["job_id"], "reason": f"{st.stage}: {st.message}"},
            )
            store.append_events(a["swarm"], a["name"], [ev])
            store.update_agent(a["name"], resume_pending=1, turn_id=None)


def _file_tail(path: str, n: int = 5) -> list[str]:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return [ln.strip() for ln in f.read().splitlines()[-n:] if ln.strip()]
    except OSError:
        return []


def _crash_reason(p: _Pass, spec: SwarmSpec, w: Row, st: JobState) -> str:
    """`STAGE: message`, plus the job's last log lines (one Hub call per crash) when the
    jobs client can fetch them: a boot failure (wheel download 401, missing key) is only
    visible there."""
    reason = f"{st.stage}: {st.message}" if st.message else st.stage
    tail = getattr(p.jobs, "logs_tail", None)
    if tail is not None and w["job_id"]:
        try:
            lines = [ln.strip() for ln in tail(w["job_id"], namespace=spec.namespace) if ln.strip()]
        except Exception as exc:  # noqa: BLE001 - logs are a nicety; never break the pass
            log.warning("logs of %s failed: %s", w["job_id"], type(exc).__name__)
            lines = []
        if not lines and w.get("log_path"):  # a job an earlier controller launched
            lines = _file_tail(w["log_path"])
        if lines:
            reason += " | " + " / ".join(lines)
    return reason[:500]


def _launch(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec, w: Row) -> None:
    # A refused launch marks the type launch_refused: no slot of it launches again until the
    # spec is re-applied. Transient errors back off per type, capped at LAUNCH_BACKOFF_CAP.
    store, s, wid = p.store, p.settings, w["id"]
    lifetime = parse_duration(ts.worker_lifetime).total_seconds()
    jitter = parse_duration(ts.lifetime_jitter).total_seconds()
    timeout = lifetime + s.rng.uniform(0, jitter)
    deadline = p.now + timeout
    # Binds polls and heartbeats to this job: a parked poll or a late heartbeat from the job
    # this one replaces must not take the new job's commands (see the API).
    incarnation = uuid.uuid4().hex
    env = {
        **ts.env,
        "LEGION_URL": s.public_url,
        "WORKER_ID": wid,
        "WORKER_INCARNATION": incarnation,
        "SWARM": spec.swarm,
        "TYPE": type_name,
        "DEADLINE": _iso(deadline),
        "DRAIN_WINDOW": ts.drain_window,  # known before the first assign arrives
        "BUCKET_ID": spec.bucket_id,
        # the inference block it holds, re-adopted with the job by a restored controller
        **({"INFERENCE_REV": rev} if (rev := inference_rev(ts)) else {}),
    }
    secrets = {"LEGION_TOKEN": s.token}
    names = list(ts.secrets)
    if ts.resolved_model and ts.resolved_model.inference.key_env:  # the key, by name, for the worker
        names.append(ts.resolved_model.inference.key_env)
    for name in dict.fromkeys(names):
        if name in s.secret_source:
            secrets[name] = s.secret_source[name]
        else:
            log.warning(
                "secret %s for %s/%s is not set on the controller", name, spec.swarm, type_name
            )
    labels = {**s.job_labels, "swarm": spec.swarm, "type": type_name, "worker": wid}
    p.budget -= 1
    store.record_launch(p.now)
    try:
        # backends configured per swarm (compute slurm) get the spec's compute block
        extra = {"compute": spec.compute} if getattr(p.jobs, "uses_compute_spec", False) else {}
        job_id = p.jobs.launch(wid, ts, env, secrets, labels, timeout, namespace=spec.namespace,
                               **extra)
    except Exception as exc:  # noqa: BLE001 - any Hub failure: keep the loop alive
        # the Hub's message says why (a 400 names the bad field); class name alone does not
        msg = f"{type(exc).__name__}: {str(exc)[:500]}"
        log.warning("launch of %s failed: %s", wid, msg)
        store.update_worker(wid, last_error=f"launch failed: {msg}"[:500])
        if launch_refused(exc):
            store.set_type_health(spec.swarm, type_name, TypeHealth.launch_refused.value)
            store.set_launch_error(spec.swarm, type_name, msg, 0)
            p.act("launch_refused", f"{spec.swarm}/{type_name}", msg)
            return
        failures = store.launch_error(spec.swarm, type_name)[1] + 1
        backoff = min(p.lv.crash_backoff * 2 ** (failures - 1), LAUNCH_BACKOFF_CAP)
        store.set_launch_error(spec.swarm, type_name, msg, failures)
        store.update_worker(wid, next_launch_at=p.now + backoff)
        p.act("launch_failed", wid, f"{msg}; backoff {backoff:.0f}s")
        return
    store.set_launch_error(spec.swarm, type_name, None, 0)
    log_path = None
    if (path_of := getattr(p.jobs, "log_path", None)) is not None:
        try:
            log_path = path_of(job_id)
        except Exception:  # noqa: BLE001 - a nicety
            log_path = None
    store.clear_commands(wid)
    store.update_worker(
        wid,
        job_id=job_id,
        incarnation=incarnation,
        launched_at=p.now,
        started_at=None,
        deadline=deadline,
        last_heartbeat=None,
        last_seen=None,
        last_inspect=None,
        draining=0,
        preempted=0,
        drained=0,
        log_path=log_path,
        state=WorkerState.launching.value,
        next_launch_at=None,
        inference_rev=inference_rev(ts),  # the block this job holds (_restart_stale)
        telemetry=None,  # the job before's: nothing this job reported
    )
    for a in store.list_agents(worker=wid):
        _send_assign(store, wid, a, type_name, ts, spec.tools_version, spec.comms, spec.world,
                     p.roster)
    p.act("launch", wid, job_id)


# --------------------------------------------------------------------------- heartbeats


def record_heartbeat(store: Store, worker_id: str, hb: WorkerHeartbeat, now: datetime) -> bool:
    """Apply a worker heartbeat. Returns False if the worker slot has no live job."""
    t = now.timestamp()
    w = store.get_worker(worker_id)
    if w is None or w["job_id"] is None:
        return False
    fields = {
        "last_heartbeat": t,
        "last_seen": t,
        "deadline": hb.deadline.timestamp(),
        "draining": int(hb.draining or w["draining"]),  # a drain we sent stays until the job ends
        "state": (WorkerState.draining if hb.draining else WorkerState.running).value,
        "telemetry": hb.model_dump_json(include={"mem", "restore", "tools", "knobs_rev",
                                                 "wants_rotation", "draining"}),
    }
    if hb.preempted:  # the next reconcile pass ends the slot as a planned restart
        fields["preempted"] = 1
    if hb.drained:  # the next reconcile pass ends the slot (a drain's planned end)
        fields["drained"] = 1
    if hb.unsynced and (hb.draining or hb.preempted):  # a final sync failed: never swallowed
        fields["last_error"] = ("final sync failed: " + ", ".join(sorted(hb.unsynced)))[:500]
        store.put_control(w["swarm"], "unsynced",
                          {**(store.get_control(w["swarm"], "unsynced") or {}), **hb.unsynced})
        store.append_events(w["swarm"], swarm_stream(w["swarm"]), [Event(
            type="sync_failed", ts=now, agent="", data={"worker": worker_id,
                                                        "agents": hb.unsynced})])
    for name in hb.interrupted:  # cut by a preemption or a drain deadline: resume after the restore
        a = store.get_agent(name)
        if a is not None and a["worker"] == worker_id:
            store.update_agent(name, resume_pending=1)
    if (w["consecutive_crashes"] or w["fast_ends"]) and t - (w["launched_at"] or t) >= STABLE_AFTER:
        fields["consecutive_crashes"] = fields["fast_ends"] = 0
    store.update_worker(worker_id, **fields)
    queued = {c["body"] for c in store.commands_for(worker_id)}
    sw = store.get_swarm(w["swarm"])
    table = type_price_table(sw["spec"].types.get(w["type"]) if sw else None)

    def cost_fn(tokens):
        return cost_usd(tokens, table)

    for ah in hb.agents:
        a = store.get_agent(ah.name)
        if (
            a is not None
            and a["worker"] is None
            and (a["swarm"], a["type"])
            == (
                w["swarm"],
                w["type"],
            )
        ):
            store.update_agent(ah.name, worker=worker_id)  # adopt (controller lost its state)
        elif a is None or a["worker"] != worker_id:
            body = UnassignCommand(agent=ah.name).model_dump_json()
            if body not in queued:
                store.enqueue_command(worker_id, body)
            continue
        store.record_agent_heartbeat(
            ah.name,
            {
                "hb_state": ah.state,
                "turn_id": ah.turn_id,
                "idle_since": ah.idle_since.timestamp() if ah.idle_since else None,
                "session_id": ah.session_id,
                "resumed": None if ah.resumed is None else int(ah.resumed),
                "turns": ah.turns,
                "last_sync": ah.last_sync.timestamp() if ah.last_sync else None,
                "msgboard_principal": ah.msgboard_principal,
                "rev": ah.rev,
                "start_rev": ah.start_rev,
                "endpoint": ah.endpoint,
                "last_heartbeat": t,
            },
            ah.tokens,
            cost_fn,
        )
    if not hb.draining and not w["retire_at"]:
        _reassign_missing(store, w, {ah.name for ah in hb.agents}, queued)
    return True


def _reassign_missing(store: Store, w: Row, reported: set[str], queued: set[str]) -> None:
    """Agents placed on this worker that its heartbeat does not list, with no assign queued,
    lost their assign on the way (a dropped poll response). Send it again; a worker treats a
    repeated assign for an agent it already hosts as a spec update, so this is idempotent."""
    missing = [a for a in store.list_agents(worker=w["id"])
               if a["name"] not in reported and not a["removing"]]
    if not missing:
        return
    sw = store.get_swarm(w["swarm"])
    ts = sw["spec"].types.get(w["type"]) if sw else None
    if ts is None:
        return
    pending = {AssignCommand.model_validate_json(b).agent for b in queued if '"assign"' in b}
    roster = None  # computed once, for the first re-sent assign
    for a in missing:
        if a["name"] not in pending:
            log.warning("re-sending assign for %s: missing from %s heartbeat", a["name"], w["id"])
            roster = roster if roster is not None else harbor_roster(store, sw["spec"])
            _send_assign(store, w["id"], a, w["type"], ts, sw["spec"].tools_version,
                         sw["spec"].comms, sw["spec"].world, roster)


# --------------------------------------------------------------------------- startup and runner


def rebuild_from_jobs(store: Store, jobs: JobsClient, now: datetime) -> list[Action]:
    """After a controller restart, re-adopt live worker jobs by label, then wait for heartbeats."""
    t = now.timestamp()
    actions: list[Action] = []
    for sw in store.list_swarms():
        spec: SwarmSpec = sw["spec"]
        live: dict[str, list] = {}
        for j in jobs.list({"swarm": spec.swarm}, namespace=spec.namespace):
            wid = j.labels.get("worker") or j.env.get("WORKER_ID")
            type_name = j.labels.get("type") or j.env.get("TYPE")
            if wid and type_name and j.stage not in ("COMPLETED", "CANCELED", "ERROR", "DELETED"):
                live.setdefault(wid, []).append((j, type_name))
        for wid, found in live.items():
            w = store.get_worker(wid)
            keep = next((x for x in found if w and x[0].job_id == w["job_id"]), found[-1])
            for j, _ in found:
                if j is not keep[0]:
                    jobs.cancel(j.job_id, namespace=spec.namespace)
                    actions.append(Action("cancel", wid, j.job_id))
            j, type_name = keep
            if w is None:
                idx = int(wid.rsplit("-w", 1)[-1]) if wid.rsplit("-w", 1)[-1].isdigit() else 0
                store.insert_worker(wid, spec.swarm, type_name, idx, t)
                store.bump_worker_index(spec.swarm, type_name, idx + 1)
                w = store.get_worker(wid)
            deadline = None
            if j.env.get("DEADLINE"):
                deadline = datetime.fromisoformat(j.env["DEADLINE"]).timestamp()
            store.update_worker(
                wid,
                job_id=j.job_id,
                incarnation=j.env.get("WORKER_INCARNATION"),
                deadline=deadline or w["deadline"],
                launched_at=w["launched_at"] or t,
                # one heartbeat interval of grace before the age rules apply
                last_heartbeat=t if w["last_heartbeat"] is not None else None,
                # the adopted job's own (unknown: None, restarted by _restart_stale)
                inference_rev=j.env.get("INFERENCE_REV"),
            )
            actions.append(Action("adopt", wid, j.job_id))
    return actions


async def run_reconciler(
    store: Store, jobs: JobsClient, settings: Settings, interval: float = 5.0, wake=None
) -> None:
    """Background task: rebuild once, then reconcile every `interval` s (or when woken)."""
    now = lambda: datetime.now(UTC)
    try:
        await asyncio.to_thread(rebuild_from_jobs, store, jobs, now())
    except Exception:
        log.exception("rebuild from jobs failed")
    while True:
        try:
            for a in await asyncio.to_thread(reconcile, store, jobs, now(), settings):
                if a.kind not in ("assign", "budget_exhausted"):
                    log.info("%s %s %s", a.kind, a.target, a.detail)
        except Exception:
            log.exception("reconcile pass failed")
        if wake is not None:
            await wake(interval)
        else:
            await asyncio.sleep(interval)
