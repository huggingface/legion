"""The reconcile loop: desired agents -> placement -> worker jobs.

`reconcile(store, jobs, now)` is one pass and returns the actions it took; the runner
calls it every few seconds in a thread (Hub calls block). A worker row is a stable *slot*
(`<swarm>-<type>-wNNNN`); its job is replaced on restart, its agents never move.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import time
import uuid
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from flock.controller import spawn as spawning
from flock.controller.jobs import JobsClient, JobState
from flock.controller.prices import cost_usd, type_price_table
from flock.controller.store import Row, Store
from flock.models import (
    CRASHLOOP_THRESHOLD,
    DEAD_AFTER,
    LAUNCHES_PER_WINDOW,
    RATE_WINDOW,
    UNKNOWN_AFTER,
    AgentState,
    AssignCommand,
    DrainCommand,
    Event,
    PromptCommand,
    SwarmSpec,
    TypeHealth,
    TypeSpec,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerState,
    agent_name,
    max_workers,
    parse_duration,
)

log = logging.getLogger("flock.reconcile")

BACKOFF_BASE = 30.0  # s, first crash backoff
BACKOFF_CAP = 900.0  # s, 15 min
LAUNCH_BACKOFF_CAP = 300.0  # s, transient launch errors back off 30 s, 60 s, ... up to 5 min
INSPECT_EVERY = 15.0  # s, min gap between job-status checks for one non-fresh worker
INSPECT_GRACE = 60.0  # s after launch before a worker with no heartbeat is checked at all
LIST_THRESHOLD = 10  # more non-fresh workers than this in a swarm: one list_jobs, not N inspects
LAUNCH_TIMEOUT = 900.0  # s, a RUNNING job that never heartbeats in this long is a crash
RETIRE_GRACE = 120.0  # s, an emptied worker gets this long to drain before it is cancelled
STABLE_AFTER = 600.0  # s, a heartbeat this long after launch resets the crash streak
_UNKNOWN = UNKNOWN_AFTER.total_seconds()
_DEAD = DEAD_AFTER.total_seconds()


@dataclass
class Settings:
    public_url: str = "http://localhost:7860"
    token: str = ""
    secret_source: Mapping[str, str] = field(default_factory=lambda: os.environ)
    launches_per_window: int = LAUNCHES_PER_WINDOW
    rate_window: timedelta = RATE_WINDOW
    rng: random.Random = field(default_factory=random.Random)
    job_labels: dict[str, str] = field(default_factory=dict)  # deployment-wide extra job labels


@dataclass
class Action:
    kind: str  # launch, planned_restart, crash, crashloop, cancel, assign, unassign, retire, ...
    target: str
    detail: str = ""


def launch_refused(exc: BaseException) -> bool:
    """Permanent launch failure: a Hub 4xx (not 408/429, which are worth retrying), or a
    client-side validation error. Anything else (5xx, network) is transient."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if isinstance(status, int):
        return 400 <= status < 500 and status not in (408, 429)
    return type(exc).__name__ in ("HFValidationError", "BadRequestError")


def worker_id_for(swarm: str, type_name: str, idx: int) -> str:
    return f"{swarm}-{type_name}-w{idx:04d}"


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, UTC).isoformat()


# --------------------------------------------------------------------------- derived states


def worker_state(w: Row, now: float) -> WorkerState:
    if w["job_id"] is None:
        return WorkerState.dead
    hb = w["last_heartbeat"]
    if hb is None:
        return WorkerState.launching
    age = now - hb
    if age > _DEAD:
        return WorkerState.dead
    if age > _UNKNOWN:
        return WorkerState.unknown
    return WorkerState.draining if w["draining"] or w["retire_at"] else WorkerState.running


def agent_state(a: Row, w: Row | None, now: float) -> AgentState:
    hb = a["last_heartbeat"]
    if w is None or w["job_id"] is None:
        return AgentState.dead if hb is not None else AgentState.unknown
    if worker_state(w, now) == WorkerState.dead:
        return AgentState.dead
    if hb is None or (w["launched_at"] and hb < w["launched_at"]):
        return AgentState.unknown  # not yet reported by the current job
    age = now - hb
    if age > _DEAD:
        return AgentState.dead
    if age > _UNKNOWN or not a["hb_state"]:
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
    listed: dict[str, JobState] | None = None  # job stages from one list_jobs, if used

    def act(self, kind: str, target: str, detail: str = "") -> None:
        self.actions.append(Action(kind, target, detail))

    def job_state(self, job_id: str, namespace: str) -> JobState | None:
        if self.listed is not None:
            return self.listed.get(job_id)
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


def _needs_check(w: Row, now: float) -> bool:
    """Fresh, non-draining workers are trusted from heartbeats; others get a job-status check."""
    if w["job_id"] is None:
        return False
    hb = w["last_heartbeat"]
    if hb is None and now - (w["launched_at"] or now) < INSPECT_GRACE:
        return False
    if hb is not None and now - hb <= _UNKNOWN and not w["draining"] and not w["retire_at"]:
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
        p.listed = None
        if sum(_needs_check(w, t) for w in store.list_workers(spec.swarm)) > LIST_THRESHOLD:
            try:
                listing = jobs.list({"swarm": spec.swarm}, namespace=spec.namespace)
                p.listed = {j.job_id: JobState(j.stage, j.message) for j in listing}
            except Exception as exc:  # noqa: BLE001
                log.warning("list_jobs for %s failed: %s", spec.swarm, type(exc).__name__)
                p.listed = {}
        for type_name in sorted(set(spec.types) | set(store.type_names(spec.swarm))):
            ts = None if sw["removing"] else spec.types.get(type_name)
            _reconcile_type(p, spec, type_name, ts)
        gone = not store.list_workers(spec.swarm) and not store.list_agents(spec.swarm)
        if sw["removing"] and gone:
            store.delete_swarm(spec.swarm)
            p.act("remove_swarm", spec.swarm)
    return p.actions


def _reconcile_type(p: _Pass, spec: SwarmSpec, type_name: str, ts: TypeSpec | None) -> None:
    store = p.store
    for w in store.list_workers(spec.swarm, type_name):
        _refresh_worker(p, spec, w, ts)

    # Desired agents; scale down removes the highest slots first. Spawned agents (parent set)
    # are not replicas: they go with their parent, their type or the swarm (docs/spawn.md).
    replicas = ts.replicas if ts else 0
    existing = {a["name"]: a for a in store.list_agents(spec.swarm, type_name)}
    reason = "type_removed" if type_name not in spec.types else "swarm_removed"
    for a in sorted(existing.values(), key=lambda a: a["slot"], reverse=True):
        if ts is None or a["removing"] or (a["parent"] is None and a["slot"] >= replicas):
            for r in spawning.remove_subtree(store, a["name"], reason, p.now):
                p.act("unassign", r["name"], r["worker"] or "")
    for slot in range(replicas):
        name = agent_name(type_name, slot)
        if name not in existing:
            store.insert_agent(name, spec.swarm, type_name, slot, p.now)

    # Placement: fill live slots in order, never move a placed agent, open new slots as needed.
    if ts is not None:
        _place(p, spec, type_name, ts)

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
    if ts is None or store.type_health(spec.swarm, type_name) != TypeHealth.ok:
        return  # crashlooping or launch_refused: wait for a new apply
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
                _send_assign(store, w["id"], a, type_name, ts)
            p.act("assign", a["name"], w["id"])

    for w in workers:
        fill(w)
    # A type that declares `workers` never has more than initial + max_extra worker slots
    # (retiring ones included); agents that do not fit wait unplaced. Others: v1, unbounded.
    slots = max_workers(ts) if ts.workers is not None else None
    while pending:
        if slots is not None and len(store.list_workers(spec.swarm, type_name)) >= slots:
            p.act("workers_full", f"{spec.swarm}/{type_name}", f"{pending} unplaced")
            return
        idx = store.next_worker_index(spec.swarm, type_name)
        wid = worker_id_for(spec.swarm, type_name, idx)
        store.insert_worker(wid, spec.swarm, type_name, idx, p.now)
        fill(store.get_worker(wid))


def _send_assign(store: Store, worker_id: str, a: Row, type_name: str, ts: TypeSpec) -> None:
    if a["removing"]:
        return  # mid-removal (docs/spawn.md): never re-assign it or re-issue its token
    resume = ts.resume_prompt if a["resume_pending"] else None
    # A provider with max_working admits a resume prompt like any other (api.py `admit`): it
    # goes to pending_prompts after the assign instead of riding inside it.
    r = ts.resolved_model
    held, resume = (resume, None) if resume and r and r.provider.max_working else (None, resume)
    cmd = AssignCommand(
        agent=a["name"], type=type_name, spec=ts, restore=True, resume_prompt=resume,
        parent=a["parent"], agent_token=spawning.agent_token(store, a["name"], a["swarm"], time.time()),
    )
    store.enqueue_command(worker_id, cmd.model_dump_json())
    if a["resume_pending"]:
        store.update_agent(a["name"], resume_pending=0)
    if held:
        prompt = PromptCommand(agent=a["name"], prompt_id=uuid.uuid4().hex, text=held)
        store.add_prompt(a["name"], prompt.model_dump_json(), time.time())


def _refresh_worker(p: _Pass, spec: SwarmSpec, w: Row, ts: TypeSpec | None) -> None:
    if not _needs_check(w, p.now):
        if w["job_id"] is not None:
            p.store.update_worker(w["id"], state=worker_state(w, p.now).value)
        return
    p.store.update_worker(w["id"], last_inspect=p.now)
    st = p.job_state(w["job_id"], spec.namespace)
    if st is None:
        return
    if st.terminal:
        _job_ended(p, spec, w, ts, st)
        return
    hb = w["last_heartbeat"]
    silent = hb is not None and p.now - hb > _DEAD
    never = hb is None and st.stage != "SCHEDULING" and p.now - w["launched_at"] > LAUNCH_TIMEOUT
    if silent or never:
        p.cancel(w["job_id"], spec.namespace)
        p.act("cancel", w["id"], w["job_id"])
        _job_ended(p, spec, w, ts, JobState("CANCELED", "unresponsive"))
        return
    p.store.update_worker(w["id"], state=worker_state(w, p.now).value)


def _job_ended(p: _Pass, spec: SwarmSpec, w: Row, ts: TypeSpec | None, st: JobState) -> None:
    store, wid = p.store, w["id"]
    store.clear_commands(wid)
    if w["retire_at"] is not None:
        store.delete_worker(wid)
        p.act("remove_worker", wid, st.stage)
        return
    drain = parse_duration(ts.drain_window).total_seconds() if ts else 0.0
    near_deadline = w["deadline"] is not None and p.now >= w["deadline"] - drain
    planned = (st.stage == "ERROR" and st.message == "Job timeout") or (
        st.stage == "COMPLETED" and (bool(w["draining"]) or near_deadline)
    )
    common = {
        "job_id": None,
        "incarnation": None,
        "state": WorkerState.dead.value,
        "last_heartbeat": None,
        "draining": 0,
    }
    if planned:
        store.update_worker(wid, **common, consecutive_crashes=0, next_launch_at=p.now)
        p.act("planned_restart", wid, f"{st.stage}: {st.message}")
        return
    streak = w["consecutive_crashes"] + 1
    backoff = min(BACKOFF_BASE * 2 ** (streak - 1), BACKOFF_CAP)
    store.update_worker(
        wid,
        **common,
        last_error=_crash_reason(p, spec, w, st),
        crashes=w["crashes"] + 1,
        consecutive_crashes=streak,
        next_launch_at=p.now + backoff,
    )
    p.act("crash", wid, f"{st.stage}: {st.message}; backoff {backoff:.0f}s")
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
    if streak >= CRASHLOOP_THRESHOLD:
        store.set_type_health(w["swarm"], w["type"], TypeHealth.crashlooping.value)
        p.act("crashloop", f"{w['swarm']}/{w['type']}")


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
    # this one replaces must not take the new job's commands (e2e run 1, see the API).
    incarnation = uuid.uuid4().hex
    env = {
        **ts.env,
        "FLOCK_URL": s.public_url,
        "WORKER_ID": wid,
        "WORKER_INCARNATION": incarnation,
        "SWARM": spec.swarm,
        "TYPE": type_name,
        "DEADLINE": _iso(deadline),
        "DRAIN_WINDOW": ts.drain_window,  # known before the first assign arrives
        "BUCKET": spec.bucket,
    }
    secrets = {"FLOCK_TOKEN": s.token}
    names = [*ts.secrets, *([ts.msgboard_invite] if ts.msgboard_invite else [])]
    if ts.resolved_model is not None:  # the provider's key, by name, for the worker to inject
        names.append(ts.resolved_model.provider.api_key_env)
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
        job_id = p.jobs.launch(wid, ts, env, secrets, labels, timeout, namespace=spec.namespace)
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
        backoff = min(BACKOFF_BASE * 2 ** (failures - 1), LAUNCH_BACKOFF_CAP)
        store.set_launch_error(spec.swarm, type_name, msg, failures)
        store.update_worker(wid, next_launch_at=p.now + backoff)
        p.act("launch_failed", wid, f"{msg}; backoff {backoff:.0f}s")
        return
    store.set_launch_error(spec.swarm, type_name, None, 0)
    store.clear_commands(wid)
    store.update_worker(
        wid,
        job_id=job_id,
        incarnation=incarnation,
        launched_at=p.now,
        deadline=deadline,
        last_heartbeat=None,
        last_inspect=None,
        draining=0,
        state=WorkerState.launching.value,
        next_launch_at=None,
    )
    for a in store.list_agents(worker=wid):
        _send_assign(store, wid, a, type_name, ts)
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
        "deadline": hb.deadline.timestamp(),
        "draining": int(hb.draining),
        "state": (WorkerState.draining if hb.draining else WorkerState.running).value,
    }
    if w["consecutive_crashes"] and t - (w["launched_at"] or t) >= STABLE_AFTER:
        fields["consecutive_crashes"] = 0
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
                "harness_estimate_usd": ah.harness_estimate_usd,
                "turns": ah.turns,
                "last_sync": ah.last_sync.timestamp() if ah.last_sync else None,
                "msgboard_principal": ah.msgboard_principal,
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
    for a in missing:
        if a["name"] not in pending:
            log.warning("re-sending assign for %s: missing from %s heartbeat", a["name"], w["id"])
            _send_assign(store, w["id"], a, w["type"], ts)


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
