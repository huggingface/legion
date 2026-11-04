from __future__ import annotations

import math
from datetime import datetime, timedelta

from pydantic import TypeAdapter

from flock import status as fs
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import (
    LAUNCH_TIMEOUT,
    NUDGE_EVERY,
    NUDGE_MAX,
    agent_state,
    rebuild_from_jobs,
    reconcile,
    record_heartbeat,
    worker_state,
)
from flock.controller.store import Store
from flock.models import (
    CRASHLOOP_THRESHOLD,
    AgentHeartbeat,
    AgentState,
    AssignCommand,
    Command,
    ComputeSpec,
    DrainCommand,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerState,
    parse_duration,
)

from .conftest import TOKEN, make_spec

CMD = TypeAdapter(Command)
W0, W1 = "demo-researcher-w0000", "demo-researcher-w0001"
LV = ComputeSpec().liveness()  # make_spec's hf-jobs defaults: unknown 90 s, dead 180 s


def queued(store: Store, wid: str) -> list:
    return [CMD.validate_json(c["body"]) for c in store.commands_for(wid)]


def heartbeat(store, clock, wid, agents, draining=False):
    hb = WorkerHeartbeat(
        worker=wid,
        deadline=clock() + timedelta(hours=24),
        draining=draining,
        agents=[AgentHeartbeat(name=n, state=s, turn_id=tid) for n, s, tid in agents],
    )
    assert record_heartbeat(store, wid, hb, clock())


def kinds(actions) -> list[str]:
    return [a.kind for a in actions]


def test_apply_creates_agents_and_launches_workers(store, jobs, settings, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 5, "agents_per_worker": 2}), 0)
    tick()
    agents = store.list_agents("demo")
    assert [a["name"] for a in agents] == [f"researcher-000{i}" for i in range(5)]
    assert len(jobs.launches) == math.ceil(5 / 2)
    assert [a["worker"] for a in agents] == [W0, W0, W1, W1, "demo-researcher-w0002"]

    launch = jobs.launches[0]
    assert launch["labels"] == {"swarm": "demo", "type": "researcher", "worker": W0}
    assert launch["namespace"] == "ns"
    assert launch["secrets"] == {"FLOCK_TOKEN": TOKEN, "ANTHROPIC_API_KEY": "sk-test"}
    env = launch["env"]
    assert env["WORKER_ID"] == W0 and env["SWARM"] == "demo" and env["TYPE"] == "researcher"
    assert env["FLOCK_URL"] == "https://controller.invalid"
    assert env["BUCKET"] == "hf://buckets/ns/sh" and env["MSGBOARD_URL"]
    assert env["DRAIN_WINDOW"] == "10m" and len(env["WORKER_INCARNATION"]) == 32
    assert "FLOCK_TOKEN" not in env
    lifetime = parse_duration("24h").total_seconds()
    assert lifetime <= launch["timeout"] <= lifetime + parse_duration("4h").total_seconds()
    deadline = datetime.fromisoformat(env["DEADLINE"])
    assert deadline == clock() + timedelta(seconds=launch["timeout"])

    cmds = queued(store, W0)
    assert all(isinstance(c, AssignCommand) and c.restore for c in cmds)
    assert [c.agent for c in cmds] == ["researcher-0000", "researcher-0001"]

    # Idempotent: a second pass launches nothing new.
    assert "launch" not in kinds(tick())


def test_launch_budget(store, jobs, settings, clock, tick):
    settings.launches_per_window = 2
    store.put_swarm(make_spec(researcher={"replicas": 5, "agents_per_worker": 1}), 0)
    actions = tick()
    assert len(jobs.launches) == 2 and "budget_exhausted" in kinds(actions)
    clock.advance(60)
    tick()
    assert len(jobs.launches) == 2
    clock.advance(settings.rate_window.total_seconds())
    tick()
    assert len(jobs.launches) == 4
    clock.advance(settings.rate_window.total_seconds() + 1)
    tick()
    assert len(jobs.launches) == 5


def test_placement_fills_in_order_and_never_moves(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 3, "agents_per_worker": 2}), 0)
    tick()
    store.put_swarm(make_spec(researcher={"replicas": 4, "agents_per_worker": 2}), 0)
    actions = tick()
    assert store.get_agent("researcher-0003")["worker"] == W1  # filled W1's free slot
    assert len(jobs.launches) == 2 and "launch" not in kinds(actions)
    assert queued(store, W1)[-1].agent == "researcher-0003"
    # Raising capacity later does not move anyone.
    store.put_swarm(make_spec(researcher={"replicas": 4, "agents_per_worker": 10}), 0)
    tick()
    assert store.get_agent("researcher-0002")["worker"] == W1


def test_heartbeat_updates_states(store, jobs, clock, tick):
    store.put_swarm(make_spec(), 0)
    tick()
    w = store.get_worker(W0)
    assert worker_state(w, clock().timestamp(), LV) == WorkerState.launching
    assert agent_state(store.get_agent("researcher-0000"), w, clock().timestamp(), LV) == "unknown"
    clock.advance(20)
    heartbeat(
        store, clock, W0, [("researcher-0000", "working", "t-1"), ("researcher-0001", "idle", None)]
    )
    now = clock().timestamp()
    w = store.get_worker(W0)
    assert worker_state(w, now, LV) == WorkerState.running
    assert agent_state(store.get_agent("researcher-0000"), w, now, LV) == AgentState.working
    assert agent_state(store.get_agent("researcher-0001"), w, now, LV) == AgentState.idle
    assert store.get_agent("researcher-0000")["turn_id"] == "t-1"


def test_unknown_and_dead_by_age(store, jobs, clock, tick):
    store.put_swarm(make_spec(), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    a = store.get_agent("researcher-0000")
    t = clock().timestamp()
    w = store.get_worker(W0)
    assert agent_state(a, w, t + 60, LV) == AgentState.idle
    assert agent_state(a, w, t + 91, LV) == AgentState.unknown
    assert worker_state(w, t + 91, LV) == WorkerState.unknown
    assert agent_state(a, w, t + 181, LV) == AgentState.dead
    assert worker_state(w, t + 181, LV) == WorkerState.dead


def test_unresponsive_worker_is_cancelled_and_relaunched(store, jobs, clock, tick):
    store.put_swarm(make_spec(), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    job = jobs.job_for(W0)
    clock.advance(200)
    actions = tick()
    assert job in jobs.cancels and "crash" in kinds(actions)
    clock.advance(31)
    tick()
    assert jobs.job_for(W0) != job


def test_crash_relaunch_restores_and_resumes_interrupted(store, jobs, clock, tick):
    spec = make_spec(researcher={"replicas": 2, "agents_per_worker": 2, "resume_prompt": "go on"})
    store.put_swarm(spec, 0)
    tick()
    heartbeat(
        store, clock, W0, [("researcher-0000", "working", "t-9"), ("researcher-0001", "idle", None)]
    )
    old = jobs.job_for(W0)
    jobs.set_stage(old, "ERROR", "OOM killed")
    clock.advance(100)  # heartbeat now stale -> inspected
    actions = tick()
    assert "crash" in kinds(actions) and "launch" not in kinds(actions)
    w = store.get_worker(W0)
    assert w["crashes"] == 1 and w["consecutive_crashes"] == 1 and w["job_id"] is None
    assert agent_state(store.get_agent("researcher-0000"), w, clock().timestamp(), LV) == "dead"
    evs = store.read_events("researcher-0000", 0)
    assert [e["type"] for e in evs] == ["turn_interrupted"] and evs[0]["turn_id"] == "t-9"
    assert store.read_events("researcher-0001", 0) == []

    clock.advance(29)
    assert "launch" not in kinds(tick())  # backoff 30 s
    clock.advance(2)
    assert "launch" in kinds(tick())
    new = jobs.job_for(W0)
    assert new != old
    cmds = {c.agent: c for c in queued(store, W0)}
    assert cmds["researcher-0000"].restore and cmds["researcher-0000"].resume_prompt == "go on"
    assert cmds["researcher-0001"].restore and cmds["researcher-0001"].resume_prompt is None
    assert store.get_agent("researcher-0000")["resume_pending"] == 0


def test_no_resume_prompt_when_type_has_none(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "working", "t-1")])
    jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
    clock.advance(100)
    tick()
    clock.advance(31)
    tick()
    (cmd,) = queued(store, W0)
    assert cmd.resume_prompt is None
    assert store.read_events("researcher-0000", 0)[0]["type"] == "turn_interrupted"


def test_job_timeout_is_planned_restart(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1, "resume_prompt": "go on"}), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "working", "t-1")])
    store.update_worker(W0, consecutive_crashes=2)
    jobs.set_stage(jobs.job_for(W0), "ERROR", "Job timeout")
    clock.advance(100)
    actions = tick()
    assert kinds(actions)[:2] == ["planned_restart", "launch"]
    assert "planned_restart" in [e["type"] for e in store.read_events("_swarm.demo", 0)]
    w = store.get_worker(W0)
    assert w["consecutive_crashes"] == 0 and w["crashes"] == 0
    (cmd,) = queued(store, W0)
    assert cmd.restore and cmd.resume_prompt is None
    assert store.read_events("researcher-0000", 0) == []


def test_drained_completed_is_planned_restart(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)], draining=True)
    jobs.set_stage(jobs.job_for(W0), "COMPLETED")
    actions = tick()  # draining workers are inspected even with a fresh heartbeat
    assert kinds(actions)[:2] == ["planned_restart", "launch"]


def test_drained_completed_in_a_released_swarm_ends(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    store.put_swarm_flags("demo", {"released": {"since": 0, "reason": "", "by": "op"}})
    assert "release" in kinds(tick())  # the drain
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)], draining=True)
    jobs.set_stage(jobs.job_for(W0), "COMPLETED")
    assert kinds(tick()) == ["end"]  # no restart, no launch
    (ev,) = [e for e in store.read_events("_swarm.demo", 0) if e["type"] == "end"]
    assert fs.normalise(ev, "demo")["name"] == "worker.ended" and ev["data"]["detail"] == "COMPLETED"


def test_completed_without_drain_is_a_crash(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    jobs.set_stage(jobs.job_for(W0), "COMPLETED")
    clock.advance(100)
    assert "crash" in kinds(tick())


def test_repeated_crashes_back_off_then_crashloop(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    backoffs = []
    for _ in range(CRASHLOOP_THRESHOLD):
        jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
        clock.advance(100)  # never heartbeated -> inspected every pass
        tick()
        w = store.get_worker(W0)
        backoffs.append(w["next_launch_at"] - clock().timestamp())
        clock.advance(w["next_launch_at"] - clock().timestamp())
        tick()
    assert backoffs == [30, 60, 120, 240, 480]
    assert store.type_health("demo", "researcher") == "crashlooping"
    evs = store.read_events("_swarm.demo", 0)
    types = [e["type"] for e in evs]
    assert types.count("crash") == CRASHLOOP_THRESHOLD and types[-1] == "crashloop"
    assert "launch" in types and "assign" not in types
    crash = next(e for e in evs if e["type"] == "crash")
    assert crash["data"] == {"target": W0, "detail": "ERROR: boom; backoff 30s"}
    n = len(jobs.launches)
    assert n == CRASHLOOP_THRESHOLD  # first launch + 4 relaunches; the 5th crash stops it
    clock.advance(3600)
    tick()
    assert len(jobs.launches) == n
    # Re-applying the spec clears it.
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    assert store.type_health("demo", "researcher") == "ok" and len(jobs.launches) == n + 1


def test_backoff_caps_at_15_minutes(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
    store.update_worker(W0, consecutive_crashes=9)
    clock.advance(100)
    tick()
    assert store.get_worker(W0)["next_launch_at"] - clock().timestamp() == 900


def test_scale_down_unassigns_highest_and_cancels_empty_workers(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 4, "agents_per_worker": 2}), 0)
    tick()
    store.clear_commands(W0)
    store.clear_commands(W1)
    store.put_swarm(make_spec(researcher={"replicas": 1, "agents_per_worker": 2}), 0)
    actions = tick()
    unassigned = [a.target for a in actions if a.kind == "unassign"]
    assert unassigned == ["researcher-0003", "researcher-0002", "researcher-0001"]
    assert [a["name"] for a in store.list_agents("demo")] == ["researcher-0000"]
    assert queued(store, W0) == [UnassignCommand(agent="researcher-0001")]
    w1_cmds = queued(store, W1)
    assert w1_cmds[:2] == [
        UnassignCommand(agent="researcher-0003"),
        UnassignCommand(agent="researcher-0002"),
    ]
    assert w1_cmds[-1] == DrainCommand()
    job1 = jobs.job_for(W1)
    clock.advance(121)
    tick()
    assert job1 in jobs.cancels and store.get_worker(W1) is None
    assert store.get_worker(W0) is not None and jobs.job_for(W0) not in jobs.cancels


def test_remove_swarm(store, jobs, clock, tick):
    store.put_swarm(make_spec(), 0)
    tick()
    store.set_removing("demo")
    tick()
    assert store.list_agents("demo") == []
    clock.advance(121)
    actions = tick()
    assert "remove_swarm" in kinds(actions)
    assert store.get_swarm("demo") is None and len(jobs.cancels) == 2


def test_heartbeat_for_unknown_agent_unassigns_it(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    store.clear_commands(W0)
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None), ("ghost-0001", "idle", None)])
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None), ("ghost-0001", "idle", None)])
    assert queued(store, W0) == [UnassignCommand(agent="ghost-0001")]


def test_rebuild_from_jobs_adopts_live_workers(tmp_path, jobs, settings, clock):
    first = Store(tmp_path / "a")
    first.put_swarm(make_spec(), 0)
    reconcile(first, jobs, clock(), settings)
    # Controller lost its database; it still has the spec (re-applied) and the jobs.
    fresh = Store(tmp_path / "b")
    fresh.put_swarm(make_spec(), 0)
    actions = rebuild_from_jobs(fresh, jobs, clock())
    assert sorted(a.target for a in actions if a.kind == "adopt") == [W0, W1]
    assert fresh.get_worker(W0)["job_id"] == jobs.job_for(W0)
    assert fresh.get_worker(W0)["incarnation"] == jobs.launches[0]["env"]["WORKER_INCARNATION"]
    hb = WorkerHeartbeat(
        worker=W0,
        deadline=clock() + timedelta(hours=1),
        agents=[AgentHeartbeat(name="researcher-0000", state="idle")],
    )
    # Agents are created by the next pass and adopted from heartbeats, without new launches.
    reconcile_actions = reconcile(fresh, jobs, clock(), settings)
    assert "launch" not in [a.kind for a in reconcile_actions]
    assert record_heartbeat(fresh, W0, hb, clock())
    assert fresh.get_agent("researcher-0000")["worker"] == W0
    assert len(jobs.launches) == 2
    first.close()
    fresh.close()


def test_fake_jobs_list_filters_labels():
    jobs = FakeJobsClient()
    spec = make_spec().types["researcher"]
    jobs.launch("w", spec, {}, {}, {"swarm": "a", "type": "t", "worker": "w"}, 10, namespace="ns")
    assert len(jobs.list({"swarm": "a"}, namespace="ns")) == 1
    assert jobs.list({"swarm": "b"}, namespace="ns") == []


def test_job_checks_are_bounded(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 30, "agents_per_worker": 1}), 0)
    tick()
    assert len(jobs.launches) == 30
    clock.advance(30)
    tick()
    assert jobs.inspects == 0 and jobs.lists == 0  # launch grace: nothing is checked yet
    clock.advance(40)
    tick()
    assert jobs.inspects == 0 and jobs.lists == 1  # 30 unheard-from workers: one list call
    tick()
    assert jobs.lists == 1  # and not again within INSPECT_EVERY
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    for i in range(1, 25):
        wid = f"demo-researcher-w{i:04d}"
        heartbeat(store, clock, wid, [(f"researcher-{i:04d}", "idle", None)])
    clock.advance(16)
    tick()
    assert jobs.inspects == 5 and jobs.lists == 1  # few left: inspect them one by one


def test_hub_errors_do_not_break_the_pass(store, jobs, clock, tick, monkeypatch):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()

    def boom(*a, **k):
        raise RuntimeError("429")

    monkeypatch.setattr(jobs, "inspect", boom)
    clock.advance(100)
    assert "crash" not in kinds(tick())
    jobs.fail_launches = True
    store.put_swarm(make_spec(researcher={"replicas": 2, "agents_per_worker": 1}), 0)
    assert "launch_failed" in kinds(tick())


def test_job_labels_from_settings_are_added_but_never_override(store, jobs, settings, tick):
    settings.job_labels = {"experiment": "flock-m1", "swarm": "spoof"}
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    assert jobs.launches[0]["labels"] == {
        "experiment": "flock-m1",
        "swarm": "demo",
        "type": "researcher",
        "worker": W0,
    }


def test_store_migrates_workers_table_without_incarnation(tmp_path):
    import sqlite3

    (tmp_path / "old").mkdir()
    db = sqlite3.connect(tmp_path / "old" / "controller.db")
    db.execute("CREATE TABLE workers (id TEXT PRIMARY KEY, swarm TEXT NOT NULL, type TEXT NOT NULL,"
               " idx INTEGER NOT NULL, job_id TEXT, created_at REAL NOT NULL)")
    db.commit()
    db.close()
    s = Store(tmp_path / "old")
    s.insert_worker("w", "sw", "t", 0, 0)
    s.update_worker("w", incarnation="abc")
    assert s.get_worker("w")["incarnation"] == "abc"


def test_store_migrates_swarms_to_generation_1(tmp_path):
    import sqlite3

    (tmp_path / "old").mkdir()
    db = sqlite3.connect(tmp_path / "old" / "controller.db")
    db.execute("CREATE TABLE swarms (name TEXT PRIMARY KEY, spec TEXT NOT NULL, "
               "applied_at REAL NOT NULL, removing INTEGER NOT NULL DEFAULT 0)")
    db.execute("INSERT INTO swarms VALUES ('demo', ?, 0, 0)", [make_spec().model_dump_json()])
    db.commit()
    db.close()
    s = Store(tmp_path / "old")
    assert s.generation("demo") == 1 and s.spec_history("demo") == []
    assert s.record_spec(make_spec(), 5, {}, ["x: 1 -> 2"], new=False) == 2


class _HubError(Exception):
    """Shaped like huggingface_hub's HfHubHTTPError: carries the response."""

    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.response = type("R", (), {"status_code": status})()


def test_permanent_launch_error_stops_retrying_until_reapply(store, jobs, clock, tick):
    from flock.controller.api import swarm_status

    jobs.launch_error = _HubError(400, "Flavor 'cpu-basic' for arch 'amd64' is not available")
    store.put_swarm(make_spec(researcher={"replicas": 2, "agents_per_worker": 1}), 0)
    acts = tick()
    assert kinds(acts).count("launch_refused") == 1 and "launch" not in kinds(acts)
    assert store.type_health("demo", "researcher") == "launch_refused"
    for _ in range(5):  # no more attempts, however long we wait
        clock.advance(600)
        assert not {"launch", "launch_failed", "launch_refused"} & set(kinds(tick()))
    spec = store.get_swarm("demo")["spec"]
    t = swarm_status(store, spec, clock().timestamp()).types["researcher"]
    assert t.health == "launch_refused" and "not available" in t.launch_error
    jobs.launch_error = None  # a new apply resets it
    store.put_swarm(make_spec(researcher={"replicas": 2, "agents_per_worker": 1}), 0)
    assert kinds(tick()).count("launch") == 2
    assert store.type_health("demo", "researcher") == "ok"
    assert store.launch_error("demo", "researcher") == (None, 0)


def test_transient_launch_errors_back_off_with_a_cap(store, jobs, clock, tick):
    jobs.launch_error = _HubError(503, "Service Unavailable")
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    delays = []
    for _ in range(8):
        acts = [a for a in tick() if a.kind == "launch_failed"]
        assert len(acts) == 1
        delay = store.get_worker(W0)["next_launch_at"] - clock().timestamp()
        delays.append(delay)
        clock.advance(delay - 1)  # just before the backoff ends: no attempt
        assert "launch_failed" not in kinds(tick())
        clock.advance(1)
    assert delays == [30, 60, 120, 240, 300, 300, 300, 300]
    assert store.type_health("demo", "researcher") == "ok"
    err, failures = store.launch_error("demo", "researcher")
    assert err == "_HubError: Service Unavailable" and failures == 8
    jobs.launch_error = ConnectionError("reset")  # no status: transient too
    assert "launch_failed" in kinds(tick())
    jobs.launch_error = None
    clock.advance(300)
    assert "launch" in kinds(tick())
    assert store.launch_error("demo", "researcher") == (None, 0)  # cleared by a launch


def test_launch_refused_classification():
    from flock.controller.reconcile import launch_refused

    assert launch_refused(_HubError(400, "x")) and launch_refused(_HubError(422, "x"))
    BadRequestError = type("BadRequestError", (Exception,), {})
    assert launch_refused(BadRequestError("x"))
    assert not launch_refused(_HubError(429, "x")) and not launch_refused(_HubError(503, "x"))
    assert not launch_refused(RuntimeError("x")) and not launch_refused(ConnectionError("x"))


def test_resume_prompt_goes_through_max_working_admission(store, jobs, clock, tick):
    """A capped provider's resume prompt is queued after the assign (pending_prompts), so the
    controller's max_working admission sees it like any API prompt."""
    from flock.models import SwarmSpec

    spec = SwarmSpec.model_validate({
        "swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": {"p": {"type": "openai_compat", "base_url": "http://fake/v1",
                            "api_key_env": "K", "self_hosted": True, "max_working": 1}},
        "types": {"researcher": {"harness": "codex", "model": "p:m", "replicas": 1,
                                 "resume_prompt": "go on"}}})
    store.put_swarm(spec, 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "working", "t-9")])
    jobs.set_stage(jobs.job_for(W0), "ERROR", "OOM killed")
    clock.advance(100)
    assert "crash" in kinds(tick())
    clock.advance(31)
    assert "launch" in kinds(tick())
    (assign,) = queued(store, W0)
    assert assign.agent == "researcher-0000" and assign.resume_prompt is None
    (held,) = store.prompts_for_worker(W0)
    assert held["agent"] == "researcher-0000" and '"text":"go on"' in held["body"]


def test_memory_rotation_drains_smallest_eta_within_budget(store, jobs, settings, clock, tick):
    from flock.controller.reconcile import swarm_stream
    from flock.models import DrainCommand, WorkerMem

    store.put_swarm(make_spec(researcher={"replicas": 6, "agents_per_worker": 2}), 0)
    tick()
    w2 = "demo-researcher-w0002"

    def hb(wid, i, eta=None, **kw):
        h = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=24),
                            agents=[AgentHeartbeat(name=f"researcher-000{j}", state="idle")
                                    for j in (2 * i, 2 * i + 1)],
                            wants_rotation=eta is not None,
                            mem=WorkerMem(limit_mb=10000, nonreclaim_mb=8000, eta_limit_s=eta), **kw)
        assert record_heartbeat(store, wid, h, clock())

    def drains(wid):
        return [c for c in queued(store, wid) if isinstance(c, DrainCommand)]

    hb(W0, 0)
    hb(W1, 1, eta=7200)
    hb(w2, 2, eta=3600)
    tick()  # budget max(1, 10% of 3 slots) = 1: only the nearest to its limit
    assert drains(w2) == [DrainCommand()] and drains(W1) == []
    events = [e for e in store.read_events(swarm_stream("demo"), 0) if e["type"] == "memory_rotation"]
    assert [e["data"] for e in events] == [{"worker": w2, "nonreclaim_mb": 8000.0,
                                            "eta_limit_s": 3600.0}]
    tick()  # w2 is draining: the budget is used up
    assert drains(W1) == []

    hb(w2, 2, draining=True)  # its drain ends in a planned restart
    jobs.set_stage(jobs.job_for(w2), "COMPLETED")
    clock.advance(20)
    for wid, i in ((W0, 0), (W1, 1)):
        hb(wid, i, eta=7200 if wid == W1 else None)
    assert "planned_restart" in kinds(tick())
    tick()
    assert drains(W1) == []  # w2's relaunch (restoring) still uses the budget
    settings.max_rotating = 2
    tick()
    assert drains(W1) == [DrainCommand()]


def test_a_stale_heartbeat_does_not_undo_a_rotation_drain(store, jobs, settings, clock, tick):
    from flock.controller.reconcile import swarm_stream
    from flock.models import DrainCommand, WorkerMem

    store.put_swarm(make_spec(researcher={"replicas": 2, "agents_per_worker": 2}), 0)
    tick()
    hb = WorkerHeartbeat(worker=W0, deadline=clock() + timedelta(hours=24), wants_rotation=True,
                         agents=[AgentHeartbeat(name=f"researcher-000{j}", state="idle") for j in (0, 1)],
                         mem=WorkerMem(limit_mb=10000, nonreclaim_mb=8000, eta_limit_s=600))
    assert record_heartbeat(store, W0, hb, clock())
    tick()
    assert record_heartbeat(store, W0, hb, clock())  # sent before the worker polled the drain
    tick()
    drains = [c for c in queued(store, W0) if isinstance(c, DrainCommand)]
    rotations = [e for e in store.read_events(swarm_stream("demo"), 0) if e["type"] == "memory_rotation"]
    assert len(drains) == 1 and len(rotations) == 1


def test_launch_timeout_counts_from_job_start_not_submit(store, jobs, clock, tick):
    """A job queued longer than LAUNCH_TIMEOUT is not cancelled the moment it starts."""
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    job = jobs.job_for(W0)
    jobs.set_stage(job, "SCHEDULING", "PENDING")
    for _ in range(70):  # 20 min in the queue
        clock.advance(20)
        tick()
    jobs.set_stage(job, "RUNNING")
    clock.advance(20)
    tick()
    clock.advance(LAUNCH_TIMEOUT - 30)
    tick()
    assert job not in jobs.cancels
    clock.advance(30)  # LAUNCH_TIMEOUT since the last queued sighting, no heartbeat: a crash
    assert "cancel" in kinds(tick()) and job in jobs.cancels


def test_launch_timeout_without_queue_state_counts_from_launch(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    job = jobs.job_for(W0)  # the fake job is RUNNING from launch, like a local process
    clock.advance(LAUNCH_TIMEOUT - 1)
    tick()
    assert job not in jobs.cancels
    clock.advance(16)  # next job check (INSPECT_EVERY)
    assert "cancel" in kinds(tick()) and job in jobs.cancels


def _stuck_tick(store, clock, tick, dt, state="stuck"):
    clock.advance(dt)
    heartbeat(store, clock, W0, [("researcher-0000", state, None)])
    return kinds(tick())


def _nudge_spec():
    return make_spec(researcher={"replicas": 1, "resume_prompt": "go on"})


def test_stuck_agent_gets_resume_prompt_after_60s_once_per_30_min(store, jobs, clock, tick):
    store.put_swarm(_nudge_spec(), 0)
    tick()
    assert "nudge" not in _stuck_tick(store, clock, tick, 1)
    assert "nudge" not in _stuck_tick(store, clock, tick, 30)  # stuck 30 s
    assert "nudge" in _stuck_tick(store, clock, tick, 30)  # stuck 60 s
    (p,) = store.prompts_for_worker(W0)
    assert '"text":"go on"' in p["body"]
    store.delete_prompts([p["id"]])
    for _ in range(3):  # still stuck: not again within 30 min
        assert "nudge" not in _stuck_tick(store, clock, tick, 60)
    assert "nudge" in _stuck_tick(store, clock, tick, NUDGE_EVERY)


def test_stuck_nudges_cap_until_a_turn_completes(store, jobs, clock, tick):
    store.put_swarm(_nudge_spec(), 0)
    tick()
    _stuck_tick(store, clock, tick, 1)
    for _ in range(NUDGE_MAX):  # each nudge starts a turn that times out again
        assert "nudge" in _stuck_tick(store, clock, tick, NUDGE_EVERY)
        store.delete_prompts([p["id"] for p in store.prompts_for_worker(W0)])
        _stuck_tick(store, clock, tick, 1, "working")
        _stuck_tick(store, clock, tick, 1)
    assert "nudge" not in _stuck_tick(store, clock, tick, NUDGE_EVERY)  # 4th episode
    _stuck_tick(store, clock, tick, 1, "idle")  # a completed turn resets the count
    _stuck_tick(store, clock, tick, 1)
    assert "nudge" in _stuck_tick(store, clock, tick, 60)


def test_no_nudge_under_budget_stop_hold_or_done_loop(store, jobs, clock, tick):
    store.put_swarm(_nudge_spec(), 0)
    tick()
    _stuck_tick(store, clock, tick, 1)
    store.put_swarm_flags("demo", {"budget_stop": "final_reserve"})
    assert "nudge" not in _stuck_tick(store, clock, tick, 60)
    store.put_swarm_flags("demo", {"holds": {"agents": {"researcher-0000": {}}}})
    assert "nudge" not in _stuck_tick(store, clock, tick, 60)
    store.put_swarm_flags("demo", {})
    store.put_loop_agent("researcher-0000", "demo", "researcher", {"phase": "done"})
    assert "nudge" not in _stuck_tick(store, clock, tick, 60)
    store.put_loop_agent("researcher-0000", "demo", "researcher", {"phase": "looping"})
    assert "nudge" in _stuck_tick(store, clock, tick, 60)


def test_no_nudge_when_type_has_no_resume_prompt(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)
    tick()
    for _ in range(3):
        assert "nudge" not in _stuck_tick(store, clock, tick, 60)
    assert store.prompts_for_worker(W0) == []
