from __future__ import annotations

import math
from datetime import datetime, timedelta

from pydantic import TypeAdapter

from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import (
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
    DrainCommand,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerState,
    parse_duration,
)

from .conftest import TOKEN, make_spec

CMD = TypeAdapter(Command)
W0, W1 = "demo-researcher-w0000", "demo-researcher-w0001"


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
    assert worker_state(w, clock().timestamp()) == WorkerState.launching
    assert agent_state(store.get_agent("researcher-0000"), w, clock().timestamp()) == "unknown"
    clock.advance(20)
    heartbeat(
        store, clock, W0, [("researcher-0000", "working", "t-1"), ("researcher-0001", "idle", None)]
    )
    now = clock().timestamp()
    w = store.get_worker(W0)
    assert worker_state(w, now) == WorkerState.running
    assert agent_state(store.get_agent("researcher-0000"), w, now) == AgentState.working
    assert agent_state(store.get_agent("researcher-0001"), w, now) == AgentState.idle
    assert store.get_agent("researcher-0000")["turn_id"] == "t-1"


def test_unknown_and_dead_by_age(store, jobs, clock, tick):
    store.put_swarm(make_spec(), 0)
    tick()
    heartbeat(store, clock, W0, [("researcher-0000", "idle", None)])
    a = store.get_agent("researcher-0000")
    t = clock().timestamp()
    w = store.get_worker(W0)
    assert agent_state(a, w, t + 60) == AgentState.idle
    assert agent_state(a, w, t + 91) == AgentState.unknown
    assert worker_state(w, t + 91) == WorkerState.unknown
    assert agent_state(a, w, t + 181) == AgentState.dead
    assert worker_state(w, t + 181) == WorkerState.dead


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
    assert agent_state(store.get_agent("researcher-0000"), w, clock().timestamp()) == "dead"
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
