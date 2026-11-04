"""`flock launch` / `flock stop` (docs/live.md "Lifecycle"): release with verified final syncs,
the fresh-run guard, resume ignoring replicas, and
`update --replicas spec` never lowering a ramp type."""

from __future__ import annotations

import asyncio
from collections import defaultdict

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli, ops
from flock.controller.api import create_app
from flock.controller.reconcile import reconcile, record_heartbeat
from flock.controller.state import StateSync
from flock.controller.store import Store
from flock.models import (
    AssignCommand,
    DrainCommand,
    PromptCommand,
    WorkerHeartbeat,
)

from .conftest import TOKEN, make_spec, wire, write_spec
from .test_worker_main import FakeController, make_worker
from .worker_fakes import MemoryStore, RecordingSync, ScriptedAdapter, spec, until

runner = CliRunner()


# ----------------------------------------------------------------- (a) stop --release


class FlakySync(RecordingSync):
    """Pushes of `bad` always fail; every push is logged with whether the worker had exited."""

    def __init__(self, store, bad: str, exited) -> None:
        super().__init__(store)
        self.bad, self.exited, self.at_exit = bad, exited, []

    async def push(self, rt, **kw):
        self.at_exit.append((rt.name, self.exited()))
        if rt.name == self.bad:
            self.pushed.append(rt.name)
            raise OSError("bucket 503")
        return await super().push(rt, **kw)


async def test_release_drain_syncs_every_agent_before_exit_and_reports_a_failed_one(tmp_path,
                                                                                    caplog):
    ctrl = FakeController()
    bucket = MemoryStore()
    done: list[bool] = []
    sync = FlakySync(bucket, "researcher-0003", lambda: bool(done))
    s = spec()
    ctrl.batches = [[AssignCommand(agent=f"researcher-000{i}", type="researcher", spec=s,
                                   restore=False) for i in (1, 2, 3)],
                    [PromptCommand(agent="researcher-0001", prompt_id="p1", text="long turn")]]
    adapters: dict = {}
    worker, _ = make_worker(tmp_path, ctrl, sync=sync, adapters=adapters)
    worker.cfg.sync_interval = 3600  # only the drain's final push
    worker.adapter_factory = lambda name, sp: adapters.setdefault(
        name, ScriptedAdapter(tmp_path / "home" / name, turn_time=0.5))
    run = asyncio.create_task(worker.run())
    await until(lambda: any(e.type == "turn_started" for e in ctrl.events), timeout=5)
    ctrl.batches.append([DrainCommand()])  # what reconcile sends a released swarm's workers
    assert await asyncio.wait_for(run, 5) == 0
    done.append(True)
    # the turn in flight finished, then every agent pushed before the worker exited
    assert any(e.type == "turn_ended" and e.agent == "researcher-0001" for e in ctrl.events)
    assert {n for n, exited in sync.at_exit if not exited} == {
        "researcher-0001", "researcher-0002", "researcher-0003"}
    for n in ("researcher-0001", "researcher-0002"):
        assert bucket.list(f"sw/researcher/{n}/"), n
    assert "researcher-0003: drain sync failed: OSError: bucket 503 (attempt 2)" in caplog.text
    last = ctrl.heartbeats[-1]
    assert last.draining and list(last.unsynced) == ["researcher-0003"]
    assert "bucket 503" in last.unsynced["researcher-0003"]


def test_stop_release_drains_every_worker_waits_and_reports_unsynced(
        tmp_path, monkeypatch, app, store, jobs, settings, clock):
    wire(monkeypatch, app)
    tick = lambda: reconcile(store, jobs, clock(), settings)
    store.put_swarm(make_spec(), clock().timestamp())  # 3 agents, 2 per worker: 2 workers
    tick()
    assert len(jobs.launches) == 2
    workers = store.list_workers("demo")

    def workers_drain_and_exit(_s):  # one CLI poll: the workers do their part
        tick()  # the release: a drain command per worker, nothing new launches
        for w in workers:
            assert any(DrainCommand().model_dump_json() == c["body"]
                       for c in store.commands_for(w["id"]))
        hb = WorkerHeartbeat(worker=workers[0]["id"], deadline=clock(), draining=True, agents=[],
                             unsynced={"researcher-0001": "drain sync failed: OSError: 503"})
        record_heartbeat(store, workers[0]["id"], hb, clock())
        for w in store.list_workers("demo"):
            jobs.set_stage(w["job_id"], "COMPLETED")
        clock.advance(120)
        tick()

    monkeypatch.setattr("flock.ops._sleep", workers_drain_and_exit)
    r = runner.invoke(cli.app, ["stop", "demo", "--release", "--reason", "night"])
    assert r.exit_code == 1, r.output
    assert "NOT synced: researcher-0001: drain sync failed: OSError: 503" in r.output
    assert all(w["job_id"] is None for w in store.list_workers("demo"))
    st = runner.invoke(cli.app, ["status", "demo"]).output
    assert "RELEASED since" in st and "NOT synced: researcher-0001" in st, st
    assert any(e["type"] == "sync_failed" for e in store.read_events("_swarm.demo", 0))
    # released: rows kept, nothing relaunches until `flock launch`
    clock.advance(600)
    tick()
    assert len(jobs.launches) == 2 and len(store.list_agents("demo")) == 3
    r = runner.invoke(cli.app, ["launch", "demo"])
    assert r.exit_code == 0 and "held: nothing" in r.output, r.output
    tick()
    assert len(jobs.launches) == 4
    assert store.swarm_flags("demo").get("released") is None


def test_a_fresh_empty_controller_never_overwrites_the_checkpoint(tmp_path):
    bucket = MemoryStore()
    s = Store(tmp_path / "fresh")
    assert StateSync(bucket, tmp_path / "fresh").push(s, force=True, final=True) == 0
    assert bucket.files == {}
    s.close()


def test_the_guard_refuses_a_reused_bucket(
        tmp_path, monkeypatch, store, clock):
    buckets: dict[str, MemoryStore] = defaultdict(MemoryStore)
    buckets["hf://buckets/ns/sh"].files["demo/researcher/researcher-0000/meta.json"] = b"{}"
    app = create_app(store, TOKEN, clock=clock, buckets=buckets.__getitem__)
    wire(monkeypatch, app)
    f = write_spec(tmp_path / "spec.yaml", make_spec())
    r = runner.invoke(cli.app, ["launch", f])  # (c) a fresh run over an old run's agents
    assert r.exit_code == 1 and "already holds agents of swarm 'demo'" in r.output, r.output
    assert "controller restores it on start" in r.output and "--adopt" not in r.output
    assert store.get_swarm("demo") is None
    f2 = write_spec(tmp_path / "other.yaml", make_spec("fresh", other={"replicas": 1}))
    assert runner.invoke(cli.app, ["launch", f2]).exit_code == 0  # an empty prefix: created


def test_two_launches_create_once(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    body = make_spec().model_dump(mode="json")
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        codes = [c.put("/swarms/demo", json=body, headers={"If-None-Match": "*"}).status_code
                 for _ in range(2)]
    assert codes == [200, 412]


# ----------------------------------------------------------------- (e) resume, (f) update


def test_update_replicas_spec_never_lowers_a_ramp_type_and_check_lists_deletions(
        tmp_path, monkeypatch, app, store, tick):
    wire(monkeypatch, app)
    ramp = {"start": 2, "add": 1, "every": "1h", "max": 10, "stagger": "0s"}
    f = write_spec(tmp_path / "spec.yaml", make_spec(
        r={"replicas": 0, "replicas_ramp": ramp}, b={"replicas": 2}))
    assert runner.invoke(cli.app, ["launch", f]).exit_code == 0
    tick()
    assert store.get_swarm("demo")["spec"].types["r"].replicas == 2
    write_spec(tmp_path / "spec.yaml", make_spec(r={"replicas": 0, "replicas_ramp": ramp},
                                             b={"replicas": 1}))
    r = runner.invoke(cli.app, ["update", f, "--replicas", "spec", "--check"])
    assert r.exit_code == 0, r.output
    assert "r: replicas kept (spec 0, live 2)" in r.output
    assert "deletes b: 1 agents (b-0001)" in r.output and "deletes r" not in r.output
    r = runner.invoke(cli.app, ["update", f, "--replicas", "spec"])
    assert r.exit_code == 0, r.output
    types = store.get_swarm("demo")["spec"].types
    assert types["r"].replicas == 2 and types["b"].replicas == 1


# ----------------------------------------------------------------- review fixes


def test_the_bucket_guard_stops_at_the_first_file(monkeypatch):
    from huggingface_hub import BucketFile

    from flock.worker.sync import HubBucketStore

    def tree(bucket_id, prefix, recursive):
        f = BucketFile.__new__(BucketFile)
        f.path = "b/demo/a/x"
        yield f
        raise AssertionError("listed past the first file")

    hub = HubBucketStore("hf://buckets/ns/name/b")
    monkeypatch.setattr(hub.api, "list_bucket_tree", tree)
    assert hub.first("demo/") == "demo/a/x"


def test_a_worker_killed_mid_release_reports_its_agents_unsynced(store, jobs, settings, clock):
    store.put_swarm(make_spec(), clock().timestamp())
    reconcile(store, jobs, clock(), settings)
    from flock.controller.loop import LoopEngine

    LoopEngine(store, send=lambda *a: None, clock=clock).pause("demo", release=True)
    reconcile(store, jobs, clock(), settings)  # drains sent
    w0, w1 = store.list_workers("demo")
    jobs.set_stage(w0["job_id"], "ERROR", "killed")  # no final heartbeat
    jobs.set_stage(w1["job_id"], "COMPLETED")  # drained and exited
    clock.advance(120)
    reconcile(store, jobs, clock(), settings)
    unsynced = store.get_control("demo", "unsynced")
    gone = {a["name"] for a in store.list_agents(worker=w0["id"])}
    assert set(unsynced) == gone and gone
    assert "ended before its final sync (ERROR: killed)" in next(iter(unsynced.values()))


def test_resume_diff_ignores_where_the_folder_is():
    from flock.models import SwarmSpec

    base = make_spec().model_dump(mode="json")

    def at(d, cmd="w"):
        return SwarmSpec.model_validate({**base, "compute": {"kind": "local"},
                                         "world": {"command": cmd, "dir": d}})

    assert ops.resume_diff(at("/runs/a/experiment/world"), at("/runs/b/experiment/world")) == []
    assert ops.resume_diff(at("/y"), at("/x", "w2"))


def test_the_losing_launch_discards_its_candidate_world(monkeypatch, store, clock):
    import flock.controller.api as api_mod
    from flock.world.host import WorldHost

    class Host(WorldHost):
        def __init__(self):
            super().__init__()
            self.discarded: list[str] = []

        def discard(self, swarm):
            self.discarded.append(swarm)

    spec_ = make_spec()

    def the_other_launch_wins(spec, **kw):  # it stores the swarm while this one resolves
        store.put_swarm(spec_, 1.0)

    monkeypatch.setattr(api_mod, "world_resolve", the_other_launch_wins)
    host = Host()
    app = create_app(store, TOKEN, clock=clock, worlds=host)
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = c.put("/swarms/demo", json=spec_.model_dump(mode="json"),
                  headers={"If-None-Match": "*"})
    assert r.status_code == 412 and host.discarded == ["demo"]


def test_local_runs_checkpoint_to_the_bucket_by_default():
    from flock.controller.localrun import state_env

    spec_, saved, none = make_spec(), lambda: "hf_saved", lambda: None
    env, note = state_env({}, spec_, saved)  # the saved hub token is enough
    assert env["FLOCK_STATE_URL"] == "hf://buckets/ns/sh/controller/demo" and note is None
    env, _ = state_env({"HF_TOKEN": "t"}, spec_, none)
    assert env["FLOCK_STATE_URL"] == "hf://buckets/ns/sh/controller/demo"
    env, note = state_env({}, spec_, none)  # a laptop without a token: off, said once
    assert "FLOCK_STATE_URL" not in env and note.startswith("checkpoint: off (no HF token")
    assert state_env({"FLOCK_STATE_URL": ""}, spec_, saved)[0]["FLOCK_STATE_URL"] == ""  # off
    assert state_env({"FLOCK_STATE_URL": "hf://x/y"}, spec_, none)[0]["FLOCK_STATE_URL"] == "hf://x/y"
    nob = make_spec().model_copy(update={"bucket": ""})
    assert state_env({}, nob, saved) == ({}, None)
