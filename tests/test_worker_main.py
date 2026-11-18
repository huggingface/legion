"""Worker loop against an in-process fake controller (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from flock.models import (
    AssignCommand,
    CommandBatch,
    DrainCommand,
    EventBatch,
    PromptCommand,
    StopCommand,
    UnassignCommand,
    WorkerHeartbeat,
    WorkerMem,
    type_spec_rev,
)
from flock.worker import main as wmain
from flock.worker.events import utcnow
from flock.worker.main import ControllerClient, Worker, WorkerConfig

from .worker_fakes import MemoryStore, RecordingSync, ScriptedAdapter, spec, until

WID = "sw-researcher-w0000"


class FakeController:
    """Serves queued command batches; records heartbeats and events; can fail on demand."""

    def __init__(self) -> None:
        self.batches: list[list] = []
        self.heartbeats: list[WorkerHeartbeat] = []
        self.events: list = []
        self.fail_next: list[int] = []  # status codes to return before succeeding
        self.polls = 0
        self.auth: set[str] = set()

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.auth.add(request.headers.get("authorization", ""))
        path = request.url.path
        if self.fail_next:
            return httpx.Response(self.fail_next.pop(0), json={"detail": "nope"})
        if path == f"/workers/{WID}/commands":
            self.polls += 1
            assert request.url.params["wait"] == "25"
            cmds = self.batches.pop(0) if self.batches else []
            return httpx.Response(200, json=CommandBatch(commands=cmds).model_dump(mode="json"))
        if path == f"/workers/{WID}/heartbeat":
            self.heartbeats.append(WorkerHeartbeat.model_validate_json(request.content))
            return httpx.Response(200, json={"ok": True})
        if path == f"/workers/{WID}/events":
            self.events += EventBatch.model_validate_json(request.content).events
            return httpx.Response(200, json={"ok": True})
        return httpx.Response(404)


def make_worker(tmp_path, ctrl: FakeController, *, deadline_in: float = 3600, sync=None, adapters=None):
    cfg = WorkerConfig(url="http://controller", token="tok", worker_id=WID, swarm="sw", type="researcher",
                       deadline=utcnow() + timedelta(seconds=deadline_in), bucket=None,
                       work_root=tmp_path / "agents", heartbeat_interval=0.1, flush_interval=0.05,
                       sync_interval=0.2, poll_min_interval=0.02)
    adapters = adapters if adapters is not None else {}

    def factory(name, spec):
        adapters[name] = ScriptedAdapter(tmp_path / "home" / name, turn_time=0.2)
        return adapters[name]

    client = ControllerClient(cfg.url, cfg.token, transport=httpx.MockTransport(ctrl.handler))
    return Worker(cfg, client, adapter_factory=factory, sync=sync), adapters


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(wmain, "_backoff", lambda attempt: 0.01)
    monkeypatch.setattr(wmain, "_retry_delay", lambda attempt, retry_after: 0.01)


def test_config_from_env(tmp_path):
    cfg = WorkerConfig.from_env({
        "FLOCK_URL": "https://c.hf.space/", "FLOCK_TOKEN": "t", "WORKER_ID": WID,
        "SWARM": "sw", "TYPE": "researcher", "DEADLINE": "2026-10-15T09:12:00Z",
        "BUCKET_ID": "ns/b"})
    assert cfg.url == "https://c.hf.space" and cfg.work_root.as_posix() == "/work/agents"
    assert cfg.deadline.year == 2026 and cfg.deadline.tzinfo is not None
    assert cfg.sync_interval == 180 and cfg.bucket == "ns/b"
    assert cfg.drain_window == timedelta(minutes=10)  # default until the env says otherwise


async def test_drain_window_from_env_applies_before_any_assign(tmp_path):
    # lifetime 6 min, drain window 5 min: must not drain at once just because no assign came yet
    cfg = WorkerConfig.from_env({
        "FLOCK_URL": "http://c", "FLOCK_TOKEN": "t", "WORKER_ID": WID,
        "SWARM": "sw", "TYPE": "researcher", "DRAIN_WINDOW": "5m",
        "DEADLINE": (utcnow() + timedelta(minutes=6)).isoformat()})
    assert cfg.drain_window == timedelta(minutes=5)
    w = Worker(cfg, ControllerClient(cfg.url, cfg.token, transport=httpx.MockTransport(
        FakeController().handler)))
    task = asyncio.create_task(w.deadline_loop())
    await asyncio.sleep(0.1)
    assert not w.draining
    task.cancel()
    await w.client.aclose()


async def test_dispatch_against_fake_controller(tmp_path):
    ctrl = FakeController()
    ctrl.fail_next = [503, 500]  # the worker must survive controller 5xx
    ctrl.batches = [
        [AssignCommand(agent="researcher-0001", type="researcher", spec=spec(), restore=False),
         AssignCommand(agent="researcher-0002", type="researcher", spec=spec(), restore=False)],
        [PromptCommand(agent="researcher-0001", prompt_id="p1", text="hello"),
         AssignCommand(agent="researcher-0001", type="researcher", spec=spec(), restore=False)],  # repeat: no-op
    ]
    worker, adapters = make_worker(tmp_path, ctrl)
    run = asyncio.create_task(worker.run())
    await until(lambda: any(e.type == "turn_ended" for e in ctrl.events), timeout=5)
    await until(lambda: any(a.state == "idle" and a.turns == 1 for hb in ctrl.heartbeats for a in hb.agents))
    assert ctrl.auth == {"Bearer tok"}
    assert adapters["researcher-0001"].sent == ["hello"] and adapters["researcher-0001"].starts == [None]
    assert adapters["researcher-0002"].starts == []
    hb = ctrl.heartbeats[-1]
    assert hb.worker == WID and hb.draining is False
    assert {a.name for a in hb.agents} == {"researcher-0001", "researcher-0002"}
    a1, a2 = sorted(hb.agents, key=lambda a: a.name)
    assert a1.rev == a1.start_rev == type_spec_rev(spec()) and a2.rev is a2.start_rev is None
    assert hb.knobs_rev == worker.knobs_rev and hb.restore is None and hb.tools is None
    types = [e.type for e in ctrl.events if e.agent == "researcher-0001"]
    assert types[:2] == ["session_started", "turn_started"] and "message" in types

    # stop, unassign (twice: the second is a no-op), unknown-agent prompt, then drain
    ctrl.batches.append([StopCommand(agent="researcher-0001"), UnassignCommand(agent="researcher-0002"),
                         UnassignCommand(agent="researcher-0002"),
                         PromptCommand(agent="nobody-0001", prompt_id="p2", text="x")])
    await until(lambda: "researcher-0002" not in worker.agents)
    ctrl.batches.append([DrainCommand()])
    assert await asyncio.wait_for(run, 5) == 0
    assert ctrl.heartbeats[-1].draining is True


async def test_heartbeat_carries_memory_restores_and_knobs(tmp_path, monkeypatch):
    from datetime import UTC, datetime

    from flock.worker.sync import BucketSync

    worker, _ = make_worker(tmp_path, FakeController(), sync=BucketSync(MemoryStore(), "sw"))
    t0 = datetime(2026, 10, 28, 9, tzinfo=UTC)
    worker.sync.restore_progress = {"a/": [3, 10, t0 + timedelta(minutes=1)], "b/": [0, 5, t0]}
    worker.mem = WorkerMem(current_mb=1, limit_mb=2)
    hb = worker.heartbeat_payload()
    assert hb.restore.model_dump() == {"agents": 2, "files_done": 3, "files_total": 15,
                                       "started": t0}
    assert hb.mem.limit_mb == 2 and hb.tools is None and len(hb.knobs_rev) == 12
    monkeypatch.setenv("FLOCK_SYNC_EXCLUDE", "big/")  # any knob changes the hash
    assert wmain.knobs_rev(worker.cfg) != hb.knobs_rev
    await worker.client.aclose()


async def test_live_worker_knobs_beat_env_and_bump_knobs_rev(tmp_path, monkeypatch):
    from flock.worker import sync as sync_mod
    from flock.worker.sync import is_excluded, restore_threads, workdir_excludes

    monkeypatch.setattr(sync_mod, "SPEC_KNOBS", {})
    monkeypatch.setenv("FLOCK_SYNC_EXCLUDE", "envonly/")
    monkeypatch.setenv("FLOCK_MEM_DRAIN", "0.8")
    worker, _ = make_worker(tmp_path, FakeController())
    rev0 = worker.knobs_rev
    knobs = {"sync_interval_s": 30, "drain_budget_s": 9, "sync_exclude": ["big/"],
             "restore_threads": 2, "memory": {"drain": 0.9}}
    worker._live_from({"type_spec": spec(worker=knobs).model_dump(mode="json")})
    assert (worker.cfg.sync_interval, worker.cfg.drain_budget) == (30, 9)
    assert restore_threads() == 2
    pats = workdir_excludes()  # the spec's list replaces the env's, the defaults stay
    assert is_excluded("big/x", pats) and not is_excluded("envonly/x", pats)
    assert is_excluded("node_modules/x", pats)
    assert worker.mem_knob("FLOCK_MEM_DRAIN", 0.85) == 0.9
    assert worker.mem_knob("FLOCK_MEM_PREEMPT", 0.93) == 0.93
    assert worker.knobs_rev != rev0 and worker.heartbeat_payload().knobs_rev == worker.knobs_rev
    worker._live_from({"type_spec": spec().model_dump(mode="json")})  # back to env, defaults
    assert worker.cfg.sync_interval == 0.2
    assert is_excluded("envonly/x", workdir_excludes()) and worker.knobs_rev == rev0
    assert worker.mem_knob("FLOCK_MEM_DRAIN", 0.85) == 0.8
    await worker.client.aclose()


async def test_memory_loop_samples_the_cgroup_and_logs(tmp_path, monkeypatch, caplog):
    from flock.worker.cgroup import CgroupMem

    from .test_worker_cgroup import v2

    v2(tmp_path / "fs")
    monkeypatch.setattr(wmain, "MEMORY_SAMPLE", 0.01)
    monkeypatch.setenv("FLOCK_MEMORY_LOG_S", "0.02")
    worker, _ = make_worker(tmp_path, FakeController())
    worker.cgroup = CgroupMem(tmp_path / "fs", env={})
    caplog.set_level("INFO", logger="flock.worker")
    task = asyncio.create_task(worker.memory_loop())
    try:
        await until(lambda: "memory: rss" in caplog.text)
    finally:
        task.cancel()
    assert worker.mem.limit_mb == 11264 and worker.mem.nonreclaim_mb == 1300
    assert worker.heartbeat_payload().mem == worker.mem
    await worker.client.aclose()


async def test_drain_sequence_at_deadline(tmp_path):
    ctrl = FakeController()
    sync = RecordingSync(MemoryStore())
    s = spec(drain_window="1s", idle_grace="10s")
    ctrl.batches = [
        [AssignCommand(agent="researcher-0001", type="researcher", spec=s, restore=False),
         AssignCommand(agent="researcher-0002", type="researcher", spec=s, restore=False)],
        [PromptCommand(agent="researcher-0001", prompt_id="p1", text="long turn")],
    ]
    adapters: dict = {}
    worker, _ = make_worker(tmp_path, ctrl, deadline_in=1.6, sync=sync, adapters=adapters)

    def factory(name, sp):
        adapters[name] = ScriptedAdapter(tmp_path / "home" / name, turn_time=0.8)
        return adapters[name]

    worker.adapter_factory = factory
    run = asyncio.create_task(worker.run())
    await until(lambda: worker.draining, timeout=5)
    # no new harness processes once draining
    await worker.dispatch(PromptCommand(agent="researcher-0002", prompt_id="p2", text="late"))
    code = await asyncio.wait_for(run, 5)
    assert code == 0
    assert adapters["researcher-0002"].starts == []
    ev1 = [e.type for e in ctrl.events if e.agent == "researcher-0001"]
    assert "turn_ended" in ev1 and "turn_interrupted" not in ev1  # the in-flight turn finished
    assert "prompt_deferred" in [e.type for e in ctrl.events if e.agent == "researcher-0002"]
    assert {"researcher-0001", "researcher-0002"} <= set(sync.pushed)  # final sync per agent
    assert ctrl.heartbeats[-1].draining is True
    assert [hb.drained for hb in ctrl.heartbeats].count(True) == 1 and ctrl.heartbeats[-1].drained
    assert adapters["researcher-0001"].stops == 1


async def test_assign_restore_goes_through_sync(tmp_path):
    ctrl = FakeController()
    order: list = []
    sync = RecordingSync(MemoryStore(), log=order)
    ctrl.batches = [[AssignCommand(agent="researcher-0001", type="researcher", spec=spec(), restore=True),
                     PromptCommand(agent="researcher-0001", prompt_id="p", text="go")]]
    worker, adapters = make_worker(tmp_path, ctrl, sync=sync)
    run = asyncio.create_task(worker.run())
    await until(lambda: "researcher-0001" in adapters and adapters["researcher-0001"].sent == ["go"])
    assert sync.restored == ["researcher-0001"] and order[0] == ("restore", "researcher-0001")
    await worker.drain("test")
    assert await asyncio.wait_for(run, 5) == 0


@pytest.mark.parametrize("status", [404, 409])
async def test_stale_slot_exits_zero_without_sync(tmp_path, status):
    ctrl = FakeController()
    sync = RecordingSync(MemoryStore())
    ctrl.batches = [[AssignCommand(agent="researcher-0001", type="researcher", spec=spec(idle_grace="10s"),
                                   restore=False),
                     PromptCommand(agent="researcher-0001", prompt_id="p", text="go")]]
    worker, adapters = make_worker(tmp_path, ctrl, sync=sync)
    run = asyncio.create_task(worker.run())
    await until(lambda: "researcher-0001" in adapters and adapters["researcher-0001"].sent == ["go"])
    ctrl.fail_next = [status] * 50
    assert await asyncio.wait_for(run, 5) == 0
    assert adapters["researcher-0001"].stops == 1 and sync.pushed == []


def test_heartbeat_json_shape(tmp_path):
    ctrl = FakeController()
    worker, _ = make_worker(tmp_path, ctrl)
    body = json.loads(worker.heartbeat_payload().model_dump_json())
    assert body["worker"] == WID and body["agents"] == [] and body["draining"] is False


async def test_worker_sends_its_incarnation_on_polls_and_heartbeats(tmp_path):
    cfg = WorkerConfig.from_env({
        "FLOCK_URL": "http://c", "FLOCK_TOKEN": "t", "WORKER_ID": WID,
        "SWARM": "sw", "TYPE": "researcher", "WORKER_INCARNATION": "inc-1",
        "DEADLINE": (utcnow() + timedelta(hours=1)).isoformat()})
    assert cfg.incarnation == "inc-1"
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, request.url.params.get("incarnation")))
        if request.url.path.endswith("/commands"):
            return httpx.Response(200, json={"commands": []})
        return httpx.Response(200, json={"ok": True})

    client = ControllerClient(cfg.url, cfg.token, transport=httpx.MockTransport(handler),
                              incarnation=cfg.incarnation)
    w = Worker(cfg, client)
    await client.commands(WID, 0)
    await client.heartbeat(w.heartbeat_payload())
    await client.aclose()
    assert seen == [(f"/workers/{WID}/commands", "inc-1"), (f"/workers/{WID}/heartbeat", "inc-1")]


def test_run_and_exit_does_not_wait_for_a_stuck_executor_thread():
    """1k run, check 5: a drained worker logged "drained; exiting" but its job stayed RUNNING.
    asyncio.run's teardown joins the default executor (and Python joins non-daemon threads at
    exit), so one blocked thread kept the process alive. The worker must exit regardless."""
    import threading

    release = threading.Event()
    exits: list[int] = []
    leftover: list[asyncio.Task] = []

    async def worker_run() -> int:
        asyncio.get_running_loop().run_in_executor(None, release.wait)  # never finishes
        leftover.append(asyncio.get_running_loop().create_task(asyncio.Event().wait()))  # ditto
        await asyncio.sleep(0)
        return 0

    def fake_exit(code: int) -> None:
        exits.append(code)
        # os._exit would end the process here; the test goes on, so finish the task it left
        # pending (else its loop is collected with it: "Task was destroyed but it is pending").
        task = leftover[0]
        task.cancel()
        task.get_loop().run_until_complete(asyncio.gather(task, return_exceptions=True))

    t = threading.Thread(target=wmain.run_and_exit, args=(worker_run(),),
                         kwargs={"exit": fake_exit}, daemon=True)
    t.start()
    t.join(5)
    release.set()
    assert exits == [0]


async def test_periodic_sync_skips_an_agent_until_its_restore_finished(tmp_path):
    """A restore wave outlasts several sync intervals: the half-restored workdir must never
    be mirrored over the bucket copy (it would delete what is not downloaded yet)."""
    from .worker_fakes import GatedStore

    ctrl = FakeController()
    store = GatedStore()
    pre = "sw/researcher/researcher-0001/"
    store.files = {pre + "workdir/a.txt": b"A", pre + "workdir/b.txt": b"B"}
    before = dict(store.files)
    sync = RecordingSync(store)
    ctrl.batches = [[AssignCommand(agent="researcher-0001", type="researcher", spec=spec(),
                                   restore=True),
                     PromptCommand(agent="researcher-0001", prompt_id="p", text="go")]]
    worker, adapters = make_worker(tmp_path, ctrl, sync=sync)
    run = asyncio.create_task(worker.run())
    try:
        await until(lambda: store.waiting > 0)
        rt = worker.agents["researcher-0001"]
        rt.dirty = True
        await asyncio.sleep(0.7)  # > 3 sync intervals while the restore hangs
        assert sync.pushed == [] and store.files == before and store.synced == []
    finally:
        store.gate.set()
    await until(lambda: adapters["researcher-0001"].sent == ["go"])
    await until(lambda: "researcher-0001" in sync.pushed)  # periodic pushes resume after setup
    assert store.files[pre + "workdir/a.txt"] == b"A" and store.files[pre + "workdir/b.txt"] == b"B"
    await worker.drain("test")
    assert await asyncio.wait_for(run, 5) == 0


def test_memory_helpers_trim_and_cap_arenas_safely(monkeypatch):
    rss = wmain.rss_mb()
    assert rss is None or rss > 0
    wmain.trim_heap()  # no-op off glibc
    assert wmain.limit_malloc_arenas({"MALLOC_ARENA_MAX": "2"}) is False  # the env wins
    calls = []

    class Libc:
        def malloc_trim(self, pad):
            calls.append(("trim", pad))

        def mallopt(self, param, value):
            calls.append(("mallopt", param, value))
            return 1

    monkeypatch.setattr(wmain, "_glibc", Libc)
    assert wmain.limit_malloc_arenas({}) is True
    wmain.trim_heap()
    assert calls == [("mallopt", -8, wmain.ARENAS), ("trim", 0)]


async def test_memory_loop_trims_and_logs_the_worker_sizes(tmp_path, monkeypatch, caplog):
    import logging

    ctrl = FakeController()
    w, _ = make_worker(tmp_path, ctrl)
    trims = []
    monkeypatch.setattr(wmain, "trim_heap", lambda: trims.append(1))
    monkeypatch.setenv("FLOCK_MEMORY_LOG_S", "0.01")
    monkeypatch.setattr(wmain, "MEMORY_SAMPLE", 0.01)
    caplog.set_level(logging.INFO, logger="flock.worker")
    task = asyncio.create_task(w.memory_loop())
    await until(lambda: len(trims) >= 2)
    task.cancel()
    report = w.memory_report()
    assert report["outbox_events"] == 0 and report["outbox_bytes"] == 0 and report["agents"] == 0
    if wmain.rss_mb() is not None:
        assert any("memory: rss" in r.message and "outbox_bytes" in r.message for r in caplog.records)
    monkeypatch.setenv("FLOCK_MEMORY_LOG_S", "0")  # log off; the cgroup is still sampled
    trims.clear()
    w.mem = None
    task = asyncio.create_task(w.memory_loop())
    await until(lambda: w.mem is not None)
    await asyncio.sleep(0.05)
    task.cancel()
    assert trims == []


async def test_memory_loop_survives_a_failed_sample(tmp_path, monkeypatch):
    monkeypatch.setattr(wmain, "MEMORY_SAMPLE", 0.01)
    worker, _ = make_worker(tmp_path, FakeController())
    calls = []

    def sample():
        calls.append(1)
        if len(calls) == 2:
            raise PermissionError("nope")
        return {"limit_mb": 1.0}

    monkeypatch.setattr(worker.cgroup, "sample", sample)
    task = asyncio.create_task(worker.memory_loop())
    await until(lambda: len(calls) >= 2)
    assert worker.mem is None  # no stale reading after a failure
    await until(lambda: len(calls) >= 3 and worker.mem is not None)
    task.cancel()
    await worker.client.aclose()


def test_restore_threads_env_is_capped_and_spec_knobs_rebind(monkeypatch):
    from flock.worker import sync as sync_mod
    monkeypatch.setenv("FLOCK_RESTORE_THREADS", "500")
    assert sync_mod.restore_threads() == 32
    monkeypatch.setenv("FLOCK_RESTORE_THREADS", "0")
    assert sync_mod.restore_threads() == sync_mod.RESTORE_THREADS


@pytest.mark.parametrize("bad", ["../x", "/x", "a/..", "."])
def test_workdir_ephemeral_rejects_escapes(bad):
    from flock.models import WorkdirSpec
    with pytest.raises(ValueError):
        WorkdirSpec(ephemeral=[bad])
    assert WorkdirSpec(ephemeral=["scratch"]).ephemeral == ["scratch/"]
