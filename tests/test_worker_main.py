"""Worker loop against an in-process fake controller (httpx.MockTransport)."""

from __future__ import annotations

import asyncio
import json
from datetime import timedelta

import httpx
import pytest

from swarm_harness.models import (
    AssignCommand,
    CommandBatch,
    DrainCommand,
    EventBatch,
    PromptCommand,
    StopCommand,
    UnassignCommand,
    WorkerHeartbeat,
)
from swarm_harness.worker import main as wmain
from swarm_harness.worker.events import utcnow
from swarm_harness.worker.main import ControllerClient, Worker, WorkerConfig

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


def test_config_from_env(tmp_path):
    cfg = WorkerConfig.from_env({
        "SWARM_HARNESS_URL": "https://c.hf.space/", "SWARM_HARNESS_TOKEN": "t", "WORKER_ID": WID,
        "SWARM": "sw", "TYPE": "researcher", "DEADLINE": "2026-10-15T09:12:00Z",
        "BUCKET": "hf://buckets/ns/b"})
    assert cfg.url == "https://c.hf.space" and cfg.work_root.as_posix() == "/work/agents"
    assert cfg.deadline.year == 2026 and cfg.deadline.tzinfo is not None
    assert cfg.sync_interval == 180 and cfg.bucket == "hf://buckets/ns/b"
    assert cfg.drain_window == timedelta(minutes=10)  # default until the env says otherwise


async def test_drain_window_from_env_applies_before_any_assign(tmp_path):
    # lifetime 6 min, drain window 5 min: must not drain at once just because no assign came yet
    cfg = WorkerConfig.from_env({
        "SWARM_HARNESS_URL": "http://c", "SWARM_HARNESS_TOKEN": "t", "WORKER_ID": WID,
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
