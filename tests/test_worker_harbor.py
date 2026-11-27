"""Worker wiring for harness harbor on a fake adapter: close() on drain and unassign (before the
final sync), no workdir instruction files, the heartbeat's `trial`, the worker's HarborHost
reaching each harbor adapter and closing at exit, and a local harbor worker's kill grace."""

from __future__ import annotations

import asyncio
import time

import pytest

from legion.controller import local
from legion.models import AssignCommand, Harness, LoopMeta, PromptCommand
from legion.worker.adapters.harbor import HarborAdapter
from legion.worker.agents import VERIFY_WAIT, AgentRuntime
from legion.worker.events import Outbox
from legion.worker.main import PREEMPT_EXIT, default_adapter_factory

from .test_worker_main import FakeController, make_worker
from .worker_fakes import MemoryStore, RecordingSync, ScriptedAdapter, event_types, spec, until

HARBOR = {"harness": "harbor", "harbor": {"task": {}, "config": {}},
          "system_prompt": "Not for the workdir."}


class TrialAdapter(ScriptedAdapter):
    harness = "harbor"

    def __init__(self, home, *, fail_close=False, closing=None, **kw):
        super().__init__(home, **kw)
        self.fail_close, self.closed, self.timeouts = fail_close, 0, []
        self.closing = closing if closing is not None else {"now": 0, "peak": 0}

    async def close(self, timeout=None):
        self.closed += 1
        self.timeouts.append(timeout)
        self.log.append(("close", self.home.name))
        c = self.closing
        c["now"] += 1
        c["peak"] = max(c["peak"], c["now"])
        await asyncio.sleep(0.05)
        c["now"] -= 1
        self.log.append(("closed", self.home.name))
        if self.fail_close:
            raise RuntimeError("verifier crashed")

    def trial_status(self):
        return {"id": "a-t1", "attempt": 1, "alive": self.closed == 0}


def runtime(tmp_path, order, **kw):
    a = TrialAdapter(tmp_path / "home", log=order, **kw)
    rt = AgentRuntime("researcher-0001", "researcher", spec(**HARBOR), swarm="sw",
                      work_root=tmp_path / "agents", adapter=a, outbox=Outbox(),
                      sync=RecordingSync(MemoryStore(), log=order))
    return rt, a


async def test_drain_closes_the_trial_before_the_final_sync(tmp_path):
    order: list = []
    rt, a = runtime(tmp_path, order)
    await rt.setup(restore=False)
    await rt.prompt("go")
    await until(lambda: rt.state.turns == 1)
    assert rt.heartbeat().trial == {"id": "a-t1", "attempt": 1, "alive": True}
    assert list(rt.workdir.iterdir()) == []  # the Trial has its instructions from Harbor
    await rt.drain()
    assert [k for k, _ in order[-4:]] == ["stop", "close", "closed", "push"]
    assert a.timeouts == [VERIFY_WAIT]
    assert rt.heartbeat().trial["alive"] is False


async def test_unassign_closes_the_trial_and_syncs_when_close_fails(tmp_path):
    order: list = []
    rt, a = runtime(tmp_path, order, fail_close=True)
    await rt.setup(restore=False)
    await rt.prompt("go")
    await until(lambda: rt.state.turns == 1)
    await rt.unassign()
    assert [k for k, _ in order[-3:]] == ["close", "closed", "push"] and a.closed == 1
    assert a.timeouts == [VERIFY_WAIT]
    assert "error" in event_types(rt)


def test_the_worker_host_reaches_harbor_adapters_only(tmp_path):
    worker, _ = make_worker(tmp_path, FakeController())
    make = default_adapter_factory(worker.cfg, worker.harbor)
    a = make("researcher-0001", spec(**HARBOR))
    assert isinstance(a, HarborAdapter) and a.host is worker.harbor
    assert worker.harbor.root == tmp_path / "agents" / "_harbor" / "_worker"


async def test_the_worker_closes_its_host_at_exit(tmp_path):
    worker, _ = make_worker(tmp_path, FakeController())
    closed = asyncio.Event()

    async def close():
        closed.set()

    worker.harbor.close = close
    run = asyncio.create_task(worker.run())
    await worker.drain("test")
    assert await asyncio.wait_for(run, 5) == 0 and closed.is_set()


@pytest.mark.parametrize("path", ["terminate", "abandon", "preempt"])
async def test_every_exit_path_closes_the_trials_in_parallel(tmp_path, path):
    ctrl, order, adapters, closing = FakeController(), [], {}, {"now": 0, "peak": 0}
    sync = RecordingSync(MemoryStore(), log=order)
    names = ["researcher-0001", "researcher-0002"]
    ts = spec(**HARBOR, worker={"sync_interval_s": 60})  # only the exit path's sync pushes
    ctrl.batches = [
        [AssignCommand(agent=n, type="researcher", spec=ts, restore=False) for n in names],
        [PromptCommand(agent=n, prompt_id=f"p-{n}", text="go") for n in names],
    ]
    worker, _ = make_worker(tmp_path, ctrl, sync=sync)

    def factory(name, sp):
        adapters[name] = TrialAdapter(tmp_path / "home" / name, log=order, closing=closing)
        return adapters[name]

    worker.adapter_factory = factory
    run = asyncio.create_task(worker.run())
    await until(lambda: len(adapters) == 2 and all(a.sent for a in adapters.values()))
    await getattr(worker, path)("test")
    assert await asyncio.wait_for(run, 5) == (PREEMPT_EXIT if path == "preempt" else 0)
    assert all(a.closed == 1 for a in adapters.values()) and closing["peak"] == 2
    assert [t for a in adapters.values() for t in a.timeouts] == [0, 0]  # cancelled, unverified
    if path == "preempt":  # the sync gets the sessions Harbor pulled as the Trial stopped
        for n in names:
            assert order.index(("closed", n)) < order.index(("push", n))


def test_a_local_harbor_worker_gets_time_to_stop_its_trials(tmp_path, monkeypatch):
    """SIGTERM, then 0.3 s until the worker has stopped its Trials: it is not SIGKILLed after
    the 0.1 s grace."""
    monkeypatch.setattr(local, "HARBOR_STOP_GRACE", 5.0)
    script = "trap 'sleep 0.3; exit 3' TERM; echo up; sleep 30 & wait"
    jobs = local.LocalJobsClient(root=tmp_path, command=["sh", "-c", script])
    codes = {}
    for harness in (Harness.harbor,):
        ts = spec().model_copy(update={"harness": harness})
        job = jobs.launch(harness.value, ts, {"WORKER_ID": harness.value}, {}, {}, 60, namespace="")
        while "up" not in "".join(jobs.logs_tail(job, namespace="")):
            time.sleep(0.01)
        jobs.kill_all(grace=0.1)
        codes[harness.value] = jobs.procs[job].popen.returncode
    assert codes == {"harbor": 3}


async def test_gateway_usage_is_the_agents_and_its_cap_refuses(tmp_path):
    """resources.gateway's per-call usage (the host's emit) goes into the agent's tokens, which
    the heartbeat reports; its cap check is the agent's `loop.while.agent_usd`."""
    ctrl = FakeController()
    ts = spec(**HARBOR, pricing=[1.0, 1.0, 0.5],
              loop={"prompt": "go", "while": {"agent_usd": 0.01}})
    ctrl.batches = [[AssignCommand(agent="researcher-0001", type="researcher", spec=ts,
                                   restore=False)]]
    worker, _ = make_worker(tmp_path, ctrl)
    worker.adapter_factory = lambda name, sp: TrialAdapter(tmp_path / "home" / name, turn_time=0.5)
    run = asyncio.create_task(worker.run())
    await until(lambda: "researcher-0001" in worker.agents)
    rt = worker.agents["researcher-0001"]
    host = worker.harbor
    assert not host.capped("researcher-0001") and host.capped("researcher-0009")  # not here
    host.emit("researcher-0001", {"resource": "gateway", "kind": "usage",
                                  "data": {"input": 9000, "output": 2000, "cache_read": 0}})
    assert rt.heartbeat().tokens["m-1"].input == 9000 and "resource" not in event_types(rt)
    assert host.capped("researcher-0001")  # 0.011 USD >= 0.01
    final = LoopMeta(kind="final", type="researcher")
    await rt.prompt("Wrap up.", loop=final)  # exempt from agent_usd until its turn ends
    assert not host.capped("researcher-0001")
    await until(lambda: rt.turn_id is not None)
    assert not host.capped("researcher-0001")
    await until(lambda: rt.state.turns == 1)
    assert host.capped("researcher-0001")

    async def broken(*args, **kw):
        raise RuntimeError("no sandbox")

    await until(lambda: rt.phase == "none")  # the burst ended (idle_grace)
    rt.adapter.start = broken  # the final prompt's burst fails: it exempts no later turn
    await rt.prompt("Wrap up.", loop=final)
    await until(lambda: "error" in event_types(rt) and rt.phase == "none")
    assert host.capped("researcher-0001")
    await worker.drain("test")
    await asyncio.wait_for(run, 5)
