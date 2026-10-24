"""Worker preemption (SIGUSR1/SIGTERM; docs/slurm.md): interrupt, parallel sync, marker, flag,
exit 75, all inside FLOCK_DRAIN_BUDGET_S; then a restart with the same name restores. Also the
controller host file (Slurm) and a container with no usable HOME."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import threading
import time
from datetime import timedelta

import httpx
import pytest

from flock.models import AssignCommand, PromptCommand
from flock.worker import main as wmain
from flock.worker.events import utcnow
from flock.worker.main import PREEMPT_EXIT, ControllerClient, Worker, WorkerConfig

from .test_worker_main import WID, FakeController
from .worker_fakes import MemoryStore, RecordingSync, ScriptedAdapter, spec, until

N_AGENTS = 25  # agents_per_worker in run 3
UPLOAD_S = 0.25  # per blocking bucket call: 25 agents x 2 calls = 12.5 s if run one by one


class SlowStore(MemoryStore):
    """A bucket whose every call blocks its thread for `delay` s (counts concurrent calls)."""

    def __init__(self, delay: float = UPLOAD_S) -> None:
        super().__init__()
        self.delay = delay
        self.active = self.peak = 0
        self._lock = threading.Lock()

    def _slow(self):
        with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)
        time.sleep(self.delay)
        with self._lock:
            self.active -= 1

    def upload(self, files):
        self._slow()
        super().upload(files)

    def sync_up(self, local, prefix):
        self._slow()
        super().sync_up(local, prefix)


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(wmain, "_backoff", lambda attempt: 0.01)
    monkeypatch.setattr(wmain, "_retry_delay", lambda attempt, retry_after: 0.01)


def make(tmp_path, ctrl, store, *, budget=20.0, turn_time=0.05, adapters=None):
    cfg = WorkerConfig(url="http://controller", token="tok", worker_id=WID, swarm="sw",
                       type="researcher", deadline=utcnow() + timedelta(hours=1), bucket=None,
                       work_root=tmp_path / "agents", heartbeat_interval=0.1,
                       flush_interval=0.05, sync_interval=3600, poll_min_interval=0.02,
                       drain_budget=budget)
    adapters = adapters if adapters is not None else {}

    def factory(name, sp):
        adapters[name] = ScriptedAdapter(tmp_path / "home" / name, turn_time=turn_time)
        return adapters[name]

    client = ControllerClient(cfg.url, cfg.token, transport=httpx.MockTransport(ctrl.handler))
    sync = RecordingSync(store)
    return Worker(cfg, client, adapter_factory=factory, sync=sync), adapters, sync


def names(n=N_AGENTS):
    return [f"researcher-{i:04d}" for i in range(n)]


async def test_sigusr1_preempts_within_budget_and_restart_restores(tmp_path):
    store = SlowStore()
    ctrl = FakeController()
    s = spec(idle_grace="10s", turn_timeout="10m")
    ctrl.batches = [
        [AssignCommand(agent=n, type="researcher", spec=s, restore=False) for n in names()],
        [PromptCommand(agent="researcher-0000", prompt_id="p1", text="a very long turn")],
    ]
    worker, adapters, sync = make(tmp_path, ctrl, store, turn_time=600)
    run = asyncio.create_task(worker.run())
    await until(lambda: len(adapters) == N_AGENTS and worker.agents["researcher-0000"].turn_id,
                timeout=5)
    for n in names():  # each agent has some work on disk
        worker.agents[n].workdir.mkdir(parents=True, exist_ok=True)
        (worker.agents[n].workdir / "notes.md").write_text(f"{n} was here\n")
    hb_before = len(ctrl.heartbeats)

    t0 = time.monotonic()
    os.kill(os.getpid(), signal.SIGUSR1)
    code = await asyncio.wait_for(run, 25)
    took = time.monotonic() - t0

    assert code == PREEMPT_EXIT
    # sequential would be N_AGENTS * 2 * UPLOAD_S = 12.5 s; in parallel it is ~2 calls' worth
    assert took < 4.0, took
    assert store.peak >= 10
    print(f"\npreempt drain: {N_AGENTS} agents in {took:.2f} s with {UPLOAD_S}s bucket calls "
          f"(peak {store.peak} concurrent); at the measured ~0.3 s/agent (7 s per 25 agents, "
          "one by one) the parallel drain stays near 1-3 s of the 20 s budget")
    # the running turn was interrupted, not waited for (it had 600 s to go)
    ev0 = [e.type for e in ctrl.events if e.agent == "researcher-0000"]
    assert "turn_interrupted" in ev0 and "turn_ended" not in ev0
    assert set(sync.pushed) == set(names())  # every agent synced
    for n in names():
        assert store.files[f"sw/researcher/{n}/workdir/notes.md"] == f"{n} was here\n".encode()
    marker = json.loads(store.files[f"sw/_workers/{WID}/preempted.json"])
    assert marker["worker"] == WID and marker["reason"] == "SIGUSR1"
    # `preempted` only on heartbeats sent after the sync: the last one
    flags = [hb.preempted for hb in ctrl.heartbeats[hb_before:]]
    assert flags[-1] is True and flags.count(True) == 1
    assert ctrl.heartbeats[-1].interrupted == ["researcher-0000"]
    assert ctrl.heartbeats[-1].draining is True

    # the relaunched job (same name) restores every agent from the bucket and resumes
    ctrl2 = FakeController()
    ctrl2.batches = [[AssignCommand(agent=n, type="researcher", spec=s, restore=True,
                                    resume_prompt="resume" if n == "researcher-0000" else None)
                      for n in names()]]
    for n in names():  # node-local /tmp is wiped: nothing left on disk
        for p in sorted((tmp_path / "agents" / n).rglob("*"), reverse=True):
            p.unlink() if p.is_file() else p.rmdir()
    store.delay = 0.0
    worker2, adapters2, sync2 = make(tmp_path / "second", ctrl2, store)
    worker2.cfg.work_root = tmp_path / "agents"  # same absolute paths as before
    run2 = asyncio.create_task(worker2.run())
    await until(lambda: "researcher-0000" in adapters2 and adapters2["researcher-0000"].sent,
                timeout=10)
    assert set(sync2.restored) == set(names())
    assert (tmp_path / "agents" / "researcher-0007" / "notes.md").read_text() == \
        "researcher-0007 was here\n"
    assert adapters2["researcher-0000"].sent[0].startswith("resume")
    await worker2.drain("test")
    assert await asyncio.wait_for(run2, 10) == 0


async def test_preempt_is_bounded_by_the_budget(tmp_path):
    store = SlowStore(delay=5.0)  # a bucket that hangs far past the budget
    ctrl = FakeController()
    ctrl.batches = [[AssignCommand(agent=n, type="researcher", spec=spec(idle_grace="10s"),
                                   restore=False) for n in names(3)]]
    worker, _adapters, _ = make(tmp_path, ctrl, store, budget=1.5)
    run = asyncio.create_task(worker.run())
    await until(lambda: len(worker.agents) == 3, timeout=5)
    t0 = time.monotonic()
    os.kill(os.getpid(), signal.SIGTERM)  # outside compute local, SIGTERM preempts too
    assert await asyncio.wait_for(run, 10) == PREEMPT_EXIT
    assert time.monotonic() - t0 < 2.5


def test_budget_from_env_and_host_file(tmp_path):
    host = tmp_path / "controller.host"
    host.write_text("ip-10-0-0-7\n")
    env = {"FLOCK_URL": "http://${CTL_HOST}:7860", "FLOCK_CTL_HOST_FILE": str(host),
           "FLOCK_TOKEN": "t", "WORKER_ID": WID, "SWARM": "sw", "TYPE": "researcher",
           "DEADLINE": "2026-10-25T00:00:00+00:00", "FLOCK_DRAIN_BUDGET_S": "12"}
    cfg = WorkerConfig.from_env(env)
    assert cfg.url == "http://ip-10-0-0-7:7860" and cfg.drain_budget == 12.0
    assert cfg.url_template == "http://${CTL_HOST}:7860"
    assert WorkerConfig.from_env({**env, "FLOCK_DRAIN_BUDGET_S": ""}).drain_budget == 20.0


async def test_client_follows_a_moved_controller(tmp_path):
    host = tmp_path / "controller.host"
    host.write_text("old-node\n")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "old-node":
            raise httpx.ConnectError("gone")
        return httpx.Response(200, json={"ok": True})

    client = ControllerClient(
        "http://old-node:7860", "tok", transport=httpx.MockTransport(handler),
        sleep=lambda s: asyncio.sleep(0),
        resolve_url=lambda: wmain.expand_env("http://${CTL_HOST}:7860",
                                             {"CTL_HOST": wmain.read_host_file(str(host))}))
    host.write_text("new-node\n")  # the controller was requeued on another node
    from flock.models import WorkerHeartbeat

    hb = WorkerHeartbeat(worker=WID, deadline=utcnow(), agents=[])
    assert await client.heartbeat(hb) == {"ok": True}
    assert seen == ["old-node", "new-node"] and client.stats["moved"] == 1
    await client.aclose()


def test_worker_startup_and_folder_tool_without_home(tmp_path, monkeypatch):
    """A container with no HOME (or a read-only one): caches go next to the work root, adapters
    never write under HOME, and a folder tool (python -I subprocess) still runs."""
    from flock.home import home
    from flock.tools.loader import load_dir
    from flock.tools.runner import run_tool

    ro_home = tmp_path / "ro-home"
    ro_home.mkdir()
    ro_home.chmod(0o500)
    monkeypatch.setenv("HOME", str(ro_home))
    for var in ("HF_HOME", "XDG_CACHE_HOME", "UV_CACHE_DIR", "FLOCK_LLM_HOME"):
        monkeypatch.delenv(var, raising=False)
    work = tmp_path / "flock" / WID / "work"
    monkeypatch.setenv("FLOCK_URL", "http://ctl:7860")
    monkeypatch.setenv("FLOCK_TOKEN", "t")
    monkeypatch.setenv("WORKER_ID", WID)
    monkeypatch.setenv("SWARM", "sw")
    monkeypatch.setenv("TYPE", "researcher")
    monkeypatch.setenv("DEADLINE", "2026-10-25T00:00:00+00:00")
    monkeypatch.setenv("WORK_ROOT", str(work))
    started = {}

    def fake_run(coro, **kw):  # startup only: do not enter the loops
        started["worker"] = coro.cr_frame.f_locals.get("self")
        coro.close()

    monkeypatch.setattr(wmain, "run_and_exit", fake_run)
    try:
        wmain.main()
        assert started["worker"] is not None and work.is_dir()
        for var, sub in (("HF_HOME", "hf"), ("XDG_CACHE_HOME", "cache"), ("UV_CACHE_DIR", "uv")):
            assert os.environ[var] == str(work.parent / "cache" / sub)
        assert home() != ro_home and os.access(home(), os.W_OK)
        from flock.worker.adapters.llm import llm_home

        assert not str(llm_home()).startswith(str(ro_home))
        monkeypatch.delenv("HOME")
        assert os.access(home(), os.W_OK)

        folder = tmp_path / "tools"
        folder.mkdir()
        (folder / "t.py").write_text('def add(a: int, b: int) -> int:\n    """Adds."""\n'
                                     "    return a + b\n")
        m = load_dir(folder)
        r = asyncio.run(run_tool(m.get("add"), {"a": 2, "b": 3}, tools_dir=folder,
                                 workdir=tmp_path / "wd"))
        assert (r.output, r.error) == ("5", False)
        assert list(ro_home.iterdir()) == []
    finally:
        ro_home.chmod(0o700)
