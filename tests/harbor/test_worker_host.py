"""The worker's HarborHost on fakes: 20 agents' Trials share one Resources, one tool bundle and
one task download, and on hf-sandbox their sandbox commands (threads, as Harbor's hf-sandbox
environment runs them) overlap instead of queueing on the default pool. Board agents join on
whichever worker their Trial starts."""

import asyncio
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

import flock.harbor
from flock.harbor import launch, sources
from flock.harbor.spec import Spec, compile_spec
from flock.harbor.tools.bundle import build_tool_bundle
from flock.worker.adapters.harbor import make_adapter
from flock.worker.harbor import HarborHost

from .. import fake_msgboard
from .test_adapter import TASK, FakeEnv, Fakes, Trial, read

BOARD = Path(flock.harbor.__file__).with_name("plugins") / "resources" / "board.py"

SPEC = """
name: host
model: huggingface/moonshotai/Kimi-K2.5
task:
  repo: https://huggingface.co/datasets/harborframework/terminal-bench-2.1
  selector: regex-log
  revision: 3e235dff6880252a587fa479c09bcb1e16edf2eb
resources_dir: resources
resources: {fake: {mount: MOUNT}}
tools_dir: tools
tools: [sample.*]
agents: {count: 20}
environment: {backend: hf-sandbox}
workers: {agents_per_worker: 20}
limits: {run_timeout_seconds: 900}
"""
RESOURCE = '''
import asyncio

class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options, self.agents, self.metadata = options, agents, {}
        self.starts = self.stops = 0

    async def start(self):
        self.starts += 1

    def bind(self, name):
        mounts = [{"type": "bind", "source": "/tmp/f.sock", "target": "/run/f.sock"}]
        return {"env": {"FAKE_TOKEN": "tok-" + name},
                "mounts": mounts if self.options["mount"] else []}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        return {}

    async def stop(self):
        self.stops += 1
'''
TOOL = '''
REQUIRES = ["fake"]

def add(x: int) -> dict:
    """Add one."""
    return {"sum": x + 1}

TOOLS = {"add": add}
'''


class Sandboxes:
    """Each `pi` call sleeps in a thread, as hf-sandbox's `sandbox.run`; records the overlap."""

    def __init__(self):
        self.lock, self.now, self.peak = threading.Lock(), 0, 0

    def call(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
        time.sleep(0.5)
        with self.lock:
            self.now -= 1


class SandboxEnv(FakeEnv):
    async def exec(self, command, **kw):
        if " pi --print " in command:
            await asyncio.to_thread(self.fakes.sandboxes.call)
        return await super().exec(command, **kw)


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    fakes = Fakes()
    fakes.sandboxes, fakes.downloads, fakes.tool_setups = Sandboxes(), [], []
    fakes.download_fails = False

    async def create(config):
        trial = Trial(config, fakes)
        trial.agent_environment = SandboxEnv(trial.agent_environment.logs_dir, fakes)
        fakes.trials.append(trial)
        return trial

    async def download(task, directory):
        fakes.downloads.append(task.task_id.git_commit_id)
        if fakes.download_fails:
            fakes.download_fails = False
            raise OSError("download failed")
        directory.mkdir(parents=True, exist_ok=True)
        task.local_path = directory
        return directory

    monkeypatch.setattr(launch.Trial, "create", create)
    monkeypatch.setattr(launch, "attach_tool_setup", lambda t, b: fakes.tool_setups.append(b))
    monkeypatch.setattr(sources, "download_task", download)
    for folder, name, text in [("resources", "fake", RESOURCE), ("tools", "sample", TOOL)]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / f"{name}.py").write_text(text)
    return fakes


def agents(tmp_path, *, mount=False, sha=None):
    (tmp_path / "spec.yaml").write_text(SPEC.replace("MOUNT", str(mount).lower()))
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    bundle = build_tool_bundle(spec.tools_dir, spec.tools, tmp_path / "launcher-tools.tar.gz")
    if sha:
        bundle = SimpleNamespace(manifest=bundle.manifest, sha256=sha)
    ts = compile_spec(spec, TASK, bundle).types["agent"]
    host = HarborHost(tmp_path / "work" / "_harbor" / "_worker")
    names = list(ts.harbor.roster)
    return host, [make_adapter(tmp_path / "work" / n, n, ts, host=host) for n in names]


async def test_twenty_agents_share_the_host_and_their_sandbox_commands_overlap(world, tmp_path):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(4))  # what a 2-vCPU job gets by default
    host, adapters = agents(tmp_path)
    await asyncio.gather(*(a.start(a.workdir, None, a.spec) for a in adapters))
    streams = [a.events() for a in adapters]
    for a in adapters:
        await a.send("go")
    for events in streams:
        assert (await read(events, "turn_ended"))[-1].status == "completed"

    assert world.sandboxes.peak == 20
    assert loop._default_executor._max_workers == 2 * 20 + 8
    assert len(world.downloads) == 1 and world.downloads[0] == "c" * 40
    fake = host.resources.providers["fake"]
    assert fake.starts == 1 and fake.agents == [a.agent for a in adapters]
    assert world.tool_setups == [host.bundle] * 20
    assert host.bundle.archive == host.root / "tools.tar.gz"
    for a, trial in zip(adapters, world.trials, strict=True):
        assert trial.config.agent.env["FAKE_TOKEN"] == f"tok-{a.agent}"
        assert trial.config.task.path == host.task_path
    assert adapters[0].trial_status() == {"id": "agent-0000-t1", "attempt": 1, "alive": True}

    await asyncio.gather(*(a.close() for a in adapters))
    assert all(t.verified for t in world.trials) and not adapters[0].trial_status()["alive"]
    await host.close()
    assert fake.stops == 1


async def test_a_failed_prepare_is_retried_by_the_next_trial(world, tmp_path):
    host, (a, *_) = agents(tmp_path)
    world.download_fails = True
    with pytest.raises(OSError, match="download failed"):
        await a.start(a.workdir, None, a.spec)
    assert world.trials == [] and host.resources is None
    await a.start(a.workdir, None, a.spec)
    assert len(world.downloads) == 2 and host.resources.providers["fake"].starts == 1
    await a.close()
    await host.close()


@pytest.mark.parametrize("case, error", [
    ("sha", "the tools built here .* are not the run's"),
    ("mount", "hf-sandbox has no mounts"),
])
async def test_refusals(world, tmp_path, case, error):
    host, (a, *_) = agents(tmp_path, mount=case == "mount", sha="0" * 64 if case == "sha" else None)
    with pytest.raises(RuntimeError, match=error):
        await a.start(a.workdir, None, a.spec)
    assert world.trials == []
    await host.close()


async def test_board_agents_join_on_any_worker_spawned_and_reassigned(world, tmp_path, monkeypatch):
    """Two workers of one type on one external board (fake), a spawned agent, and an agent
    re-assigned from the first worker to the second: each joins where its Trial starts (a
    re-join: same identity, a fresh token) and its mentions come from that worker's poller.
    hf-sandbox has no mounts, so the board's rules are not enforced: each Trial says so."""
    board, server, url = fake_msgboard.serve()
    monkeypatch.setenv("MSGBOARD_OPERATOR_TOKEN", board.token_for("operator", "user"))
    shutil.copy(BOARD, tmp_path / "resources" / "board.py")
    (tmp_path / "spec.yaml").write_text(SPEC.replace(
        "{fake: {mount: MOUNT}}", f"{{fake: {{mount: false}}, board: {{url: '{url}', "
        "channels: [flock-test], rules: {max_len: 2000}}}"))
    ts = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK, None).types["agent"]
    events, hosts = [], []
    for w in ("w1", "w2"):
        hosts.append(HarborHost(tmp_path / w / "_harbor" / "_worker",
                                emit=lambda agent, e, w=w: events.append((w, agent, e["kind"]))))

    def adapter(host, name):
        return make_adapter(host.root.parent.parent / name, name, ts, host=host)

    def token(trial):
        return trial.config.agent.env["MSGBOARD_TOKEN"]

    def kinds(kind):
        return sorted((w, agent) for w, agent, k in events if k == kind)

    async def until(check):
        async with asyncio.timeout(5):
            while not check():
                await asyncio.sleep(0.01)

    first, second, spawned, moved = (
        adapter(hosts[0], "agent-0000"), adapter(hosts[1], "agent-0001"),
        adapter(hosts[1], "helper-0001"),  # spawned: not in the roster
        adapter(hosts[1], "agent-0000"))  # `first`, re-assigned to w2
    try:
        for a in (first, second, spawned):
            await a.start(a.workdir, None, a.spec)
        assert all(token(t) in board.tokens for t in world.trials)
        assert sorted(board.names()) == ["agent-0000", "agent-0001", "helper-0001", "operator"]
        await first.close()
        await moved.start(moved.workdir, None, moved.spec)
        old, new = token(world.trials[0]), token(world.trials[-1])
        assert old not in board.tokens and board.tokens[new] == board.principal("agent-0000")["id"]
        await until(lambda: sum(p == "/v1/activity" and "after" in q
                                for _, p, q, _ in board.requests) >= 2)  # both pollers wait
        board.post("carol", 12, "@agent-0000 @helper-0001 review please")
        await until(lambda: kinds("mention") == [("w2", "agent-0000"), ("w2", "helper-0001")])
        await asyncio.sleep(0.05)  # and nothing from w1
        assert kinds("mention") == [("w2", "agent-0000"), ("w2", "helper-0001")]
        assert kinds("warning") == [("w1", "agent-0000"), ("w2", "agent-0000"),
                                    ("w2", "agent-0001"), ("w2", "helper-0001")]
    finally:
        for a in (second, spawned, moved):
            await a.close()
        for host in hosts:
            await host.close()
        server.shutdown()
