"""The worker's HarborHost on fakes: 20 agents' Trials share one Resources, one tool bundle and
one task download, and on hf-sandbox their sandbox commands (threads, as Harbor's hf-sandbox
environment runs them) overlap instead of queueing on the default pool. Board agents join on
whichever worker their Trial starts."""

import asyncio
import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import legion.harbor
from legion import secrets_file
from legion.harbor import agent as legion_agent
from legion.harbor import launch, sources
from legion.harbor.spec import Spec, compile_spec
from legion.harbor.tools.bundle import build_tool_bundle
from legion.worker.adapters.harbor import make_adapter
from legion.worker.harbor import HarborHost

from .. import fake_msgboard
from ..worker_fakes import until
from . import test_gateway
from .test_adapter import TASK, FakeEnv, Fakes, Trial, read

BOARD = Path(legion.harbor.__file__).with_name("builtin") / "resources" / "message_board.py"
GATEWAY = BOARD.with_name("gateway.py")
Model, model = test_gateway.Model, test_gateway.model  # the fake model server and its fixture

SPEC = """
name: host
model: moonshotai/Kimi-K2.5
inference: {api_key_env: HF_INFERENCE_TOKEN}
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
    """Each `pi` call blocks a thread, as hf-sandbox's `sandbox.run`, until `want` calls run at
    once (or 10 s pass); records the overlap."""

    def __init__(self):
        self.lock, self.now, self.peak, self.want = threading.Lock(), 0, 0, 1
        self.full = threading.Event()

    def call(self):
        with self.lock:
            self.now += 1
            self.peak = max(self.peak, self.now)
            if self.now >= self.want:
                self.full.set()
        self.full.wait(10)
        with self.lock:
            self.now -= 1


class SandboxEnv(FakeEnv):
    async def exec(self, command, **kw):
        if " pi --print " in command:
            await asyncio.to_thread(self.fakes.sandboxes.call)
        return await super().exec(command, **kw)


@pytest.fixture
def world(monkeypatch, tmp_path):
    monkeypatch.setenv("HF_INFERENCE_TOKEN", "test-inference-secret")
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


def agents(tmp_path, *, mount=False, sha=None, resources=None):
    text = SPEC.replace("MOUNT", str(mount).lower())
    (tmp_path / "spec.yaml").write_text(
        text.replace("{fake: {mount: false}}", resources) if resources else text)
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    bundle = build_tool_bundle(spec.tools_dir, spec.tools, tmp_path / "launcher-tools.tar.gz")
    if sha:
        bundle = SimpleNamespace(manifest=bundle.manifest, sha256=sha)
    ts = compile_spec(spec, TASK, bundle).types["agent"]
    host = HarborHost(tmp_path / "work" / "_harbor" / "_worker")
    names = [f"agent-{i:04d}" for i in range(20)]  # opaque: the controller's would be words
    ts.harbor.roster = dict.fromkeys(names)  # as the controller fills it in each assign
    return host, [make_adapter(tmp_path / "work" / n, n, ts, host=host) for n in names]


async def test_twenty_agents_share_the_host_and_their_sandbox_commands_overlap(world, tmp_path):
    loop = asyncio.get_running_loop()
    loop.set_default_executor(ThreadPoolExecutor(4))  # what a 2-vCPU job gets by default
    host, adapters = agents(tmp_path)
    world.sandboxes.want = 20
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
        assert trial.config.environment.import_path == "legion.harbor.envs:HFSandbox"
    status = adapters[0].trial_status()
    assert status.pop("last_seen") and status == {"id": "agent-0000-t1", "attempt": 1, "alive": True}

    await asyncio.gather(*(a.close() for a in adapters))
    assert all(t.verified for t in world.trials) and not adapters[0].trial_status()["alive"]
    await host.close()
    assert fake.stops == 1


EXPORTING = '''
class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.metadata = {}

    async def start(self):
        pass

    def bind(self, name):
        return {}

    async def wait(self):
        import asyncio
        await asyncio.Event().wait()

    async def export(self, directory):
        (directory / "board.jsonl").write_text("{}\\n")
        return {"board": "board.jsonl"}

    async def stop(self):
        pass
'''


@pytest.mark.parametrize("steps", [False, True])
async def test_a_trials_grading_and_the_exports_outlive_the_worker(world, tmp_path, monkeypatch,
                                                                   steps):
    """LEGION_RESULTS_DIR (a Slurm worker's <run dir>/results, on the shared disk): each ended
    Trial's result and verifier files (a multi-step Trial's under steps/<name>/) and, at exit,
    the resources' exports are copied there, so a run is scored after the node's WORK_ROOT is
    gone."""
    world.steps = steps
    for key, value in (("LEGION_RESULTS_DIR", str(tmp_path / "results")), ("SWARM", "host"),
                       ("WORKER_ID", "host-agent-w0000")):
        monkeypatch.setenv(key, value)
    (tmp_path / "resources" / "exporting.py").write_text(EXPORTING)
    host, (a, *_) = agents(tmp_path, resources="\n  fake:\n    mount: false\n  exporting: {}")
    await a.start(a.workdir, None, a.spec)
    events = a.events()
    await a.send("go")
    await read(events, "turn_ended")
    await a.close()
    await host.close()
    shutil.rmtree(tmp_path / "work")  # the node's WORK_ROOT
    trial = tmp_path / "results" / "host" / "agent-0000" / "trials" / "host-agent-0000-t1"
    assert json.loads((trial / "result.json").read_text())["verifier_result"] == {
        "rewards": {"reward": 1.0}}
    verifier = trial / ("steps/check/verifier" if steps else "verifier")
    assert (verifier / "reward.txt").read_text() == "1"
    export = tmp_path / "results" / "host" / "host-agent-w0000" / "resources" / "exporting"
    assert (export / "board.jsonl").read_text() == "{}\n"


async def test_one_task_per_agent_gives_each_trial_only_its_own(world, tmp_path, monkeypatch):
    """A list selector with naming: consecutive-id (sources.pin_tasks): agent k runs the task
    the k-th entry matches, in the list's order, the worker downloads them at once, and each
    agent's Trial gets only its own task. A list that isn't one entry per agent is refused."""
    from harbor.models.registry import DatasetMetadata
    from harbor.models.task.id import GitTaskId

    async def metadata(self, name):  # the registry's order isn't the agents'
        return DatasetMetadata(name="tasks", version="1", task_ids=[GitTaskId(
            git_url="https://huggingface.co/datasets/org/tasks", git_commit_id="a" * 40,
            path=Path(f"tasks/team-s{k:02d}")) for k in (3, 1, 2)])

    downloads = []

    async def download_tasks(tasks, directory):
        downloads.append([t.task_id.get_name() for t in tasks])
        for t in tasks:
            t.local_path = directory / t.task_id.get_name()
            t.local_path.mkdir(parents=True)
        return [t.local_path for t in tasks]

    monkeypatch.setattr(sources.GitRepoRegistryClient, "get_dataset_metadata", metadata)
    monkeypatch.setattr(sources, "download_tasks", download_tasks)
    text = SPEC.replace("MOUNT", "false").replace("regex-log", "[team-s02, 'team-s0[3]', team-s01]")
    (tmp_path / "spec.yaml").write_text(text.replace(
        "agents: {count: 20}", "naming: consecutive-id\nagents:\n  count: 3").replace(
        "workers: {agents_per_worker: 20}", "workers:\n  agents_per_worker: 3"))
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    with pytest.raises(ValueError, match="no roles and agents.count: 3"):
        await sources.pin_tasks(spec.task, ["agent-001", "agent-002"])
    task = await sources.pin_tasks(spec.task, list(spec.agents.roster()))
    bundle = build_tool_bundle(spec.tools_dir, spec.tools, tmp_path / "launcher-tools.tar.gz")
    ts = compile_spec(spec, task, bundle).types["agent"]
    host = HarborHost(tmp_path / "work" / "_harbor" / "_worker")
    adapters = [make_adapter(tmp_path / "work" / n, n, ts, host=host) for n in ts.harbor.names()]
    await asyncio.gather(*(a.start(a.workdir, None, a.spec) for a in adapters))
    assert downloads == [["team-s02", "team-s03", "team-s01"]]
    assert len(world.trials) == 3 and [a.agent for a in adapters] == [
        "agent-001", "agent-002", "agent-003"]
    for a, slot in zip(adapters, ("02", "03", "01"), strict=True):  # the k-th entry's, only
        assert a.trial.config.task.path.name == f"team-s{slot}"
    await asyncio.gather(*(a.close() for a in adapters))
    await host.close()


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
    shutil.copy(BOARD, tmp_path / "resources" / "message_board.py")
    (tmp_path / "spec.yaml").write_text(SPEC.replace(
        "{fake: {mount: MOUNT}}", f"{{fake: {{mount: false}}, message_board: {{url: '{url}', "
        "channels: [legion-test], rules: {max_len: 2000}}}"))
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


GATED = '''
import asyncio

class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.agents, self.metadata = set(agents), {}
        self.adding, self.gate = asyncio.Event(), asyncio.Event()

    async def start(self):
        pass

    async def add(self, names):
        self.adding.set()
        await self.gate.wait()
        self.agents.update(names)

    async def remove(self, names):
        self.agents.difference_update(names)

    def bind(self, name):
        return {}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        return {}

    async def stop(self):
        pass
'''


async def test_a_close_while_the_agent_joins_leaves_it_out(world, tmp_path):
    """close() lands while the resources add the agent: its remove ran first, so the start
    removes the agent again and fails, and no Trial is created."""
    (tmp_path / "resources" / "gated.py").write_text(GATED)
    host, (a, *_) = agents(tmp_path, resources="{fake: {mount: false}, gated: {}}")
    starting = asyncio.create_task(a.start(a.workdir, None, a.spec))
    async with asyncio.timeout(5):
        while host.resources is None:
            await asyncio.sleep(0.01)
        gated = host.resources.providers["gated"]
        await gated.adding.wait()
    await a.close(0)
    gated.gate.set()
    with pytest.raises(RuntimeError, match="agent closed"):
        await starting
    assert gated.agents == set() and world.trials == []
    await host.close()


TOKENED = '''
import asyncio, os

class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.token, self.metadata = os.environ.get("FAKE_OPERATOR_TOKEN"), {}

    async def start(self):
        pass

    def bind(self, name):
        return {}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        return {}

    async def stop(self):
        pass
'''


async def test_secrets_from_the_workers_secrets_file_reach_the_trial(world, tmp_path, monkeypatch):
    """A Slurm worker has its secrets in a file, not os.environ (worker.sbatch): the upstream
    build_trial_config and a resource plugin, which read os.environ, still get them, and
    os.environ is without them again afterwards."""
    monkeypatch.delenv("HF_INFERENCE_TOKEN")
    monkeypatch.setattr(secrets_file, "_secrets", {})
    f = tmp_path / "secrets"
    f.write_text("HF_INFERENCE_TOKEN=key-from-file\nFAKE_OPERATOR_TOKEN=op-from-file\n")
    f.chmod(0o600)
    assert secrets_file.load({secrets_file.SECRETS_FILE_ENV: str(f)}) == [
        "FAKE_OPERATOR_TOKEN", "HF_INFERENCE_TOKEN"]
    (tmp_path / "resources" / "tokened.py").write_text(TOKENED)
    host, (a, *_) = agents(tmp_path, resources="{fake: {mount: false}, tokened: {}}")
    await a.start(a.workdir, None, a.spec)
    assert world.trials[0].config.agent.env["HF_INFERENCE_TOKEN"] == "key-from-file"
    assert host.resources.providers["tokened"].token == "op-from-file"
    assert not {"HF_INFERENCE_TOKEN", "FAKE_OPERATOR_TOKEN"} & set(os.environ)
    await a.close()
    await host.close()


class GatewayEnv(FakeEnv):
    """A Pi turn makes one streamed call as Pi would from its models.json, `$NAME` header values
    resolved from its env, without `Authorization` (the jobs proxy drops it)."""

    async def exec(self, command, **kw):
        if " pi --print " in command:
            trial = self.fakes.trials[-1]
            (provider,) = json.loads((trial.agent.logs_dir / "models.json").read_text())[
                "providers"].values()
            headers = {k: kw["env"][v[1:]] if v.startswith("$") else v
                       for k, v in provider["headers"].items()}
            ask = {"model": provider["models"][0]["id"], "stream": True, "messages": []}
            Model.go.set()
            async with httpx.AsyncClient() as client:
                r = await client.post(f"{provider['baseUrl']}/chat/completions", json=ask,
                                      headers=headers)
                r.raise_for_status()
        return await super().exec(command, **kw)


async def test_a_gateway_keeps_the_key_and_counts_each_call_once(world, model, tmp_path,
                                                                 monkeypatch):
    """On e2b with resources.gateway the Trial holds only its gateway token and URL; Pi's call
    is counted by the gateway (the host's emit), not again at the turn's end (metered)."""
    monkeypatch.setattr(launch.Trial, "create", lambda config: create(config))

    async def create(config):
        trial = Trial(config, world)
        trial.agent_environment = GatewayEnv(trial.agent_environment.logs_dir, world)
        world.trials.append(trial)
        return trial

    shutil.copy(GATEWAY, tmp_path / "resources" / "gateway.py")
    spec = (SPEC.replace("{fake: {mount: MOUNT}}", "{fake: {mount: false}, gateway: {port: 0}}")
            .replace("backend: hf-sandbox", "backend: e2b")
            .replace("inference: {api_key_env: HF_INFERENCE_TOKEN}",
                     "inference: {urls: ['https://models.example/v1'], "
                     "api_key_env: HF_INFERENCE_TOKEN}")
            + "compute: hf-jobs\nnamespace: ns\nbucket_id: ns/b\n")
    (tmp_path / "spec.yaml").write_text(spec)
    ts = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK,
                      build_tool_bundle(tmp_path / "tools", ["sample.*"],
                                        tmp_path / "launcher-tools.tar.gz")).types["agent"]
    ts.resolved_model.inference.urls = [f"{model}/v1"]  # the fake model server, on this machine
    monkeypatch.setattr(legion_agent, "PROBE", 0.02)  # an idle Trial probes its sandbox often
    usage = []
    host = HarborHost(tmp_path / "work" / "_harbor" / "_worker",
                      emit=lambda agent, event: usage.append((agent, event)))
    a = make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts, host=host)
    await a.start(a.workdir, None, a.spec)
    events = a.events()
    await a.send("go")
    assert "usage" not in [e.kind for e in await read(events, "turn_ended")]
    gateway = host.resources.providers["gateway"]
    agent = world.trials[0].config.agent
    assert agent.base_url == f"{gateway.url}/inference" and Model.seen[0]["headers"][
        "Authorization"] == "Bearer test-inference-secret"
    held = json.dumps([agent.env, world.pi_env, (a.trial.agent.logs_dir / "models.json")
                       .read_text()])
    assert "test-inference-secret" not in held
    assert usage == [("agent-0000", {"resource": "gateway", "kind": "usage", "data": {
        "input": 1000, "output": 100, "cache_read": 500, "cache_write": 0, "reasoning": 40}})]
    world.probe_ok = False  # the sandbox stops answering: the Trial ends, and so does its token
    await until(lambda: not a.alive())
    async with httpx.AsyncClient() as client:
        r = await client.get(f"{gateway.url}/world/manifest",
                             headers={"X-World-Token": agent.env[agent.model_api_key_name]})
    assert r.status_code == 401
    await a.close()
    await host.close()
