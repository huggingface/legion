"""HarborAdapter + LegionPi on fakes: a FakeTrial (test_launch's) that runs the real LegionPi
against a FakeEnv (after harbor-core's test_agent.py) whose `pi --print` writes a canned pi.txt.
Container networks are faked at their command boundary (legion.harbor.networking.command)."""

import asyncio
import json
import math
import shlex
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.result import ExceptionInfo
from harbor.models.verifier.result import VerifierResult

from legion.harbor import agent as legion_agent
from legion.harbor import launch, networking
from legion.harbor.agent import LegionPi
from legion.harbor.spec import Spec, compile_spec
from legion.models import Inference
from legion.worker.adapters.base import ResumeError, transient_error
from legion.worker.adapters.harbor import HarborAdapter, make_adapter

from .test_envs import compose
from .test_launch import FakeTrial

EXAMPLES = Path(__file__).parent / "examples"
TASK = {
    "registry_commit": "d" * 40,
    "dataset": "terminal-bench-2.1@2.1.0",
    "task": {"git_url": "https://example.com/tasks.git", "git_commit_id": "c" * 40, "path": "t"},
}
SESSIONS = "/logs/agent/pi/sessions"


class Fakes:
    """What every fake Trial and environment of a test shares, and what they record."""

    def __init__(self):
        self.trials, self.pi, self.root, self.uploads, self.net = [], [], [], [], []
        self.reward = 1.0
        self.success_response = None
        self.success_gate = None
        self.success_calls = []
        self.pi_env = []  # each `pi --print` exec's env
        self.probe_ok, self.upload_fails, self.kill_fails = True, False, False
        self.fail = None  # pi call number -> "exit" | "provider"
        self.hold = set()  # pi call numbers that wait for release(n) or a kill
        self.gates = {}
        self.mounted = True  # False: /logs/agent lives in the env (hf-sandbox), not on the host
        self.start_gate = self.verify_gate = None  # set: Trial.run waits for them
        self.route_gate = None  # set: `ip route` (a Trial network's first command) waits for it
        self.steps = False  # True: the verifier's files land as a multi-step Trial's

    def gate(self, n):
        return self.gates.setdefault(n, asyncio.Event())

    def release(self, n):
        self.gate(n).set()


class FakeEnv:
    def __init__(self, logs_dir, fakes):
        self.logs_dir, self.fakes, self.default_user = logs_dir, fakes, None
        self.killed = False

    async def service_exec(self, command, *, service, timeout_sec):
        # Compose mixes a banner from stderr into stdout; the command frames its JSON body.
        from legion.worker.adapters.harbor import CHECK_END, CHECK_START

        parts = shlex.split(command)
        original = parts[parts.index("-c") + 1]
        self.fakes.success_calls.append((original, service, timeout_sec))
        if self.fakes.success_gate is not None:
            await self.fakes.success_gate.wait()
        if original == "after-first-turn":
            from ..worker_fakes import until

            await until(lambda: bool(self.fakes.pi))
            original = '{"success": true}'
        return ExecResult(return_code=0, stdout="compose provider banner" + CHECK_START
                          + (self.fakes.success_response or original) + CHECK_END)

    async def upload_file(self, source, target):
        pass

    async def upload_dir(self, source, target):
        if self.fakes.upload_fails:
            raise OSError("upload failed")
        self.fakes.uploads.append((Path(source), target))
        local = self.logs_dir / "pi" / "sessions"  # /logs/agent is mounted at logs_dir
        local.mkdir(parents=True, exist_ok=True)
        for path in Path(source).glob("*.jsonl"):
            (local / path.name).write_text(path.read_text())

    def is_path_mounted(self, environment_path, host_path):
        return self.fakes.mounted

    def _remote(self, path):
        return self.logs_dir / Path(path).relative_to("/logs/agent")

    async def download_file(self, source, target):
        shutil.copy2(self._remote(source), target)

    async def download_dir(self, source, target):
        shutil.copytree(self._remote(source), target, dirs_exist_ok=True)

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        f = self.fakes
        if command == "true":
            return ExecResult(return_code=0 if f.probe_ok else 1)
        if user == "root":
            f.root.append(command)
        if command.endswith(legion_agent.KILL_PI):
            if f.kill_fails:
                return ExecResult(return_code=1)
            self.killed = True
            for gate in f.gates.values():
                gate.set()
            return ExecResult(return_code=0)
        if " pi --print " not in command:
            return ExecResult(return_code=0)
        f.pi.append(command)
        f.pi_env.append(env)
        n = len(f.pi)
        words = shlex.split(command)
        prompt = words[words.index("2>&1") - 1]
        if n in f.hold or prompt.endswith("[hold]"):  # [hold]: a tiny run's long turn
            await f.gate(n).wait()
        if self.killed:
            self.killed = False
            return ExecResult(return_code=143, stderr="Terminated")
        failure = f.fail and f.fail(n)
        if failure == "exit":
            return ExecResult(return_code=1, stderr="boom")
        message = {
            "role": "assistant",
            "content": [{"type": "text", "text": f"reply-{n}"}],
            "usage": {"input": 1000, "cacheRead": 500, "output": 100},
        }
        if failure == "provider":
            message.update(content=[], errorMessage="429 Too Many Requests")
        if failure == "server":  # a 500 before any token
            message.update(content=[], errorMessage="500 Internal Server Error", usage={})
        if failure == "rejected":  # the model refuses the request (e.g. the history)
            message.update(content=[], errorMessage="400 Bad Request", usage={})
        (self.logs_dir / "pi.txt").write_text(
            json.dumps({"type": "message_end", "message": message}) + "\n"
        )
        sessions = self.logs_dir / "pi" / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        with (sessions / "s.jsonl").open("a") as out:
            # the prompt too: the tiny runs' "what the model was told" (tests/tinyrun.py)
            out.write(json.dumps({"call": n, "prompt": prompt}) + "\n")
        return ExecResult(return_code=0)


class Trial(FakeTrial):
    """A Trial's life as Harbor runs it: hooks, the agent's run(), then the verifier."""

    def __init__(self, config, fakes):
        super().__init__(config, [])
        self.logs_dir = logs_dir = config.trials_dir / config.trial_name / "agent"
        logs_dir.mkdir(parents=True, exist_ok=True)  # as Harbor: a new worker reuses t1
        self.agent = AgentFactory.create_agent_from_config(config.agent, logs_dir=logs_dir)
        remote = logs_dir if fakes.mounted else logs_dir.parent / "remote"
        remote.mkdir(exist_ok=True)
        self.agent_environment = FakeEnv(remote, fakes)
        self.fakes, self.verified, self.cancelled = fakes, False, False

    async def run(self):
        try:
            return await self._life()
        except asyncio.CancelledError:
            if not self.fakes.mounted:  # Harbor downloads /logs/agent, stops the environment
                shutil.copytree(self.agent_environment.logs_dir, self.logs_dir, dirs_exist_ok=True)
            await asyncio.sleep(0.05)
            self.cancelled = True
            raise

    async def _life(self):
        if self.fakes.start_gate:
            await self.fakes.start_gate.wait()
        for hook in self.hooks.values():
            await hook(None)
        instruction = "TASK\n\n" + self.config.extra_instructions[0]
        try:
            await self.agent.run(instruction, self.agent_environment, AgentContext())
        except Exception as exc:  # noqa: BLE001 - Harbor records any agent failure
            self.result.exception_info = ExceptionInfo.from_exception(exc)
            return self.result
        finally:
            if not self.fakes.mounted:  # Harbor downloads /logs/agent after the agent phase
                remote = self.agent_environment.logs_dir
                shutil.copytree(remote, self.logs_dir, dirs_exist_ok=True)
        if self.fakes.verify_gate:
            await self.fakes.verify_gate.wait()
        self.verified = True
        self.result.verifier_result = VerifierResult(rewards={"reward": self.fakes.reward})
        trial_dir = self.config.trials_dir / self.config.trial_name  # as Harbor leaves it
        verifier = trial_dir / ("steps/check/verifier" if self.fakes.steps else "verifier")
        verifier.mkdir(parents=True, exist_ok=True)
        (verifier / "reward.txt").write_text("1")
        (trial_dir / "result.json").write_text(self.result.model_dump_json())
        return self.result


def install(setattr) -> Fakes:
    """Harbor's Trial.create and the container-network commands faked through `setattr` (the
    `fakes` fixture; tests/harbor/fake_worker.py in a worker process)."""
    shared = Fakes()

    async def create(config):
        trial = Trial(config, shared)
        shared.trials.append(trial)
        return trial

    async def command(*args):  # docker or podman, and `ip route`
        shared.net.append(args)
        if args[0] == "ip" and shared.route_gate:
            await shared.route_gate.wait()
        if args[0] == "ip":
            return 0, "[]", ""
        if args[2] == "inspect":  # legion-<run id>-<participant>
            return 0, json.dumps([{"Labels": {"io.legion.run": args[3].split("-")[1]}}]), ""
        return 0, "", ""

    setattr(launch.Trial, "create", create)
    setattr(networking, "command", command)
    return shared


@pytest.fixture
def fakes(monkeypatch):
    for name in ("HF_TOKEN", "HF_INFERENCE_TOKEN"):
        monkeypatch.setenv(name, "test-inference-secret")
    return install(monkeypatch.setattr)


@pytest.fixture
def clock(monkeypatch):
    """LegionPi's monotonic clock, moved by hand; an idle Trial looks at it every 20 ms."""
    now = [0.0]
    monkeypatch.setattr(legion_agent, "time", SimpleNamespace(monotonic=lambda: now[0]))
    monkeypatch.setattr(legion_agent, "PROBE", 0.02)
    return now


def type_spec(backend="docker", **limits):
    spec = Spec.load(EXAMPLES / "pi-hf.yaml", env={})
    spec.environment.backend = backend
    if backend in ("hf-sandbox", "e2b"):  # never the worker's Hub token
        spec.inference.api_key_env = "HF_INFERENCE_TOKEN"
    for key, value in limits.items():
        setattr(spec.limits, key, value)
    return compile_spec(spec, TASK).types["agent"]


def adapter(tmp_path, **kw) -> HarborAdapter:
    return make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", type_spec(**kw))


async def read(events, until, count=1):
    """Events up to the count-th one of kind `until`."""
    out = []
    while count:
        ev = await asyncio.wait_for(anext(events), 5)
        out.append(ev)
        count -= ev.kind == until
    return out


async def start(a, session_id=None):
    return await a.start(a.workdir, session_id, a.spec), a.events()


async def test_a_trial_is_named_for_its_swarm(fakes, tmp_path, monkeypatch):
    """Harbor's compose project is `<trial_name>__env` and its start takes down whatever runs
    under that name: another swarm's agent-0000 on this host must not share it."""
    from harbor.environments.docker.docker import _sanitize_docker_compose_project_name

    monkeypatch.setenv("SWARM", "crew-2")  # the worker's env
    a = adapter(tmp_path)
    _, events = await start(a)
    await a.send("Go.")
    assert (await read(events, "turn_ended"))[0].data == {"trial": "crew-2-agent-0000-t1"}
    project = f"{fakes.trials[0].config.trial_name}__env"
    assert _sanitize_docker_compose_project_name(project) == project  # a valid name as it is
    await a.stop()


async def test_one_trial_across_bursts(fakes, tmp_path):
    a = adapter(tmp_path)
    sid, events = await start(a)
    await a.send("First prompt.")
    evs = await read(events, "turn_ended")
    assert [e.kind for e in evs] == ["turn_started", "message", "usage", "turn_ended"]
    assert evs[0].data == {"trial": "agent-0000-t1"} and evs[-1].status == "completed"
    usage = evs[2]
    assert usage.model == "moonshotai/Kimi-K2.5"  # priced under the model's Hub id
    assert (usage.usage.input, usage.usage.cache_read, usage.usage.output) == (1000, 500, 100)
    assert evs[1].message["content"][0]["text"] == "reply-1"
    await a.stop()
    assert (await read(events, "exited"))[-1].kind == "exited"
    assert a.alive() and isinstance(fakes.trials[0].agent, LegionPi)

    assert (await start(a, sid))[0] == sid
    events = a.events()
    await a.send("Second.")
    assert (await read(events, "turn_ended"))[-1].status == "completed"
    assert len(fakes.trials) == 1
    first, second = fakes.pi
    assert "--continue" not in first and "TASK" in first and "First prompt." in first
    assert "You are agent-0000, working on your own isolated copy of this task." in first
    assert "--continue" in second and "TASK" not in second and "Second." in second
    (session,) = a.session_files(sid)
    assert session == a.home / "sessions" / "s.jsonl" and a.session_dir == a.root  # purge()
    assert len(session.read_text().splitlines()) == 2


def pi_models(fakes, n=0):
    """The models.json the n-th Trial's Pi was last given (LegionPi's copy)."""
    (provider,) = json.loads((fakes.trials[n].agent.logs_dir / "models.json").read_text())[
        "providers"].values()
    return provider


def endpoint_spec(inference, backend="docker", window=None, namespace=""):
    spec = Spec.load(EXAMPLES / "pi-hf.yaml", env={})
    spec.environment.backend, spec.inference = backend, Inference.model_validate(inference)
    spec.namespace = namespace
    ts = compile_spec(spec, TASK).types["agent"]
    ts.resolved_model.context_window = window  # as the controller's apply pins it
    return ts


@pytest.mark.parametrize("namespace", ["", "huggingface"])
async def test_pi_gets_the_router_provider_bill_to_and_context_window(fakes, tmp_path, namespace):
    ts = endpoint_spec({"hf_router": {"provider": "deepinfra"}},
                       window=262144, namespace=namespace)
    a = make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts)
    _, events = await start(a)
    await a.send("go")
    await read(events, "turn_ended")
    provider = pi_models(fakes)
    assert provider["baseUrl"] == "https://router.huggingface.co/v1"
    assert provider.get("headers", {}) == ({"X-HF-Bill-To": namespace} if namespace else {})
    assert provider["models"] == [{"id": "moonshotai/Kimi-K2.5:deepinfra", "contextWindow": 262144}]
    assert "--model moonshotai/Kimi-K2.5:deepinfra" in fakes.pi[0]
    await a.close()


async def test_pi_reaches_this_machine_through_podman(fakes, tmp_path):
    """`localhost` is the host as podman containers reach it, allowlisted; a server you run takes
    no key (Pi gets a placeholder). A running Trial keeps its endpoint and allowlist: a changed
    block restarts the type's workers instead (the controller)."""
    ts = endpoint_spec({"urls": ["http://localhost:8000/v1"]}, "podman", 32768)
    a = make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts)
    sid, events = await start(a)
    await a.send("go")
    await read(events, "turn_ended")
    agent = fakes.trials[0].config.agent
    assert agent.base_url == pi_models(fakes)["baseUrl"] == "http://host.containers.internal:8000/v1"
    assert agent.extra_allowed_hosts == ["host.containers.internal"]
    assert agent.env["LEGION_INFERENCE_KEY"] == "none" and agent.model_api_key_name == "LEGION_INFERENCE_KEY"
    assert pi_models(fakes)["models"] == [{"id": "moonshotai/Kimi-K2.5", "contextWindow": 32768}]
    await a.stop()
    await read(events, "exited")

    moved = endpoint_spec({"urls": ["http://10.53.0.9:8000/v1"]}, "podman", 32768)
    assert await a.start(a.workdir, sid, moved) == sid
    await a.send("again")
    await read(a.events(), "turn_ended")
    assert len(fakes.trials) == 1
    assert pi_models(fakes)["baseUrl"] == "http://host.containers.internal:8000/v1"
    await a.close()


async def test_an_agent_keeps_its_server_across_trials_and_workers(fakes, tmp_path, monkeypatch):
    """The server is picked from the swarm and the agent's name, so the agent's Trial on the
    worker it is restored on after a restart calls the server of its first one (whose prefix
    cache holds its context), the one its heartbeat names."""
    from legion.worker.agents import AgentRuntime
    from legion.worker.events import Outbox

    monkeypatch.setenv("SWARM", "s3-maze-07")  # the workers' env
    ts = endpoint_spec({"urls": [f"http://10.0.0.{i}:8000/v1" for i in range(1, 15)]})
    old, new = (make_adapter(tmp_path / w / "agent-001", "agent-001", ts) for w in ("w1", "w2"))
    sid, events = await start(old)
    await old.send("go")
    await read(events, "turn_ended")
    await old.close()
    shutil.copytree(old.home / "sessions", new.home / "sessions")  # the bucket restore
    _, events = await start(new, sid)
    await new.send("go on")
    await read(events, "turn_ended")
    await new.close()
    first, second = (t.config.agent.base_url for t in fakes.trials)
    assert first == second == pi_models(fakes, 1)["baseUrl"]
    rt = AgentRuntime("agent-001", "agent", ts, swarm="s3-maze-07", work_root=tmp_path / "w2",
                      adapter=new, outbox=Outbox(), secret_source={})
    assert rt.heartbeat().endpoint == first.removeprefix("http://").removesuffix("/v1")


async def test_prompts_during_a_turn_join_the_next_turn(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.hold = {1}
    await a.send("a")
    await read(events, "turn_started")
    await a.send("b")
    await a.send("c")
    fakes.release(1)
    evs = await read(events, "turn_ended", 2)
    assert [e.status for e in evs if e.kind == "turn_ended"] == ["completed", "completed"]
    assert len(fakes.pi) == 2 and shlex.quote("b\n\nc") in fakes.pi[1]


async def test_stop_when_idle_keeps_the_trial_and_close_runs_the_verifier(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    await a.send("a")
    await read(events, "turn_ended")
    await a.stop()
    await a.stop()  # idempotent: one `exited`
    trial = fakes.trials[0]
    assert a.alive() and not trial.verified
    result = await a.close()
    assert trial.verified and result.verifier_result.rewards == {"reward": 1.0}
    assert not a.alive() and trial.agent.why == "closed"
    assert [e.kind for e in await read(events, "exited")] == ["exited"]


async def test_stop_kills_a_turn_in_flight(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.hold = {1}
    await a.send("a")
    await read(events, "turn_started")
    await a.send("queued while stopping")
    await a.stop()
    evs = await read(events, "exited")
    assert [(e.kind, e.status) for e in evs] == [("turn_ended", "interrupted"), ("exited", None)]
    assert any(c.endswith(legion_agent.KILL_PI) for c in fakes.root)
    assert a.alive()
    _, events = await start(a)  # same Trial; the queued prompt was dropped
    await a.send("b")
    await read(events, "turn_ended")
    assert len(fakes.trials) == 1 and len(fakes.pi) == 2 and "queued" not in fakes.pi[1]
    assert "TASK" in fakes.pi[1]  # the killed first turn used no tokens: the task goes again


async def test_a_new_trial_gets_the_saved_sessions(fakes, tmp_path):
    a = adapter(tmp_path)
    (a.home / "sessions").mkdir(parents=True)
    (a.home / "sessions" / "s.jsonl").write_text('{"call": 0}\n')
    sid, events = await start(a, "s-1")
    assert sid == "s-1" and fakes.uploads == [(a.home / "sessions", SESSIONS)]
    await a.send("go on")
    await read(events, "turn_ended")
    assert "--continue" in fakes.pi[0] and "TASK" not in fakes.pi[0]
    assert len((a.home / "sessions" / "s.jsonl").read_text().splitlines()) == 2


async def test_resume_errors(fakes, tmp_path):
    a = adapter(tmp_path)
    with pytest.raises(ResumeError, match="no Pi session files"):
        await a.start(a.workdir, "gone", a.spec)
    assert fakes.trials == []
    (a.home / "sessions").mkdir(parents=True)
    (a.home / "sessions" / "s.jsonl").write_text("{}\n")
    fakes.upload_fails = True
    with pytest.raises(ResumeError, match="upload failed"):
        await a.start(a.workdir, "s-1", a.spec)
    sid, _ = await start(a)  # the runtime's fallback: a fresh session, nothing put back
    assert sid not in ("gone", "s-1") and len(fakes.trials) == 2 and a.alive()


async def test_a_failed_probe_reports_and_the_next_start_takes_a_new_trial(
    fakes, tmp_path, monkeypatch
):
    monkeypatch.setattr(legion_agent, "PROBE", 0.05)
    a = adapter(tmp_path)
    sid, events = await start(a)
    await a.send("a")
    await read(events, "turn_ended")
    fakes.probe_ok = False
    evs = await read(events, "exited")
    assert [e.kind for e in evs] == ["error", "exited"] and "idle probe failed" in evs[0].error
    fakes.probe_ok = True
    await a.stop()  # the burst's end; no second `exited`
    _, events = await start(a, sid)
    assert [t.config.trial_name for t in fakes.trials] == ["agent-0000-t1", "agent-0000-t2"]
    assert not fakes.trials[0].verified and fakes.uploads == [(a.home / "sessions", SESSIONS)]
    await a.send("b")
    await read(events, "turn_ended")
    assert "--continue" in fakes.pi[-1]


@pytest.mark.parametrize("failure, again", [("exit", True), ("provider", False)])
async def test_a_failed_first_turn_sends_the_task_again_only_without_tokens(
    fakes, tmp_path, failure, again
):
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.fail = lambda n: failure if n == 1 else None
    await a.send("a")
    ended = (await read(events, "turn_ended"))[-1]
    assert ended.status == "error"
    assert ended.data == ({} if failure == "exit" else {"reason": "model_transport"})
    await a.send("b")
    await read(events, "turn_ended")
    assert ("TASK" in fakes.pi[1]) is again and ("--continue" in fakes.pi[1]) is not again


async def test_three_failed_turns_end_the_trial(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.fail = lambda n: "exit" if n <= 3 else None
    for prompt in "abc":
        await a.send(prompt)
        await read(events, "turn_ended")
    evs = await read(events, "exited")
    assert [e.kind for e in evs] == ["error", "exited"] and "3 failed turns" in evs[0].error
    await asyncio.wait_for(asyncio.shield(a._run), 5)
    assert fakes.trials[0].verified  # LegionPi returned: Harbor ran the verifier
    _, events = await start(a, None)
    await a.send("d")
    assert (await read(events, "turn_ended"))[-1].status == "completed"
    assert len(fakes.trials) == 2


async def test_trial_hours_roll_over_to_a_new_trial(fakes, tmp_path, clock):
    a = adapter(tmp_path, trial_hours=1)
    _, events = await start(a)
    for prompt in ("a", "b"):
        clock[0] += 3600  # the Trial's time is up: it ends idle, before any prompt
        await asyncio.wait_for(asyncio.shield(a._run), 5)
        await a.send(prompt)  # starts the next Trial
        evs = await read(events, "turn_ended")
        assert [e.kind for e in evs] == ["turn_started", "message", "usage", "turn_ended"]
    assert [t.config.trial_name for t in fakes.trials] == [
        "agent-0000-t1", "agent-0000-t2", "agent-0000-t3",
    ]
    assert all(t.verified for t in fakes.trials[:2])
    assert "TASK" in fakes.pi[0] and "--continue" in fakes.pi[1]
    assert fakes.uploads == [(a.home / "sessions", SESSIONS)]  # t3 got t2's session


async def test_the_worker_runtime_drives_one_trial_across_bursts(fakes, tmp_path):
    from legion.worker.agents import AgentRuntime
    from legion.worker.events import Outbox

    from ..worker_fakes import event_types, until

    a = adapter(tmp_path)
    spec = a.spec.model_copy(update={"idle_grace": "200ms"})
    rt = AgentRuntime("agent-0000", "agent", spec, swarm="sw", work_root=a.workdir.parent,
                      adapter=a, outbox=Outbox())
    await rt.setup(restore=False)
    for n, prompt in enumerate(("one", "two"), 1):
        await rt.prompt(prompt)
        await until(lambda n=n: rt.state.turns == n and rt.phase == "none")
    types = event_types(rt)
    assert (types.count("session_started"), types.count("session_resumed")) == (1, 1)
    assert types.count("turn_ended") == 2 and "error" not in types
    assert len(fakes.trials) == 1 and a.alive() and "--continue" in fakes.pi[1]
    assert rt.state.tokens["moonshotai/Kimi-K2.5"].input == 2000
    assert rt.state.last_summary == "reply-2"


def lines(a):
    return len((a.home / "sessions" / "s.jsonl").read_text().splitlines())


async def test_unmounted_sessions_reach_home_at_rollover_and_close(fakes, tmp_path, clock):
    fakes.mounted = False  # hf-sandbox: LegionPi downloads the sessions every SYNC s only
    a = adapter(tmp_path, trial_hours=1)
    _, events = await start(a)
    for prompt in ("a", "b"):
        await a.send(prompt)
        await read(events, "turn_ended")
    assert lines(a) == 1  # the second turn's session is not downloaded yet
    clock[0] += 3600
    await asyncio.wait_for(asyncio.shield(a._run), 5)
    assert lines(a) == 2  # Harbor's download at the end, kept before the next Trial starts
    for prompt in ("c", "d"):
        await a.send(prompt)
        await read(events, "turn_ended")
    assert lines(a) == 3 and fakes.uploads == [(a.home / "sessions", SESSIONS)]
    await a.close()
    assert lines(a) == 4 and len(fakes.trials) == 2  # what BucketSync pushes after a drain


@pytest.mark.parametrize("message, transient", [
    ("Connection error.", True),
    ("connect ECONNREFUSED 10.0.0.5:8000", True),
    ("other side closed", True),
    ("504 Gateway Time-out", True),
    ("429 Too Many Requests", True),
    ("400 Bad Request: invalid tool schema", False),
    ("400 tool output: curl: (7) Failed to connect: Connection refused", False),  # quoted agent text
    ("Error: Unterminated string in JSON at position 12", False),
])
def test_model_server_failures_are_transient(message, transient):
    assert transient_error(message) is transient


async def test_rate_limited_turns_do_not_end_the_trial(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.fail = lambda n: "provider" if n <= 3 else None
    ended = []
    for prompt in "abcd":
        await a.send(prompt)
        ended.append((await read(events, "turn_ended"))[-1])
    assert [e.data.get("reason") for e in ended] == ["model_transport"] * 3 + [None]
    assert ended[-1].status == "completed" and len(fakes.trials) == 1 and a.alive()


async def test_a_prompt_sent_while_the_verifier_runs_ends_as_a_failed_turn(fakes, tmp_path):
    from ..worker_fakes import until

    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.fail, fakes.verify_gate = (lambda n: "exit"), asyncio.Event()
    for prompt in "abc":
        await a.send(prompt)
        await read(events, "turn_ended")
    await until(lambda: a.trial.agent.why is not None)  # LegionPi returned; Harbor verifies
    await a.send("late")
    fakes.verify_gate.set()
    evs = await read(events, "exited")
    assert [(e.kind, e.status) for e in evs] == [
        ("turn_started", None), ("turn_ended", "error"), ("error", None), ("exited", None),
    ]
    assert "3 failed turns" in evs[1].error and len(fakes.pi) == 3


async def test_an_expired_trial_leaves_queued_prompts_to_the_next(fakes, tmp_path, clock):
    a = adapter(tmp_path, trial_hours=1)
    _, events = await start(a)
    fakes.hold = {1}
    await a.send("first")
    await read(events, "turn_started")
    await a.send("queued-during-the-turn")
    clock[0] += 3600
    fakes.release(1)
    evs = await read(events, "turn_ended", 2)
    assert [e.data for e in evs if e.kind == "turn_started"] == [{"trial": "agent-0000-t2"}]
    assert "queued-during-the-turn" in fakes.pi[1] and fakes.trials[0].verified


async def test_a_failed_continue_restarts_the_session(fakes, tmp_path):
    a = adapter(tmp_path)
    (a.home / "sessions").mkdir(parents=True)
    (a.home / "sessions" / "s.jsonl").write_text('{"call": 0}\n')
    _, events = await start(a, "s-1")
    fakes.fail = lambda n: "exit" if n == 1 else None
    await a.send("go on")
    evs = await read(events, "exited")
    assert [e.kind for e in evs] == ["turn_started", "turn_ended", "error", "exited"]
    assert "resume failed" in evs[2].error
    with pytest.raises(ResumeError, match="--continue failed"):
        await a.start(a.workdir, "s-1", a.spec)
    sid, events = await start(a)  # the runtime's session_restarted: fresh, nothing put back
    assert sid != "s-1"
    await a.send("again")
    assert (await read(events, "turn_ended"))[-1].status == "completed"
    assert len(fakes.trials) == 2 and len(fakes.uploads) == 1 and "TASK" in fakes.pi[1]


async def test_a_model_error_on_the_first_resumed_turn_keeps_the_session(fakes, tmp_path):
    a = adapter(tmp_path)
    (a.home / "sessions").mkdir(parents=True)
    (a.home / "sessions" / "s.jsonl").write_text('{"call": 0}\n')
    _, events = await start(a, "s-1")
    fakes.fail = lambda n: "server" if n == 1 else None
    await a.send("go on")
    assert (await read(events, "turn_ended"))[-1].status == "error"
    assert a.alive() and a.trial.agent.channel.restore_error is None
    await a.stop()
    await read(events, "exited")
    assert await a.start(a.workdir, "s-1", a.spec) == "s-1"  # no ResumeError
    await a.send("again")
    assert (await read(a.events(), "turn_ended"))[-1].status == "completed"
    assert len(fakes.trials) == 1 and "--continue" in fakes.pi[1]


@pytest.mark.parametrize("failure", ["rejected", "server"])
async def test_three_failed_turns_on_a_restored_session(fakes, tmp_path, failure):
    """Three 400s on the restored session: the next start begins a fresh one. Three 500s (an
    outage) say nothing about the history: the next start restores it again."""
    a = adapter(tmp_path)
    (a.home / "sessions").mkdir(parents=True)
    (a.home / "sessions" / "s.jsonl").write_text('{"call": 0}\n')
    _, events = await start(a, "s-1")
    fakes.fail = lambda n: failure  # every turn on the restored session fails
    for prompt in "abc":
        await a.send(prompt)
        await read(events, "turn_ended")
    assert [e.kind for e in await read(events, "exited")] == ["error", "exited"]
    if failure == "rejected":
        with pytest.raises(ResumeError, match="3 failed turns in a row on the restored sessions"):
            await a.start(a.workdir, "s-1", a.spec)
        return
    assert await a.start(a.workdir, "s-1", a.spec) == "s-1"
    assert len(fakes.trials) == 2 and a.trial.agent.channel.restore
    await a.close()


async def test_stop_pulls_unmounted_sessions_for_the_next_push(fakes, tmp_path):
    fakes.mounted = False
    a = adapter(tmp_path)
    _, events = await start(a)
    for prompt in ("a", "b"):
        await a.send(prompt)
        await read(events, "turn_ended")
    assert lines(a) == 1  # the second turn's session is not downloaded yet (SYNC)
    await a.stop()
    assert lines(a) == 2 and a.alive()


async def test_keep_sessions_copies_changed_files_only(fakes, tmp_path):
    a = adapter(tmp_path)
    _, events = await start(a)
    await a.send("a")
    await read(events, "turn_ended")
    kept = a.home / "sessions" / "s.jsonl"
    inode = kept.stat().st_ino
    a.trial.agent.keep_sessions()
    assert kept.stat().st_ino == inode  # unchanged: not copied again
    with (a.trial.logs_dir / "pi" / "sessions" / "s.jsonl").open("a") as out:
        out.write("{}\n")
    a.trial.agent.keep_sessions()
    assert kept.stat().st_ino != inode and lines(a) == 2


async def test_each_trial_gets_its_own_network_removed_after_it(fakes, tmp_path, clock):
    a = adapter(tmp_path, trial_hours=1)
    _, events = await start(a)
    clock[0] += 3600  # t1 ends; the next prompt starts t2
    await asyncio.wait_for(asyncio.shield(a._run), 5)
    await a.send("a")
    await read(events, "turn_ended")
    await a.close()
    calls = [c[2:] for c in fakes.net if c[0] == "docker" and c[2] != "inspect"]
    t1, t2 = (f"legion-{a._network_id}-agent-0000-t{n}" for n in (1, 2))
    assert [(c[0], c[-1]) for c in calls] == [
        ("create", t1), ("rm", t1), ("create", t2), ("rm", t2)]
    (overlay,) = fakes.trials[1].config.environment.extra_docker_compose
    assert json.loads(overlay.read_text())["networks"]["default"] == {"external": True, "name": t2}
    # the agent's env names: the environment keeps their values out of every exec's argv
    assert fakes.trials[1].config.environment.kwargs["secret_names"] == [
        "HF_TOKEN", "LEGION_TOOLS_MANIFEST"]


@pytest.mark.parametrize("backend", ["hf-sandbox", "e2b"])
async def test_a_sandbox_trial_gets_no_network(fakes, tmp_path, backend):
    a = adapter(tmp_path, backend=backend)
    await start(a)
    await a.close()
    env = fakes.trials[0].config.environment
    assert fakes.net == [] and env.extra_docker_compose == []
    if backend == "e2b":  # Harbor's own E2B environment, living as long as the Trial
        assert env.import_path == "legion.harbor.envs:E2B"
        assert env.kwargs["sandbox_timeout_secs"] == math.ceil(
            fakes.trials[0].config.agent.override_timeout_sec) + 600


async def test_close_now_cancels_unverified_and_keeps_the_sessions(fakes, tmp_path):
    fakes.mounted = False
    a = adapter(tmp_path)
    _, events = await start(a)
    for prompt in ("a", "b"):
        await a.send(prompt)
        await read(events, "turn_ended")
    assert lines(a) == 1
    await asyncio.wait_for(a.close(0), 5)
    trial = fakes.trials[0]
    assert not a.alive() and trial.cancelled and not trial.verified
    assert lines(a) == 2  # Harbor's download as it stopped the Trial


async def test_close_cancels_a_slow_verifier_after_its_timeout(fakes, tmp_path):
    a = adapter(tmp_path)
    await start(a)
    fakes.verify_gate = asyncio.Event()
    assert await asyncio.wait_for(a.close(timeout=0.1), 5) is None
    assert not a.alive() and not fakes.trials[0].verified and fakes.trials[0].cancelled


async def test_close_while_a_trial_starts_does_not_raise(fakes, tmp_path, monkeypatch):
    from ..worker_fakes import until

    monkeypatch.setattr(legion_agent, "PROBE", 0.05)
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.probe_ok = False
    await read(events, "exited")  # t1 failed its probe
    fakes.probe_ok, fakes.start_gate = True, asyncio.Event()
    await a.send("b")  # starts t2, which hangs before LegionPi is ready
    await until(lambda: len(fakes.trials) == 2)
    burst = asyncio.create_task(a.start(a.workdir, None, a.spec))  # waits for t2
    await asyncio.sleep(0)
    assert await asyncio.wait_for(a.close(), 5) is None
    assert not a.alive() and fakes.trials[1].cancelled
    with pytest.raises(RuntimeError, match="agent closed"):  # a failed start, not a cancel
        await burst


async def test_a_trial_closed_while_created_leaves_no_network(fakes, tmp_path):
    fakes.route_gate = asyncio.Event()
    a = adapter(tmp_path)
    starting = asyncio.create_task(a.start(a.workdir, None, a.spec))
    await asyncio.sleep(0.01)  # its network is being set up
    await a.close(0)  # finds no Trial yet
    fakes.route_gate.set()
    with pytest.raises(RuntimeError, match="agent closed"):
        await starting
    created = [c[-1] for c in fakes.net if c[1:3] == ("network", "create")]
    removed = [c[-1] for c in fakes.net if c[1:3] == ("network", "rm")]
    assert created and removed == created and a.trial is None  # created, never run


async def test_a_failed_kill_is_reported(fakes, tmp_path, monkeypatch):
    monkeypatch.setattr(legion_agent, "STOP_GRACE", 0.05)
    a = adapter(tmp_path)
    _, events = await start(a)
    fakes.hold, fakes.kill_fails = {1}, True
    await a.send("a")
    await read(events, "turn_started")
    await a.stop()
    evs = await read(events, "exited")
    assert [e.kind for e in evs] == ["error", "exited"] and "kill pi" in evs[0].error
    assert "pkill" not in legion_agent.KILL_PI
    fakes.release(1)


def test_read_turn_counts_compaction(tmp_path):
    path = tmp_path / "pi.txt"
    path.write_text("\n".join(json.dumps(e) for e in [
        {"type": "message_end",
         "message": {"role": "assistant", "usage": {"input": 10, "output": 1}}},
        {"type": "compaction_end", "result": {"usage": {"input": 100, "output": 20}}},
    ]))
    messages, usage = legion_agent.read_turn(path)
    assert len(messages) == 1 and (usage.input, usage.output) == (110, 21)


def test_the_kill_script_needs_no_procps_and_spares_itself(tmp_path):
    """KILL_PI on a fake /proc, `kill` stubbed: nothing on this machine is signalled. It ends
    `pi` and its tool processes (12 -> 11 -> 20 -> 21), not the rest."""
    import subprocess

    proc = tmp_path / "proc"
    script = legion_agent.KILL_PI.replace("/proc", str(proc))
    for pid, ppid, argv in [("11", "12", ["node", "/usr/bin/pi", "--print", "--mode", "json"]),
                            ("12", "1", ["bash", "-c", "set -o pipefail; pi --print 'hi'"]),
                            ("20", "11", ["bash", "-c", "make test"]), ("21", "20", ["make"]),
                            ("13", "1", ["sleep", "100"]), ("14", "1", ["bash", "-c", script])]:
        (proc / pid).mkdir(parents=True)
        (proc / pid / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (proc / pid / "stat").write_text(f"{pid} (a) b) S {ppid} {pid} 0\n")  # comm may hold ") "

    def run():
        return subprocess.run(["sh", "-c", f'kill() {{ echo "$@"; }}; {script}'],
                              capture_output=True, text=True)

    killed = run()
    assert killed.returncode == 0 and sorted(killed.stdout.split()) == ["11", "12", "20", "21"]
    for pid in ("11", "12"):
        (proc / pid / "cmdline").unlink()
    assert run().returncode == 1  # nothing to kill: reported


def runtime(tmp_path):
    from legion.worker.agents import AgentRuntime
    from legion.worker.events import Outbox

    a = adapter(tmp_path)
    spec = a.spec.model_copy(update={"idle_grace": "200ms"})
    return AgentRuntime("agent-0000", "agent", spec, swarm="sw", work_root=a.workdir.parent,
                        adapter=a, outbox=Outbox()), a


async def test_a_restart_prompt_names_no_worker_path(fakes, tmp_path):
    from ..worker_fakes import event_types, until

    rt, _ = runtime(tmp_path)
    await rt.setup(restore=False)
    rt.state.session_id = "gone"  # no session files: ResumeError, then a fresh session
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    assert "session_restarted" in event_types(rt) and rt.state.session_id != "gone"
    assert "transcript is not available here" in fakes.pi[0]
    assert str(rt.log.path) not in fakes.pi[0]  # the Trial cannot read the worker's files


async def test_no_trial_starts_after_the_exit_path_closed_the_agent(fakes, tmp_path):
    """stop and close(0) run together on exit: a burst still on its way to adapter.start (here
    in harness_env) must not start a Trial after close(0) found none."""
    from ..worker_fakes import event_types, until

    rt, _ = runtime(tmp_path)
    await rt.setup(restore=False)
    gate, harness_env = asyncio.Event(), rt.harness_env

    async def parked():
        await gate.wait()
        return await harness_env()

    rt.harness_env = parked
    await rt.prompt("go")
    await until(lambda: rt._burst is not None)
    stopping = asyncio.gather(rt.stop("terminated"), rt.close(0))
    await asyncio.sleep(0.01)
    gate.set()
    await asyncio.wait_for(stopping, 3)
    assert fakes.trials == [] and "error" in event_types(rt)


async def test_drain_cancels_a_slow_verifier(fakes, tmp_path, monkeypatch):
    from legion.worker import agents

    from ..worker_fakes import until

    monkeypatch.setattr(agents, "VERIFY_WAIT", 0.1)
    rt, a = runtime(tmp_path)
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    fakes.verify_gate = asyncio.Event()  # never opens
    await asyncio.wait_for(rt.drain(), 3)
    trial = fakes.trials[0]
    assert not a.alive() and trial.cancelled and not trial.verified


def harbor_worker(tmp_path, *texts, sync=None):
    """A worker with agent-0000 (this file's fakes) assigned, then sent `texts`; its bursts
    stay open 2 s after a turn."""
    from legion.models import AssignCommand, PromptCommand

    from ..test_worker_main import FakeController, make_worker

    ctrl = FakeController()
    ctrl.batches = [
        [AssignCommand(agent="agent-0000", type="researcher", restore=False,
                       spec=type_spec().model_copy(update={"idle_grace": "2s"}))],
        [PromptCommand(agent="agent-0000", prompt_id=f"p{n}", text=t) for n, t in enumerate(texts)],
    ]
    worker, _ = make_worker(tmp_path, ctrl, sync=sync)
    worker.adapter_factory = lambda name, sp: make_adapter(worker.cfg.work_root / name, name, sp)
    return worker


@pytest.mark.parametrize("path", ["terminate", "preempt"])
async def test_an_exit_while_a_trial_starts_cancels_it(fakes, tmp_path, path):
    from ..worker_fakes import until

    fakes.start_gate = asyncio.Event()  # never opens: Pi is never ready
    worker = harbor_worker(tmp_path, "go")
    worker.cfg.drain_budget = 2.0
    run = asyncio.create_task(worker.run())
    await until(lambda: len(fakes.trials) == 1)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await asyncio.wait_for(getattr(worker, path)("test"), 3)
    assert fakes.trials[0].cancelled and loop.time() - t0 < 1  # not at the end of the budget
    await asyncio.wait_for(run, 3)


async def test_preempt_pushes_the_last_turns_unmounted_sessions(fakes, tmp_path):
    from ..worker_fakes import MemoryStore, RecordingSync, until

    fakes.mounted = False
    store = MemoryStore()
    worker = harbor_worker(tmp_path, "a", sync=RecordingSync(store))
    run = asyncio.create_task(worker.run())
    await until(lambda: "agent-0000" in worker.agents and worker.agents["agent-0000"].state.turns)
    rt = worker.agents["agent-0000"]
    await rt.prompt("b")
    await until(lambda: rt.state.turns == 2)
    await worker.preempt("test")
    await asyncio.wait_for(run, 3)
    (session,) = [k for k in store.files if k.endswith("s.jsonl")]
    assert len(store.files[session].splitlines()) == 2  # the turn after the last SYNC download


async def test_a_pi_turn_leaves_no_value_in_an_argv(fakes, monkeypatch, tmp_path):
    """The env the real LegionPi gives a `pi --print` turn (Harbor's Pi passes the inference key
    as OPENAI_API_KEY, not under the spec's name) through legion's docker environment with the
    adapter's secret_names: every name is passed bare, every value through the environment."""
    a = adapter(tmp_path)
    _, events = await start(a)
    await a.send("go")
    await read(events, "turn_ended")
    await a.close()
    (pi_env,) = fakes.pi_env
    runs, env = compose(monkeypatch, tmp_path, "docker",
                        fakes.trials[0].config.environment.kwargs["secret_names"])
    await env.exec("pi --print go", env=pi_env, user="agent")
    ((argv, compose_env),) = runs
    assert pi_env and all(compose_env[k] == v for k, v in pi_env.items())
    assert not [x for x in argv for v in pi_env.values() if v in x], argv


@pytest.mark.parametrize("body,success,reward", [
    ('{"success": true, "evidence": {"won": true}}', True, 1),
    ('{"success": true}', True, 0),
    ('{"success": false}', False, 1),
    ('{"success": "true"}', False, 1),
    ('not-json', False, 1),
])
async def test_sidecar_success_survives_until_verification(fakes, tmp_path, body, success, reward):
    fakes.success_response, fakes.reward = body, reward
    ts = type_spec()
    ts.harbor.config["success_check"] = {"service": "puzzle", "command": "check"}
    a = make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts)
    try:
        await a.start(a.workdir, None, ts)
        # Checks also run while no model turn is active.
        from ..worker_fakes import until

        await until(lambda: bool(fakes.success_calls))
        status = a.trial_status()
        assert bool(status["success"]) is success
        assert fakes.success_calls == [("check", "puzzle", 5)]
        await a.close(timeout=2)
        if success:
            assert a.trial_status()["success"]["verification"] == ("passed" if reward else "failed")
            assert fakes.trials[0].verified
            saved = a.home / "results" / a.trial.config.trial_name / "result.json"
            assert saved in a.artifact_files()
            assert "config" not in json.loads(saved.read_text())
    finally:
        await a.close(0)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("late_check", [False, True])
async def test_success_at_trial_expiry_is_retained_until_release(
        fakes, tmp_path, clock, enabled, late_check):
    from ..worker_fakes import until

    ts = type_spec(trial_hours=1)
    ts.harbor.config.update(stop_on_success=enabled,
                           success_check={"service": "puzzle", "command": '{"success": true}'})
    a = make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts)
    if late_check:
        fakes.success_gate = asyncio.Event()
    try:
        _, events = await start(a)
        await until(lambda: bool(fakes.success_calls))
        winning_trial = a.trial_status()["id"]
        clock[0] += 3600
        await until(lambda: not a.alive())
        if late_check:
            fakes.success_gate.set()
        await until(lambda: a.trial_status()["success"] is not None
                    and a.trial_status()["success"]["verification"] == "passed")
        # A prompt between detection and the controller's heartbeat must not replace the win.
        await a.start(a.workdir, None, ts)
        await a.send("next turn")
        if enabled:
            assert a.trial_status()["id"] == winning_trial
            assert a.trial_status()["success"]["verification"] == "passed"
            assert len(fakes.trials) == 1
        else:
            await read(events, "turn_ended")
            assert len(fakes.trials) == 2
        await a.close(timeout=2)
        assert any(p.name == "result.json" for p in a.artifact_files())
    finally:
        await a.close(0)
