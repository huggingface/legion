"""HarborAdapter + FlockPi on fakes: a FakeTrial (test_launch's) that runs the real FlockPi
against a FakeEnv (after harbor-core's test_agent.py) whose `pi --print` writes a canned pi.txt."""

import asyncio
import json
import shlex
from pathlib import Path

import pytest

pytest.importorskip("harbor")

from harbor.agents.factory import AgentFactory
from harbor.environments.base import ExecResult
from harbor.models.agent.context import AgentContext
from harbor.models.trial.result import ExceptionInfo
from harbor.models.verifier.result import VerifierResult

from flock.harbor import agent as flock_agent
from flock.harbor import launch
from flock.harbor.agent import FlockPi
from flock.harbor.spec import Spec, compile_spec
from flock.worker.adapters.base import ResumeError
from flock.worker.adapters.harbor import HarborAdapter, make_adapter

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
        self.trials, self.pi, self.root, self.uploads = [], [], [], []
        self.probe_ok, self.upload_fails = True, False
        self.fail = None  # pi call number -> "exit" | "provider"
        self.hold = set()  # pi call numbers that wait for release(n) or a kill
        self.gates = {}

    def gate(self, n):
        return self.gates.setdefault(n, asyncio.Event())

    def release(self, n):
        self.gate(n).set()


class FakeEnv:
    def __init__(self, logs_dir, fakes):
        self.logs_dir, self.fakes, self.default_user = logs_dir, fakes, None
        self.killed = False

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
        return True

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        f = self.fakes
        if command == "true":
            return ExecResult(return_code=0 if f.probe_ok else 1)
        if user == "root":
            f.root.append(command)
        if "pkill" in command:
            self.killed = True
            for gate in f.gates.values():
                gate.set()
            return ExecResult(return_code=0)
        if " pi --print " not in command:
            return ExecResult(return_code=0)
        f.pi.append(command)
        n = len(f.pi)
        if n in f.hold:
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
        (self.logs_dir / "pi.txt").write_text(
            json.dumps({"type": "message_end", "message": message}) + "\n"
        )
        sessions = self.logs_dir / "pi" / "sessions"
        sessions.mkdir(parents=True, exist_ok=True)
        with (sessions / "s.jsonl").open("a") as out:
            out.write(json.dumps({"call": n}) + "\n")
        return ExecResult(return_code=0)


class Trial(FakeTrial):
    """A Trial's life as Harbor runs it: hooks, the agent's run(), then the verifier."""

    def __init__(self, config, fakes):
        super().__init__(config, [])
        logs_dir = config.trials_dir / config.trial_name / "agent"
        logs_dir.mkdir(parents=True)
        self.agent = AgentFactory.create_agent_from_config(config.agent, logs_dir=logs_dir)
        self.agent_environment = FakeEnv(logs_dir, fakes)
        self.verified = False

    async def run(self):
        for hook in self.hooks.values():
            await hook(None)
        instruction = "TASK\n\n" + self.config.extra_instructions[0]
        try:
            await self.agent.run(instruction, self.agent_environment, AgentContext())
        except Exception as exc:  # noqa: BLE001 - Harbor records any agent failure
            self.result.exception_info = ExceptionInfo.from_exception(exc)
            return self.result
        self.verified = True
        self.result.verifier_result = VerifierResult(rewards={"reward": 1.0})
        return self.result


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    shared = Fakes()

    async def create(config):
        trial = Trial(config, shared)
        shared.trials.append(trial)
        return trial

    monkeypatch.setattr(launch.Trial, "create", create)
    return shared


def adapter(tmp_path, **limits) -> HarborAdapter:
    spec = Spec.load(EXAMPLES / "pi-hf.yaml", env={})
    for key, value in limits.items():
        setattr(spec.limits, key, value)
    ts = compile_spec(spec, TASK).types["agent"]
    return make_adapter(tmp_path / "work" / "agent-0000", "agent-0000", ts)


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


async def test_one_trial_across_bursts(fakes, tmp_path):
    a = adapter(tmp_path)
    sid, events = await start(a)
    await a.send("First prompt.")
    evs = await read(events, "turn_ended")
    assert [e.kind for e in evs] == ["turn_started", "message", "usage", "turn_ended"]
    assert evs[0].data == {"trial": "agent-0000-t1"} and evs[-1].status == "completed"
    usage = evs[2]
    assert usage.model == "huggingface/moonshotai/Kimi-K2.5"  # priced under the inference provider
    assert (usage.usage.input, usage.usage.cache_read, usage.usage.output) == (1000, 500, 100)
    assert evs[1].message["content"][0]["text"] == "reply-1"
    await a.stop()
    assert (await read(events, "exited"))[-1].kind == "exited"
    assert a.alive() and isinstance(fakes.trials[0].agent, FlockPi)

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
    assert session == a.home / "sessions" / "s.jsonl"
    assert len(session.read_text().splitlines()) == 2


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
    assert any("pkill -f '[p]i --print'" in c for c in fakes.root)
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
    (a.home / "sessions" / "s.jsonl").unlink()
    sid, _ = await start(a)  # the runtime's fallback: a fresh session
    assert sid not in ("gone", "s-1") and len(fakes.trials) == 2 and a.alive()


async def test_a_failed_probe_reports_and_the_next_start_takes_a_new_trial(
    fakes, tmp_path, monkeypatch
):
    monkeypatch.setattr(flock_agent, "PROBE", 0.05)
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
    assert fakes.trials[0].verified  # FlockPi returned: Harbor ran the verifier
    _, events = await start(a, None)
    await a.send("d")
    assert (await read(events, "turn_ended"))[-1].status == "completed"
    assert len(fakes.trials) == 2


async def test_trial_hours_roll_over_to_a_new_trial(fakes, tmp_path):
    a = adapter(tmp_path, trial_hours=1e-9)  # each Trial ends once it is idle
    _, events = await start(a)
    for prompt in ("a", "b"):
        await asyncio.wait_for(asyncio.shield(a._run), 5)  # t1 ends idle, before any prompt
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
    from flock.worker.agents import AgentRuntime
    from flock.worker.events import Outbox

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
    assert rt.state.tokens["huggingface/moonshotai/Kimi-K2.5"].input == 2000
    assert rt.state.last_summary == "reply-2"
