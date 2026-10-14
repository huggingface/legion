"""AgentRuntime: process-per-burst, stuck, heartbeat, restore, unassign."""

from __future__ import annotations

import asyncio
import json

from swarm_harness.models import TokenUsage
from swarm_harness.worker.agents import AgentState

from .worker_fakes import MemoryStore, RecordingSync, ScriptedAdapter, event_types, runtime, until


async def test_burst_lifecycle(tmp_path):
    adapter = ScriptedAdapter(tmp_path / "home", start_delay=0.1, turn_time=0.3)
    rt, _ = runtime(tmp_path, adapter=adapter)
    await rt.setup(restore=False)

    # first prompt starts the process; prompts during startup are buffered, then forwarded in order
    await rt.prompt("a")
    assert rt.phase == "starting"
    await until(lambda: adapter.starts == [None])
    await rt.prompt("b")
    await rt.prompt("c")
    assert adapter.sent == []
    assert rt.heartbeat().state == "working"
    await until(lambda: adapter.sent == ["a", "b", "c"])
    await until(lambda: rt.turn_id == "t-1")

    # a prompt during a turn is forwarded immediately
    await rt.prompt("d")
    assert adapter.sent[-1] == "d"
    assert rt.buffer == []

    # turn ends, then the process stops after idle_grace (200ms)
    await until(lambda: rt.turn_id is None)
    assert rt.phase == "running"
    await until(lambda: rt.phase == "none")
    assert adapter.stops == 1

    # the next prompt restarts the harness, resuming the session
    await rt.prompt("e")
    await until(lambda: adapter.sent[-1:] == ["e"])
    assert adapter.starts == [None, "s-1"]
    assert rt.resumed is True
    await until(lambda: rt.state.turns == 2)
    types = event_types(rt)
    assert types.count("session_started") == 1 and types.count("session_resumed") == 1
    assert types.count("turn_started") == 2 and types.count("turn_ended") == 2
    # turn ids are t-<n> per agent and land on the events
    ended = [e for e in rt.outbox._q if e.type == "turn_ended"]
    assert [e.turn_id for e in ended] == ["t-1", "t-2"]


async def test_stuck_on_turn_timeout(tmp_path):
    adapter = ScriptedAdapter(tmp_path / "home", hang=True)
    rt, _ = runtime(tmp_path, adapter=adapter, turn_timeout="300ms", idle_grace="5s")
    await rt.setup(restore=False)
    await rt.prompt("loop forever")
    await until(lambda: rt.turn_id == "t-1")
    await until(lambda: rt.phase == "none", timeout=3)
    hb = rt.heartbeat()
    assert hb.state == "stuck" and hb.turn_id is None and hb.idle_since is not None
    assert adapter.stops == 1
    interrupted = [e for e in rt.outbox._q if e.type == "turn_interrupted"]
    assert interrupted[0].data["data"]["reason"] == "timeout" and interrupted[0].turn_id == "t-1"
    # cleared by the next turn
    adapter.hang = False
    await rt.prompt("again")
    await until(lambda: rt.turn_id == "t-2")
    assert rt.heartbeat().state == "working"


async def test_per_prompt_timeout_overrides_spec(tmp_path):
    adapter = ScriptedAdapter(tmp_path / "home", hang=True)
    rt, _ = runtime(tmp_path, adapter=adapter, turn_timeout="30m", idle_grace="5s")
    await rt.setup(restore=False)
    await rt.prompt("x", timeout="200ms")
    await until(lambda: rt.stuck, timeout=3)


async def test_heartbeat_payload(tmp_path):
    rt, _ = runtime(tmp_path)
    await rt.setup(restore=False)
    hb = rt.heartbeat()
    assert hb.state == "idle" and hb.session_id is None and hb.resumed is None and hb.turns == 0
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    await rt.prompt("two")
    await until(lambda: rt.state.turns == 2)
    hb = rt.heartbeat()
    assert hb.name == "researcher-0001" and hb.state == "idle" and hb.idle_since is not None
    assert hb.session_id == "s-1" and hb.resumed is False and hb.turns == 2
    # cumulative usage from the harness is differenced per turn, reported cumulative by model
    assert hb.tokens == {"m-1": TokenUsage(input=20, output=10, cache_read=200)}
    assert hb.harness_estimate_usd == 0.01
    ended = [e for e in rt.outbox._q if e.type == "turn_ended"]
    assert ended[1].data["data"]["tokens"]["m-1"]["input"] == 10
    # size budget: about 100-200 bytes per agent
    assert len(hb.model_dump_json(exclude_none=True)) < 400


async def test_state_persists_and_reloads(tmp_path):
    rt, _ = runtime(tmp_path)
    await rt.setup(restore=False)
    await rt.prompt("one")
    await until(lambda: rt.state.turns == 1)
    await rt.stop()
    saved = AgentState.model_validate_json(rt.state_path.read_text())
    assert saved.session_id == "s-1" and saved.turn_seq == 1
    rt2, _ = runtime(tmp_path, adapter=ScriptedAdapter(tmp_path / "home"))
    await rt2.setup(restore=False)
    await rt2.prompt("two")
    await until(lambda: rt2.state.turns == 2)
    assert rt2.adapter.starts == ["s-1"]
    assert rt2.state.turn_seq == 2  # turn ids keep counting across workers


async def test_assign_with_restore_restores_before_first_start(tmp_path):
    order: list = []
    store = MemoryStore()
    pre = "sw/researcher/researcher-0001/"
    store.files[pre + "workdir/notes.txt"] = b"PELICAN-7"
    store.files[pre + "sessions/researcher-0001.state.json"] = AgentState(
        session_id="old", turn_seq=7, turns=7, tokens={"m-1": TokenUsage(input=5)}).model_dump_json().encode()
    store.files[pre + "harness/old.jsonl"] = b"{}\n"
    sync = RecordingSync(store, log=order)
    adapter = ScriptedAdapter(tmp_path / "home", log=order)
    rt, _ = runtime(tmp_path, adapter=adapter, sync=sync)
    setup = asyncio.create_task(rt.setup(restore=True))
    await rt.prompt("continue")  # arrives before setup finished: waits for the restore
    await setup
    await until(lambda: adapter.sent == ["continue"])
    assert order[0] == ("restore", "researcher-0001") and order[1] == ("start", "old")
    assert (rt.workdir / "notes.txt").read_text() == "PELICAN-7"
    assert rt.resumed is True and rt.turn_id == "t-8"
    assert rt.heartbeat().tokens["m-1"].input == 5


async def test_restore_without_harness_files_restarts_fresh(tmp_path):
    store = MemoryStore()
    pre = "sw/researcher/researcher-0001/"
    store.files[pre + "sessions/researcher-0001.state.json"] = AgentState(session_id="gone").model_dump_json().encode()
    rt, adapter = runtime(tmp_path, sync=RecordingSync(store))
    await rt.setup(restore=True)
    await rt.prompt("hello")
    await until(lambda: len(adapter.sent) == 2)
    assert adapter.starts == ["gone", None]
    assert "could not be resumed" in adapter.sent[0] and adapter.sent[1] == "hello"
    restarted = next(e for e in rt.outbox._q if e.type == "session_restarted")
    assert restarted.data["data"]["resumed"] is False
    assert restarted.data["data"]["previous_session_id"] == "gone"
    assert rt.heartbeat().resumed is False


async def test_resume_prompt_sent_after_restore(tmp_path):
    rt, adapter = runtime(tmp_path)
    await rt.setup(restore=False, resume_prompt="Your previous turn was interrupted.")
    await until(lambda: adapter.sent == ["Your previous turn was interrupted."])


async def test_unassign_stops_and_pushes(tmp_path):
    store = MemoryStore()
    sync = RecordingSync(store)
    adapter = ScriptedAdapter(tmp_path / "home", turn_time=5)
    rt, _ = runtime(tmp_path, adapter=adapter, sync=sync, idle_grace="10s")
    await rt.setup(restore=False)
    (rt.workdir).mkdir(parents=True, exist_ok=True)
    (rt.workdir / "out.txt").write_text("result")
    await rt.prompt("long job")
    await until(lambda: rt.turn_id is not None)
    await rt.unassign()
    assert adapter.stops == 1 and rt.phase == "none"
    assert sync.pushed == ["researcher-0001"]
    pre = "sw/researcher/researcher-0001/"
    assert store.files[pre + "workdir/out.txt"] == b"result"
    assert pre + "harness/s-1.jsonl" in store.files
    log_lines = store.files[pre + "sessions/researcher-0001.jsonl"].decode().splitlines()
    assert json.loads(log_lines[0])["type"] == "session"
    assert "turn_interrupted" in event_types(rt)
    await rt.prompt("after unassign")  # no new process
    assert rt.phase == "none" and "prompt_deferred" in event_types(rt)


async def test_stop_keeps_agent_assigned(tmp_path):
    adapter = ScriptedAdapter(tmp_path / "home", turn_time=5)
    rt, _ = runtime(tmp_path, adapter=adapter, idle_grace="10s")
    await rt.setup(restore=False)
    await rt.prompt("x")
    await until(lambda: rt.turn_id is not None)
    await rt.stop()
    assert rt.phase == "none" and adapter.stops == 1
    await rt.prompt("y")
    await until(lambda: adapter.sent[-1:] == ["y"])
    assert adapter.starts == [None, "s-1"]


async def test_drain_finishes_inflight_turn(tmp_path):
    sync = RecordingSync(MemoryStore())
    adapter = ScriptedAdapter(tmp_path / "home", turn_time=0.3)
    rt, _ = runtime(tmp_path, adapter=adapter, sync=sync, idle_grace="10s")
    await rt.setup(restore=False)
    await rt.prompt("finish me")
    await until(lambda: rt.turn_id is not None)
    await rt.drain()
    types = event_types(rt)
    assert "turn_ended" in types and "turn_interrupted" not in types
    assert sync.pushed == ["researcher-0001"] and rt.phase == "none"
    await rt.prompt("too late")
    assert adapter.starts == [None] and "prompt_deferred" in event_types(rt)


async def test_msgboard_env_injected(tmp_path):
    class FakeIds:
        async def env_for(self, agent, spec):
            return {"MSGBOARD_URL": "https://mb", "MSGBOARD_TOKEN": "tok"}

        def principal(self, agent):
            return agent

    rt, adapter = runtime(tmp_path, env={"FOO": "1"})
    rt.identities = FakeIds()
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: adapter.sent == ["hi"])
    assert adapter.envs[0] == {"FOO": "1", "MSGBOARD_URL": "https://mb", "MSGBOARD_TOKEN": "tok"}
    assert rt.heartbeat().msgboard_principal == "researcher-0001"


async def test_system_prompt_written(tmp_path):
    prompt = tmp_path / "researcher.md"
    prompt.write_text("You are a researcher.")
    rt, adapter = runtime(tmp_path, system_prompt_file=str(prompt))
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: adapter.sent == ["hi"])
    assert (rt.workdir / "CLAUDE.md").read_text() == "You are a researcher."
