"""Pi adapter: mapping against the recorded RPC fixture, models.json, env allow-list, session
files and resume checks; then the real `pi` binary against tests/fake_openai.py.

The `pi_bin` tests need Pi on PATH or FLOCK_PI_CLI pointing at it, e.g.
`FLOCK_PI_CLI=$AM_LOCAL/pi/node_modules/.bin/pi pytest tests/test_worker_pi.py`.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

from flock.models import TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, ResumeError
from flock.worker.adapters.pi import (
    PiAdapter,
    PiMapper,
    PiModelConfig,
    _descendants,
    harness_env,
    write_models_json,
)
from flock.worker.agents import INSTRUCTIONS_FILE, AgentRuntime, pi_model_config
from flock.worker.events import Outbox

from . import fake_openai
from .worker_fakes import ScriptedAdapter, until

FIX = Path(__file__).parent / "fixtures"
PI_BIN = os.environ.get("FLOCK_PI_CLI") or shutil.which("pi")
pi_bin = pytest.mark.skipif(not (PI_BIN and os.access(PI_BIN, os.X_OK)),
                            reason="pi binary not found (set FLOCK_PI_CLI)")
SID = "00000000-0000-4000-8000-000000000001"


def pi_spec(**kw) -> TypeSpec:
    return TypeSpec.model_validate({"harness": "pi", "model": "fake-model", "idle_grace": "300ms",
                                    "turn_timeout": "20s", **kw})


def replay(records: list[dict]) -> list[AdapterEvent]:
    m = PiMapper("fake/fake-model")
    out: list[AdapterEvent] = []
    for r in records:
        if r.get("type") == "response" and r.get("command") == "prompt":
            out += m.on_prompt_response(r)
        else:
            out += m.on_record(r)
    return out


# --------------------------------------------------------------------------- fixture mapping


def test_fixture_maps_to_turns_messages_and_per_call_usage():
    recs = [json.loads(x) for x in (FIX / "pi_rpc.jsonl").read_text().splitlines()]
    evs = replay(recs)
    kinds = [e.kind for e in evs]
    # hello | tool call + two queued steers (one turn) | endpoint 400 | abort mid-tool
    assert kinds.count("turn_started") == 4
    assert [e.status for e in evs if e.kind == "turn_ended"] == ["completed", "completed", "error", "interrupted"]
    err = next(e for e in evs if e.kind == "turn_ended" and e.status == "error")
    assert "fake failure" in err.error
    # turn 2 carries the tool call, its result and both steered prompts, in order
    t2 = evs[kinds.index("turn_started", 1) + 1: kinds.index("turn_ended", kinds.index("turn_started", 1))]
    roles = [(e.message["role"], e.message["content"][0].get("text") if e.message["content"] else None)
             for e in t2 if e.kind == "message"]
    assert roles == [("user", "bash: sleep 1; echo slept"), ("assistant", None), ("toolResult", "slept\n"),
                     ("user", "also this"), ("user", "and this"), ("assistant", "echo: and this")]
    # messages pass through untouched: byte-identical to Pi's message_end payload
    ends = [r["message"] for r in recs if r["type"] == "message_end"]
    assert [e.message for e in evs if e.kind == "message"] == ends
    # usage: one event per model call that reported tokens, never cumulative
    usage = [e for e in evs if e.kind == "usage"]
    calls = [m for m in ends if m["role"] == "assistant" and m["usage"]["totalTokens"]]
    assert len(usage) == len(calls) == 4
    assert all(not e.cumulative and not e.baseline and e.model == "fake/fake-model" for e in usage)
    assert usage[0].usage == TokenUsage(input=62, output=7, cache_read=40)
    stats = next(r for r in recs if r.get("command") == "get_session_stats")["data"]["tokens"]
    total = sum((e.usage for e in usage), TokenUsage())
    assert (total.input, total.output, total.cache_read) == (stats["input"], stats["output"], stats["cacheRead"])


def test_mapper_errors_and_ignored_records():
    m = PiMapper()
    assert m.on_record({"type": "agent_settled", "aborted": False}) == []  # no open turn
    assert [e.kind for e in m.on_record({"type": "agent_start"})] == ["turn_started"]
    assert m.on_record({"type": "agent_start"}) == []  # one flock turn per run
    for t in ("turn_start", "turn_end", "message_update", "tool_execution_start", "queue_update", "agent_end"):
        assert m.on_record({"type": t}) == []
    e = m.on_record({"type": "auto_retry_end", "success": False, "finalError": "overloaded"})
    assert e[0].kind == "error" and e[0].error == "overloaded"
    e = m.on_record({"type": "response", "command": "compact", "success": False, "error": "nope"})
    assert e[0].kind == "error"
    e = m.on_record({"type": "compaction_end", "result": {"usage": {"input": 5, "output": 2}}})
    assert e[0].kind == "usage" and e[0].usage == TokenUsage(input=5, output=2)


# --------------------------------------------------------------------------- config, env, files


def test_models_json_has_env_reference_not_key(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_KEY", "sk-very-secret")
    cfg = PiModelConfig(base_url="https://h/v1", api_key_env="VLLM_KEY", model_id="Qwen/Q-27B",
                        reasoning="low", extra={"contextWindow": 131072})
    path = write_models_json(tmp_path, cfg)
    data = json.loads(path.read_text())
    assert data == {"providers": {"flock": {"baseUrl": "https://h/v1", "api": "openai-completions",
                                            "apiKey": "$VLLM_KEY",
                                            "models": [{"id": "Qwen/Q-27B", "reasoning": True,
                                                        "contextWindow": 131072}]}}}
    assert "sk-very-secret" not in path.read_text()
    assert [p.name for p in tmp_path.iterdir()] == ["models.json"]
    assert not PiModelConfig("u", "K", "m", reasoning="off").models_json()["providers"]["flock"]["models"][0]["reasoning"]
    assert cfg.model_ref == "flock/Qwen/Q-27B"


def test_harness_env_is_an_allow_list():
    base = {"PATH": "/bin", "HOME": "/root", "LANG": "C.UTF-8", "FLOCK_TOKEN": "worker-secret",
            "FLOCK_URL": "https://c", "HF_TOKEN": "hf_x", "VLLM_KEY": "sk-1", "GH_TOKEN": "g",
            "ANTHROPIC_API_KEY": "a", "PI_OFFLINE": "1"}
    spec = pi_spec(secrets=["ANTHROPIC_API_KEY", "FLOCK_TOKEN"], env={"FOO": "bar"})
    cfg = PiModelConfig("u", "VLLM_KEY", "m")
    env = harness_env(spec, {"FOO": "bar", "MSGBOARD_TOKEN": "mb", "FLOCK_TOKEN": "x"}, Path("/state/pi"), cfg,
                      base=base)
    assert env == {"PATH": "/bin", "HOME": "/root", "LANG": "C.UTF-8", "PI_OFFLINE": "1",
                   "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0", "ANTHROPIC_API_KEY": "a",
                   "VLLM_KEY": "sk-1", "FOO": "bar", "MSGBOARD_TOKEN": "mb",
                   "PI_CODING_AGENT_DIR": "/state/pi"}


def test_pi_model_config_glue():
    assert pi_model_config(pi_spec(model="anthropic/claude-haiku-5-5")) is None
    c = pi_model_config(pi_spec(model="openai_compat:http://h:8000/v1|Qwen/Q",
                                env={"PI_API_KEY_ENV": "VLLM_KEY", "PI_REASONING": "high"}))
    assert c == PiModelConfig(base_url="http://h:8000/v1", api_key_env="VLLM_KEY", model_id="Qwen/Q",
                              reasoning="high")
    c = pi_model_config(pi_spec(model="ignored", env={"PI_BASE_URL": "http://r/v1", "PI_MODEL": "m-2"}))
    assert (c.base_url, c.model_id, c.api_key_env, c.reasoning) == ("http://r/v1", "m-2", "OPENAI_API_KEY", None)
    with pytest.raises(ValueError):
        pi_model_config(pi_spec(model="openai_compat:http://h/v1"))


def _session(home: Path, agent: str, sid: str, cwd: Path, ts: str = "2026-10-16T08-00-00-000Z") -> Path:
    p = home / "sessions" / agent / f"{ts}_{sid}.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"type": "session", "version": 3, "id": sid, "timestamp": "t", "cwd": str(cwd)}) + "\n")
    return p


def test_session_files_and_lookup(tmp_path):
    home, wd = tmp_path / "pi", tmp_path / "work" / "agents" / "a-0001"
    wd.mkdir(parents=True)
    a = PiAdapter(wd, home=home)
    assert a.session_dir == home / "sessions" / "a-0001"
    p = _session(home, "a-0001", SID, wd)
    _session(home, "a-0001", "other-sid", wd)
    _session(home, "a-0002", SID, tmp_path / "work" / "agents" / "a-0002")  # another agent's dir
    assert a.find_session(SID) == p
    assert a.session_files(SID) == [p]
    assert a.session_files(None) == []
    assert a.session_files("nope") == []


async def test_resume_error_before_spawn(tmp_path):
    home, wd = tmp_path / "pi", tmp_path / "wd"
    wd.mkdir()
    a = PiAdapter(wd, home=home, pi_bin="/nonexistent/pi")  # never spawned
    with pytest.raises(ResumeError, match="no session file"):
        await a.start(wd, SID, pi_spec())
    # same id written from another cwd: Pi would silently fork a second file, so refuse
    _session(home, "wd", SID, tmp_path / "elsewhere")
    with pytest.raises(ResumeError, match="elsewhere"):
        await a.start(wd, SID, pi_spec())


async def test_runtime_writes_agents_md_for_pi(tmp_path):
    assert INSTRUCTIONS_FILE["pi"] == "AGENTS.md"
    adapter = ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime("a-0001", "a", pi_spec(system_prompt="be terse"), swarm="sw", work_root=tmp_path,
                      adapter=adapter, outbox=Outbox())
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    assert (tmp_path / "a-0001" / "AGENTS.md").read_text() == "be terse"
    await rt.stop()


def test_worker_factory_builds_pi_adapter(tmp_path):
    from datetime import UTC, datetime

    from flock.worker.main import WorkerConfig, default_adapter_factory

    cfg = WorkerConfig(url="u", token="t", worker_id="w", swarm="s", type="a",
                       deadline=datetime.now(UTC), bucket=None, work_root=tmp_path)
    a = default_adapter_factory(cfg)("a-0001", pi_spec(model="openai_compat:http://h/v1|m"))
    assert isinstance(a, PiAdapter) and a.agent == "a-0001" and a.model_config.model_id == "m"


# --------------------------------------------------------------------------- real pi + fake server


@pytest.fixture
def fake_server(tmp_path):
    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    srv.log = log
    yield srv
    srv.shutdown()


def requests_seen(srv) -> list[dict]:
    if not srv.log.exists():
        return []
    return [json.loads(x) for x in srv.log.read_text().splitlines()]


def make_adapter(tmp_path, srv, monkeypatch, home: Path | None = None) -> tuple[PiAdapter, Path]:
    monkeypatch.setenv("FLOCK_TEST_PI_KEY", "sk-test-key")
    monkeypatch.setenv("FLOCK_TOKEN", "worker-only-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "nope")
    wd = tmp_path / "work" / "agents" / "pi-0001"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "AGENTS.md").write_text("You are pi-0001. MARKER-AGENTS-MD\n")
    cfg = PiModelConfig(base_url=f"http://127.0.0.1:{srv.server_port}/v1", api_key_env="FLOCK_TEST_PI_KEY",
                        model_id="fake-model")
    return PiAdapter(wd, home=home or tmp_path / "pi", pi_bin=PI_BIN, model_config=cfg), wd


async def until_turn_ended(a: PiAdapter, timeout: float = 30) -> list[AdapterEvent]:
    got: list[AdapterEvent] = []

    async def collect():
        async for ev in a.events():
            got.append(ev)
            if ev.kind in ("turn_ended", "exited"):
                return

    await asyncio.wait_for(collect(), timeout)
    return got


@pi_bin
async def test_pi_happy_path(tmp_path, fake_server, monkeypatch):
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, pi_spec(), env={"MSGBOARD_TOKEN": "mb-1"})
    assert len(sid) == 36 and "--session-id" in a.argv and "--no-extensions" in a.argv
    await a.send("bash: env | sort > env.txt; echo wrote")
    evs = await until_turn_ended(a)
    assert [e.kind for e in evs if e.kind != "message"] == ["turn_started", "usage", "usage", "turn_ended"]
    assert evs[-1].status == "completed"
    msgs = [e.message for e in evs if e.kind == "message"]
    assert [m["role"] for m in msgs if m["role"] != "system"] == ["user", "assistant", "toolResult", "assistant"]
    assert msgs[-1]["content"][0]["text"] == "done: wrote"
    assert all(e.model == "flock/fake-model" and not e.cumulative for e in evs if e.kind == "usage")
    # the tool ran in the workdir with the allow-listed env: key and identity yes, FLOCK_TOKEN no
    env_txt = (wd / "env.txt").read_text()
    assert "FLOCK_TEST_PI_KEY=sk-test-key" in env_txt and "MSGBOARD_TOKEN=mb-1" in env_txt
    assert "FLOCK_TOKEN" not in env_txt and "UNRELATED_SECRET" not in env_txt
    # the endpoint got the key from the env reference and AGENTS.md in the system prompt
    reqs = requests_seen(fake_server)
    assert {r["auth"] for r in reqs} == {"Bearer sk-test-key"}
    assert "MARKER-AGENTS-MD" in reqs[0]["body"]["messages"][0]["content"]
    assert "sk-test-key" not in (tmp_path / "pi" / "models.json").read_text()
    await a.stop()
    kinds = [ev.kind async for ev in a.events()]
    assert kinds[-1] == "exited"
    files = a.session_files(sid)
    assert len(files) == 1 and files[0].parent == tmp_path / "pi" / "sessions" / "pi-0001"
    assert files[0] == a.session_file


@pi_bin
async def test_pi_resume_from_restored_file(tmp_path, fake_server, monkeypatch):
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, pi_spec())
    await a.send("the codeword is PELICAN-7")
    await until_turn_ended(a)
    await a.stop()
    [src] = a.session_files(sid)
    # "another worker": a fresh Pi home that only gets the synced file back at the same path
    home2 = tmp_path / "pi2"
    dst = home2 / src.relative_to(tmp_path / "pi")
    dst.parent.mkdir(parents=True)
    shutil.copy(src, dst)
    b, _ = make_adapter(tmp_path, fake_server, monkeypatch, home=home2)
    assert await b.start(wd, sid, pi_spec()) == sid
    assert b.session_file == dst
    await b.send("what is the codeword?")
    evs = await until_turn_ended(b)
    assert evs[-1].status == "completed"
    await b.stop()
    last = requests_seen(fake_server)[-1]["body"]["messages"]
    texts = [json.dumps(m.get("content")) for m in last]
    assert any("PELICAN-7" in t for t in texts) and "what is the codeword?" in texts[-1]
    assert b.session_files(sid) == [dst]  # appended to, not a second file
    assert len(dst.read_text().splitlines()) > len(src.read_text().splitlines())


@pi_bin
async def test_pi_send_while_busy_is_queued_into_the_turn(tmp_path, fake_server, monkeypatch):
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, pi_spec())
    await a.send("bash: sleep 1; echo slept")
    await asyncio.sleep(0.4)
    await a.send("steer one")
    await a.send("steer two")
    evs = await until_turn_ended(a)
    assert [e.kind for e in evs].count("turn_started") == 1
    users = [e.message["content"][0]["text"] for e in evs if e.kind == "message" and e.message["role"] == "user"]
    assert users == ["bash: sleep 1; echo slept", "steer one", "steer two"]
    # steering mode "all": both queued prompts went out in one model call
    reqs = requests_seen(fake_server)
    assert len(reqs) == 2
    await a.stop()


@pi_bin
async def test_pi_stop_mid_tool_interrupts_and_leaves_no_process(tmp_path, fake_server, monkeypatch):
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, pi_spec())
    pid = a._proc.pid
    await a.send("bash: sleep 47; echo never")
    await until(lambda: any("sleep 47" in Path(f"/proc/{p}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                            for p, _ in _descendants(pid)), timeout=10)
    tool = [p for p, _ in _descendants(pid)]
    await a.stop()
    evs = [ev async for ev in a.events()]
    assert [e.status for e in evs if e.kind == "turn_ended"] == ["interrupted"]
    assert evs[-1].kind == "exited"
    await asyncio.sleep(0.2)
    assert not [p for p in tool if Path(f"/proc/{p}").exists()]


@pi_bin
async def test_pi_runtime_accounts_per_call_usage(tmp_path, fake_server, monkeypatch):
    a, _ = make_adapter(tmp_path, fake_server, monkeypatch)
    rt = AgentRuntime("pi-0001", "pi", pi_spec(), swarm="sw", work_root=tmp_path / "work" / "agents",
                      adapter=a, outbox=Outbox())
    await rt.setup(restore=False)
    await rt.prompt("bash: echo hi")
    await until(lambda: rt.state.turns == 1, timeout=30)
    await rt.prompt("again")
    await until(lambda: rt.state.turns == 2, timeout=30)
    # three model calls: 62/63/64 uncached input (+1 message each time), 40 cached, 7 out
    u = rt.state.tokens["flock/fake-model"]
    assert (u.output, u.cache_read) == (21, 120)
    await rt.stop()
    assert rt.state.session_id and rt.harness_files() == a.session_files(rt.state.session_id)


@pi_bin
async def test_pi_hung_process_is_killed_with_its_tool_processes(tmp_path, fake_server, monkeypatch):
    """SIGKILL alone would orphan the tool (its bash runs in its own session); stop() kills the
    descendants it saw before killing Pi."""
    import signal

    import flock.worker.adapters.pi as pimod

    monkeypatch.setattr(pimod, "STOP_GRACE", 0.5)
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, pi_spec())
    pid = a._proc.pid
    await a.send("bash: sleep 48; echo never")
    await until(lambda: len(_descendants(pid)) >= 2, timeout=10)
    tool = [p for p, _ in _descendants(pid)]
    os.kill(pid, signal.SIGSTOP)  # Pi stops answering: no abort response, stdin close ignored
    await a.stop()
    await asyncio.sleep(0.2)
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
    assert not [p for p in tool if Path(f"/proc/{p}").exists()]


@pi_bin
async def test_pi_rejected_prompt_is_a_failed_turn(tmp_path, fake_server, monkeypatch):
    """Pi refuses to prompt when the key variable is unset; that must end as a failed turn (not
    an exception, which would make the runtime re-buffer the prompt and restart forever)."""
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    monkeypatch.delenv("FLOCK_TEST_PI_KEY")
    await a.start(wd, None, pi_spec())
    await a.send("hello")
    evs = await until_turn_ended(a)
    assert [e.kind for e in evs] == ["turn_started", "turn_ended"]
    assert evs[-1].status == "error" and "No API key" in evs[-1].error
    assert requests_seen(fake_server) == []
    await a.stop()


@pi_bin
async def test_pi_thinking_level_reaches_the_endpoint(tmp_path, fake_server, monkeypatch):
    a, wd = make_adapter(tmp_path, fake_server, monkeypatch)
    a.model_config = PiModelConfig(**{**a.model_config.__dict__, "reasoning": "high"})
    await a.start(wd, None, pi_spec())
    assert a.argv[-2:] == ["--thinking", "high"]
    await a.send("hello")
    await until_turn_ended(a)
    await a.stop()
    assert requests_seen(fake_server)[0]["body"]["reasoning_effort"] == "high"
