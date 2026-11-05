"""`harness: llm`: the in-process chat-completions loop against tests/fake_openai.py (no real
model calls), plus its pure helpers (text tool calls, compaction, repair) and spec checks."""

from __future__ import annotations

import asyncio
import itertools
import json
from pathlib import Path

import pytest

from flock.controller.prices import price_for
from flock.models import SwarmSpec, TokenUsage, TypeSpec
from flock.worker.adapters.base import AdapterEvent, ResumeError
from flock.worker.adapters.llm import (
    LLMAdapter,
    chat_usage,
    compact,
    estimate_tokens,
    harness_env,
    redact,
    repair,
    split_thinking,
    text_tool_calls,
    wire_messages,
)
from flock.worker.agents import INSTRUCTIONS_FILE, AgentRuntime
from flock.worker.events import Outbox

from . import fake_openai
from .worker_fakes import until

KEY_ENV = "FLOCK_TEST_LLM_KEY"


def llm_spec(base_url: str = "http://127.0.0.1:9/v1", *, models: dict | None = None,
             model: str = "gpu:fake-model", pricing: dict | None = None, **kw) -> TypeSpec:
    providers = {"gpu": {"type": "openai_compat", "base_url": base_url, "api_key_env": KEY_ENV,
                         "self_hosted": pricing is None, "max_retries": 3,
                         **({"pricing": pricing} if pricing else {})}}
    return SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b", "providers": providers, "models": models or {},
        "types": {"a": {"harness": "llm", "model": model, "idle_grace": "300ms", "turn_timeout": "20s",
                        **kw}}}).types["a"]


# --------------------------------------------------------------------------- pure helpers


@pytest.mark.parametrize(("content", "command"), [
    ('{"tool": "bash", "command": "ls -la"}', "ls -la"),
    ('```json\n{"tool": "bash", "command": "pwd"}\n```', "pwd"),
    ('{"name": "bash", "arguments": {"command": "echo hi"}}', "echo hi"),
    ('{"name": "bash", "arguments": "{\\"command\\": \\"date\\"}"}', "date"),
    ('<tool_call>{"name": "bash", "arguments": {"command": "id"}}</tool_call>', "id"),
    ('[bash(command="cat a.txt")]', "cat a.txt"),
    ('<|tool_call_start|>[bash(command="ls")]<|tool_call_end|>', "ls"),
])
def test_text_tool_calls_accepts_calls_written_as_text(content, command):
    [call] = text_tool_calls(content)
    assert call == {"name": "bash", "arguments": {"command": command}}


@pytest.mark.parametrize("content", [
    "Done. I ran ls.",
    'Use {"tool": "bash", "command": "ls"} to list files.',  # prose around the JSON: an answer
    '{"tool": "python", "command": "1"}',
    '{"tool": "bash", "command": ',
    "[1, 2, 3]",
    '[bash(__import__("os").system("x"))]',  # positional / non-literal: never evaluated
    'bash(command=open("x").read())',
    None,
])
def test_text_tool_calls_rejects_answers(content):
    assert text_tool_calls(content) == []


def test_chat_usage_splits_cached_tokens():
    u = chat_usage({"prompt_tokens": 100, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 3}})
    assert u == TokenUsage(input=60, output=7, cache_read=40, reasoning=3)
    assert chat_usage(None) == TokenUsage()


def test_wire_drops_flock_keys():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "N", "flock_summary": True},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "flock_from_text": "{..}", "tool_calls": []}]
    assert wire_messages(msgs) == [{"role": "system", "content": "S"}, {"role": "user", "content": "N"},
                                   {"role": "user", "content": "u"},
                                   {"role": "assistant", "content": None, "tool_calls": []}]


@pytest.mark.parametrize(("content", "thinking", "visible"), [
    ("<think>hmm</think>\n\nanswer", "hmm", "answer"),
    ("hmm</think>answer", "hmm", "answer"),  # the template opened the tag (LFM2)
    ("<think>cut off mid-thought", "cut off mid-thought", ""),
    ("plain answer", None, "plain answer"),
    ('hmm</think>{"tool": "bash", "command": "ls"}', "hmm", '{"tool": "bash", "command": "ls"}'),
    (None, None, None),
])
def test_split_thinking(content, thinking, visible):
    assert split_thinking(content) == (thinking, visible)


def test_repair_closes_dangling_tool_calls():
    call = {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "tool_calls": [call, {**call, "id": "c2"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
    out = repair(msgs)
    assert out[-1] == {"role": "tool", "tool_call_id": "c2", "content": "[interrupted: no result]"}
    assert repair(out) == out


def test_compaction_keeps_system_note_and_recent_messages(tmp_path):
    sys_msg = {"role": "system", "content": "S"}
    body = []
    for i in range(20):
        body += [{"role": "user", "content": f"q{i} " + "x" * 400},
                 {"role": "assistant", "content": f"a{i} " + "y" * 400}]
    msgs = [sys_msg, *body]
    assert compact(msgs[:3], 1000, tmp_path) is None  # fits: untouched
    out = compact(msgs, 1000, tmp_path)
    assert out[0] == sys_msg and out[1]["flock_summary"] and out[1]["role"] == "user"
    assert out[2]["role"] == "user" and [m["role"] for m in out].count("system") == 1
    assert out[-1] == body[-1] and estimate_tokens(out) <= 1000
    n = out[1]["flock_dropped"]
    assert n == len(msgs) - 1 - (len(out) - 2) and f"{n} earlier message(s)" in out[1]["content"]
    # a second compaction counts what the first dropped and keeps one note
    again = compact([*out, *body], 1000, tmp_path)
    assert sum(1 for m in again if m.get("flock_summary")) == 1 and again[1]["flock_dropped"] > n


def test_compaction_shortens_old_tool_outputs_within_one_turn(tmp_path):
    call = {"id": "c", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "go"}]
    for _ in range(4):
        msgs += [{"role": "assistant", "content": None, "tool_calls": [call]},
                 {"role": "tool", "tool_call_id": "c", "content": "z" * 3000}]
    out = compact(msgs, 1500, tmp_path)
    assert out[2]["content"] == "go"  # one user message: nothing to drop, outputs shortened
    assert len(out[4]["content"]) < 1100 and out[-1]["content"] == "z" * 3000


def test_harness_env_is_the_shared_allow_list():
    base = {"PATH": "/bin", "HOME": "/root", "FLOCK_TOKEN": "worker-secret", "HF_TOKEN": "hf",
            "UNRELATED": "x", "LC_ALL": "C"}
    env = harness_env({"FOO": "bar", KEY_ENV: "sk", "FLOCK_TOKEN": "x"}, base=base)
    assert env == {"PATH": "/bin", "HOME": "/root", "LC_ALL": "C", "FOO": "bar", KEY_ENV: "sk"}


# --------------------------------------------------------------------------- spec


def test_spec_llm_needs_an_openai_compat_provider():
    s = llm_spec(context_tokens=8000)
    assert s.harness.value == "llm" and s.context_tokens == 8000 and s.resolved_model.model_id == "fake-model"
    with pytest.raises(ValueError, match="needs a provider of type openai_compat"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                  "providers": {"an": {"type": "anthropic", "api_key_env": "K",
                                                       "pricing": {"*": [1, 1, 1]}}},
                                  "types": {"a": {"harness": "llm", "model": "an:m"}}})
    with pytest.raises(ValueError, match="harness 'llm' needs a provider-backed model"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                  "types": {"a": {"harness": "llm", "model": "gpt-5"}}})
    with pytest.raises(ValueError, match="context_tokens must be >= 256"):
        llm_spec(context_tokens=10)
    assert llm_spec(tool_choice="bash").tool_choice == "bash"
    with pytest.raises(ValueError, match="tool_choice"):
        llm_spec(tool_choice="paint")
    with pytest.raises(ValueError, match="for harness llm only"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                  "types": {"a": {"harness": "pi", "model": "m", "tool_choice": "required"}}})


def test_example_spec_validates():
    import yaml

    raw = yaml.safe_load((Path(__file__).parent.parent / "examples" / "llm-small-model.yaml").read_text())
    s = SwarmSpec.model_validate(raw)
    t = next(t for t in s.types.values() if t.harness.value == "llm")
    assert t.resolved_model.model_id == "LiquidAI/LFM2.5-2.6B"
    assert t.max_turns and t.max_budget_usd and t.context_tokens


def test_worker_factory_builds_llm_adapter(tmp_path):
    from datetime import UTC, datetime

    from flock.worker.main import WorkerConfig, default_adapter_factory

    cfg = WorkerConfig(url="u", token="t", worker_id="w", swarm="s", type="a",
                       deadline=datetime.now(UTC), bucket=None, work_root=tmp_path)
    a = default_adapter_factory(cfg)("a-0001", llm_spec(pricing={"*": [0.5, 1.5, 0.1]}))
    assert isinstance(a, LLMAdapter) and a.agent == "a-0001" and a.workdir == tmp_path / "a-0001"
    assert price_for("fake-model", a.prices).input == 0.5  # the pinned provider price
    assert INSTRUCTIONS_FILE["llm"] == "AGENTS.md"


# --------------------------------------------------------------------------- against the fake server


@pytest.fixture
def fake_server(tmp_path):
    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    srv.log = log
    srv.url = f"http://127.0.0.1:{srv.server_port}/v1"
    yield srv
    srv.shutdown()


def requests_seen(srv) -> list[dict]:
    return [json.loads(x) for x in srv.log.read_text().splitlines()] if srv.log.exists() else []


def make(tmp_path, srv, monkeypatch, **spec_kw) -> tuple[LLMAdapter, Path, TypeSpec]:
    monkeypatch.setenv("FLOCK_TOKEN", "worker-only-secret")
    monkeypatch.setenv("UNRELATED_SECRET", "nope")
    wd = tmp_path / "work" / "agents" / "llm-0001"
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "AGENTS.md").write_text("You are llm-0001. MARKER-AGENTS-MD\n")
    a = LLMAdapter(wd, home=tmp_path / "llm", backoff_s=0.01)
    return a, wd, llm_spec(srv.url, **spec_kw)


async def turn(a: LLMAdapter, timeout: float = 20) -> list[AdapterEvent]:
    got: list[AdapterEvent] = []

    async def collect():
        async for ev in a.events():
            got.append(ev)
            if ev.kind in ("turn_ended", "exited"):
                return

    await asyncio.wait_for(collect(), timeout)
    return got


def roles(evs: list[AdapterEvent]) -> list[str]:
    return [e.message["role"] for e in evs if e.kind == "message"]


async def test_plain_answer(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "sk-test"})
    await a.send("hello")
    evs = await turn(a)
    assert [e.kind for e in evs] == ["turn_started", "message", "usage", "message", "turn_ended"]
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 1
    assert evs[3].message["content"] == [{"type": "text", "text": "echo: hello"}]
    assert evs[3].message["stopReason"] == "stop" and evs[3].message["provider"] == "gpu"
    assert evs[2].model == "fake-model"
    assert evs[2].usage == TokenUsage(input=62, output=7, cache_read=40)  # 102 prompt, 40 cached
    [req] = requests_seen(fake_server)
    assert req["auth"] == "Bearer sk-test"
    body = req["body"]
    assert body["model"] == "fake-model" and body["temperature"] == 0 and "stream" not in body
    assert body["tools"][0]["function"]["name"] == "bash"
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert "MARKER-AGENTS-MD" in body["messages"][0]["content"]
    lines = [json.loads(x) for x in a.session_path(sid).read_text().splitlines()]
    assert [m["role"] for m in lines] == ["system", "user", "assistant"]
    assert a.session_files(sid) == [tmp_path / "llm" / "sessions" / "llm-0001" / f"{sid}.jsonl"]
    await a.stop()
    assert [e.kind async for e in a.events()] == ["exited"]


async def test_tool_call_round_trip_and_env(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "sk-test", "MSGBOARD_TOKEN": "mb-1"})
    await a.send("bash: env | sort > env.txt; echo wrote")
    evs = await turn(a)
    assert roles(evs) == ["user", "assistant", "toolResult", "assistant"]
    call = evs[[e.kind for e in evs].index("message", 2)].message
    assert call["stopReason"] == "toolUse" and call["content"][0]["type"] == "toolCall"
    res = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult")
    assert res["content"][0]["text"] == "wrote\n[exit code 0]" and res["isError"] is False
    assert sum(e.kind == "usage" for e in evs) == 2 and evs[-1].data["num_turns"] == 2
    env = dict(line.split("=", 1) for line in (wd / "env.txt").read_text().splitlines() if "=" in line)
    assert env["MSGBOARD_TOKEN"] == "mb-1"
    assert "FLOCK_TOKEN" not in env and "UNRELATED_SECRET" not in env and KEY_ENV not in env
    second = requests_seen(fake_server)[1]["body"]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assert second[3]["tool_call_id"] == second[2]["tool_calls"][0]["id"]
    lines = [json.loads(x) for x in a.session_path(sid).read_text().splitlines()]
    assert [m["role"] for m in lines] == ["system", "user", "assistant", "tool", "assistant"]
    await a.stop()


@pytest.mark.parametrize("prefix", ["json", "py"])
async def test_tool_call_written_as_text(tmp_path, fake_server, monkeypatch, prefix):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send(f"{prefix}: echo from-text > t.txt")
    evs = await turn(a)
    assert evs[-1].status == "completed"
    assert (wd / "t.txt").read_text() == "from-text\n"
    asst = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "assistant")
    assert asst.get("toolCallsFromText") and asst["content"][-1]["arguments"] == {"command": "echo from-text > t.txt"}
    # the next request shows the call in native form, flock_* keys stripped
    m = requests_seen(fake_server)[1]["body"]["messages"]
    assert m[2]["tool_calls"][0]["function"]["name"] == "bash" and "flock_from_text" not in m[2]
    await a.stop()


async def test_max_turns_stops_before_the_next_call(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, max_turns=1)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo hi")
    evs = await turn(a)
    end = evs[-1]
    assert end.status == "error" and end.data["reason"] == "max_turns" and end.data["num_turns"] == 1
    assert "max_turns exceeded (1)" in end.error
    assert len(requests_seen(fake_server)) == 1  # the tool ran, its result was never sent
    assert roles(evs) == ["user", "assistant", "toolResult"]
    await a.send("next")  # the transcript stays valid: tool result then a new user message
    assert (await turn(a))[-1].status == "completed"
    await a.stop()


async def test_max_budget_uses_the_price_table(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, max_budget_usd=0.0001,
                       pricing={"*": [10.0, 10.0, 10.0]})
    from flock.controller.prices import type_price_table

    a.prices = type_price_table(spec)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo hi")
    end = (await turn(a))[-1]
    assert end.status == "error" and end.data["reason"] == "max_budget_usd"  # 109 tokens * $10/M
    assert len(requests_seen(fake_server)) == 1
    await a.stop()


async def test_resume_from_jsonl(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("remember 42")
    await turn(a)
    await a.stop()
    (wd / "AGENTS.md").write_text("NEW-INSTRUCTIONS\n")
    b = LLMAdapter(wd, home=tmp_path / "llm", backoff_s=0.01)
    with pytest.raises(ResumeError, match="no file"):
        await b.start(wd, "00000000-dead", spec, env={KEY_ENV: "k"})
    assert await b.start(wd, sid, spec, env={KEY_ENV: "k"}) == sid
    await b.send("what was it?")
    await turn(b)
    await b.stop()
    msgs = requests_seen(fake_server)[-1]["body"]["messages"]
    assert [m["content"] for m in msgs[1:]] == ["remember 42", "echo: remember 42", "what was it?"]
    assert "NEW-INSTRUCTIONS" in msgs[0]["content"]  # system prompt rebuilt on start


async def test_compaction_during_a_session(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, context_tokens=600)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "k"})
    for i in range(6):
        await a.send(f"msg{i} realtok! " + "x" * 600)
        assert (await turn(a))[-1].status == "completed"
    await a.stop()
    last = requests_seen(fake_server)[-1]["body"]["messages"]
    assert [m["role"] for m in last] == ["system", "user", "user"]  # one system message, first
    assert "earlier message(s) of this conversation were removed" in last[1]["content"]
    assert last[2]["content"].startswith("msg5")
    lines = [json.loads(x) for x in a.session_path(sid).read_text().splitlines()]
    assert lines[1]["flock_summary"] and lines[1]["flock_dropped"] == 10


async def test_steering_while_busy_is_the_next_user_message(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: sleep 0.5; echo slept")
    await asyncio.sleep(0.2)
    assert a.in_turn
    await a.send("also this")
    evs = await turn(a)
    assert sum(e.kind == "turn_started" for e in evs) == 1
    texts = [(e.message["role"], (e.message["content"] or [{}])[-1].get("text")) for e in evs if e.kind == "message"]
    assert texts == [("user", "bash: sleep 0.5; echo slept"), ("assistant", None),
                     ("toolResult", "slept\n[exit code 0]"), ("user", "also this"),
                     ("assistant", "echo: also this")]
    assert [m["role"] for m in requests_seen(fake_server)[1]["body"]["messages"]][-2:] == ["tool", "user"]
    await a.stop()


async def test_429_is_retried(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("429! please")
    evs = await turn(a)
    assert evs[-1].status == "completed" and len(requests_seen(fake_server)) == 2
    await a.send("fail! now")  # a 4xx is not retried: the turn fails
    end = (await turn(a))[-1]
    assert end.status == "error" and end.data["reason"] == "api_error" and "HTTP 400" in end.error
    assert len(requests_seen(fake_server)) == 3
    await a.stop()


async def test_retries_are_bounded(tmp_path, monkeypatch):
    import httpx

    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(503, text="overloaded")

    spec = llm_spec("http://fake/v1")
    a = LLMAdapter(tmp_path / "wd", home=tmp_path / "llm", backoff_s=0.001,
                   transport=httpx.MockTransport(handler))
    await a.start(tmp_path / "wd", None, spec, env={})
    await a.send("hi")
    end = (await turn(a))[-1]
    assert end.status == "error" and "gave up after 4 attempts" in end.error and len(calls) == 4
    assert "authorization" not in calls[0].headers  # no key in env: none sent
    await a.stop()


async def test_extra_is_sent_and_reasoning_ignored(tmp_path, fake_server, monkeypatch):
    spec_kw = {"models": {"small": {"model": "gpu:fake-model", "reasoning": "high",
                                    "extra": {"chat_template_kwargs": {"enable_thinking": False},
                                              "max_tokens": 256, "stream": True}}},
               "model": "small"}
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, **spec_kw)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("hi")
    await turn(a)
    await a.stop()
    body = requests_seen(fake_server)[0]["body"]
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["max_tokens"] == 256
    assert "stream" not in body and "reasoning" not in body and "reasoning_effort" not in body


async def test_stop_mid_tool_interrupts_and_kills_the_tool(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: sleep 30 & echo $! > pid; wait")
    await until(lambda: (wd / "pid").exists() and (wd / "pid").read_text().strip(), timeout=10)
    pid = int((wd / "pid").read_text())
    await a.stop()
    evs = [e async for e in a.events()]
    assert [e.status for e in evs if e.kind == "turn_ended"] == ["interrupted"] and evs[-1].kind == "exited"
    from flock.controller.local import _alive

    await until(lambda: not _alive(pid), timeout=10)  # killed; a loaded host reaps it late
    lines = [json.loads(x) for x in a.session_path(sid).read_text().splitlines()]
    assert lines[-1]["role"] == "tool" and "interrupted" in lines[-1]["content"]  # repaired
    with pytest.raises(RuntimeError):
        await a.send("after stop")


async def test_runtime_accounts_usage_and_reports_limits(tmp_path, fake_server, monkeypatch):
    monkeypatch.setenv(KEY_ENV, "sk-from-worker")
    spec = llm_spec(fake_server.url, max_turns=1, system_prompt="be terse")
    a = LLMAdapter(tmp_path / "work" / "llm-0001", agent="llm-0001", home=tmp_path / "llm")
    rt = AgentRuntime("llm-0001", "llm", spec, swarm="sw", work_root=tmp_path / "work",
                      adapter=a, outbox=Outbox())
    await rt.setup(restore=False)
    await rt.prompt("hello")
    await until(lambda: rt.state.turns == 1, timeout=20)
    await rt.prompt("bash: echo hi")
    await until(lambda: rt.state.turns == 2, timeout=20)
    await rt.stop()
    assert (tmp_path / "work" / "llm-0001" / "AGENTS.md").read_text() == "be terse"
    u = rt.state.tokens["fake-model"]
    assert (u.output, u.cache_read) == (14, 80)  # two model calls
    ended = [r["data"] for r in map(json.loads, rt.log.path.read_text().splitlines())
             if r.get("customType") == "turn_ended"]
    assert [e["status"] for e in ended] == ["completed", "error"] and ended[1]["reason"] == "max_turns"
    assert requests_seen(fake_server)[0]["auth"] == "Bearer sk-from-worker"
    assert rt.harness_files() == a.session_files(rt.state.session_id) != []


async def test_bash_timeout_and_truncation(tmp_path):
    a = LLMAdapter(tmp_path / "wd", home=tmp_path / "llm")
    out, err = await a._bash("sleep 5", 0.2)
    assert err and "timed out after 0.2s" in out
    out, err = await a._bash("head -c 40000 /dev/zero | tr '\\0' a; exit 3", None)
    assert err and out.endswith("[exit code 3]") and "chars truncated" in out and len(out) < 16_200


@pytest.mark.parametrize(("where", "wire"), [
    ({"tool_choice": "required", "done_tool": False}, "required"),
    ({"tool_choice": "bash"}, {"type": "function", "function": {"name": "bash"}}),
    ({"models": {"m": {"model": "gpu:fake-model", "extra": {"tool_choice": "required"}}}, "model": "m",
      "done_tool": False}, "required"),
])
async def test_forced_tool_choice_is_one_action_per_prompt(tmp_path, fake_server, monkeypatch, where, wire):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, **where)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo painted > px.txt")
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 1
    assert roles(evs) == ["user", "assistant", "toolResult"]
    assert (wd / "px.txt").read_text() == "painted\n"
    [req] = requests_seen(fake_server)
    assert req["body"]["tool_choice"] == wire
    await a.send("again")  # the next prompt continues the transcript, one action again
    assert (await turn(a))[-1].status == "completed" and len(requests_seen(fake_server)) == 2
    await a.stop()


async def test_response_format_and_no_thinking_pass_through_extra(tmp_path, fake_server, monkeypatch):
    schema = {"type": "object", "properties": {"tool": {"const": "bash"}, "command": {"type": "string"}},
              "required": ["tool", "command"]}
    extra = {"response_format": {"type": "json_schema", "json_schema": {"name": "call", "schema": schema}},
             "chat_template_kwargs": {"enable_thinking": False}}
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, model="m",
                       models={"m": {"model": "gpu:fake-model", "extra": extra}})
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("paint")
    evs = await turn(a)
    body = requests_seen(fake_server)[0]["body"]
    assert body["response_format"] == extra["response_format"]  # nested objects intact
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    res = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult")
    assert res["content"][0]["text"] == "rf\n[exit code 0]"  # the schema'd JSON ran as the call
    await a.stop()


@pytest.mark.parametrize("trigger", ["think!", "think2!"])
async def test_thinking_is_shown_not_resent(tmp_path, fake_server, monkeypatch, trigger):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    sid = await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send(f"{trigger} hi")
    evs = await turn(a)
    asst = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "assistant")
    assert asst["content"] == [{"type": "thinking", "thinking": "pondering"},
                               {"type": "text", "text": f"echo: {trigger} hi"}]
    assert next(e for e in evs if e.kind == "usage").usage.output == 7  # tokens still counted
    await a.send("next")
    await turn(a)
    await a.stop()
    sent = requests_seen(fake_server)[1]["body"]["messages"]
    assert sent[2] == {"role": "assistant", "content": f"echo: {trigger} hi"}
    assert "pondering" not in a.session_path(sid).read_text()


async def test_required_with_done_runs_several_steps(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required")
    sid = await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo x >> px.txt done after 3")
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 4  # 3 bash + done
    assert (wd / "px.txt").read_text().count("x") == 3
    final = [e.message for e in evs if e.kind == "message"][-1]
    assert final["role"] == "assistant" and final["content"] == [{"type": "text", "text": "painted 3"}]
    body = requests_seen(fake_server)[0]["body"]
    assert [t["function"]["name"] for t in body["tools"]] == ["bash", "done"]
    assert "call the `done` tool" in body["messages"][0]["content"]
    lines = [json.loads(x) for x in a.session_path(sid).read_text().splitlines()]
    assert [m["role"] for m in lines[-3:]] == ["assistant", "tool", "assistant"]  # done -> ok -> summary
    await a.send("next, done after 0")  # the transcript stays valid for the next prompt
    end = (await turn(a))[-1]
    assert end.status == "completed" and end.data["num_turns"] == 1
    await a.stop()


async def test_required_without_done_hits_max_turns(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required", max_turns=3)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo again")  # the model never calls done
    end = (await turn(a))[-1]
    assert end.status == "error" and end.data["reason"] == "max_turns" and end.data["num_turns"] == 3
    await a.stop()


async def test_done_tool_defaults_and_text_form(tmp_path, fake_server, monkeypatch):
    a, _, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="bash")
    a._spec = spec
    assert not a.done_enabled  # named bash: one action, unchanged
    a._spec = llm_spec(fake_server.url, tool_choice="bash", done_tool=True)
    assert a.done_enabled
    a._spec = llm_spec(fake_server.url)
    assert not a.done_enabled
    assert text_tool_calls('{"tool": "done", "summary": "ok"}', "done") == [
        {"name": "done", "arguments": {"summary": "ok"}}]
    with pytest.raises(ValueError, match="for harness llm only"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                  "types": {"a": {"harness": "pi", "model": "m", "done_tool": True}}})


async def test_forced_with_done_has_a_default_call_cap(tmp_path, fake_server, monkeypatch):
    import flock.worker.adapters.llm as llm_mod

    monkeypatch.setattr(llm_mod, "FORCED_DEFAULT_MAX_TURNS", 2)
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required")
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("never done")
    end = (await turn(a))[-1]
    assert end.data["reason"] == "max_turns" and end.data["num_turns"] == 2 and "(2)" in end.error
    await a.stop()


async def test_non_2xx_bodies_are_reported_redacted(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, spec, env={KEY_ENV: "sk-test-secret-key-123"})
    await a.send("fail! please")
    evs = await turn(a)
    [err] = [e for e in evs if e.kind == "error"]
    assert err.data == {"status": 400, "attempt": 1, "retrying": False}
    assert err.error.startswith("HTTP 400 from http://127.0.0.1") and "bad request" in err.error
    assert "sk-test-secret-key-123" not in err.error and "[redacted]" in err.error
    body = err.error.split(": ", 1)[1]
    assert len(body) == 500  # the first 500 chars of the body
    end = evs[-1]
    assert end.status == "error" and end.data["reason"] == "api_error"  # handling unchanged
    assert "sk-test-secret-key-123" not in end.error

    await a.send("503! now")  # retried as before (3 retries), one error event per response
    evs = await turn(a)
    errs = [e for e in evs if e.kind == "error"]
    assert [e.data["retrying"] for e in errs] == [True, True, True, False]
    assert all(e.data["status"] == 503 and "engine overloaded" in e.error for e in errs)
    assert "gave up after 4 attempts" in evs[-1].error
    await a.stop()


def test_redact():
    assert redact("Authorization: Bearer abc.def-123 ok", None) == "Authorization: Bearer [redacted] ok"
    assert redact("key hf_abcdefghijk and sk-abcdefghij12", None) == "key [redacted] and [redacted]"
    assert redact("my-own-key-value appears", "my-own-key-value") == "[redacted] appears"


def test_warn_extra_skips_the_llm_harness(capsys):
    from flock import ops

    spec = SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b",
        "providers": {"p": {"type": "openai_compat", "base_url": "http://x/v1", "api_key_env": "K",
                            "self_hosted": True}},
        "models": {"m": {"model": "p:m", "extra": {"chat_template_kwargs": {"enable_thinking": False}}}},
        "types": {"pix": {"harness": "llm", "model": "m"}, "cx": {"harness": "pi", "model": "m"}}})
    err = "\n".join(ops.extra_warnings(spec))
    assert "type cx: `extra`" in err and "pix" not in err


# --------------------------------------------------------------------------- context bounding (pixel run 3)

STEP = ('n=$(( $(cat c 2>/dev/null || echo 0) + 1 )); echo $n > c; '
        'python3 -c "import json; print(json.dumps({\'ok\': True, \'pixels\': [[i, i, \'#a0b0c0\'] '
        'for i in range(30)]}))"; [ $n -ge 12 ] && echo DONE-NOW; true')


def no_repeat_guard(monkeypatch) -> None:
    import flock.worker.adapters.llm as llm_mod

    monkeypatch.setattr(llm_mod, "REPEAT_NUDGE_AT", 1000)
    monkeypatch.setattr(llm_mod, "REPEAT_SKIP_AFTER", 1000)


def test_compaction_drops_old_steps_inside_one_turn(tmp_path):
    def call(i):
        return {"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "paint"}]
    for i in range(10):
        msgs += [call(i), {"role": "tool", "tool_call_id": f"c{i}", "content": "p" * 600}]
    out = compact(msgs, 1000, tmp_path)
    assert [m["role"] for m in out[:3]] == ["system", "user", "user"] and out[2]["content"] == "paint"
    ids = [m["tool_call_id"] for m in out if m["role"] == "tool"]
    assert ids == [f"c{i}" for i in range(10 - len(ids), 10)] and len(ids) >= 2  # newest steps
    assert all(m["tool_calls"][0]["id"] in ids for m in out if m["role"] == "assistant")  # never split
    assert estimate_tokens(out) <= 1000 and out[1]["flock_dropped"] == 2 * (10 - len(ids))
    agg = compact(msgs, 100_000, tmp_path, aggressive=True)  # after a context error: last 2 steps
    assert [m.get("tool_call_id") for m in agg if m["role"] == "tool"] == ["c8", "c9"]
    assert compact(msgs, 100_000, tmp_path) is None and compact(msgs, 10, tmp_path, over=False) is None


def tokens_sent(srv) -> list[int]:
    return [fake_openai.prompt_tokens(r["body"]) for r in requests_seen(srv)]


async def test_growing_turn_is_compacted_before_the_bound(tmp_path, fake_server, monkeypatch):
    """Run 3: ~600-char JSON tool results at ~2 chars/token; a chars/4 estimate let a 3000-token
    budget reach 7.9k and overflow 8192. Projection from the server's count keeps it bounded."""
    no_repeat_guard(monkeypatch)  # STEP is one stateful command repeated on purpose
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required", context_tokens=2500,
                       max_turns=20)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send(f"realtok! ctxlimit=3000 bash: {STEP}")  # each step adds ~570 tokens
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 13  # 12 steps + done
    assert not [e for e in evs if e.kind == "error"]  # never a 400
    sent = tokens_sent(fake_server)
    assert max(sent) <= 2500 * 1.05 and sent[-1] < sent[-3] + 400  # bounded, not monotonic growth
    assert any("earlier message(s)" in m["content"] for r in requests_seen(fake_server)[-1:]
               for m in r["body"]["messages"] if m["role"] == "user")
    assert 1.5 < a.chars_per_token < 2.5  # calibrated toward the server's ~2 chars/token
    await a.stop()


async def test_context_error_compacts_and_retries_once(tmp_path, fake_server, monkeypatch):
    no_repeat_guard(monkeypatch)  # STEP is one stateful command repeated on purpose
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required", max_turns=30,
                       context_tokens=100_000)  # no proactive compaction: the server is the bound
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send(f"realtok! ctxlimit=2200 bash: {STEP}")
    evs = await turn(a)
    errs = [e for e in evs if e.kind == "error"]
    assert errs and all(e.data["status"] == 400 and "maximum context length" in e.error for e in errs)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 13  # each 400 retried once
    statuses = [fake_openai.prompt_tokens(r["body"]) > 2200 for r in requests_seen(fake_server)]
    assert not any(x and y for x, y in itertools.pairwise(statuses))  # retry fits
    await a.stop()


async def test_context_error_that_cannot_shrink_ends_with_context_overflow(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("realtok! ctxlimit=50 hello")  # system + prompt alone exceed the server's limit
    end = (await turn(a))[-1]
    assert end.status == "error" and end.data["reason"] == "context_overflow"
    assert len(requests_seen(fake_server)) == 1  # nothing to drop: no retry
    await a.stop()


async def test_calibration_and_settings(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, chars_per_token=3.0, tool_output_chars=300)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    assert a.chars_per_token == 3.0 and a.projected_tokens() > 0
    await a.send("realtok! bash: python3 -c \"print('y' * 1000)\"")
    evs = await turn(a)
    assert 2.0 < a.chars_per_token < 3.0  # moved toward the server's ~2
    res = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult")
    text = res["content"][0]["text"]
    assert "chars truncated" in text and len(text) < 400  # tool_output_chars bounds the stored result
    tokens, _ = a._last_prompt
    assert a.projected_tokens() >= tokens  # projection starts from the server's count
    await a.stop()
    with pytest.raises(ValueError, match="for harness llm only"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                  "types": {"a": {"harness": "pi", "model": "m", "chars_per_token": 2}}})
    with pytest.raises(ValueError, match="tool_output_chars must be >= 200"):
        llm_spec(tool_output_chars=10)


async def test_forced_empty_reply_gets_one_nudge(tmp_path, fake_server, monkeypatch):
    """A provider that drops the tool call (an empty assistant message) under a forcing
    tool_choice with `done`: one user nudge, counted as a model call; it then calls done."""
    from flock.worker.adapters.llm import NUDGE

    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required", max_turns=12)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("empty! please")
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 2  # empty + done
    users = [e.message for e in evs if e.kind == "message" and e.message["role"] == "user"]
    assert users[-1]["content"] == [{"type": "text", "text": NUDGE}]
    final = [e.message for e in evs if e.kind == "message"][-1]
    assert final["content"] == [{"type": "text", "text": "nudged"}]
    # a script that drops the call after a tool result (what Qwen3.5-9B did after a verdict)
    await a.send('script: [["bash", {"command": "echo v"}], ["EMPTY", {}]]')
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 3  # bash, empty, done
    # still empty after the nudge: the turn ends as before (no second nudge)
    await a.send("empty-always! now")
    evs = await turn(a)
    assert evs[-1].status == "completed" and evs[-1].data["num_turns"] == 2
    assert sum(e.kind == "message" and e.message["role"] == "user"
               and e.message["content"][0]["text"] == NUDGE for e in evs) == 1
    await a.stop()


async def test_forced_empty_reply_nudge_counts_toward_max_turns(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required", max_turns=1)
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("empty! please")
    end = (await turn(a))[-1]
    assert end.status == "error" and end.data["reason"] == "max_turns" and end.data["num_turns"] == 1
    await a.stop()


async def test_repeated_identical_calls_are_nudged_then_not_run(tmp_path, fake_server, monkeypatch):
    """A model stuck repeating one call: from the third identical call in a turn its result starts
    with a nudge; after the fifth the call is not run. `repeated_calls` counts them."""
    from flock.worker.adapters.llm import REPEAT_NUDGE

    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required")
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo x >> rep.txt done after 7")  # the same bash call 7 times, then done
    evs = await turn(a)
    end = evs[-1]
    assert end.status == "completed" and end.data["num_turns"] == 8
    assert end.data["repeated_calls"] == 5  # calls 3 to 7
    results = [e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult"]
    texts = [r["content"][0]["text"] for r in results[:7]]
    assert [t.startswith(REPEAT_NUDGE) for t in texts] == [False, False, True, True, True, True, True]
    assert texts[2] == f"{REPEAT_NUDGE}\n[exit code 0]" and not results[2]["isError"]
    assert "[not run: the same call was made 5 times already" in texts[5] and results[5]["isError"]
    assert (wd / "rep.txt").read_text().count("x") == 5  # the 6th and 7th were not run
    sent = requests_seen(fake_server)[3]["body"]["messages"][-1]
    assert sent["role"] == "tool" and sent["content"].startswith(REPEAT_NUDGE)
    await a.send("bash: echo y >> rep.txt done after 2")  # a new turn counts afresh
    end = (await turn(a))[-1]
    assert end.status == "completed" and "repeated_calls" not in end.data
    await a.stop()


async def test_two_identical_calls_are_not_nudged(tmp_path, fake_server, monkeypatch):
    a, wd, spec = make(tmp_path, fake_server, monkeypatch, tool_choice="required")
    await a.start(wd, None, spec, env={KEY_ENV: "k"})
    await a.send("bash: echo x >> r.txt\ndone after 2")
    end = (await turn(a))[-1]
    assert "repeated_calls" not in end.data
    await a.send("bash: echo y >> r.txt\ndone after 3")  # a third identical call: nudged
    evs = await turn(a)
    assert evs[-1].data["repeated_calls"] == 1
    assert (wd / "r.txt").read_text() == "x\nx\ny\ny\ny\n"
    await a.stop()


async def test_loop_turn_at_max_turns_sets_an_unfinished_summary(tmp_path, fake_server, monkeypatch):
    """The worker loop with the llm harness: a round that stops at max_turns (no `done`) makes
    the next round's `{last_summary}` say so instead of repeating the previous summary."""
    import uuid
    from datetime import UTC, datetime

    from flock.models import LoopMeta, LoopStartCommand

    monkeypatch.setenv(KEY_ENV, "k")
    spec = llm_spec(fake_server.url, tool_choice="required", max_turns=2)
    a = LLMAdapter(tmp_path / "work" / "llm-0001", agent="llm-0001", home=tmp_path / "llm")
    rt = AgentRuntime("llm-0001", "llm", spec, swarm="sw", work_root=tmp_path / "work",
                      adapter=a, outbox=Outbox())
    await rt.setup(restore=False)
    rt.state.last_summary = "stale summary from hours ago"
    rt.start_loop(LoopStartCommand(
        agent="llm-0001", text="bash: echo r{round}\nlast: {last_summary}", max_rounds=2,
        run=uuid.uuid4().hex[:12],
        meta=LoopMeta(kind="loop", type="llm", round=0, why="turn_end", started_at=datetime.now(UTC))))
    await until(lambda: rt.state.turns == 2, timeout=20)
    await rt.stop()
    prompts = [m["content"] for r in requests_seen(fake_server) for m in r["body"]["messages"]
               if m["role"] == "user" and m["content"].startswith("bash: echo r")]
    assert prompts[0] == "bash: echo r1\nlast: stale summary from hours ago"
    assert prompts[-1] == ("bash: echo r2\nlast: previous turn ended (max_turns) after 2 tool call(s) "
                           "without finishing (last calls: bash, bash)")
