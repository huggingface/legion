"""`harness: llm`: the in-process chat-completions loop against tests/fake_openai.py (no real
model calls), plus its pure helpers (text tool calls, compaction, repair) and spec checks."""

from __future__ import annotations

import asyncio
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
    repair,
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


def test_wire_merges_the_compaction_note_and_drops_flock_keys():
    msgs = [{"role": "system", "content": "S"}, {"role": "system", "content": "N", "flock_summary": True},
            {"role": "user", "content": "u"},
            {"role": "assistant", "content": None, "flock_from_text": "{..}", "tool_calls": []}]
    assert wire_messages(msgs) == [{"role": "system", "content": "S\n\nN"}, {"role": "user", "content": "u"},
                                   {"role": "assistant", "content": None, "tool_calls": []}]


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
    assert out[0] == sys_msg and out[1]["flock_summary"] and out[2]["role"] == "user"
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
    assert evs[2].model == "fake-model" and not evs[2].cumulative
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
        await a.send(f"msg{i} " + "x" * 600)
        assert (await turn(a))[-1].status == "completed"
    await a.stop()
    last = requests_seen(fake_server)[-1]["body"]["messages"]
    assert "earlier message(s) of this conversation were removed" in last[0]["content"]
    assert last[1]["content"].startswith("msg5") and len(last) == 2
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
    await asyncio.sleep(0.2)
    assert not Path(f"/proc/{pid}").exists() or "Z" in Path(f"/proc/{pid}/stat").read_text().split()[2]
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
