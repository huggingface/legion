"""Controller side of `providers:`: job secrets, pinned pricing, status, preflight."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from flock.controller.api import create_app
from flock.models import AgentHeartbeat, SwarmSpec, TokenUsage, WorkerHeartbeat

from .conftest import TOKEN

PROVIDERS = {
    "hf": {"type": "openai_compat", "base_url": "https://router.example/v1", "api_key_env": "HF_TOKEN",
           "pricing": {"Qwen/Qwen3.8-27B": [1.0, 2.0, 0.5]}},
    "gpu1": {"type": "openai_compat", "base_url": "http://qwen-job:8000/v1",
             "api_key_env": "VLLM_API_KEY", "self_hosted": True, "max_working": 2},
    "anth": {"type": "anthropic", "base_url": "https://vllm-anthropic.example",
             "api_key_env": "ANTH_KEY", "pricing": {"claude-x": [3, 15, 0.3]}},
}


def pspec(providers=None, **types) -> SwarmSpec:
    types = types or {
        "worker": {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B", "replicas": 2,
                   "agents_per_worker": 2, "secrets": ["HF_TOKEN"]},
        "local": {"harness": "codex", "model": "gpu1:Qwen/Small", "replicas": 1},
        "native": {"harness": "claude", "model": "claude-opus-5-5", "replicas": 1},
    }
    return SwarmSpec.model_validate({"swarm": "p", "namespace": "ns", "bucket": "hf://buckets/ns/b",
                                     "providers": providers or PROVIDERS, "types": types})


def hb(clock, wid, agents) -> dict:
    return WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=24),
                           agents=agents).model_dump(mode="json")


async def put(client, spec) -> httpx.Response:
    return await client.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))


async def test_provider_key_is_added_to_job_secrets(client, jobs, settings, tick):
    settings.secret_source = {"HF_TOKEN": "hf-x", "VLLM_API_KEY": "v-x", "ANTHROPIC_API_KEY": "a"}
    assert (await put(client, pspec())).status_code == 200
    tick()
    by_type = {x["labels"]["type"]: x for x in jobs.launches}
    assert by_type["worker"]["secrets"] == {"FLOCK_TOKEN": TOKEN, "HF_TOKEN": "hf-x"}  # deduped
    assert by_type["local"]["secrets"] == {"FLOCK_TOKEN": TOKEN, "VLLM_API_KEY": "v-x"}
    assert by_type["native"]["secrets"] == {"FLOCK_TOKEN": TOKEN}
    assert by_type["worker"]["spec"].resolved_model.provider.api_key_env == "HF_TOKEN"


async def test_invalid_spec_is_rejected_at_apply(client):
    raw = pspec().model_dump(mode="json")
    raw["types"]["worker"]["model"] = "gpu9:Qwen/X"
    r = await client.put("/swarms/p", json=raw)
    assert r.status_code == 422
    assert "types.worker.model" in r.text and "unknown provider 'gpu9'" in r.text


async def test_cost_uses_pinned_provider_price_and_does_not_reprice_history(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-worker-w0000"
    mtok = TokenUsage(input=1_000_000, output=1_000_000, cache_read=1_000_000)
    agents = [AgentHeartbeat(name="worker-0000", state="working", turn_id="t-1",
                             tokens={"Qwen/Qwen3.8-27B": mtok})]
    assert (await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))).status_code == 200
    a = (await client.get("/agents/worker-0000")).json()
    assert a["cost_usd"] == 3.5 and a["provider"] == "hf" and a["model"] == "hf:Qwen/Qwen3.8-27B"

    # Re-apply with a new price: only tokens after the change are priced at it.
    providers = {**PROVIDERS, "hf": {**PROVIDERS["hf"], "pricing": {"Qwen/Qwen3.8-27B": [10, 20, 5]}}}
    assert (await put(client, pspec(providers))).status_code == 200
    agents[0].tokens = {"Qwen/Qwen3.8-27B": mtok + TokenUsage(input=1_000_000)}
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    assert (await client.get("/agents/worker-0000")).json()["cost_usd"] == 13.5


async def test_self_hosted_is_free_and_status_shows_providers(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-local-w0000"
    agents = [AgentHeartbeat(name="local-0000", state="working", turn_id="t-1",
                             tokens={"Qwen/Small": TokenUsage(input=5_000_000, output=1_000_000)})]
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    a = (await client.get("/agents/local-0000")).json()
    assert a["cost_usd"] == 0.0 and a["provider"] == "gpu1"
    assert (await client.get("/agents/native-0000")).json()["provider"] is None
    st = (await client.get("/swarms/p")).json()
    assert st["spec"]["providers"]["gpu1"]["pricing"] == {"Qwen/Small": [0.0, 0.0, 0.0]}  # pinned
    gpu1 = st["status"]["providers"]["gpu1"]
    assert gpu1 == {"name": "gpu1", "type": "openai_compat", "self_hosted": True,
                    "types": ["local"], "working": 1, "max_working": 2, "zero_usage_agents": 0}
    assert st["status"]["providers"]["anth"]["types"] == []
    page = (await client.get("/")).text
    assert "gpu1:Qwen/Small" in page and "working 1 / max 2" in page


# --------------------------------------------------------------------------- preflight


def _sse(events: list[dict], named: bool = False):
    """A streamed reply; `named` adds `event:` lines (Anthropic style; the HF router sends none)."""
    def lines():
        for e in events:
            yield (f"event: {e['type']}\n" if named else "") + f"data: {json.dumps(e)}\n\n"
    return StreamingResponse(lines(), media_type="text/event-stream")


def fake_provider(tool_call: bool = True, status: int = 200, follow_up_fails: bool = False,
                  zero_usage: bool = False, text_call: bool = False) -> tuple[FastAPI, list[dict]]:
    """A tiny streaming OpenAI-compatible (Responses) and Anthropic (Messages) server.
    `follow_up_fails` answers the turn carrying the tool result with `response.failed` inside
    HTTP 200 (or an Anthropic `error` event), as the HF router did on its auto provider."""
    fake = FastAPI()
    seen: list[dict] = []
    usage = {"input_tokens": 0 if zero_usage else 12, "output_tokens": 0 if zero_usage else 5}

    @fake.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        seen.append({"path": "responses", "auth": request.headers.get("authorization"), **body})
        if status != 200:
            return JSONResponse({"error": {"message": "model not found"}}, status_code=status)
        follow_up = any(i.get("type") == "function_call_output" for i in body["input"])
        if follow_up and follow_up_fails:
            return _sse([{"type": "response.failed", "response": {
                "status": "failed", "error": {"code": "server_error", "message": "400 status code (no body)"}}}])
        if follow_up:
            out = [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "It is noon UTC."}]}]
        elif tool_call:
            out = [{"type": "reasoning", "id": "rs_1", "summary": [],
                    "content": [{"type": "reasoning_text", "text": "call it"}]},
                   {"type": "function_call", "id": "fc_1", "call_id": "c1", "name": "get_time",
                    "arguments": '{"timezone": "UTC"}'}]
        else:
            out = [{"type": "message", "role": "assistant",
                    "content": [{"type": "output_text", "text": "It is noon."}]}]
        events = [{"type": "response.created", "response": {"status": "in_progress", "output": []}}]
        events += [{"type": "response.output_item.done", "item": it} for it in out]
        events.append({"type": "response.completed",
                       "response": {"id": "r1", "status": "completed", "output": out, "usage": usage}})
        return _sse(events)

    @fake.post("/v1/chat/completions")
    async def chat(request: Request):
        """Non-streamed Chat Completions (the llm harness). `text_call`: the model writes the
        call as JSON text instead of `tool_calls`, as small models do."""
        body = await request.json()
        seen.append({"path": "chat", "auth": request.headers.get("authorization"), **body})
        if status != 200:
            return JSONResponse({"error": {"message": "model not found"}}, status_code=status)
        follow_up = body["messages"][-1]["role"] == "tool"
        if follow_up and follow_up_fails:
            return JSONResponse({"error": {"message": "bad tool message"}}, status_code=400)
        msg: dict = {"role": "assistant", "content": "It is noon UTC."}
        if tool_call and not follow_up:
            msg = ({"role": "assistant", "content": 'hmm</think>{"tool": "get_time", "timezone": "UTC"}'}
                   if text_call
                   else {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "call_1", "type": "function",
                        "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'}}]})
        u = {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"]}
        return JSONResponse({"id": "c1", "object": "chat.completion", "usage": u,
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})

    @fake.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        seen.append({"path": "messages", "key": request.headers.get("x-api-key"),
                     "auth": request.headers.get("authorization"), **body})
        last = body["messages"][-1]["content"]
        follow_up = isinstance(last, list) and any(b.get("type") == "tool_result" for b in last)
        if follow_up and follow_up_fails:
            return _sse([{"type": "message_start", "message": {"usage": {"input_tokens": 1}}},
                         {"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}],
                        named=True)
        ev = [{"type": "message_start", "message": {"id": "m1", "role": "assistant", "content": [],
                                                   "usage": {"input_tokens": usage["input_tokens"]}}}]
        if tool_call and not follow_up:
            ev += [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "ok"}},
                   {"type": "content_block_stop", "index": 0},
                   {"type": "content_block_start", "index": 1,
                    "content_block": {"type": "tool_use", "id": "t1", "name": "get_time", "input": {}}},
                   {"type": "content_block_delta", "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"timezone": '}},
                   {"type": "content_block_delta", "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '"UTC"}'}},
                   {"type": "content_block_stop", "index": 1}]
        else:
            ev += [{"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "noon"}},
                   {"type": "content_block_stop", "index": 0}]
        ev += [{"type": "message_delta", "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": usage["output_tokens"]}}, {"type": "message_stop"}]
        return _sse(ev, named=True)

    return fake, seen


def preflight_providers():
    return {
        "hf": {**PROVIDERS["hf"], "base_url": "http://fake/v1"},
        "gpu1": {**PROVIDERS["gpu1"], "base_url": "http://fake/v1"},
        "anth": {**PROVIDERS["anth"], "base_url": "http://fake"},
    }


def preflight_spec(**extra_types) -> SwarmSpec:
    return pspec(preflight_providers(), **{
        "worker": {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B"},
        "worker2": {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B"},
        "local": {"harness": "codex", "model": "gpu1:Qwen/Small", "reasoning": "off"},
        "writer": {"harness": "claude", "model": "anth:claude-x"},
        "native": {"harness": "claude", "model": "claude-opus-5-5"},
        **extra_types,
    })


@pytest.fixture
def preflight_client(store, clock):
    async def make(fake: FastAPI, secrets: dict):
        app = create_app(store, TOKEN, clock=clock, secret_source=secrets,
                         preflight_transport=httpx.ASGITransport(app=fake))
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"})
    return make


SECRETS = {"HF_TOKEN": "hf-secret", "VLLM_API_KEY": "v", "ANTH_KEY": "ak"}


async def test_preflight_tool_call_seen(preflight_client):
    fake, seen = fake_provider()
    async with await preflight_client(fake, SECRETS) as c:
        r = await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))
    assert r.status_code == 200, r.text
    rep = r.json()
    assert rep["ok"] is True
    by_model = {x["model"]: x for x in rep["results"]}
    assert set(by_model) == {"hf:Qwen/Qwen3.8-27B", "gpu1:Qwen/Small", "anth:claude-x"}  # native skipped
    assert by_model["hf:Qwen/Qwen3.8-27B"]["types"] == ["worker", "worker2"]  # one request, two types
    assert all(x["tool_call_seen"] and x["follow_up_ok"] and x["usage_reported"]
               and x["latency_ms"] is not None and x["warning"] is None for x in rep["results"])
    assert by_model["anth:claude-x"]["api"] == "messages"
    assert len(seen) == 6  # two turns per model
    first = {(x["path"], x["model"]): x for x in seen[::-1]}  # earliest request per model wins
    hf = first[("responses", "Qwen/Qwen3.8-27B")]
    assert hf["auth"] == "Bearer hf-secret" and hf["tools"][0]["name"] == "get_time"
    assert hf["stream"] is True and hf["store"] is False and "reasoning" not in hf
    assert first[("responses", "Qwen/Small")]["reasoning"] == {"effort": "none"}
    assert first[("messages", "claude-x")]["key"] == "ak" and first[("messages", "claude-x")]["stream"]
    # turn two carries the model's own output and the tool result, as a harness would
    follow = [x for x in seen if x["path"] == "responses" and x["model"] == "Qwen/Qwen3.8-27B"][-1]
    assert [i["type"] for i in follow["input"]] == ["message", "reasoning", "function_call",
                                                    "function_call_output"]
    assert follow["input"][-1]["call_id"] == "c1"
    m2 = [x for x in seen if x["path"] == "messages"][-1]["messages"]
    assert [m["role"] for m in m2] == ["user", "assistant", "user"]
    assert m2[1]["content"][1] == {"type": "tool_use", "id": "t1", "name": "get_time",
                                   "input": {"timezone": "UTC"}}
    assert m2[2]["content"][0]["tool_use_id"] == "t1"


async def test_preflight_fails_when_the_follow_up_turn_fails(preflight_client):
    fake, _ = fake_provider(follow_up_fails=True)
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is False
    for x in rep["results"]:
        assert x["tool_call_seen"] and not x["follow_up_ok"] and not x["ok"]
        assert x["error"].startswith("follow-up turn")
    hf = {x["model"]: x for x in rep["results"]}["hf:Qwen/Qwen3.8-27B"]
    assert "400 status code" in hf["error"]


async def test_preflight_warns_on_zero_usage(preflight_client):
    fake, _ = fake_provider(zero_usage=True)
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is True  # a warning, not a failure
    for x in rep["results"]:
        assert x["usage_reported"] is False and "zero token usage" in x["warning"]


async def test_preflight_reports_missing_tool_call(preflight_client):
    fake, seen = fake_provider(tool_call=False)
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is False
    for x in rep["results"]:
        assert x["ok"] is False and x["tool_call_seen"] is False and "no tool call" in x["error"]
    assert len(seen) == 3  # no follow-up without a tool call


async def test_preflight_http_error_and_missing_secret(preflight_client):
    fake, seen = fake_provider(status=404)
    async with await preflight_client(fake, {"HF_TOKEN": "x"}) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    by_model = {x["model"]: x for x in rep["results"]}
    assert by_model["hf:Qwen/Qwen3.8-27B"]["error"].startswith("first turn: HTTP 404")
    small = by_model["gpu1:Qwen/Small"]  # no key on the controller: unverified, not failed
    assert small["unverified"] is True and small["ok"] is False and small["error"] is None
    assert "unverified: secret VLLM_API_KEY is not set on the controller" in small["warning"]
    assert rep["ok"] is False and rep["unverified"] == 2  # the 404 still fails the report
    assert all(x["model"] != "Qwen/Small" for x in seen)  # no request without a key


async def test_preflight_missing_key_alone_is_unverified_not_failed(preflight_client):
    fake, seen = fake_provider()
    async with await preflight_client(fake, {}) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is True and rep["unverified"] == 3 and not seen
    assert all(x["unverified"] and not x["ok"] for x in rep["results"])


async def test_preflight_uses_applied_spec_without_body(preflight_client):
    fake, _ = fake_provider()
    async with await preflight_client(fake, SECRETS) as c:
        assert (await c.post("/swarms/p/preflight")).status_code == 404
        assert (await put(c, preflight_spec())).status_code == 200
        rep = (await c.post("/swarms/p/preflight")).json()
        assert rep["ok"] and len(rep["results"]) == 3
        other = preflight_spec().model_dump(mode="json")
        other["swarm"] = "q"
        assert (await c.post("/swarms/p/preflight", json=other)).status_code == 400


async def test_preflight_bearer_auth_for_anthropic(preflight_client):
    fake, seen = fake_provider()
    providers = {"anth": {**PROVIDERS["anth"], "base_url": "http://fake", "auth": "bearer"}}
    spec = pspec(providers, writer={"harness": "claude", "model": "anth:claude-x"})
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()
    assert rep["ok"] is True
    assert all(x["auth"] == "Bearer ak" and x["key"] is None for x in seen)


async def test_preflight_probes_chat_completions_for_llm_types(preflight_client):
    fake, seen = fake_provider()
    extra = {"chat_template_kwargs": {"enable_thinking": False}, "response_format": {"type": "json_object"}}
    spec = SwarmSpec.model_validate({
        "swarm": "p", "namespace": "ns", "bucket": "hf://buckets/ns/b", "providers": preflight_providers(),
        "models": {"small": {"model": "gpu1:Qwen/Small", "extra": extra}},
        "types": {"pixel": {"harness": "llm", "model": "small"},
                  "pixel2": {"harness": "llm", "model": "small"},
                  "coder": {"harness": "codex", "model": "gpu1:Qwen/Small"}}})
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()
    assert rep["ok"] is True
    by_api = {x["api"]: x for x in rep["results"]}
    assert set(by_api) == {"chat_completions", "responses"}  # same model, two wires
    chat = by_api["chat_completions"]
    assert chat["types"] == ["pixel", "pixel2"] and chat["tool_call_seen"] and chat["follow_up_ok"]
    assert chat["usage_reported"] and chat["warning"] is None
    chats = [x for x in seen if x["path"] == "chat"]
    assert len(chats) == 2 and chats[0]["auth"] == "Bearer v"
    assert "stream" not in chats[0] and chats[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "response_format" not in chats[0]  # the agent's output schema, not the probe's
    assert chats[0]["tools"][0]["function"]["name"] == "get_time" and chats[0]["temperature"] == 0
    m2 = chats[1]["messages"]
    assert [m["role"] for m in m2] == ["user", "assistant", "tool"]
    assert m2[2]["tool_call_id"] == "call_1" == m2[1]["tool_calls"][0]["id"]


def test_api_for_pi_speaks_chat_completions_on_openai_compat():
    from flock.controller.preflight import api_for, models_to_probe
    from flock.models import Harness
    spec = pspec(preflight_providers(), **{h: {"harness": h, "model": "gpu1:Qwen/Small"}
                                           for h in ("llm", "pi", "codex")})
    assert [api_for(Harness(h)) for h in ("llm", "pi", "codex")] == [
        "chat_completions", "chat_completions", "responses"]
    probes = {tuple(types): (api, choice) for _, types, api, choice in models_to_probe(spec)}
    assert probes == {("llm", "pi"): ("chat_completions", None), ("codex",): ("responses", None)}
    anth = pspec({"anth": {**PROVIDERS["anth"], "base_url": "http://fake"}},
                 a={"harness": "claude", "model": "anth:claude-x"})
    assert api_for(anth.types["a"].harness) == "messages"


async def test_preflight_llm_accepts_a_tool_call_written_as_text(preflight_client):
    spec = pspec(preflight_providers(), pixel={"harness": "llm", "model": "gpu1:Qwen/Small"})
    fake, seen = fake_provider(text_call=True)
    async with await preflight_client(fake, SECRETS) as c:
        [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
    assert x["ok"] and x["api"] == "chat_completions" and "wrote its tool call as text" in x["warning"]
    assert seen[1]["messages"][1]["tool_calls"][0]["function"]["name"] == "get_time"
    # a forced tool_choice on the type: named -> the probe's own tool; follow-up back to auto
    forced = pspec(preflight_providers(), pixel={"harness": "llm", "model": "gpu1:Qwen/Small",
                                                 "tool_choice": "bash"})
    fake, seen = fake_provider()
    async with await preflight_client(fake, SECRETS) as c:
        [x] = (await c.post("/swarms/p/preflight", json=forced.model_dump(mode="json"))).json()["results"]
    assert x["ok"] and seen[0]["tool_choice"] == {"type": "function", "function": {"name": "get_time"}}
    assert seen[1]["tool_choice"] == "auto"
    for kw, err in (({"tool_call": False}, "no tool call"), ({"follow_up_fails": True}, "follow-up turn")):
        fake, _ = fake_provider(**kw)
        async with await preflight_client(fake, SECRETS) as c:
            [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
        assert not x["ok"] and err in x["error"]


async def test_status_counts_agents_with_zero_usage(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-worker-w0000"
    agents = [AgentHeartbeat(name="worker-0000", state="idle", turns=3,
                             tokens={"Qwen/Qwen3.8-27B": TokenUsage()}),
              AgentHeartbeat(name="worker-0001", state="idle", turns=0)]
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    st = (await client.get("/swarms/p")).json()["status"]
    assert st["providers"]["hf"]["zero_usage_agents"] == 1  # turns > 0 and no tokens
    assert st["providers"]["gpu1"]["zero_usage_agents"] == 0
    assert "report zero tokens (cost unknown)" in (await client.get("/")).text


# max_working at prompt admission; the reviewer's provider_cap_repro.py is the first case.
CAP_PROVIDERS = {"p": {"type": "openai_compat", "base_url": "http://fake/v1", "api_key_env": "K",
                       "self_hosted": True, "max_working": 1}}


async def _cap_client(store, clock, enforce):
    app = create_app(store, TOKEN, clock=clock, enforce_max_working=enforce)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                             headers={"Authorization": f"Bearer {TOKEN}"})


async def _delivered(c, wid) -> list[str]:
    cmds = (await c.get(f"/workers/{wid}/commands?wait=0")).json()["commands"]
    return [x["agent"] for x in cmds if x["kind"] == "prompt"]


@pytest.mark.parametrize("env, enforced", [(None, True), ("1", True), ("0", False)])
async def test_max_working_holds_prompts_for_idle_agents(store, clock, tick, monkeypatch, env,
                                                         enforced):
    if env is None:
        monkeypatch.delenv("FLOCK_MAX_WORKING_ENFORCE", raising=False)
    else:
        monkeypatch.setenv("FLOCK_MAX_WORKING_ENFORCE", env)
    spec = pspec(CAP_PROVIDERS, worker={"harness": "codex", "model": "p:test", "replicas": 2})
    wid = "p-worker-w0000"
    async with await _cap_client(store, clock, None) as c:
        assert (await put(c, spec)).status_code == 200
        tick()
        await _delivered(c, wid)  # drain the assigns
        busy = [AgentHeartbeat(name="worker-0000", state="working", turn_id="t1"),
                AgentHeartbeat(name="worker-0001", state="idle")]
        await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, busy))
        st = (await c.get("/swarms/p")).json()["status"]["providers"]["p"]
        assert st["working"] == st["max_working"] == 1
        ack = (await c.post("/agents/worker-0001/prompt", json={"text": "start"})).json()
        assert ack["status"] == "queued"  # accepted, never rejected
        if not enforced:  # FLOCK_MAX_WORKING_ENFORCE=0: report only, as before
            assert await _delivered(c, wid) == ["worker-0001"]
            return
        assert await _delivered(c, wid) == []  # provider full: held in the controller
        await c.post("/agents/worker-0000/prompt", json={"text": "more for the working one"})
        assert await _delivered(c, wid) == ["worker-0000"]  # a working agent takes no new slot
        clock.advance(30)
        idle = [AgentHeartbeat(name="worker-0000", state="idle"),
                AgentHeartbeat(name="worker-0001", state="idle")]
        await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, idle))
        assert await _delivered(c, wid) == ["worker-0001"]  # slot freed: dispatched
        assert store.pending_counts("p") == {}  # acked once the response went out


async def test_max_working_counts_admissions_between_heartbeats(store, clock, tick):
    spec = pspec(CAP_PROVIDERS, worker={"harness": "codex", "model": "p:test", "replicas": 2,
                                        "agents_per_worker": 1})
    async with await _cap_client(store, clock, True) as c:
        assert (await put(c, spec)).status_code == 200
        tick()
        for i, wid in enumerate(("p-worker-w0000", "p-worker-w0001")):
            await _delivered(c, wid)
            idle = [AgentHeartbeat(name=f"worker-000{i}", state="idle")]
            await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, idle))
        clock.advance(1)
        for name in ("worker-0000", "worker-0001"):
            await c.post(f"/agents/{name}/prompt", json={"text": "go"})
        # both idle per heartbeat, but the first poll took the only slot
        assert await _delivered(c, "p-worker-w0000") == ["worker-0000"]
        assert await _delivered(c, "p-worker-w0001") == []
        clock.advance(30)  # the next heartbeat says worker-0000 is working: still full
        busy = [AgentHeartbeat(name="worker-0000", state="working", turn_id="t")]
        await c.post("/workers/p-worker-w0000/heartbeat", json=hb(clock, "p-worker-w0000", busy))
        assert await _delivered(c, "p-worker-w0001") == []
        clock.advance(30)
        idle = [AgentHeartbeat(name="worker-0000", state="idle")]
        await c.post("/workers/p-worker-w0000/heartbeat", json=hb(clock, "p-worker-w0000", idle))
        assert await _delivered(c, "p-worker-w0001") == ["worker-0001"]
