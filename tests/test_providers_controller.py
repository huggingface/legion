"""Controller side of `providers:`: job secrets, pinned pricing, status, preflight."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

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
                    "types": ["local"], "working": 1, "max_working": 2}
    assert st["status"]["providers"]["anth"]["types"] == []
    page = (await client.get("/")).text
    assert "gpu1:Qwen/Small" in page and "working 1 / max 2" in page


# --------------------------------------------------------------------------- preflight


def fake_provider(tool_call: bool = True, status: int = 200) -> tuple[FastAPI, list[dict]]:
    """A tiny OpenAI-compatible (Responses) and Anthropic (Messages) server."""
    fake = FastAPI()
    seen: list[dict] = []

    @fake.post("/v1/responses")
    async def responses(request: Request):
        body = await request.json()
        seen.append({"path": "responses", "auth": request.headers.get("authorization"), **body})
        if status != 200:
            return JSONResponse({"error": {"message": "model not found"}}, status_code=status)
        out = [{"type": "function_call", "call_id": "c1", "name": "get_time",
                "arguments": '{"timezone": "UTC"}'}] if tool_call else [
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "It is noon."}]}]
        return {"id": "r1", "object": "response", "status": "completed", "output": out}

    @fake.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        seen.append({"path": "messages", "key": request.headers.get("x-api-key"), **body})
        content = [{"type": "tool_use", "id": "t1", "name": "get_time", "input": {"timezone": "UTC"}}]
        return {"id": "m1", "type": "message", "role": "assistant",
                "content": content if tool_call else [{"type": "text", "text": "noon"}]}

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
    assert all(x["tool_call_seen"] and x["latency_ms"] is not None for x in rep["results"])
    assert by_model["anth:claude-x"]["api"] == "messages"
    req = {(x["path"], x["model"]): x for x in seen}
    assert len(seen) == 3
    hf = req[("responses", "Qwen/Qwen3.8-27B")]
    assert hf["auth"] == "Bearer hf-secret" and hf["tools"][0]["name"] == "get_time"
    assert "reasoning" not in hf
    assert req[("responses", "Qwen/Small")]["reasoning"] == {"effort": "none"}
    assert req[("messages", "claude-x")]["key"] == "ak"


async def test_preflight_reports_missing_tool_call(preflight_client):
    fake, _ = fake_provider(tool_call=False)
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is False
    for x in rep["results"]:
        assert x["ok"] is False and x["tool_call_seen"] is False and "no tool call" in x["error"]


async def test_preflight_http_error_and_missing_secret(preflight_client):
    fake, seen = fake_provider(status=404)
    async with await preflight_client(fake, {"HF_TOKEN": "x"}) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    by_model = {x["model"]: x for x in rep["results"]}
    assert by_model["hf:Qwen/Qwen3.8-27B"]["error"].startswith("HTTP 404")
    assert by_model["gpu1:Qwen/Small"]["error"] == "secret VLLM_API_KEY is not set on the controller"
    assert rep["ok"] is False
    assert all(x["model"] != "Qwen/Small" for x in seen)  # no request without a key


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
