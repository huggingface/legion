"""Controller side of `inference:`: job secrets, pinned pricing, the apply's model check and
context window, status, preflight, max_working."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient

from flock.controller.api import create_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import rebuild_from_jobs, reconcile
from flock.controller.store import Store
from flock.models import AgentHeartbeat, SwarmSpec, TokenUsage, WorkerHeartbeat

from . import fake_models
from .conftest import HB, TOKEN

ROUTER = {"hf_router": {"provider": "deepinfra"}}
GPU1 = {"urls": ["http://qwen-job:8000/v1"], "api_key_env": "VLLM_API_KEY", "max_working": 2}


def pspec(inference=None, **types) -> SwarmSpec:
    types = types or {
        "worker": {**HB, "model": "Qwen/Qwen3.8-27B", "pricing": [1.0, 2.0, 0.5], "replicas": 2,
                   "agents_per_worker": 2, "secrets": ["HF_TOKEN"]},
        "local": {**HB, "model": "Qwen/Small", "inference": GPU1, "replicas": 1},
    }
    return SwarmSpec.model_validate({"swarm": "p", "namespace": "ns", "bucket": "hf://buckets/ns/b",
                                     "inference": inference or ROUTER, "types": types})


def hb(clock, wid, agents) -> dict:
    return WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=24),
                           agents=agents).model_dump(mode="json")


async def put(client, spec) -> httpx.Response:
    return await client.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))


async def test_the_key_is_added_to_job_secrets(client, jobs, settings, tick):
    settings.secret_source = {"HF_TOKEN": "hf-x", "VLLM_API_KEY": "v-x", "ANTHROPIC_API_KEY": "a"}
    assert (await put(client, pspec())).status_code == 200
    tick()
    by_type = {x["labels"]["type"]: x for x in jobs.launches}
    assert by_type["worker"]["secrets"] == {"FLOCK_TOKEN": TOKEN, "HF_TOKEN": "hf-x"}  # deduped
    assert by_type["local"]["secrets"] == {"FLOCK_TOKEN": TOKEN, "VLLM_API_KEY": "v-x"}
    assert by_type["worker"]["spec"].resolved_model.inference.key_env == "HF_TOKEN"


async def test_invalid_spec_is_rejected_at_apply(client):
    raw = pspec().model_dump(mode="json")
    raw["types"]["worker"]["model"] = "Qwen/X:deepinfra"
    r = await client.put("/swarms/p", json=raw)
    assert r.status_code == 422
    assert "types.worker.model" in r.text and "hf_router.provider" in r.text


async def test_cost_uses_pinned_price_and_does_not_reprice_history(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-worker-w0000"
    mtok = TokenUsage(input=1_000_000, output=1_000_000, cache_read=1_000_000)
    agents = [AgentHeartbeat(name="worker-0000", state="working", turn_id="t-1",
                             tokens={"Qwen/Qwen3.8-27B": mtok})]
    assert (await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))).status_code == 200
    a = (await client.get("/agents/worker-0000")).json()
    assert a["cost_usd"] == 3.5 and a["provider"] == "inference" and a["model"] == "Qwen/Qwen3.8-27B"

    # Re-apply with a new price: only tokens after the change are priced at it.
    spec = pspec()
    spec.types["worker"].pricing = [10, 20, 5]
    assert (await put(client, SwarmSpec.model_validate(spec.model_dump()))).status_code == 200
    agents[0].tokens = {"Qwen/Qwen3.8-27B": mtok + TokenUsage(input=1_000_000)}
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    assert (await client.get("/agents/worker-0000")).json()["cost_usd"] == 13.5


async def test_servers_you_run_are_free_and_status_shows_each_block(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-local-w0000"
    agents = [AgentHeartbeat(name="local-0000", state="working", turn_id="t-1",
                             tokens={"Qwen/Small": TokenUsage(input=5_000_000, output=1_000_000)})]
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    a = (await client.get("/agents/local-0000")).json()
    assert a["cost_usd"] == 0.0 and a["provider"] == "types.local.inference"
    st = (await client.get("/swarms/p")).json()
    assert st["spec"]["types"]["local"]["resolved_model"]["pricing"] == [0.0, 0.0, 0.0]  # pinned
    gpu1 = st["status"]["providers"]["types.local.inference"]
    assert gpu1 == {"name": "types.local.inference", "endpoint": "urls http://qwen-job:8000/v1",
                    "types": ["local"], "models": {"Qwen/Small": fake_models.WINDOW},
                    "working": 1, "max_working": 2, "zero_usage_agents": 0}
    assert st["status"]["providers"]["inference"]["endpoint"] == "hf_router deepinfra"
    page = (await client.get("/")).text
    assert "types.local.inference:Qwen/Small" in page and "working 1 / max 2" in page


def local_spec(**inference) -> SwarmSpec:
    return pspec(ROUTER, local={**HB, "model": "Qwen/Small", "replicas": 1,
                                "inference": {**GPU1, **inference}})


MOVED = {"urls": ["http://qwen-job-2:8000/v1"], "api_key_env": "NEW_KEY"}


def drains(store, wid) -> int:
    return sum(json.loads(c["body"])["kind"] == "drain" for c in store.commands_for(wid))


async def test_a_changed_block_restarts_its_types_workers(client, store, jobs, settings, tick,
                                                          clock):
    """Workers hold the block's key (job secrets) and their Trials its endpoints and allowlist:
    a change restarts the type's workers at the next pass, and they relaunch with the new
    block; max_working (the controller's) restarts nothing."""
    settings.secret_source = {"VLLM_API_KEY": "v", "NEW_KEY": "n"}
    assert (await put(client, local_spec())).status_code == 200
    tick()
    (first,) = jobs.launches
    wid = first["worker_id"]
    assert first["secrets"]["VLLM_API_KEY"] == "v"
    assert (await put(client, local_spec(max_working=5))).status_code == 200
    tick()
    assert drains(store, wid) == 0
    assert (await put(client, local_spec(**MOVED))).status_code == 200
    tick()
    assert drains(store, wid) == 1
    events = (await client.get("/swarms/p/events")).json()["events"]
    assert [e["data"] for e in events if e["type"] == "restart"] == [
        {"target": wid, "detail": "inference changed"}]
    jobs.set_stage(first["job_id"], "COMPLETED")  # the drained worker exits
    tick()
    clock.advance(60)
    tick()
    second = jobs.launches[-1]
    assert second["worker_id"] == wid and second["job_id"] != first["job_id"]
    assert second["secrets"]["NEW_KEY"] == "n" and "VLLM_API_KEY" not in second["secrets"]
    assert second["spec"].resolved_model.inference.urls == ["http://qwen-job-2:8000/v1"]


def test_a_launch_that_raced_an_update_restarts(app, store, clock, settings):
    """The update lands while the backend creates the job, so the job holds the old block: the
    next pass restarts it."""
    settings.secret_source = {"VLLM_API_KEY": "v", "NEW_KEY": "n"}
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})

    class Racing(FakeJobsClient):
        def launch(self, *args, **kwargs):
            job = super().launch(*args, **kwargs)
            if len(self.launches) == 1:  # the job of the old block is being created
                assert c.put("/swarms/p", json=local_spec(**MOVED).model_dump(
                    mode="json")).status_code == 200
            return job

    jobs = Racing()
    assert c.put("/swarms/p", json=local_spec().model_dump(mode="json")).status_code == 200
    reconcile(store, jobs, clock(), settings)
    (first,) = jobs.launches
    assert "VLLM_API_KEY" in first["secrets"] and drains(store, first["worker_id"]) == 0
    reconcile(store, jobs, clock(), settings)
    assert drains(store, first["worker_id"]) == 1


async def test_a_lost_drain_is_sent_again(client, store, jobs, settings, tick, clock):
    """A drain whose delivery was lost (the poll's response) goes again until the worker
    reports draining; the worker's drain is idempotent."""
    settings.secret_source = {"VLLM_API_KEY": "v", "NEW_KEY": "n"}
    assert (await put(client, local_spec())).status_code == 200
    tick()
    wid = jobs.launches[0]["worker_id"]
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, []))
    assert (await put(client, local_spec(**MOVED))).status_code == 200
    tick()
    assert drains(store, wid) == 1
    await client.get(f"/workers/{wid}/commands?wait=0")  # delivered, and the response lost
    clock.advance(30)
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, []))  # not draining
    tick()
    assert drains(store, wid) == 1  # sent again
    tick()
    assert drains(store, wid) == 1  # once while it is queued
    await client.get(f"/workers/{wid}/commands?wait=0")
    clock.advance(30)
    beat = {**hb(clock, wid, []), "draining": True}
    await client.post(f"/workers/{wid}/heartbeat", json=beat)
    tick()
    assert drains(store, wid) == 0  # it drains


async def test_a_drain_lost_before_the_first_heartbeat_is_sent_again(client, store, jobs,
                                                                      settings, tick):
    """Before the job's first heartbeat its drain is unacknowledged too."""
    settings.secret_source = {"VLLM_API_KEY": "v", "NEW_KEY": "n"}
    assert (await put(client, local_spec())).status_code == 200
    tick()
    wid = jobs.launches[0]["worker_id"]
    assert (await put(client, local_spec(**MOVED))).status_code == 200
    tick()
    assert drains(store, wid) == 1
    await client.get(f"/workers/{wid}/commands?wait=0")  # delivered, and the response lost
    tick()
    assert drains(store, wid) == 1  # sent again, with no heartbeat yet


@pytest.mark.parametrize("carried", [True, False])
def test_a_restored_controller_restarts_a_readopted_job_of_another_block(tmp_path, clock,
                                                                         settings, carried):
    """A controller restored from a checkpoint taken before a launch re-adopts the job by its
    labels, with the inference revision the job carries (`INFERENCE_REV`): the job restarts
    when the block changed since, and when its revision is unknown."""
    settings.secret_source = {"VLLM_API_KEY": "v", "NEW_KEY": "n"}
    jobs, first = FakeJobsClient(), Store(tmp_path / "first")
    c = TestClient(create_app(first, TOKEN, clock=clock), headers={"Authorization": f"Bearer {TOKEN}"})
    assert c.put("/swarms/p", json=local_spec().model_dump(mode="json")).status_code == 200
    checkpoint = first.get_swarm("p")["spec"]  # the swarm, before its worker launched
    reconcile(first, jobs, clock(), settings)
    (launch,) = jobs.launches
    assert launch["env"]["INFERENCE_REV"]
    if not carried:  # e.g. a job launched by another controller version
        del launch["env"]["INFERENCE_REV"]
    restored = Store(tmp_path / "restored")
    restored.put_swarm(checkpoint, clock().timestamp())
    assert [a.kind for a in rebuild_from_jobs(restored, jobs, clock())] == ["adopt"]
    reconcile(restored, jobs, clock(), settings)
    assert drains(restored, launch["worker_id"]) == (0 if carried else 1)
    c = TestClient(create_app(restored, TOKEN, clock=clock),
                   headers={"Authorization": f"Bearer {TOKEN}"})
    assert c.put("/swarms/p", json=local_spec(**MOVED).model_dump(mode="json")).status_code == 200
    reconcile(restored, jobs, clock(), settings)
    assert drains(restored, launch["worker_id"]) == 1
    first.close()
    restored.close()


async def test_a_controller_only_change_keeps_the_pinned_window(store, clock, jobs, settings):
    """max_working is the controller's: its change keeps the window the workers' Trials hold
    (no call, even when the server now reports another) and restarts nothing."""
    reply = {"qwen-job": fake_models.listing(["Qwen/Small"], 32768)}
    seen: list[str] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        return httpx.Response(200, json=reply[request.url.host])

    app = create_app(store, TOKEN, clock=clock, secret_source={"VLLM_API_KEY": "v"},
                     inference_transport=httpx.MockTransport(handle))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        assert (await put(c, local_spec())).status_code == 200
        reconcile(store, jobs, clock(), settings)
        reply["qwen-job"] = fake_models.listing(["Qwen/Small"], 8192)
        seen.clear()
        assert (await put(c, local_spec(max_working=5))).status_code == 200
        reconcile(store, jobs, clock(), settings)
        st = (await c.get("/swarms/p")).json()
    assert seen == [] and st["spec"]["types"]["local"]["resolved_model"]["context_window"] == 32768
    assert drains(store, jobs.launches[0]["worker_id"]) == 0


# --------------------------------------------------------------------------- the apply's check


def router_listing(*providers: dict, model: str = "Qwen/Qwen3.8-27B") -> dict:
    return {"object": "list", "data": [{"id": model, "providers": list(providers)}]}


def provider(name: str, ctx: int | None = None, tools: bool | None = True) -> dict:
    return {"provider": name, "status": "live",
            **({"context_length": ctx} if ctx else {}),
            **({"supports_tools": tools} if tools is not None else {})}


def checked(store, clock, reply: dict, seen: list | None = None):
    """A client whose apply asks `GET /models` of a fake answering `reply`."""
    app = create_app(store, TOKEN, clock=clock, secret_source={"VLLM_API_KEY": "v"},
                     inference_transport=fake_models.transport(reply, seen))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": f"Bearer {TOKEN}"})


def one(inference: dict, model: str = "Qwen/Qwen3.8-27B") -> SwarmSpec:
    return pspec(inference, worker={**HB, "model": model})


ROUTER_TOOLS = router_listing(provider("deepinfra", 262144), provider("cerebras", 65536),
                              provider("novita", 1000000), provider("featherless-ai", tools=None),
                              provider("slowtools", 8192, tools=False))


@pytest.mark.parametrize(("hf_router", "window"), [
    ({"provider": "cerebras"}, 65536),  # the pinned provider's
    ({}, 65536), ({"provider": "cheapest"}, 65536),  # a policy: the smallest with tools
    ({"provider": "novita"}, 1000000),
])
async def test_apply_pins_the_routers_context_window(store, clock, hf_router, window):
    async with checked(store, clock, ROUTER_TOOLS) as c:
        r = await put(c, one({"hf_router": hf_router}))
        assert r.status_code == 200, r.text
        st = (await c.get("/swarms/p")).json()
    assert st["spec"]["types"]["worker"]["resolved_model"]["context_window"] == window
    assert st["status"]["providers"]["inference"]["models"] == {"Qwen/Qwen3.8-27B": window}


@pytest.mark.parametrize(("hf_router", "model", "needle"), [
    ({}, "Qwen/Qwen9", "the HF router does not list 'Qwen/Qwen9'"),
    ({}, "qwen/qwen3.8-27b", "did you mean 'Qwen/Qwen3.8-27B'?"),
    ({"provider": "groq"}, "Qwen/Qwen3.8-27B",
     "'groq' does not serve 'Qwen/Qwen3.8-27B'; providers with tool support: cerebras, deepinfra, novita"),
    ({"provider": "featherless-ai"}, "Qwen/Qwen3.8-27B", "does not list tool support"),
    ({"provider": "slowtools"}, "Qwen/Qwen3.8-27B", "does not list tool support"),
])
async def test_apply_refuses_a_model_the_router_cannot_serve_with_tools(store, clock, hf_router,
                                                                         model, needle):
    async with checked(store, clock, ROUTER_TOOLS) as c:
        r = await put(c, one({"hf_router": hf_router}, model))
    assert r.status_code == 400 and needle in r.json()["detail"]
    assert r.json()["detail"].startswith(f"types.worker.model {model!r} on inference: ")


async def test_apply_checks_every_url_and_pins_the_smallest_window(store, clock):
    urls = {"urls": ["http://a:8000/v1", "http://b:8000/v1"], "api_key_env": "VLLM_API_KEY"}
    seen: list[str] = []
    replies = {"a": fake_models.listing(["Qwen/Small"], 40960),
               "b": fake_models.listing(["Qwen/Small", "Qwen/Other"], 32768)}

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.host, request.headers.get("authorization")))
        return httpx.Response(200, json=replies[request.url.host])

    app = create_app(store, TOKEN, clock=clock, secret_source={"VLLM_API_KEY": "v"},
                     inference_transport=httpx.MockTransport(handle))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        assert (await put(c, one(urls, "Qwen/Small"))).status_code == 200
        st = (await c.get("/swarms/p")).json()
        assert st["spec"]["types"]["worker"]["resolved_model"]["context_window"] == 32768
        assert sorted(seen) == [("a", "Bearer v"), ("b", "Bearer v")]
        r = await put(c, one(urls, "Qwen/Other"))  # a's listing does not have it
        assert r.status_code == 400 and r.json()["detail"] == (
            "types.worker.model 'Qwen/Other' on inference: http://a:8000/v1 does not serve "
            "'Qwen/Other'; it serves: Qwen/Small")
        seen.clear()  # a re-apply of the same block and model keeps the window without a call
        spec = one(urls, "Qwen/Small")
        spec.types["worker"].replicas = 3
        assert (await put(c, spec)).status_code == 200 and seen == []


def by_host(replies: dict[str, dict]) -> httpx.MockTransport:
    """`GET /models` answered per host."""
    return httpx.MockTransport(lambda request: httpx.Response(200, json=replies[request.url.host]))


QWEN = fake_models.listing(["qwen3.5:2b"], None)  # Ollama's: no window
WITH_TOOLS_NO_WINDOW = router_listing(provider("cerebras", 65536), provider("newco"))


@pytest.mark.parametrize(("inference", "replies", "fallback", "window_or_error"), [
    ({"urls": ["http://box:11434/v1"]}, {"box": QWEN}, None,
     "http://box:11434/v1 reports no context window for 'qwen3.5:2b': set inference.context_window"),
    ({"urls": ["http://box:11434/v1"]}, {"box": QWEN}, 16384, 16384),
    # each server takes the fallback when it reports none, and the smallest wins
    ({"urls": ["http://a:8000/v1", "http://box:11434/v1"]},
     {"a": fake_models.listing(["qwen3.5:2b"], 40960), "box": QWEN}, None,
     "http://box:11434/v1 reports no context window"),
    ({"urls": ["http://a:8000/v1", "http://box:11434/v1"]},
     {"a": fake_models.listing(["qwen3.5:2b"], 40960), "box": QWEN}, 8192, 8192),
    ({"urls": ["http://a:8000/v1", "http://box:11434/v1"]},
     {"a": fake_models.listing(["qwen3.5:2b"], 40960), "box": QWEN}, 65536, 40960),
    # under a router policy, every provider with tools may serve a request
    ({"hf_router": {}}, {"router.huggingface.co": WITH_TOOLS_NO_WINDOW}, None,
     "HF router provider 'newco' reports no context window"),
    ({"hf_router": {}}, {"router.huggingface.co": WITH_TOOLS_NO_WINDOW}, 32768, 32768),
    ({"hf_router": {"provider": "cerebras"}}, {"router.huggingface.co": WITH_TOOLS_NO_WINDOW},
     None, 65536),
])
async def test_a_server_without_a_window_takes_context_window_or_is_refused(
        store, clock, inference, replies, fallback, window_or_error):
    model = "Qwen/Qwen3.8-27B" if "hf_router" in inference else "qwen3.5:2b"
    block = {**inference, **({"context_window": fallback} if fallback else {})}
    app = create_app(store, TOKEN, clock=clock, inference_transport=by_host(replies))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = await put(c, one(block, model))
        if isinstance(window_or_error, str):
            assert r.status_code == 400 and window_or_error in r.json()["detail"], r.text
            return
        assert r.status_code == 200, r.text
        st = (await c.get("/swarms/p")).json()
    assert st["spec"]["types"]["worker"]["resolved_model"]["context_window"] == window_or_error


async def test_a_server_that_wants_a_key_the_controller_lacks(store, clock):
    def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "Unauthorized"})

    app = create_app(store, TOKEN, clock=clock, secret_source={},
                     inference_transport=httpx.MockTransport(handle))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = await put(c, one({**GPU1}, "Qwen/Small"))
    assert r.status_code == 400 and "HTTP 401" in r.json()["detail"]
    assert r.json()["detail"].endswith("(VLLM_API_KEY is not set on the controller)")


# --------------------------------------------------------------------------- preflight


def fake_provider(tool_call: bool = True, status: int = 200, follow_up_fails: bool = False,
                  zero_usage: bool = False, text_call: bool = False,
                  no_call_on: str | None = None) -> tuple[FastAPI, list[dict]]:
    """A tiny OpenAI-compatible Chat Completions server. `follow_up_fails` refuses the turn
    carrying the tool result, as the HF router did on its auto provider."""
    fake = FastAPI()
    seen: list[dict] = []
    usage = {"input_tokens": 0 if zero_usage else 12, "output_tokens": 0 if zero_usage else 5}

    @fake.get("/v1/models")
    async def models():
        return fake_models.listing()

    @fake.post("/v1/chat/completions")
    async def chat(request: Request):
        """Non-streamed Chat Completions (Harbor's Pi). `text_call`: the model writes the
        call as JSON text instead of `tool_calls`, as small models do."""
        body = await request.json()
        seen.append({"path": "chat", "auth": request.headers.get("authorization"),
                     "bill_to": request.headers.get("x-hf-bill-to"), **body})
        if status != 200:
            return JSONResponse({"error": {"message": "model not found"}}, status_code=status)
        follow_up = body["messages"][-1]["role"] == "tool"
        if follow_up and follow_up_fails:
            return JSONResponse({"error": {"message": "bad tool message"}}, status_code=400)
        msg: dict = {"role": "assistant", "content": "It is noon UTC."}
        if tool_call and not follow_up and request.url.hostname != no_call_on:
            msg = ({"role": "assistant", "content": 'hmm</think>{"tool": "get_time", "timezone": "UTC"}'}
                   if text_call
                   else {"role": "assistant", "content": None, "tool_calls": [
                       {"id": "call_1", "type": "function",
                        "function": {"name": "get_time", "arguments": '{"timezone": "UTC"}'}}]})
        u = {"prompt_tokens": usage["input_tokens"], "completion_tokens": usage["output_tokens"]}
        return JSONResponse({"id": "c1", "object": "chat.completion", "usage": u,
                             "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]})

    return fake, seen


FAKE_GPU = {**GPU1, "urls": ["http://fake/v1"]}  # the router's requests reach the same fake


def preflight_spec(**extra_types) -> SwarmSpec:
    return pspec(ROUTER, **{
        "worker": {**HB, "model": "Qwen/Qwen3.8-27B"},
        "worker2": {**HB, "model": "Qwen/Qwen3.8-27B"},
        "local": {**HB, "model": "Qwen/Small", "inference": FAKE_GPU, "reasoning": "off"},
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
    assert set(by_model) == {"inference:Qwen/Qwen3.8-27B", "types.local.inference:Qwen/Small"}
    assert by_model["inference:Qwen/Qwen3.8-27B"]["types"] == ["worker", "worker2"]  # one request, two types
    assert all(x["tool_call_seen"] and x["follow_up_ok"] and x["usage_reported"]
               and x["latency_ms"] is not None and x["warning"] is None for x in rep["results"])
    assert len(seen) == 4  # two turns per model
    hf = next(x for x in seen if x["model"] == "Qwen/Qwen3.8-27B:deepinfra")  # the router's provider
    assert hf["auth"] == "Bearer hf-secret" and hf["tools"][0]["function"]["name"] == "get_time"


async def test_preflight_fails_when_the_follow_up_turn_fails(preflight_client):
    fake, _ = fake_provider(follow_up_fails=True)
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is False
    for x in rep["results"]:
        assert x["tool_call_seen"] and not x["follow_up_ok"] and not x["ok"]
        assert x["error"].startswith("follow-up turn")
    hf = {x["model"]: x for x in rep["results"]}["inference:Qwen/Qwen3.8-27B"]
    assert "bad tool message" in hf["error"] and "hf_router.provider" in hf["error"]


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
    assert len(seen) == 2  # no follow-up without a tool call


async def test_preflight_http_error_and_missing_secret(preflight_client):
    fake, seen = fake_provider(status=404)
    async with await preflight_client(fake, {"HF_TOKEN": "x"}) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    by_model = {x["model"]: x for x in rep["results"]}
    assert by_model["inference:Qwen/Qwen3.8-27B"]["error"].startswith("first turn: HTTP 404")
    small = by_model["types.local.inference:Qwen/Small"]  # no key on the controller: unverified
    assert small["unverified"] is True and small["ok"] is False and small["error"] is None
    assert "unverified: secret VLLM_API_KEY is not set on the controller" in small["warning"]
    assert rep["ok"] is False and rep["unverified"] == 1  # the 404 still fails the report
    assert all(x["model"] != "Qwen/Small" for x in seen)  # no request without a key


async def test_preflight_missing_key_alone_is_unverified_not_failed(preflight_client):
    fake, seen = fake_provider()
    async with await preflight_client(fake, {}) as c:
        rep = (await c.post("/swarms/p/preflight", json=preflight_spec().model_dump(mode="json"))).json()
    assert rep["ok"] is True and rep["unverified"] == 2 and not seen
    assert all(x["unverified"] and not x["ok"] for x in rep["results"])


async def test_preflight_uses_applied_spec_without_body(preflight_client):
    fake, _ = fake_provider()
    async with await preflight_client(fake, SECRETS) as c:
        assert (await c.post("/swarms/p/preflight")).status_code == 404
        assert (await put(c, preflight_spec())).status_code == 200
        rep = (await c.post("/swarms/p/preflight")).json()
        assert rep["ok"] and len(rep["results"]) == 2
        other = preflight_spec().model_dump(mode="json")
        other["swarm"] = "q"
        assert (await c.post("/swarms/p/preflight", json=other)).status_code == 400


async def test_preflight_probes_chat_completions(preflight_client):
    fake, seen = fake_provider()
    spec = pspec(FAKE_GPU, pixel={**HB, "model": "Qwen/Small"}, pixel2={**HB, "model": "Qwen/Small"})
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()
    assert rep["ok"] is True
    [chat] = rep["results"]
    assert chat["api"] == "chat_completions"
    assert chat["types"] == ["pixel", "pixel2"] and chat["tool_call_seen"] and chat["follow_up_ok"]
    assert chat["usage_reported"] and chat["warning"] is None
    chats = [x for x in seen if x["path"] == "chat"]
    assert len(chats) == 2 and chats[0]["auth"] == "Bearer v"
    assert "stream" not in chats[0] and chats[0]["model"] == "Qwen/Small"
    assert chats[0]["tools"][0]["function"]["name"] == "get_time" and chats[0]["temperature"] == 0
    m2 = chats[1]["messages"]
    assert [m["role"] for m in m2] == ["user", "assistant", "tool"]
    assert m2[2]["tool_call_id"] == "call_1" == m2[1]["tool_calls"][0]["id"]


def test_api_for_every_harness_is_chat_completions():
    from flock.controller.preflight import api_for
    from flock.models import Harness
    assert [api_for(h) for h in Harness] == ["chat_completions"] * len(Harness)


async def test_preflight_sends_the_router_provider_and_bill_to_as_pi_does(preflight_client):
    fake, seen = fake_provider()
    spec = pspec({"hf_router": {"provider": "novita", "bill_to": "huggingface"}},
                 h={**HB, "model": "deepseek-ai/DeepSeek-V4.1-Flash-1017"})
    async with await preflight_client(fake, SECRETS) as c:
        rep = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()
    assert len(rep["results"]) == 1 and rep["ok"]
    assert {(x["model"], x["bill_to"]) for x in seen} == {
        ("deepseek-ai/DeepSeek-V4.1-Flash-1017:novita", "huggingface")}


async def test_preflight_needs_a_tool_call_from_every_server(preflight_client):
    fake, _ = fake_provider(no_call_on="a")
    urls = {"urls": ["http://a/v1", "http://b/v1"], "api_key_env": "VLLM_API_KEY"}
    spec = pspec(urls, pixel={**HB, "model": "Qwen/Small"})
    async with await preflight_client(fake, SECRETS) as c:
        [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
    assert not x["ok"] and not x["tool_call_seen"]
    assert x["error"].startswith("http://a/v1: the reply has no tool call")
    urls["urls"].reverse()  # a server after one that passed
    spec = pspec(urls, pixel={**HB, "model": "Qwen/Small"})
    async with await preflight_client(fake, SECRETS) as c:
        [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
    assert not x["ok"] and x["error"].startswith("http://a/v1: the reply has no tool call")


async def test_preflight_checks_the_model_is_served_first(preflight_client):
    fake, seen = fake_provider()
    spec = pspec(FAKE_GPU, pixel={**HB, "model": "Qwen/Nope"})
    async with await preflight_client(fake, SECRETS) as c:
        [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
    assert not x["ok"] and x["error"].startswith("http://fake/v1 does not serve 'Qwen/Nope'; it serves: ")
    assert not seen  # no chat request


async def test_preflight_fails_without_a_tool_call_or_its_follow_up(preflight_client):
    spec = pspec(FAKE_GPU, pixel={**HB, "model": "Qwen/Small"})
    for kw, err in (({"tool_call": False}, "no tool call"), ({"text_call": True}, "no tool call"),
                    ({"follow_up_fails": True}, "follow-up turn")):
        fake, _ = fake_provider(**kw)
        async with await preflight_client(fake, SECRETS) as c:
            [x] = (await c.post("/swarms/p/preflight", json=spec.model_dump(mode="json"))).json()["results"]
        assert not x["ok"] and err in x["error"] and "--tool-call-parser" in x["error"]


async def test_status_counts_agents_with_zero_usage(client, clock, tick):
    await put(client, pspec())
    tick()
    wid = "p-worker-w0000"
    agents = [AgentHeartbeat(name="worker-0000", state="idle", turns=3,
                             tokens={"Qwen/Qwen3.8-27B": TokenUsage()}),
              AgentHeartbeat(name="worker-0001", state="idle", turns=0)]
    await client.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, agents))
    st = (await client.get("/swarms/p")).json()["status"]
    assert st["providers"]["inference"]["zero_usage_agents"] == 1  # turns > 0 and no tokens
    assert st["providers"]["types.local.inference"]["zero_usage_agents"] == 0
    assert "report zero tokens (cost unknown)" in (await client.get("/")).text


# max_working at prompt admission; the reviewer's provider_cap_repro.py is the first case.
CAP = {"urls": ["http://fake/v1"], "max_working": 1}


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
    spec = pspec(CAP, worker={**HB, "model": "m", "replicas": 2})
    wid = "p-worker-w0000"
    async with await _cap_client(store, clock, None) as c:
        assert (await put(c, spec)).status_code == 200
        tick()
        await _delivered(c, wid)  # drain the assigns
        busy = [AgentHeartbeat(name="worker-0000", state="working", turn_id="t1"),
                AgentHeartbeat(name="worker-0001", state="idle")]
        await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, busy))
        st = (await c.get("/swarms/p")).json()["status"]["providers"]["inference"]
        assert st["working"] == st["max_working"] == 1
        ack = (await c.post("/agents/worker-0001/prompt", json={"text": "start"})).json()
        assert ack["status"] == "queued"  # accepted, never rejected
        if not enforced:  # FLOCK_MAX_WORKING_ENFORCE=0: report only
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
    spec = pspec(CAP, worker={**HB, "model": "m", "replicas": 2, "agents_per_worker": 1})
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


@pytest.mark.parametrize("api_format", ["openai_chat_completions", "openai_responses", "anthropic_messages"])
async def test_task_local_endpoints_are_checked_in_the_trial_not_from_the_controller(api_format):
    from flock.controller.inference import check
    from flock.controller.preflight import run_preflight

    s = local_spec(task_local=True, context_window=128000, api_format=api_format)
    requests = []
    def unreachable(request):
        requests.append(request)
        raise AssertionError("task-local endpoint must not be contacted by the controller")
    transport = httpx.MockTransport(unreachable)
    check(s, {"VLLM_API_KEY": "fake"}, transport=transport)
    assert s.types["local"].resolved_model.context_window == 128000
    async with httpx.AsyncClient(transport=transport) as http:
        report = await run_preflight(s, {"VLLM_API_KEY": "fake"}, http)
    assert report.ok and report.unverified == 1
    assert report.results[0].unverified
    assert requests == []


@pytest.mark.parametrize("api_format", ["openai_responses", "anthropic_messages"])
async def test_native_api_preflight_echoes_the_provider_tool_call_and_checks_its_result(api_format):
    from flock.controller.preflight import probe

    r = local_spec(api_format=api_format, context_window=128000).types["local"].resolved_model
    messages = api_format == "anthropic_messages"
    content = ([{"type": "thinking", "thinking": "opaque", "signature": "keep-this"},
                {"type": "tool_use", "id": "call", "name": "get_time", "input": {"timezone": "UTC"}}]
               if messages else [{"type": "reasoning", "id": "r", "summary": []},
                                  {"type": "function_call", "id": "f", "call_id": "call",
                                   "name": "get_time", "arguments": '{"timezone":"UTC"}'}])
    seen = []
    def provider(request):
        if request.method == "GET":
            assert request.url.path.endswith("/v1/models")
            return httpx.Response(200, json={"data": [{"id": r.model_id, "max_model_len": 128000}]})
        body = json.loads(request.content)
        assert request.url.path.endswith("/v1/messages" if messages else "/v1/responses")
        assert request.headers["x-api-key" if messages else "authorization"] == ("fake" if messages else "Bearer fake")
        if seen:
            if messages:
                assert body["messages"][-2]["content"] == content
                assert body["messages"][-1]["content"][0]["tool_use_id"] == "call"
            else:
                assert body["input"][1:-1] == content
                assert body["input"][-1]["call_id"] == "call"
        seen.append(body)
        return httpx.Response(200, json={"content" if messages else "output": content if len(seen) == 1 else [],
                                        "usage": {"input_tokens": 2, "output_tokens": 1}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        result = await probe(http, r, "fake", ["local"])
    assert result.ok and result.tool_call_seen and result.follow_up_ok
    assert len(seen) == 2
