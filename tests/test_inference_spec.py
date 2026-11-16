"""`inference:` in a swarm spec: the block, resolution and pinning (docs/open-models.md)."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from flock.models import HF_ROUTER_URL, Inference, SwarmSpec, TypeSpec

from .conftest import HB

URLS = {"urls": ["http://vllm-a.example:8000/v1", "http://vllm-b.example:8000/v1"], "max_working": 3}


def spec(types: dict, inference: dict | None = None, **extra) -> SwarmSpec:
    return SwarmSpec.model_validate({
        "swarm": "s", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        **({"inference": inference} if inference is not None else {}), "types": types, **extra})


def err(types: dict, inference: dict | None = None, **extra) -> str:
    with pytest.raises(ValidationError) as e:
        spec(types, inference, **extra)
    return str(e.value)


def test_the_hf_router_is_the_default():
    r = spec({"a": {**HB, "model": "Qwen/Qwen3.8-27B"}}).types["a"].resolved_model
    assert r.inference.hf_router is not None and r.inference.endpoints == [HF_ROUTER_URL]
    assert r.request_model == r.model_id == "Qwen/Qwen3.8-27B"
    assert r.inference.key_env == "HF_TOKEN" and r.provider_name == "inference"
    assert r.pricing == [0, 0, 0]


@pytest.mark.parametrize(("provider", "sent", "policy"), [
    ("deepinfra", "Qwen/Qwen3.8-27B:deepinfra", False), ("cheapest", "Qwen/Qwen3.8-27B:cheapest", True)])
def test_a_router_provider_or_policy_is_added_to_the_model(provider, sent, policy):
    s = spec({"a": {**HB, "model": "Qwen/Qwen3.8-27B", "pricing": [1, 2, 0.5]}},
             {"hf_router": {"provider": provider, "bill_to": "huggingface"}})
    r = s.types["a"].resolved_model
    assert r.request_model == sent and r.inference.hf_router.policy is policy
    assert r.model_id == "Qwen/Qwen3.8-27B" and r.pricing == [1, 2, 0.5]  # priced as the Hub id
    assert r.inference.headers == {"X-HF-Bill-To": "huggingface"}


@pytest.mark.parametrize("model", ["huggingface/Qwen/Qwen3.8-27B", "Qwen/Qwen3.8-27B:deepinfra"])
def test_the_router_takes_a_plain_hub_id(model):
    msg = err({"a": {**HB, "model": model}})
    assert "types.a.model" in msg and "inference.hf_router.provider" in msg


def test_servers_you_run_take_any_name_and_no_key():
    r = spec({"a": {**HB, "model": "qwen3.5:2b"}}, URLS).types["a"].resolved_model
    assert r.request_model == "qwen3.5:2b" and r.inference.key_env is None
    assert r.inference.endpoints == URLS["urls"] and r.inference.headers == {}


def test_agents_spread_over_several_urls():
    one = Inference(urls=["http://a:8000/v1", "http://b:8000/v1", "http://c:8000/v1"])
    agents = [f"agent-{i:03d}" for i in range(60)]
    picked = {a: one.endpoint_for(a) for a in agents}
    assert set(picked.values()) == set(one.urls)  # every server gets agents
    assert all(one.endpoint_for(a) == u for a, u in picked.items())  # stable
    fewer = Inference(urls=one.urls[:2])  # a server removed: only its agents move
    assert all(fewer.endpoint_for(a) == u for a, u in picked.items() if u in fewer.urls)


@pytest.mark.parametrize(("inference", "needle"), [
    ({"urls": ["http://a/v1"], "hf_router": {}}, "set hf_router or urls, not both"),
    ({"urls": ["ftp://a/v1"]}, "HTTP(S) URL without credentials"),
    ({"urls": ["http://user:pw@a/v1"]}, "HTTP(S) URL without credentials"),
    ({"urls": ["http://a/v1", "http://a/v1/"]}, "must not repeat a URL"),
    ({"urls": []}, "at least one URL"),
    ({"metrics": True}, "it needs urls"),
    ({"api_key_env": "hf_abc-123"}, "env var name"),
    ({"max_working": 0}, "greater than or equal to 1"),
    ({"hf_router": {"provider": "deepinfra:fast"}}, "must be a name"),
    ({"base_url": "http://a/v1"}, "Extra inputs are not permitted"),
])
def test_block_validation(inference, needle):
    assert needle in err({"a": {**HB, "model": "m"}}, inference)


def test_a_type_runs_on_its_own_block():
    s = spec({"a": {**HB, "model": "m"}, "b": {**HB, "model": "m", "inference": URLS}},
             {"hf_router": {"provider": "deepinfra"}})
    assert s.types["a"].provider_name == "inference"
    assert s.types["b"].provider_name == "types.b.inference"
    assert s.inference_blocks()["types.b.inference"].max_working == 3


@pytest.mark.parametrize(("compute", "backend", "url", "needle"), [
    ("hf-jobs", "hf-sandbox", "http://localhost:8000/v1", "cannot reach"),
    ("local", "hf-sandbox", "http://10.53.0.7:8000/v1", "cannot reach"),
    ("slurm", "podman", "http://127.0.0.1:8000/v1", "each worker's own node"),
])
def test_agents_must_reach_the_servers(compute, backend, url, needle):
    t = {**HB, "model": "m", "harbor": {**HB["harbor"], "config": {
        "task": {}, "environment": {"backend": backend}}}}
    msg = err({"a": t}, {"urls": [url]}, compute={"kind": compute})
    assert "inference.urls" in msg and needle in msg
    if compute == "slurm":  # a cluster address is fine there
        spec({"a": t}, {"urls": ["http://10.53.0.7:8000/v1"]}, compute={"kind": compute})


def test_the_pinned_context_window_stays_while_block_and_model_do():
    s = spec({"a": {**HB, "model": "m"}}, URLS)
    s.types["a"].resolved_model.context_window = 32768  # as the controller's apply pins it
    stored = SwarmSpec.model_validate_json(s.model_dump_json())
    assert stored.types["a"].resolved_model.context_window == 32768
    moved = json.loads(s.model_dump_json())
    moved["inference"]["urls"] = ["http://vllm-c.example:8000/v1"]
    assert SwarmSpec.model_validate(moved).types["a"].resolved_model.context_window is None


def test_hand_written_resolved_model_is_replaced():
    raw = {**HB, "model": "Qwen/Qwen3.8-27B", "pricing": [0.3, 1.2, 0.3],
           "resolved_model": {"provider_name": "x", "inference": URLS, "model_id": "evil",
                              "pricing": [0, 0, 0]}}
    r = spec({"w": raw}).types["w"].resolved_model
    assert (r.model_id, r.pricing, r.provider_name) == ("Qwen/Qwen3.8-27B", [0.3, 1.2, 0.3],
                                                        "inference")


def test_standalone_typespec_is_unresolved():
    ts = TypeSpec(**HB, model="Qwen/Qwen3.8-27B")
    assert ts.resolved_model is None and ts.harness_model == "Qwen/Qwen3.8-27B"


def test_a_stored_swarm_that_does_not_load_is_skipped(tmp_path, caplog):
    from flock.controller.store import Store

    store = Store(tmp_path)
    good = spec({"w": {**HB, "model": "m"}}, URLS)
    store.put_swarm(good, 0)
    parsed = store.get_swarm(good.swarm)["spec"]  # parsed once per stored JSON
    assert store.get_swarm(good.swarm)["spec"] is parsed is store.list_swarms()[0]["spec"]
    # a swarm stored before `inference:` (its `providers:` dropped, its model prefixed) does not load
    old = good.model_dump(mode="json") | {"swarm": "old", "providers": {"inference": {
        "type": "openai_compat", "base_url": HF_ROUTER_URL, "api_key_env": "HF_TOKEN"}}}
    old.pop("inference")
    old["types"]["w"]["model"] = "inference:huggingface/Qwen/Qwen3.8-27B:deepinfra"
    store._x("INSERT INTO swarms (name, spec, applied_at) VALUES (?, ?, 0)",
             ["old", json.dumps(old)])
    with caplog.at_level("ERROR", logger="flock.controller.store"):
        assert [r["name"] for r in store.list_swarms()] == [good.swarm]
        assert [r["name"] for r in store.list_swarms()] == [good.swarm]  # logged once
    assert len(caplog.records) == 1 and "swarm old: stored spec" in caplog.text
    with pytest.raises(ValidationError):
        store.get_swarm("old")
    store.close()
