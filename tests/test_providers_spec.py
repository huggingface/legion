"""`providers:` / `models:` in the spec: resolution, apply-time validation, pinning."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from flock.models import SwarmSpec, TypeSpec, resolve_model

from .conftest import HB

PROVIDERS = {
    "hf": {"type": "openai_compat", "base_url": "https://router.huggingface.co/v1/",
           "api_key_env": "HF_TOKEN", "pricing": {"Qwen/Qwen3.8-27B": [0.3, 1.2, 0.3]}},
    "gpu1": {"type": "openai_compat", "base_url": "http://qwen-job:8000/v1",
             "api_key_env": "VLLM_API_KEY", "self_hosted": True, "max_working": 256},
    "anth": {"type": "anthropic", "api_key_env": "ANTHROPIC_API_KEY",
             "pricing": {"claude-opus-5-5": [4, 20, 0.2]}},
}
MODELS = {"qwen-27b": {"model": "gpu1:Qwen/Qwen3.8-27B-FP8", "reasoning": "medium",
                       "extra": {"chat_template_kwargs": {"enable_thinking": False}}}}


def spec(types: dict, providers: dict | None = None, models: dict | None = None) -> SwarmSpec:
    return SwarmSpec.model_validate({
        "swarm": "s", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": PROVIDERS if providers is None else providers,
        "models": MODELS if models is None else models,
        "types": types,
    })


def err(types: dict, **kw) -> str:
    with pytest.raises(ValidationError) as e:
        spec(types, **kw)
    return str(e.value)


def test_resolution_order_alias_then_prefix():
    s = spec({
        "a": {**HB, "model": "qwen-27b"},
        "b": {**HB, "model": "hf:Qwen/Qwen3.8-27B"},
    })
    assert s.types["a"].resolved_model.qualified == "gpu1:Qwen/Qwen3.8-27B-FP8"
    assert s.types["b"].resolved_model.qualified == "hf:Qwen/Qwen3.8-27B"
    assert resolve_model(s, s.types["b"]).provider.base_url == "https://router.huggingface.co/v1"


def test_type_reasoning_overrides_alias():
    s = spec({"a": {**HB, "model": "qwen-27b", "reasoning": "off"}})
    assert s.types["a"].effective_reasoning == "off"


def test_router_suffix_stays_in_model_id():
    s = spec({"a": {**HB, "model": "hf:Qwen/Qwen3.8-27B:ovhcloud"}},
             providers={**PROVIDERS, "hf": {**PROVIDERS["hf"], "pricing": {"*": [1, 2, 0.5]}}})
    r = s.types["a"].resolved_model
    assert r.model_id == "Qwen/Qwen3.8-27B:ovhcloud" and r.pricing == [1, 2, 0.5]


def test_unknown_prefix_names_type_and_field():
    msg = err({"worker": {**HB, "model": "gpu9:Qwen/X"}})
    assert "types.worker.model" in msg and "unknown provider 'gpu9'" in msg


def test_models_entry_with_unknown_prefix():
    msg = err({"a": {**HB, "model": "gpt-5.5-codex"}},
              models={"q": {"model": "gpu9:Qwen/X"}})
    assert "models.q.model" in msg and "unknown provider 'gpu9'" in msg


def test_models_entry_must_be_prefixed():
    msg = err({"a": {**HB, "model": "q"}}, models={"q": {"model": "Qwen/X"}})
    assert "models.q.model" in msg and "<provider>:<model id>" in msg


def test_missing_pricing_is_an_error_unless_self_hosted():
    msg = err({"worker": {**HB, "model": "hf:Qwen/Other"}})
    assert "types.worker.model" in msg and "no pricing" in msg and "providers.hf.pricing" in msg
    msg = err({"worker": {**HB, "model": "q"}},
              models={"q": {"model": "hf:Qwen/Other"}})
    assert "models.q.pricing" in msg
    # alias pricing satisfies it and overrides the provider's
    s = spec({"w": {**HB, "model": "q"}},
             models={"q": {"model": "hf:Qwen/Qwen3.8-27B", "pricing": [9, 9, 9]}})
    assert s.types["w"].resolved_model.pricing == [9, 9, 9]


def test_self_hosted_default_price_is_pinned_into_the_spec():
    s = spec({"w": {**HB, "model": "gpu1:Qwen/Small"}})
    assert s.types["w"].resolved_model.pricing == [0, 0, 0]
    assert s.providers["gpu1"].pricing["Qwen/Small"] == [0, 0, 0]
    # the stored (dumped) spec round-trips with the pins, and re-validation is stable
    again = SwarmSpec.model_validate_json(s.model_dump_json())
    assert again.types["w"].resolved_model == s.types["w"].resolved_model
    assert again.providers["gpu1"].pricing == {"Qwen/Small": [0, 0, 0]}


def test_hand_written_resolved_model_is_replaced():
    raw = {**HB, "model": "hf:Qwen/Qwen3.8-27B",
           "resolved_model": {"provider_name": "x", "provider": PROVIDERS["gpu1"],
                              "model_id": "evil", "pricing": [0, 0, 0]}}
    s = spec({"w": raw})
    assert s.types["w"].resolved_model.model_id == "Qwen/Qwen3.8-27B"
    assert s.types["w"].resolved_model.pricing == [0.3, 1.2, 0.3]


def test_harness_provider_compatibility():
    msg = err({"t1": {**HB, "model": "anth:claude-opus-5-5"}})
    assert "types.t1.model" in msg and "needs a provider of type openai_compat" in msg


@pytest.mark.parametrize("model", ["claude-opus-5-5", "anthropic.claude-opus-v1:0"])  # Bedrock-style
def test_a_model_without_a_provider_is_refused(model):
    msg = err({"c": {**HB, "model": model}}, providers={}, models={})
    assert "types.c.model" in msg and "names no provider" in msg


@pytest.mark.parametrize(("provider", "needle"), [
    ({"type": "openai_compat", "api_key_env": "K"}, "base_url is required"),
    ({"type": "anthropic", "api_key_env": "sk-ant-123"}, "env var name"),
    ({"type": "anthropic", "api_key_env": "K", "pricing": {"m": [1, 2]}}, "[input, output, cached]"),
    ({"type": "anthropic", "api_key_env": "K", "max_working": 0}, "max_working must be >= 1"),
    ({"type": "vertex", "api_key_env": "K"}, "openai_compat"),
])
def test_provider_field_validation(provider, needle):
    msg = err({"c": {**HB, "model": "claude-opus-5-5"}},
              providers={"p": provider}, models={})
    assert needle in msg


def test_provider_and_alias_names():
    msg = err({"c": {**HB, "model": "claude-opus-5-5"}},
              providers={"my.vllm": PROVIDERS["gpu1"]}, models={})
    assert "providers.my.vllm" in msg
    msg = err({"c": {**HB, "model": "gpt-5.5-codex"}},
              models={"a:b": {"model": "gpu1:x"}})
    assert "models.a:b" in msg


def test_standalone_typespec_is_unresolved():
    ts = TypeSpec(**HB, model="hf:Qwen/Qwen3.8-27B")
    assert ts.resolved_model is None and ts.harness_model == "hf:Qwen/Qwen3.8-27B"


def test_removed_typespec_keys_are_dropped_with_a_warning(monkeypatch, caplog):
    from flock import models

    monkeypatch.setattr(models, "_warned", set())
    with caplog.at_level("WARNING", logger="flock.models"):
        s = spec({"w": {**HB, "model": "qwen-27b", "tool_memory_mb": 512,
                        "node_heap_mb": None}})
    assert "tool_memory_mb" not in s.model_dump_json()
    assert [r.getMessage() for r in caplog.records] == [
        "spec: TypeSpec.tool_memory_mb was removed from flock; ignoring it"]


@pytest.mark.parametrize("harness", ["pi", "claude", "codex"])
def test_a_removed_harness_gets_a_clear_refusal(harness):
    msg = err({"w": {"harness": harness, "model": "qwen-27b"}})
    assert f"harness '{harness}' was removed from flock; relaunch the type as harbor" in msg


def test_a_stored_swarm_that_no_longer_loads_is_skipped(tmp_path, caplog):
    from flock.controller.store import Store

    store = Store(tmp_path)
    good = spec({"w": {**HB, "model": "qwen-27b"}})
    store.put_swarm(good, 0)
    old = good.model_dump(mode="json") | {"swarm": "old"}
    old["types"]["w"]["harness"] = "pi"
    store._x("INSERT INTO swarms (name, spec, applied_at) VALUES (?, ?, 0)",
             ["old", json.dumps(old)])
    with caplog.at_level("ERROR", logger="flock.controller.store"):
        assert [r["name"] for r in store.list_swarms()] == [good.swarm]
        assert [r["name"] for r in store.list_swarms()] == [good.swarm]
    assert len(caplog.records) == 1 and "swarm old: stored spec no longer loads" in caplog.text
    with pytest.raises(ValidationError, match="harness 'pi' was removed from flock"):
        store.get_swarm("old")
    store.close()


def test_auth_option():
    base = {"swarm": "s", "namespace": "n", "bucket": "b", "types": {}}
    ok = SwarmSpec.model_validate({**base, "providers": {
        "a": {"type": "anthropic", "api_key_env": "K", "auth": "bearer"}}})
    assert ok.providers["a"].auth == "bearer"
    with pytest.raises(ValidationError, match="openai_compat always sends a bearer token"):
        SwarmSpec.model_validate({**base, "providers": {
            "o": {"type": "openai_compat", "base_url": "http://x/v1", "api_key_env": "K",
                  "auth": "x-api-key"}}})
    with pytest.raises(ValidationError):
        SwarmSpec.model_validate({**base, "providers": {
            "a": {"type": "anthropic", "api_key_env": "K", "auth": "basic"}}})
