"""`providers:` / `models:` in the spec: resolution, apply-time validation, pinning."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flock.models import SwarmSpec, TypeSpec, resolve_model

EXAMPLE = Path(__file__).parent.parent / "examples" / "open-models.yaml"

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


def test_example_spec_resolves():
    s = SwarmSpec.model_validate(yaml.safe_load(EXAMPLE.read_text()))
    w, a, wr = s.types["worker"], s.types["analyst"], s.types["writer"]
    assert (w.provider_name, w.harness_model, w.resolved_model.pricing) == (
        "hf", "Qwen/Qwen3.8-27B:deepinfra", [0.2, 2.5, 0.2])  # router provider pinned
    assert (a.provider_name, a.harness_model, a.effective_reasoning) == (
        "gpu1", "Qwen/Qwen3.8-27B-FP8", "medium")
    assert a.resolved_model.extra == {"chat_template_kwargs": {"enable_thinking": False}}
    assert wr.resolved_model is None and wr.harness_model == "claude-opus-5-5"


def test_resolution_order_alias_then_prefix_then_native():
    s = spec({
        "a": {"harness": "codex", "model": "qwen-27b"},
        "b": {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B"},
        "c": {"harness": "codex", "model": "gpt-5.5-codex"},
        "d": {"harness": "claude", "model": "anth:claude-opus-5-5"},
        "e": {"harness": "claude", "model": "anthropic.claude-opus-v1:0"},  # Bedrock-style id
    })
    assert s.types["a"].resolved_model.qualified == "gpu1:Qwen/Qwen3.8-27B-FP8"
    assert s.types["b"].resolved_model.qualified == "hf:Qwen/Qwen3.8-27B"
    assert s.types["c"].resolved_model is None
    assert s.types["d"].harness_model == "claude-opus-5-5"
    assert s.types["e"].resolved_model is None
    assert resolve_model(s, s.types["b"]).provider.base_url == "https://router.huggingface.co/v1"


def test_type_reasoning_overrides_alias():
    s = spec({"a": {"harness": "codex", "model": "qwen-27b", "reasoning": "off"},
              "n": {"harness": "codex", "model": "gpt-5.5-codex", "reasoning": "high"}})
    assert s.types["a"].effective_reasoning == "off"
    assert s.types["n"].effective_reasoning == "high"


def test_router_suffix_stays_in_model_id():
    s = spec({"a": {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B:ovhcloud"}},
             providers={**PROVIDERS, "hf": {**PROVIDERS["hf"], "pricing": {"*": [1, 2, 0.5]}}})
    r = s.types["a"].resolved_model
    assert r.model_id == "Qwen/Qwen3.8-27B:ovhcloud" and r.pricing == [1, 2, 0.5]


def test_unknown_prefix_names_type_and_field():
    msg = err({"worker": {"harness": "codex", "model": "gpu9:Qwen/X"}})
    assert "types.worker.model" in msg and "unknown provider 'gpu9'" in msg


def test_models_entry_with_unknown_prefix():
    msg = err({"a": {"harness": "codex", "model": "gpt-5.5-codex"}},
              models={"q": {"model": "gpu9:Qwen/X"}})
    assert "models.q.model" in msg and "unknown provider 'gpu9'" in msg


def test_models_entry_must_be_prefixed():
    msg = err({"a": {"harness": "codex", "model": "q"}}, models={"q": {"model": "Qwen/X"}})
    assert "models.q.model" in msg and "<provider>:<model id>" in msg


def test_missing_pricing_is_an_error_unless_self_hosted():
    msg = err({"worker": {"harness": "codex", "model": "hf:Qwen/Other"}})
    assert "types.worker.model" in msg and "no pricing" in msg and "providers.hf.pricing" in msg
    msg = err({"worker": {"harness": "codex", "model": "q"}},
              models={"q": {"model": "hf:Qwen/Other"}})
    assert "models.q.pricing" in msg
    # alias pricing satisfies it and overrides the provider's
    s = spec({"w": {"harness": "codex", "model": "q"}},
             models={"q": {"model": "hf:Qwen/Qwen3.8-27B", "pricing": [9, 9, 9]}})
    assert s.types["w"].resolved_model.pricing == [9, 9, 9]


def test_self_hosted_default_price_is_pinned_into_the_spec():
    s = spec({"w": {"harness": "codex", "model": "gpu1:Qwen/Small"}})
    assert s.types["w"].resolved_model.pricing == [0, 0, 0]
    assert s.providers["gpu1"].pricing["Qwen/Small"] == [0, 0, 0]
    # the stored (dumped) spec round-trips with the pins, and re-validation is stable
    again = SwarmSpec.model_validate_json(s.model_dump_json())
    assert again.types["w"].resolved_model == s.types["w"].resolved_model
    assert again.providers["gpu1"].pricing == {"Qwen/Small": [0, 0, 0]}


def test_hand_written_resolved_model_is_replaced():
    raw = {"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B",
           "resolved_model": {"provider_name": "x", "provider": PROVIDERS["gpu1"],
                              "model_id": "evil", "pricing": [0, 0, 0]}}
    s = spec({"w": raw})
    assert s.types["w"].resolved_model.model_id == "Qwen/Qwen3.8-27B"
    assert s.types["w"].resolved_model.pricing == [0.3, 1.2, 0.3]


@pytest.mark.parametrize(("harness", "model", "needs"), [
    ("claude", "hf:Qwen/Qwen3.8-27B", "anthropic"),
    ("claude", "qwen-27b", "anthropic"),
    ("codex", "anth:claude-opus-5-5", "openai_compat"),
])
def test_harness_provider_compatibility(harness, model, needs):
    msg = err({"t1": {"harness": harness, "model": model}})
    assert "types.t1.model" in msg and f"needs a provider of type {needs}" in msg


def test_native_models_are_always_fine():
    s = spec({"c": {"harness": "claude", "model": "claude-opus-5-5"},
              "x": {"harness": "codex", "model": "gpt-5.5-codex"}}, providers={}, models={})
    assert all(t.resolved_model is None for t in s.types.values())


@pytest.mark.parametrize(("provider", "needle"), [
    ({"type": "openai_compat", "api_key_env": "K"}, "base_url is required"),
    ({"type": "anthropic", "api_key_env": "sk-ant-123"}, "env var name"),
    ({"type": "anthropic", "api_key_env": "K", "pricing": {"m": [1, 2]}}, "[input, output, cached]"),
    ({"type": "anthropic", "api_key_env": "K", "max_working": 0}, "max_working must be >= 1"),
    ({"type": "vertex", "api_key_env": "K"}, "openai_compat"),
])
def test_provider_field_validation(provider, needle):
    msg = err({"c": {"harness": "claude", "model": "claude-opus-5-5"}},
              providers={"p": provider}, models={})
    assert needle in msg


def test_provider_and_alias_names():
    msg = err({"c": {"harness": "claude", "model": "claude-opus-5-5"}},
              providers={"my.vllm": PROVIDERS["gpu1"]}, models={})
    assert "providers.my.vllm" in msg
    msg = err({"c": {"harness": "codex", "model": "gpt-5.5-codex"}},
              models={"a:b": {"model": "gpu1:x"}})
    assert "models.a:b" in msg


def test_standalone_typespec_is_native():
    ts = TypeSpec(harness="codex", model="hf:Qwen/Qwen3.8-27B")
    assert ts.resolved_model is None and ts.harness_model == "hf:Qwen/Qwen3.8-27B"


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
