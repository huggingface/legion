"""Worker side of `providers:`: the harness environment (no worker secrets leak), the provider
key handed to the harness."""

from __future__ import annotations

import pytest

from flock.models import SwarmSpec, TypeSpec
from flock.worker.adapters.base import minimal_env
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox

from .worker_fakes import ScriptedAdapter, until

PROVIDERS = {
    "hf": {"type": "openai_compat", "base_url": "https://router.example/v1", "api_key_env": "HF_TOKEN",
           "pricing": {"*": [1, 2, 0.5]}, "max_retries": 3, "timeout_s": 45},
    "openai": {"type": "openai_compat", "base_url": "https://api.openai.example/v1",
               "api_key_env": "OPENAI_API_KEY", "pricing": {"*": [1, 2, 0.5]}},
    "anth": {"type": "anthropic", "base_url": "http://vllm:8000", "api_key_env": "VLLM_KEY",
             "self_hosted": True, "timeout_s": 600, "max_retries": 1},
    "anthd": {"type": "anthropic", "api_key_env": "MY_ANTHROPIC", "pricing": {"*": [4, 20, 0.2]}},
    "anthb": {"type": "anthropic", "base_url": "https://gw.example", "api_key_env": "VLLM_KEY",
              "self_hosted": True, "auth": "bearer"},
}
WORKER_ENV = {
    "PATH": "/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "xterm",
    "FLOCK_TOKEN": "flock-secret", "WORKER_INCARNATION": "inc", "HF_TOKEN": "hf-secret",
    "VLLM_KEY": "vllm-secret", "MY_ANTHROPIC": "ant-secret", "GITHUB_TOKEN": "gh-secret",
    "MSGBOARD_INVITE_X": "invite", "CLAUDE_CONFIG_DIR": "/state/claude", "CODEX_HOME": "/state/codex",
}


def typespec(**types) -> dict[str, TypeSpec]:
    return SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket": "b",
                                     "providers": PROVIDERS, "types": types}).types


@pytest.fixture
def worker_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k not in ("PATH", "HOME", "TMPDIR", "PYTHONPATH", "VIRTUAL_ENV"):
            monkeypatch.delenv(k, raising=False)
    for k, v in WORKER_ENV.items():
        if k not in ("PATH", "HOME"):
            monkeypatch.setenv(k, v)
    return WORKER_ENV


# --------------------------------------------------------------------------- base


def test_minimal_env_is_an_allowlist():
    env = minimal_env(WORKER_ENV, {"X": "1", "HF_TOKEN": "explicit"}, keep=("CODEX_HOME",))
    assert env["PATH"] == "/usr/bin:/bin" and env["LC_ALL"] == "C.UTF-8" and env["X"] == "1"
    assert env["CODEX_HOME"] == "/state/codex" and env["HF_TOKEN"] == "explicit"
    for leaked in ("FLOCK_TOKEN", "WORKER_INCARNATION", "VLLM_KEY", "GITHUB_TOKEN",
                   "MSGBOARD_INVITE_X", "CLAUDE_CONFIG_DIR"):
        assert leaked not in env


# --------------------------------------------------------------------------- runtime


async def test_runtime_hands_over_declared_secrets_and_provider_key(tmp_path):
    ts = typespec(w={"harness": "llm", "model": "hf:Qwen/Q", "secrets": ["GITHUB_TOKEN"],
                     "env": {"A": "1"}})["w"]
    adapter = ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents", adapter=adapter,
                      outbox=Outbox(), secret_source=WORKER_ENV)
    assert await rt.harness_env() == {"GITHUB_TOKEN": "gh-secret", "A": "1", "HF_TOKEN": "hf-secret"}
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: adapter.envs)
    await rt.stop()
    assert "FLOCK_TOKEN" not in adapter.envs[0] and adapter.envs[0]["HF_TOKEN"] == "hf-secret"
    started = next(e for e in rt.outbox._q if e.type == "session_started")
    assert started.data["data"]["provider"] == "hf" and started.data["data"]["model_id"] == "Qwen/Q"


async def test_runtime_reports_a_missing_provider_key(tmp_path):
    ts = typespec(w={"harness": "llm", "model": "hf:Qwen/Q"})["w"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                      adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(), secret_source={})
    assert await rt.harness_env() == {}
    err = next(e for e in rt.outbox._q if e.type == "error")
    assert "HF_TOKEN is not set on the worker" in err.data["data"]["error"]


async def test_zero_usage_turn_is_flagged_and_logged_once(tmp_path, caplog):
    from flock.models import TokenUsage
    from flock.worker.adapters.base import AdapterEvent

    def rt_for(model):
        ts = typespec(w={"harness": "llm", "model": model})["w"]
        return AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                            adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(),
                            secret_source=WORKER_ENV)

    def turn(rt, usage):
        rt.on_adapter_event(AdapterEvent("turn_started"))
        rt.on_adapter_event(AdapterEvent("usage", model="Qwen/Q", usage=usage))
        rt.on_adapter_event(AdapterEvent("turn_ended", status="completed"))
        return [e for e in rt.outbox._q if e.type == "turn_ended"][-1].data["data"]

    rt = rt_for("hf:Qwen/Q")
    with caplog.at_level("WARNING"):
        assert turn(rt, TokenUsage())["usage_missing"] is True
        assert turn(rt, TokenUsage())["usage_missing"] is True
    assert sum("zero reported token usage" in r.message for r in caplog.records) == 1  # once
    assert "usage_missing" not in turn(rt_for("hf:Qwen/Q"), TokenUsage(input=10, output=2))
    native = AgentRuntime("w-0002", "w", TypeSpec(harness="llm", model="gpt-5.5-codex"), swarm="s",
                          work_root=tmp_path / "agents2", adapter=ScriptedAdapter(tmp_path / "h2"),
                          outbox=Outbox(), secret_source={})
    assert "usage_missing" not in turn(native, TokenUsage())  # only provider-backed types


async def test_env_dropped_event_once_per_spec_rev(tmp_path):
    ts = typespec(w={"harness": "llm", "model": "hf:Qwen/Q"})["w"]
    adapter = ScriptedAdapter(tmp_path / "home")
    adapter.env_dropped_names = ["JUNK", "MY_TOKEN"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents", adapter=adapter,
                      outbox=Outbox(), secret_source={})
    rt._report_env_dropped()
    rt._report_env_dropped()
    evs = [e for e in rt.outbox._q if e.type == "env_dropped"]
    assert len(evs) == 1 and evs[0].data["data"]["names"] == ["JUNK", "MY_TOKEN"]


def test_heartbeat_carries_the_pinned_server(tmp_path):
    from flock.worker.adapters.llm import pick_endpoint

    ts = SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b", "providers": {"v": {
            "type": "openai_compat", "base_urls": ["http://g1:8000/v1", "http://g2:8000/v1"],
            "api_key_env": "K", "self_hosted": True}},
        "types": {"w": {"harness": "llm", "model": "v:Qwen/Q"}}}).types["w"]
    adapter = ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                      adapter=adapter, outbox=Outbox(), secret_source={})
    want = pick_endpoint("w-0001", ["http://g1:8000/v1", "http://g2:8000/v1"])
    assert rt.heartbeat().endpoint == want.split("/")[2]  # before any turn: where it will go
    adapter._endpoint = "http://g1:8000/v1"  # the llm adapter: where its last turn went
    assert rt.heartbeat().endpoint == "g1:8000"
    native = AgentRuntime("w-0002", "w", TypeSpec(harness="llm", model="gpt-5.5-codex"),
                          swarm="s", work_root=tmp_path / "a2", adapter=ScriptedAdapter(tmp_path / "h2"),
                          outbox=Outbox(), secret_source={})
    assert native.heartbeat().endpoint is None
