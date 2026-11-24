"""Worker side of `inference:`: the harness environment (no worker secrets leak), the inference
key handed to the harness, the server an agent calls."""

from __future__ import annotations

import pytest

from legion.models import SwarmSpec, TypeSpec
from legion.worker.agents import AgentRuntime
from legion.worker.events import Outbox

from .conftest import HB
from .worker_fakes import ScriptedAdapter, until

GPU = {"urls": ["http://g1:8000/v1", "http://g2:8000/v1"], "api_key_env": "K"}
WORKER_ENV = {
    "PATH": "/usr/bin:/bin", "HOME": "/root", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "TERM": "xterm",
    "LEGION_TOKEN": "legion-secret", "WORKER_INCARNATION": "inc", "HF_TOKEN": "hf-secret",
    "VLLM_KEY": "vllm-secret", "MY_ANTHROPIC": "ant-secret", "GITHUB_TOKEN": "gh-secret",
    "MSGBOARD_INVITE_X": "invite",
}


def typespec(**types) -> dict[str, TypeSpec]:
    return SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "bucket_id": "b",
                                     "types": types}).types


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


# --------------------------------------------------------------------------- runtime


async def test_runtime_hands_over_declared_secrets_and_the_inference_key(tmp_path):
    ts = typespec(w={**HB, "model": "Qwen/Q", "secrets": ["GITHUB_TOKEN"],
                     "env": {"A": "1"}})["w"]
    adapter = ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents", adapter=adapter,
                      outbox=Outbox(), secret_source=WORKER_ENV)
    assert await rt.harness_env() == {"GITHUB_TOKEN": "gh-secret", "A": "1", "HF_TOKEN": "hf-secret"}
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: adapter.envs)
    await rt.stop()
    assert "LEGION_TOKEN" not in adapter.envs[0] and adapter.envs[0]["HF_TOKEN"] == "hf-secret"
    started = next(e for e in rt.outbox._q if e.type == "session_started")
    assert started.data["data"]["provider"] == "inference"
    assert started.data["data"]["model_id"] == "Qwen/Q"


async def test_runtime_reports_a_missing_key(tmp_path):
    ts = typespec(w={**HB, "model": "Qwen/Q"})["w"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                      adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(), secret_source={})
    assert await rt.harness_env() == {}
    err = next(e for e in rt.outbox._q if e.type == "error")
    assert "HF_TOKEN is not set on the worker" in err.data["data"]["error"]
    keyless = typespec(w={**HB, "model": "m", "inference": {"urls": GPU["urls"]}})["w"]
    rt = AgentRuntime("w-0002", "w", keyless, swarm="s", work_root=tmp_path / "a2",
                      adapter=ScriptedAdapter(tmp_path / "h2"), outbox=Outbox(), secret_source={})
    assert await rt.harness_env() == {} and not [e for e in rt.outbox._q if e.type == "error"]


async def test_zero_usage_turn_is_flagged_and_logged_once(tmp_path, caplog):
    from legion.models import TokenUsage
    from legion.worker.adapters.base import AdapterEvent

    def rt_for(model):
        ts = typespec(w={**HB, "model": model})["w"]
        return AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                            adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(),
                            secret_source=WORKER_ENV)

    def turn(rt, usage):
        rt.on_adapter_event(AdapterEvent("turn_started"))
        rt.on_adapter_event(AdapterEvent("usage", model="Qwen/Q", usage=usage))
        rt.on_adapter_event(AdapterEvent("turn_ended", status="completed"))
        return [e for e in rt.outbox._q if e.type == "turn_ended"][-1].data["data"]

    rt = rt_for("Qwen/Q")
    with caplog.at_level("WARNING"):
        assert turn(rt, TokenUsage())["usage_missing"] is True
        assert turn(rt, TokenUsage())["usage_missing"] is True
    assert sum("zero reported token usage" in r.message for r in caplog.records) == 1  # once
    assert "usage_missing" not in turn(rt_for("Qwen/Q"), TokenUsage(input=10, output=2))
    native = AgentRuntime("w-0002", "w", TypeSpec(**HB, model="gpt-5.5-codex"), swarm="s",
                          work_root=tmp_path / "agents2", adapter=ScriptedAdapter(tmp_path / "h2"),
                          outbox=Outbox(), secret_source={})
    assert "usage_missing" not in turn(native, TokenUsage())  # only provider-backed types


def test_heartbeat_carries_the_model_server(tmp_path):
    ts = typespec(w={**HB, "model": "Qwen/Q", "inference": GPU})["w"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                      adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(), secret_source={})
    picked = ts.resolved_model.inference.endpoint_for("s", "w-0001")  # the server its Trial calls
    assert rt.heartbeat().endpoint == picked.removeprefix("http://").removesuffix("/v1")
    native = AgentRuntime("w-0002", "w", TypeSpec(**HB, model="gpt-5.5-codex"),
                          swarm="s", work_root=tmp_path / "a2", adapter=ScriptedAdapter(tmp_path / "h2"),
                          outbox=Outbox(), secret_source={})
    assert native.heartbeat().endpoint is None
