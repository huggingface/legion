"""Worker side of `providers:`: the harness environment (no worker secrets leak), the provider
key handed to the harness, and each adapter's provider configuration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from flock.models import SwarmSpec, TypeSpec
from flock.worker.adapters.base import minimal_env
from flock.worker.adapters.claude import DROP_VAR, ClaudeAdapter, claude_env
from flock.worker.adapters.codex import CodexAdapter, codex_config_args
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox

from .test_worker_adapters import FAKE_APP_SERVER
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
    ts = typespec(w={"harness": "codex", "model": "hf:Qwen/Q", "secrets": ["GITHUB_TOKEN"],
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
    ts = typespec(w={"harness": "codex", "model": "hf:Qwen/Q"})["w"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                      adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(), secret_source={})
    assert await rt.harness_env() == {}
    err = next(e for e in rt.outbox._q if e.type == "error")
    assert "HF_TOKEN is not set on the worker" in err.data["data"]["error"]


# --------------------------------------------------------------------------- codex


def test_codex_config_args_for_a_provider():
    ts = typespec(w={"harness": "codex", "model": "hf:Qwen/Qwen3.8-27B:ovhcloud", "reasoning": "off"})["w"]
    args = codex_config_args(ts)
    assert args[::2] == ["-c"] * (len(args) // 2)
    sets = dict(a.split("=", 1) for a in args[1::2])
    assert sets == {
        "model_providers.hf.name": '"hf"',
        "model_providers.hf.base_url": '"https://router.example/v1"',
        "model_providers.hf.env_key": '"HF_TOKEN"',
        "model_providers.hf.wire_api": '"responses"',
        "model_providers.hf.request_max_retries": "3",
        "model_providers.hf.stream_max_retries": "3",
        "model_providers.hf.stream_idle_timeout_ms": "45000",
        "model_provider": '"hf"',
        "model": '"Qwen/Qwen3.8-27B:ovhcloud"',
        # OpenAI-only tool types off: the HF router rejects the request when they are sent
        "features.multi_agent": "false",
        "web_search": '"disabled"',
        "model_reasoning_effort": '"none"',
    }


def test_toml_literals():
    from flock.worker.adapters.codex import _toml
    assert (_toml(True), _toml(False), _toml(3), _toml("a\"b")) == ("true", "false", "3", '"a\\"b"')


def test_codex_config_args_reserved_name_native_and_reasoning():
    ts = typespec(o={"harness": "codex", "model": "openai:gpt-x"})["o"]
    sets = dict(a.split("=", 1) for a in codex_config_args(ts)[1::2])
    assert sets["model_provider"] == '"flock-openai"'  # Codex refuses to redefine `openai`
    assert sets["model_providers.flock-openai.name"] == '"openai"'
    assert codex_config_args(TypeSpec(harness="codex", model="gpt-5.5-codex")) == []
    assert codex_config_args(TypeSpec(harness="codex", model="gpt-5.5-codex", reasoning="high")) == [
        "-c", 'model_reasoning_effort="high"']


DUMPING_APP_SERVER = FAKE_APP_SERVER.replace(
    '''    if meth == "initialize":''',
    '''    if meth == "initialize":
        import os
        open(os.environ["DUMP"], "w").write(json.dumps({"env": dict(os.environ), "argv": sys.argv}))''',
).replace(
    '''send({"id": rid, "result": {"model": p["model"], "thread": {"id": "thr-1", "path": None}}})''',
    '''open(os.environ["DUMP"] + ".start", "w").write(json.dumps(p))
        send({"id": rid, "result": {"model": p["model"], "thread": {"id": "thr-1", "path": None}}})''',
)


async def test_codex_process_gets_no_worker_secrets(tmp_path, worker_env, monkeypatch):
    dump = tmp_path / "dump.json"
    script = tmp_path / "fake-codex"
    script.write_text(f"#!{sys.executable}\n" + DUMPING_APP_SERVER)
    script.chmod(0o755)
    wd = tmp_path / "wd"
    wd.mkdir()
    ts = typespec(w={"harness": "codex", "model": "hf:Qwen/Q"})["w"]
    a = CodexAdapter(wd, home=tmp_path / "codex", codex_bin=str(script))
    await a.start(wd, None, ts, env={"HF_TOKEN": "hf-secret", "DUMP": str(dump)})
    await a.stop()
    got = json.loads(dump.read_text())
    env = got["env"]
    assert "FLOCK_TOKEN" not in env and "VLLM_KEY" not in env and "GITHUB_TOKEN" not in env
    assert env["HF_TOKEN"] == "hf-secret" and env["CODEX_HOME"] == str(tmp_path / "codex")
    assert env["LC_ALL"] == "C.UTF-8"
    assert 'model_provider="hf"' in got["argv"]
    start = json.loads(Path(str(dump) + ".start").read_text())
    assert start["model"] == "Qwen/Q"


# --------------------------------------------------------------------------- claude


def test_claude_env_for_an_anthropic_provider(worker_env):
    ts = typespec(c={"harness": "claude", "model": "anth:Qwen/Q-anthropic"})["c"]
    env = claude_env(ts, {"VLLM_KEY": "vllm-secret", "A": "1"})
    assert env["ANTHROPIC_API_KEY"] == "vllm-secret" and "VLLM_KEY" not in env
    assert env["ANTHROPIC_BASE_URL"] == "http://vllm:8000"
    assert env["API_TIMEOUT_MS"] == "600000" and env["CLAUDE_CODE_MAX_RETRIES"] == "1"
    assert env["CLAUDE_CONFIG_DIR"] == "/state/claude" and env["A"] == "1"
    assert "FLOCK_TOKEN" not in env and "HF_TOKEN" not in env
    # default Anthropic endpoint: key only, no base URL
    ts = typespec(c={"harness": "claude", "model": "anthd:claude-opus-5-5"})["c"]
    env = claude_env(ts, {"MY_ANTHROPIC": "ant-secret"})
    assert env["ANTHROPIC_API_KEY"] == "ant-secret" and "ANTHROPIC_BASE_URL" not in env


def test_claude_env_bearer_auth(worker_env):
    ts = typespec(c={"harness": "claude", "model": "anthb:Qwen/Q"})["c"]
    env = claude_env(ts, {"VLLM_KEY": "vllm-secret"})
    assert env["ANTHROPIC_AUTH_TOKEN"] == "vllm-secret" and "ANTHROPIC_API_KEY" not in env
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example"


def test_claude_options_drop_worker_env_and_map_reasoning(tmp_path, worker_env):
    ts = typespec(c={"harness": "claude", "model": "anth:Qwen/Q-anthropic", "reasoning": "low"})["c"]
    a = ClaudeAdapter(tmp_path, home=tmp_path / "claude", cli_path="/bin/true")
    o = a._options(ts, "sid", False, {"VLLM_KEY": "vllm-secret"})
    assert o.model == "Qwen/Q-anthropic" and o.effort == "low" and o.thinking is None
    drop = set(o.env[DROP_VAR].split(","))
    assert {"FLOCK_TOKEN", "WORKER_INCARNATION", "HF_TOKEN", "GITHUB_TOKEN", "VLLM_KEY"} <= drop
    assert "FLOCK_TOKEN" not in o.env and o.env["ANTHROPIC_API_KEY"] == "vllm-secret"
    assert o.cli_path != "/bin/true" and Path(o.cli_path).is_file()  # the env wrapper
    off = typespec(c={"harness": "claude", "model": "claude-opus-5-5", "reasoning": "off"})["c"]
    o = a._options(off, "sid", False, {})
    assert o.thinking == {"type": "disabled"} and o.effort is None and o.model == "claude-opus-5-5"


FAKE_CLAUDE = r'''
import json, os, sys
if "-v" in sys.argv or "--version" in sys.argv:
    print("2.1.296 (Claude Code)"); sys.exit(0)
open(os.environ["DUMP"], "w").write(json.dumps({"env": dict(os.environ), "argv": sys.argv}))
sys.exit(1)
'''


async def test_claude_cli_gets_no_worker_secrets_through_the_real_sdk(tmp_path, worker_env, monkeypatch):
    """The SDK merges os.environ into the CLI's env; the wrapper must take it back out."""
    pytest.importorskip("claude_agent_sdk")
    dump = tmp_path / "dump.json"
    cli = tmp_path / "claude"
    cli.write_text(f"#!{sys.executable}\n" + FAKE_CLAUDE)
    cli.chmod(0o755)
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")
    wd = tmp_path / "wd"
    wd.mkdir()
    ts = typespec(c={"harness": "claude", "model": "anth:Qwen/Q-anthropic"})["c"]
    a = ClaudeAdapter(wd, home=tmp_path / "claude-home", cli_path=str(cli))
    with pytest.raises(Exception):  # noqa: B017 - the fake CLI exits instead of speaking the protocol
        await a.start(wd, None, ts, env={"VLLM_KEY": "vllm-secret", "DUMP": str(dump)})
    env = json.loads(dump.read_text())["env"]
    for leaked in ("FLOCK_TOKEN", "WORKER_INCARNATION", "HF_TOKEN", "GITHUB_TOKEN", "VLLM_KEY",
                   "MSGBOARD_INVITE_X", "CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", DROP_VAR):
        assert leaked not in env, leaked
    assert env["ANTHROPIC_API_KEY"] == "vllm-secret" and env["ANTHROPIC_BASE_URL"] == "http://vllm:8000"
    assert env["CLAUDE_CONFIG_DIR"] == "/state/claude" and env["LANG"] == "C.UTF-8"
    assert env["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"  # the SDK's own variables survive


async def test_zero_usage_turn_is_flagged_and_logged_once(tmp_path, caplog):
    from flock.models import TokenUsage
    from flock.worker.adapters.base import AdapterEvent

    def rt_for(model):
        ts = typespec(w={"harness": "codex", "model": model})["w"]
        return AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents",
                            adapter=ScriptedAdapter(tmp_path / "home"), outbox=Outbox(),
                            secret_source=WORKER_ENV)

    def turn(rt, usage):
        rt.on_adapter_event(AdapterEvent("turn_started"))
        rt.on_adapter_event(AdapterEvent("usage", model="Qwen/Q", usage=usage, cumulative=True))
        rt.on_adapter_event(AdapterEvent("turn_ended", status="completed"))
        return [e for e in rt.outbox._q if e.type == "turn_ended"][-1].data["data"]

    rt = rt_for("hf:Qwen/Q")
    with caplog.at_level("WARNING"):
        assert turn(rt, TokenUsage())["usage_missing"] is True
        assert turn(rt, TokenUsage())["usage_missing"] is True
    assert sum("zero reported token usage" in r.message for r in caplog.records) == 1  # once
    assert "usage_missing" not in turn(rt_for("hf:Qwen/Q"), TokenUsage(input=10, output=2))
    native = AgentRuntime("w-0002", "w", TypeSpec(harness="codex", model="gpt-5.5-codex"), swarm="s",
                          work_root=tmp_path / "agents2", adapter=ScriptedAdapter(tmp_path / "h2"),
                          outbox=Outbox(), secret_source={})
    assert "usage_missing" not in turn(native, TokenUsage())  # only provider-backed types


async def test_env_dropped_event_once_per_spec_rev(tmp_path):
    ts = typespec(w={"harness": "codex", "model": "hf:Qwen/Q"})["w"]
    adapter = ScriptedAdapter(tmp_path / "home")
    adapter.env_dropped_names = ["JUNK", "MY_TOKEN"]
    rt = AgentRuntime("w-0001", "w", ts, swarm="s", work_root=tmp_path / "agents", adapter=adapter,
                      outbox=Outbox(), secret_source={})
    rt._report_env_dropped()
    rt._report_env_dropped()
    evs = [e for e in rt.outbox._q if e.type == "env_dropped"]
    assert len(evs) == 1 and evs[0].data["data"]["names"] == ["JUNK", "MY_TOKEN"]
