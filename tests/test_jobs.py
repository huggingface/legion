"""HubJobsClient launch arguments: bootstrap mode vs a prebuilt image."""

from __future__ import annotations

import subprocess

import pytest

from swarm_harness.controller import jobs as jobs_mod
from swarm_harness.controller.jobs import (
    BOOTSTRAP_IMAGE,
    CLAUDE_CODE_VERSION,
    CODEX_VERSION,
    WORKER_COMMAND,
    HubJobsClient,
    bootstrap_command,
)
from swarm_harness.controller.main import parse_labels
from swarm_harness.models import TypeSpec

WHEEL = "hf://buckets/ns/sh/dist/abc/swarm_harness-0.0.1-py3-none-any.whl"


def _spec(**kw) -> TypeSpec:
    return TypeSpec(harness="claude", model="claude-haiku-5-5", **kw)


def test_bootstrap_command_installs_pinned_tools_then_execs_worker(tmp_path):
    cmd = bootstrap_command()
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert f"@anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}" in script
    assert f"@openai/codex@{CODEX_VERSION}" in script
    assert "astral.sh/uv/install.sh" in script
    assert 'hf buckets cp "$SWARM_HARNESS_WHEEL_URL"' in script
    assert "download_bucket_files" in script  # Python API fallback
    assert 'uv tool install -q "$WHEEL[worker]"' in script and "pip install" in script
    for env in ("IS_SANDBOX=1", "CLAUDE_CONFIG_DIR=/state/claude", "CODEX_HOME=/state/codex"):
        assert env in script
    assert script.rstrip().endswith("exec swarm-harness-worker")
    assert "{" + "claude}" not in script and "{" + "hub}" not in script  # all pins filled in
    f = tmp_path / "bootstrap.sh"
    f.write_text(script)
    subprocess.run(["bash", "-n", str(f)], check=True)


def _client(monkeypatch, image, bootstrap_env=None, wheel=WHEEL) -> HubJobsClient:
    if bootstrap_env is None:
        monkeypatch.delenv("SWARM_HARNESS_WORKER_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv("SWARM_HARNESS_WORKER_BOOTSTRAP", bootstrap_env)
    return HubJobsClient(token="x", image=image, wheel_url=wheel)


def test_launch_args_bootstrap_when_no_image(monkeypatch):
    c = _client(monkeypatch, image=None)
    args = c.launch_args(_spec(), {"WORKER_ID": "w"})
    assert args["image"] == BOOTSTRAP_IMAGE == "node:22-bookworm"
    assert args["command"] == bootstrap_command()
    assert args["env"] == {"WORKER_ID": "w", "SWARM_HARNESS_WHEEL_URL": WHEEL}


def test_launch_args_bootstrap_flag_overrides_default_image(monkeypatch):
    c = _client(monkeypatch, image="ghcr.io/x/worker:1", bootstrap_env="1")
    assert c.launch_args(_spec(), {})["image"] == BOOTSTRAP_IMAGE
    # a type's own image always wins
    args = c.launch_args(_spec(image="me/custom:2"), {"A": "1"})
    assert args == {"image": "me/custom:2", "command": WORKER_COMMAND, "env": {"A": "1"}}


def test_launch_args_prebuilt_image(monkeypatch):
    c = _client(monkeypatch, image="ghcr.io/x/worker:1", wheel=None)
    args = c.launch_args(_spec(), {"A": "1"})
    assert args == {"image": "ghcr.io/x/worker:1", "command": WORKER_COMMAND, "env": {"A": "1"}}


def test_bootstrap_without_wheel_url_fails_fast(monkeypatch):
    monkeypatch.delenv("SWARM_HARNESS_WHEEL_URL", raising=False)
    with pytest.raises(ValueError, match="SWARM_HARNESS_WHEEL_URL"):
        _client(monkeypatch, image=None, wheel=None)


def test_launch_counts_hub_calls(monkeypatch):
    c = _client(monkeypatch, image=None)
    seen = {}

    class Api:
        def run_job(self, **kw):
            seen.update(kw)
            return type("J", (), {"id": "job-1"})()

    c.api = Api()
    job = c.launch("w0", _spec(), {"X": "1"}, {"S": "s"}, {"swarm": "e"}, 2400.7, namespace="ns")
    assert job == "job-1" and c.calls["run_job"] == 1
    assert seen["image"] == BOOTSTRAP_IMAGE and seen["timeout"] == 2400
    assert seen["namespace"] == "ns" and seen["labels"] == {"swarm": "e"}
    assert jobs_mod.DEFAULT_WORKER_IMAGE is None or isinstance(jobs_mod.DEFAULT_WORKER_IMAGE, str)


def test_parse_job_labels():
    assert parse_labels("experiment=swarm-harness-m1, team = eval") == {
        "experiment": "swarm-harness-m1",
        "team": "eval",
    }
    assert parse_labels("") == {}
