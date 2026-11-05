"""HubJobsClient launch arguments: bootstrap mode vs a prebuilt image."""

from __future__ import annotations

import subprocess

import pytest

from flock.controller.jobs import (
    BOOTSTRAP_IMAGE,
    CLAUDE_CODE_VERSION,
    CODEX_VERSION,
    PI_VERSION,
    WORKER_COMMAND,
    HubJobsClient,
    bootstrap_command,
)
from flock.controller.main import parse_labels
from flock.models import TypeSpec

WHEEL = "hf://buckets/ns/sh/dist/abc/flock-0.0.1-py3-none-any.whl"


def _spec(**kw) -> TypeSpec:
    return TypeSpec(harness="claude", model="claude-haiku-5-5", **kw)


def test_bootstrap_command_installs_pinned_tools_then_execs_worker(tmp_path):
    cmd = bootstrap_command()
    assert cmd[:2] == ["bash", "-c"]
    script = cmd[2]
    assert f"@anthropic-ai/claude-code@{CLAUDE_CODE_VERSION}" in script
    assert f"@openai/codex@{CODEX_VERSION}" in script
    assert "astral.sh/uv/install.sh" in script
    assert 'hf buckets cp "$FLOCK_WHEEL_URL"' in script
    assert "download_bucket_files" in script  # Python API fallback
    assert 'uv tool install -q "$WHEEL[worker]"' in script and "pip install" in script
    for env in ("IS_SANDBOX=1", "CLAUDE_CONFIG_DIR=/state/claude", "CODEX_HOME=/state/codex"):
        assert env in script
    assert script.rstrip().endswith("exec flock-worker")
    assert "{" + "claude}" not in script and "{" + "hub}" not in script  # all pins filled in
    f = tmp_path / "bootstrap.sh"
    f.write_text(script)
    subprocess.run(["bash", "-n", str(f)], check=True)


def _client(monkeypatch, image, bootstrap_env=None, wheel=WHEEL) -> HubJobsClient:
    if bootstrap_env is None:
        monkeypatch.delenv("FLOCK_WORKER_BOOTSTRAP", raising=False)
    else:
        monkeypatch.setenv("FLOCK_WORKER_BOOTSTRAP", bootstrap_env)
    return HubJobsClient(token="x", image=image, wheel_url=wheel)


def test_bootstrap_installs_pi_only_on_pi_workers(monkeypatch):
    script = bootstrap_command()[2]
    assert 'if [ "${FLOCK_HARNESS:-}" = pi ]' in script
    assert f"@earendil-works/pi-coding-agent@{PI_VERSION}" in script
    for env in ("PI_CODING_AGENT_DIR=/state/pi", "PI_OFFLINE=1", "PI_TELEMETRY=0", "FLOCK_PI_CLI="):
        assert env in script
    c = _client(monkeypatch, image=None)
    args = c.launch_args(TypeSpec(harness="pi", model="m"), {"WORKER_ID": "w"})
    assert args["env"]["FLOCK_HARNESS"] == "pi"


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
    monkeypatch.delenv("FLOCK_WHEEL_URL", raising=False)
    with pytest.raises(ValueError, match="FLOCK_WHEEL_URL"):
        _client(monkeypatch, image=None, wheel=None)


def test_parse_job_labels():
    assert parse_labels("experiment=flock-m1, team = eval") == {
        "experiment": "flock-m1",
        "team": "eval",
    }
    assert parse_labels("") == {}
