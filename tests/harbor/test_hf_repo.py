"""Regression checks for the collaboration tools' destructive/error boundaries."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "repo_tool", Path(__file__).parents[2] / "src/flock/harbor/builtin/tools/hf_repo.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("FLOCK_HF_REPO", "example/collaboration")
    monkeypatch.setenv("FLOCK_HF_REPO_AGENT", "agent-001")
    monkeypatch.setenv("FLOCK_HF_REPO_TOKEN", "secret-test-token")
    return module


@pytest.fixture
def checkout(tool, tmp_path):
    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-b", "main")
    git("remote", "add", "origin", tool._url())
    git(
        "-c",
        "user.name=test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "--allow-empty",
        "-m",
        "initial",
    )
    return tmp_path


def test_explicit_staging_and_dirty_checkout(tool, checkout):
    (checkout / "code.py").write_text("print('original')\n")
    (checkout / "private.txt").write_text("not selected\n")
    tool.commit(str(checkout), ["code.py"], "Add original code")
    assert (
        subprocess.check_output(
            ["git", "-C", str(checkout), "ls-files"], text=True
        ).strip()
        == "code.py"
    )
    with pytest.raises(ValueError, match="local changes"):
        tool.sync(str(checkout))
    with pytest.raises(ValueError, match="relative file"):
        tool.commit(str(checkout), ["../private.txt"], "bad")
    with pytest.raises(ValueError, match="individual files"):
        tool.commit(str(checkout), ["." + "/"], "bad")
    (checkout / "executable").write_text("reference")
    with pytest.raises(ValueError, match="reference"):
        tool.commit(str(checkout), ["executable"], "bad")


def test_staged_changes_not_swept_into_commit(tool, checkout):
    (checkout / "private.txt").write_text("not selected")
    subprocess.run(["git", "-C", str(checkout), "add", "private.txt"], check=True)
    (checkout / "code.py").write_text("pass")
    with pytest.raises(ValueError, match="staged changes"):
        tool.commit(str(checkout), ["code.py"], "Add code")


def test_foreign_push_remote_rejected(tool, checkout):
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
            "remote",
            "set-url",
            "--push",
            "origin",
            "https://example.invalid/other",
        ],
        check=True,
    )
    with pytest.raises(ValueError, match="does not match"):
        tool.status(str(checkout))


def test_missing_git_is_actionable_and_redacted(tool, checkout, monkeypatch):
    monkeypatch.setenv("PATH", str(checkout))
    with pytest.raises(RuntimeError, match="Git is required") as exc:
        tool.status(str(checkout))
    assert "secret-test-token" not in str(exc.value)


async def test_shared_resource_across_workers_and_restart(tmp_path, monkeypatch, respx_mock):
    import json

    import httpx
    from harbor.models.trial.config import TrialConfig

    from flock.harbor.config import BUILTIN
    from flock.harbor.resources import Resources
    from flock.harbor.spec import Spec, resource_secrets

    monkeypatch.setenv("TEAM_REPO_TOKEN", "test-resource-credential")
    options = {"hf_repo": {"repo_id": "example/team", "token_env": "TEAM_REPO_TOKEN"}}
    spec = Spec.model_validate({
        "name": "team", "model": "test-model",
        "task": {"repo": "https://huggingface.co/datasets/example/tasks", "selector": "task"},
        "agents": {"count": 2}, "resources": options,
    })
    assert resource_secrets(spec) == ["TEAM_REPO_TOKEN"]
    info = respx_mock.get("https://huggingface.co/api/datasets/example/team/revision/main")
    info.return_value = httpx.Response(200, json={"id": "example/team", "sha": "a" * 40, "private": True})
    respx_mock.get("https://huggingface.co/api/datasets/example/team/discussions").respond(
        200, json={"discussions": [], "count": 0, "start": 0})
    for worker in ("worker-one", "worker-two", "worker-one"):
        resources = Resources(BUILTIN / "resources", options, run_dir=tmp_path / worker,
                              agents=[], secrets=[], required=["hf_repo"], incremental=True)
        try:
            await resources.start()
            await resources.add(["amber-otter"])
            config = TrialConfig(task={"path": tmp_path / "task"})
            resources.bind("amber-otter", config)
            assert config.agent.env["FLOCK_HF_REPO"] == "example/team"
            assert config.agent.env["FLOCK_HF_REPO_AGENT"] == "amber-otter"
            assert config.agent.env["FLOCK_HF_REPO_TOKEN"] == "test-resource-credential"
            assert "huggingface.co" in config.agent.extra_allowed_hosts
            await resources.remove(["amber-otter"])
        finally:
            assert not await resources.finish()
        result = (tmp_path / worker / "resources/hf_repo/repository.json").read_text()
        assert json.loads(result)["main_sha"] == "a" * 40
        assert "test-resource-credential" not in result
    assert all(call.request.method == "GET" for call in respx_mock.calls)
    info.return_value = httpx.Response(200, json={"id": "example/team", "sha": "a" * 40, "private": False})
    resources = Resources(BUILTIN / "resources", options, run_dir=tmp_path / "public",
                          agents=[], secrets=[])
    try:
        with pytest.raises(ValueError, match="must be private"):
            await resources.start()
    finally:
        await resources.finish()


def test_repository_bundle_runs_without_host_site_packages(tmp_path):
    import json
    import os
    import sys
    import tarfile

    from flock.harbor.config import BUILTIN
    from flock.harbor.tools.bundle import build_tool_bundle

    bundle = build_tool_bundle(BUILTIN / "tools", ["hf_repo.*", "message_board.*"],
                               tmp_path / "tools.tar.gz")
    with tarfile.open(bundle.archive) as archive:
        archive.extractall(tmp_path / "bundle", filter="data")
    result = subprocess.run(
        [sys.executable, "-S", str(tmp_path / "bundle/bin/flock-tool"), "--list"],
        env={**os.environ, "FLOCK_TOOLS_MANIFEST": str(tmp_path / "bundle/manifest.json")},
        capture_output=True, text=True, check=False, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    tools = json.loads(result.stdout)
    assert len([name for name in tools if name.startswith("hf_repo.")]) == 12
    assert "message_board.post" in tools
