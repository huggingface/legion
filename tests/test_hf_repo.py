"""Regression checks for the collaboration tools' destructive/error boundaries."""

import importlib.util
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def tool(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "repo_tool", Path(__file__).parents[1] / "src/flock/builtin/tools/hf_repo.py"
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
