"""FLOCK_WORKER_SECRETS_FILE: worker credentials from a 0600 file instead of the exec environment."""

import os
import subprocess
import sys

import pytest

from flock import secrets_file
from flock.secrets_file import SecretsFileError, load, parse, read_file, secret_env


@pytest.fixture(autouse=True)
def _clean():
    secrets_file._secrets.clear()
    yield
    secrets_file._secrets.clear()


def write(path, text, mode=0o600):
    path.write_text(text)
    path.chmod(mode)
    return path


def test_format_comments_export_and_quotes():
    assert parse("# c\n\nexport A=1\nB = 'two words'\nC=\"x=y\"\nD=\n") == \
        {"A": "1", "B": "two words", "C": "x=y", "D": ""}
    with pytest.raises(SecretsFileError, match="line 1") as e:
        parse("not a line with secret-value\n")
    assert "secret-value" not in str(e.value)


def test_file_must_be_private(tmp_path):
    path = write(tmp_path / "s.env", "HF_TOKEN=hf_x\n", mode=0o640)
    with pytest.raises(SecretsFileError, match="0600"):
        read_file(path)
    path.chmod(0o600)
    assert read_file(path) == {"HF_TOKEN": "hf_x"}
    link = tmp_path / "link.env"
    link.symlink_to(path)
    with pytest.raises(SecretsFileError):
        read_file(link)


def test_file_values_win_and_leave_the_process_environment(tmp_path):
    path = write(tmp_path / "s.env", "HF_TOKEN=from-file\nOPENAI_API_KEY=key\n")
    env = {"FLOCK_WORKER_SECRETS_FILE": str(path), "HF_TOKEN": "from-env", "FLOCK_TOKEN": "env-only"}
    assert load(env) == ["HF_TOKEN", "OPENAI_API_KEY"]
    assert "HF_TOKEN" not in env and env["FLOCK_TOKEN"] == "env-only"
    assert path.exists()
    env2 = {"FLOCK_WORKER_SECRETS_FILE": str(path), "FLOCK_WORKER_SECRETS_FILE_UNLINK": "1"}
    load(env2)
    assert not path.exists()


def test_without_the_file_secrets_come_from_the_environment(monkeypatch):
    monkeypatch.delenv("FLOCK_WORKER_SECRETS_FILE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert load() == [] and secret_env()["OPENAI_API_KEY"] == "env-key"


def test_children_and_proc_environ_never_see_file_secrets(tmp_path):
    path = write(tmp_path / "s.env", "FLOCK_WORLD_TOKEN=world-secret\n")
    script = (
        "import os, subprocess, sys\n"
        "from flock.secrets_file import load, secret_env\n"
        "load()\n"
        "assert secret_env()['FLOCK_WORLD_TOKEN'] == 'world-secret'\n"
        "own = open(f'/proc/{os.getpid()}/environ', 'rb').read()\n"
        "child = subprocess.run([sys.executable, '-c', 'import os; print(sorted(os.environ))'],\n"
        "                       capture_output=True, text=True).stdout\n"
        "print(b'world-secret' in own, 'FLOCK_WORLD_TOKEN' in child)\n")
    env = {k: v for k, v in os.environ.items() if k != "FLOCK_WORLD_TOKEN"}
    env["FLOCK_WORKER_SECRETS_FILE"] = str(path)
    out = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True,
                         check=True).stdout
    assert out.split() == ["False", "False"]


async def test_agent_harness_env_reads_the_file_secrets(tmp_path, monkeypatch):
    from .worker_fakes import runtime

    path = write(tmp_path / "s.env", "MY_SECRET=file-value\n")
    monkeypatch.setenv("FLOCK_WORKER_SECRETS_FILE", str(path))
    monkeypatch.setenv("MY_SECRET", "env-value")
    load()
    assert "MY_SECRET" not in os.environ
    rt, _ = runtime(tmp_path)
    rt.spec = rt.spec.model_copy(update={"secrets": ["MY_SECRET"]})
    assert (await rt.harness_env())["MY_SECRET"] == "file-value"
