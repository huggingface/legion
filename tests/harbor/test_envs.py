"""flock.harbor.envs, built through Harbor's factory as a Trial builds them: HFSandbox on a
fake Sandbox.create (`kwargs.namespace` and the secrets file's HF token reach it; unset: None,
Harbor's default), and the docker and podman environments on a fake subprocess (no secret value
in an argv)."""

import asyncio
import importlib.metadata
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.environments import hf_sandbox
from harbor.environments.base import ExecResult
from harbor.environments.docker import docker
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from flock import secrets_file
from flock.harbor.envs import ENVIRONMENTS, HFSandbox


@pytest.mark.parametrize("kwargs, namespace, file_token, token", [
    ({}, None, None, None),
    ({"namespace": "my-org", "flavor": "cpu-upgrade"}, "my-org", "hf_from_file", "hf_from_file"),
])
async def test_namespace_and_token_reach_sandbox_create(
    monkeypatch, tmp_path, kwargs, namespace, file_token, token
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(secrets_file, "_secrets", {"HF_TOKEN": file_token} if file_token else {})
    calls = []

    def create(**kw):
        calls.append(kw)
        return SimpleNamespace(files=SimpleNamespace(mkdir=lambda path: None))

    monkeypatch.setattr(hf_sandbox.Sandbox, "create", staticmethod(create))
    monkeypatch.setattr(HFSandbox, "ensure_dirs", lambda self, dirs: _done())
    monkeypatch.setattr(HFSandbox, "_upload_environment_dir_after_start", lambda self: _done())
    (tmp_path / "environment").mkdir()
    env = EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig(type="hf-sandbox", import_path="flock.harbor.envs:HFSandbox",
                               kwargs=kwargs),
        environment_dir=tmp_path / "environment", environment_name="t", session_id="s",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.12"))
    assert isinstance(env, HFSandbox)
    await env.start(force_build=False)
    (kw,) = calls
    assert (kw["namespace"], kw["token"], kw["image"]) == (namespace, token, "python:3.12")
    assert kw["flavor"] == kwargs.get("flavor", "cpu-basic")


async def _done():
    return None


def test_the_copied_start_matches_the_pinned_harbor():
    """HFSandbox.start is Harbor 0.25.0's, copied: a Harbor upgrade must re-copy it (and check
    that _EnvOffArgv still sees `compose exec -e NAME=value`)."""
    assert importlib.metadata.version("harbor") == "0.25.0"


@pytest.mark.parametrize("backend", ["docker", "podman"])
async def test_no_secret_value_in_any_argv(monkeypatch, tmp_path, backend):
    """Every compose exec (the agent's env holds the inference key and resource tokens) passes
    `-e NAME` only; the values reach the compose process through its environment. Concurrent
    execs keep their own values, and a non-exec command gets none."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    runs = []

    async def spawn(*argv, env=None, **kw):
        runs.append((argv, env))
        await asyncio.sleep(0.01)  # the other exec runs meanwhile
        return SimpleNamespace(argv=argv)

    async def collect(self, process, **kw):
        return ExecResult(stdout="", stderr="", return_code=0)

    monkeypatch.setattr(docker.asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(docker.DockerEnvironment, "_collect_buffered_output", collect)
    monkeypatch.setattr(docker.DockerEnvironment, "_collect_streamed_output", collect)
    (tmp_path / "environment").mkdir()
    env = EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig(type=backend, import_path=ENVIRONMENTS[backend],
                               kwargs={"secret_names": ["HF_TOKEN", "PLAIN", "DOCKER_TOKEN"]}),
        environment_dir=tmp_path / "environment", environment_name="t", session_id="s",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.12"))
    await asyncio.gather(
        env.exec("pi --print hi", env={"HF_TOKEN": "s3cret-a", "PLAIN": "x=1", "PATH": "/t/bin",
                                       "STORAGE_DRIVER": "vfs", "DOCKER_TOKEN": "s3cret-c"},
                 user="agent"),
        env.exec("true", env={"HF_TOKEN": "s3cret-b"}, cwd="/app"))
    await env._run_docker_compose_command(["ps"])
    (a, env_a), (b, env_b), (ps, env_ps) = runs
    for argv in (a, b, ps):
        assert not any("s3cret" in arg for arg in argv), argv
    assert a[a.index("exec"):][:13] == ("exec", "-e", "HF_TOKEN", "-e", "PLAIN", "-e",
                                        "PATH=/t/bin", "-e", "STORAGE_DRIVER=vfs", "-e",
                                        "DOCKER_TOKEN", "-u", "agent")
    assert (env_a["HF_TOKEN"], env_a["PLAIN"], env_b["HF_TOKEN"]) == ("s3cret-a", "x=1", "s3cret-b")
    # only the agent's own names move: a task's PATH or STORAGE_DRIVER never steers the host
    assert env_a.get("PATH") != "/t/bin" and "STORAGE_DRIVER" not in env_a
    assert "-e" in b and "PLAIN" not in env_b and "HF_TOKEN" not in env_ps
