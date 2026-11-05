"""flock.harbor.envs.HFSandbox on a fake Sandbox.create, built through Harbor's factory as a
Trial builds it: `kwargs.namespace` reaches Sandbox.create (unset: None, Harbor's default), and
so does the HF token from the worker's secrets file."""

from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.environments import hf_sandbox
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from flock import secrets_file
from flock.harbor.envs import HFSandbox


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
