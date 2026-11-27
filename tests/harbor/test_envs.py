"""legion.harbor.envs, built through Harbor's factory as a Trial builds them: HFSandbox on a
fake Sandbox.create (`kwargs.namespace` and the secrets file's HF token reach it; unset: None,
Harbor's default), and the docker and podman environments on a fake subprocess (no secret value
in an argv)."""

import asyncio
import collections
import importlib.metadata
import re
from types import SimpleNamespace

import pytest
from harbor.environments import e2b, hf_sandbox
from harbor.environments.base import ExecResult
from harbor.environments.docker import docker
from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
from harbor.models.trial.paths import TrialPaths

from legion import secrets_file
from legion.harbor import envs
from legion.harbor.envs import E2B, ENVIRONMENTS, HFSandbox, sandbox_name


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
        TrialEnvironmentConfig(type="hf-sandbox", import_path="legion.harbor.envs:HFSandbox",
                               kwargs=kwargs),
        environment_dir=tmp_path / "environment", environment_name="t", session_id="s",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.12"))
    assert isinstance(env, HFSandbox)
    await env.start(force_build=False)
    (kw,) = calls
    assert (kw["namespace"], kw["token"], kw["image"]) == (namespace, token, "python:3.12")
    assert kw["flavor"] == kwargs.get("flavor", "cpu-basic")
    assert kw["labels"]["name"].startswith("s-")


def test_sandbox_name_is_short_for_a_digest_image():
    image = "ghcr.io/org/img@sha256:" + "a" * 64
    name = sandbox_name("trial/x" * 30, image)
    assert len(name) <= 100 and re.fullmatch(r"[A-Za-z0-9_-]+", name)
    assert name == sandbox_name("trial/x" * 30, image) != sandbox_name("trial/x" * 30, "other")


async def test_e2b_trials_of_one_task_build_its_template_once(monkeypatch, tmp_path):
    """Two Trials starting at once: one builds the template, the other waits for it (E2B
    cancels a build when a second one of the same template starts)."""
    built, builds = set(), []

    async def build(self):
        builds.append(self._template_name)
        await asyncio.sleep(0)  # the other Trial's turn, mid-build
        built.add(self._template_name)

    async def exists(self):
        return self._template_name in built

    monkeypatch.setattr(E2B, "_create_template", build)
    monkeypatch.setattr(E2B, "_does_template_exist", exists)
    monkeypatch.setattr(e2b.E2BEnvironment, "start", lambda self, force_build: _done())
    monkeypatch.setattr(envs, "_E2B_BUILDS", collections.defaultdict(asyncio.Lock))  # this loop's
    (tmp_path / "environment").mkdir()

    def env(session):
        return EnvironmentFactory.create_environment_from_config(
            TrialEnvironmentConfig(type="e2b", import_path=ENVIRONMENTS["e2b"]),
            environment_dir=tmp_path / "environment", environment_name="t", session_id=session,
            trial_paths=TrialPaths(tmp_path / session),
            task_env_config=EnvironmentConfig(docker_image="python:3.12"))

    await asyncio.gather(env("a").start(force_build=False), env("b").start(force_build=False))
    assert len(builds) == 1


async def _done():
    return None


def test_the_copied_start_matches_the_pinned_harbor():
    """HFSandbox.start is Harbor 0.25.0's, copied: a Harbor upgrade must re-copy it (and check
    that _EnvOffArgv still sees `compose exec -e NAME=value`, and whether Podman still needs
    `--no-TTY` rewritten)."""
    assert importlib.metadata.version("harbor") == "0.25.0"


def compose(monkeypatch, tmp_path, backend, secret_names, **env_kwargs):
    """legion's docker or podman environment, as a Trial builds it, on a fake subprocess; the
    list gets each compose call's (argv, env)."""
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
    return runs, EnvironmentFactory.create_environment_from_config(
        TrialEnvironmentConfig(type=backend, import_path=ENVIRONMENTS[backend],
                               kwargs={"secret_names": secret_names}),
        environment_dir=tmp_path / "environment", environment_name="t", session_id="s",
        trial_paths=TrialPaths(tmp_path / "trial"),
        task_env_config=EnvironmentConfig(docker_image="python:3.12"), **env_kwargs)


@pytest.mark.parametrize("backend", ["docker", "podman"])
async def test_no_secret_value_in_any_argv(monkeypatch, tmp_path, backend):
    """Every compose exec (the agent's env holds the inference key and resource tokens) passes
    `-e NAME` only; the values reach the compose process through its environment. Concurrent
    execs keep their own values, and a non-exec command gets none."""
    monkeypatch.delenv("HF_TOKEN", raising=False)
    runs, env = compose(monkeypatch, tmp_path, backend, ["HF_TOKEN", "PLAIN", "DOCKER_TOKEN"])
    await asyncio.gather(
        env.exec("pi --print hi", env={"HF_TOKEN": "s3cret-a", "PLAIN": "x=1", "PATH": "/t/bin",
                                       "STORAGE_DRIVER": "vfs", "DOCKER_TOKEN": "s3cret-c",
                                       "VERIFIER_API_KEY": "s3cret-d"}, user="agent"),
        env.exec("true", env={"HF_TOKEN": "s3cret-b"}, cwd="/app"))
    await env._run_docker_compose_command(["ps"])
    (a, env_a), (b, env_b), (ps, env_ps) = runs
    for argv in (a, b, ps):
        assert not any("s3cret" in arg for arg in argv), argv
    assert a[a.index("exec"):][:15] == ("exec", "-e", "HF_TOKEN", "-e", "PLAIN", "-e",
                                        "PATH=/t/bin", "-e", "STORAGE_DRIVER=vfs", "-e",
                                        "DOCKER_TOKEN", "-e", "VERIFIER_API_KEY", "-u", "agent")
    assert (env_a["HF_TOKEN"], env_a["PLAIN"], env_b["HF_TOKEN"]) == ("s3cret-a", "x=1", "s3cret-b")
    # only the agent's own names move: a task's PATH or STORAGE_DRIVER never steers the host
    assert env_a.get("PATH") != "/t/bin" and "STORAGE_DRIVER" not in env_a
    assert "-e" in b and "PLAIN" not in env_b and "HF_TOKEN" not in env_ps


@pytest.mark.parametrize("backend, flag", [("docker", "--no-TTY"), ("podman", "-T")])
async def test_network_policy_exec_flag(monkeypatch, tmp_path, backend, flag):
    """Harbor's `compose exec --no-TTY` (a non-public network policy, set on the egress sidecar)
    reaches podman-compose, which rejects `--no-TTY`, as `-T`; docker keeps `--no-TTY`."""
    # the probe runs a container on the engine's kernel
    monkeypatch.setattr(docker.DockerEnvironment, "_egress_control_kernel_support",
                        classmethod(lambda cls: True))
    runs, env = compose(monkeypatch, tmp_path, backend, [],
                        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
    await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.ALLOWLIST,
                                               allowed_hosts=["example.com"]))
    (argv, _), = runs
    assert argv[argv.index("exec"):] == ("exec", flag, "harbor-docker-egress-control-sidecar",
                                         "network-policy", "allow", "example.com")
