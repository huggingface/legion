"""flock's Harbor environments, loaded through `EnvironmentConfig.import_path` (the adapter sets
it for every Trial: `ENVIRONMENTS`)."""

import asyncio
import collections
import contextvars
import hashlib
import re
import shlex
import uuid

from harbor.environments import e2b, hf_sandbox
from harbor.environments.docker.docker import DockerEnvironment
from harbor.environments.podman import PodmanEnvironment
from harbor.environments.tar_transfer import pack_dir_to_bytes
from harbor.utils.env import is_sensitive_env_key

from flock.secrets_file import secret_env

_EXEC_ENV = contextvars.ContextVar("flock_exec_env", default=None)


class _EnvOffArgv:
    """Harbor's `compose exec -e NAME=value` (the agent's env: the inference key, resource
    tokens) puts the value in an argv every user of the node reads (`ps`). For the agent's env
    names (`secret_names`, from the adapter) and every sensitive-looking one (Harbor's
    `is_sensitive_env_key`: the key Harbor's Pi passes as OPENAI_API_KEY, a task's *_TOKEN) it
    passes `-e NAME`, and the value goes in the compose process's own environment, which podman
    and docker read a bare NAME from (podman-compose 1.5: `--env NAME`). Any other name (a
    task's PATH or STORAGE_DRIVER, which would steer the host's compose there) keeps
    `-e NAME=value`."""

    def __init__(self, *args, secret_names=(), **kwargs):
        super().__init__(*args, **kwargs)
        self._secret_names = frozenset(secret_names)

    async def _run_docker_compose_command(self, command, *args, **kwargs):
        if command[:1] != ["exec"]:
            return await super()._run_docker_compose_command(command, *args, **kwargs)
        command, values, i = list(command), {}, 1
        while i + 1 < len(command) and command[i] in ("-w", "-e", "-u"):  # flags, then service
            name, eq, value = command[i + 1].partition("=")
            if command[i] == "-e" and eq and (name in self._secret_names
                                              or is_sensitive_env_key(name)):
                values[name], command[i + 1] = value, name
            i += 2
        token = _EXEC_ENV.set(values)  # per task: concurrent execs keep their own
        try:
            return await super()._run_docker_compose_command(command, *args, **kwargs)
        finally:
            _EXEC_ENV.reset(token)

    def _compose_env_vars(self, *args, **kwargs):
        return {**super()._compose_env_vars(*args, **kwargs), **(_EXEC_ENV.get() or {})}


class Docker(_EnvOffArgv, DockerEnvironment):
    pass


class Podman(_EnvOffArgv, PodmanEnvironment):
    pass


def sandbox_name(session_id: str, image: str) -> str:
    """A short, stable job name. The Hub derives one from the image's last path segment, which
    for `repo@sha256:<64 hex>` is over the 100-character `labels.name` limit."""
    return f"{re.sub(r'[^a-zA-Z0-9_-]', '-', session_id)[:60]}-{hashlib.sha256(image.encode()).hexdigest()[:8]}"


class HFSandbox(hf_sandbox.HFSandboxEnvironment):
    """Harbor 0.25.0's hf-sandbox, whose `Sandbox.create` also gets
    - `namespace` (`environment.kwargs.namespace`): the user or org the sandboxes run and bill
      under; unset, the token's owner, as in Harbor;
    - the HF token through `secret_env()`: a Slurm worker has it in its secrets file, not in
      os.environ; unset, huggingface_hub's own lookup, as in Harbor.
    The sandbox keeps both for its later calls (kill cancels the job in its owner's namespace).
    `upload_dir` does not keep the files' owner (below)."""

    def __init__(self, *args, namespace: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._namespace = namespace

    async def start(self, force_build: bool) -> None:
        # Harbor 0.25.0's HFSandboxEnvironment.start, with `namespace` and `token` added
        image = self.task_env_config.docker_image
        if image is None:
            raise ValueError(
                "HF Sandbox requires a prebuilt Docker image. "
                "Set [environment].docker_image in task.toml."
            )
        self._sandbox = await asyncio.to_thread(
            hf_sandbox.Sandbox.create,
            image=image,
            flavor=self._flavor,
            idle_timeout=self._idle_timeout,
            forward_hf_token=self._forward_hf_token,
            namespace=self._namespace,
            labels={"name": sandbox_name(self.session_id, image)},
            token=secret_env().get("HF_TOKEN") or None,
        )
        try:
            if workdir := self.task_env_config.workdir:
                await asyncio.to_thread(self._sandbox.files.mkdir, workdir)
            await self.ensure_dirs(self._mount_targets(writable_only=True))
            await self._upload_environment_dir_after_start()
        except Exception:
            try:  # not left running until its idle timeout
                await asyncio.to_thread(self._sandbox.kill)
            except Exception as kill_exc:
                self.logger.error(f"Failed to kill HF Sandbox during cleanup: {kill_exc}")
            self._sandbox = None
            raise


    async def upload_dir(self, source_dir, target_dir: str) -> None:
        """Harbor 0.25.0's, extracting with `--no-same-owner`. The archive carries the files'
        owner on this host, and root in the sandbox cannot take a uid over 65535 (a cluster
        user's, e.g. 150017), so the verifier's tests (and restored sessions) failed to
        upload."""
        sandbox = self._require_sandbox()
        remote_tar = f"/tmp/.hb-upload-{uuid.uuid4().hex[:8]}.tar.gz"
        buffer = await asyncio.to_thread(pack_dir_to_bytes, source_dir, compress=True)
        await asyncio.to_thread(sandbox.files.write, remote_tar, buffer.getvalue())
        tar, target = shlex.quote(remote_tar), shlex.quote(target_dir)
        result = await self.exec(f"mkdir -p {target} && tar --no-same-owner -xzf {tar} -C {target}"
                                 f" && rm -f {tar}", timeout_sec=120)
        if result.return_code != 0:
            raise RuntimeError(f"Failed to extract upload archive into {target_dir!r}: "
                               f"{result.stderr or result.stdout}")


_E2B_BUILDS = collections.defaultdict(asyncio.Lock)  # template -> lock, in this worker


class E2B(e2b.E2BEnvironment):
    """Harbor 0.25.0's E2B, whose Trials of one task build its template once. Each Trial that
    finds no template builds it, and E2B cancels a build when another of the same template
    starts ("build was cancelled"); a sandbox started before the build ends gets a 404."""
    async def start(self, force_build: bool) -> None:
        # shortcut: one worker's Trials wait for each other; workers that start one new task's
        # Trials at once still race (prebuild the template, or lock on E2B, if that matters)
        async with _E2B_BUILDS[self._template_name]:
            if force_build or not await self._does_template_exist():
                await self._create_template()
        await super().start(force_build=False)


ENVIRONMENTS = {"docker": "flock.harbor.envs:Docker", "podman": "flock.harbor.envs:Podman",
                "hf-sandbox": "flock.harbor.envs:HFSandbox",
                "e2b": "flock.harbor.envs:E2B"}  # backend -> import_path
