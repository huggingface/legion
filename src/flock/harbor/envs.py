"""flock's Harbor environments, loaded through `EnvironmentConfig.import_path` (the adapter sets
it for every hf-sandbox Trial)."""

import asyncio

from harbor.environments import hf_sandbox

from flock.secrets_file import secret_env


class HFSandbox(hf_sandbox.HFSandboxEnvironment):
    """Harbor 0.25.0's hf-sandbox, whose `Sandbox.create` also gets
    - `namespace` (`environment.kwargs.namespace`): the user or org the sandboxes run and bill
      under; unset, the token's owner, as in Harbor;
    - the HF token through `secret_env()`: a Slurm worker has it in its secrets file, not in
      os.environ; unset, huggingface_hub's own lookup, as in Harbor.
    The sandbox keeps both for its later calls (kill cancels the job in its owner's namespace)."""

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
