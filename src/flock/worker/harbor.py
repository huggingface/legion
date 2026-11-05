"""What a worker's `harness: harbor` agents share: one `Resources` (shepherd's resources; the
board's and world's sockets and proxies run on this host, each agent's Trial is bound to them),
shepherd's tool bundle, the task downloaded once, and on hf-sandbox a thread pool sized for
every sandbox call. Prepared at the worker's first Trial start; a failed attempt is retried by
the next one. The worker stops the resources when it exits.

Each agent is added to the resources as its Trial starts and removed when it leaves (another
worker of the type, a spawned agent and a re-assigned one alike), and the resources' events for
an agent (the board's mentions) go to `emit(agent, event)`, which the worker forwards as the
agent's `resource` event.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flock.models import TypeSpec
from flock.secrets_file import exposed, secret_env

log = logging.getLogger(__name__)


class HarborHost:
    def __init__(self, root: Path, emit: Callable[[str, dict], None] | None = None) -> None:
        self.root = Path(root)  # bundle, sources, resource state and exports
        self.emit = emit  # (agent, {"resource", "kind", "data"}): a resource's event
        self.resources: Any = None  # flock.harbor Resources, started; None until prepared
        self.bundle: Any = None  # shepherd's ToolBundle; None: the run has no tools
        self.task_path: Path | None = None  # the pinned task, downloaded for every agent here
        self._prepared: asyncio.Task | None = None

    async def prepare(self, spec: TypeSpec) -> None:
        """Once per worker; concurrent Trial starts wait for the same attempt."""
        t = self._prepared
        if t is None or (t.done() and (t.cancelled() or t.exception() is not None)):
            t = self._prepared = asyncio.create_task(self._prepare(spec), name="harbor-host")
        await asyncio.shield(t)

    async def _prepare(self, spec: TypeSpec) -> None:
        from harbor.models.task.id import GitTaskId

        from flock.harbor import sources
        from flock.harbor.resources import Resources
        from flock.harbor.spec import Spec
        from flock.harbor.tools.bundle import build_tool_bundle

        h = spec.harbor
        config = Spec.model_validate(h.config)
        if config.environment.backend == "hf-sandbox":
            # every sandbox call runs in a thread (asyncio.to_thread); the default pool has
            # about 6 on 2 vCPUs, so N agents' commands would queue behind each other
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(
                2 * spec.agents_per_worker + 8, thread_name_prefix="harbor"))
        self.root.mkdir(parents=True, exist_ok=True)
        if h.tools_sha256 and self.bundle is None:
            # shortcut: built from the spec's tools_dir (the controller's tools transport ships
            # flock's own bundle format); upgrade when HF Jobs workers lack that folder
            bundle = await asyncio.to_thread(build_tool_bundle, config.tools_dir, config.tools,
                                             self.root / "tools.tar.gz")
            if bundle.sha256 != h.tools_sha256:
                raise RuntimeError(f"the tools built here ({bundle.sha256[:12]}) are not the "
                                   f"run's ({h.tools_sha256[:12]}): {config.tools_dir} differs")
            self.bundle = bundle
        if self.task_path is None:
            task = sources.ResolvedTask(GitTaskId.model_validate(h.task["task"]),
                                        h.task["registry_commit"], h.task["dataset"])
            self.task_path = await sources.download_task(task, self.root / "sources")
        if self.resources is None:
            key = secret_env().get(config.inference.api_key_env)
            # a resource without the add hook gets the type's roster, as shepherd's launcher
            # gives it; one with it is told each agent placed here (adapters/harbor.py)
            with exposed():  # the plugins read their tokens (token_env) from os.environ
                resources = Resources(
                    config.resources_dir, h.resources, run_dir=self.root,
                    agents=[name for name, role in h.roster.items() if role == h.role],
                    secrets=[key] if key else [],
                    required=self.bundle.manifest["requires"] if self.bundle else (),
                    incremental=True)
            try:
                await resources.start()
            except BaseException:
                await resources.finish()
                raise
            if self.emit is not None:
                resources.watch(self.emit)
            for w in resources.watchers:  # a resource that dies is logged; its binds stay
                w.add_done_callback(lambda t: t.cancelled() or log.error("%s", t.exception()))
            self.resources = resources

    async def close(self) -> None:
        """Export and stop the resources (the worker exits)."""
        if self._prepared is not None:
            self._prepared.cancel()
            await asyncio.gather(self._prepared, return_exceptions=True)
        if self.resources is not None:
            await self.resources.finish()
