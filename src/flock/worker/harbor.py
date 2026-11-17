"""What a worker's `harness: harbor` agents share: one `Resources` (the vendored resources; the
board's and world's sockets and proxies run on this host, each agent's Trial is bound to them),
the vendored tool bundle, the task downloaded once, and on hf-sandbox a thread pool sized for
every sandbox call. Prepared at the worker's first Trial start; a failed attempt is retried by
the next one. The worker stops the resources when it exits.

With FLOCK_RESULTS_DIR set (a Slurm worker: `<run dir>/results`, on the shared disk), what a run
is graded on outlives the worker's WORK_ROOT (node-local): each ended Trial's `result.json`,
`config.json` and `verifier/` (a multi-step Trial's `steps/<name>/verifier/`) go to
`<dir>/<swarm>/<agent>/trials/<trial>/` (adapters/harbor.py) and the resources' exports, at exit,
to `<dir>/<swarm>/<worker id>/resources/`.

Each agent is added to the resources as its Trial starts and removed when it leaves (another
worker of the type, a spawned agent and a re-assigned one alike), and the resources' events for
an agent (the board's mentions) go to `emit(agent, event)`, which the worker forwards as the
agent's `resource` event. A resource with a `controller` attribute (resources.spawn) gets
`controller`, the worker's call to the controller as one of its agents (that agent's token).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from collections.abc import Awaitable, Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from flock.models import TypeSpec
from flock.secrets_file import exposed, secret_env

log = logging.getLogger(__name__)

RESULTS_ENV = "FLOCK_RESULTS_DIR"
# what a Trial is graded on; Harbor 0.25.0 moves a multi-step Trial's verifier files under steps/
TRIAL_RESULTS = ("result.json", "config.json", "verifier", "steps/*/verifier")


def results_dir() -> Path | None:
    """`$FLOCK_RESULTS_DIR/<swarm>`, where results outlive the worker; None when unset."""
    root = os.environ.get(RESULTS_ENV)
    return Path(root) / os.environ.get("SWARM", "swarm") if root else None


def keep_results(src: Path, dest: Path, names: Iterable[str]) -> None:
    """Copy `names` (files or folders, relative to `src`; `*` matches one level) from `src` to the
    same places under `dest`, links as links, missing ones skipped. A failure is logged: a run
    without its copy is still a run."""
    try:
        for name in names:
            for path in sorted(src.glob(name)) if "*" in name else [src / name]:
                target = dest / path.relative_to(src)
                if path.is_dir() and not path.is_symlink():
                    shutil.copytree(path, target, symlinks=True, dirs_exist_ok=True)
                elif path.exists() or path.is_symlink():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, target, follow_symlinks=False)
    except OSError as e:
        log.warning("results of %s not kept in %s: %s", src, dest, e)


class HarborHost:
    def __init__(self, root: Path, emit: Callable[[str, dict], None] | None = None,
                 folders: Callable[[str], Awaitable[Path]] | None = None,
                 controller: Callable[[str, str, dict], Awaitable[Any]] | None = None) -> None:
        self.root = Path(root)  # bundle, sources, resource state and exports
        self.emit = emit  # (agent, {"resource", "kind", "data"}): a resource's event
        # version -> the run's tools and resources folders unpacked here (ToolsManager.folder)
        self.folders = folders
        # the worker's controller_call(agent, entry, args): a call as that agent (its own token)
        self.controller = controller
        self.resources: Any = None  # flock.harbor Resources, started; None until prepared
        self.bundle: Any = None  # the vendored ToolBundle; None: the run has no tools
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
        if h.folders and self.folders is not None:  # shipped by `flock launch`, not the paths
            d = await self.folders(h.folders)
            config.tools_dir, config.resources_dir = d / "tools", d / "resources"
        if config.environment.backend == "hf-sandbox":
            # every sandbox call runs in a thread (asyncio.to_thread); the default pool has
            # about 6 on 2 vCPUs, so N agents' commands would queue behind each other
            asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(
                2 * spec.agents_per_worker + 8, thread_name_prefix="harbor"))
        self.root.mkdir(parents=True, exist_ok=True)
        if h.tools_sha256 and self.bundle is None:
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
            name = spec.resolved_model.inference.key_env if spec.resolved_model else None
            key = secret_env().get(name) if name else None
            # a resource without the add hook gets the type's roster, as the upstream launcher
            # gives it; one with it is told each agent placed here (adapters/harbor.py)
            with exposed():  # the plugins read their tokens (token_env) from os.environ
                resources = Resources(
                    config.resources_dir, h.resources, run_dir=self.root,
                    agents=[name for name, role in h.roster.items() if role == h.role],
                    secrets=[key] if key else [],
                    required=self.bundle.manifest["requires"] if self.bundle else (),
                    incremental=True)
            for provider in resources.providers.values():
                if hasattr(provider, "controller"):  # resources.spawn: calls as one agent
                    provider.controller = self.controller
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
        """Export and stop the resources (the worker exits); the exports go to the results
        dir too, when there is one."""
        if self._prepared is not None:
            self._prepared.cancel()
            await asyncio.gather(self._prepared, return_exceptions=True)
        if self.resources is not None:
            await self.resources.finish()
            if (results := results_dir()) is not None:
                worker = os.environ.get("WORKER_ID", "worker")
                await asyncio.to_thread(keep_results, self.root, results / worker,
                                        ["resources"])
