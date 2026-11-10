# vendored from eabb46f:src/flock/resources.py
"""Load experiment resources and manage their lifecycle without provider-specific code."""

import asyncio
import importlib.util
import re

from .results import safe_error


def load_module(path):
    spec = importlib.util.spec_from_file_location(f"_flock_tool_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Resources:
    # flock: with `incremental`, a resource with an `add` hook starts with no agents and is told
    # each agent placed here through add() (another worker's agents, spawned, re-assigned)
    def __init__(self, directory, options, *, run_dir, agents, secrets, required=(),
                 incremental=False):
        missing = set(required) - options.keys()
        if missing:
            raise ValueError(
                f"Selected tools require resources: {', '.join(sorted(missing))}"
            )
        self.run_dir = run_dir
        self.secrets = secrets
        self.providers = {}
        self.records = {}
        self.attempted = []
        self.watchers = []
        for name, settings in options.items():
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
                raise ValueError(f"Invalid resource name: {name}")
            path = directory / f"{name}.py"
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Missing resource module: {path}")
            module = load_module(path)
            self.providers[name] = module.Resource(
                settings,
                state_dir=run_dir / "state" / "resources" / name,
                agents=[] if incremental and hasattr(module.Resource, "add") else agents,  # flock:
                secrets=secrets,
            )
            self.records[name] = {"status": "pending", "metadata": {}, "artifacts": {}}

    async def _watch(self, name, provider):
        try:
            await provider.wait()
            raise RuntimeError("Resource exited unexpectedly")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.records[name].update(
                status="failed", error=safe_error(exc, self.secrets)
            )
            raise RuntimeError(
                f"Resource {name} failed: {safe_error(exc, self.secrets)}"
            ) from None

    async def start(self):
        for name, provider in self.providers.items():
            self.attempted.append(name)
            record = self.records[name]
            record["status"] = "starting"
            # Resources may need longer, e.g. to build a Space.
            limit = getattr(provider, "startup_timeout_seconds", 120)
            try:
                async with asyncio.timeout(limit) as deadline:
                    await provider.start()
                record.update(status="ready", metadata=provider.metadata)
                self.watchers.append(asyncio.create_task(self._watch(name, provider)))
            except BaseException as exc:
                if isinstance(exc, TimeoutError) and deadline.expired():
                    exc = RuntimeError(
                        f"Resource {name} did not start within {limit:g} s"
                    )
                record.update(
                    status="failed",
                    error=safe_error(exc, self.secrets),
                    metadata=provider.metadata,
                )
                raise exc

    def bind(self, name, config):
        metadata = {}
        mounts = list(config.environment.mounts or [])
        targets = {m["target"] for m in mounts}
        for resource_name, provider in self.providers.items():
            binding = provider.bind(name)
            env = binding.get("env", {})
            overlap = config.agent.env.keys() & env.keys()
            if overlap:
                raise ValueError(
                    f"Resource environment collision: {', '.join(sorted(overlap))}"
                )
            config.agent.env.update(env)
            for mount in binding.get("mounts", []):
                if mount["target"] in targets:
                    raise ValueError(f"Resource mount collision: {mount['target']}")
                targets.add(mount["target"])
                mounts.append(mount)
            for host in binding.get("allowed_hosts", []):
                if host not in config.agent.extra_allowed_hosts:
                    config.agent.extra_allowed_hosts.append(host)
            metadata[resource_name] = binding.get("metadata", {})
        config.environment.mounts = mounts or None
        return metadata

    # flock: the optional hooks; the upstream launcher calls none of them
    async def add(self, names):
        """Agents that now run here: each resource with `add` sets them up (idempotent)."""
        for provider in self.providers.values():
            if hasattr(provider, "add"):
                await provider.add(names)

    async def remove(self, names):
        """Agents that left: each resource with `remove` lets them go."""
        for provider in self.providers.values():
            if hasattr(provider, "remove"):
                await provider.remove(names)

    def watch(self, emit):
        """Run each resource's `watch(emit)`; it calls emit(agent, kind, data) per event, passed on
        as emit(agent, {"resource", "kind", "data"}). Stopped by finish()."""
        for name, provider in self.providers.items():
            if hasattr(provider, "watch"):
                def forward(agent, kind, data, resource=name):
                    emit(agent, {"resource": resource, "kind": kind, "data": data})

                self.watchers.append(asyncio.create_task(provider.watch(forward)))

    async def finish(self):
        """Export every started resource, then stop in reverse order, retaining errors."""
        for watcher in self.watchers:
            watcher.cancel()
        await asyncio.gather(*self.watchers, return_exceptions=True)
        failed = False
        for name in self.attempted:
            record = self.records[name]
            directory = self.run_dir / "resources" / name
            try:
                async with asyncio.timeout(120) as deadline:
                    directory.mkdir(parents=True, exist_ok=True)
                    artifacts = await self.providers[name].export(directory)
                record["artifacts"] = {
                    key: str((directory / path).relative_to(self.run_dir))
                    for key, path in artifacts.items()
                }
            except Exception as exc:
                if isinstance(exc, TimeoutError) and deadline.expired():
                    exc = RuntimeError("Export did not finish within 120 s")
                record["export_error"] = safe_error(exc, self.secrets)
                failed = True
        for name in reversed(self.attempted):
            record = self.records[name]
            limit = getattr(self.providers[name], "stop_timeout_seconds", 30)
            try:
                async with asyncio.timeout(limit) as deadline:
                    await self.providers[name].stop()
                if record["status"] == "ready":
                    record["status"] = "stopped"
            except Exception as exc:
                if isinstance(exc, TimeoutError) and deadline.expired():
                    exc = RuntimeError(f"Stop did not finish within {limit:g} s")
                record["cleanup_error"] = safe_error(exc, self.secrets)
                failed = True
        return failed
