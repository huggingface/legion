"""Run-local, one-worker barrier before Pi emits its first model-turn event."""

import asyncio
import json
import shlex
import time

from legion.harbor import networking
from legion.harbor.agent import LegionPi


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.expected = int(options["agents"])
        self.directory = state_dir
        self.ready = {}
        self.open = asyncio.Event()
        self.original = None
        self.metadata = {"expected_agents": self.expected, "barrier": "before_first_model_turn"}

    async def start(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.original = LegionPi._turn
        original = self.original
        resource = self
        self.network_command = networking.command
        original_command = self.network_command

        async def podman_compatible(*args):
            code, out, err = await original_command(*args)
            if (
                code
                and args[:3] == ("podman", "network", "create")
                and "subnet" in err.lower()
                and "already used" in err.lower()
            ):
                err += " (subnet overlap)"
            return code, out, err

        networking.command = podman_compatible

        async def synchronized(agent, text, first, environment, totals):
            if first:
                from pathlib import Path

                extension = Path(__file__).with_name("capture.js").read_text()
                await agent.exec_as_agent(
                    environment,
                    "mkdir -p /tmp/harbor-pi-agent/extensions && printf %s "
                    + shlex.quote(extension)
                    + " > /tmp/harbor-pi-agent/extensions/qwen35.js",
                    timeout_sec=60,
                )
                name = agent.logs_dir.parent.name
                if name in resource.ready:
                    raise RuntimeError("Duplicate first turn at start barrier")
                resource.ready[name] = time.time()
                if len(resource.ready) > resource.expected:
                    raise RuntimeError("Too many agents at start barrier")
                if len(resource.ready) == resource.expected:
                    resource.metadata.update(released_at=time.time(), ready=dict(resource.ready))
                    (resource.directory / "released.json").write_text(
                        json.dumps(resource.metadata, indent=2)
                    )
                    resource.open.set()
                (resource.directory / "ready.json").write_text(json.dumps(resource.ready))
                await asyncio.wait_for(resource.open.wait(), 2700)
            return await original(agent, text, first, environment, totals)

        LegionPi._turn = synchronized

    def bind(self, name):
        return {}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        (directory / "barrier.json").write_text(json.dumps(self.metadata, indent=2))
        return {"barrier": "barrier.json"}

    async def stop(self):
        if hasattr(self, "network_command"):
            networking.command = self.network_command
        if self.original:
            LegionPi._turn = self.original
