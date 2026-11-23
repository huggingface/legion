"""legion's controller for Harbor agents (docs/spawn.md): spawn children, prompt agents, read
and remove them, through the `spawn.*` tools.

Each agent gets its own Unix socket, served on the worker and mounted alone at SOCKET. One
JSON line in (`{method, path, params?, json?}`), one out (`{status, body}` or `{error}`). The
worker sends the call to the controller with that agent's own token (legion.worker.main's
controller_call: the routes an agent token may use), and the controller checks what the agent
may do and the spawn caps. So an agent acts only as itself: the socket names it, the request
cannot. Needs a mount: docker or podman, not hf-sandbox.
"""

import asyncio
import contextlib
import itertools
import json
import tempfile
from functools import partial
from pathlib import Path

SOCKET = "/run/legion-spawn.sock"
LINE_MAX = 1 << 20  # a request line (a long first prompt fits)
READ_S = 60  # a client that never sends its line


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        if options:
            raise ValueError("resources.spawn takes no options: use `spawn: {}`")
        self.metadata = {}
        # legion: set by the worker (legion.worker.harbor): controller_call(agent, entry, args)
        self.controller = None
        self.directory = tempfile.TemporaryDirectory(prefix="legion-spawn-")
        self.servers, self.paths, self.slots = {}, {}, {}
        self.numbers = itertools.count()  # socket file names: AF_UNIX paths are short

    async def start(self):
        if self.controller is None:
            raise RuntimeError("resources.spawn runs on a legion worker only")

    async def add(self, names):
        for name in names:
            if name not in self.servers:
                path = Path(self.directory.name) / f"{next(self.numbers)}.sock"
                self.servers[name] = await asyncio.start_unix_server(
                    partial(self._serve, name), path, limit=LINE_MAX)
                path.chmod(0o666)  # the containing host directory is private
                self.paths[name] = path
                # one call at a time per agent: the worker's controller client has a few
                # connections, shared with its heartbeat and poll
                self.slots.setdefault(name, asyncio.Semaphore(1))  # kept: a call in flight

    async def remove(self, names):
        for name in names:
            if (server := self.servers.pop(name, None)) is not None:
                server.close()
                self.paths.pop(name).unlink(missing_ok=True)

    async def _serve(self, agent, reader, writer):
        try:
            try:
                request = json.loads(await asyncio.wait_for(reader.readline(), READ_S))
                if not isinstance(request, dict):
                    raise ValueError("a request is a JSON object")
                async with self.slots[agent]:
                    result = await self.controller(agent, "controller:request", request)
                answer = result.output
            except Exception as exc:  # a bad line; never the controller's own answer
                answer = json.dumps({"error": f"bad request: {type(exc).__name__}: {exc}"})
            with contextlib.suppress(ConnectionError):  # the client hung up
                writer.write(answer.encode() + b"\n")
                await writer.drain()
        finally:
            writer.close()

    def bind(self, name):
        return {
            "env": {"SPAWN_SOCKET": SOCKET, "SPAWN_AGENT": name},
            "mounts": [{"type": "bind", "source": str(self.paths[name]), "target": SOCKET,
                        "read_only": True}],
            "metadata": {"agent": name},
        }

    async def wait(self):
        await asyncio.Event().wait()  # nothing to watch: the sockets live in this process

    async def export(self, directory):
        return {}

    async def stop(self):
        await self.remove(list(self.servers))
        self.directory.cleanup()
