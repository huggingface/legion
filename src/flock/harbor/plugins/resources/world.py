"""A running World API world (docs/world.md), shared by every agent; flock never starts or stops it.

Each agent acts only as itself. `transport: unix` (docker, podman): a per-agent Unix socket,
served by a launcher-side proxy that adds the world secret and the agent's name, so no world
credential enters the container and any World API v1 world works. `transport: http` (backends
without mounts, e.g. hf-sandbox): WORLD_URL, WORLD_AGENT and the agent's own token, which a
flock.world server accepts only for that agent's tool calls (flock.world.protocol.agent_token).
"""

import asyncio
import http.server
import os
import socketserver
import tempfile
import threading
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator

from flock.harbor.results import write_json
from flock.world.protocol import AGENT_HEADER, TOKEN_HEADER, WorldManifest, agent_token

SOCKET = "/run/flock-world.sock"
HEALTH_EVERY_S, HEALTH_MISSES = 5, 12  # a minute of failed health checks fails the run
# What the proxy passes on: tracing headers in, the body's type and a retry hint out.
FORWARD_IN = ("content-type", "accept", "x-request-id", "x-flock-type", "x-flock-swarm",
              "x-flock-turn")
FORWARD_OUT = ("content-type", "retry-after")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    url: str
    token_env: str = Field(default="FLOCK_WORLD_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    transport: Literal["unix", "http"] = "unix"

    @field_validator("url")
    @classmethod
    def endpoint(cls, value):
        url = urlsplit(value)
        if (
            url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError(
                "world.url must be HTTP(S) without embedded credentials, query or fragment"
            )
        return value.rstrip("/")


def _proxy(url, client, token, agent):
    class Handler(http.server.BaseHTTPRequestHandler):
        def forward(self):
            # A request target such as "@host/x" would move the credentials to another host.
            if not self.path.startswith("/"):
                self.send_error(400)
                return
            headers = {k: v for k, v in self.headers.items() if k.lower() in FORWARD_IN}
            headers.update({TOKEN_HEADER: token, AGENT_HEADER: agent})
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                r = client.request(self.command, url + self.path, content=body, headers=headers)
            except httpx.HTTPError as exc:
                self.send_error(502, f"world unreachable ({type(exc).__name__})")
                return
            self.send_response(r.status_code)
            for key in FORWARD_OUT:
                if key in r.headers:
                    self.send_header(key, r.headers[key])
            self.send_header("Content-Length", str(len(r.content)))
            self.end_headers()
            self.wfile.write(r.content)

        do_GET = do_POST = forward

        def log_message(self, *args):
            pass

    return Handler


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options)
        self.url = self.options.url
        self.secret = os.environ.get(self.options.token_env)
        if not self.secret:
            raise ValueError(f"Set {self.options.token_env} to the secret of world.url")
        self.tokens = {name: agent_token(self.secret, name) for name in agents}
        secrets.extend([self.secret, *self.tokens.values()])
        self.metadata = {}
        self.client = self.proxy_client = self.socket_directory = None
        self.sockets, self.servers = {}, []

    async def _health(self):
        r = await self.client.get("/healthz")
        r.raise_for_status()
        return r.json()["status"]

    async def start(self):
        self.client = httpx.AsyncClient(
            base_url=self.url, headers={TOKEN_HEADER: self.secret}, timeout=10
        )
        while await self._health() != "ready":  # parked: still building; Resources times out
            await asyncio.sleep(1)
        r = await self.client.get("/manifest")
        r.raise_for_status()
        manifest = WorldManifest.model_validate(r.json())
        self.metadata = {
            "url": self.url,
            "transport": self.options.transport,
            "world": manifest.name,
            "version": manifest.version,
            "tools": manifest.names(),
            "capabilities": manifest.capabilities,
        }
        if self.options.transport == "unix":
            self._serve_sockets()
        print(f"World: {self.url} ({manifest.name} {manifest.version})", flush=True)

    def _serve_sockets(self):
        self.socket_directory = tempfile.TemporaryDirectory(prefix="flock-world-")
        self.proxy_client = httpx.Client(timeout=httpx.Timeout(1800, connect=10))
        for name in self.tokens:
            path = Path(self.socket_directory.name) / f"{name}.sock"
            server = socketserver.ThreadingUnixStreamServer(
                str(path), _proxy(self.url, self.proxy_client, self.secret, name)
            )
            server.daemon_threads = True
            # The containing host directory is private; the socket is mounted alone.
            path.chmod(0o666)
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self.servers.append(server)
            self.sockets[name] = path

    def bind(self, name):
        if self.options.transport == "http":
            return {
                "env": {
                    "WORLD_URL": self.url,
                    "WORLD_AGENT": name,
                    "WORLD_TOKEN": self.tokens[name],
                },
                "allowed_hosts": [urlsplit(self.url).hostname],
                "metadata": {"agent": name},
            }
        return {
            "env": {"WORLD_SOCKET": SOCKET},
            "mounts": [
                {
                    "type": "bind",
                    "source": str(self.sockets[name]),
                    "target": SOCKET,
                    "read_only": True,
                }
            ],
            "metadata": {"agent": name},
        }

    async def wait(self):
        misses = 0
        while True:
            await asyncio.sleep(HEALTH_EVERY_S)
            try:
                await self._health()  # parked counts as up: the world answers its own 503s
                misses = 0
            except Exception:
                misses += 1
                if misses >= HEALTH_MISSES:
                    raise

    async def export(self, directory):
        r = await self.client.get("/metrics")
        if r.status_code == 404:  # metrics are optional in World API v1
            return {}
        r.raise_for_status()
        write_json(directory / "metrics.json", r.json())
        return {"metrics": "metrics.json"}

    async def stop(self):
        for server in self.servers:
            await asyncio.to_thread(server.shutdown)
            server.server_close()
        if self.proxy_client:
            self.proxy_client.close()
        if self.client:
            await self.client.aclose()
        if self.socket_directory:
            self.socket_directory.cleanup()
