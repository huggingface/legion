# from shepherd f3970bc:resources/board.py
"""A managed local msgboard or a connection to a dedicated existing board."""

import asyncio
import importlib.metadata
import json
import os
import re
import socket
import sys
import tempfile
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

from msgboard.client import Client
from pydantic import BaseModel, ConfigDict, Field, field_validator

from flock.harbor.results import write_json


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    url: str | None = None
    token_env: str = Field(
        default="MSGBOARD_OPERATOR_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    channels: list[str] = Field(default_factory=lambda: ["general"], min_length=1)

    @field_validator("url")
    @classmethod
    def endpoint(cls, value):
        if value is None:
            return value
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
                "board.url must be HTTP(S) without embedded credentials, query or fragment"
            )
        return value.rstrip("/")

    @field_validator("channels")
    @classmethod
    def channel_names(cls, value):
        if len(value) != len(set(value)) or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) for name in value
        ):
            raise ValueError("channels must be unique lowercase channel names")
        return value


async def call(function, *args):
    """Finish an in-flight SDK operation before cancellation closes its client."""
    task = asyncio.create_task(asyncio.to_thread(function, *args))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        with suppress(Exception):
            await task
        raise


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.options = Options.model_validate(options)
        self.state_dir = state_dir
        self.data_dir = state_dir / "data"
        self.agents = agents
        self.secrets = secrets
        self.process = self.client = self.log = None
        self.socket_directory = self.socket_path = None
        self.url = self.options.url or ""
        self.credentials = {}
        self.channel_ids = set()
        self.metadata = {}
        self.operator_token = None
        if self.options.url:
            self.operator_token = os.environ.get(self.options.token_env)
            if not self.operator_token:
                raise ValueError(
                    f"Set {self.options.token_env} to an operator token for board.url"
                )
            self.secrets.append(self.operator_token)

    async def start(self):
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        if not self.options.url:
            await self._start_local()
        self.client = Client(self.url, self.operator_token, timeout=5, max_retries=2)
        identity = await call(self.client.me)
        if identity.get("principal", identity).get("role") != "admin":
            raise ValueError("Board resource requires an operator/admin token")
        if self.options.url:
            people = await call(self.client.people)
            conflicts = {p["external_id"] for p in people} & set(self.agents)
            if conflicts:
                raise ValueError(
                    f"Board already has participant identities: {', '.join(sorted(conflicts))}; use a dedicated board"
                )
        await call(self.provision, self.agents)
        dist = importlib.metadata.distribution("msgboard")
        self.metadata = {
            "url": self.url,
            "mode": "external" if self.options.url else "managed",
            "transport": "http" if self.options.url else "unix",
            "channels": self.options.channels,
            "client_version": dist.version,
            "client_revision": json.loads(dist.read_text("direct_url.json") or "{}")
            .get("vcs_info", {})
            .get("commit_id"),
        }
        print(f"Board: {self.url}", flush=True)
        if not self.options.url:
            print(f"Board token file: {self.data_dir / 'operator.token'}", flush=True)

    async def _start_local(self):
        self.log = (self.state_dir / "board.log").open("ab")
        init = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "msgboard",
            "init",
            "--data-dir",
            str(self.data_dir),
            stdout=self.log,
            stderr=self.log,
        )
        try:
            if await init.wait():
                raise RuntimeError("msgboard init failed; see resource board.log")
        finally:
            if init.returncode is None:
                with suppress(ProcessLookupError):
                    init.terminate()
                await init.wait()
        self.operator_token = (self.data_dir / "operator.token").read_text().strip()
        self.secrets.append(self.operator_token)
        self.socket_directory = tempfile.TemporaryDirectory(prefix="flock-board-")
        self.socket_path = Path(self.socket_directory.name) / "board.sock"
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("board_server.py")),
            str(self.data_dir),
            "127.0.0.1",
            str(port),
            str(self.socket_path),
            stdout=self.log,
            stderr=self.log,
        )
        with Client(self.url, self.operator_token, timeout=0.5, max_retries=0) as probe:
            for _ in range(100):
                if self.process.returncode is not None:
                    raise RuntimeError(
                        "msgboard exited during startup; see resource board.log"
                    )
                try:
                    await call(probe.me)
                    return
                except Exception:
                    await asyncio.sleep(0.1)
        raise RuntimeError("msgboard did not become ready; see resource board.log")

    def provision(self, names):
        visible = {channel["name"]: channel for channel in self.client.list_channels()}
        for name in self.options.channels:
            channel = visible.get(name) or self.client.create_channel(name)
            self.channel_ids.add(channel.get("channel", channel)["id"])
        participants = []
        for name in names:
            invite = self.client.create_invite(
                agent_name=name,
                role="member",
                can_create_channels=True,
                channels=sorted(self.channel_ids),
                max_uses=1,
            )
            with Client.join(self.url, invite["code"]) as agent:
                credential = {"identity": agent.identity, "token": agent.token}
                self.credentials[name] = credential
                participants.append(credential)
                self.secrets.append(agent.token)
                write_json(self.state_dir / "participants.json", self.credentials)
                (self.state_dir / "participants.json").chmod(0o600)
        return participants

    def bind(self, name):
        credential = self.credentials[name]
        env = {"MSGBOARD_URL": self.url, "MSGBOARD_TOKEN": credential["token"]}
        mounts = []
        hosts = [urlsplit(self.url).hostname]
        if self.socket_path:
            env["MSGBOARD_SOCKET"] = "/run/flock-board.sock"
            mounts = [
                {
                    "type": "bind",
                    "source": str(self.socket_path),
                    "target": env["MSGBOARD_SOCKET"],
                    "read_only": True,
                }
            ]
            hosts = []
        return {
            "env": env,
            "mounts": mounts,
            "allowed_hosts": hosts,
            "metadata": {"identity": credential["identity"]},
        }

    async def wait(self):
        if self.process:
            code = await self.process.wait()
            raise RuntimeError(f"Board exited unexpectedly ({code})")
        # Remote failures are detected through authenticated HTTP health checks.
        with Client(self.url, self.operator_token, timeout=5, max_retries=0) as probe:
            while True:
                await asyncio.sleep(5)
                await call(probe.me)

    async def export(self, directory):
        if not self.client or (self.process and self.process.returncode is not None):
            raise RuntimeError(
                "Board unavailable; retained any local database in resource state"
            )
        await call(self._export, directory)
        return {"board": "board.jsonl", "channels": "channels.json"}

    def _export(self, directory):
        ids = {
            c["identity"].get("principal", c["identity"])["id"]
            for c in self.credentials.values()
        }
        channels = []
        for channel in self.client.list_channels():
            info = self.client.channel_info(channel["id"])
            if (
                channel["id"] in self.channel_ids
                or (info.get("created_by") or {}).get("id") in ids
            ):
                channel["members"] = info["members"]
                channels.append(channel)
        write_json(directory / "channels.json", channels)
        temporary = directory / "board.jsonl.tmp"
        with temporary.open("w") as stream:
            for channel in channels:
                after = 0
                while True:
                    messages = self.client.read(channel["id"], after=after, limit=100)[
                        "messages"
                    ]
                    if not messages:
                        break
                    for message in messages:
                        stream.write(
                            json.dumps(
                                {"channel_id": channel["id"], "message": message}
                            )
                            + "\n"
                        )
                    next_seq = max(m["seq"] for m in messages)
                    if next_seq <= after:
                        raise RuntimeError("Non-advancing msgboard history cursor")
                    after = next_seq
        temporary.replace(directory / "board.jsonl")

    async def stop(self):
        if self.process and self.process.returncode is None:
            with suppress(ProcessLookupError):
                self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=10)
            except TimeoutError:
                with suppress(ProcessLookupError):
                    self.process.kill()
                await self.process.wait()
        if self.client:
            self.client.close()
        if self.log:
            self.log.close()
        if self.socket_directory:
            self.socket_directory.cleanup()
