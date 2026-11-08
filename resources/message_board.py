"""The run's msgboard: a local board managed by the run, or a board on a Hugging Face Space."""

import asyncio
import importlib.metadata
import json
import os
import re
import socket
import sys
import tempfile
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import get_token
from huggingface_hub.utils import validate_repo_id
from msgboard import auth
from msgboard.client import BoardError, Client
from msgboard.space import Deployment
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flock.results import write_json

HEALTH_CHECK_SECONDS = 5
# While agents run: a Space hand-over keeps answering reads, health checks
# included. A new container that isn't handed the board (the old one crashed, or
# its uploads are throttled) waits until the old lease is 5 minutes stale, then
# restores and starts, so only a longer outage fails the run.
SPACE_OUTAGE_GRACE_SECONDS = 420
# After the run, Resources.finish gives export 120 s and a Space board's stop()
# 120 s, so an outage then can fail the run sooner. Export only reads, which a
# hand-over serves except while the new container restores (a few seconds); a
# failing request gives up within about 45 s. Revocation retries for 15 s, or for
# 100 s once the board itself says it is handing over: a hand-over pauses writes
# for a few seconds, and longer when bucket uploads are throttled (75 s was seen;
# there is no fixed bound). Each request step has a 2 s timeout, but connecting
# tries every address the host resolves to and DNS has no timeout, so the last
# attempt can end later. stop()'s limit leaves room for that.
EXPORT_CLIENT = {"timeout": 5, "max_retries": 5}
REVOKE_SECONDS = 15
HANDOVER_SECONDS = 100
HANDOVER_CODES = ("shutting_down", "starting")


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    channels: list[str] = Field(default_factory=lambda: ["general"], min_length=1)
    space_id: str | None = None
    private: bool | None = None
    startup_timeout_seconds: float | None = Field(default=None, gt=0)

    @field_validator("channels")
    @classmethod
    def channel_names(cls, value):
        if len(value) != len(set(value)) or any(
            not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", name) for name in value
        ):
            raise ValueError("channels must be unique lowercase channel names")
        return value

    @field_validator("space_id")
    @classmethod
    def hub_space_id(cls, value):
        if value is not None:
            validate_repo_id(value)  # `name` or `namespace/name`
        return value

    @model_validator(mode="after")
    def space_only_options(self):
        if self.private is not None and self.space_id is None:
            raise ValueError("message_board.private is only valid with space_id")
        return self


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
        self.url = ""
        self.credentials = {}
        self.channel_ids = set()
        self.invite_ids = set()
        self.metadata = {}
        self.operator_token = None
        self.space = self.hf_token = self.deploy_process = None
        self.startup_timeout_seconds = self.options.startup_timeout_seconds or (
            900 if self.options.space_id else 120
        )
        self.stop_timeout_seconds = (
            HANDOVER_SECONDS + 20 if self.options.space_id else 30
        )
        if self.options.space_id:
            self.hf_token = get_token()
            if not self.hf_token:
                raise ValueError(
                    "Set HF_TOKEN or run `hf auth login` for message_board.space_id"
                )
            self.secrets.append(self.hf_token)

    def _gate(self):
        """Client options for a private Space: its gate takes the launcher's HF token."""
        if not (self.space and self.space.private):
            return {}
        return {
            "headers": {"Authorization": f"Bearer {self.hf_token}"},
            "token_header": "X-Msgboard-Token",
        }

    async def start(self):
        self.state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.state_dir.chmod(0o700)
        if self.options.space_id:
            await self._start_space()
            # Retries outlast a restart that follows wait_ready().
            self.client = Client(
                self.url,
                self.operator_token,
                timeout=10,
                max_retries=10,
                **self._gate(),
            )
        else:
            await self._start_local()
            self.client = Client(
                self.url, self.operator_token, timeout=5, max_retries=2
            )
        identity = await call(self.client.me)
        identity = identity.get("principal", identity)
        if identity.get("role") != "admin":
            raise ValueError("Board resource requires an operator/admin token")
        await call(self.provision, self.agents)
        invite = await call(self._admin_invite, identity["id"]) if self.space else None
        dist = importlib.metadata.distribution("msgboard")
        self.metadata.update(
            {
                "url": self.url,
                "mode": "space" if self.space else "local",
                "transport": "http" if self.space else "unix",
                "channels": self.options.channels,
                "client_version": dist.version,
                "client_revision": json.loads(dist.read_text("direct_url.json") or "{}")
                .get("vcs_info", {})
                .get("commit_id"),
            }
        )
        print(f"Board: {self.url}", flush=True)
        if self.space:
            self.metadata["space"]["admin_invite_created"] = invite is not None
            if invite:
                print(
                    "Board admin invite (single use; open it and click "
                    f'"Join with Hugging Face"): {invite}',
                    flush=True,
                )
            print(f"Board token file: {self.state_dir / 'operator.token'}", flush=True)
        else:
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
            str(Path(__file__).with_name("message_board_server.py")),
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

    async def _start_space(self):
        loop = asyncio.get_running_loop()
        # Keep time to provision before Resources stops the startup: a quarter of
        # the limit, at most 30 s.
        limit = self.startup_timeout_seconds
        ready_by = loop.time() + limit - min(30, limit / 4)
        # A new operator token on every launch: deploy() makes it the Space's
        # admin credential, so a relaunch needs nothing from earlier runs.
        _, self.operator_token, _ = auth.new_token()
        self.secrets.append(self.operator_token)
        # Saved before deploy() replaces the Space's secret, so it is never lost.
        descriptor = os.open(
            self.state_dir / "operator.token",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        with os.fdopen(descriptor, "w") as stream:
            stream.write(self.operator_token + "\n")
        # deploy() puts a bare name in the token's account; Deployment has the full ID.
        self.space = await self._deploy()
        self.url = self.space.url
        # Recorded now, so a failed startup still names the Space and its bucket.
        self.metadata = {
            "url": self.url,
            "mode": "space",
            "space": {
                "id": self.space.space_id,
                "bucket": self.space.bucket_id,
                "private": self.space.private,
                "created": self.space.created,
                "commit": self.space.commit,
            },
        }
        page = f"https://huggingface.co/spaces/{self.space.space_id}"
        print(f"Board Space: {page} (waiting for it to build and start)", flush=True)
        # Short waits, so cancelling the launch doesn't wait out a long one.
        with Client(self.url, self.operator_token, timeout=5, **self._gate()) as probe:
            while True:
                try:
                    await call(
                        probe.wait_ready, max(1, min(10, ready_by - loop.time()))
                    )
                    return
                except TimeoutError as exc:
                    if loop.time() >= ready_by:
                        raise RuntimeError(
                            f"Board Space {self.space.space_id} was not ready within "
                            f"message_board.startup_timeout_seconds ({exc}); see {page}"
                        ) from None

    async def _deploy(self):
        """deploy() in a worker process, killed when the startup limit runs out or
        the launch is cancelled, since its Hub calls have no timeout."""
        request = {
            "space_id": self.options.space_id,
            "operator_token": self.operator_token,
            "private": self.options.private,
            "hf_token": self.hf_token,
        }
        self.deploy_process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path(__file__).with_name("message_board_deploy.py")),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
        )
        try:
            output, _ = await self.deploy_process.communicate(
                json.dumps(request).encode()
            )
        finally:
            if self.deploy_process.returncode is None:
                with suppress(ProcessLookupError):
                    self.deploy_process.kill()
                await self.deploy_process.wait()
        try:
            result = json.loads(output.decode().splitlines()[-1])
        except (IndexError, ValueError):
            result = {}
        if "deployment" not in result:
            # DeployError, Hub HTTP errors, HFValidationError and network errors alike.
            error = result.get("error") or (
                f"the deploy process exited with code {self.deploy_process.returncode}"
            )
            raise RuntimeError(
                f"Could not deploy the board to Space {self.options.space_id}: {error}"
            )
        return Deployment(**result["deployment"])

    def _admin_invite(self, operator_id):
        """A single-use admin invite link for a person, unless a person is already admin."""
        if any(
            person["identity_type"] == "user"
            and person["role"] == "admin"
            and not person["disabled"]
            and person["id"] != operator_id
            for person in self.client.people()
        ):
            return None
        invite = self.client.create_invite(
            "human", role="admin", max_uses=1, label="Flock admin invite"
        )
        link = f"{self.url}/?invite={invite['code']}"
        self.secrets.extend([invite["code"], link])
        return link

    def provision(self, names):
        visible = {channel["name"]: channel for channel in self.client.list_channels()}
        for name in self.options.channels:
            channel = visible.get(name) or self.client.create_channel(name)
            self.channel_ids.add(channel.get("channel", channel)["id"])
        participants = []
        for name in names:
            # Joining a named invite again re-keys an agent that an earlier run made.
            invite = self.client.create_invite(
                agent_name=name,
                role="member",
                can_create_channels=True,
                channels=sorted(self.channel_ids),
                max_uses=1,
            )
            self.invite_ids.add(invite["id"])
            with Client.join(
                self.url,
                invite["code"],
                timeout=self.client.timeout,
                max_retries=self.client.max_retries,
                **self._gate(),
            ) as agent:
                credential = {"identity": agent.identity, "token": agent.token}
                self.credentials[name] = credential
                participants.append(credential)
                self.secrets.append(agent.token)
                write_json(self.state_dir / "participants.json", self.credentials)
                (self.state_dir / "participants.json").chmod(0o600)
            # A re-key doesn't use up the invite, so it would keep making tokens.
            self.client.revoke_invite(invite["id"])
        return participants

    def bind(self, name):
        credential = self.credentials[name]
        env = {"MSGBOARD_URL": self.url, "MSGBOARD_TOKEN": credential["token"]}
        if self.space and self.space.private:
            # Client.from_env sends it to the Space's gate, and the board token apart.
            env["MSGBOARD_HF_TOKEN"] = self.hf_token
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
        # A Space board is watched through authenticated HTTP health checks.
        with Client(
            self.url, self.operator_token, timeout=5, max_retries=0, **self._gate()
        ) as probe:
            healthy = time.monotonic()
            while True:
                await asyncio.sleep(HEALTH_CHECK_SECONDS)
                try:
                    await call(probe.me)
                    healthy = time.monotonic()
                except Exception as exc:
                    if time.monotonic() - healthy > SPACE_OUTAGE_GRACE_SECONDS:
                        raise RuntimeError(
                            f"Board Space unreachable for {SPACE_OUTAGE_GRACE_SECONDS} s: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

    async def export(self, directory):
        if self.options.space_id and not self.client:
            raise RuntimeError(
                f"Board Space {self.options.space_id} did not start, so there is "
                "nothing to export"
            )
        if not self.client or (self.process and self.process.returncode is not None):
            raise RuntimeError(
                "Board unavailable; retained any local database in resource state"
            )
        await call(self._export, directory)
        return {"board": "board.jsonl", "channels": "channels.json"}

    def _export(self, directory):
        # A Space board gets an export client that fits Resources.finish's limit,
        # also when startup failed before it finished.
        board = (
            Client(self.url, self.operator_token, **EXPORT_CLIENT, **self._gate())
            if self.space
            else nullcontext(self.client)
        )
        with board as client:
            self._write_export(client, directory)

    def _write_export(self, client, directory):
        # A Space restored from a snapshot older than this run lacks what it made.
        if self.invite_ids - {invite["id"] for invite in client.invites()}:
            raise RuntimeError(
                "The board no longer has this run's agent invites; it was restored "
                "from a snapshot older than the run"
            )
        ids = {
            c["identity"].get("principal", c["identity"])["id"]
            for c in self.credentials.values()
        }
        channels = []
        for channel in client.list_channels():
            info = client.channel_info(channel["id"])
            if (
                channel["id"] in self.channel_ids
                or (info.get("created_by") or {}).get("id") in ids
            ):
                channel["members"] = info["members"]
                channels.append(channel)
        lost = self.channel_ids - {channel["id"] for channel in channels}
        if lost:
            raise RuntimeError(
                f"The board no longer has the run's channels with IDs {sorted(lost)}"
            )
        write_json(directory / "channels.json", channels)
        temporary = directory / "board.jsonl.tmp"
        with temporary.open("w") as stream:
            for channel in channels:
                after = 0
                while True:
                    messages = client.read(channel["id"], after=after, limit=100)[
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

    def _revoke_agent_tokens(self):
        """The Space outlives the run, so the agents' tokens must not. Every token
        is tried, and tried again, until REVOKE_SECONDS run out, or HANDOVER_SECONDS
        once the board says it is handing over."""
        start = time.monotonic()
        deadline = start + REVOKE_SECONDS
        pending = dict(self.credentials)
        error = None
        with Client(
            self.url, self.operator_token, max_retries=0, **self._gate()
        ) as client:
            while pending and time.monotonic() < deadline:
                for name, credential in list(pending.items()):
                    left = deadline - time.monotonic()
                    if left <= 0:
                        break
                    client.timeout = min(2, left)
                    try:
                        client.revoke_token(auth.parse_token(credential["token"])[0])
                    except BoardError as exc:
                        # Only the board's own 404 means it has no such token. A
                        # proxy's (a moved Space, a refused HF token) says nothing.
                        if (exc.status, exc.code) != (404, "not_found"):
                            error = exc
                            if exc.status == 503 and exc.code in HANDOVER_CODES:
                                deadline = max(deadline, start + HANDOVER_SECONDS)
                            continue
                    except Exception as exc:
                        error = exc
                        continue
                    del pending[name]
                if pending:
                    time.sleep(max(0, min(1, deadline - time.monotonic())))
        if pending:
            reason = (
                f"{type(error).__name__}: {error}"
                if error
                else "not tried before the deadline"
            )
            raise RuntimeError(
                f"Could not revoke the board tokens of {', '.join(pending)}, which "
                f"may still work: {reason}"
            )

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
        try:
            if self.space and self.credentials:
                await call(self._revoke_agent_tokens)
        finally:
            if self.client:
                self.client.close()
            if self.log:
                self.log.close()
            if self.socket_directory:
                self.socket_directory.cleanup()
