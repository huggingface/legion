"""The run's msgboard: a local board managed by the run, or a board on a Hugging Face Space.

flock's additions (lines marked `flock:`; unused options keep the upstream behaviour):
- `url` / `token_env`: connect to a board that is already up. Each flock worker runs its own resources, and a
  `space_id` board is deployed (and its operator token rotated) by every one that starts: a run
  of more than one worker deploys its board once and connects every worker to it this way.
- `add` / `remove`: agents placed after start join under their names; a name the board already
  has re-joins with a fresh token (its old one stops working). So give each run its own board,
  or another run's agents of the same names take over its tokens.
- `watch(emit)`: one activity poller (the operator's feed of `channels`) turns each new message
  into `emit(agent, kind, data)` for the agents added here: `mention` (named directly), `thread`
  (a reply in a thread the agent posted in) or `channel` (a new topic), the first that applies
  among the kinds in `wake`. The feed leaves out the operator's own posts: a message posted
  with the operator token (the kickoff, or a person using it) wakes no one.
- `rules` (flock's comms rules): each agent reaches the board (local, Space or `url`) only
  through its own Unix socket, served by a proxy that checks every post with `check_content` /
  `check_rate` and every new channel against `max_channels_per_agent` / `max_channels`, opens
  only the board tools' routes, refuses an agent's ninth call in flight (429) and sends the
  agent's token, which never enters the container. `threads: per_topic` is not enforced
  (message_board.post names no thread). With `transport: http` (no mounts, e.g. hf-sandbox)
  there is no proxy: the rules are not enforced, and the resource says so in its metadata and in
  every agent's binding.
- `kickoff`: posted once to the first channel at start (every worker posts it under one key).
- `private: false` with a reused Space that is private: start fails (a private Space's gate
  would need the HF token in every agent's container).
- A worker restarted on the same state dir (local compute shares it) renames the earlier
  `operator.token` to `operator.token.<ns>` instead of failing to create it.
"""

import asyncio
import contextlib
import hashlib
import http.server
import importlib.metadata
import json
import os
import re
import socket
import socketserver
import sys
import tempfile
import threading
import time
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from huggingface_hub import get_token
from huggingface_hub.utils import validate_repo_id
from msgboard import auth
from msgboard.client import BoardError, Client
from msgboard.space import Deployment
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from flock.comms.tools import GROUP_NAMES, Refused, check_content, check_rate
from flock.harbor.results import write_json
from flock.models import CommsRules

HEALTH_CHECK_SECONDS = 5
# While agents run, a Space board can be unreachable for minutes and come back: a
# restart, or a crash, after which msgboard waits for the old container's lease to
# go stale before it restores. Only a longer outage fails the run.
SPACE_OUTAGE_GRACE_SECONDS = 420
# After the run, Resources.finish gives export 120 s and a Space board's stop()
# 120 s, so an outage then can fail the run sooner. A failing export request gives
# up within about 45 s. Revocation retries for 15 s, or for 100 s once the board
# itself says it is shutting down or starting, that is, restarting. Each request
# step has a 2 s timeout, but connecting tries every address the host resolves to
# and DNS has no timeout, so the last attempt can end later. stop()'s limit leaves
# room for that.
EXPORT_CLIENT = {"timeout": 5, "max_retries": 5}
REVOKE_SECONDS = 15
RESTART_SECONDS = 100
RESTART_CODES = ("shutting_down", "starting")
# flock: the constants below
SOCKET = "/run/flock-board.sock"
POLL_WAIT = 25  # s, one activity long-poll
# What the rule proxy opens, after the board's own path: the routes of the message_board.* and
# message_board_extra.* tools. Never invites, tokens or admin routes (an invite an agent made
# would give it an identity without rules).
ROUTES = {
    "GET": r"/v1/(me|principals|channels|search|activity|channels/\d+/(messages|members|doc))",
    "POST": r"/v1/channels(/\d+/(messages|members))?",
    "PUT": r"/v1/channels/\d+/(doc|messages/\d+/reactions/[^/]+)",
    "DELETE": r"/v1/channels/\d+/members/\d+",
}
FORWARD_IN = ("content-type", "accept", "idempotency-key")  # never the agent's own auth headers
FORWARD_OUT = ("content-type", "retry-after", "idempotent-replayed")
BODY_MAX = 1 << 20
CALLS_MAX = 8  # board calls one agent has in flight; more are refused with 429


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    channels: list[str] = Field(default_factory=lambda: ["general"], min_length=1)
    space_id: str | None = None
    private: bool | None = None
    startup_timeout_seconds: float | None = Field(default=None, gt=0)
    # flock: a board that is already up (the run's, shared by its workers), its operator token's
    # env name; comms rules, the kinds watch() emits, the kickoff text, and whether agents can
    # mount a socket (unix) or not (http: no rule proxy)
    url: str | None = None
    token_env: str = Field(
        default="MSGBOARD_OPERATOR_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    rules: CommsRules | None = None
    wake: list[Literal["mention", "thread", "channel"]] = ["mention", "thread"]
    kickoff: str | None = None
    transport: Literal["unix", "http"] = "unix"

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

    @field_validator("url")  # flock: as f3970bc's board.url
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
                "message_board.url must be HTTP(S) without embedded credentials, query or "
                "fragment"
            )
        return value.rstrip("/")

    @model_validator(mode="after")
    def space_only_options(self):
        if self.private is not None and self.space_id is None:
            raise ValueError("message_board.private is only valid with space_id")
        if self.url and self.space_id:  # flock:
            raise ValueError("message_board: set space_id or url, not both")
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
        # flock: the agents added here -> the threads each posted in (watch); rule proxies
        self.threads = {}
        self.proxies = {}
        self.proxy_client = self.proxy_directory = None
        self.adding = asyncio.Lock()
        self.warning = None
        if self.options.rules is not None and self.options.transport == "http":
            self.warning = "board rules are not enforced: transport http has no rule proxy"
        if self.options.url:  # flock: as f3970bc's board.url
            self.url = self.options.url
            self.operator_token = os.environ.get(self.options.token_env)
            if not self.operator_token:
                raise ValueError(
                    f"Set {self.options.token_env} to an operator token for message_board.url"
                )
            self.secrets.append(self.operator_token)
        self.startup_timeout_seconds = self.options.startup_timeout_seconds or (
            900 if self.options.space_id else 120
        )
        self.stop_timeout_seconds = (
            RESTART_SECONDS + 20 if self.options.space_id else 30
        )
        if self.options.space_id:
            self.hf_token = get_token()
            if not self.hf_token:
                raise ValueError(
                    "Set HF_TOKEN or run `hf auth login` for message_board.space_id"
                )
            self.secrets.append(self.hf_token)

    def _ruled(self):  # flock:
        return self.options.rules is not None and self.options.transport == "unix"

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
        if self.options.space_id or self.options.url:  # flock: or url
            if self.options.space_id:
                await self._start_space()
                if self.options.private is False and self.space.private:  # flock:
                    # a reused Space keeps its visibility, and a private one's gate would need
                    # the HF token in every agent's container
                    raise RuntimeError(
                        f"Board Space {self.space.space_id} is private (message_board.private: "
                        "false): make it public or use another Space"
                    )
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
        if self.options.kickoff:  # flock:
            await call(self._kickoff)
        invite = await call(self._admin_invite, identity["id"]) if self.space else None
        dist = importlib.metadata.distribution("msgboard")
        self.metadata.update(
            {
                "url": self.url,
                "mode": "space" if self.space else "external" if self.options.url else "local",
                # flock: a rule proxy's socket for any board
                "transport": "unix" if self.socket_path or self._ruled() else "http",
                "channels": self.options.channels,
                "client_version": dist.version,
                "client_revision": json.loads(dist.read_text("direct_url.json") or "{}")
                .get("vcs_info", {})
                .get("commit_id"),
            }
        )
        if self.warning:  # flock:
            self.metadata["warning"] = self.warning
            print(f"Warning: {self.warning}", flush=True)
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
        elif not self.options.url:  # flock: or url
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
        token_file = self.state_dir / "operator.token"
        if token_file.exists():  # flock: a restarted local worker shares its state dir; keep both
            token_file.rename(token_file.with_name(f"operator.token.{time.time_ns()}"))
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
            "token": self.hf_token,
        }
        # -P keeps resources/ off the worker's sys.path, so a provider module named
        # like one it imports can't shadow it. It exits if this process dies first.
        self.deploy_process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-P",
            str(Path(__file__).with_name("message_board_deploy.py")),
            str(os.getpid()),
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
        source = self._proxy(name) if self._ruled() else self.socket_path  # flock: or a proxy
        if self._ruled():  # flock: the agent's token, the board's path and gate stay with its proxy
            env = {"MSGBOARD_URL": "http://board", "MSGBOARD_TOKEN": "via-proxy"}
        if source:
            env["MSGBOARD_SOCKET"] = SOCKET
            mounts = [
                {
                    "type": "bind",
                    "source": str(source),
                    "target": env["MSGBOARD_SOCKET"],
                    "read_only": True,
                }
            ]
            hosts = []
        return {
            "env": env,
            "mounts": mounts,
            "allowed_hosts": hosts,
            "metadata": {"identity": credential["identity"]}
            | ({"warning": self.warning} if self.warning else {}),  # flock:
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
        is tried, and tried again, until REVOKE_SECONDS run out, or RESTART_SECONDS
        once the board says it is restarting."""
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
                            if exc.status == 503 and exc.code in RESTART_CODES:
                                deadline = max(deadline, start + RESTART_SECONDS)
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
        await self.remove(list(self.proxies))  # flock: the rule proxies
        if self.proxy_client:
            self.proxy_client.close()
        if self.proxy_directory:
            self.proxy_directory.cleanup()
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

    # flock: the hooks flock's worker calls (flock.harbor.resources); the upstream launcher does not
    def _kickoff(self):
        channel = self.client.channel_id(self.options.channels[0])
        key = hashlib.sha256(f"{channel}\n{self.options.kickoff}".encode()).hexdigest()[:32]
        self.client.post(channel, self.options.kickoff, request_id=f"flock-kickoff-{key}")

    async def add(self, names):
        """Agents that now run here join (again: a fresh token); idempotent while they stay."""
        async with self.adding:  # provision writes participants.json
            new = [name for name in names if name not in self.threads]
            if new:
                await call(self.provision, new)
            for name in new:
                self.threads[name] = set()

    async def remove(self, names):
        for name in names:
            self.threads.pop(name, None)
        proxies = [self.proxies.pop(name) for name in names if name in self.proxies]
        await asyncio.gather(*(asyncio.to_thread(server.shutdown) for server, _ in proxies))
        for server, path in proxies:
            server.server_close()
            path.unlink(missing_ok=True)

    async def watch(self, emit):
        # shortcut: the operator's feed of `channels` only (no DMs, no channels agents created),
        # and an agent's threads are those it posted in on this worker; upgrade with per-agent
        # feeds when a run needs DM or thread wakes across worker moves
        params = {"channels": ",".join(str(c) for c in sorted(self.channel_ids)), "limit": 100}
        gate = self._gate()  # a private Space's gate takes the HF token, the board's goes apart
        headers = {**gate.get("headers", {}),
                   gate.get("token_header", "Authorization"): (
                       self.operator_token if gate else f"Bearer {self.operator_token}")}
        cursor, failures = None, 0
        async with httpx.AsyncClient(base_url=self.url, headers=headers,
                                     timeout=POLL_WAIT + 15) as client:
            while True:
                after = {} if cursor is None else {"after": cursor, "wait": POLL_WAIT}
                try:
                    r = await client.get("/v1/activity", params=params | after)
                    r.raise_for_status()
                    page = r.json()
                except (httpx.HTTPError, ValueError) as exc:
                    failures += 1
                    print(f"Board poll failed ({type(exc).__name__}); retrying", flush=True)
                    await asyncio.sleep(min(30, 2**failures))
                    continue
                failures = 0
                if cursor is not None:  # the first page is history, not news
                    for message in page.get("messages") or []:
                        if isinstance(message, dict) and isinstance(message.get("id"), int):
                            self._fan_out(message, emit)
                cursor = page.get("next_cursor") or cursor or 0

    def _fan_out(self, message, emit):
        sender = message.get("sender") or {}
        author, parent = sender.get("external_id"), message.get("thread_id")
        thread = parent or message["id"]
        if author in self.threads:
            self.threads[author].add(thread)
        data = {"msg_id": message["id"], "thread": thread, "from": author,
                "identity_type": sender.get("identity_type")}
        for name, threads in self.threads.items():
            identity = self.credentials[name]["identity"]
            me = identity.get("principal", identity)["id"]
            direct = any(
                (m.get("id") == me or m.get("external_id") == name)
                and str(m.get("as") or "").lstrip("@").lower() not in GROUP_NAMES
                for m in message.get("mentions") or []
            )
            found = {"mention": direct, "thread": parent in threads, "channel": not parent}
            kind = next((k for k in found if found[k] and k in self.options.wake), None)
            if kind and name != author:
                emit(name, kind, data)

    def _proxy(self, name):
        """The agent's rule-proxy socket, started at its first bind."""
        if name in self.proxies:
            return self.proxies[name][1]
        if self.proxy_directory is None:
            self.proxy_directory = tempfile.TemporaryDirectory(prefix="flock-board-rules-")
            # an activity long-poll waits up to 25 s
            self.proxy_client = httpx.Client(timeout=httpx.Timeout(60, connect=10))
        path = Path(self.proxy_directory.name) / f"{name}.sock"
        server = socketserver.ThreadingUnixStreamServer(str(path), self._handler(name))
        server.daemon_threads = True
        # The containing host directory is private; the socket is mounted alone.
        path.chmod(0o666)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.proxies[name] = (server, path)
        return path

    def _handler(self, name):
        resource, url, client = self, self.url, self.proxy_client
        base = httpx.URL(url)
        prefix = re.escape(base.path.rstrip("/"))
        allowed = {method: re.compile(prefix + route) for method, route in ROUTES.items()}
        rules = self.options.rules
        # shortcut: the agent's posts, last post time and channels made live in this proxy, so
        # max_posts, min_post_gap and max_channels_per_agent restart when the agent moves; keep
        # them in its state if runs need that
        made = {"posts": 0, "last_post": None, "channels": 0}
        lock, slots = threading.Lock(), threading.BoundedSemaphore(CALLS_MAX)

        def check(kind, body, headers):
            """Raises Refused: the post's rules, or the channel caps (as the comms hub's)."""
            if kind == "post":
                sent = json.loads(body or b"{}")
                # shortcut: message_board.post has no thread argument, so a post without
                # reply_to starts a topic and `threads: per_topic` never refuses; enforce it
                # when the board tools can name a thread
                check_content(rules, sent.get("body"), sent.get("reply_to") or "new")
                check_rate(rules, made["posts"], made["last_post"], time.time())
            elif made["channels"] >= rules.max_channels_per_agent:
                raise Refused("max_channels_per_agent", "you reached your channel creation limit")
            else:
                listed = client.get(url + "/v1/channels", headers=headers)
                listed.raise_for_status()
                if sum(c.get("kind") != "dm" for c in listed.json()["channels"]) >= rules.max_channels:
                    raise Refused("max_channels",
                                  "the channel limit is reached; join an existing channel")

        class Handler(http.server.BaseHTTPRequestHandler):
            timeout = 60  # a client that stops sending or reading

            def handle(self):
                # A few calls at once per agent, never queued: one agent's long-polls must not
                # hold the shared client or the worker's threads that the others need.
                if not slots.acquire(blocking=False):
                    self.wfile.write(b"HTTP/1.0 429 Too Many Requests\r\nRetry-After: 1\r\n"
                                     b"Content-Length: 0\r\n\r\n")
                    return
                try:
                    super().handle()
                finally:
                    slots.release()

            def reply(self, status, body, headers=()):
                self.send_response(status)
                for key, value in headers:
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def refuse(self, message):
                error = {"error": {"code": "refused", "message": message}}
                self.reply(403, json.dumps(error).encode(), [("Content-Type", "application/json")])

            def forward(self):
                # Checked on the URL as sent (dot segments resolved), as resources.world's proxy.
                try:
                    target = httpx.URL(url + self.path) if self.path.startswith("/") else None
                except httpx.InvalidURL:
                    target = None
                route = allowed.get(self.command)
                if (target is None or route is None or target.host != base.host
                        or target.port != base.port or not route.fullmatch(target.path)):
                    self.send_error(403)
                    return
                try:
                    length = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    length = -1
                if not 0 <= length <= BODY_MAX:
                    self.send_error(413 if length > 0 else 400)
                    return
                body = self.rfile.read(length)
                headers = {k: v for k, v in self.headers.items() if k.lower() in FORWARD_IN}
                gate = resource._gate()  # a private Space's gate: the HF token, the board's apart
                headers.update(gate.get("headers", {}))
                token = resource.credentials[name]["token"]
                headers[gate.get("token_header", "Authorization")] = (
                    token if gate else f"Bearer {token}")
                kind = None
                if self.command == "POST":
                    kind = "post" if target.path.endswith("/messages") else (
                        "channel" if target.path.endswith("/v1/channels") else None)
                with lock if kind else contextlib.nullcontext():
                    try:
                        if kind:
                            check(kind, body, headers)
                        r = client.request(self.command, target, content=body, headers=headers)
                    except Refused as exc:
                        self.refuse(str(exc))
                        return
                    except (ValueError, AttributeError, KeyError, TypeError):
                        self.send_error(400)
                        return
                    except httpx.HTTPError as exc:
                        self.send_error(502, f"board unreachable ({type(exc).__name__})")
                        return
                    if kind and r.status_code == 201:  # made (200: a replayed post)
                        if kind == "post":
                            made.update(posts=made["posts"] + 1, last_post=time.time())
                        else:
                            made["channels"] += 1
                self.reply(r.status_code, r.content,
                           [(k, r.headers[k]) for k in FORWARD_OUT if k in r.headers])

            do_GET = do_POST = do_PUT = do_DELETE = forward

            def log_message(self, *args):
                pass

        return Handler
