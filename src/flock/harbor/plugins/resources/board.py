# from shepherd f3970bc:resources/board.py
"""A managed local msgboard or a connection to a dedicated existing board.

flock's additions (lines marked `flock:`; unused options keep shepherd's behaviour):
- `add` / `remove`: agents placed after start join under their names; a name the board already
  has re-joins with a fresh token (its old one stops working). So shepherd's dedicated-board
  check cannot run here (another worker's or a re-assigned agent's name is on the board too):
  give each run its own board, or another run's agents of the same names take over its tokens.
- `watch(emit)`: one activity poller (the operator's feed of `channels`) turns each new message
  into `emit(agent, kind, data)` for the agents added here: `mention` (named directly), `thread`
  (a reply in a thread the agent posted in) or `channel` (a new topic), the first that applies
  among the kinds in `wake`. The feed leaves out the operator's own posts: a message posted
  with the operator token (the kickoff, or a person using it) wakes no one.
- `rules` (flock's comms rules): each agent reaches the board only through its own Unix socket,
  served by a proxy that checks every post with `check_content` / `check_rate` and every new
  channel against `max_channels_per_agent` / `max_channels`, opens only the board tools'
  routes, refuses an agent's ninth call in flight (429) and sends the agent's token, which never
  enters the container. `threads: per_topic` is not enforced (shepherd's board.post names no
  thread). With `transport: http` (no mounts, e.g. hf-sandbox) there is no proxy: the rules are
  not enforced, and the resource says so in its metadata and in every agent's binding.
- `kickoff`: posted once to the first channel at start (every worker posts it under one key).
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
from contextlib import suppress
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
from msgboard.client import Client
from pydantic import BaseModel, ConfigDict, Field, field_validator

from flock.comms.tools import GROUP_NAMES, Refused, check_content, check_rate
from flock.harbor.results import write_json
from flock.models import CommsRules

# flock: the constants below
SOCKET = "/run/flock-board.sock"
POLL_WAIT = 25  # s, one activity long-poll
# What the rule proxy opens, after the board's own path: the routes of shepherd's board.*
# and board_extra.* tools. Never invites, tokens or admin routes (an invite an agent made would
# give it an identity without rules).
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
    url: str | None = None
    token_env: str = Field(
        default="MSGBOARD_OPERATOR_TOKEN", pattern=r"^[A-Za-z_][A-Za-z0-9_]*$"
    )
    channels: list[str] = Field(default_factory=lambda: ["general"], min_length=1)
    # flock: comms rules, the kinds watch() emits, the kickoff text, and whether agents can
    # mount a socket (unix) or not (http: no rule proxy)
    rules: CommsRules | None = None
    wake: list[Literal["mention", "thread", "channel"]] = ["mention", "thread"]
    kickoff: str | None = None
    transport: Literal["unix", "http"] = "unix"

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
        # flock: the agents added here -> the threads each posted in (watch); rule proxies
        self.threads = {}
        self.proxies = {}
        self.proxy_client = self.proxy_directory = None
        self.adding = asyncio.Lock()
        self.warning = None
        if self.options.rules is not None and self.options.transport == "http":
            self.warning = "board rules are not enforced: transport http has no rule proxy"

    def _ruled(self):
        return self.options.rules is not None and self.options.transport == "unix"

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
        if self.options.kickoff:  # flock:
            await call(self._kickoff)
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
        if self.warning:  # flock:
            self.metadata["warning"] = self.warning
            print(f"Warning: {self.warning}", flush=True)
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
        source = self._proxy(name) if self._ruled() else self.socket_path  # flock: or a proxy
        if self._ruled():  # flock: the agent's token and the board's path stay with its proxy
            env.update(MSGBOARD_URL="http://board", MSGBOARD_TOKEN="via-proxy")
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
        if self.client:
            self.client.close()
        if self.log:
            self.log.close()
        if self.socket_directory:
            self.socket_directory.cleanup()


    # flock: the hooks flock's worker calls (flock.harbor.resources); shepherd's launcher does not
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
        headers = {"Authorization": f"Bearer {self.operator_token}"}
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
                # shortcut: shepherd's board.post has no thread argument, so a post without
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
                headers["Authorization"] = f"Bearer {resource.credentials[name]['token']}"
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
