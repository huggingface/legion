"""The worker's comms hub (docs/comms.md): board identities, the one activity poller, fan-out,
the built-in `comms:*` tools with their rules, and the local gateway `flock-tool` calls.

- Join at an agent's first prompt (`on_prompt`, called before its harness starts), not at
  assign, so a 500-agent placement does not hit `/v1/join` in one burst. The agent joins under
  its flock name with the swarm's invite (`comms.invite_env`, a job secret on the worker); a
  relaunch on another worker joins again by name (same principal, new token). The token lives
  here, in memory: never in the agent's environment, workdir or pin file.
- One activity long-poll per worker (`GET /v1/activity?after=&wait=25&channels=<id>`), with the
  token of one joined agent on this worker. The board leaves the caller's own messages out of
  its feed, so posts made through this hub are fed to the fan-out directly. Each new channel
  message is matched against every joined agent here: a direct @mention (group mentions do
  not count) emits `comms_mention {msg_id, thread, from}` on that agent's event stream, a reply
  in a thread the agent posted in emits `comms_thread_reply {msg_id, thread, from}` (one event
  per message and agent, mention first; only the kinds in `comms.wake`). The controller hands
  those to its comms hook (`flock.controller.api`, `on_comms`), where the loop's wake queue
  plugs in.
- Tools: `post`, `read`, `search` (flock.comms.tools). Rules are checked before any board call;
  a refusal is the tool result `{"error": "refused: ..."}`. `max_posts` and the last post time
  are kept in the agent's synced state, so they survive a relaunch.
- Live settings: `configure` takes the swarm's comms block from every assign and heartbeat
  response; rules apply from the next call, protocols from the next turn boundary.
"""

from __future__ import annotations

import asyncio
import contextlib
import functools
import hmac
import json
import logging
import os
import re
import secrets
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from flock.comms.board import AsyncBoard, BoardError
from flock.comms.tools import (
    GROUP_NAMES,
    READ_DEFAULT,
    READ_MAX,
    Refused,
    as_id,
    author,
    brief,
    check_content,
    check_rate,
    relaxed_search,
    thread_of,
    tool_def,
)
from flock.models import CommsSpec, comms_name
from flock.tools.runner import Result, check_args
from flock.tools.toolset import GATEWAY_LINE_MAX, Toolset
from flock.worker.identity import Identities, MsgboardIdentity

if TYPE_CHECKING:
    from flock.worker.agents import AgentRuntime

log = logging.getLogger(__name__)

POLL_WAIT_S = 25.0
JOIN_RETRY_S = 30.0
CACHE_MESSAGES = 2000
THREADS_KEPT = 500
SECTION_START = "<!-- flock:comms -->"
SECTION_END = "<!-- /flock:comms -->"


@dataclass
class Member:
    """One agent of this worker on the board."""

    name: str
    harness: str = "llm"
    ident: MsgboardIdentity | None = None
    rejoin: bool = False  # the token was refused (rotated elsewhere): join again
    error: str | None = None
    retry_at: float = 0.0
    threads: set[int] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # fallbacks when the agent has no runtime (tests); otherwise its synced AgentState
    posts: int = 0
    last_post: float | None = None


def _ok(out: Any, t0: float) -> Result:
    return Result(json.dumps(out, ensure_ascii=False), False, time.monotonic() - t0)


def _err(msg: str, t0: float) -> Result:
    return Result(json.dumps({"error": msg}, ensure_ascii=False), True, time.monotonic() - t0)


class CommsHub:
    def __init__(self, *, swarm: str, identities: Identities | None = None,
                 runtime: Callable[[str], AgentRuntime | None] | None = None,
                 env: Mapping[str, str] | None = None,
                 transport: httpx.AsyncBaseTransport | None = None,
                 clock: Callable[[], float] = time.time, poll_wait: float = POLL_WAIT_S,
                 backoff_s: float = 1.0) -> None:
        self.swarm = swarm
        self.identities = identities or Identities(client=httpx.AsyncClient(transport=transport,
                                                                             timeout=20))
        self._runtime = runtime or (lambda name: None)
        self._env = env if env is not None else os.environ
        self._transport = transport
        self._clock = clock
        self.poll_wait = poll_wait
        self.backoff_s = backoff_s
        self.spec: CommsSpec | None = None
        self.entries: list[str] = []
        self.board: AsyncBoard | None = None
        self.members: dict[str, Member] = {}
        self._keys: dict[str, str] = {}
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self._cursor: int | None = None
        self._poller: asyncio.Task | None = None
        self._poll_agent: str | None = None
        self._server: asyncio.base_events.Server | None = None
        self.address: str | None = None
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------ configuration
    @property
    def active(self) -> bool:
        return self.spec is not None and self.spec.channel_id is not None and bool(self.entries)

    def configure(self, spec: CommsSpec | None, entries: list[str] | None) -> None:
        """The swarm's comms block (from an assign or a heartbeat response) and this worker
        type's comms tools. A changed board or channel starts over (joins, poller, cursor)."""
        old = self.spec
        moved = old is not None and (spec is None or (spec.url, spec.channel_id) != (old.url, old.channel_id))
        self.spec, self.entries = spec, list(entries or []) if spec is not None else []
        if moved:
            self._reset()

    def _reset(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
        self._poller, self._poll_agent, self._cursor = None, None, None
        for m in self.members.values():
            m.ident, m.error, m.retry_at = None, None, 0.0
        self._cache.clear()
        if self.board is not None:
            board, self.board = self.board, None
            with contextlib.suppress(RuntimeError):
                asyncio.get_running_loop().create_task(board.aclose())

    def _board(self) -> AsyncBoard:
        assert self.spec is not None
        if self.board is None:
            self.board = AsyncBoard(self.spec.url, transport=self._transport, backoff_s=self.backoff_s)
        return self.board

    def tooldefs(self) -> tuple:
        return tuple(tool_def(e) for e in self.entries) if self.active else ()

    def key(self, agent: str) -> str:
        if agent not in self._keys:
            self._keys[agent] = secrets.token_urlsafe(24)
        return self._keys[agent]

    def toolset_for(self, agent: str, base: Toolset) -> Toolset:
        """`base` (the folder tools of this turn) plus this agent's comms tools."""
        defs = self.tooldefs()
        if not defs:
            return base
        names = set(base.names())
        return replace(base, tools=base.tools + tuple(d for d in defs if d.name not in names),
                       gateway=self.address, gateway_key=self.key(agent),
                       builtin_call=functools.partial(self.call, agent))

    def section(self, agent: str) -> str:
        """The comms section of the agent's instructions: who it is on the board, the tools,
        the rules in force, then the protocol files (`comms.protocols`)."""
        if not self.active:
            return ""
        c = self.spec
        assert c is not None
        names = [f"`{comms_name(e)}`" for e in self.entries]
        r = c.rules
        rules = []
        if r.min_post_gap:
            rules.append(f"at most one post every {r.min_post_gap}")
        if r.max_posts:
            rules.append(f"at most {r.max_posts} posts in total")
        if r.max_len:
            rules.append(f"at most {r.max_len} characters per post")
        if r.threads == "per_topic":
            rules.append('one thread per topic: post with thread: "new" to open a topic, or '
                         "thread: <root id> to reply in an existing one")
        if r.no_group_mentions:
            rules.append("no @all, @agents or @humans; mention people by name")
        lines = [SECTION_START, "## Team board", "",
                 (f"You are @{agent} on the message board channel #{c.channel}. Use the "
                  f"{', '.join(names)} tools for it (you hold no board token, so not curl).")]
        if c.kickoff_id:
            lines.append(f"The kickoff is message {c.kickoff_id}; its thread is the run's main thread.")
        if rules:
            lines += ["", "Rules (the post tool refuses a post that breaks one and says why):"]
            lines += [f"- {x}" for x in rules]
        for path, text in zip(c.protocols, c.protocols_text, strict=False):
            lines += ["", f"### Protocol: {Path(path).stem}", "", text.strip()]
        lines.append(SECTION_END)
        return "\n".join(lines)

    # ------------------------------------------------------------------ joining
    async def on_prompt(self, agent: str, harness: str = "llm") -> None:
        """An agent's harness is about to start for a prompt: join the board now (first time on
        this worker), and open the gateway for a CLI harness. Never raises."""
        if not self.active:
            return
        if harness != "llm":
            await self.start_gateway()
        self.members.setdefault(agent, Member(agent)).harness = harness
        with contextlib.suppress(BoardError, ValueError):
            await self._member(agent)

    async def _member(self, agent: str) -> Member:
        m = self.members.setdefault(agent, Member(agent))
        if m.ident is not None:
            return m
        async with m.lock:
            if m.ident is not None:
                return m
            if self._clock() < m.retry_at and m.error:
                raise BoardError(f"not on the board yet: {m.error} (retrying in "
                                 f"{int(m.retry_at - self._clock()) + 1} s)")
            c = self.spec
            assert c is not None and c.channel_id is not None
            try:
                code = self._env.get(c.invite_env)
                if not code:
                    raise BoardError(f"the invite secret {c.invite_env} is not set on this worker")
                try:
                    ident = await self.identities.join(agent, c.url, code, again=m.rejoin)
                except httpx.HTTPStatusError as e:
                    raise BoardError(f"join refused: HTTP {e.response.status_code}") from None
                except (httpx.HTTPError, KeyError, ValueError) as e:
                    raise BoardError(f"join failed: {type(e).__name__}") from None
                board = self._board()
                with contextlib.suppress(BoardError):  # the invite usually adds the channel
                    chans = await board.channels(ident.token, c.channel)
                    if chans and not chans[0].get("member") and ident.id is not None:
                        await board.add_member(ident.token, c.channel_id, ident.id)
                    if ident.id is not None:
                        await board.set_runtime(ident.token, ident.id, {
                            "harness": f"flock/{m.harness}", "host": f"flock swarm {self.swarm}"})
            except BoardError as e:
                m.error, m.retry_at = str(e), self._clock() + JOIN_RETRY_S
                self._emit(agent, "comms_join_failed", {"error": m.error})
                log.warning("%s: comms join failed: %s", agent, m.error)
                raise
            m.ident, m.rejoin, m.error = ident, False, None
            rt = self._runtime(agent)
            if rt is not None:
                m.threads.update(rt.state.comms_threads)
            self.stats["joins"] += 1
            self._emit(agent, "comms_joined", {"principal": ident.principal, "principal_id": ident.id,
                                               "channel": c.channel, "channel_id": c.channel_id})
            self._ensure_poller()
            return m

    def forget(self, agent: str) -> None:
        """The agent left this worker (unassign)."""
        self.members.pop(agent, None)
        self._keys.pop(agent, None)
        self.identities.forget(agent)
        if self._poll_agent == agent and self._poller is not None:
            self._poller.cancel()
            self._poller, self._poll_agent = None, None
            self._ensure_poller()

    def _refused_token(self, m: Member) -> None:
        """A 401: the token was rotated (the agent joined elsewhere) or revoked."""
        m.ident, m.rejoin = None, True

    # ------------------------------------------------------------------ events
    def _emit(self, agent: str, kind: str, data: dict[str, Any]) -> None:
        rt = self._runtime(agent)
        if rt is not None:
            rt.emit(kind, data)

    def _state(self, m: Member) -> tuple[int, float | None]:
        rt = self._runtime(m.name)
        if rt is not None:
            return rt.state.comms_posts, rt.state.comms_last_post
        return m.posts, m.last_post

    def _record_post(self, m: Member, when: float) -> int:
        rt = self._runtime(m.name)
        if rt is None:
            m.posts, m.last_post = m.posts + 1, when
            return m.posts
        rt.state.comms_posts += 1
        rt.state.comms_last_post = when
        with contextlib.suppress(OSError):
            rt.save_state()
        return rt.state.comms_posts

    def _add_thread(self, m: Member, root: int) -> None:
        if root in m.threads:
            return
        m.threads.add(root)
        rt = self._runtime(m.name)
        if rt is not None:
            rt.state.comms_threads = [*rt.state.comms_threads, root][-THREADS_KEPT:]

    def _mentions(self, msg: dict, m: Member) -> bool:
        """A direct @mention of this agent (group mentions such as @agents do not count)."""
        if "mentions" in msg:
            for x in msg.get("mentions") or []:
                who = (m.ident is not None and x.get("id") == m.ident.id) or x.get("external_id") == m.name
                if who and str(x.get("as") or "").lstrip("@").lower() not in GROUP_NAMES:
                    return True
            return False
        return re.search(rf"(?<![\w@])@{re.escape(m.name)}(?![\w-])", msg.get("body") or "") is not None

    def fanout(self, msgs: list[dict]) -> None:
        """New channel messages (from the poller or this hub's own posts) -> wake events."""
        c = self.spec
        if c is None:
            return
        for x in sorted(msgs, key=lambda x: x.get("id") or 0):
            mid = x.get("id")
            if not isinstance(mid, int) or mid in self._cache or x.get("channel_id") not in (None, c.channel_id):
                continue
            self._cache[mid] = x
            while len(self._cache) > CACHE_MESSAGES:
                self._cache.popitem(last=False)
            sender, root = author(x), thread_of(x)
            if sender in self.members:
                self._add_thread(self.members[sender], root)
            for name, m in self.members.items():
                if name == sender or m.ident is None:
                    continue
                data = {"msg_id": mid, "thread": root, "from": sender}
                if "mention" in c.wake and self._mentions(x, m):
                    self.stats["mentions"] += 1
                    self._emit(name, "comms_mention", data)
                elif "thread" in c.wake and x.get("thread_id") and x["thread_id"] in m.threads:
                    self.stats["thread_replies"] += 1
                    self._emit(name, "comms_thread_reply", data)

    # ------------------------------------------------------------------ the poller
    def _ensure_poller(self) -> None:
        if self._poller is None or self._poller.done():
            self._poller = asyncio.get_running_loop().create_task(self._poll_loop(), name="comms-poll")

    def _poll_member(self) -> Member | None:
        cur = self.members.get(self._poll_agent or "")
        if cur is not None and cur.ident is not None:
            return cur
        m = next((m for m in self.members.values() if m.ident is not None), None)
        self._poll_agent = m.name if m else None
        return m

    async def _poll_loop(self) -> None:
        attempt = 0
        while self.active:
            m = self._poll_member()
            if m is None or m.ident is None:
                return  # the next join starts it again
            c = self.spec
            assert c is not None and c.channel_id is not None
            try:
                if self._cursor is None:  # "now": history before the poller is not news
                    page = await self._board().activity(m.ident.token, c.channel_id, None)
                    self._cursor = int(page.get("next_cursor") or 0)
                    continue
                self.stats["polls"] += 1
                page = await self._board().activity(m.ident.token, c.channel_id, self._cursor,
                                                    wait=self.poll_wait)
                attempt = 0
                self._cursor = int(page.get("next_cursor") or self._cursor)
                self.fanout([x for x in page.get("messages") or [] if isinstance(x, dict)])
            except asyncio.CancelledError:
                raise
            except BoardError as e:
                if e.status == 401:
                    self._refused_token(m)
                    continue
                attempt += 1
                log.warning("comms poll failed: %s", e)
                await asyncio.sleep(min(30.0, self.backoff_s * 2 ** min(attempt, 5)))
            except Exception as e:  # noqa: BLE001 - keep polling
                attempt += 1
                log.warning("comms poll failed: %s: %s", type(e).__name__, e)
                await asyncio.sleep(min(30.0, self.backoff_s * 2 ** min(attempt, 5)))

    # ------------------------------------------------------------------ tools
    async def call(self, agent: str, entry: str, args: dict[str, Any]) -> Result:
        """One built-in comms call by `agent` (the llm harness in process, flock-tool through the
        gateway). Never raises: errors and refusals are results the model reads."""
        t0 = time.monotonic()
        if not self.active or entry not in self.entries:
            return _err(f"{entry} is not enabled for this agent", t0)
        if err := check_args(tool_def(entry).parameters, args):
            return _err(f"{comms_name(entry)}: {err}", t0)
        for attempt in (0, 1):
            try:
                m = await self._member(agent)
                fn = {"comms:post": self._post, "comms:read": self._read,
                      "comms:search": self._search}[entry]
                return _ok(await fn(m, args), t0)
            except Refused as e:
                self.stats["refused"] += 1
                self._emit(agent, "comms_refused", {"rule": e.rule, "error": str(e)})
                return _err(str(e), t0)
            except BoardError as e:
                if e.status == 401 and attempt == 0 and agent in self.members:
                    self._refused_token(self.members[agent])
                    continue
                return _err(str(e), t0)
            except ValueError as e:
                return _err(f"{comms_name(entry)}: {e}", t0)
        return _err("unreachable", t0)

    async def _message(self, m: Member, msg_id: int) -> dict | None:
        if msg_id in self._cache:
            return self._cache[msg_id]
        assert m.ident is not None
        try:
            msg = await self._board().message(m.ident.token, msg_id)
        except BoardError as e:
            if e.status in (403, 404):
                return None
            raise
        return msg if isinstance(msg, dict) and "id" in msg else None

    async def _post(self, m: Member, args: dict[str, Any]) -> dict[str, Any]:
        c = self.spec
        assert c is not None and c.channel_id is not None and m.ident is not None
        rules = c.rules
        text, raw = args.get("text"), args.get("thread")
        check_content(rules, text, raw)
        new = isinstance(raw, str) and raw.strip().lower() == "new"
        thread = None if new else as_id(raw, "thread")
        reply = as_id(args.get("reply_to"), "reply_to")
        if new and reply is not None:
            raise ValueError('thread: "new" opens a topic; leave out reply_to')
        async with m.lock:
            posts, last = self._state(m)
            check_rate(rules, posts, last, self._clock())
            reply_to = reply
            if thread is not None:
                root = await self._message(m, thread)
                if root is None or root.get("channel_id") not in (None, c.channel_id):
                    raise ValueError(f"no message {thread} in #{c.channel}")
                if root.get("thread_id"):
                    if rules.threads == "per_topic":
                        raise Refused("threads", f"{thread} is a reply in thread {root['thread_id']}; "
                                                 f"use thread: {root['thread_id']}")
                    thread = int(root["thread_id"])
                if reply is not None and reply != thread:
                    r = await self._message(m, reply)
                    if r is None or thread_of(r) != thread:
                        raise ValueError(f"reply_to {reply} is not in thread {thread}")
                reply_to = reply or thread
            elif reply is not None and rules.threads == "per_topic":
                raise Refused("threads", "say the thread too: thread: <root id> with reply_to")
            msg = await self._board().post(m.ident.token, c.channel_id, text, reply_to)
            n = self._record_post(m, self._clock())
        self.stats["posts"] += 1
        if isinstance(msg, dict) and isinstance(msg.get("id"), int):
            msg.setdefault("channel_id", c.channel_id)
            self.fanout([msg])
        out: dict[str, Any] = {"posted": msg.get("id"), "thread": thread_of(msg) if msg.get("id") else None}
        if rules.max_posts:
            out["posts_left"] = max(0, rules.max_posts - n)
        return out

    async def _read(self, m: Member, args: dict[str, Any]) -> dict[str, Any]:
        c = self.spec
        assert c is not None and c.channel_id is not None and m.ident is not None
        thread = as_id(args.get("thread"), "thread")
        board = self._board()
        if thread is not None:
            msgs = await board.thread(m.ident.token, c.channel_id, thread)
            if len(msgs) == 1 and msgs[0].get("thread_id"):  # a reply's id: read its thread
                thread = int(msgs[0]["thread_id"])
                msgs = await board.thread(m.ident.token, c.channel_id, thread)
            if not msgs:
                raise ValueError(f"no thread {thread} in #{c.channel}")
            return {"channel": f"#{c.channel}", "thread": thread, "messages": [brief(x) for x in msgs]}
        n = args.get("n") or READ_DEFAULT
        n = max(1, min(int(n), READ_MAX))
        msgs = await board.latest(m.ident.token, c.channel_id, n)
        return {"channel": f"#{c.channel}", "messages": [brief(x) for x in msgs]}

    async def _search(self, m: Member, args: dict[str, Any]) -> dict[str, Any]:
        c = self.spec
        assert c is not None and m.ident is not None
        token = m.ident.token

        async def search(q: str, limit: int) -> dict:
            self.stats["searches"] += 1
            return await self._board().search(token, q, sort="relevance", limit=limit)

        return await relaxed_search(search, query=args["query"], channel=c.channel,
                                    thread=as_id(args.get("thread"), "thread"),
                                    author_=args.get("from") or None, since=args.get("since") or None,
                                    limit=args.get("limit") or 0)

    # ------------------------------------------------------------------ gateway
    async def start_gateway(self) -> str:
        """The local endpoint `flock-tool` calls for built-ins (127.0.0.1, one JSON line each
        way; each agent's key is in its pin file)."""
        if self._server is None:
            self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0,
                                                      limit=GATEWAY_LINE_MAX)
            port = self._server.sockets[0].getsockname()[1]
            self.address = f"127.0.0.1:{port}"
        assert self.address is not None
        return self.address

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            req = json.loads(await reader.readline())
            agent, key = str(req.get("agent")), str(req.get("key") or "")
            if agent not in self._keys or not hmac.compare_digest(key, self._keys[agent]):
                r = Result(json.dumps({"error": "gateway: unknown agent or key"}), True)
            else:
                r = await self.call(agent, str(req.get("tool")), req.get("args") or {})
            writer.write(json.dumps({"output": r.output, "error": r.error,
                                     "duration_s": r.duration_s}).encode() + b"\n")
            await writer.drain()
        except (OSError, ValueError, AttributeError) as e:
            log.warning("comms gateway: bad request: %s", type(e).__name__)
        finally:
            writer.close()

    async def aclose(self) -> None:
        if self._poller is not None:
            self._poller.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._poller
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        if self.board is not None:
            await self.board.aclose()
