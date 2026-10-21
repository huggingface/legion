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

from flock.comms.board import AsyncBoard, BoardError, same_board
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
from flock.models import CommsSpec, comms_name, normalize_board_url
from flock.tools.runner import Result, check_args
from flock.tools.toolset import GATEWAY_LINE_MAX, Toolset
from flock.worker.identity import Identities, MsgboardIdentity

if TYPE_CHECKING:
    from flock.worker.agents import AgentRuntime

log = logging.getLogger(__name__)

POLL_WAIT_S = 25.0
JOIN_RETRY_S = 30.0
JOIN_ATTEMPTS = 3  # joins in a row made stale by a configure() before giving up
BACKLOG_MESSAGES = 500  # per agent, kept for an agent whose re-join is pending
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
    # set when a reconfiguration dropped the agent's identity: messages for it that arrive
    # before it is back are kept in `backlog` and delivered once it has re-joined
    pending: bool = False
    backlog: list[dict] = field(default_factory=list)
    # fallbacks when the agent has no runtime (tests); otherwise its synced AgentState
    posts: int = 0
    last_post: float | None = None


class _Stale(Exception):
    """A join attempt overtaken by a reconfiguration: dropped, nothing more is sent."""


@dataclass
class _Op:
    """One tool call: the member, its identity and the spec, captured once at the start; every
    request of the call goes to the identity's own board."""

    m: Member
    ident: MsgboardIdentity
    c: CommsSpec
    gen: int  # the hub's configuration generation when the call started

    def board(self, hub: CommsHub) -> AsyncBoard:
        return hub._board(self.ident.url)

    @property
    def source(self) -> Source:
        assert self.c.channel_id is not None
        return Source(self.ident.url, self.c.channel_id, self.gen)


@dataclass(frozen=True)
class Source:
    """Where messages came from: their board and channel, and the hub generation of the request
    that returned them. Message ids are only unique on one board, so every cache key carries
    the board and channel, and data from a superseded generation is never fanned out."""

    url: str
    channel_id: int
    gen: int

    def __post_init__(self) -> None:  # compared and keyed on the one spelling of the url
        object.__setattr__(self, "url", normalize_board_url(self.url))

    def key(self, msg_id: int) -> tuple[str, int, int]:
        return (self.url, self.channel_id, msg_id)


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
        # one client per board url; a credential is only ever sent through its own board's
        # client (and AsyncBoard refuses a credential issued by another board)
        self._boards: dict[str, AsyncBoard] = {}
        self.members: dict[str, Member] = {}
        self._keys: dict[str, str] = {}
        self._cache: OrderedDict[tuple[str, int, int], dict] = OrderedDict()  # (board, channel, id)
        self._cursor: int | None = None
        self._poller: asyncio.Task | None = None
        self._poll_agent: str | None = None
        self._server: asyncio.base_events.Server | None = None
        self._gateway_lock = asyncio.Lock()  # one listener, however many first prompts race
        self._restore: asyncio.Task | None = None
        self._gen = 0  # bumped by every reconfiguration that invalidates identities
        self.address: str | None = None
        self.stats: Counter[str] = Counter()

    # ------------------------------------------------------------------ configuration
    @property
    def active(self) -> bool:
        return self.spec is not None and self.spec.channel_id is not None and bool(self.entries)

    def configure(self, spec: CommsSpec | None, entries: list[str] | None) -> None:
        """The swarm's comms block (from an assign or a heartbeat response) and this worker
        type's comms tools. A changed board, channel or invite starts over: the cached
        identities are dropped, the poller and its cursor reset, and every agent that already
        joined here joins again at once with the (new) invite, so its subscription comes back
        without waiting for its next harness start. Tools switched back on restart a stopped
        poller. Agents that never had a prompt still join only at their first one."""
        old, was_active = self.spec, self.active
        key = (lambda c: (normalize_board_url(c.url), c.channel_id, c.invite_env))
        moved = old is not None and (spec is None or key(spec) != key(old))
        self.spec, self.entries = spec, list(entries or []) if spec is not None else []
        if moved:
            assert old is not None
            self._reset(old, same_board=spec is not None and same_board(spec.url, old.url))
        if self.active and self.members and (moved or not was_active):
            self._restore_members()

    def _reset(self, old: CommsSpec, *, same_board: bool) -> None:
        """Message ids grow across the whole board, so on the same board the poller keeps its
        cursor: a message posted in the new channel after the change is never missed, however
        long the joins take. A new board starts from "now"."""
        self._gen += 1
        if self._poller is not None:
            self._poller.cancel()
        self._poller, self._poll_agent = None, None
        if not same_board:
            self._cursor = None
        for m in self.members.values():
            # only agents that had joined wait for their messages (one that never joined here
            # has no subscription to restore); a new board has no shared cursor to keep
            m.pending = same_board and (m.ident is not None or m.pending)
            if not m.pending:
                m.backlog.clear()
            m.ident, m.error, m.retry_at, m.rejoin = None, None, 0.0, True
            self.identities.forget(m.name, url=old.url)  # redeem the new invite, never reuse
        self._cache.clear()

    def _restore_members(self) -> None:
        """(Re)join every agent that joined before, in the background; the poller starts with
        the first one back."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        if self._restore is not None and not self._restore.done():
            self._restore.cancel()

        async def run() -> None:
            for name in list(self.members):
                with contextlib.suppress(BoardError, ValueError):
                    await self._member(name)
            self._ensure_poller()

        self._restore = loop.create_task(run(), name="comms-restore")

    def _board(self, url: str) -> AsyncBoard:
        """The client for the board at `url`: always the url of the credential in hand (an
        identity's own url, or the spec captured at the start of a join), never "the current
        spec" read after an await."""
        key = normalize_board_url(url)
        if key not in self._boards:
            self._boards[key] = AsyncBoard(key, transport=self._transport, backoff_s=self.backoff_s)
        return self._boards[key]

    def tooldefs(self) -> tuple:
        return tuple(tool_def(e) for e in self.entries) if self.active else ()

    def deny_env(self) -> list[str]:
        """Variables a tool call never inherits: the comms invite (and MSGBOARD_TOKEN, the
        older msgboard_invite's token). The comms token itself is in no environment at all."""
        if self.spec is None:
            return []
        return sorted({self.spec.invite_env, "MSGBOARD_TOKEN"})

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
            self._ensure_poller()  # it may have stopped while the tools were off
            return m
        async with m.lock:
            if m.ident is not None:
                self._ensure_poller()
                return m
            if self._clock() < m.retry_at and m.error:
                raise BoardError(f"not on the board yet: {m.error} (retrying in "
                                 f"{int(m.retry_at - self._clock()) + 1} s)")
            for _ in range(JOIN_ATTEMPTS):
                # A configure() during the join's I/O (a new channel or invite) makes its
                # result stale: never publish it, join again against the current settings.
                gen, c = self._gen, self.spec
                if not self.active or c is None:
                    raise BoardError("comms is off for this agent")
                try:
                    ident = await self._join_once(m, c, gen)
                except _Stale:
                    m.rejoin = True
                    continue
                except BoardError as e:
                    if gen != self._gen:
                        m.rejoin = True
                        continue
                    m.error, m.retry_at = str(e), self._clock() + JOIN_RETRY_S
                    self._emit(agent, "comms_join_failed", {"error": m.error})
                    log.warning("%s: comms join failed: %s", agent, m.error)
                    raise
                if gen != self._gen:
                    m.rejoin = True
                    continue
                break
            else:
                raise BoardError("the comms settings kept changing during the join; retry")
            m.ident, m.rejoin, m.error = ident, False, None
            rt = self._runtime(agent)
            if rt is not None:
                m.threads.update(rt.state.comms_threads)
            self.stats["joins"] += 1
            self._emit(agent, "comms_joined", {"principal": ident.principal, "principal_id": ident.id,
                                               "channel": c.channel, "channel_id": c.channel_id})
            self._replay(m)
            self._ensure_poller()
            return m

    async def _join_once(self, m: Member, c: CommsSpec, gen: int) -> MsgboardIdentity:
        """Redeem the invite under the agent's name and check the channel is readable (raises
        BoardError). Joined means readable: a lookup that does not find the channel (an invite
        not scoped to a private one) is a failed join, never a success.

        Every request of the attempt goes to the board captured at its start (`c.url`), and
        after each await the generation is checked before anything else is sent: a
        reconfiguration during the attempt drops it (_Stale) without one more request, so a
        token is never sent to a board that did not issue it."""
        assert c.channel_id is not None
        board = self._board(c.url)

        def fresh() -> None:
            if gen != self._gen:
                raise _Stale
        code = self._env.get(c.invite_env)
        if not code:
            raise BoardError(f"the invite secret {c.invite_env} is not set on this worker")
        try:
            ident = await self.identities.join(m.name, c.url, code, again=m.rejoin)
        except httpx.HTTPStatusError as e:
            fresh()
            raise BoardError(f"join refused: HTTP {e.response.status_code}") from None
        except (httpx.HTTPError, KeyError, ValueError) as e:
            fresh()
            raise BoardError(f"join failed: {type(e).__name__}") from None
        fresh()
        chans = await board.channels(ident, c.channel)
        fresh()
        ch = next((x for x in chans if x.get("id") == c.channel_id), None)
        if ch is None:
            raise BoardError(f"#{c.channel} (id {c.channel_id}) is not visible to {m.name}: "
                             f"is the invite in {c.invite_env} scoped to it?")
        if not ch.get("member") and ident.id is not None:
            try:
                await board.add_member(ident, c.channel_id, ident.id)
            except BoardError as e:
                fresh()
                raise BoardError(f"cannot join #{c.channel}: {e}") from None
            fresh()
        if ident.id is not None:
            with contextlib.suppress(BoardError):  # cosmetic: the profile's runtime
                await board.set_runtime(ident, ident.id, {
                    "harness": f"flock/{m.harness}", "host": f"flock swarm {self.swarm}"})
            fresh()
        return ident

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

    def current(self) -> Source | None:
        """The subscription messages are fanned out for: the current board, channel and
        generation."""
        c = self.spec
        if c is None or c.channel_id is None:
            return None
        return Source(c.url, c.channel_id, self._gen)

    def fanout(self, msgs: list[dict], source: Source) -> None:
        """New channel messages (from the poller or this hub's own posts) -> wake events. Only
        messages of the current subscription count: data returned by a request of an older
        generation, another board or another channel is dropped (no cache entry, no event)."""
        c = self.spec
        if c is None or source != self.current():
            return
        for x in sorted(msgs, key=lambda x: x.get("id") or 0):
            mid = x.get("id")
            if not isinstance(mid, int) or x.get("channel_id") not in (None, source.channel_id):
                continue
            key = source.key(mid)
            if key in self._cache:
                continue
            self._cache[key] = x
            while len(self._cache) > CACHE_MESSAGES:
                self._cache.popitem(last=False)
            sender, root = author(x), thread_of(x)
            if sender in self.members:
                self._add_thread(self.members[sender], root)
            for name, m in self.members.items():
                if name == sender:
                    continue
                if m.ident is None:
                    if m.pending:  # re-joining after a reconfiguration: deliver once it is back
                        m.backlog = [*m.backlog, x][-BACKLOG_MESSAGES:]
                    continue
                self._deliver(m, x)

    def _deliver(self, m: Member, x: dict) -> None:
        c = self.spec
        assert c is not None
        sender = author(x)
        data = {"msg_id": x["id"], "thread": thread_of(x), "from": sender}
        if "mention" in c.wake and self._mentions(x, m):
            self.stats["mentions"] += 1
            self._emit(m.name, "comms_mention", data)
        elif "thread" in c.wake and x.get("thread_id") and x["thread_id"] in m.threads:
            self.stats["thread_replies"] += 1
            self._emit(m.name, "comms_thread_reply", data)

    def _replay(self, m: Member) -> None:
        """An agent is back after a re-join: the messages the poller saw meanwhile."""
        backlog, m.backlog, m.pending = m.backlog, [], False
        for x in backlog:
            if author(x) != m.name:
                self._deliver(m, x)

    # ------------------------------------------------------------------ the poller
    def _ensure_poller(self) -> None:
        if not self.active:
            return
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
            ident, c = (m.ident if m else None), self.spec
            if m is None or ident is None or c is None:
                return  # the next join starts it again
            assert c.channel_id is not None
            board = self._board(ident.url)  # the identity's own board, whatever the spec says
            source = Source(ident.url, c.channel_id, self._gen)
            try:
                if self._cursor is None:  # "now": history before the poller is not news
                    page = await board.activity(ident, c.channel_id, None)
                    self._cursor = int(page.get("next_cursor") or 0)
                    continue
                self.stats["polls"] += 1
                page = await board.activity(ident, c.channel_id, self._cursor, wait=self.poll_wait)
                attempt = 0
                self._cursor = int(page.get("next_cursor") or self._cursor)
                self.fanout([x for x in page.get("messages") or [] if isinstance(x, dict)], source)
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
                op = _Op(m, m.ident, self.spec, self._gen)  # one identity, its own board
                if op.ident is None or op.c is None or not same_board(op.ident.url, op.c.url):
                    raise BoardError("comms settings changed during the call; try again")
                fn = {"comms:post": self._post, "comms:read": self._read,
                      "comms:search": self._search}[entry]
                return _ok(await fn(op, args), t0)
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

    async def _message(self, op: _Op, msg_id: int) -> dict | None:
        """A message of the call's own board and channel (the cache is keyed by both)."""
        if (cached := self._cache.get(op.source.key(msg_id))) is not None:
            return cached
        try:
            msg = await op.board(self).message(op.ident, msg_id)
        except BoardError as e:
            if e.status in (403, 404):
                return None
            raise
        return msg if isinstance(msg, dict) and "id" in msg else None

    async def _post(self, op: _Op, args: dict[str, Any]) -> dict[str, Any]:
        m, c = op.m, op.c
        assert c.channel_id is not None
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
                root = await self._message(op, thread)
                if root is None or root.get("channel_id") not in (None, c.channel_id):
                    raise ValueError(f"no message {thread} in #{c.channel}")
                if root.get("thread_id"):
                    if rules.threads == "per_topic":
                        raise Refused("threads", f"{thread} is a reply in thread {root['thread_id']}; "
                                                 f"use thread: {root['thread_id']}")
                    thread = int(root["thread_id"])
                if reply is not None and reply != thread:
                    r = await self._message(op, reply)
                    if r is None or thread_of(r) != thread:
                        raise ValueError(f"reply_to {reply} is not in thread {thread}")
                reply_to = reply or thread
            elif reply is not None and rules.threads == "per_topic":
                raise Refused("threads", "say the thread too: thread: <root id> with reply_to")
            msg = await op.board(self).post(op.ident, c.channel_id, text, reply_to)
            n = self._record_post(m, self._clock())
        self.stats["posts"] += 1
        if isinstance(msg, dict) and isinstance(msg.get("id"), int):
            msg.setdefault("channel_id", c.channel_id)
            self.fanout([msg], op.source)  # dropped if the settings changed during the post
        out: dict[str, Any] = {"posted": msg.get("id"), "thread": thread_of(msg) if msg.get("id") else None}
        if rules.max_posts:
            out["posts_left"] = max(0, rules.max_posts - n)
        return out

    async def _read(self, op: _Op, args: dict[str, Any]) -> dict[str, Any]:
        c, ident = op.c, op.ident
        assert c.channel_id is not None
        thread = as_id(args.get("thread"), "thread")
        board = op.board(self)
        # the newest n (a thread: READ_MAX unless asked), read from the tail of the history;
        # `truncated` says older messages exist, never a first page passed off as the whole
        n = args.get("n") or (READ_MAX if thread is not None else READ_DEFAULT)
        n = max(1, min(int(n), READ_MAX))
        out: dict[str, Any] = {"channel": f"#{c.channel}"}
        if thread is not None:
            page = await board.thread(ident, c.channel_id, thread, n)
            msgs = [x for x in page.get("messages") or [] if isinstance(x, dict)]
            if msgs and msgs[0].get("id") == thread and msgs[0].get("thread_id"):
                thread = int(msgs[0]["thread_id"])  # a reply's id: read its thread
                page = await board.thread(ident, c.channel_id, thread, n)
                msgs = [x for x in page.get("messages") or [] if isinstance(x, dict)]
            if not msgs:
                raise ValueError(f"no thread {thread} in #{c.channel}")
            out["thread"] = thread
        else:
            page = await board.latest(ident, c.channel_id, n)
            msgs = [x for x in page.get("messages") or [] if isinstance(x, dict)]
        out["messages"] = [brief(x) for x in msgs]
        if page.get("has_more"):
            out["truncated"] = True
            out["note"] = (f"only the newest {len(msgs)} messages; older ones exist"
                           + (f" (n is at most {READ_MAX})" if n < READ_MAX else ""))
        return out

    async def _search(self, op: _Op, args: dict[str, Any]) -> dict[str, Any]:
        c, board = op.c, op.board(self)

        async def search(q: str, limit: int) -> dict:
            self.stats["searches"] += 1
            return await board.search(op.ident, q, sort="relevance", limit=limit)

        return await relaxed_search(search, query=args["query"], channel=c.channel,
                                    thread=as_id(args.get("thread"), "thread"),
                                    author_=args.get("from") or None, since=args.get("since") or None,
                                    limit=args.get("limit") or 0)

    # ------------------------------------------------------------------ gateway
    async def start_gateway(self) -> str:
        """The local endpoint `flock-tool` calls for built-ins (127.0.0.1, one JSON line each
        way; each agent's key is in its pin file)."""
        async with self._gateway_lock:
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
        if self._restore is not None:
            self._restore.cancel()
        if self._poller is not None:
            self._poller.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._poller
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
        for board in self._boards.values():
            await board.aclose()
