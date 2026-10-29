"""Multi-channel tools and one rotating activity task per worker.

Each identity keeps its own cursor: another agent's token cannot see its DMs.
The single polling task rotates identities on a `comms.poll_cycle_s` cycle (default 300 s),
even on busy feeds.
No service credential, per-agent poll tasks, or per-channel pollers are needed.
"""

import asyncio
import contextlib
import logging
import random
import re
from collections import OrderedDict
from dataclasses import replace
from time import monotonic

from flock.comms.board import BoardError
from flock.comms.tools import Refused, as_id, author, brief, check_content, check_rate, thread_of

log = logging.getLogger(__name__)

POLL_BACKOFF_MAX_S = 120.0
CACHE_TTL_S = 60.0  # per agent: its channel list and the principals directory
TOOL_CACHE_MAX = 64  # entries kept per agent (channel list, directory, DM pages); oldest out
READ_DMS_CHANNELS = 10  # read_dms reads at most this many DM channels (the newest)


def poll_backoff(failures: int, base: float, retry_after: float | None = None,
                 rng=random.uniform) -> float:
    """Full jitter: uniform(0, min(cap, base * 2^failures)), never shorter than Retry-After."""
    d = rng(0.0, min(POLL_BACKOFF_MAX_S, base * 2 ** min(failures, 16)))
    return max(d, retry_after or 0.0)


class ExtendedComms:
    async def _cached(self, m, key, fetch, *, fresh=False):
        """`fetch()` at most once per CACHE_TTL_S per agent and key. Returns (value, hit)."""
        now = monotonic()
        hit = self._recall(m.name, key)
        if not fresh and hit is not None and now - hit[0] < CACHE_TTL_S:
            return hit[1], True
        value = await fetch()
        self._remember(m.name, key, (now, value))
        return value, False

    def _remember(self, agent, key, entry):
        """Store a tool cache entry, newest last; an agent keeps at most TOOL_CACHE_MAX (a
        read_dms page per DM channel and page size would otherwise pile up for the run)."""
        cache = self._tool_cache.setdefault(agent, {})
        cache.pop(key, None)
        cache[key] = entry
        while len(cache) > TOOL_CACHE_MAX:
            cache.pop(next(iter(cache)))

    def _recall(self, agent, key):
        return self._tool_cache.get(agent, {}).get(key)

    def _forget(self, m, key):
        self._tool_cache.get(m.name, {}).pop(key, None)

    async def _channels(self, op, *, fresh=False):
        return await self._cached(op.m, "channels", lambda: op.board(self).channels(op.ident),
                                  fresh=fresh)

    async def _channel_op(self, op, channel):
        if channel is None:
            return op
        name = str(channel).lstrip("#")
        channels, hit = await self._channels(op)

        def find(chs):
            return next((c for c in chs if str(c["id"]) == name or c["name"] == name), None)
        match = find(channels)
        if match is None and hit:  # created or opened since the cached list
            channels, _ = await self._channels(op, fresh=True)
            match = find(channels)
        if match is None or match.get("kind") == "dm":
            raise ValueError("channel is not visible; use channels to find it")
        op.m.channels.update(c["id"] for c in channels if c.get("member"))
        return replace(op, c=op.c.model_copy(update={"channel": match["name"],
                                                    "channel_id": match["id"], "kickoff_id": None}))

    async def _extended_call(self, op, entry, args):
        board, ident, m = op.board(self), op.ident, op.m
        if entry == "comms:channels":
            channels, _ = await self._channels(op)
            m.channels.update(c["id"] for c in channels if c.get("member"))
            return {"channels": channels}
        if entry == "comms:join_channel":
            selected = await self._channel_op(op, args["name"])
            await board.add_member(ident, selected.c.channel_id, ident.id)
            self._forget(m, "channels")  # membership changed
            m.channels.add(selected.c.channel_id)
            return {"joined": selected.c.channel, "id": selected.c.channel_id}
        if entry == "comms:create_channel":
            if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", args["name"]):
                raise ValueError("use a lowercase channel slug")
            async with self._channel_lock:
                rt = self._runtime(m.name)
                created = rt.state.comms_created_channels if rt else m.created_channels
                if len(created) >= op.c.rules.max_channels_per_agent:
                    raise Refused("max_channels_per_agent", "you reached your channel creation limit")
                channels = await board.channels(ident)
                if len([c for c in channels if c.get("kind") != "dm"]) >= op.c.rules.max_channels:
                    raise Refused("max_channels", "the channel limit is reached; join an existing channel")
                # The count is read from the board, but two workers can both pass it before
                # either creates: the cap can be exceeded by at most one channel per other
                # worker creating at the same moment (docs/comms.md).
                channel = await board.request(("POST", "/v1/channels", {"json": {
                    "name": args["name"], "description": args["purpose"], "private": False}}), ident)
                if channel["id"] not in created:  # a retried create returns the same channel
                    created.append(channel["id"])
                self._forget(m, "channels")
                m.channels.add(channel["id"])
                if rt:
                    rt.save_state()
                return {"created": channel["name"], "id": channel["id"]}
        if entry == "comms:dm":
            async with m.lock:
                check_content(op.c.rules.model_copy(update={"threads": None}), args["text"], None)
                posts, last = self._state(m)
                check_rate(op.c.rules, posts, last, self._clock())
                data, _ = await self._cached(m, "principals", lambda: board.request(
                    ("GET", "/v1/principals", {}), ident))
                name = args["to"].lstrip("@")
                people = data.get("principals") or []
                person = next((p for p in people if name in (p.get("external_id"), p.get("display_name"))), None)
                if person is None:
                    raise ValueError("no agent or human with that name")
                channel = await board.request(("POST", "/v1/dms", {"json": {"with": person["id"]}}), ident)
                self._forget(m, "channels")  # possibly a new DM channel
                msg = await board.post(ident, channel["id"], args["text"])
                self._record_post(m, self._clock())
                msg["channel"] = channel
                msg["channel_id"] = channel["id"]
                self._extended_fanout([msg], op.source)
                return {"posted": msg["id"], "channel": channel["id"]}
        if entry == "comms:read_dms":
            # The newest DM channels only, and a channel's page again only when its last_seq
            # moved: 1 + (changed channels) requests instead of 1 + every DM channel.
            listed, _ = await self._channels(op, fresh=True)
            channels = sorted((c for c in listed if c.get("kind") == "dm"),
                              key=lambda c: c["id"], reverse=True)[:READ_DMS_CHANNELS]
            n = max(1, min(args.get("n", 20), 100))
            since = as_id(args.get("since"), "since") or 0
            messages = []
            for channel in channels:
                key = ("dm", channel["id"], n)
                seen = self._recall(m.name, key)
                if seen is not None and channel.get("last_seq") is not None \
                        and seen[0] == channel["last_seq"]:
                    page = seen[1]
                else:
                    page = await board.latest(ident, channel["id"], n)
                    self._remember(m.name, key, (channel.get("last_seq"), page))
                messages.extend({**brief(x), "channel": channel["id"]}
                                for x in page.get("messages", []) if x["id"] > since)
            return {"messages": sorted(messages, key=lambda x: x["id"])[-n:]}
        if entry == "comms:react":
            msg = await board.message(ident, as_id(args["msg_id"], "msg_id"))
            await board.react(ident, msg["channel_id"], msg["id"], args["emoji"])
            return {"reacted": msg["id"], "emoji": args["emoji"]}
        raise ValueError("unknown board tool")

    def _extended_deliver(self, m, msg):
        key = (self.spec.url, m.name, msg["id"])
        if key in self._delivered or author(msg) == m.name:
            return
        if msg.get("reason") == "reply":
            self._add_thread(m, thread_of(msg))
        self._delivered[key] = True
        while len(self._delivered) > 10000:
            self._delivered.popitem(last=False)
        self._deliver(m, msg)

    def _extended_fanout(self, messages, source):
        c = self.spec
        if c is None or source.url != c.url or source.gen != self._gen:
            return
        for msg in messages:
            sender = self.members.get(author(msg))
            if sender is not None:
                self._add_thread(sender, thread_of(msg))
            channel = msg.get("channel") or {"id": msg.get("channel_id", source.channel_id)}
            msg["channel"] = channel
            for m in list(self.members.values()):
                if not m.ident:
                    continue
                if channel.get("kind") == "dm":
                    allowed = m.ident.id in [p["id"] for p in channel.get("members", []) if p]
                else:
                    allowed = channel["id"] in m.channels
                if allowed:
                    self._extended_deliver(m, msg)

    async def _extended_join(self, m, c):
        m.channels.add(c.channel_id)
        rt = self._runtime(m.name)
        if rt and rt.state.comms_board == c.url and rt.state.comms_cursor is not None:
            m.cursor = rt.state.comms_cursor
        else:
            page = await self._board(c.url).activity(m.ident, "joined", None)
            m.cursor = int(page.get("next_cursor") or 0)

    async def _poll_extended(self):
        index = failures = 0
        while self.active and self.spec.extended:
            members = [m for m in self.members.values() if m.ident is not None]
            if not members:
                return
            started = monotonic()
            m = members[index % len(members)]
            index += 1
            ident, c, gen = m.ident, self.spec, self._gen
            # A failed poll sleeps its backoff first; the next start still waits for the
            # spacing, so a backoff can only slow polling down, never speed it up.
            cycle = self.poll_cycle_s or c.poll_cycle_s
            spacing = max(self.poll_min_interval, cycle / len(members))
            try:
                self.stats["polls"] += 1
                page = await self._board(ident.url).activity(ident, "joined", m.cursor,
                                                            wait=min(self.poll_wait, spacing))
                if gen != self._gen or self.spec is not c:
                    await asyncio.sleep(max(0, spacing - (monotonic() - started)))
                    continue
                for msg in page.get("messages") or []:
                    if isinstance(msg, dict):
                        # This response was authorized for m, including private channels/DMs.
                        self._extended_deliver(m, msg)
                m.cursor = int(page.get("next_cursor") or m.cursor or 0)
                rt = self._runtime(m.name)
                if rt:
                    rt.state.comms_cursor, rt.state.comms_board = m.cursor, c.url
                    with contextlib.suppress(OSError):
                        rt.save_state()
                failures = 0
            except asyncio.CancelledError:
                raise
            except BoardError as exc:
                if exc.status == 401:
                    self._refused_token(m)
                    with contextlib.suppress(Exception):
                        await self._member(m.name)
                else:
                    # One task polls for every agent of the worker: this spaces all of them.
                    failures += 1
                    delay = poll_backoff(failures, self.backoff_s, exc.retry_after)
                    log.warning("comms poll for %s failed: %s; next poll in %.1f s", m.name, exc, delay)
                    await asyncio.sleep(delay)
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop every agent's notices
                failures += 1
                delay = poll_backoff(failures, self.backoff_s)
                log.warning("comms poll for %s failed: %s: %s; next poll in %.1f s", m.name,
                            type(exc).__name__, exc, delay)
                await asyncio.sleep(delay)
            await asyncio.sleep(max(0, spacing - (monotonic() - started)))

    def _init_extended(self):
        self._channel_lock = asyncio.Lock()
        self._tool_cache: dict[str, dict] = {}  # agent -> {key: (stamp, value)}
        self._delivered: OrderedDict[tuple[str, str, int], bool] = OrderedDict()
