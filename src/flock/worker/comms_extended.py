"""Multi-channel tools and one rotating activity task per worker.

Each identity keeps its own cursor: another agent's token cannot see its DMs.
The single polling task rotates identities on a 60-second cycle, even on busy feeds.
No service credential, per-agent poll tasks, or per-channel pollers are needed.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from collections import OrderedDict
from dataclasses import replace
from time import monotonic

from flock.comms.board import BoardError
from flock.comms.tools import Refused, as_id, author, brief, check_content, check_rate, thread_of

log = logging.getLogger(__name__)


class ExtendedComms:
    async def _channel_op(self, op, channel):
        if channel is None:
            return op
        name = str(channel).lstrip("#")
        channels = await op.board(self).channels(op.ident)
        match = next((c for c in channels if str(c["id"]) == name or c["name"] == name), None)
        if match is None or match.get("kind") == "dm":
            raise ValueError("channel is not visible; use channels to find it")
        op.m.channels.update(c["id"] for c in channels if c.get("member"))
        return replace(op, c=op.c.model_copy(update={"channel": match["name"],
                                                    "channel_id": match["id"], "kickoff_id": None}))

    async def _extended_call(self, op, entry, args):
        board, ident, m = op.board(self), op.ident, op.m
        if entry == "comms:channels":
            channels = await board.channels(ident)
            m.channels.update(c["id"] for c in channels if c.get("member"))
            return {"channels": channels}
        if entry == "comms:join_channel":
            selected = await self._channel_op(op, args["name"])
            await board.add_member(ident, selected.c.channel_id, ident.id)
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
                channel = await board.request(("POST", "/v1/channels", {"json": {
                    "name": args["name"], "description": args["purpose"], "private": False}}), ident)
                created.append(channel["id"])
                m.channels.add(channel["id"])
                if rt:
                    rt.save_state()
                return {"created": channel["name"], "id": channel["id"]}
        if entry == "comms:dm":
            async with m.lock:
                check_content(op.c.rules.model_copy(update={"threads": None}), args["text"], None)
                posts, last = self._state(m)
                check_rate(op.c.rules, posts, last, self._clock())
                data = await board.request(("GET", "/v1/principals", {}), ident)
                name = args["to"].lstrip("@")
                people = data.get("principals") or []
                person = next((p for p in people if name in (p.get("external_id"), p.get("display_name"))), None)
                if person is None:
                    raise ValueError("no agent or human with that name")
                channel = await board.request(("POST", "/v1/dms", {"json": {"with": person["id"]}}), ident)
                msg = await board.post(ident, channel["id"], args["text"])
                self._record_post(m, self._clock())
                msg["channel"] = channel
                msg["channel_id"] = channel["id"]
                self._extended_fanout([msg], op.source)
                return {"posted": msg["id"], "channel": channel["id"]}
        if entry == "comms:read_dms":
            channels = [c for c in await board.channels(ident) if c.get("kind") == "dm"]
            n = max(1, min(args.get("n", 20), 100))
            since = as_id(args.get("since"), "since") or 0
            messages = []
            for channel in channels:
                page = await board.latest(ident, channel["id"], n)
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
            spacing = max(self.poll_min_interval, self.poll_cycle_s / len(members))
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
                    failures += 1
                    log.warning("comms poll for %s failed: %s", m.name, exc)
                    await asyncio.sleep(min(30.0, self.backoff_s * 2 ** min(failures, 5)))
            except Exception as exc:  # noqa: BLE001 - one bad page must not stop every agent's notices
                failures += 1
                log.warning("comms poll for %s failed: %s: %s", m.name, type(exc).__name__, exc)
                await asyncio.sleep(min(30.0, self.backoff_s * 2 ** min(failures, 5)))
            await asyncio.sleep(max(0, spacing - (monotonic() - started)))

    def _init_extended(self):
        self._channel_lock = asyncio.Lock()
        self._delivered: OrderedDict[tuple[str, str, int], bool] = OrderedDict()
