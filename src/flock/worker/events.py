"""Per-agent event stream: pi session format v3 on disk, `Event` records to the controller.

Encoding follows swarmlab's export.py: a header line
`{"type":"session","version":3,"id","timestamp","cwd","harness","name"}`, then entries chained
by `parentId` whose ids are the first 8 hex chars of sha256("<swarm>/<agent>/<n>") (widened
on collision). Harness messages are `message` entries; the worker's harness-neutral events
(session_started, session_resumed, session_restarted, turn_started, turn_ended,
turn_interrupted, sync, error, prompt_deferred) are pi `custom` entries with
`customType` = the event type. Entries carry our turn id as an extra `turnId` field.

An `Event` carries the whole pi entry in `data`; `Event.type` is the entry type for
messages and the `customType` for custom entries, so the two forms round-trip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from flock.models import Event

log = logging.getLogger(__name__)

NEUTRAL_EVENTS = (
    "session_started", "session_resumed", "session_restarted", "turn_started", "turn_ended",
    "turn_interrupted", "sync",
)


def utcnow() -> datetime:
    return datetime.now(UTC)


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s)  # 3.11+ parses a trailing Z


def entry_to_event(agent: str, entry: dict[str, Any]) -> Event:
    etype = entry["customType"] if entry.get("type") == "custom" else entry["type"]
    return Event(type=etype, ts=parse_iso(entry["timestamp"]), agent=agent,
                 turn_id=entry.get("turnId"), data=entry)


def event_to_entry(event: Event) -> dict[str, Any]:
    return event.data


class SessionLog:
    """Append-only pi v3 session file for one agent."""

    def __init__(self, path: Path, *, swarm: str, agent: str, harness: str, cwd: str) -> None:
        self.path = Path(path)
        self.swarm, self.agent, self.harness, self.cwd = swarm, agent, harness, cwd
        self.reload()

    def reload(self) -> None:
        """(Re)read ids from disk, e.g. after a restore replaced the file."""
        self._ids: set[str] = set()
        self._n = 0
        self._last: str | None = None
        if self.path.is_file():
            for line in self.path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "session" or "id" not in e:
                    continue
                self._ids.add(e["id"])
                self._last = e["id"]
                self._n += 1

    def _new_id(self) -> str:
        digest = hashlib.sha256(f"{self.swarm}/{self.agent}/{self._n}".encode()).hexdigest()
        for width in (8, 12, 16, 64):
            if digest[:width] not in self._ids:
                return digest[:width]
        raise AssertionError("unreachable")

    def append(self, entry_type: str, *, ts: datetime | None = None, turn_id: str | None = None,
               **fields: Any) -> Event:
        ts = ts or utcnow()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        if not self.path.exists() or self.path.stat().st_size == 0:
            header = {"type": "session", "version": 3, "id": f"{self.swarm}/{self.agent}",
                      "timestamp": iso(ts), "cwd": self.cwd, "harness": self.harness,
                      "name": self.agent}
            lines.append(json.dumps(header, ensure_ascii=False))
        eid = self._new_id()
        entry: dict[str, Any] = {"type": entry_type, "id": eid, "parentId": self._last,
                                 "timestamp": iso(ts), **fields}
        if turn_id is not None:
            entry["turnId"] = turn_id
        lines.append(json.dumps(entry, ensure_ascii=False, default=str))
        with self.path.open("a") as f:
            f.write("\n".join(lines) + "\n")
        self._ids.add(eid)
        self._last = eid
        self._n += 1
        return entry_to_event(self.agent, entry)

    def message(self, message: dict[str, Any], *, turn_id: str | None = None,
                ts: datetime | None = None) -> Event:
        ts = ts or utcnow()
        message = {**message}
        message.setdefault("timestamp", int(ts.timestamp() * 1000))
        return self.append("message", ts=ts, turn_id=turn_id, message=message)

    def custom(self, custom_type: str, data: dict[str, Any] | None = None, *,
               turn_id: str | None = None, ts: datetime | None = None) -> Event:
        return self.append("custom", ts=ts, turn_id=turn_id, customType=custom_type,
                           data=data or {})


def read_session(path: Path, agent: str) -> tuple[dict[str, Any], list[Event]]:
    """Header and events of a session file (for tests and tooling)."""
    header: dict[str, Any] = {}
    events = []
    for line in Path(path).read_text().splitlines():
        e = json.loads(line)
        if e.get("type") == "session":
            header = e
        else:
            events.append(entry_to_event(agent, e))
    return header, events


OUTBOX_MAX_EVENTS = 50_000
OUTBOX_MAX_BYTES = 256 * 1024 * 1024  # FLOCK_OUTBOX_MAX_BYTES


def event_size(ev: Event) -> int:
    """About what the event costs in the outbox: its entry's JSON length."""
    try:
        return len(json.dumps(ev.data, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return 1024


class Outbox:
    """Events waiting to be POSTed to the controller. Bounded by count and by size (a message
    entry can carry a large tool result, and a controller that falls behind would otherwise let
    50k of them pile up in the worker): drops the oldest when either bound is reached."""

    def __init__(self, maxlen: int = OUTBOX_MAX_EVENTS, max_bytes: int | None = None) -> None:
        self._q: deque[Event] = deque(maxlen=maxlen)
        self._sizes: deque[int] = deque()
        self.bytes = 0
        if max_bytes is None:
            max_bytes = int(os.environ.get("FLOCK_OUTBOX_MAX_BYTES") or OUTBOX_MAX_BYTES)
        self.max_bytes = max_bytes
        self.dropped = 0

    def _drop_oldest(self) -> None:
        self._q.popleft()
        self.bytes -= self._sizes.popleft()
        self.dropped += 1

    def add(self, ev: Event) -> None:
        size = event_size(ev)
        if len(self._q) == self._q.maxlen or (self._q and self.bytes + size > self.max_bytes):
            log.warning("event outbox full (%d events, %d bytes); dropping oldest event",
                        len(self._q), self.bytes)
        while self._q and (len(self._q) >= (self._q.maxlen or 0)
                           or self.bytes + size > self.max_bytes):
            self._drop_oldest()
        self._q.append(ev)
        self._sizes.append(size)
        self.bytes += size

    def take(self, n: int = 500) -> list[Event]:
        out = []
        for _ in range(min(n, len(self._q))):
            out.append(self._q.popleft())
            self.bytes -= self._sizes.popleft()
        return out

    def put_back(self, events: list[Event]) -> None:
        """Failed batch back at the front. Bounded too: what does not fit (newer events came
        in meanwhile) is dropped from the batch's oldest end."""
        for ev in reversed(events):
            size = event_size(ev)
            if len(self._q) >= (self._q.maxlen or 0) or (self._q and self.bytes + size > self.max_bytes):
                self.dropped += 1
                log.warning("event outbox full; dropping a re-queued event")
                continue
            self._q.appendleft(ev)
            self._sizes.appendleft(size)
            self.bytes += size

    def __len__(self) -> int:
        return len(self._q)
