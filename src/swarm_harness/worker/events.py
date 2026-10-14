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
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from swarm_harness.models import Event

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


class Outbox:
    """Events waiting to be POSTed to the controller. Bounded: drops the oldest when full."""

    def __init__(self, maxlen: int = 50_000) -> None:
        self._q: deque[Event] = deque(maxlen=maxlen)

    def add(self, ev: Event) -> None:
        if len(self._q) == self._q.maxlen:
            log.warning("event outbox full; dropping oldest event")
        self._q.append(ev)

    def take(self, n: int = 500) -> list[Event]:
        return [self._q.popleft() for _ in range(min(n, len(self._q)))]

    def put_back(self, events: list[Event]) -> None:
        for ev in reversed(events):
            self._q.appendleft(ev)

    def __len__(self) -> int:
        return len(self._q)
