"""The seam between the worker and a harness process.

An adapter owns at most one live harness process at a time (one burst). It is reusable:
`start` after `stop` begins a new burst on the same instance. Events come out of
`events()` until the process exits; the last event of a burst is always `exited`.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from legion.models import TokenUsage, TypeSpec

EventKind = Literal["turn_started", "message", "usage", "turn_ended", "error", "exited"]


@dataclass
class AdapterEvent:
    """One harness-neutral event.

    - turn_started: the harness began a turn (data may carry the harness's own turn id).
    - message: `message` is a pi-format AgentMessage (role user/assistant/toolResult).
    - usage: `usage` tokens for `model`, used since the last usage event.
    - turn_ended: `status` is completed | error | interrupted | ...
    - error: `error` text; the process may still be alive.
    - exited: the process is gone; no more events this burst.
    """

    kind: EventKind
    message: dict[str, Any] | None = None
    model: str | None = None
    usage: TokenUsage | None = None
    status: str | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


HTTP_TRANSIENT = re.compile(r"(?:^\s*|\bhttp\s+|\bstatus(?: code)?\s*[:=]?\s*)(?:429|502|503)\b",
                            re.IGNORECASE)


def transient_error(message: str) -> bool:
    """A model transport failure (HTTP 429/502/503, a timeout, a dropped connection): a turn
    that ends on one is `reason: model_transport`, which the worker-side loop retries with
    backoff instead of counting it as a failed turn."""
    return bool(HTTP_TRANSIENT.search(message)) or any(x in message.lower() for x in (
        "timeout", "timed out", "fetch failed", "econnreset", "connection reset",
        "connection refused", "socket", "abort"))


class ResumeError(RuntimeError):
    """The harness could not resume the requested session; start fresh instead."""


@runtime_checkable
class HarnessAdapter(Protocol):
    harness: str
    home: Path  # root of the harness's own state; session_files are synced relative to it

    async def start(
        self,
        workdir: Path,
        session_id: str | None,
        spec: TypeSpec,
        *,
        env: dict[str, str] | None = None,
    ) -> str:
        """Start a process. Resume `session_id` (raise ResumeError if impossible) or create a
        new session. Returns the session id in use. `env` is what the worker passes on
        (spec.env, declared secrets, the provider key under `api_key_env`)."""
        ...

    async def send(self, text: str) -> None: ...

    async def stop(self) -> None:
        """End the process (graceful, then killed). Idempotent."""
        ...

    def events(self) -> AsyncIterator[AdapterEvent]: ...

    def session_files(self, session_id: str | None) -> list[Path]:
        """Harness files that must be synced for `session_id` to resume elsewhere."""
        ...


class EventQueue:
    """A queue whose iterator ends after `exited`."""

    def __init__(self) -> None:
        self._q: asyncio.Queue[AdapterEvent] = asyncio.Queue()

    def put(self, ev: AdapterEvent) -> None:
        self._q.put_nowait(ev)

    async def __aiter__(self) -> AsyncIterator[AdapterEvent]:
        while True:
            ev = await self._q.get()
            yield ev
            if ev.kind == "exited":
                return


def pi_tokens(u: dict[str, Any] | None) -> TokenUsage:
    """A pi `usage` object as TokenUsage (`input` excludes cached tokens; `reasoning` is
    already inside `output`)."""
    u = u or {}
    return TokenUsage(input=int(u.get("input") or 0), output=int(u.get("output") or 0),
                      cache_read=int(u.get("cacheRead") or 0),
                      cache_write=int(u.get("cacheWrite") or 0),
                      reasoning=int(u.get("reasoning") or 0))
