"""The seam between the worker and a harness process.

An adapter owns at most one live harness process at a time (one burst). It is reusable:
`start` after `stop` begins a new burst on the same instance. Events come out of
`events()` until the process exits; the last event of a burst is always `exited`.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from flock.models import TokenUsage, TypeSpec

EventKind = Literal["turn_started", "message", "usage", "turn_ended", "error", "exited"]


@dataclass
class AdapterEvent:
    """One harness-neutral event.

    - turn_started: the harness began a turn (data may carry the harness's own turn id).
    - message: `message` is a pi-format AgentMessage (role user/assistant/toolResult).
    - usage: `usage` tokens for `model`. `cumulative` means a running total for the whole
      session (the runtime subtracts the previous snapshot); `baseline` means a snapshot that
      predates this burst and must only reset the reference point, never be counted.
      `cost_usd` is the harness's own cumulative estimate, if any.
    - turn_ended: `status` is completed | error | interrupted | ...
    - error: `error` text; the process may still be alive.
    - exited: the process is gone; no more events this burst.
    """

    kind: EventKind
    message: dict[str, Any] | None = None
    model: str | None = None
    usage: TokenUsage | None = None
    cumulative: bool = False
    baseline: bool = False
    cost_usd: float | None = None
    status: str | None = None
    error: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    raw: Any = None


# What a harness process inherits from the worker's own environment. Everything else (FLOCK_TOKEN,
# job secrets, the incarnation) stays in the worker; the harness gets only these plus what the
# worker passes explicitly (spec.env, the type's declared secrets, identity, the provider key).
BASE_ENV = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "SHELL", "TERM", "LANG", "LANGUAGE", "TZ", "TMPDIR",
    "IS_SANDBOX",  # Claude's bypassPermissions as root
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "NODE_EXTRA_CA_CERTS",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy",
})
BASE_ENV_PREFIXES = ("LC_", "XDG_")


def minimal_env(
    parent: Mapping[str, str], explicit: Mapping[str, str], keep: Iterable[str] = ()
) -> dict[str, str]:
    """The harness environment: allowlisted names from `parent` (BASE_ENV, LC_*, XDG_*, plus
    `keep`, e.g. the harness's config-dir variable), then `explicit` on top."""
    names = BASE_ENV | set(keep)
    env = {k: v for k, v in parent.items() if k in names or k.startswith(BASE_ENV_PREFIXES)}
    env.update(explicit)
    return env


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
        new session. Returns the session id in use. `env` is everything the worker passes on
        (spec.env, declared secrets, identity, the provider key under `api_key_env`); the
        process gets that plus `minimal_env`'s allowlist, never the worker's whole environment."""
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
    """Small helper both adapters use: a queue whose iterator ends after `exited`."""

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


def usage_sub(a: TokenUsage, b: TokenUsage | None) -> TokenUsage:
    """a - b, clamped at zero per field (cumulative counters never go backwards)."""
    if b is None:
        return a
    return TokenUsage(**{k: max(0, getattr(a, k) - getattr(b, k)) for k in TokenUsage.model_fields})


def pi_usage(u: TokenUsage) -> dict[str, Any]:
    """TokenUsage as a pi `usage` object (cost left at zero; the controller prices tokens)."""
    d: dict[str, Any] = {
        "input": u.input,
        "output": u.output,
        "cacheRead": u.cache_read,
        "cacheWrite": u.cache_write,
        "totalTokens": u.input + u.output + u.cache_read + u.cache_write,
        "cost": {"input": 0.0, "output": 0.0, "cacheRead": 0.0, "cacheWrite": 0.0, "total": 0.0},
    }
    if u.reasoning:
        d["reasoning"] = u.reasoning
    return d
