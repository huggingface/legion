"""msgboard API client (the routes comms uses; the board's guide is `GET /v1/guide`).

Writes carry an `Idempotency-Key` that stays the same across retries, so a retried post never
posts twice. 429, 5xx and transport errors are retried with jittered backoff, honouring
`Retry-After` (its value plus jitter, up to RETRY_AFTER_MAX_S), except that the worker's
AsyncBoard raises a 429 at once (BoardError.retry_after) instead of retrying; other errors raise `BoardError` with the board's own message. Tokens and invite
codes are never part of an error message.
"""

from __future__ import annotations

import asyncio
import math
import random
import time
import uuid
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from flock.models import normalize_board_url

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
ATTEMPTS = 4
BACKOFF_S = 1.0
BACKOFF_MAX_S = 20.0
RETRY_AFTER_MAX_S = 120.0  # a board's Retry-After is honoured up to this (plus jitter)
GATE_DEFAULT_S = 5.0  # a 429 without Retry-After closes the gate this long (plus jitter)
TOOL_WAIT_S = 10.0  # a foreground request waits at most this for the worker's request budget
NEWEST = 2 ** 53  # `before=` that means "the newest messages"


class BoardError(RuntimeError):
    """`retry_after`: seconds from the response's Retry-After header, when it had one."""

    def __init__(self, message: str, status: int | None = None, code: str | None = None,
                 retry_after: float | None = None) -> None:
        super().__init__(message)
        self.status, self.code, self.retry_after = status, code, retry_after


def parse_retry_after(value: str | None, now: datetime | None = None) -> float | None:
    """Retry-After as seconds (>= 0): delta-seconds or an HTTP-date; None if absent or invalid."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - (now or datetime.now(UTC))).total_seconds())


def _busy(what: str, wait: float) -> BoardError:
    return BoardError(f"board busy ({what}), retry in {max(1, math.ceil(wait))} s", 429,
                      "board_busy", wait)


class RateLimiter:
    """A worker's request budget for the board: a token bucket (`rate` requests/s, `burst`),
    shared by the poller and every tool call. Background requests (the poller) yield: they
    take a token only while no foreground request is waiting."""

    def __init__(self, rate: float, burst: float = 5.0, clock=time.monotonic) -> None:
        self.rate, self.burst, self._clock = rate, max(1.0, burst), clock
        self.tokens, self._at, self.waiting = self.burst, clock(), 0

    def _refill(self) -> None:
        now = self._clock()
        self.tokens = min(self.burst, self.tokens + (now - self._at) * self.rate)
        self._at = now

    async def acquire(self, *, background: bool = False, max_wait: float | None = TOOL_WAIT_S) -> None:
        start = self._clock()
        if not background:
            self.waiting += 1
        try:
            while True:
                self._refill()
                if self.tokens >= 1 and (not background or self.waiting == 0):
                    self.tokens -= 1
                    return
                wait = max(0.05, (1 - self.tokens) / self.rate)
                if max_wait is not None and self._clock() - start + wait > max_wait:
                    raise _busy("worker request budget", wait)
                await asyncio.sleep(wait)
        finally:
            if not background:
                self.waiting -= 1


def idem_key(prefix: str = "flock") -> str:
    return f"{prefix}-{uuid.uuid4().hex}"[:128]


def _error(r: httpx.Response, method: str, path: str) -> BoardError:
    try:
        err = (r.json() or {}).get("error") or {}
    except ValueError:
        err = {}
    msg = err.get("message") if isinstance(err, dict) else None
    code = err.get("code") if isinstance(err, dict) else None
    return BoardError(f"board: {method} {path.split('?')[0]}: HTTP {r.status_code}"
                      + (f" {code}: {msg}" if msg else ""), r.status_code, code,
                      parse_retry_after(r.headers.get("retry-after")))


def _delay(attempt: int, r: httpx.Response | None, base: float) -> float:
    """Jittered exponential backoff; with a Retry-After, that wait plus up to 20% (at least
    1 s) of jitter, so clients told the same time do not come back together. Capped at
    RETRY_AFTER_MAX_S."""
    d = min(BACKOFF_MAX_S, base * 2 ** attempt) * (0.5 + random.random() / 2)
    ra = parse_retry_after(r.headers.get("retry-after")) if r is not None else None
    if ra is None:
        return d
    return min(RETRY_AFTER_MAX_S, max(d, ra + random.uniform(0, max(1.0, 0.2 * ra))))


def _query(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


class Credential(Protocol):
    """A board token together with the board it belongs to (MsgboardIdentity)."""

    url: str
    token: str


def same_board(a: str, b: str) -> bool:
    """Same scheme, host and port (and path prefix): where a token may be sent."""
    return normalize_board_url(a) == normalize_board_url(b)


class _Routes:
    """Request shapes shared by both clients: (method, path, kwargs)."""

    @staticmethod
    def join(code: str, name: str) -> tuple[str, str, dict]:
        return "POST", "/v1/join", {"json": {"code": code, "name": name, "display_name": name}}

    @staticmethod
    def channels(name: str | None = None) -> tuple[str, str, dict]:
        return "GET", "/v1/channels", {"params": _query({"name": name})}

    @staticmethod
    def add_member(channel: int, principal: int) -> tuple[str, str, dict]:
        return "POST", f"/v1/channels/{channel}/members", {"json": {"principal_id": principal}}

    @staticmethod
    def runtime(principal: int, runtime: dict) -> tuple[str, str, dict]:
        return "PATCH", f"/v1/principals/{principal}", {"json": {"runtime": runtime}}

    @staticmethod
    def post(channel: int, body: str, reply_to: int | None) -> tuple[str, str, dict]:
        return "POST", f"/v1/channels/{channel}/messages", {"json": _query({"body": body,
                                                                             "reply_to": reply_to})}

    @staticmethod
    def message(msg_id: int) -> tuple[str, str, dict]:
        return "GET", f"/v1/messages/{msg_id}", {}

    @staticmethod
    def latest(channel: int, n: int) -> tuple[str, str, dict]:
        return "GET", f"/v1/channels/{channel}/messages", {"params": {"before": NEWEST, "limit": n}}

    @staticmethod
    def thread(channel: int, root: int, n: int) -> tuple[str, str, dict]:
        """The newest `n` of a thread (`before=` reads from the tail; the page comes back
        oldest first with `has_more` when older messages exist)."""
        return "GET", f"/v1/channels/{channel}/messages", {"params": {"thread": root, "before": NEWEST,
                                                                       "limit": n}}

    @staticmethod
    def activity(channel: int | str, after: int | None, wait: float) -> tuple[str, str, dict]:
        params = _query({"after": after, "wait": wait if after is not None and wait else None,
                         "channels": channel, "limit": 200})
        return "GET", "/v1/activity", {"params": params}

    @staticmethod
    def search(q: str, sort: str, limit: int) -> tuple[str, str, dict]:
        return "GET", "/v1/search", {"params": {"q": q, "sort": sort, "limit": limit}}

    @staticmethod
    def react(channel: int, msg_id: int, emoji: str) -> tuple[str, str, dict]:
        return "PUT", f"/v1/channels/{channel}/messages/{msg_id}/reactions/{quote(emoji)}", {}


class AsyncBoard:
    """One board, async; a credential per call (a worker speaks for many agents). A
    credential carries the url of the board that issued it, and `request` refuses to send it
    anywhere else."""

    def __init__(self, url: str, *, transport: httpx.AsyncBaseTransport | None = None,
                 backoff_s: float = BACKOFF_S, limiter: RateLimiter | None = None) -> None:
        self.url = normalize_board_url(url)
        self.http = httpx.AsyncClient(base_url=self.url, transport=transport,
                                      timeout=httpx.Timeout(20, read=45))
        self.backoff_s = backoff_s
        self.limiter = limiter
        # Shared gate: a 429 closes it until Retry-After (plus jitter) for every caller of
        # this board; meanwhile requests fail at once, without being sent.
        self.blocked_until = 0.0

    async def aclose(self) -> None:
        await self.http.aclose()

    def _check_gate(self) -> None:
        left = self.blocked_until - time.monotonic()
        if left > 0:
            raise _busy("rate limited", left)

    def _close_gate(self, error: BoardError) -> None:
        ra = error.retry_after if error.retry_after is not None else GATE_DEFAULT_S
        wait = min(RETRY_AFTER_MAX_S, ra + random.uniform(0, max(1.0, 0.2 * ra)))
        self.blocked_until = max(self.blocked_until, time.monotonic() + wait)
        error.retry_after = max(ra, wait)

    async def request(self, route: tuple[str, str, dict], cred: Credential | None = None, *,
                      background: bool = False) -> dict:
        """`background`: the poller's request, which yields to tool calls for the budget
        and waits for it as long as needed; a tool call waits at most TOOL_WAIT_S."""
        method, path, kw = route
        self._check_gate()
        headers = {}
        if cred is not None:
            # a token is only ever sent to the board that issued it
            if not same_board(cred.url, self.url):
                raise BoardError(f"board: refusing to send a token issued by {cred.url} to "
                                 f"{self.url}", code="wrong_board")
            headers["Authorization"] = f"Bearer {cred.token}"
        if method in ("POST", "PUT", "PATCH"):
            headers["Idempotency-Key"] = idem_key()
        for attempt in range(ATTEMPTS):
            if self.limiter is not None:
                await self.limiter.acquire(background=background,
                                           max_wait=None if background else TOOL_WAIT_S)
                self._check_gate()
            r: httpx.Response | None = None
            try:
                r = await self.http.request(method, path, headers=headers, **kw)
            except httpx.TransportError as e:
                if attempt == ATTEMPTS - 1:
                    raise BoardError(f"board: {method} {path}: {type(e).__name__}") from None
            if r is not None and r.status_code == 429:
                # Rate limited: never retried here (a retry is another request against the
                # same limit); the gate closes for every caller and the error carries
                # retry_after.
                error = _error(r, method, path)
                self._close_gate(error)
                raise error
            if r is not None and r.status_code not in RETRY_STATUSES:
                if r.status_code >= 400:
                    raise _error(r, method, path)
                return r.json() if r.content else {}
            if attempt == ATTEMPTS - 1 and r is not None:
                raise _error(r, method, path)
            await asyncio.sleep(_delay(attempt, r, self.backoff_s))
        raise BoardError(f"board: {method} {path}: gave up")

    async def join(self, code: str, name: str) -> dict:
        return await self.request(_Routes.join(code, name))

    async def channels(self, cred: Credential, name: str | None = None) -> list[dict]:
        return (await self.request(_Routes.channels(name), cred)).get("channels") or []

    async def add_member(self, cred: Credential, channel: int, principal: int) -> None:
        await self.request(_Routes.add_member(channel, principal), cred)

    async def set_runtime(self, cred: Credential, principal: int, runtime: dict) -> None:
        await self.request(_Routes.runtime(principal, runtime), cred)

    async def post(self, cred: Credential, channel: int, body: str, reply_to: int | None = None) -> dict:
        return await self.request(_Routes.post(channel, body, reply_to), cred)

    async def message(self, cred: Credential, msg_id: int) -> dict:
        return await self.request(_Routes.message(msg_id), cred)

    async def latest(self, cred: Credential, channel: int, n: int) -> dict:
        """{messages (the newest n, oldest first), has_more, next_cursor}."""
        return await self.request(_Routes.latest(channel, n), cred)

    async def thread(self, cred: Credential, channel: int, root: int, n: int = 100) -> dict:
        """{messages (the thread's newest n, oldest first), has_more, next_cursor}."""
        return await self.request(_Routes.thread(channel, root, n), cred)

    async def activity(self, cred: Credential, channel: int | str, after: int | None, wait: float = 0) -> dict:
        return await self.request(_Routes.activity(channel, after, wait), cred,
                                  background=after is not None and bool(wait))

    async def search(self, cred: Credential, q: str, *, sort: str = "relevance", limit: int = 20) -> dict:
        return await self.request(_Routes.search(q, sort, limit), cred)

    async def react(self, cred: Credential, channel: int, msg_id: int, emoji: str) -> dict:
        return await self.request(_Routes.react(channel, msg_id, emoji), cred)


class SyncBoard:
    """The controller's client (apply runs in a thread): join, channels, the kickoff post."""

    def __init__(self, url: str, *, transport: httpx.BaseTransport | None = None,
                 backoff_s: float = BACKOFF_S) -> None:
        self.http = httpx.Client(base_url=normalize_board_url(url), transport=transport, timeout=20)
        self.backoff_s = backoff_s

    def close(self) -> None:
        self.http.close()

    def request(self, route: tuple[str, str, dict], token: str | None = None, *,
                key: str | None = None) -> dict:
        method, path, kw = route
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        if method in ("POST", "PUT", "PATCH"):
            headers["Idempotency-Key"] = key or idem_key()
        for attempt in range(ATTEMPTS):
            r: httpx.Response | None = None
            try:
                r = self.http.request(method, path, headers=headers, **kw)
            except httpx.TransportError as e:
                if attempt == ATTEMPTS - 1:
                    raise BoardError(f"board: {method} {path}: {type(e).__name__}") from None
            if r is not None and r.status_code not in RETRY_STATUSES:
                if r.status_code >= 400:
                    raise _error(r, method, path)
                return r.json() if r.content else {}
            if attempt == ATTEMPTS - 1 and r is not None:
                raise _error(r, method, path)
            time.sleep(_delay(attempt, r, self.backoff_s))
        raise BoardError(f"board: {method} {path}: gave up")

    def join(self, code: str, name: str) -> dict:
        return self.request(_Routes.join(code, name))

    def channels(self, token: str, name: str | None = None) -> list[dict]:
        return self.request(_Routes.channels(name), token).get("channels") or []

    def add_member(self, token: str, channel: int, principal: int) -> None:
        self.request(_Routes.add_member(channel, principal), token)

    def post(self, token: str, channel: int, body: str, *, key: str,
             reply_to: int | None = None) -> dict:
        return self.request(_Routes.post(channel, body, reply_to), token, key=key)
