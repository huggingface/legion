"""Board client 429 handling: Retry-After (seconds or HTTP-date) plus jitter, capped."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx
import pytest

from flock.comms import board as board_mod
from flock.comms.board import AsyncBoard, BoardError, _delay, parse_retry_after


def test_retry_after_seconds_and_http_date():
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    assert parse_retry_after("45") == 45.0
    assert parse_retry_after(format_datetime(now + timedelta(seconds=90), usegmt=True), now) == 90.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=5), usegmt=True), now) == 0.0
    assert parse_retry_after("soon") is None and parse_retry_after(None) is None


def test_delay_waits_retry_after_plus_jitter_up_to_the_cap(monkeypatch):
    r = httpx.Response(429, headers={"retry-after": "60"})
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: b)
    assert _delay(0, r, 1.0) == 60.0 + 12.0  # no longer capped at 20 s, 20% jitter
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: a)
    assert _delay(0, r, 1.0) == 60.0
    assert _delay(0, httpx.Response(429, headers={"retry-after": "600"}), 1.0) == 120.0
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: b)
    assert _delay(0, httpx.Response(429, headers={"retry-after": "0"}), 1.0) <= 1.0  # >= 1 s jitter
    assert _delay(0, httpx.Response(429, headers={"retry-after": "0"}), 1.0) > 0.0


async def test_async_board_raises_429_at_once_but_retries_5xx(monkeypatch):
    sleeps, seen = [], []

    async def no_sleep(s):
        sleeps.append(s)

    status = [429]

    def handler(r):
        seen.append(r)
        if status[0] == 503 and len(seen) > 2:
            return httpx.Response(200, json={"channels": []})
        return httpx.Response(status[0], headers={"retry-after": "30"},
                              json={"error": {"code": "rate_limited", "message": "slow down"}})

    monkeypatch.setattr(board_mod.asyncio, "sleep", no_sleep)
    board = AsyncBoard("http://b.invalid", transport=httpx.MockTransport(handler))
    cred = type("C", (), {"url": "http://b.invalid", "token": "t"})()
    try:
        with pytest.raises(BoardError) as e:
            await board.channels(cred)
        assert e.value.status == 429 and 30.0 <= e.value.retry_after <= 36.0  # + gate jitter
        assert len(seen) == 1 and sleeps == []  # one request, no in-client retry
        board.blocked_until = 0.0  # (the 429 closed the gate)
        status[0] = 503
        seen.clear()
        assert await board.channels(cred) == []
        assert len(seen) == 3 and len(sleeps) == 2  # 5xx still retried in the client
    finally:
        await board.aclose()


async def test_a_429_closes_the_gate_for_every_caller_without_sending(monkeypatch):
    seen = []

    def handler(r):
        seen.append(r.url.path)
        return httpx.Response(429, headers={"retry-after": "40"})

    board = AsyncBoard("http://b.invalid", transport=httpx.MockTransport(handler))
    cred = type("C", (), {"url": "http://b.invalid", "token": "t"})()
    try:
        with pytest.raises(BoardError):
            await board.post(cred, 12, "hello")
        assert 40.0 <= board.blocked_until - board_mod.time.monotonic() <= 49.0  # + jitter
        for call in (board.channels(cred), board.activity(cred, "joined", 5, wait=25),
                     board.search(cred, "x")):
            with pytest.raises(BoardError) as e:
                await call
            assert e.value.code == "board_busy" and e.value.retry_after >= 39
            assert str(e.value).startswith("board busy (rate limited), retry in ")
        assert seen == ["/v1/channels/12/messages"]  # nothing else was sent
    finally:
        await board.aclose()


async def test_worker_budget_tool_calls_wait_briefly_and_the_poller_yields(monkeypatch):
    now = [0.0]
    limiter = board_mod.RateLimiter(0.5, burst=2, clock=lambda: now[0])

    async def sleep(s):
        now[0] += s

    monkeypatch.setattr(board_mod.asyncio, "sleep", sleep)
    await limiter.acquire()
    await limiter.acquire()  # the burst
    await limiter.acquire()  # waits 2 s for the next token (0.5 req/s)
    assert now[0] == 2.0
    limiter.waiting = 1  # a tool call is waiting: the poller does not take the token
    now[0] += 2.0
    limiter._refill()
    assert limiter.tokens >= 1
    with pytest.raises(BoardError) as e:  # (bounded here only to end the test)
        await limiter.acquire(background=True, max_wait=1.0)
    limiter.waiting = 0
    await limiter.acquire(background=True, max_wait=None)
    with pytest.raises(BoardError) as e:  # a tool call never waits more than TOOL_WAIT_S
        await limiter.acquire(max_wait=1.0)
    assert e.value.code == "board_busy" and "worker request budget" in str(e.value)


async def test_tool_call_reports_board_busy_to_the_model(monkeypatch):
    from flock.models import ALL_COMMS_TOOLS, CommsSpec
    from flock.worker.comms import CommsHub

    def handler(r):
        if r.url.path == "/v1/join":
            return httpx.Response(200, json={"token": "tok", "principal": {"id": 1, "external_id": "dev-0000"}})
        if r.url.path == "/v1/channels":
            return httpx.Response(200, json={"channels": [{"id": 12, "name": "team", "member": True}]})
        if r.url.path == "/v1/activity":
            return httpx.Response(200, json={"messages": [], "next_cursor": 0})
        return httpx.Response(429, headers={"retry-after": "30"})

    hub = CommsHub(swarm="sw", env={"INV": "c"}, transport=httpx.MockTransport(handler), rate=0.5)
    hub.configure(CommsSpec(url="http://b.invalid", invite_env="INV", channel="team", channel_id=12),
                  list(ALL_COMMS_TOOLS))
    try:
        first = await hub.call("dev-0000", "comms:read", {})
        assert first.error
        second = await hub.call("dev-0000", "comms:read", {})
        assert second.error and "board busy (rate limited), retry in" in second.output
        assert hub.limiter.rate == 0.5
    finally:
        await hub.aclose()


async def test_429_log_lines_never_carry_tokens_cookies_or_queries(monkeypatch, caplog):
    import logging

    monkeypatch.setattr(board_mod, "_seen_429", 0)
    headers = {"retry-after": "7", "x-ratelimit-remaining": "0", "cf-ray": "abc", "server": "nginx",
               "set-cookie": "session=cookie-secret", "x-other": "other-secret"}
    board = AsyncBoard("http://b.invalid", transport=httpx.MockTransport(
        lambda r: httpx.Response(429, headers=headers)))
    cred = type("C", (), {"url": "http://b.invalid", "token": "bearer-secret"})()
    caplog.set_level(logging.WARNING, logger="flock.comms.board")
    try:
        for i in range(6):
            board.blocked_until = 0.0
            caplog.clear()
            with pytest.raises(BoardError):
                await board.activity(cred, "joined", 5, wait=0)
            lines = [r.getMessage() for r in caplog.records]
            assert lines[0] == "board 429 GET /v1/activity retry_after=7"
            text = "\n".join(lines)
            for secret in ("bearer-secret", "cookie-secret", "other-secret", "after=5", "joined"):
                assert secret not in text
            if i < 5:
                assert len(lines) == 2 and "x-ratelimit-remaining" in lines[1] and "cf-ray" in lines[1]
            else:
                assert len(lines) == 1  # the 6th: the short line only
    finally:
        await board.aclose()


def test_board_auth_uses_the_hf_token_on_hf_spaces(monkeypatch):
    from flock.comms.board import _auth

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    space = "https://huggingfaceswarms-board-demo.hf.space/boards/demo"
    assert _auth("fbt_b", space) == {"X-Msgboard-Token": "fbt_b", "Authorization": "Bearer hf_x"}
    assert _auth(None, space) == {"Authorization": "Bearer hf_x"}  # join on a private Space
    # elsewhere the HF token is never sent; old public boards still get the Bearer fallback
    assert _auth("fbt_b", "https://board.invalid") == {"X-Msgboard-Token": "fbt_b",
                                                        "Authorization": "Bearer fbt_b"}
    monkeypatch.delenv("HF_TOKEN")
    assert _auth("fbt_b", space)["Authorization"] == "Bearer fbt_b"


async def test_worker_join_sends_the_hf_token_only_to_hf_spaces(monkeypatch):
    from flock.worker.identity import join

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    seen = []

    def handler(req):
        seen.append((req.url.host, req.headers.get("authorization")))
        return httpx.Response(200, json={"token": "t", "principal": {"id": 3, "external_id": "a"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await join("https://x.hf.space/boards/b", "code", "a", client=c)
        await join("https://board.invalid/boards/b", "code", "a", client=c)
    assert seen == [("x.hf.space", "Bearer hf_x"), ("board.invalid", None)]
