"""Board client 429 handling: Retry-After (seconds or HTTP-date) plus jitter, capped."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from email.utils import format_datetime

import httpx

from legion.comms import board as board_mod
from legion.comms.board import _delay, parse_retry_after


def test_retry_after_seconds_and_http_date():
    now = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
    assert parse_retry_after("45") == 45.0
    assert parse_retry_after(format_datetime(now + timedelta(seconds=90), usegmt=True), now) == 90.0
    assert parse_retry_after(format_datetime(now - timedelta(seconds=5), usegmt=True), now) == 0.0
    assert parse_retry_after("soon") is None and parse_retry_after(None) is None


def test_delay_waits_retry_after_plus_jitter_up_to_the_cap(monkeypatch):
    r = httpx.Response(429, headers={"retry-after": "60"})
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: b)
    assert _delay(0, r, 1.0) == 60.0 + 12.0  # Retry-After is honoured in full, 20% jitter
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: a)
    assert _delay(0, r, 1.0) == 60.0
    assert _delay(0, httpx.Response(429, headers={"retry-after": "600"}), 1.0) == 120.0
    monkeypatch.setattr(board_mod.random, "uniform", lambda a, b: b)
    assert _delay(0, httpx.Response(429, headers={"retry-after": "0"}), 1.0) <= 1.0  # >= 1 s jitter
    assert _delay(0, httpx.Response(429, headers={"retry-after": "0"}), 1.0) > 0.0


