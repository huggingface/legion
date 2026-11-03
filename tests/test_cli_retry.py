"""The `flock` CLI retries 429 / 5xx with backoff honouring Retry-After (bounded), so 1,000
agents calling it through the Space edge do not fail on throttling."""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli
from flock import client as flock_client

from .conftest import TOKEN

LEADER = "leader-0000"
runner = CliRunner()


def _edge(replies):
    """A fake controller behind the edge: `replies` statuses first, then 200."""
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.method)
        if replies:
            status, headers = replies.pop(0)
            return httpx.Response(status, headers=headers, json={"detail": "slow down"})
        return httpx.Response(200, json={"name": "leader-0000-analyst-0001", "created": True})

    return seen, handler


def _edge_client(monkeypatch, handler):
    monkeypatch.setattr("flock.client.connect", lambda *_: httpx.Client(
        base_url="http://c", transport=httpx.MockTransport(handler)))


@pytest.fixture
def sleeps(monkeypatch):
    out: list[float] = []
    monkeypatch.setattr("flock.client._sleep", out.append)
    return out


def test_cli_retries_429_then_succeeds(monkeypatch, sleeps):
    seen, handler = _edge([(429, {"retry-after": "3"}), (429, {}), (503, {})])
    _edge_client(monkeypatch, handler)
    r = runner.invoke(cli.agent_app, ["spawn", "analyst", "--parent", LEADER])
    assert r.exit_code == 0, r.output
    assert r.output.strip() == "leader-0000-analyst-0001"
    assert seen == ["POST"] * 4
    assert sleeps[0] >= 3 and len(sleeps) == 3
    assert all(0 <= s <= flock_client.RETRY_CAP for s in sleeps)


def test_cli_gives_up_after_five_attempts(monkeypatch, sleeps):
    seen, handler = _edge([(429, {})] * 10)
    _edge_client(monkeypatch, handler)
    r = runner.invoke(cli.app, ["status", "--agent", "leader-0000"])
    assert r.exit_code == 1 and "error 429" in r.output
    assert len(seen) == flock_client.RETRY_ATTEMPTS == 5 and len(sleeps) == 4


def test_cli_does_not_retry_a_post_500(monkeypatch, sleeps):
    seen, handler = _edge([(500, {})])
    _edge_client(monkeypatch, handler)
    r = runner.invoke(cli.app, ["prompt", "leader-0000", "x"])
    assert r.exit_code == 1 and seen == ["POST"] and not sleeps  # may have been processed


def test_cli_against_the_app_still_works(monkeypatch, app, sleeps):
    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0 and not sleeps
