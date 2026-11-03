"""flock.client.Client against the in-process controller (the CLI's only way to it)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from flock.client import Client, ClientError
from flock.models import Event

from .conftest import TOKEN, make_spec


@pytest.fixture
def api(monkeypatch, app) -> Client:
    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    return Client()


def test_create_prompt_stop_resume(api, store, tick):
    spec = make_spec()
    assert api.create(spec)["swarm"] == "demo"
    with pytest.raises(ClientError) as e:  # If-None-Match: never over an existing swarm
        api.create(spec)
    assert e.value.status == 412
    tick()
    assert api.swarm("demo")["spec"]["swarm"] == "demo" and api.is_swarm("demo")
    assert [a["name"] for a in api.agents("demo")] == [f"researcher-000{i}" for i in range(3)]
    assert api.prompt("researcher-0001", "hello")["status"] == "pending"
    assert api.prompt_type("demo", "researcher", "all of you")["matched"] == 3

    held = api.stop("demo", types=["researcher"], reason="night")["held"]
    assert list(held["types"]) == ["researcher"]
    assert api.resume("demo")["held"] == {"types": {}, "agents": {}}


def test_events_follow_stops_at_the_timeout(api, store):
    api.create(make_spec())
    store.append_events("demo", "researcher-0001", [
        Event(type=k, ts=datetime.now(UTC), agent="researcher-0001", data={})
        for k in ("turn_started", "turn_ended")])
    pages = list(api.events("researcher-0001", follow=True, timeout=0))
    assert len(pages) == 1 and [e["type"] for e in pages[0]["events"]] == [
        "turn_started", "turn_ended"]
    (page,) = api.events("researcher-0001", after=pages[0]["next"])  # no follow: one page
    assert page["events"] == []


def test_404_raises_client_error(api):
    with pytest.raises(ClientError) as e:
        api.agent("nobody-0001")
    assert e.value.status == 404 and str(e.value).startswith("error 404: ")
    assert e.value.detail != e.value.body  # the controller's `detail`, unwrapped
