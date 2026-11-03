"""flock.client.Client against the in-process controller (the CLI's only way to it)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from flock.client import ClientError
from flock.models import Event

from .conftest import make_spec


def test_events_follow_stops_at_the_timeout(api, store):
    api.create(make_spec())
    store.append_events("demo", "researcher-0001", [
        Event(type=k, ts=datetime.now(UTC), agent="researcher-0001", data={})
        for k in ("turn_started", "turn_ended")])
    pages = list(api.events("researcher-0001", swarm=False, follow=True, timeout=0))
    assert len(pages) == 1 and [e["type"] for e in pages[0]["events"]] == [
        "turn_started", "turn_ended"]
    (page,) = api.events("researcher-0001", swarm=False, after=pages[0]["next"])  # no follow: one page
    assert page["events"] == []


def test_404_raises_client_error(api):
    with pytest.raises(ClientError) as e:
        api.agent("nobody-0001")
    assert e.value.status == 404 and str(e.value).startswith("error 404: ")
    assert e.value.detail != e.value.body  # the controller's `detail`, unwrapped
