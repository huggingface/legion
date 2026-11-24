"""legion.client.Client against the in-process controller (the CLI's only way to it)."""

from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest

from legion.client import Client, ClientError
from legion.models import Event

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


def test_answers_only_a_legion_controller(api, monkeypatch):
    assert api.answers()  # the in-process controller's /healthz
    for reply in ({"text": "<html>a Gradio app</html>"}, {"json": {"status": "ready"}},
                  {"json": ["ok", "process"]}, {"status_code": 404, "json": {"ok": 1, "process": 1}}):
        def other_app(*_, kw=reply):  # another app on the port
            return httpx.Client(base_url="http://localhost:7860", transport=httpx.MockTransport(
                lambda req: httpx.Response(**{"status_code": 200, **kw})))
        monkeypatch.setattr("legion.client.connect", other_app)
        assert not api.answers()


def test_a_run_without_a_controller_host_never_falls_back_to_localhost(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .legion/current here either
    run = tmp_path / "run"
    (run / "hosts").mkdir(parents=True)
    calls = []

    def refused(env):
        calls.append(env.get("LEGION_URL"))
        raise httpx.ConnectError("refused")

    monkeypatch.setattr("legion.client.connect", refused)
    assert Client.for_run(run, need_host=False, env={"LEGION_URL": "http://x:1"}).answers() is False
    assert calls == []  # no request at all: `launch` goes to `up`
    with pytest.raises(ClientError, match="launch first"):
        Client.for_run(run, env={})
    (run / "hosts" / "controller.host").write_text("node3:7999\n")
    assert Client.for_run(run, env={}).answers() is False
    assert calls == ["http://node3:7999"]  # the host file's controller is asked
