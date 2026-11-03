"""Agent -> controller calls through the worker gateway (docs/spawn.md, "CLI inside an agent"),
and the worker following a controller that moved host."""

from __future__ import annotations

import httpx
import pytest

from flock.worker import main as wmain

from .test_worker_main import FakeController, make_worker


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(wmain, "_retry_delay", lambda attempt, retry_after: 0.01)


class Moving(FakeController):
    """A FakeController reachable only at `host`; any other host refuses the connection."""

    def __init__(self, host: str = "new-node") -> None:
        super().__init__()
        self.host, self.seen = host, []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request.url.host)
        if request.url.host != self.host:
            raise httpx.ConnectError("No route to host")
        return super().handler(request)


async def test_heartbeat_rereads_the_host_before_any_error(tmp_path):
    ctrl = Moving("old-node")
    worker, _ = make_worker(tmp_path, ctrl)
    where = {"host": "old-node"}
    worker.client.resolve_url = lambda: f"http://{where['host']}:7860"
    assert await worker.send_heartbeat()
    where["host"] = ctrl.host = "new-node"  # moved between two heartbeats
    assert await worker.send_heartbeat()
    assert ctrl.seen == ["old-node", "new-node"]  # no failed attempt at the old host
    assert len(ctrl.heartbeats) == 2
    await worker.client.aclose()


AGENT = "leader-0000"


class AgentRoutes(Moving):
    """Also answers an agent's spawn and status; records the token each one came with."""

    def __init__(self, host: str = "new-node") -> None:
        super().__init__(host)
        self.agent_calls: list[tuple[str, str, str, dict]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.startswith("/agents/"):
            return super().handler(request)
        self.seen.append(request.url.host)
        if request.url.host != self.host:
            raise httpx.ConnectError("No route to host")
        self.agent_calls.append((request.method, request.url.path,
                                 request.headers["authorization"], dict(request.url.params)))
        if request.url.path.endswith("/spawn"):
            return httpx.Response(201, json={"name": f"{AGENT}-analyst-0001", "created": True})
        if request.url.path == "/agents/someone-else":
            return httpx.Response(403, json={"detail": "agent leader-0000 may not call"})
        return httpx.Response(200, json={"name": AGENT, "state": "idle"})


async def gateway_worker(tmp_path, ctrl):
    from flock.models import AssignCommand

    from .worker_fakes import spec

    worker, _ = make_worker(tmp_path, ctrl)
    await worker.dispatch(AssignCommand(agent=AGENT, type="researcher", spec=spec(),
                                        restore=False, agent_token="flock_at_leader"))
    await worker.comms.start_gateway()
    return worker


async def call(worker, args, agent=AGENT):
    import json

    from flock.tools.toolset import gateway_call

    r = await gateway_call(worker.comms.address, worker.comms.key(agent), agent,
                           "controller:request", args)
    return r.error, json.loads(r.output)


async def test_gateway_forwards_with_the_agent_token_and_follows_a_move(tmp_path):
    ctrl = AgentRoutes("new-node")
    worker = await gateway_worker(tmp_path, ctrl)
    host = tmp_path / "controller.host"
    host.write_text("old-node\n")
    worker.client.http.base_url = "http://old-node:7860"
    worker.client.resolve_url = lambda: f"http://{host.read_text().strip()}:7860"
    host.write_text("new-node\n")  # the controller moved; this worker has not noticed yet

    err, out = await call(worker, {"method": "POST", "path": f"/agents/{AGENT}/spawn",
                                   "json": {"type": "analyst", "key": "a1"}})
    assert not err and out == {"status": 201, "body": {"name": f"{AGENT}-analyst-0001",
                                                       "created": True}}
    assert ctrl.seen == ["old-node", "new-node"]  # failed once, re-resolved, succeeded
    assert ctrl.agent_calls == [("POST", f"/agents/{AGENT}/spawn", "Bearer flock_at_leader", {})]

    err, out = await call(worker, {"method": "GET", "path": f"/agents/{AGENT}/events",
                                   "params": {"after": 3, "wait": 0}})
    assert not err and ctrl.agent_calls[-1][3] == {"after": "3", "wait": "0"}
    err, out = await call(worker, {"method": "GET", "path": "/agents/someone-else"})
    assert err and out["status"] == 403  # the controller's answer comes back as is
    await worker.client.aclose()


@pytest.mark.parametrize("method,path", [
    ("PUT", "/swarms/sw"), ("GET", "/swarms/sw"), ("GET", "/workers"),
    ("POST", "/workers/sw-researcher-w0000/heartbeat"), ("DELETE", "/swarms/sw"),
    ("GET", "/agents/x/../../swarms"), ("POST", "/agents/leader-0000/spawn?x=1"),
    ("GET", "/healthz"), ("POST", "/swarms/sw/prompts"),
])
async def test_gateway_refuses_other_routes(tmp_path, method, path):
    ctrl = AgentRoutes("controller")
    worker = await gateway_worker(tmp_path, ctrl)
    err, out = await call(worker, {"method": method, "path": path})
    assert err and "not allowed" in out["error"]
    assert ctrl.agent_calls == [] and "controller" not in ctrl.seen
    await worker.client.aclose()


async def test_wrong_key_is_refused(tmp_path):
    from flock.tools.toolset import gateway_call

    worker = await gateway_worker(tmp_path, AgentRoutes("controller"))
    r = await gateway_call(worker.comms.address, "guess", AGENT, "controller:request",
                           {"method": "GET", "path": "/whoami"})
    assert r.error and "unknown agent or key" in r.output
    await worker.client.aclose()


async def test_persistent_failure_emits_controller_call_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(wmain, "_retry_delay", lambda attempt, retry_after: 20.0)
    ctrl = AgentRoutes("nowhere")  # never reachable, wherever the host file points
    worker = await gateway_worker(tmp_path, ctrl)
    worker.client._sleep = lambda s: __import__("asyncio").sleep(0)
    err, out = await call(worker, {"method": "POST", "path": f"/agents/{AGENT}/prompt",
                                   "json": {"text": "hi"}})
    assert err and "controller unreachable" in out["error"]
    events = [e for e in worker.outbox.take() if e.type == "controller_call_failed"]
    assert len(events) == 1 and events[0].agent == AGENT
    assert events[0].data["data"]["path"] == f"/agents/{AGENT}/prompt"
    assert "ConnectError" in events[0].data["data"]["error"]
    await worker.client.aclose()


async def test_cli_call_goes_through_the_gateway_when_the_pin_names_one(tmp_path, monkeypatch):
    import asyncio

    from flock.client import Client, gateway

    ctrl = AgentRoutes("controller")
    worker = await gateway_worker(tmp_path, ctrl)
    pin = worker.agents[AGENT].pin_path
    worker.agents[AGENT]._pin_tools()  # what the turn start writes
    monkeypatch.setenv("FLOCK_TOOLS_PIN", str(pin))
    monkeypatch.setenv("FLOCK_URL", "http://stale-host:7860")  # frozen at harness start
    monkeypatch.delenv("FLOCK_AGENT_TOKEN", raising=False)
    monkeypatch.setattr("flock.client.connect", lambda *_: pytest.fail("direct call"))
    ack = await asyncio.to_thread(Client()._call, "POST", f"/agents/{AGENT}/spawn",
                                  json={"type": "analyst"})
    assert ack["name"] == f"{AGENT}-analyst-0001"
    assert Client().as_agent  # no FLOCK_AGENT_TOKEN needed any more

    monkeypatch.delenv("FLOCK_TOOLS_PIN")  # no gateway: the direct path, as before
    assert gateway() is None
    await worker.client.aclose()


async def test_forwarded_post_is_not_resent_once_it_may_have_arrived(tmp_path):
    """A read timeout or a 500 may come after the controller acted: a POST is not sent again
    (a prompt would land twice); no connection or a 503 is retried; a GET is retried on both."""
    ctrl = AgentRoutes("controller")
    worker = await gateway_worker(tmp_path, ctrl)
    script: list = []
    base = ctrl.handler

    def handler(request):
        if request.url.path.startswith("/agents/") and script:
            step = script.pop(0)
            ctrl.agent_calls.append((request.method, request.url.path, "", {}))
            if isinstance(step, Exception):
                raise step
            return httpx.Response(step, json={"detail": "x"})
        return base(request)

    ctrl.handler = handler
    worker.client.http._transport = httpx.MockTransport(handler)
    prompt = {"method": "POST", "path": f"/agents/{AGENT}/prompt", "json": {"text": "hi"}}
    for first in (httpx.ReadTimeout("slow"), 500):
        script[:] = [first]
        ctrl.agent_calls.clear()
        err, _ = await call(worker, prompt)
        assert err and len(ctrl.agent_calls) == 1, first  # sent once, never resent
    for first in (httpx.ConnectError("refused"), 503):
        script[:] = [first]
        ctrl.agent_calls.clear()
        err, out = await call(worker, prompt)
        assert not err and out["status"] == 200 and len(ctrl.agent_calls) == 2, first
    script[:] = [httpx.ReadTimeout("slow")]
    ctrl.agent_calls.clear()
    err, _ = await call(worker, {"method": "GET", "path": f"/agents/{AGENT}"})
    assert not err and len(ctrl.agent_calls) == 2
    await worker.client.aclose()
