"""Worker and controller under HTTP 429 from the Space edge: the
client retries with backoff honouring Retry-After, caps requests in flight, delivers events
once, and the controller does not declare a worker dead while any of its requests get in."""

from __future__ import annotations

import asyncio
from datetime import timedelta

import httpx
import pytest

from legion.controller.api import create_app
from legion.controller.reconcile import worker_state
from legion.models import (
    AgentHeartbeat,
    AgentState,
    AssignCommand,
    PromptCommand,
    WorkerHeartbeat,
    WorkerState,
)
from legion.worker import main as wmain
from legion.worker.events import utcnow
from legion.worker.main import RETRY_CAP, ControllerClient

from .conftest import TOKEN, make_spec
from .test_worker_main import WID, FakeController, make_worker
from .worker_fakes import spec, until


def _hb(wid: str = WID, agents=()) -> WorkerHeartbeat:
    return WorkerHeartbeat(worker=wid, deadline=utcnow() + timedelta(hours=1), agents=list(agents))


async def test_client_retries_429_and_5xx_honouring_retry_after():
    replies = [(429, {"retry-after": "2"}), (429, {"retry-after": "3"}), (503, {}), (200, {})]
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        status, headers = replies.pop(0)
        return httpx.Response(status, headers=headers, json={"ok": True})

    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    c = ControllerClient("http://c", "t", transport=httpx.MockTransport(handler), sleep=sleep)
    await c.heartbeat(_hb())
    assert len(seen) == 4 and not replies
    assert sleeps[0] >= 2 and sleeps[1] >= 3  # never sooner than the server asked
    assert 0 <= sleeps[2] <= RETRY_CAP  # 503 without Retry-After: jittered backoff
    assert c.stats == {"requests": 4, "throttled": 2, "retries": 3}
    await c.aclose()


async def test_client_gives_up_after_its_budget_and_backoff_grows():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    sleeps: list[float] = []

    async def sleep(s: float) -> None:
        sleeps.append(s)

    c = ControllerClient("http://c", "t", transport=httpx.MockTransport(handler), sleep=sleep)
    with pytest.raises(httpx.HTTPStatusError) as e:
        await c.heartbeat(_hb(), budget=60)
    assert e.value.response.status_code == 429
    assert sum(sleeps) <= 60 and len(sleeps) >= 3
    assert all(s <= RETRY_CAP for s in sleeps)
    await c.aclose()


def test_retry_after_http_date_and_jitter_bounds():
    later = (utcnow() + timedelta(seconds=10)).strftime("%a, %d %b %Y %H:%M:%S GMT")
    assert 8 <= wmain._retry_after(httpx.Response(429, headers={"retry-after": later})) <= 10
    assert wmain._retry_after(httpx.Response(429, headers={"retry-after": "junk"})) is None
    for attempt in range(10):
        d = wmain._retry_delay(attempt, None)
        assert 0 <= d <= min(RETRY_CAP, wmain.RETRY_BASE * 2 ** attempt)
    assert wmain._retry_delay(0, 120) == RETRY_CAP


async def test_client_caps_requests_in_flight():
    in_flight = peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.02)
        in_flight -= 1
        return httpx.Response(200, json={"ok": True})

    c = ControllerClient("http://c", "t", transport=httpx.MockTransport(handler))
    await asyncio.gather(*(c.heartbeat(_hb()) for _ in range(12)))
    assert peak == wmain.MAX_CONCURRENCY == 3
    await c.aclose()


async def test_worker_delivers_every_event_once_through_429_bursts(tmp_path, monkeypatch):
    monkeypatch.setattr(wmain, "_retry_delay", lambda attempt, retry_after: 0.01)
    ctrl = FakeController()
    ctrl.fail_next = [429] * 8  # a burst on whatever comes first: polls, heartbeats, events
    ctrl.batches = [
        [AssignCommand(agent="researcher-0001", type="researcher", spec=spec(), restore=False)],
        [PromptCommand(agent="researcher-0001", prompt_id="p1", text="one")],
    ]
    worker, _ = make_worker(tmp_path, ctrl)
    run = asyncio.create_task(worker.run())
    await until(lambda: any(e.type == "turn_ended" for e in ctrl.events), timeout=5)
    ctrl.fail_next = [429] * 8  # a second burst mid-run, then a second turn
    ctrl.batches.append([PromptCommand(agent="researcher-0001", prompt_id="p2", text="two")])
    await until(lambda: sum(e.type == "turn_ended" for e in ctrl.events) == 2, timeout=5)
    await worker.drain("test")
    await asyncio.wait_for(run, 5)
    ids = [e.data["id"] for e in ctrl.events]
    assert len(ids) == len(set(ids))  # once each
    assert [e.type for e in ctrl.events].count("turn_started") == 2
    assert worker.client.stats["throttled"] >= 16 and ctrl.heartbeats


# ----------------------------------------------------------------------- controller side


class EdgeThrottle(httpx.AsyncBaseTransport):
    """The Space edge in front of the app: 429 for heartbeats while `throttle` is set."""

    def __init__(self, app) -> None:
        self.inner = httpx.ASGITransport(app=app)
        self.throttle = False
        self.passed: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self.throttle and request.url.path.endswith("/heartbeat"):
            return httpx.Response(429, headers={"retry-after": "5"})
        self.passed.append(request.url.path.rsplit("/", 1)[-1])
        return await self.inner.handle_async_request(request)


async def test_controller_keeps_a_worker_whose_heartbeats_are_throttled(store, clock, tick):
    spec = make_spec(researcher={"replicas": 2, "agents_per_worker": 2})
    edge = EdgeThrottle(create_app(store, TOKEN, clock=clock))
    wid = "demo-researcher-w0000"
    async with httpx.AsyncClient(transport=edge, base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as admin:
        assert (await admin.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    tick()

    async def sleep(s: float) -> None:  # backoff waits pass on the controller's clock
        clock.advance(s)

    c = ControllerClient("http://t", TOKEN, transport=edge, sleep=sleep)
    agents = [AgentHeartbeat(name=f"researcher-000{i}", state="idle") for i in range(2)]
    await c.commands(wid, 0)
    await c.heartbeat(_hb(wid, agents))
    edge.throttle = True
    for _ in range(12):  # 6 min of throttled heartbeats; polls still get through
        clock.advance(25)
        await c.commands(wid, 0)
        with pytest.raises(httpx.HTTPStatusError):
            await c.heartbeat(_hb(wid, agents), budget=20)
        actions = tick()
        assert not [a for a in actions if a.kind in ("cancel", "crash", "launch")], actions
        w = store.get_worker(wid)
        assert worker_state(w, clock().timestamp(), store.liveness(w["swarm"])) \
            == WorkerState.running
    status = {a["name"]: a for a in (await _agents(edge))}
    assert {s["state"] for s in status.values()} == {AgentState.unknown.value}  # stale, not dead
    assert c.stats["throttled"] >= 12
    # total silence for the full grace period is still death
    clock.advance(181)
    assert any(a.kind == "cancel" for a in tick())
    await c.aclose()


async def _agents(edge: EdgeThrottle) -> list[dict]:
    async with httpx.AsyncClient(transport=edge, base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        return (await c.get("/swarms/demo/agents")).json()["agents"]
