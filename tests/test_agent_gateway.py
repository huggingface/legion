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
