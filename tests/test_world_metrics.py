"""World server: structured retry hints, /metrics and admission control (docs/world.md)."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import ClassVar

import httpx
from fastapi.testclient import TestClient

from flock.world import ToolError, World, server, tool
from flock.world.server import WINDOW_S, _Metrics, create_app

from .test_world import SECRET, Served

H = {"X-World-Token": SECRET, "X-Flock-Agent": "dev-0001"}


class Hinted(World):
    name, version = "hinted", "1"

    @tool
    def later(self) -> str:
        """Refuses with a retry hint."""
        raise ToolError("budget spent; next slot in ~2 min", "open_budget", retry_after_s=120)


def test_tool_error_retry_after_is_in_the_error_envelope():
    c = TestClient(create_app(Hinted(), SECRET))
    assert c.post("/tools/later", json={"args": {}}, headers=H).json() == {"ok": False, "error": {
        "code": "open_budget", "message": "budget spent; next slot in ~2 min", "retry_after_s": 120}}


class Gauged(Hinted):
    name = "gauged"

    @tool(read_only=True)
    def echo(self, text: str) -> str:
        """Echo."""
        return text

    def metrics(self) -> dict[str, float]:
        return {"open_prs": 3, "note": "not a number", "flag": True, "bad": float("nan"),
                "huge": float("inf")}


def test_metrics_json_and_prometheus_text():
    c = TestClient(create_app(Gauged(), SECRET))
    assert c.get("/metrics").status_code == 401
    assert "metrics-v1" in c.get("/manifest", headers=H).json()["capabilities"]
    assert c.post("/tools/echo", json={"args": {"text": "hi"}}, headers=H).json()["result"] == "hi"
    c.post("/tools/later", json={"args": {}}, headers=H)
    m = c.get("/metrics", headers=H).json()
    assert set(m) == {"uptime_s", "rss_mb", "cpu_pct", "loop_lag_s", "loop_lag_max_s",
                      "overloaded_429_total", "upload_v1_total", "upload_v2_total", "window_s",
                      "tools", "pools", "world", "parked", "attempt", "paused"}
    assert m["world"] == {"open_prs": 3} and m["overloaded_429_total"] == 0 and m["rss_mb"] > 0
    echo, later = m["tools"]["echo"], m["tools"]["later"]
    assert set(echo) == {"calls", "errors_by_code", "latency_p50_s", "latency_p95_s",
                         "latency_max_s", "bytes_in", "bytes_out", "calls_total", "errors_total"}
    assert (echo["calls"], echo["errors_by_code"], echo["calls_total"]) == (1, {}, 1)
    assert echo["bytes_in"] > 0 and echo["bytes_out"] > 0
    assert later["errors_by_code"] == {"open_budget": 1} and later["errors_total"] == 1
    assert set(m["pools"]) == {"fast", "slow"} and m["pools"]["fast"]["limit"] == 40
    assert set(m["pools"]["slow"]) == {"in_flight", "queued", "limit", "queue_wait_oldest_s",
                                       "queue_wait_max_s", "queue_wait_p95_s",
                                       "overloaded_429_total"}
    text = c.get("/metrics", headers={**H, "Accept": "text/plain"})
    assert text.headers["content-type"].startswith("text/plain")
    lines = text.text.splitlines()
    assert 'flock_world_tool_calls{tool="echo"} 1' in lines
    assert 'flock_world_tool_errors{tool="later",code="open_budget"} 1' in lines
    assert 'flock_world_pool_limit{pool="slow"} 64' in lines
    assert "flock_world_world_open_prs 3" in lines and "flock_world_overloaded_429_total 0" in lines


def test_metrics_window_and_percentiles_with_a_fake_clock():
    now = [1000.0]
    m = _Metrics(clock=lambda: now[0])
    for i in range(1, 101):
        m.call("t", i / 100, "busy" if i % 10 == 0 else None, 10, 20)
    m.call("t", None, "overloaded", 5, 0)  # a refusal: counted, no latency
    m.waited("slow", 4.0)
    assert m.waits["fast"].maxlen == m.lags.maxlen == m.calls["t"].maxlen == 20_000
    now[0] += 60
    pools = {"slow": {"queue_wait_oldest_s": 0.0}}
    t = m.snapshot(pools)["tools"]["t"]
    assert (t["calls"], t["latency_p50_s"], t["latency_p95_s"], t["latency_max_s"]) == (101, 0.5, 0.95, 1.0)
    assert t["errors_by_code"] == {"busy": 10, "overloaded": 1} and t["bytes_in"] == 1005
    assert pools["slow"]["queue_wait_max_s"] == 4.0 == pools["slow"]["queue_wait_p95_s"]
    now[0] += WINDOW_S
    m.call("t", 2.0, None, 1, 1)
    snap = m.snapshot({"slow": {"queue_wait_oldest_s": 0.5}})
    t = snap["tools"]["t"]
    assert (t["calls"], t["latency_p95_s"], t["calls_total"], t["errors_total"]) == (1, 2.0, 102, 11)
    assert snap["pools"]["slow"]["queue_wait_max_s"] == 0.5 and snap["uptime_s"] == 60 + WINDOW_S


class Crowded(World):
    name, version = "crowded", "1"
    concurrent = True
    capabilities: ClassVar[list[str]] = ["workspace-upload-v1"]

    def __init__(self) -> None:
        self.release = threading.Event()
        self.uploads = 0

    @tool(pool="slow")
    def slow(self) -> str:
        """Blocks for up to 30 s."""
        self.release.wait(30)
        return "slow done"

    @tool(pool="slow")
    def push(self, path: str) -> str:
        """An upload tool."""
        return "pushed"

    @tool
    def quick(self) -> str:
        """Blocks a fast thread for up to 30 s."""
        self.release.wait(30)
        return "quick done"

    def manifest(self) -> dict:
        m = super().manifest()
        m["tools"][1]["x-flock-workspace"] = {"arg": "path"}
        return m

    def call_workspace(self, name, args, agent, snapshot):
        self.uploads += 1
        return "pushed"


async def test_admission_refuses_a_full_pool_with_retry_after(monkeypatch):
    monkeypatch.setattr(server, "SLOW_THREADS", 2)
    monkeypatch.setattr(server, "FAST_THREADS", 1)
    monkeypatch.setenv("FLOCK_WORLD_SLOW_QUEUE", "1")
    monkeypatch.setenv("FLOCK_WORLD_FAST_MAX_WAIT_S", "0.3")
    served = Served(Crowded())
    try:
        async with httpx.AsyncClient(base_url=served.url, headers=H, timeout=60) as c:
            slow = [asyncio.create_task(c.post("/tools/slow", json={"args": {}})) for _ in range(3)]
            for _ in range(100):  # 2 running + 1 queued: the slow pool is full
                if (await c.get("/metrics")).json()["pools"]["slow"]["in_flight"] == 3:
                    break
                await asyncio.sleep(0.05)
            r = await c.post("/tools/slow", json={"args": {}})
            assert r.status_code == 429 and r.headers["Retry-After"] == "30"
            assert r.json()["error"]["code"] == "overloaded"
            assert r.json()["error"]["retry_after_s"] == 30
            assert (await c.get("/manifest")).status_code == 200

            async def garbage():  # not JSON: a 400 would mean the body was parsed
                for _ in range(64):
                    yield b"x" * 65536

            r = await c.post("/tools/push", content=garbage())
            assert r.status_code == 429 and served.world.uploads == 0
            # The fast pool: one thread, busy; a queued call waits past 0.3 s.
            quick = [asyncio.create_task(c.post("/tools/quick", json={"args": {}})) for _ in range(2)]
            await asyncio.sleep(0.6)
            r = await c.post("/tools/quick", json={"args": {}})
            assert r.status_code == 429 and "wait" in r.json()["error"]["message"]
            m = (await c.get("/metrics")).json()
            assert m["overloaded_429_total"] == 3
            assert m["tools"]["slow"]["errors_by_code"] == {"overloaded": 1}
            assert m["pools"]["fast"]["queue_wait_oldest_s"] > 0.3
            served.world.release.set()
            assert all(r.status_code == 200 for r in await asyncio.gather(*slow, *quick))
            r = await c.post("/tools/slow", json={"args": {}})  # admitted again
            assert r.json() == {"ok": True, "result": "slow done"}
    finally:
        served.world.release.set()
        served.close()


def test_refusals_log_their_pool_and_reason_at_most_every_10_s(monkeypatch, caplog):
    monkeypatch.setattr(server, "SLOW_THREADS", 2)
    monkeypatch.setenv("FLOCK_WORLD_SLOW_QUEUE", "-2")  # the slow pool is always full
    c = TestClient(create_app(Crowded(), SECRET))
    with caplog.at_level(logging.WARNING, logger="flock.world"):
        for _ in range(3):
            assert c.post("/tools/slow", json={"args": {}}, headers=H).status_code == 429
        monkeypatch.setattr(server, "REFUSAL_LOG_EVERY_S", 0)  # as if 10 s went by
        assert c.post("/tools/push", json={"args": {"path": "x"}}, headers=H).status_code == 429
    why = "slow pool overloaded (0 slow calls running or queued)"
    assert [r.getMessage() for r in caplog.records] == [
        f"{why}: refused slow; 0 more since the last line",
        f"{why}: refused push; 2 more since the last line"]
    pools = c.get("/metrics", headers=H).json()["pools"]
    assert (pools["slow"]["overloaded_429_total"], pools["fast"]["overloaded_429_total"]) == (4, 0)


def test_prometheus_groups_metrics_and_escapes_labels():
    from flock.world.server import _prometheus
    text = _prometheus({"tools": {"a": {"calls": 1, "p": 2}, 'b"\\x': {"calls": 3, "p": 4}}})
    names = [l.split("{")[0] for l in text.splitlines()]
    assert names == ["flock_world_tool_calls"] * 2 + ["flock_world_tool_p"] * 2
    assert 'tool="b\\"\\\\x"' in text
