"""World server: structured retry hints, /metrics and admission control (docs/world.md)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from flock.world import ToolError, World, tool
from flock.world.server import WINDOW_S, _Metrics, create_app

from .test_world import SECRET

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
        return {"open_prs": 3, "note": "not a number"}


def test_metrics_json_and_prometheus_text():
    c = TestClient(create_app(Gauged(), SECRET))
    assert c.get("/metrics").status_code == 401
    assert "metrics-v1" in c.get("/manifest", headers=H).json()["capabilities"]
    assert c.post("/tools/echo", json={"args": {"text": "hi"}}, headers=H).json()["result"] == "hi"
    c.post("/tools/later", json={"args": {}}, headers=H)
    m = c.get("/metrics", headers=H).json()
    assert set(m) == {"uptime_s", "rss_mb", "cpu_pct", "loop_lag_s", "loop_lag_max_s",
                      "overloaded_429_total", "window_s", "tools", "pools", "world"}
    assert m["world"] == {"open_prs": 3} and m["overloaded_429_total"] == 0 and m["rss_mb"] > 0
    echo, later = m["tools"]["echo"], m["tools"]["later"]
    assert set(echo) == {"calls", "errors_by_code", "latency_p50_s", "latency_p95_s",
                         "latency_max_s", "bytes_in", "bytes_out", "calls_total", "errors_total"}
    assert (echo["calls"], echo["errors_by_code"], echo["calls_total"]) == (1, {}, 1)
    assert echo["bytes_in"] > 0 and echo["bytes_out"] > 0
    assert later["errors_by_code"] == {"open_budget": 1} and later["errors_total"] == 1
    assert set(m["pools"]) == {"fast", "slow"} and m["pools"]["fast"]["limit"] == 40
    assert set(m["pools"]["slow"]) == {"in_flight", "queued", "limit", "queue_wait_oldest_s",
                                       "queue_wait_max_s", "queue_wait_p95_s"}
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
