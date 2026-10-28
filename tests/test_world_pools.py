"""World server thread pools and read-only tools (docs/world.md)."""

from __future__ import annotations

import asyncio
import threading
import time

import httpx
import pytest

from flock.world import Agent, World, tool
from flock.world.protocol import WorldManifest

from .test_world import SECRET, Served

H = {"X-World-Token": SECRET, "X-Flock-Agent": "dev-0001"}


class Jammed(World):
    name, version = "jammed", "1"
    concurrent = True

    def __init__(self) -> None:
        self.release = threading.Event()

    @tool(pool="slow")
    def slow(self) -> str:
        """Blocks for up to 30 s."""
        self.release.wait(30)
        return "slow done"

    @tool(read_only=True)
    def peek(self, agent: Agent) -> dict:
        """Quick read."""
        return {"by": agent.name}


def test_read_only_is_in_the_manifest_and_pools_are_checked():
    m = WorldManifest.model_validate(Jammed().manifest())
    assert m.get("peek").read_only and m.get("peek").annotations == {"readOnlyHint": True}
    assert not m.get("slow").read_only and "annotations" not in Jammed().manifest()["tools"][0]
    assert Jammed().pool("slow") == "slow" and Jammed().pool("peek") == "fast"
    with pytest.raises(ValueError, match="pool"):
        tool(pool="medium")


async def test_manifest_and_fast_tools_answer_under_100_slow_calls():
    served = Served(Jammed())
    try:
        async with httpx.AsyncClient(base_url=served.url, headers=H, timeout=60,
                                     limits=httpx.Limits(max_connections=200)) as c:
            slow = [asyncio.create_task(c.post("/tools/slow", json={"args": {}}))
                    for _ in range(100)]
            await asyncio.sleep(1)  # every slow call is in a thread or queued
            lat = []
            for _ in range(100):
                t0 = time.monotonic()
                r = await c.get("/manifest")
                lat.append(time.monotonic() - t0)
                assert r.status_code == 200 and r.json()["name"] == "jammed"
            lat.sort()
            assert lat[98] < 0.1, f"p99 /manifest {lat[98]:.3f} s"
            r = await asyncio.wait_for(c.post("/tools/peek", json={"args": {}}), 2)
            assert r.json() == {"ok": True, "result": {"by": "dev-0001"}}
            assert not any(t.done() for t in slow)
            served.world.release.set()
            done = await asyncio.gather(*slow)
            assert all(r.json() == {"ok": True, "result": "slow done"} for r in done)
    finally:
        served.world.release.set()
        served.close()
