"""World server thread pools and read-only tools (docs/world.md)."""

from __future__ import annotations

import asyncio
import threading

import httpx
import pytest

from flock.world import Agent, World, server, tool
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


class Patient(World):
    name, version = "patient", "1"

    def __init__(self) -> None:
        self.cancelled = threading.Event()

    @tool
    async def wait(self) -> str:
        """Waits 30 s."""
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            self.cancelled.set()
            raise
        return "waited"

    def cancellable(self, tool: str) -> bool:
        return True


def test_a_caller_that_gives_up_cancels_the_tool(monkeypatch):
    """The client stops waiting (its own timeout): the server sees the disconnect and cancels
    the tool's async work."""
    monkeypatch.setattr(server, "DISCONNECT_POLL_S", 0.05)
    served = Served(Patient())
    try:
        with pytest.raises(httpx.TimeoutException):
            httpx.post(f"{served.url}/tools/wait", json={"args": {}}, headers=H, timeout=0.3)
        assert served.world.cancelled.wait(5)
    finally:
        served.close()
