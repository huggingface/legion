"""World server: structured retry hints, /metrics and admission control (docs/world.md)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from flock.world import ToolError, World, tool
from flock.world.server import create_app

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
