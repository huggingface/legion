"""World server: parked mode, /healthz, and the admin API (knobs, lane pauses)."""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from flock.world import Admin, World, server, tool
from flock.world.protocol import admin_token
from flock.world.server import create_app

SECRET = "world-secret"
ADMIN = {"X-World-Token": admin_token(SECRET)}
H = {"X-World-Token": SECRET, "X-Flock-Agent": "dev-0001"}


class Knobbed(World):
    name, version = "knobbed", "1"

    def __init__(self) -> None:
        self.slots_seen: list[int] = []
        self.slots = self.knob("hub.push.slots", int, 1, min=0, max=4, doc="Hub write slots.",
                               on_change=self.slots_seen.append)
        self.knob("save.interval_s", float, 2.0, min=0)

    @tool
    def push(self) -> str:
        """Push unless paused."""
        return "paused" if self.paused("push") else f"pushed with {self.knobs['hub.push.slots']}"


def wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.01)


def test_parked_until_the_world_builds(monkeypatch):
    monkeypatch.setattr(server, "PARK_FIRST_S", 0.05)
    monkeypatch.setenv("FAKE_TOKEN", "s3cr3t-value-123")
    tries, gate = [], []

    def build(admin: Admin) -> World:
        tries.append(admin)
        if len(tries) < 3 or not gate:  # raises at least twice, then until the test opens it
            raise RuntimeError("restore failed: token s3cr3t-value-123\nsecond line")
        w = Knobbed.__new__(Knobbed)
        w.__dict__["_flock_admin"] = admin
        w.__init__()
        return w

    with TestClient(create_app(build, SECRET)) as c:
        wait_for(lambda: len(tries) >= 2)
        assert c.get("/healthz").json() == {"status": "parked"}
        r = c.get("/manifest", headers=H)
        assert r.status_code == 503 and int(r.headers["Retry-After"]) >= 1
        body = r.json()
        assert body["ok"] is False and body["parked"] is True and body["attempt"] >= 1
        assert body["error"] == "RuntimeError: restore failed: token [redacted]"
        assert isinstance(body["next_retry_s"], int)
        assert c.post("/tools/push", json={"args": {}}, headers=H).status_code == 503
        assert c.post("/hooks", json={"event": "turn"}, headers=H).status_code == 503
        assert c.get("/metrics", headers=H).json()["parked"] == 1
        assert c.get("/admin/knobs", headers=ADMIN).json() == {}
        gate.append(1)
        wait_for(lambda: c.get("/healthz").json()["status"] == "ready")
        assert len(tries) >= 3 and len({id(a) for a in tries}) == 1
        m = c.get("/manifest", headers=H).json()
        assert "admin-v1" in m["capabilities"] and m["knobs"] == ["hub.push.slots", "save.interval_s"]
        assert c.post("/tools/push", json={"args": {}}, headers=H).json()["result"] == "pushed with 1"
        assert c.get("/metrics", headers=H).json()["parked"] == 0


def test_pause_while_parked_carries_over(monkeypatch, tmp_path):
    monkeypatch.setattr(server, "PARK_FIRST_S", 0.05)
    monkeypatch.setenv("FLOCK_WORLD_STATE_DIR", str(tmp_path))
    gate = []

    def build(admin: Admin) -> World:
        if not gate:
            raise RuntimeError("not yet")
        w = Knobbed.__new__(Knobbed)
        admin.attach(tmp_path)
        w.__dict__["_flock_admin"] = admin
        w.__init__()
        return w

    with TestClient(create_app(build, SECRET)) as c:
        assert c.get("/healthz").json() == {"status": "parked"}
        assert c.post("/admin/pause", json={"lanes": ["push"], "reason": "hub down"},
                      headers=ADMIN).json() == {"ok": True, "paused": {"push": "hub down"}}
        gate.append(1)
        wait_for(lambda: c.get("/healthz").json()["status"] == "ready", 10)
        assert c.post("/tools/push", json={"args": {}}, headers=H).json()["result"] == "paused"


def test_healthz_needs_no_token_and_admin_needs_the_admin_secret():
    c = TestClient(create_app(Knobbed(), SECRET))
    assert c.get("/healthz").json() == {"status": "ready"}
    assert c.get("/admin/knobs", headers={"X-World-Token": SECRET}).status_code == 401
    assert c.get("/admin/knobs").status_code == 401
    assert c.get("/manifest", headers=ADMIN).status_code == 401  # admin secret is not the world's
    assert c.get("/admin/knobs", headers=ADMIN).json()["hub.push.slots"] == {
        "value": 1, "default": 1, "type": "int", "min": 0, "max": 4, "doc": "Hub write slots."}


def test_admin_token_from_the_environment(monkeypatch):
    monkeypatch.setenv("FLOCK_WORLD_ADMIN_TOKEN", "operator-only")
    c = TestClient(create_app(Knobbed(), SECRET))
    assert c.get("/admin/knobs", headers=ADMIN).status_code == 401
    assert c.get("/admin/knobs", headers={"X-World-Token": "operator-only"}).status_code == 200


def test_knob_validation_and_persistence_across_restart(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOCK_WORLD_STATE_DIR", str(tmp_path))
    w = Knobbed()
    c = TestClient(create_app(w, SECRET))
    for bad in ({"hub.push.slots": 9}, {"hub.push.slots": "2"}, {"hub.push.slots": True},
                {"hub.push.slots": 1.5}, {"nope": 1}, {}, [1]):
        r = c.patch("/admin/knobs", json=bad, headers=ADMIN)
        assert r.status_code == 400 and r.json()["ok"] is False, bad
    assert w.slots_seen == [] and not (tmp_path / "knobs.json").exists()
    r = c.patch("/admin/knobs", json={"hub.push.slots": 3, "save.interval_s": 1}, headers=ADMIN)
    assert r.json()["changes"] == [{"name": "hub.push.slots", "old": 1, "new": 3},
                                   {"name": "save.interval_s", "old": 2.0, "new": 1.0}]
    assert w.slots_seen == [3] and w.knobs["hub.push.slots"] == 3
    assert c.post("/tools/push", json={"args": {}}, headers=H).json()["result"] == "pushed with 3"
    assert c.post("/admin/pause", json={"lanes": ["push", "merge"]}, headers=ADMIN).status_code == 200
    assert c.post("/admin/resume", json={"lanes": ["merge"]}, headers=ADMIN).json()["paused"] == {
        "push": ""}
    assert c.post("/admin/pause", json={"lanes": ["Bad Lane"]}, headers=ADMIN).status_code == 400
    assert c.post("/admin/explode", json={}, headers=ADMIN).status_code == 404

    w2 = Knobbed()  # a restart (a Spot requeue): the set values and pauses come back
    assert w2.slots == 3 and w2.knobs["save.interval_s"] == 1.0 and w2.paused("push")
    assert not w2.paused("merge")
    c2 = TestClient(create_app(w2, SECRET))
    assert c2.get("/admin/paused", headers=ADMIN).json() == {"parked": False, "paused": {"push": ""}}
    assert c2.post("/tools/push", json={"args": {}}, headers=H).json()["result"] == "paused"


def test_load_hands_the_admin_to_the_world(monkeypatch, tmp_path):
    (tmp_path / "kw.py").write_text("from tests.test_world_admin import Knobbed\n")
    monkeypatch.setenv("FLOCK_WORLD_STATE_DIR", str(tmp_path / "state"))
    (tmp_path / "state").mkdir()
    admin = Admin()
    w = server.load("kw:Knobbed", str(tmp_path), admin=admin)
    assert w.admin is admin and admin.path == tmp_path / "state" / "knobs.json"


def test_knob_types_are_checked():
    with pytest.raises(ValueError):
        World().knob("x", list, [])
    with pytest.raises(ValueError):
        World().knob("x", int, 9, max=4)  # the default itself is out of range
