"""World server: parked mode, /healthz, and the admin API (knobs, lane pauses)."""

from __future__ import annotations

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock.controller.api import create_app as controller_app
from flock.models import SwarmSpec
from flock.world import Admin, World, server, tool
from flock.world.__main__ import world_app
from flock.world.host import derived_token, world_admin_token
from flock.world.server import create_app

from .conftest import TOKEN, wire

SECRET = "world-secret"
ADMIN_SECRET = "admin-secret-value"
ADMIN = {"X-World-Token": ADMIN_SECRET}


@pytest.fixture(autouse=True)
def admin_secret(monkeypatch):
    monkeypatch.setenv("FLOCK_WORLD_ADMIN_TOKEN", ADMIN_SECRET)
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


def test_admin_is_403_without_an_admin_secret(monkeypatch):
    monkeypatch.delenv("FLOCK_WORLD_ADMIN_TOKEN")
    for token in (SECRET, None):
        c = TestClient(create_app(Knobbed(), token))
        for h in (ADMIN, {"X-World-Token": SECRET}, {}):
            assert c.get("/admin/knobs", headers=h).status_code == 403
        assert c.get("/manifest", headers={"X-World-Token": SECRET}).status_code == 200


def test_admin_bodies_are_capped():
    c = TestClient(create_app(Knobbed(), SECRET))
    big = {"lanes": ["push"], "reason": "x" * (65 << 10)}
    r = c.post("/admin/pause", json=big, headers=ADMIN)
    assert r.status_code == 400 and "over 65536 bytes" in r.json()["error"]
    assert c.patch("/admin/knobs", json={"hub.push.slots": 2, "pad": "x" * (65 << 10)},
                   headers=ADMIN).status_code == 400


def test_a_failing_on_change_rolls_the_whole_patch_back(monkeypatch, tmp_path):
    monkeypatch.setenv("FLOCK_WORLD_STATE_DIR", str(tmp_path))
    w = Knobbed()
    seen = []
    w.admin.knobs["save.interval_s"].on_change = lambda v: (seen.append(v), 1 / 0)
    c = TestClient(create_app(w, SECRET))
    r = c.patch("/admin/knobs", json={"hub.push.slots": 3, "save.interval_s": 5}, headers=ADMIN)
    assert r.status_code == 400 and r.json()["error"].startswith("save.interval_s: the world refused 5")
    assert w.slots_seen == [3, 1] and seen == [5.0]  # applied, then given its old value back
    assert w.knobs == {"hub.push.slots": 1, "save.interval_s": 2.0}
    assert not (tmp_path / "knobs.json").exists()


@pytest.mark.parametrize("text", ['{"paused": ["x"]}', '{"knobs": [1]}', '[]', '{"paused": {"a": 1}}',
                                  'not json'])
def test_a_wrong_shaped_knobs_file_falls_back_to_the_defaults(monkeypatch, tmp_path, text):
    monkeypatch.setenv("FLOCK_WORLD_STATE_DIR", str(tmp_path))
    (tmp_path / "knobs.json").write_text(text)
    w = Knobbed()
    assert w.slots == 1 and w.admin.paused == {}


def test_a_bad_target_exits_before_serving(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notworld.py").write_text("class X:\n    pass\n")
    for target, exc in (("nosuchmodule:X", ImportError), ("notworld:X", TypeError),
                        ("notworld", ValueError)):
        with pytest.raises(exc):
            server.serve(target, insecure=True)


def test_sys_exit_in_init_parks(monkeypatch):
    monkeypatch.setattr(server, "PARK_FIRST_S", 60)

    def build(admin):
        raise SystemExit(3)

    with TestClient(create_app(build, SECRET)) as c:
        wait_for(lambda: c.get("/manifest", headers=H).json()["error"] == "SystemExit: 3")
        assert c.get("/healthz").json() == {"status": "parked"}


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


async def test_hub_treats_a_parked_503_as_a_world_holdoff(monkeypatch, tmp_path):
    from .test_world import _holdoff_hub

    now, seen, captured, parked = [100.0], [], [], [True]

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.path)
        if parked[0]:
            return httpx.Response(503, headers={"Retry-After": "120"}, json={
                "ok": False, "parked": True, "error": "OSError: restore failed", "attempt": 3,
                "next_retry_s": 120})
        return httpx.Response(200, json={"ok": True, "text": "fine"})

    monkeypatch.setattr("flock.world.hub.capture", lambda workdir, path: captured.append(path) or {})
    ws = {"arg": "ref", "default": "main"}
    hub = _holdoff_hub(handler, [
        {"name": "up", "parameters": {"type": "object", "properties": {"ref": {"type": "string"}}},
         "x-flock-workspace": ws},
        {"name": "peek", "annotations": {"readOnlyHint": True}}], lambda: now[0],
        runtime=lambda name: type("Rt", (), {"workdir": tmp_path})())
    r = await hub.call("x", "world:peek", {})  # read-only: not retried against a parked world
    assert json.loads(r.output) == {"error": "world w unavailable (OSError: restore failed); work "
                                    "locally", "code": "parked", "retry_after_s": 120}
    assert len(seen) == 1
    now[0] += 30
    for name, args in (("up", {"ref": "b"}), ("peek", {})):  # held: no request, no upload
        body = json.loads((await hub.call("x", f"world:{name}", args)).output)
        assert body["local"] and body["code"] == "parked" and "work locally" in body["error"]
    assert len(seen) == 1 and captured == []
    now[0] += 91
    parked[0] = False
    assert not (await hub.call("x", "world:up", {"ref": "b"})).error and len(seen) == 2
    await hub.aclose()


def test_world_admin_token_is_never_the_world_secret():
    def spec(world):
        return SwarmSpec.model_validate({"swarm": "sw", "compute": "local", "world": world,
                                         "types": {"dev": {"harness": "claude", "model": "m"}}})
    local = spec({"command": ["python", "w.py"]})
    assert world_admin_token(local, TOKEN, {}) == derived_token(TOKEN, "flock-world-admin:sw")
    assert world_admin_token(local, TOKEN, {}) != derived_token(TOKEN, "sw")  # the world secret
    url = spec({"url": "http://w.invalid", "token_env": "WORLD_SECRET"})
    assert world_admin_token(url, TOKEN, {"WORLD_SECRET": SECRET}) is None
    assert world_admin_token(url, TOKEN, {"FLOCK_WORLD_ADMIN_TOKEN": "a"}) == "a"


def _forward(world_client: TestClient):
    """An httpx transport that sends the controller's world requests to an in-process world."""
    def handler(req: httpx.Request) -> httpx.Response:
        r = world_client.request(req.method, req.url.path, content=req.content,
                                 headers={k: v for k, v in req.headers.items() if k != "host"})
        return httpx.Response(r.status_code, content=r.content, headers=r.headers)
    return httpx.MockTransport(handler)


def _swarm(world_client, store, clock):
    app = controller_app(store, TOKEN, clock=clock, secret_source={
        "WORLD_SECRET": SECRET, "FLOCK_WORLD_ADMIN_TOKEN": ADMIN_SECRET},
                         loop_interval=0, world_transport=_forward(world_client))
    spec = SwarmSpec.model_validate({
        "swarm": "sw", "compute": "local", "world": {"url": "http://w.invalid", "token_env": "WORLD_SECRET"},
        "types": {"dev": {"harness": "claude", "model": "m", "replicas": 1}}})
    return app, spec.model_dump(mode="json")


def test_controller_proxies_the_admin_api_and_records_events(store, clock, monkeypatch):
    world = Knobbed()
    app, spec = _swarm(TestClient(create_app(world, SECRET)), store, clock)
    wire(monkeypatch, app)
    monkeypatch.setenv("USER", "carol")
    run = CliRunner()
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    assert c.put("/swarms/sw", json=spec).status_code == 200

    r = run.invoke(world_app, ["set", "sw", "hub.push.slots=2", "--reason", "hub is fine"])
    assert r.exit_code == 0 and "hub.push.slots: 1 -> 2" in r.output, r.output
    assert world.slots_seen == [2]
    r = run.invoke(world_app, ["set", "sw", "hub.push.slots=7"])
    assert r.exit_code == 1 and "maximum" in r.output
    world.admin.pause(["push"], "429s")
    r = run.invoke(world_app, ["knobs", "sw"])
    assert r.exit_code == 0 and "hub.push.slots" in r.output and "0..4" in r.output
    assert "paused: push (429s)" in r.output and world.paused("push")

    evs = c.get("/swarms/sw/events").json()["events"]
    mine = [(e["type"], e["data"]) for e in evs if e["type"].startswith("world_")]
    assert mine == [
        ("world_knob", {"name": "hub.push.slots", "old": 1, "new": 2, "by": "carol", "reason": "hub is fine"})]
    assert c.get("/swarms/nope/world/knobs").status_code == 404


def test_apply_keeps_the_last_manifest_while_the_world_is_parked(store, clock, monkeypatch):
    monkeypatch.setattr(server, "PARK_FIRST_S", 60)
    gate = []

    def build(admin):
        if not gate:
            raise OSError("bucket restore failed")
        return Knobbed()

    with TestClient(create_app(build, SECRET)) as wc:
        app, spec = _swarm(wc, store, clock)
        c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        wait_for(lambda: "bucket" in wc.get("/manifest", headers=H).json()["error"])
        r = c.put("/swarms/sw", json=spec)  # never applied: nothing to keep
        assert r.status_code == 400 and "world parked" in r.json()["detail"]
        assert "bucket restore failed" in r.json()["detail"]
    with TestClient(create_app(Knobbed(), SECRET)) as wc:
        app, spec = _swarm(wc, store, clock)
        c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        assert c.put("/swarms/sw", json=spec).status_code == 200
    with TestClient(create_app(build, SECRET)) as wc:
        app, spec = _swarm(wc, store, clock)
        c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
        wait_for(lambda: wc.get("/manifest", headers=H).status_code == 503)
        assert c.put("/swarms/sw", json=spec).status_code == 200  # parked: the last manifest
        assert store.get_swarm("sw")["spec"].world.manifest.name == "knobbed"
