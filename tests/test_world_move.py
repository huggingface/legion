"""A world whose host moves (Slurm requeue, docs/slurm.md): world.url with ${WORLD_HOST},
FLOCK_WORLD_HOST_FILE on the controller. The controller follows the file and republishes the
endpoint; workers keep their credential (no relaunch) and call the new host."""

from __future__ import annotations

import httpx
import pytest

from flock.controller.api import create_app
from flock.models import SwarmSpec, WorkerHeartbeat
from flock.world.host import WorldError, follow_world_host, world_endpoint
from flock.world.hub import WorldHub
from flock.world.protocol import TOKEN_ENV, credential_id, origin_matches

from .conftest import TOKEN
from .test_world import SECRET, Recorder, Served

TEMPLATE = "http://${WORLD_HOST}"


def test_origin_matches():
    assert origin_matches("http://a:1", "http://a:1")
    assert not origin_matches("http://a:1", "http://b:1")
    assert origin_matches("http://${WORLD_HOST}:7860", "http://ip-10-0-0-9:7860")
    assert not origin_matches("http://${WORLD_HOST}:7860", "http://evil.com/x:7860")
    assert not origin_matches("http://${WORLD_HOST}:7860", "https://h:7860")
    assert not origin_matches("http://${WORLD_HOST}:7860", "http://:7860")
    t = "http://${WORLD_HOST}:7860"  # a host:port file fills in the free port the job bound
    assert origin_matches(t, "http://ip-10-53-93-119:44951")
    assert origin_matches(t, "http://ip-10-53-93-119:7860")
    assert not origin_matches(t, "http://h")  # fill_host keeps :7860 for a host-only file
    assert not origin_matches(t, "http://evil.com:1/x")
    assert not origin_matches(t, "http://a@b:1")
    assert not origin_matches(t, "http://evil/x?:7860")
    assert not origin_matches(t, "http://h:1:2")
    assert origin_matches("http://${WORLD_HOST}/p", "http://h:5/p")
    assert not origin_matches("http://${WORLD_HOST}/p", "http://h:5/q")


def test_world_endpoint_needs_the_host_file(tmp_path):
    with pytest.raises(WorldError, match="FLOCK_WORLD_HOST_FILE"):
        world_endpoint(TEMPLATE + ":7860", {})
    f = tmp_path / "world.host"
    f.write_text("")
    with pytest.raises(WorldError, match="empty"):
        world_endpoint(TEMPLATE, {"FLOCK_WORLD_HOST_FILE": str(f)})
    f.write_text("node-3\n")
    assert world_endpoint(TEMPLATE + ":7860", {"FLOCK_WORLD_HOST_FILE": str(f)}) == \
        "http://node-3:7860"  # an old host-only file: the url's port
    f.write_text("node-3:41234\n")  # host:port: the job bound a free port
    assert world_endpoint(TEMPLATE + ":7860/x", {"FLOCK_WORLD_HOST_FILE": str(f)}) == \
        "http://node-3:41234/x"
    assert world_endpoint(TEMPLATE, {"FLOCK_WORLD_HOST_FILE": str(f)}) == "http://node-3:41234"
    assert world_endpoint("http://fixed:1", {}) == "http://fixed:1"


async def test_moved_world_is_republished_without_a_relaunch(tmp_path, store, clock, monkeypatch):
    host_file = tmp_path / "world.host"
    first, second = Served(Recorder()), Served(Recorder())
    try:
        host_file.write_text(first.url.removeprefix("http://") + "\n")
        monkeypatch.setenv("FLOCK_WORLD_HOST_FILE", str(host_file))
        monkeypatch.setenv("WORLD_TOKEN", SECRET)
        spec = SwarmSpec.model_validate({
            "swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh",
            "world": {"url": TEMPLATE, "token_env": "WORLD_TOKEN"},
            "types": {"researcher": {"harness": "claude", "model": "claude-opus-5-5",
                                     "replicas": 1, "world_tools": True}},
        })
        app = create_app(store, TOKEN, clock=clock)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://t",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            r = await c.put("/swarms/demo", json=spec.model_dump(mode="json"))
            assert r.status_code == 200, r.text
            stored = store.get_swarm("demo")["spec"].world
            assert stored.endpoint == first.url and stored.credential_origin == TEMPLATE
            env = {"FLOCK_WORLD_HOST_FILE": str(host_file)}
            assert follow_world_host(store.get_swarm("demo")["spec"], TOKEN,
                                     {"WORLD_TOKEN": SECRET}, env=env) is None  # unchanged

            # the world job is requeued on another node
            host_file.write_text(second.url.removeprefix("http://") + "\n")
            new = follow_world_host(store.get_swarm("demo")["spec"], TOKEN,
                                    {"WORLD_TOKEN": SECRET}, env=env)
            assert new is not None and new.world.endpoint == second.url
            assert new.world.credential_id == stored.credential_id  # same credential
            store.update_swarm_spec(new)

            # a worker gets the new endpoint with its next heartbeat
            from flock.controller.jobs import FakeJobsClient
            from flock.controller.reconcile import reconcile

            reconcile(store, FakeJobsClient(), clock(), _settings())
            wid = "demo-researcher-w0000"
            hb = WorkerHeartbeat(worker=wid, deadline=clock(), agents=[])
            r = await c.post(f"/workers/{wid}/heartbeat", json=hb.model_dump(mode="json"))
            assert r.status_code == 200
            block = r.json()["world"]
            assert block["endpoint"] == second.url

        hub = WorldHub(swarm="demo", type_name="researcher", env={TOKEN_ENV: SECRET})
        hub.configure_raw(block)
        assert not hub.needs_restart()  # the secret still binds: no drain, no relaunch
        snap = hub.snapshot()
        assert snap is not None and snap.endpoint == second.url and snap.send_token
        # a different (unlisted) origin would not be trusted
        hub.configure_raw({**block, "endpoint": "https://elsewhere.example/x"})
        assert hub.needs_restart()
        t = "http://${WORLD_HOST}:7860"  # the endpoint's port is the one the job bound
        hub.configure_raw({**block, "credential_origin": t, "credential_id": credential_id(SECRET, t),
                           "endpoint": "http://ip-10-53-93-119:44951"})
        assert not hub.needs_restart()
    finally:
        first.close()
        second.close()


def _settings():
    from flock.controller.reconcile import Settings

    return Settings(public_url="http://ctl:7860", token=TOKEN, secret_source={})
