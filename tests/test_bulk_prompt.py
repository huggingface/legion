"""`POST /swarms/{swarm}/prompts` and `flock prompt --swarm S --type T`: one request prompts
every agent of a type; prompts are held and admitted exactly as single prompts are."""

from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from flock import cli
from flock.client import Client
from flock.models import AgentHeartbeat

from .conftest import HB, make_spec, wire
from .test_api import W0, apply, hb_body
from .test_providers_controller import CAP_PROVIDERS, _cap_client, _delivered, hb, pspec, put


async def _prompts(client, wid) -> list[dict]:
    cmds = (await client.get(f"/workers/{wid}/commands", params={"wait": 0})).json()["commands"]
    return [c for c in cmds if c["kind"] == "prompt"]


async def test_bulk_prompt_by_type_and_state(client, store, clock, tick):
    await apply(client)  # researcher x3: 0000, 0001 on w0000; 0002 on w0001 (never reported)
    tick()
    store.clear_commands(W0)
    agents = [AgentHeartbeat(name="researcher-0000", state="idle"),
              AgentHeartbeat(name="researcher-0001", state="working", turn_id="t-1")]
    assert (await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))).status_code == 200

    r = await client.post("/swarms/demo/prompts", json={"type": "researcher", "state": "idle", "text": "go",
                                                         "reply_to": "pixel"})
    assert r.status_code == 200, r.text
    assert r.json() == {"swarm": "demo", "type": "researcher", "parent": None, "matched": 3, "queued": 1, "pending": 0,
                        "skipped_state": 2, "skipped_removing": 0,
                        "skipped_pending": 0}
    [p] = await _prompts(client, W0)
    assert p["agent"] == "researcher-0000" and p["text"] == "go" and p["reply_to"] == "pixel"

    store.mark_removing("researcher-0001")
    ack = (await client.post("/swarms/demo/prompts", json={"type": "researcher", "text": "all"})).json()
    assert (ack["matched"], ack["queued"], ack["pending"], ack["skipped_removing"]) == (3, 1, 1, 1)
    assert [p["agent"] for p in await _prompts(client, W0)] == ["researcher-0000"]
    assert store.pending_counts("demo") == {"researcher-0002": 1}  # held for its worker

    assert (await client.post("/swarms/demo/prompts", json={"type": "nope", "text": "x"})).status_code == 404
    assert (await client.post("/swarms/zz/prompts", json={"type": "researcher", "text": "x"})).status_code == 404
    bad = await client.post("/swarms/demo/prompts", json={"type": "researcher", "state": "sleepy", "text": "x"})
    assert bad.status_code == 422
    await client.delete("/swarms/demo")
    assert (await client.post("/swarms/demo/prompts", json={"type": "researcher", "text": "x"})).status_code == 409


async def test_bulk_prompts_respect_max_working(store, clock, tick):
    spec = pspec(CAP_PROVIDERS, worker={**HB, "model": "p:test", "replicas": 3})
    wid = "p-worker-w0000"
    async with await _cap_client(store, clock, True) as c:
        assert (await put(c, spec)).status_code == 200
        tick()
        await _delivered(c, wid)  # drain the assigns
        idle = [AgentHeartbeat(name=f"worker-000{i}", state="idle") for i in range(3)]
        await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, idle))
        ack = (await c.post("/swarms/p/prompts", json={"type": "worker", "text": "paint"})).json()
        assert ack["queued"] == 3
        assert await _delivered(c, wid) == ["worker-0000"]  # max_working 1: one admitted
        assert sum(store.pending_counts("p").values()) == 2  # the rest wait in the controller


async def test_agent_tokens_cannot_bulk_prompt(client, app, store, tick):
    import httpx

    from flock.controller.spawn import agent_may

    await apply(client)
    tick()
    agent = store.get_agent("researcher-0000")
    assert not agent_may(store, agent, "POST", "/swarms/{swarm}/prompts", {"swarm": "demo"}, {})
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as anon:
        r = await anon.post("/swarms/demo/prompts", json={"type": "researcher", "text": "x"})
        assert r.status_code == 401


runner = CliRunner()


def test_cli_prompt_by_type(tmp_path, monkeypatch, app, store, tick):
    wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(make_spec().model_dump(mode="json")))
    assert runner.invoke(cli.app, ["launch", str(spec_file)]).exit_code == 0
    tick()
    r = runner.invoke(cli.app, ["prompt", "--swarm", "demo", "--type", "researcher", "hello all"])
    assert r.exit_code == 0, r.output
    ack = json.loads(r.output)
    assert ack["matched"] == 3 and ack["pending"] == 3  # no worker reported yet: held
    assert store.pending_counts("demo") == {f"researcher-000{i}": 1 for i in range(3)}
    r = runner.invoke(cli.app, ["prompt", "--swarm", "demo", "--type", "researcher", "--state", "idle", "x"])
    assert r.exit_code == 0 and json.loads(r.output)["skipped_state"] == 3  # all unknown
    r = runner.invoke(cli.app, ["prompt", "researcher-0001", "one"])  # the single form still works
    assert r.exit_code == 0 and json.loads(r.output)["agent"] == "researcher-0001"
    for bad in (["prompt", "--type", "researcher", "x"],  # no --swarm
                ["prompt", "--swarm", "demo", "--type", "researcher", "a", "b"],
                ["prompt", "only-text"],
                ["prompt", "--state", "idle", "researcher-0001", "x"]):
        assert runner.invoke(cli.app, bad).exit_code == 2, bad


async def test_repeated_idle_bulk_calls_do_not_stack_prompts(store, clock, tick):
    """With admission blocked, repeated `--state idle` calls must not queue a prompt each."""
    spec = pspec(CAP_PROVIDERS, worker={**HB, "model": "p:test", "replicas": 3})
    wid = "p-worker-w0000"
    body = {"type": "worker", "state": "idle", "text": "paint"}
    async with await _cap_client(store, clock, True) as c:
        assert (await put(c, spec)).status_code == 200
        tick()
        await _delivered(c, wid)
        idle = [AgentHeartbeat(name=f"worker-000{i}", state="idle") for i in range(3)]
        await c.post(f"/workers/{wid}/heartbeat", json=hb(clock, wid, idle))
        first = (await c.post("/swarms/p/prompts", json=body)).json()
        assert (first["queued"], first["skipped_pending"]) == (3, 0)
        assert await _delivered(c, wid) == ["worker-0000"]  # max_working 1: two stay pending
        for _ in range(5):  # the driver's ticks while agents are blocked
            ack = (await c.post("/swarms/p/prompts", json=body)).json()
            # 0000 was just handed its prompt (heartbeat still says idle): working, not re-prompted
            assert (ack["queued"], ack["skipped_pending"], ack["skipped_state"]) == (0, 2, 1)
        assert store.pending_counts("p") == {"worker-0001": 1, "worker-0002": 1}
        # its turn ends (event, no heartbeat yet): idle again, prompted once
        evs = [{"type": k, "ts": clock().isoformat(), "agent": "worker-0000"}
               for k in ("turn_started", "turn_ended")]
        await c.post(f"/workers/{wid}/events", json={"worker": wid, "events": evs})
        ack = (await c.post("/swarms/p/prompts", json=body)).json()
        assert (ack["queued"], ack["skipped_pending"]) == (1, 2)
        # explicit skip_pending=false stacks
        ack = (await c.post("/swarms/p/prompts", json={**body, "skip_pending": False})).json()
        assert ack["queued"] == 3 and ack["skipped_pending"] == 0


def test_cli_skip_pending_flag(monkeypatch, app, store, tick, tmp_path):
    sent = []

    def fake_call(self, method, path, **kw):
        sent.append(kw["json"])
        return {}

    monkeypatch.setattr(Client, "_call", fake_call)
    for flags in ([], ["--stack"], ["--skip-pending"]):
        r = runner.invoke(cli.app, ["prompt", "--swarm", "s", "--type", "t", "--state", "idle", *flags, "x"])
        assert r.exit_code == 0, r.output
    assert ["skip_pending" in b for b in sent] == [False, True, True]
    assert [b.get("skip_pending") for b in sent[1:]] == [False, True]
