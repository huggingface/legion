from __future__ import annotations

import asyncio
from datetime import timedelta

from swarm_harness.models import AgentHeartbeat, Event, TokenUsage, WorkerHeartbeat

from .conftest import make_spec

W0 = "demo-researcher-w0000"


def hb_body(clock, wid, agents, draining=False) -> dict:
    return WorkerHeartbeat(
        worker=wid,
        deadline=clock() + timedelta(hours=24),
        draining=draining,
        agents=agents,
    ).model_dump(mode="json")


async def apply(client, spec=None):
    spec = spec or make_spec()
    r = await client.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    return r.json()


async def test_auth(client, app):
    import httpx

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as anon:
        assert (await anon.get("/healthz")).status_code == 200
        assert (await anon.get("/")).status_code == 200
        assert (await anon.get("/swarms")).status_code == 401
        bad = {"Authorization": "Bearer nope"}
        assert (await anon.get("/workers", headers=bad)).status_code == 401
    assert (await client.get("/swarms")).status_code == 200


async def test_apply_and_get(client, tick):
    st = await apply(client)
    assert st["swarm"] == "demo"
    tick()
    body = (await client.get("/swarms/demo")).json()
    assert body["spec"]["types"]["researcher"]["replicas"] == 3
    assert body["status"]["agents_total"] == 3 and body["status"]["workers_total"] == 2
    assert body["status"]["types"]["researcher"]["agents_by_state"] == {"unknown": 3}
    other = make_spec(swarm="other")
    r = await client.put("/swarms/other", json=other.model_dump(mode="json"))
    assert r.status_code == 409  # type names are global because agent names are
    assert (
        await client.put("/swarms/x", json=make_spec().model_dump(mode="json"))
    ).status_code == 400


async def test_pending_prompt_delivered_on_new_workers_first_poll(client, store, jobs, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    # Worker crashed: prompt is held.
    jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
    tick()
    assert store.get_worker(W0)["job_id"] is None
    ack = (await client.post("/agents/researcher-0000/prompt", json={"text": "hello"})).json()
    assert ack["status"] == "pending" and ack["agent"] == "researcher-0000"
    # Poll from a stale process for a slot without a job gets nothing.
    assert (await client.get(f"/workers/{W0}/commands", params={"wait": 0})).json() == {
        "commands": []
    }
    agent = (await client.get("/agents/researcher-0000")).json()
    assert agent["pending"] == 1

    store.update_worker(W0, next_launch_at=0)
    tick()  # relaunch
    batch = (await client.get(f"/workers/{W0}/commands", params={"wait": 0})).json()["commands"]
    assert [c["kind"] for c in batch] == ["assign", "prompt"]
    assert batch[1]["text"] == "hello" and batch[1]["prompt_id"] == ack["prompt_id"]
    # Delivered: removed after the batch went out.
    assert (await client.get(f"/workers/{W0}/commands", params={"wait": 0})).json() == {
        "commands": []
    }
    assert (await client.get("/agents/researcher-0000")).json()["pending"] == 0


async def test_prompt_to_live_worker_wakes_long_poll(client, store, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    store.clear_commands(W0)
    agents = [AgentHeartbeat(name="researcher-0000", state="idle")]
    r = await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))
    assert r.status_code == 200, r.text

    poll = asyncio.create_task(client.get(f"/workers/{W0}/commands", params={"wait": 5}))
    await asyncio.sleep(0.05)
    assert not poll.done()
    ack = (
        await client.post("/agents/researcher-0000/prompt", json={"text": "hi", "reply_to": "x"})
    ).json()
    assert ack["status"] == "queued"
    batch = (await asyncio.wait_for(poll, 2)).json()["commands"]
    assert batch == [
        {
            "kind": "prompt",
            "agent": "researcher-0000",
            "prompt_id": ack["prompt_id"],
            "text": "hi",
            "timeout": None,
            "reply_to": "x",
        }
    ]


async def test_draining_worker_does_not_get_prompts(client, store, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    store.clear_commands(W0)
    agents = [AgentHeartbeat(name="researcher-0000", state="idle")]
    await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents, draining=True))
    ack = (await client.post("/agents/researcher-0000/prompt", json={"text": "hi"})).json()
    assert ack["status"] == "pending"
    assert (await client.get(f"/workers/{W0}/commands", params={"wait": 0})).json() == {
        "commands": []
    }


async def test_heartbeat_updates_agent_status_and_cost(client, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 2}))
    tick()
    tokens = {"claude-opus-5-5": TokenUsage(input=1_000_000, output=100_000)}
    agents = [
        AgentHeartbeat(
            name="researcher-0000", state="working", turn_id="t-1", tokens=tokens, turns=3
        ),
        AgentHeartbeat(name="researcher-0001", state="idle", idle_since=clock()),
    ]
    assert (
        await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))
    ).status_code == 200
    a = (await client.get("/agents/researcher-0000")).json()
    assert a["state"] == "working" and a["turn_id"] == "t-1" and a["worker"] == W0
    assert a["cost_usd"] == 22.5 and a["turns"] == 3 and a["model"] == "claude-opus-5-5"
    page = (await client.get("/swarms/demo/agents", params={"state": "idle"})).json()
    assert [x["name"] for x in page["agents"]] == ["researcher-0001"]
    st = (await client.get("/swarms/demo")).json()["status"]
    assert st["cost_usd"] == 22.5
    assert st["types"]["researcher"]["workers_by_state"] == {"running": 1}

    clock.advance(100)
    assert (await client.get("/agents/researcher-0000")).json()["state"] == "unknown"
    clock.advance(100)
    assert (await client.get("/agents/researcher-0000")).json()["state"] == "dead"
    workers = (await client.get("/workers", params={"swarm": "demo"})).json()
    assert workers[0]["state"] == "dead" and workers[0]["agents"] == [
        "researcher-0000",
        "researcher-0001",
    ]
    assert (await client.get(f"/workers/{W0}")).json()["id"] == W0

    unknown = hb_body(clock, "nope", [])
    assert (await client.post("/workers/nope/heartbeat", json=unknown)).status_code == 404


async def test_agents_pagination(client, tick):
    await apply(client, make_spec(researcher={"replicas": 5}))
    tick()
    names, cursor = [], None
    while True:
        params = {"limit": 2, **({"cursor": cursor} if cursor else {})}
        page = (await client.get("/swarms/demo/agents", params=params)).json()
        names += [a["name"] for a in page["agents"]]
        cursor = page["cursor"]
        if not cursor:
            break
    assert names == [f"researcher-000{i}" for i in range(5)]


async def test_events_append_and_long_poll(client, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()

    def batch(*types):
        evs = [Event(type=t, ts=clock(), agent="researcher-0000") for t in types]
        return {"worker": W0, "events": [e.model_dump(mode="json") for e in evs]}

    r = await client.post(f"/workers/{W0}/events", json=batch("session_started", "turn_started"))
    assert r.json()["accepted"] == 2
    page = (await client.get("/agents/researcher-0000/events")).json()
    assert [e["type"] for e in page["events"]] == ["session_started", "turn_started"]
    assert page["next"] == 2
    page = (await client.get("/agents/researcher-0000/events", params={"after": 1})).json()
    assert [e["type"] for e in page["events"]] == ["turn_started"] and page["next"] == 2

    poll = asyncio.create_task(
        client.get("/agents/researcher-0000/events", params={"after": 2, "wait": 5})
    )
    await asyncio.sleep(0.05)
    assert not poll.done()
    await client.post(f"/workers/{W0}/events", json=batch("turn_ended"))
    page = (await asyncio.wait_for(poll, 2)).json()
    assert [e["type"] for e in page["events"]] == ["turn_ended"] and page["next"] == 3
    empty = (await client.get("/agents/researcher-0000/events", params={"after": 3})).json()
    assert empty == {"events": [], "next": 3}
    assert (await client.get("/agents/nobody/events")).status_code == 404


async def test_status_page_renders(client, store, clock, tick):
    await apply(
        client,
        make_spec(
            researcher={"replicas": 2},
            reviewer={"replicas": 1, "harness": "codex", "model": "gpt-5.5-codex"},
        ),
    )
    tick()
    await client.post("/agents/researcher-0000/prompt", json={"text": "x"})
    r = await client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    html = r.text
    assert "<h2>demo</h2>" in html and "reviewer" in html and "researcher" in html
    assert "1 pending prompts" in html and 'href="/agents/researcher-0001"' in html


async def test_delete_swarm(client, store, clock, tick):
    await apply(client)
    tick()
    assert (await client.delete("/swarms/demo")).status_code == 202
    tick()
    clock.advance(121)
    tick()
    assert (await client.get("/swarms/demo")).status_code == 404
    assert (await client.delete("/swarms/demo")).status_code == 404
