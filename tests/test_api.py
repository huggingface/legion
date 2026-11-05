from __future__ import annotations

import asyncio
from datetime import timedelta

from flock.controller.reconcile import FAST_END
from flock.models import AgentHeartbeat, Event, TokenUsage, WorkerHeartbeat

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


async def test_pending_prompt_delivered_on_new_workers_first_poll(client, store, jobs, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    # Worker crashed: prompt is held.
    jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
    clock.advance(61)
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
    # opus 5.5: 1M input at $4 + 0.1M output at $20
    assert a["cost_usd"] == 6.0 and a["turns"] == 3 and a["model"] == "claude-opus-5-5"
    page = (await client.get("/swarms/demo/agents", params={"state": "idle"})).json()
    assert [x["name"] for x in page["agents"]] == ["researcher-0001"]
    st = (await client.get("/swarms/demo")).json()["status"]
    assert st["cost_usd"] == 6.0
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
    assert '<div class="v">1</div><div class="l">pending prompts</div>' in html
    assert 'href="/agents/researcher-0001"' in html and 'class="chip ' in html


async def test_delete_swarm(client, store, clock, tick):
    await apply(client)
    tick()
    assert (await client.delete("/swarms/demo")).status_code == 202
    tick()
    clock.advance(121)
    tick()
    assert (await client.get("/swarms/demo")).status_code == 404
    assert (await client.delete("/swarms/demo")).status_code == 404


async def test_parked_poll_of_a_finished_job_gets_nothing_from_its_replacement(
    client, store, jobs, clock, tick
):
    """e2e run 1: a drained job's last long-poll was still parked on the controller when the
    planned relaunch queued the new job's assigns. It woke, took two assigns and a held prompt
    to a client that had exited, and the ack deleted them. Polls are now bound to the job's
    incarnation, so the parked poll gets 409 and the replacement gets everything."""
    await apply(client, make_spec(researcher={"replicas": 2}))
    tick()
    old = jobs.launches[-1]["env"]["WORKER_INCARNATION"]
    store.clear_commands(W0)
    agents = [AgentHeartbeat(name=f"researcher-000{i}", state="idle") for i in range(2)]
    r = await client.post(
        f"/workers/{W0}/heartbeat",
        params={"incarnation": old},
        json=hb_body(clock, W0, agents, draining=True),
    )
    assert r.status_code == 200, r.text
    ack = (await client.post("/agents/researcher-0001/prompt", json={"text": "held"})).json()
    assert ack["status"] == "pending"

    parked = asyncio.create_task(
        client.get(f"/workers/{W0}/commands", params={"wait": 5, "incarnation": old})
    )
    await asyncio.sleep(0.05)
    assert not parked.done()
    jobs.set_stage(jobs.job_for(W0), "COMPLETED")  # the drained worker exited
    clock.advance(FAST_END)  # after a run of normal length
    tick()  # planned restart: new job, new incarnation, assigns queued
    new = jobs.launches[-1]["env"]["WORKER_INCARNATION"]
    assert new != old and len(jobs.launches) == 2
    r = await asyncio.wait_for(parked, 6)
    assert r.status_code == 409

    batch = (
        await client.get(f"/workers/{W0}/commands", params={"wait": 0, "incarnation": new})
    ).json()["commands"]
    assert [(c["kind"], c["agent"]) for c in batch] == [
        ("assign", "researcher-0000"),
        ("assign", "researcher-0001"),
        ("prompt", "researcher-0001"),
    ]
    # the old process cannot heartbeat into the new job's slot either
    r = await client.post(
        f"/workers/{W0}/heartbeat", params={"incarnation": old}, json=hb_body(clock, W0, agents)
    )
    assert r.status_code == 409


async def test_heartbeat_telemetry_reaches_the_worker_and_agent_status(client, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    body = hb_body(clock, W0, [AgentHeartbeat(name="researcher-0000", state="idle",
                                              rev="r2", start_rev="r1")])
    body.update(mem={"current_mb": 5000, "limit_mb": 11264, "nonreclaim_mb": 3000,
                     "slope_mb_h": 120, "eta_limit_s": 82000, "psi_some_avg10": 0.5,
                     "oom_kills": 0},
                restore={"agents": 2, "files_done": 10, "files_total": 40,
                         "started": "2026-10-28T10:00:00Z"},
                tools={"open_pr": {"n": 3, "err": 1, "refused": 1, "timeouts": 0,
                                   "sum_s": 4.5, "max_s": 3.0}},
                knobs_rev="abc")
    assert (await client.post(f"/workers/{W0}/heartbeat", json=body)).status_code == 200
    w = (await client.get(f"/workers/{W0}")).json()
    assert w["mem"]["nonreclaim_mb"] == 3000 and w["mem"]["eta_limit_s"] == 82000
    assert w["restore"]["files_total"] == 40 and w["tools"]["open_pr"]["refused"] == 1
    assert w["knobs_rev"] == "abc"
    assert (await client.get("/workers", params={"swarm": "demo"})).json()[0]["knobs_rev"] == "abc"
    a = (await client.get("/agents/researcher-0000")).json()
    assert (a["rev"], a["start_rev"]) == ("r2", "r1")
    # a worker that sends none of it clears the last values
    await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, []))
    w = (await client.get(f"/workers/{W0}")).json()
    assert w["mem"] is None and w["tools"] is None and w["knobs_rev"] is None


async def test_heartbeat_missing_assigned_agents_requeues_their_assign(client, store, clock, tick):
    await apply(client, make_spec(researcher={"replicas": 2}))
    tick()
    store.clear_commands(W0)  # as if both assigns were delivered, one of them into the void
    agents = [AgentHeartbeat(name="researcher-0000", state="idle")]
    r = await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))
    assert r.status_code == 200
    batch = (await client.get(f"/workers/{W0}/commands", params={"wait": 0})).json()["commands"]
    assert [(c["kind"], c["agent"]) for c in batch] == [("assign", "researcher-0001")]
    # already queued or reported: nothing more
    r = await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))
    r = await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents))
    assert len(store.commands_for(W0)) == 1
    # a draining worker ignores assigns, so none are sent to it
    store.clear_commands(W0)
    await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, agents, draining=True))
    assert store.commands_for(W0) == []


async def test_healthz_reports_requests_and_process(client, tick):
    await apply(client)
    tick()
    assert (await client.get("/agents/researcher-0000")).status_code == 200
    await client.get("/agents/nobody")
    h = (await client.get("/healthz")).json()
    routes = h["requests"]["routes"]
    assert routes["GET /agents/{name}"]["n"] == 2
    assert routes["PUT /swarms/{swarm}"]["n"] == 1
    assert h["requests"]["by_status"]["4xx"] == 1
    assert h["process"]["rss_mb"] > 0 and h["process"]["cpu_s"] > 0


async def test_heartbeat_during_restore_does_not_double_count_cost(client, clock, tick):
    """1k run: after a relaunch the worker heartbeats agents whose restore has not finished yet,
    with an empty token map. That read as "counters went backwards" and folded the old totals
    into the base; once restored, the worker reported its cumulative totals again (they survive
    in the bucket state), so the agent's cost was counted twice (+$0.67 on $2.00)."""
    await apply(client, make_spec(researcher={"replicas": 1}))
    tick()
    t1 = {"claude-opus-5-5": TokenUsage(input=1_000_000)}  # $4
    t2 = {"claude-opus-5-5": TokenUsage(input=1_500_000)}  # $6, cumulative across workers
    for tokens in (t1, {}, t2):  # before the relaunch, while restoring, after the restore
        hb = AgentHeartbeat(name="researcher-0000", state="idle", idle_since=clock(), tokens=tokens)
        r = await client.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, [hb]))
        assert r.status_code == 200
        clock.advance(30)
    a = (await client.get("/agents/researcher-0000")).json()
    assert a["cost_usd"] == 6.0
    assert a["tokens"]["claude-opus-5-5"]["input"] == 1_500_000
