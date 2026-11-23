"""Fair `max_working` admission: FIFO across workers, slots released on turn end, and a
many-worker simulation."""

from __future__ import annotations

import asyncio
import random
import time
from datetime import timedelta

import httpx

from flock.controller.api import create_app
from flock.controller.reconcile import agent_state
from flock.models import AgentHeartbeat, AgentState, Event, PromptCommand, WorkerHeartbeat

from .conftest import HB, TOKEN
from .test_inference_controller import CAP, pspec, put

W0, W1 = "p-worker-w0000", "p-worker-w0001"


def _spec():
    return pspec(CAP, worker={**HB, "model": "m", "replicas": 2,
                                        "agents_per_worker": 1})


def _client(store, clock):
    app = create_app(store, TOKEN, clock=clock, enforce_max_working=True)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test",
                             headers={"Authorization": f"Bearer {TOKEN}"})


async def _prompts(c, wid, wait=0) -> list[str]:
    cmds = (await c.get(f"/workers/{wid}/commands", params={"wait": wait})).json()["commands"]
    return [x["agent"] for x in cmds if x["kind"] == "prompt"]


async def _hb(c, clock, wid, name, state):
    body = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=1),
                           agents=[AgentHeartbeat(name=name, state=state)]).model_dump(mode="json")
    assert (await c.post(f"/workers/{wid}/heartbeat", json=body)).status_code == 200


async def _events(c, clock, wid, name, *kinds):
    evs = [Event(type=k, ts=clock(), agent=name).model_dump(mode="json") for k in kinds]
    r = await c.post(f"/workers/{wid}/events", json={"worker": wid, "events": evs})
    assert r.status_code == 200


async def _two_idle_workers(c, clock, tick):
    assert (await put(c, _spec())).status_code == 200
    tick()
    for i, wid in enumerate((W0, W1)):
        await _prompts(c, wid)  # take the assign
        await _hb(c, clock, wid, f"worker-000{i}", "idle")
    clock.advance(1)


async def test_free_slot_goes_to_the_oldest_pending_prompt_not_the_first_poller(store, clock, tick):
    async with _client(store, clock) as c:
        await _two_idle_workers(c, clock, tick)
        await c.post("/agents/worker-0001/prompt", json={"text": "first"})  # on w0001
        await c.post("/agents/worker-0000/prompt", json={"text": "second"})  # on w0000
        assert await _prompts(c, W0) == []  # polls first, but its prompt is younger
        assert await _prompts(c, W1) == ["worker-0001"]
        assert await _prompts(c, W0) == []  # the one slot is taken


async def test_turn_end_event_releases_the_slot_before_the_next_heartbeat(store, clock, tick):
    async with _client(store, clock) as c:
        await _two_idle_workers(c, clock, tick)
        await c.post("/agents/worker-0000/prompt", json={"text": "a"})
        await c.post("/agents/worker-0001/prompt", json={"text": "b"})
        assert await _prompts(c, W0) == ["worker-0000"]
        await _events(c, clock, W0, "worker-0000", "turn_started")
        assert await _prompts(c, W1) == []
        # W1's long-poll is parked; the turn_ended event wakes it with the prompt, with no
        # heartbeat in between.
        t0 = time.monotonic()
        poll = asyncio.create_task(_prompts(c, W1, wait=10))
        await asyncio.sleep(0.05)
        await _events(c, clock, W0, "worker-0000", "turn_ended")
        assert await asyncio.wait_for(poll, 5) == ["worker-0001"]
        assert time.monotonic() - t0 < 5


async def test_stale_turn_end_right_after_admission_keeps_the_slot(store, clock, tick):
    async with _client(store, clock) as c:
        await _two_idle_workers(c, clock, tick)
        await c.post("/agents/worker-0000/prompt", json={"text": "a"})
        await c.post("/agents/worker-0001/prompt", json={"text": "b"})
        assert await _prompts(c, W0) == ["worker-0000"]
        # a late event of the previous turn, before the admitted turn started: not a release
        await _events(c, clock, W0, "worker-0000", "turn_ended")
        assert await _prompts(c, W1) == []
        # the admitted turn starts and ends in one batch: released
        await _events(c, clock, W0, "worker-0000", "turn_started", "turn_ended")
        assert await _prompts(c, W1) == ["worker-0001"]


async def test_deferred_prompt_releases_and_early_idle_heartbeat_does_not(store, clock, tick):
    async with _client(store, clock) as c:
        await _two_idle_workers(c, clock, tick)
        await c.post("/agents/worker-0000/prompt", json={"text": "a"})
        await c.post("/agents/worker-0001/prompt", json={"text": "b"})
        assert await _prompts(c, W0) == ["worker-0000"]
        # a heartbeat built before the worker saw the prompt still says idle: slot kept
        await _hb(c, clock, W0, "worker-0000", "idle")
        assert await _prompts(c, W1) == []
        # the worker could not start the turn (draining): released
        await _events(c, clock, W0, "worker-0000", "prompt_deferred")
        assert await _prompts(c, W1) == ["worker-0001"]


async def test_prompts_on_a_dead_worker_do_not_hold_the_window(store, clock, tick):
    async with _client(store, clock) as c:
        await _two_idle_workers(c, clock, tick)
        await c.post("/agents/worker-0000/prompt", json={"text": "old, worker gone"})
        await c.post("/agents/worker-0001/prompt", json={"text": "new"})
        clock.advance(200)  # both silent past DEAD_AFTER; only W1 comes back
        await _hb(c, clock, W1, "worker-0001", "idle")
        assert await _prompts(c, W1) == ["worker-0001"]


async def test_many_workers_fair_admission_every_agent_gets_a_turn(store, clock, tick):
    """3 servers x max_working 4, one type of 8 agents on 2 workers each. Every 3 s round the
    driver prompts each agent the controller lists idle, the workers poll in a fixed order, turns
    last 5-15 rounds, events go out every round and heartbeats every 10 rounds (phased). Each
    server's cap holds, stays near full, and every agent gets a turn."""
    rng = random.Random(0)
    types = {f"painter-{x}": {**HB, "model": "m", "replicas": 8, "agents_per_worker": 4,
                              "inference": {"urls": [f"http://gpu-{x}/v1"], "max_working": 4}}
             for x in "abc"}
    async with _client(store, clock) as c:
        assert (await put(c, pspec(CAP, **types))).status_code == 200
        tick()
        hosted: dict[str, list[str]] = {}
        for a in store.list_agents("p"):
            hosted.setdefault(a["worker"], []).append(a["name"])
        workers = sorted(hosted)
        left: dict[str, int] = {}  # agent -> rounds left in its turn (0: idle)
        turns: dict[str, int] = {}
        outbox: dict[str, list[dict]] = {w: [] for w in workers}

        def event(kind, n):
            return Event(type=kind, ts=clock(), agent=n,
                         turn_id=f"t-{turns.get(n, 0)}").model_dump(mode="json")

        async def heartbeat(wid):
            agents = [AgentHeartbeat(name=n, state="working" if left.get(n) else "idle",
                                     turn_id=f"t-{turns.get(n, 0)}" if left.get(n) else None,
                                     turns=turns.get(n, 0)) for n in hosted[wid]]
            body = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=2),
                                   agents=agents).model_dump(mode="json")
            assert (await c.post(f"/workers/{wid}/heartbeat", json=body)).status_code == 200

        for wid in workers:  # take the assigns, report every agent idle
            await _prompts(c, wid)
            await heartbeat(wid)
        busy, peak, all_by = dict.fromkeys(types, 0), dict.fromkeys(types, 0), None
        for rnd in range(20):
            now = clock.advance(3).timestamp()
            wrows = {w["id"]: w for w in store.list_workers("p")}
            pending, lv = store.pending_counts("p"), store.liveness("p")
            for a in store.list_agents("p"):
                if not pending.get(a["name"]) and \
                        agent_state(a, wrows.get(a["worker"]), now, lv) == AgentState.idle:
                    cmd = PromptCommand(agent=a["name"], prompt_id=f"{a['name']}-{rnd}", text="go")
                    store.add_prompt(a["name"], cmd.model_dump_json(), now)
            for wid in workers:  # a prompt starts a turn unless the agent is in one
                for n in await _prompts(c, wid):
                    if not left.get(n):
                        turns[n] = turns.get(n, 0) + 1
                        left[n] = rng.randint(5, 15)
                        outbox[wid].append(event("turn_started", n))
            for i, wid in enumerate(workers):
                for n in hosted[wid]:
                    if left.get(n):
                        left[n] -= 1
                        if not left[n]:
                            outbox[wid].append(event("turn_ended", n))
                if outbox[wid]:
                    r = await c.post(f"/workers/{wid}/events",
                                     json={"worker": wid, "events": outbox[wid]})
                    assert r.status_code == 200
                    outbox[wid] = []
                if (rnd + i) % 10 == 0:
                    await heartbeat(wid)
            for t in types:
                k = sum(1 for n, v in left.items() if v and n.startswith(t + "-"))
                busy[t] += k
                peak[t] = max(peak[t], k)
            if all_by is None and len(turns) == 24:
                all_by = rnd + 1
    assert all_by is not None and all_by <= 18
    assert max(peak.values()) <= 4  # each server's cap holds exactly
    assert min(busy.values()) >= 3 * 20  # a mean of 3 of 4 in a turn
