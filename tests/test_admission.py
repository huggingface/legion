"""Fair `max_working` admission: FIFO across workers, slots released on turn end (run-1k
finding 1), and the 1k-shaped simulation in tests/admission_sim.py."""

from __future__ import annotations

import asyncio
import time
from datetime import timedelta

import httpx

from flock.controller.api import create_app
from flock.models import AgentHeartbeat, Event, WorkerHeartbeat

from .admission_sim import simulate
from .conftest import TOKEN
from .test_providers_controller import CAP_PROVIDERS, pspec, put

W0, W1 = "p-worker-w0000", "p-worker-w0001"


def _spec():
    return pspec(CAP_PROVIDERS, worker={"harness": "codex", "model": "p:test", "replicas": 2,
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


async def test_run1k_shaped_simulation_every_agent_gets_a_turn():
    """The run-1k shape at a quarter of its size (see admission_sim.py): 3 providers x
    max_working 32, 253 agents on 11 workers. Before this change the full 1,000-agent shape
    reached 410 of 1,000 agents in 60 rounds (stuck at 384 for the first 15), with 87 of 128
    slots in use on average; this one reached 133 of 253 in 40 rounds, with 22 of 32 in use."""
    res = await simulate(rounds=40, max_working=32, agents=253)
    assert res.agents == 253
    assert res.all_by is not None and res.all_by <= 30, res.coverage
    assert max(res.peak_in_turn.values()) <= 32  # the cap holds exactly here
    assert min(res.mean_in_turn.values()) >= 25
