"""Pause / resume holds (docs/loop.md "Pause and resume") and the automatic hold while the
world is down (docs/health.md), on the fake controller app and a fake clock."""

from __future__ import annotations

import httpx

from legion.controller.api import create_app
from legion.controller.health import Health
from legion.controller.reconcile import swarm_stream
from legion.models import SwarmSpec

from . import test_loop_controller
from .conftest import make_spec
from .test_health import FakeWorld
from .test_loop_controller import A0, A1, prompts

sim = test_loop_controller.sim  # the fixture


async def pause(sim, **body) -> dict:
    r = await sim.c.post("/swarms/demo/pause", json=body)
    assert r.status_code == 200, r.text
    return r.json()["held"]


async def resume(sim, **body) -> dict:
    r = await sim.c.post("/swarms/demo/resume", json=body)
    assert r.status_code == 200, r.text
    return r.json()["held"]


async def test_hold_defers_rounds_prompts_and_wakes_without_counting(sim):
    await sim.apply({"prompt": "r{round}", "every": "30s", "on": ["message_board.mention"]})
    await sim.heartbeat()
    sim.tick()
    r1 = prompts(await sim.commands())
    assert set(r1) == {A0, A1}
    held = await pause(sim, types=["researcher"], reason="maintenance")
    assert held["types"]["researcher"]["reason"] == "maintenance"
    await sim.turn(A0, r1[A0], tool_calls=0)  # the turn in flight ends under the hold
    await sim.turn(A1, r1[A1], tool_calls=0)
    assert sim.loops.wake(A0, "message_board.mention")  # queued, not lost
    await sim.c.post(f"/agents/{A1}/prompt", json={"text": "by hand"})
    for _ in range(4):
        sim.clock.advance(30)
        await sim.heartbeat()  # idle and fresh: only the hold keeps them unprompted
        sim.tick()
        assert prompts(await sim.commands()) == {}  # no round, no hand prompt, no wake
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["rounds_total"] == 2
    status = (await sim.c.get("/swarms/demo")).json()["status"]
    assert status["held"]["types"]["researcher"]["by"] == "operator"

    assert await resume(sim) == {"types": {}, "agents": {}}
    assert (await sim.c.get("/swarms/demo")).json()["status"]["held"] is None
    got = prompts(await sim.commands())
    assert got[A1]["text"] == "by hand"  # queued during the hold, delivered now
    await sim.heartbeat()  # fresh: the agents are idle
    sim.tick()
    got = prompts(await sim.commands())
    assert got[A0]["loop"]["why"] == "wake: message_board.mention" and got[A0]["loop"]["round"] == 1
    kinds = [(k, d) for k, d in sim.swarm_events() if k in ("paused", "resumed")]
    assert kinds == [("paused", {"types": ["researcher"], "agents": [], "reason": "maintenance",
                                 "by": "operator", "auto": False, "interrupt": False}),
                     ("resumed", {"types": ["researcher"], "agents": [], "by": "operator",
                                  "auto": False})]


async def test_agent_hold_with_interrupt_stops_only_that_agent(sim):
    await sim.apply({"prompt": "go", "every": "30s"})
    await sim.heartbeat(working=(A0, A1))
    await pause(sim, agents=[A0], interrupt=True, reason="look")
    cmds = await sim.commands()
    assert [(c["kind"], c["agent"]) for c in cmds] == [("stop", A0)]
    r = await sim.c.post("/swarms/demo/pause", json={"agents": ["nobody"], "reason": "x"})
    assert r.status_code == 404
    await pause(sim, types=["researcher"], reason="look")
    r = await sim.c.post("/swarms/demo/resume", json={"agents": [A0, A1]})
    assert r.json()["not_lifted"] == {A0: "still held by its type", A1: "still held by its type"}
    r = await sim.c.post("/swarms/demo/resume", json={"types": ["researcher"]})
    assert r.json()["not_lifted"] == {}
    r = await sim.c.post("/swarms/demo/resume", json={"agents": [A0]})
    assert r.json()["not_lifted"] == {A0: "not held"}


async def test_interrupt_ends_a_worker_side_loop_before_the_stop(sim):
    await sim.apply({"prompt": "go"})  # turn_end: the worker re-prompts
    await sim.heartbeat()
    sim.tick()
    assert {c["kind"] for c in await sim.commands()} == {"loop_start"}
    await sim.heartbeat(working=(A0,))
    await pause(sim, agents=[A0], interrupt=True, reason="look")
    cmds = await sim.commands()
    assert [(c["kind"], c["agent"], c.get("reason")) for c in cmds] == [
        ("loop_stop", A0, "held"), ("stop", A0, None)]
    sim.tick()
    assert await sim.commands() == []  # the tick does not send a second loop_stop


async def test_budget_stop_and_raise_keep_the_holds(sim):
    await sim.apply({"prompt": "go", "every": "30s"}, limits={"budget_usd": 1})
    await pause(sim, types=["researcher"], reason="r")
    sim.cost[A0] = 10_000_000
    await sim.heartbeat()
    sim.tick()
    assert sim.store.swarm_flags("demo")["budget_stop"] == "hard"
    await sim.apply({"prompt": "go", "every": "30s"}, limits={"budget_usd": 100_000})
    sim.tick()
    flags = sim.store.swarm_flags("demo")
    assert "budget_stop" not in flags and set(flags["holds"]["types"]) == {"researcher"}


async def test_worker_loop_is_stopped_by_a_hold_and_restarted_with_its_rounds(sim):
    await sim.apply({"prompt": "go", "while": {"max_rounds": 9}})
    await sim.heartbeat()
    sim.tick()
    assert {c["kind"] for c in await sim.commands()} == {"loop_start"}
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": 1, "why": "turn_end"}),
                     ("turn_started", {}), ("turn_ended", {"status": "completed"}))
    await pause(sim, reason="r")
    sim.tick()
    cmds = await sim.commands()
    assert sorted((c["kind"], c["reason"]) for c in cmds) == [("loop_stop", "held")] * 2
    sim.tick(60)
    assert await sim.commands() == []  # sent once, no stall re-start while held
    for a in (A0, A1):
        await sim.events(a, ("loop_stopped", {"reason": "held", "round": 1}))
    await resume(sim)
    sim.tick()
    starts = {c["agent"]: c for c in await sim.commands()}
    assert starts[A0]["kind"] == "loop_start" and starts[A0]["meta"]["round"] == 1


def world_types_spec(**world) -> SwarmSpec:
    data = make_spec(researcher={"replicas": 1}, writer={"replicas": 1, "world": False},
                     tester={"replicas": 1}).model_dump(mode="json")
    data["world"] = {"url": "http://world", "endpoint": "http://world", **world,
                     "manifest": {"world_api": "v1", "name": "w", "version": "1",
                                  "tools": [{"name": "open_pr"}]}}
    return SwarmSpec.model_validate(data)


async def test_world_down_holds_world_types_after_the_threshold_and_resumes_its_own(store, clock):
    T = clock().timestamp()
    store.put_swarm(world_types_spec(), T)
    world = FakeWorld()
    h = Health(store, secrets={"LEGION_WORLD_DOWN_PAUSE_S": "600"},
               transport=httpx.MockTransport(world))
    app = create_app(store, "tok", clock=clock, loop_interval=0, health=h)
    loops = app.state.loops
    loops.pause("demo", types=["tester"], reason="ops")  # an operator hold stays through it all
    world.up = False
    await h.probe(T)
    await h.probe(T + 30)
    assert h.down_since["demo"] == T
    await h.probe(T + 570)
    assert "researcher" not in (store.swarm_flags("demo").get("holds") or {}).get("types", {})
    await h.probe(T + 600)  # down 10 min since the first failed probe
    held = store.swarm_flags("demo")["holds"]["types"]
    assert set(held) == {"researcher", "tester"} and held["researcher"]["auto"]
    assert held["tester"]["reason"] == "ops"  # not replaced by the automatic hold
    world.up = True
    await h.probe(T + 630)
    assert set(store.swarm_flags("demo")["holds"]["types"]) == {"tester"}
    evs = [(e["type"], e["data"]) for e in store.read_events(swarm_stream("demo"), 0)
           if e["type"] in ("paused", "resumed")]
    assert evs[1:] == [
        ("paused", {"types": ["researcher"], "agents": [], "reason": "world_down",
                    "by": "controller", "auto": True, "interrupt": False}),
        ("resumed", {"types": ["researcher"], "agents": [], "by": "controller", "auto": True})]


async def test_world_down_pause_off_per_swarm(store, clock):
    T = clock().timestamp()
    store.put_swarm(world_types_spec(down_pause_s=0), T)
    assert "down_pause_s" in store.get_swarm("demo")["spec"].world.model_dump()
    world = FakeWorld()
    world.up = False
    h = Health(store, transport=httpx.MockTransport(world))
    create_app(store, "tok", clock=clock, loop_interval=0, health=h)
    for i in range(5):
        await h.probe(T + i * 3600)
    assert "holds" not in store.swarm_flags("demo")


