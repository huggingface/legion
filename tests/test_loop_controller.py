"""Controller side of `loop:` (docs/loop.md), driven tick by tick on a fake clock: first /
loop / final ordering, `every` as a duration and as turn_end (worker- and controller-side),
`while` conditions, the wake queue's debounce, `flock loop --stop/--start` and the
`limits.budget_usd` thresholds. The worker is simulated by posting its heartbeats and events."""

from __future__ import annotations

import json

import httpx
import pytest

from flock.controller.api import create_app
from flock.models import AgentHeartbeat, SwarmSpec, TokenUsage

from .conftest import TOKEN, make_spec
from .test_api import W0, hb_body

A0, A1 = "researcher-0000", "researcher-0001"


class Sim:
    """The app on a fake clock plus a pretend worker W0 hosting both agents."""

    def __init__(self, store, clock, tick, client) -> None:
        self.store, self.clock, self.reconcile, self.c = store, clock, tick, client
        self.loops = None
        self.cost = {A0: 0, A1: 0}  # output tokens, for heartbeats

    async def apply(self, loop: dict | None = None, limits: dict | None = None, **type_kw):
        types = {"researcher": {"replicas": 2, "agents_per_worker": 2,
                                **({"loop": loop} if loop else {}), **type_kw}}
        spec = make_spec(**types)
        if limits:
            spec.limits = spec.limits.model_validate(limits)
        r = await self.c.put("/swarms/demo", json=spec.model_dump(mode="json"))
        assert r.status_code == 200, r.text
        self.reconcile()
        await self.commands()  # drop the assigns
        return r

    async def heartbeat(self, working: tuple[str, ...] = ()) -> None:
        agents = [AgentHeartbeat(name=n, state="working" if n in working else "idle",
                                 tokens={"claude-opus-5-5": TokenUsage(output=self.cost[n])})
                  for n in (A0, A1)]
        r = await self.c.post(f"/workers/{W0}/heartbeat", json=hb_body(self.clock, W0, agents))
        assert r.status_code == 200, r.text

    async def commands(self) -> list[dict]:
        r = await self.c.get(f"/workers/{W0}/commands", params={"wait": 0})
        return r.json()["commands"]

    async def events(self, agent: str, *kinds: tuple[str, dict]) -> None:
        """Post worker custom events (pi `custom` entries) for `agent`."""
        evs = [{"type": k, "ts": self.clock().isoformat(), "agent": agent,
                "data": {"type": "custom", "customType": k, "data": d}} for k, d in kinds]
        r = await self.c.post(f"/workers/{W0}/events", json={"worker": W0, "events": evs})
        assert r.status_code == 200, r.text

    async def turn(self, agent: str, prompt: dict | None = None, status: str = "completed",
                   tool_calls: int = 1) -> None:
        """The worker delivered `prompt` (its loop meta) and ran one turn."""
        kinds = []
        if prompt is not None and prompt.get("loop"):
            m = prompt["loop"]
            kinds.append(("loop_prompt", {"kind": m["kind"], "round": m["round"], "why": m["why"]}))
        kinds += [("turn_started", {}), ("turn_ended", {"status": status, "tool_calls": tool_calls})]
        await self.events(agent, *kinds)

    def tick(self, seconds: float = 0) -> None:
        if seconds:
            self.clock.advance(seconds)
        self.loops.tick()
        assert all(n <= 1 for n in self.store.pending_counts("demo").values())  # never stacked

    def swarm_events(self) -> list[tuple[str, dict]]:
        return [(e["type"], e["data"]) for e in self.store.read_events("_swarm.demo", 0)]


@pytest.fixture
async def sim(store, clock, tick):
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        s = Sim(store, clock, tick, c)
        s.loops = app.state.loops
        yield s


def prompts(cmds: list[dict]) -> dict[str, dict]:
    out = {}
    for c in cmds:
        if c["kind"] == "prompt":
            assert c["agent"] not in out, "two prompts for one agent in one batch"
            out[c["agent"]] = c
    return out


async def test_duration_loop_first_rounds_then_final(sim):
    await sim.apply({"first": "set up", "prompt": "round {round}", "every": "30s",
                     "while": {"max_rounds": 2}, "final": {"prompt": "wrap up", "add_tools": ["submit"]}})
    sim.tick()
    assert await sim.commands() == []  # no heartbeat yet: unknown agents are not prompted
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert set(got) == {A0, A1} and got[A0]["text"] == "set up"
    assert got[A0]["loop"]["kind"] == "first" and got[A0]["loop"]["round"] == 0
    sim.tick()
    assert await sim.commands() == []  # handed out: working as far as the controller knows

    # A0 finishes its first turn; A1 is still in it. Only A0 gets round 1, and only after it.
    await sim.turn(A0, got[A0])
    sim.tick()
    r1 = prompts(await sim.commands())
    assert list(r1) == [A0] and r1[A0]["text"] == "round {round}"  # rendered on the worker
    assert r1[A0]["loop"] | {"started_at": None} == {
        "kind": "loop", "type": "researcher", "round": 1, "started_at": None, "wall_s": None,
        "cap_usd": None, "why": "start", "add_tools": []}
    await sim.turn(A1, got[A1])
    await sim.turn(A0, r1[A0])
    sim.tick(10)
    r = prompts(await sim.commands())
    assert list(r) == [A1] and r[A1]["loop"]["round"] == 1  # A0: 30 s not up yet
    await sim.turn(A1, r[A1])
    sim.tick(21)
    r2 = prompts(await sim.commands())
    assert list(r2) == [A0] and r2[A0]["loop"] | {"started_at": None} == {
        **r1[A0]["loop"], "started_at": None, "round": 2, "why": "every 30s"}
    await sim.turn(A0, r2[A0])
    sim.tick(10)
    r = prompts(await sim.commands())
    assert list(r) == [A1] and r[A1]["loop"]["round"] == 2
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["state"] == "running" and st["per_agent"][A0] == {
        "phase": "done", "rounds": 2, "done_reason": "max_rounds", "final": None, "errors": 0,
        "idle_streak": 0}
    sim.tick()
    assert await sim.commands() == []  # A0 waits for A1: the final round is type-wide

    await sim.turn(A1, r[A1])
    sim.tick()
    fin = prompts(await sim.commands())
    assert set(fin) == {A0, A1} and fin[A0]["text"] == "wrap up"
    assert fin[A0]["loop"]["kind"] == "final" and fin[A0]["loop"]["why"] == "final: max_rounds"
    assert fin[A0]["loop"]["add_tools"] == ["submit"] and fin[A0]["loop"]["round"] == 2
    for a in (A0, A1):
        await sim.turn(a, fin[a])
    sim.tick()
    sim.tick(600)
    assert await sim.commands() == []  # then idle
    st = (await sim.c.get("/swarms/demo")).json()["status"]["types"]["researcher"]["loop"]
    assert (st["state"], st["reason"], st["final_sent"], st["final_done"]) == ("done", "max_rounds", 2, 2)
    kinds = [k for k, _ in sim.swarm_events()]
    assert kinds == ["loop_started", *["loop_tick"] * 5, "loop_stopped", "final_round",
                     "final_round", "final_done"]
    assert dict(sim.swarm_events())["loop_stopped"]["reason"] == "max_rounds"
    agent_kinds = [e["type"] for e in sim.store.read_events(A0, 0)]
    assert agent_kinds.index("final_round") > max(
        i for i, k in enumerate(agent_kinds) if k == "loop_prompt" and i < agent_kinds.index("final_round"))


async def test_turn_end_runs_on_the_worker(sim):
    await sim.apply({"prompt": "go {round}", "while": {"max_rounds": 3}, "final": {"prompt": "fin"}})
    await sim.heartbeat()
    sim.tick()
    cmds = await sim.commands()
    assert {c["kind"] for c in cmds} == {"loop_start"}
    start = {c["agent"]: c for c in cmds}
    assert start[A0]["text"] == "go {round}" and start[A0]["max_rounds"] == 3
    assert start[A0]["meta"]["round"] == 0 and start[A0]["meta"]["why"] == "turn_end"
    sim.tick()
    assert await sim.commands() == []  # the worker owns the cadence now
    for a in (A0, A1):  # the worker's own re-prompts, reported as events
        for n in (1, 2, 3):
            await sim.events(a, ("loop_prompt", {"kind": "loop", "round": n, "why": "turn_end"}),
                             ("turn_started", {}), ("turn_ended", {"status": "completed"}))
    sim.tick()
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["mode"] == "worker" and st["rounds_total"] == 6 and st["state"] == "running"
    await sim.events(A0, ("loop_stopped", {"reason": "max_rounds", "round": 3}))
    sim.tick()
    assert await sim.commands() == []  # A1's worker loop has not said it stopped
    await sim.events(A1, ("loop_stopped", {"reason": "max_rounds", "round": 3}))
    sim.tick()
    fin = prompts(await sim.commands())
    assert set(fin) == {A0, A1} and fin[A0]["loop"]["why"] == "final: max_rounds"


async def test_worker_loop_is_resent_when_it_stalls(sim):
    await sim.apply({"prompt": "go", "while": {"max_rounds": 9}})
    await sim.heartbeat()
    sim.tick()
    assert len(await sim.commands()) == 2
    sim.tick(29)
    assert await sim.commands() == []
    sim.tick(2)  # idle, nothing heard for 30 s: the loop_start was lost (or a new worker)
    assert [c["kind"] for c in await sim.commands()] == ["loop_start", "loop_start"]


async def test_turn_end_under_max_working_runs_on_the_controller(store, clock, tick):
    from .test_providers_controller import CAP_PROVIDERS

    app = create_app(store, TOKEN, clock=clock, loop_interval=0, enforce_max_working=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        sim = Sim(store, clock, tick, c)
        sim.loops = app.state.loops
        spec = SwarmSpec.model_validate({
            "swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh",
            "providers": CAP_PROVIDERS, "types": {"researcher": {
                "replicas": 2, "agents_per_worker": 2, "harness": "codex", "model": "p:test",
                "loop": {"prompt": "go", "while": {"max_rounds": 2}}}}})
        assert (await c.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
        sim.reconcile()
        await sim.commands()
        await sim.heartbeat()
        sim.tick()
        st = (await c.get("/swarms/demo/loop")).json()[0]
        assert st["mode"] == "controller"
        got = prompts(await sim.commands())
        assert list(got) == [A0]  # max_working 1: admission holds A1's round 1
        assert store.pending_counts("demo") == {A1: 1}
        await sim.turn(A0, got[A0])
        sim.tick()  # A0's turn ended: its round 2 is due at once, but A1 is first in line
        got = prompts(await sim.commands())
        assert list(got) == [A1] and got[A1]["loop"]["round"] == 1


async def test_while_until_stops_and_runs_final(sim):
    await sim.apply({"prompt": "p", "every": "10s", "while": {"until": "60s"},
                     "final": {"prompt": "fin"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    assert got[A0]["loop"]["wall_s"] == 60
    await sim.turn(A0, got[A0])
    await sim.turn(A1, got[A1])
    sim.tick(61)
    fin = prompts(await sim.commands())
    assert {p["loop"]["kind"] for p in fin.values()} == {"final"}
    assert fin[A0]["loop"]["why"] == "final: until"


async def test_agent_usd_and_idle_rounds_take_agents_out(sim):
    # claude-opus-5-5 output: $20 per 1M tokens, so 25,000 tokens = $0.50
    await sim.apply({"prompt": "p", "every": "5s", "while": {"agent_usd": 0.5, "idle_rounds": 2}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a], tool_calls=0)
    sim.cost[A0] = 25_000
    await sim.heartbeat()
    sim.tick(5)
    got = prompts(await sim.commands())
    assert list(got) == [A1]  # A0 reached agent_usd
    await sim.turn(A1, got[A1], tool_calls=0)  # second idle loop turn in a row
    sim.tick(5)
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["per_agent"][A0]["done_reason"] == "agent_usd"
    assert st["per_agent"][A1]["done_reason"] == "idle_rounds"
    assert st["state"] == "done" and st["reason"] == "agents_done"


async def test_budget_thresholds_final_reserve_then_hard_stop(sim):
    # $1 budget, $0.20 held for the final round: loops stop at $0.80, everything at $1
    await sim.apply({"prompt": "p", "every": "5s", "final": {"prompt": "fin", "reserve_usd": 0.2}},
                    limits={"budget_usd": 1.0})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    r = await sim.c.post(f"/agents/{A0}/prompt", json={"text": "manual"})
    assert r.status_code == 200
    sim.cost.update({A0: 20_000, A1: 20_000})  # $0.40 each: $0.80
    await sim.heartbeat()
    sim.tick(5)
    kinds = [k for k, _ in sim.swarm_events()]
    assert kinds[-4:] == ["budget_stop", "loop_stopped", "final_round", "final_round"]
    stop = dict(sim.swarm_events())["budget_stop"]
    assert stop == {"phase": "final_reserve", "cost_usd": 0.8, "budget_usd": 1.0, "threshold_usd": 0.8}
    fin = prompts(await sim.commands())  # the manual prompt was dropped; only finals go out
    assert {a: p["text"] for a, p in fin.items()} == {A0: "fin", A1: "fin"}
    assert fin[A0]["loop"]["why"] == "final: budget"
    assert (await sim.c.get("/swarms/demo")).json()["status"]["budget_stop"] == "final_reserve"

    sim.cost[A0] = 30_000  # $1.00
    await sim.heartbeat()
    sim.tick()
    cmds = await sim.commands()
    assert sorted((c["kind"], c["agent"]) for c in cmds) == [("stop", A0), ("stop", A1)]
    assert dict(sim.swarm_events())["budget_stop"]["phase"] == "hard"
    r = await sim.c.post("/swarms/demo/prompts", json={"type": "researcher", "text": "x"})
    assert r.status_code == 409 and "budget" in r.text
    assert (await sim.c.post(f"/agents/{A1}/prompt", json={"text": "x"})).status_code == 409
    r = await sim.c.post("/swarms/demo/loop", json={"type": "researcher", "action": "start"})
    assert r.status_code == 409

    await sim.apply({"prompt": "p", "every": "5s", "final": {"prompt": "fin", "reserve_usd": 0.2}},
                    limits={"budget_usd": 5.0})  # more room: the stop is lifted
    sim.tick()
    assert (await sim.c.get("/swarms/demo")).json()["status"]["budget_stop"] is None
    assert (await sim.c.post(f"/agents/{A1}/prompt", json={"text": "x"})).status_code == 200


async def test_budget_without_reserve_is_one_hard_stop(sim):
    await sim.apply(None, limits={"budget_usd": 0.5})
    sim.cost[A0] = 25_000
    await sim.heartbeat()
    sim.tick()
    assert [e for e, _ in sim.swarm_events()] == ["budget_stop"]
    assert {c["kind"] for c in await sim.commands()} == {"stop"}


async def test_wake_queue_debounces_and_skips_pending(sim):
    await sim.apply({"prompt": "p {why}", "every": "1h", "on": ["comms.mention"]})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    assert not sim.loops.wake(A0, "comms.thread")  # not in `on`
    assert sim.loops.wake(A0, "comms.mention")
    sim.tick(1)
    assert sim.loops.wake(A0, "comms.mention:42")  # a burst of mentions...
    sim.tick(1.5)
    assert await sim.commands() == []  # ...is held until 2 s of quiet
    sim.tick(0.6)
    got = prompts(await sim.commands())
    assert list(got) == [A0] and got[A0]["loop"]["why"] == "wake: comms.mention, comms.mention:42"
    assert got[A0]["loop"]["round"] == 2
    # wakes while the prompt is out (working) wait for idle, and give one prompt, not several
    for _ in range(3):
        sim.loops.wake(A0, "comms.mention")
        sim.tick(3)
    assert await sim.commands() == []
    await sim.turn(A0, got[A0])
    sim.tick()
    got = prompts(await sim.commands())
    assert list(got) == [A0] and got[A0]["loop"]["round"] == 3
    # a steady stream of wakes is not held forever: at most 10 s
    await sim.turn(A0, got[A0])
    for _ in range(12):
        sim.loops.wake(A0, "comms.mention")
        sim.tick(1)
    assert list(prompts(await sim.commands())) == [A0]


async def test_cli_loop_stop_start_and_status(sim, monkeypatch):
    from fastapi.testclient import TestClient
    from typer.testing import CliRunner

    from flock import cli

    await sim.apply({"prompt": "p", "every": "5s", "final": {"prompt": "fin"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    app = sim.c._transport.app
    monkeypatch.setattr(cli, "client", lambda: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    run = CliRunner()
    r = run.invoke(cli.app, ["status", "demo"])
    assert r.exit_code == 0 and "loop researcher: running (controller, every 5s), round 1, 0/2" in r.output
    r = run.invoke(cli.app, ["loop", "demo", "--type", "researcher", "--stop", "--no-final"])
    assert r.exit_code == 0 and "loop researcher: done" in r.output and "stopped: manual" in r.output
    sim.tick(10)
    assert await sim.commands() == []  # stopped without a final round
    r = run.invoke(cli.app, ["loop", "demo", "--type", "researcher", "--start"])
    assert r.exit_code == 0 and "running" in r.output, r.output
    sim.tick()
    got = prompts(await sim.commands())
    assert {p["loop"]["round"] for p in got.values()} == {1}  # a new run of the loop: from round 1
    r = run.invoke(cli.app, ["loop", "demo"])
    assert r.exit_code == 0 and A0 in r.output and "looping" in r.output
    assert run.invoke(cli.app, ["loop", "demo", "--stop"]).exit_code == 1  # needs --type
    r = run.invoke(cli.app, ["events", "demo"])
    assert [json.loads(x)["type"] for x in r.output.splitlines()][:2] == ["loop_started", "loop_tick"]


async def test_live_change_and_removal(sim):
    loop = {"prompt": "v1", "while": {"max_rounds": 5}}
    await sim.apply(loop)
    await sim.heartbeat()
    sim.tick()
    assert {c["text"] for c in await sim.commands()} == {"v1"}
    await sim.apply({**loop, "prompt": "v2"})  # re-applied: workers get the new prompt
    sim.tick()
    cmds = await sim.commands()
    assert {(c["kind"], c["text"]) for c in cmds} == {("loop_start", "v2")}
    assert "loop_updated" in [k for k, _ in sim.swarm_events()]
    await sim.apply(None)  # loop removed from the spec
    sim.tick()
    assert {c["kind"] for c in await sim.commands()} == {"loop_stop"}
    assert sim.swarm_events()[-1] == ("loop_stopped", {"type": "researcher", "reason": "loop_removed"})
    assert (await sim.c.get("/swarms/demo/loop")).json() == []
