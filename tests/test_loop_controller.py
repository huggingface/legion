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
                                 turn_id="t-1" if n in working else None,
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
                   tool_calls: int = 1, **ended) -> None:
        """The worker delivered `prompt` (its loop meta) and ran one turn."""
        kinds = []
        if prompt is not None and prompt.get("loop"):
            m = prompt["loop"]
            kinds.append(("loop_prompt", {"kind": m["kind"], "round": m["round"], "why": m["why"]}))
        kinds += [("turn_started", {}), ("turn_ended", {"status": status, "tool_calls": tool_calls, **ended})]
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
        "cap_usd": None, "why": "start", "add_tools": [], "counted": True, "wake": False}
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
        "idle_streak": 0, "wakes": 0}
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
        # A1's prompt is deferred on the worker (draining): no turn_ended ever comes for it
        await sim.events(A1, ("prompt_deferred", {"reason": "draining"}))
        sim.tick(29)
        assert list(prompts(await sim.commands())) == [A0]  # A0's round 2
        sim.tick(2)  # queued again after 30 s, behind A0's slot (max_working 1)
        [row] = [r for r in store.prompts_for_worker(W0) if r["agent"] == A1]
        assert json.loads(row["body"])["loop"]["round"] == 2


async def test_max_turns_is_not_a_failed_turn_but_is_logged(sim):
    """Defect 5: a controller-side loop counts real errors (3 in a row end the agent's loop)
    but not turns that ended at max_turns; each of those is a `loop_max_turns` event."""
    await sim.apply({"prompt": "p", "every": "5s"})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for _ in range(4):
        await sim.turn(A0, got[A0], status="error", reason="max_turns", num_turns=8)
        sim.tick(5)
        got = prompts(await sim.commands())
        assert A0 in got
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"][A0]
    assert (st["phase"], st["errors"]) == ("looping", 0)
    hits = [d for k, d in sim.swarm_events() if k == "loop_max_turns"]
    assert hits == [{"type": "researcher", "kind": "loop", "round": r, "calls": 8} for r in (1, 2, 3, 4)]
    assert [e["type"] for e in sim.store.read_events(A0, 0)].count("loop_max_turns") == 4

    for i in range(3):  # real errors still count
        await sim.turn(A0, got[A0], status="error", reason="api_error")
        sim.tick(5)
        got = prompts(await sim.commands())
        assert (A0 in got) == (i < 2)
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"][A0]
    assert (st["phase"], st["done_reason"], st["errors"]) == ("done", "errors", 3)


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
    # the queue as before `wake:` existed: no gap, no cap, wakes are rounds
    await sim.apply({"prompt": "p {why}", "every": "1h", "on": ["comms.mention"],
                     "wake": {"min_gap": "0s", "max": None, "counts_as_round": True}})
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


async def test_comms_events_wake_the_loop(sim):
    """The worker's comms events reach the wake queue (`on_comms` -> `wake(agent, comms.<reason>)`):
    two mentions of A0 give it one prompt; a thread reply to A1 (not in `on:`) gives none."""
    await sim.apply({"prompt": "p {why}", "every": "1h", "on": ["comms.mention"]})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    msg = {"msg_id": 7, "thread": 7, "from": "carlos"}
    await sim.events(A0, ("comms_mention", msg), ("comms_mention", {**msg, "msg_id": 8}))
    await sim.events(A1, ("comms_thread_reply", msg))
    sim.tick(2.1)
    got = prompts(await sim.commands())
    assert list(got) == [A0] and got[A0]["loop"]["why"] == "wake: comms.mention"
    # by default a wake is not a round: it carries round 1 again and leaves the count alone
    assert got[A0]["loop"]["round"] == 1 and got[A0]["loop"]["counted"] is False
    await sim.turn(A0, got[A0])
    sim.tick(3)
    assert await sim.commands() == []
    wake = [d for k, d in sim.swarm_events() if k == "loop_wake"]
    assert wake == [{"type": "researcher", "agent": A0, "reason": "comms.mention", "events": 2,
                     "coalesced": 1, "round": 1, "counted": False, "wakes": 1}]
    assert [e["type"] for e in sim.store.read_events(A0, 0)].count("loop_wake") == 1
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["rounds_total"] == 2 and st["wakes_total"] == 1


async def test_hard_stopped_swarm_ignores_comms_wakes(sim):
    await sim.apply({"prompt": "p {why}", "every": "1h", "on": ["comms.mention"]},
                    limits={"budget_usd": 1.0})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    sim.store.put_swarm_flags("demo", {"budget_stop": "hard", "budget_usd": 1.0, "at": 0})
    assert not sim.loops.wake(A0, "comms.mention")  # the loop has not seen the stop yet
    await sim.events(A0, ("comms_mention", {"msg_id": 7, "thread": 7, "from": "carlos"}))
    sim.tick(2.1)
    assert not [c for c in await sim.commands() if c["kind"] == "prompt"]


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


async def test_stagger_spreads_the_first_prompts(sim):
    await sim.apply({"first": "boot", "prompt": "p", "every": "1h", "stagger": "4s"})
    await sim.heartbeat()
    sim.tick()
    assert list(prompts(await sim.commands())) == [A0]
    sim.tick(3)
    assert await sim.commands() == []
    await sim.heartbeat()  # keep A1 fresh on the fake clock
    sim.tick(1)
    got = prompts(await sim.commands())
    assert list(got) == [A1] and got[A1]["text"] == "boot"


async def test_type_budget_usd_stops_the_loop(sim):
    await sim.apply({"prompt": "p", "every": "5s", "while": {"budget_usd": 0.6},
                     "final": {"prompt": "fin"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    sim.cost.update({A0: 15_000, A1: 15_000})  # $0.30 each
    await sim.heartbeat()
    sim.tick(5)
    fin = prompts(await sim.commands())
    assert {p["loop"]["why"] for p in fin.values()} == {"final: budget_usd"}


# --------------------------------------------------------------------------- after review (PR #15)


def provider_spec(loop: dict, max_working: int | None = None) -> dict:
    prov = {"type": "openai_compat", "base_url": "http://unused/v1", "api_key_env": "K",
            "self_hosted": True, **({"max_working": max_working} if max_working else {})}
    return {"swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh", "providers": {"p": prov},
            "types": {"researcher": {"harness": "codex", "model": "p:m", "replicas": 2,
                                     "agents_per_worker": 2, "loop": loop}}}


@pytest.mark.parametrize("change", ["max_working", "every"])
async def test_live_change_moves_the_loop_between_worker_and_controller(store, clock, tick, change):
    """Finding 1: a provider gaining max_working, or `every` leaving turn_end, takes the
    cadence away from the workers (loop_stop first), and the controller sends the rounds."""
    app = create_app(store, TOKEN, clock=clock, loop_interval=0, enforce_max_working=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        sim = Sim(store, clock, tick, c)
        sim.loops = app.state.loops
        loop = {"prompt": "go", "every": "turn_end"}
        assert (await c.put("/swarms/demo", json=provider_spec(loop))).status_code == 200
        sim.reconcile()
        await sim.commands()
        await sim.heartbeat()
        sim.tick()
        assert {x["kind"] for x in await sim.commands()} == {"loop_start"}
        body = (provider_spec(loop, max_working=1) if change == "max_working"
                else provider_spec({**loop, "every": "30s"}))
        assert (await c.put("/swarms/demo", json=body)).status_code == 200
        sim.tick()
        assert {(x["kind"], x["reason"]) for x in await sim.commands()} == {("loop_stop", "mode_change")}
        st = (await c.get("/swarms/demo/loop")).json()[0]
        assert st["mode"] == "controller" and st["every"] == ("turn_end" if change == "max_working" else "30s")
        sim.tick(31)
        assert await sim.commands() == []  # no worker loop_start again, and no prompt before the ack
        for a in (A0, A1):
            await sim.events(a, ("loop_stopped", {"reason": "mode_change", "round": 0, "active": True}))
        sim.tick()
        got = prompts(await sim.commands())
        assert got and all(p["loop"]["kind"] == "loop" for p in got.values())
        if change == "max_working":
            assert len(got) == 1 and sum(store.pending_counts("demo").values()) == 1  # admitted one at a time
        back = provider_spec(loop)  # and back to the workers: queued controller rounds are dropped
        assert (await c.put("/swarms/demo", json=back)).status_code == 200
        sim.tick()
        assert store.pending_counts("demo") == {}
        assert (await c.get("/swarms/demo/loop")).json()[0]["mode"] == "worker"


async def test_hard_stop_survives_a_worker_relaunch(sim, jobs):
    """Finding 2: a worker crashing before its `stop` arrived is relaunched without its resume
    prompt running, and an agent seen working again under the stop gets `stop` again."""
    await sim.apply(None, limits={"budget_usd": 0.5}, resume_prompt="continue after crash")
    sim.cost[A0] = 25_000
    await sim.heartbeat(working=(A0,))
    sim.tick()
    assert sim.loops.hard_stopped("demo")
    jobs.set_stage(jobs.job_for(W0), "ERROR", "crash before the stop arrived")
    for dt in (61, 600, 61):
        sim.clock.advance(dt)
        sim.reconcile()
    cmds = await sim.commands()
    assert [x["kind"] for x in cmds] == ["assign", "assign"]
    assert all(x["resume_prompt"] is None for x in cmds)  # held, not run inside the assign
    assert [r["agent"] for r in sim.store.prompts_for_worker(W0)] == [A0]
    sim.tick()
    assert await sim.commands() == []  # and not delivered while the budget is exhausted
    await sim.heartbeat(working=(A1,))  # something runs anyway: stop it again
    sim.tick()
    assert [(x["kind"], x["agent"]) for x in await sim.commands()] == [("stop", A1)]
    sim.tick(5)
    assert await sim.commands() == []  # at most every 30 s
    sim.tick(30)
    assert [(x["kind"], x["agent"]) for x in await sim.commands()] == [("stop", A1)]


@pytest.mark.parametrize("case", ["first", "last_round", "final"])
async def test_a_deferred_loop_prompt_is_sent_again(sim, case):
    """Finding 3: the worker took a first / last-round / final prompt, then deferred it
    (draining): the same prompt goes again, not counted as another round, and the loop ends."""
    lp = {"prompt": "go", "every": "1s", "while": {"max_rounds": 1}, "final": {"prompt": "finish"}}
    if case == "first":
        lp["first"] = "setup"
    await sim.apply(lp)
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    if case == "final":
        for a in (A0, A1):
            await sim.turn(a, got[a])
        sim.tick()
        got = prompts(await sim.commands())
    kind = got[A0]["loop"]["kind"]
    for a in (A0, A1):
        m = got[a]["loop"]
        await sim.events(a, ("loop_prompt", {"kind": m["kind"], "round": m["round"]}),
                         ("prompt_deferred", {"reason": "draining",
                                              "loop": {"kind": m["kind"], "round": m["round"]}}))
    sim.clock.advance(60)
    await sim.heartbeat()
    sim.tick()
    again = prompts(await sim.commands())
    assert {(p["loop"]["kind"], p["loop"]["round"], p["text"]) for p in again.values()} == {
        (kind, got[A0]["loop"]["round"], got[A0]["text"])}
    assert set(again) == {A0, A1}
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["rounds_max"] <= 1
    # now the turns run: the loop goes on to its end
    for _ in range(4):
        for a, p in again.items():
            await sim.turn(a, p)
        sim.tick(1)
        again = prompts(await sim.commands())
        if not again:
            break
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert (st["state"], st["rounds_max"], st["final_done"]) == ("done", 1, 2)


async def test_removing_final_during_the_final_round_ends_it(sim):
    """Finding 5: a re-apply without `final` while the final round is in flight."""
    await sim.apply({"prompt": "round", "every": "30s", "final": {"prompt": "final"}})
    await sim.heartbeat()
    sim.tick()
    r = await sim.c.post("/swarms/demo/loop", json={"type": "researcher", "action": "stop"})
    assert r.json()["state"] == "final"
    await sim.apply({"prompt": "round", "every": "30s"})
    sim.tick()
    sim.tick(10)
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["state"] == "done"
    assert sim.swarm_events()[-1][1]["note"] == "final removed from the spec"


async def test_handover_to_the_workers_rolls_back_dropped_rounds(store, clock, tick, tmp_path):
    """Re-review: capped turn_end with max_rounds 1, round 1 queued for both agents, then
    max_working removed before any command poll. The queued rounds are dropped, so they were
    never rounds: the workers start from round 0 and each sends exactly one prompt."""
    from flock.models import LoopStartCommand

    from .worker_fakes import runtime, until

    app = create_app(store, TOKEN, clock=clock, loop_interval=0, enforce_max_working=True)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        sim = Sim(store, clock, tick, c)
        sim.loops = app.state.loops
        loop = {"prompt": "go {round}", "every": "turn_end", "while": {"max_rounds": 1}}
        assert (await c.put("/swarms/demo", json=provider_spec(loop, max_working=1))).status_code == 200
        sim.reconcile()
        await sim.commands()
        await sim.heartbeat()
        sim.tick()
        assert store.pending_counts("demo") == {A0: 1, A1: 1}  # round 1 queued, not yet polled
        assert (await c.put("/swarms/demo", json=provider_spec(loop))).status_code == 200
        sim.tick()
        assert store.pending_counts("demo") == {}
        starts = {x["agent"]: x for x in await sim.commands()}
        assert {x["kind"] for x in starts.values()} == {"loop_start"}
        assert {(x["meta"]["round"], x["max_rounds"]) for x in starts.values()} == {(0, 1)}
        st = (await c.get("/swarms/demo/loop")).json()[0]
        assert (st["mode"], st["rounds_total"], st["agents_done"]) == ("worker", 0, 0)

        rt, ad = runtime(tmp_path, A0, idle_grace="5s")  # the worker side of that command
        await rt.setup(restore=False)
        rt.start_loop(LoopStartCommand.model_validate(starts[A0]))
        await until(lambda: [e for e in list(rt.outbox._q) if e.type == "loop_stopped"], timeout=5)
        assert ad.sent == ["go 1"]
        await rt.stop()


async def test_removing_final_drops_its_queued_prompts(sim):
    """Third pass: finals queued, `final` removed before any worker fetched them, a tick: the
    queued finals are dropped (a command poll delivers nothing); one already fetched by its
    worker is not sent again."""
    await sim.apply({"prompt": "round", "every": "30s", "final": {"prompt": "submit final"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    r = await sim.c.post("/swarms/demo/loop", json={"type": "researcher", "action": "stop"})
    assert r.json()["state"] == "final"
    sim.tick()
    assert sim.store.pending_counts("demo") == {A0: 1, A1: 1}  # finals queued, not fetched
    spec = make_spec(researcher={"replicas": 2, "agents_per_worker": 2,
                                 "loop": {"prompt": "round", "every": "30s"}})  # final removed
    assert (await sim.c.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
    sim.tick()
    assert await sim.commands() == []
    assert sim.store.pending_counts("demo") == {}
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert (st["state"], st["final_sent"]) == ("done", 0)
    assert sim.swarm_events()[-1][1]["dropped"] == 2


async def test_finish_never_resends_a_fetched_final(sim):
    await sim.apply({"prompt": "round", "every": "30s", "final": {"prompt": "submit final"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    await sim.c.post("/swarms/demo/loop", json={"type": "researcher", "action": "stop"})
    sim.tick()
    fin = prompts(await sim.commands())  # both fetched by the worker
    assert set(fin) == {A0, A1}
    await sim.turn(A0, fin[A0])  # A0 ran it; A1's is still running
    await sim.apply({"prompt": "round", "every": "30s"})
    sim.tick()
    sim.tick(60)
    assert await sim.commands() == []
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert (st["state"], st["final_done"]) == ("done", 1)


async def test_until_idle_ignores_a_dormant_looping_type(sim):
    """`flock run --until idle` (localrun.Tracker): a `replicas: 0` type with a loop and no
    agents never runs a round, so it must not hold a finished run open. A type whose replicas
    are not created yet, or one just spawned into, still counts (review of PR #17)."""
    from flock.controller.api import swarm_status
    from flock.controller.localrun import Tracker, loops_settled

    lp = {"prompt": "work", "every": "30s", "while": {"max_rounds": 1}}
    spec = make_spec(researcher={"replicas": 2, "agents_per_worker": 2, "loop": lp},
                     unused={"replicas": 0, "loop": lp})
    r = await sim.c.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    sim.loops.tick()
    now = sim.clock().timestamp()
    assert not loops_settled(spec, swarm_status(sim.store, spec, now))  # replicas not created yet
    sim.reconcile()
    await sim.commands()
    await sim.heartbeat()
    sim.tick()
    for a, p in prompts(await sim.commands()).items():
        await sim.turn(a, p)
    tracker = Tracker(sim.store, "demo")
    for _ in range(3):
        sim.clock.advance(60)
        await sim.heartbeat()
        sim.tick()
    assert {k: v["state"] for k, v in sim.store.get_loops("demo").items()} == {
        "researcher": "done", "unused": "running"}
    snap = tracker.snapshot(spec, sim.clock().timestamp())
    assert snap["completed"] == 2 and snap["pending"] == 0
    assert snap["loops_settled"] and tracker.idle(snap)

    # spawned into: the type has an agent now, and its running loop counts again
    sim.store.insert_child("researcher-0000-unused-1", "demo", "unused", 1, A0, 1,
                           sim.clock().timestamp())
    snap = tracker.snapshot(spec, sim.clock().timestamp())
    assert not snap["loops_settled"] and not tracker.idle(snap)


async def test_wake_min_gap_coalesces_and_max_caps(sim):
    """Defaults: a wake is not a round, two wake prompts are at least 30 s apart (wakes in
    between become one prompt when the gap ends), and `wake.max` caps them per run. Each wake
    prompt is a `loop_wake` event (swarm and agent stream) and counts in `wakes_total`."""
    await sim.apply({"prompt": "p {why}", "every": "1h", "on": ["comms.mention"],
                     "while": {"max_rounds": 2}, "wake": {"max": 2}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])  # round 1 each
    assert sim.loops.wake(A0, "comms.mention")
    sim.tick(2.1)
    w1 = prompts(await sim.commands())
    assert list(w1) == [A0] and (w1[A0]["loop"]["round"], w1[A0]["loop"]["counted"]) == (1, False)
    await sim.turn(A0, w1[A0])
    for i in range(3):  # a thread's worth of chatter inside the gap
        assert sim.loops.wake(A0, f"comms.mention:{i}")
        sim.tick(5)
    sim.tick(10)  # 25 s after the first wake prompt: still held
    assert await sim.commands() == []
    sim.tick(5.5)
    w2 = prompts(await sim.commands())
    assert list(w2) == [A0] and w2[A0]["loop"]["round"] == 1
    assert w2[A0]["loop"]["why"] == "wake: comms.mention:0, comms.mention:1, comms.mention:2"
    await sim.turn(A0, w2[A0])
    assert not sim.loops.wake(A0, "comms.mention")  # wake.max: 2 reached for this run
    sim.tick(60)
    assert await sim.commands() == []
    wakes = [d for k, d in sim.swarm_events() if k == "loop_wake"]
    assert [(d["agent"], d["events"], d["coalesced"], d["wakes"]) for d in wakes] == [
        (A0, 1, 0, 1), (A0, 3, 2, 2)]
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert (st["state"], st["rounds_total"], st["wakes_total"]) == ("running", 2, 2)
    assert st["per_agent"][A0]["phase"] == "looping"  # two wakes used no rounds


async def test_wake_window_cap_and_deferred_wake(sim):
    """`wake.max` per `wake.window`; a wake prompt the worker deferred is taken back from the
    wake count and never from the round count."""
    await sim.apply({"prompt": "p", "every": "1h", "on": ["comms.mention"],
                     "wake": {"min_gap": "0s", "max": 1, "window": "1m"}})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for a in (A0, A1):
        await sim.turn(a, got[a])
    assert sim.loops.wake(A0, "comms.mention")
    sim.tick(2.1)
    w = prompts(await sim.commands())[A0]["loop"]
    await sim.events(A0, ("loop_prompt", {"kind": "loop", "round": w["round"], "why": w["why"],
                                          "counted": False}),
                     ("prompt_deferred", {"reason": "draining",
                                          "loop": {"kind": "loop", "round": 1, "counted": False}}))
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"][A0]
    assert (st["rounds"], st["wakes"]) == (1, 0)
    sim.tick()
    assert await sim.commands() == []  # no cadence round in its place
    assert sim.loops.wake(A0, "comms.mention")  # under the cap again
    sim.tick(2.1)
    w = prompts(await sim.commands())[A0]
    assert w["loop"]["why"] == "wake: comms.mention" and w["loop"]["round"] == 1
    await sim.turn(A0, w)
    assert not sim.loops.wake(A0, "comms.mention")  # 1 per minute
    sim.tick(61)
    assert sim.loops.wake(A0, "comms.mention")
    sim.tick(2.1)
    assert list(prompts(await sim.commands())) == [A0]


async def test_idle_rounds_ignore_read_only_calls(sim):
    """A loop turn whose calls are all read-only (`state_calls: 0`, e.g. a board read) is idle;
    a turn with one state-changing call is not."""
    await sim.apply({"prompt": "p", "every": "5s", "while": {"idle_rounds": 2}})
    await sim.heartbeat()
    sim.tick()
    for _ in range(2):
        got = prompts(await sim.commands())
        await sim.turn(A0, got[A0], tool_calls=3, state_calls=0)
        await sim.turn(A1, got[A1], tool_calls=3, state_calls=1)
        sim.tick(5)
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"]
    assert (st[A0]["phase"], st[A0]["done_reason"]) == ("done", "idle_rounds")
    assert (st[A1]["phase"], st[A1]["idle_streak"]) == ("looping", 0)


async def test_finished_agents_leave_only_under_stop_when(sim, monkeypatch):
    """`done(finished: true)` (turn_ended `finished`) takes the agent out of the loop only when
    `stop_when.all_agents_done` is set; the loop stops (`done`) once every agent has."""
    monkeypatch.setattr("flock.controller.api.loop_refusals", lambda spec: [])  # llm-only key
    await sim.apply({"prompt": "p", "every": "5s"})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    await sim.turn(A0, got[A0], finished=True)
    await sim.turn(A1, got[A1])
    sim.tick(5)
    got = prompts(await sim.commands())
    assert set(got) == {A0, A1}  # `finished` not asked for: ignored
    await sim.apply({"prompt": "p", "every": "5s", "stop_when": {"all_agents_done": True},
                     "final": {"prompt": "fin"}})
    await sim.turn(A0, got[A0], finished=True)
    await sim.turn(A1, got[A1])
    sim.tick(5)
    assert list(prompts(await sim.commands())) == [A1]  # A0 is out
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert st["per_agent"][A0]["done_reason"] == "done" and st["state"] == "running"
    await sim.turn(A1, got[A1], finished=True)
    sim.tick()
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]
    assert (st["state"], st["reason"]) == ("final", "done")


@pytest.mark.parametrize("how", ["deferred", "dropped"])
async def test_a_counted_wake_taken_back_restores_wake_and_round(sim, how):
    """Review of #19: with `counts_as_round: true` and `wake.max: 1`, a wake prompt the worker
    deferred, or one dropped while still queued, gives back both its round and its wake, so
    the next wake is accepted and is round 2 again."""
    await sim.apply({"prompt": "round {round}", "every": "100s", "on": ["comms.mention"],
                     "wake": {"max": 1, "counts_as_round": True}})
    await sim.heartbeat()
    sim.tick()
    for a, p in prompts(await sim.commands()).items():
        await sim.turn(a, p)
    sim.clock.advance(50)
    await sim.heartbeat()
    assert sim.loops.wake(A0, "comms.mention")
    sim.tick(2.1)
    if how == "deferred":
        w = prompts(await sim.commands())[A0]["loop"]
        assert (w["round"], w["counted"], w["wake"]) == (2, True, True)
        await sim.events(A0, ("loop_prompt", w), ("prompt_deferred", {
            "reason": "draining", "loop": {"kind": "loop", "round": 2, "wake": True}}))
    else:  # still queued for the worker when the loop drops its prompts
        with sim.loops._lock:
            assert sim.loops._drop("demo", "researcher") == 1
    st = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"][A0]
    assert (st["rounds"], st["wakes"]) == (1, 0)
    assert sim.loops.wake(A0, "comms.mention")  # the allowance is back
    sim.tick(2.1)
    w = prompts(await sim.commands())[A0]["loop"]
    assert (w["round"], w["why"], w["wake"]) == (2, "wake: comms.mention", True)


async def test_transport_stalls_do_not_exhaust_controller_loop(sim):
    await sim.apply({"prompt": "p", "every": "5s"})
    await sim.heartbeat()
    sim.tick()
    got = prompts(await sim.commands())
    for reason in ("model_transport", "timeout") * 3:
        await sim.turn(A0, got[A0], status="error", reason=reason)
        sim.tick(5)
        got = prompts(await sim.commands())
        assert A0 in got
    state = (await sim.c.get("/swarms/demo/loop")).json()[0]["per_agent"][A0]
    assert (state["phase"], state["errors"]) == ("looping", 0)
