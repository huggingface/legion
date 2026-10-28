"""Policies, digest and steering (docs/policies.md, docs/live.md, docs/loop.md): rule
evaluation on scripted snapshots, drains through the rotation budget, the digest, the ramp,
tracked broadcasts, the loop banner and `types.T.max_working`."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest

from flock.controller.api import create_app
from flock.controller.health import Health
from flock.controller.policy import Policies, digest
from flock.controller.reconcile import record_heartbeat, swarm_stream
from flock.models import (
    AgentHeartbeat,
    DrainCommand,
    SwarmSpec,
    WorkerHeartbeat,
    WorkerMem,
)

from .conftest import TOKEN, make_spec
from .test_api import W0, hb_body
from .test_loop_controller import A0, Sim, prompts


def spec_with(policies=None, report=None, **types) -> SwarmSpec:
    data = make_spec(**types).model_dump(mode="json")
    data["policies"], data["report"] = policies, report
    return SwarmSpec.model_validate(data)


def events(store, *kinds):
    return [(e["type"], e["data"]) for e in store.read_events(swarm_stream("demo"), 0)
            if e["type"] in kinds]


def body(world=None, **workers):
    """A snapshot body: workers as wid=(nonreclaim_mb of 1000, heartbeat time)."""
    return {"world": world, "types": {}, "tools": {},
            "workers": {w: {"hb": hb, "mem": {"nonreclaim_mb": mb, "limit_mb": 1000}}
                        for w, (mb, hb) in workers.items()}}


MEM = {"name": "mem", "metric": "worker.nonreclaim_frac", "op": ">", "value": 0.9, "for": "60s",
       "clear": 0.8, "do": "notify"}


def test_for_hysteresis_and_stale_guard(store):
    spec = spec_with({"rules": [MEM]})
    store.put_swarm(spec, 0)
    p = Policies(store, secrets={})
    run = lambda t, mb, hb: p.evaluate(spec, body(w=(mb, hb)), t, 30)
    assert run(0, 950, 1) == []  # 1 of 2 fresh samples
    assert run(30, 950, 1) == []  # stale heartbeat: does not count
    assert [k for k, _ in run(60, 950, 2)] == ["alert", "policy_action"]
    assert run(90, 850, 3) == []  # under value, over clear: still firing
    assert [k for k, _ in run(120, 750, 4)] == ["alert_cleared"]
    assert events(store, "alert")[0][1] == {"rule": "mem", "target": "w", "value": 0.95}
    assert events(store, "policy_action")[0][1]["mode"] == "done"  # notify always runs
    assert run(150, 950, 5) == [] and run(180, 960, 5) == []  # restarted count, stale again


class FakeWorldAdmin:
    def __init__(self):
        self.calls = []

    def __call__(self, swarm, method, path, body):
        self.calls.append((method, path, body))
        return {"changes": [{"name": k, "old": None, "new": v} for k, v in body.items()]} \
            if method == "PATCH" else {}


def slow_rule(**kw):
    return {"name": "slow", "metric": "world.manifest_ms", "op": ">", "value": 1000,
            "do": {"world_knob": {"pause_pushes": True}},
            "clear_do": {"world_knob": {"pause_pushes": False}}, **kw}


def world(ms, t):
    return {"up": True, "manifest_ms": ms, "ts": t}


def test_world_knob_cooldown_per_hour_and_budget(store):
    spec = spec_with({"mode": "enforce", "rules": [slow_rule(cooldown="10m", max_per_hour=3)]})
    store.put_swarm(spec, 0)
    p = Policies(store, secrets={"FLOCK_POLICY_MAX_ACTIONS_PER_HOUR": "2"})
    p.world_admin = admin = FakeWorldAdmin()
    t = 0.0

    def run(ms):
        nonlocal t
        t += 30
        return [(k, d.get("mode")) for k, d in p.evaluate(spec, body(world(ms, t)), t, 30)]

    assert run(5000) == [("alert", None), ("policy_action", "done")]
    assert admin.calls == [("PATCH", "/admin/knobs", {"pause_pushes": True})]
    assert run(10) == [("alert_cleared", None), ("policy_action", "done")]  # clear_do
    assert run(5000) == [("alert", None)]  # cooldown: 10 min since the last enforced episode
    for _ in range(17):
        assert run(5000) == []
    assert run(5000) == [("policy_action", "done")]  # cooldown over, still firing
    run(10)
    t += 600
    assert run(5000) == [("alert", None), ("policy_budget_exhausted", None),
                         ("policy_action", "would_do")]  # global budget of 2 per hour
    assert len([c for c in admin.calls if c[2] == {"pause_pushes": True}]) == 2
    assert run(10)[0] == ("alert_cleared", None)
    assert events(store, "world_knob")[0][1]["by"] == "policy"


def test_alert_mode_and_unenforced_actions_are_would_do(store):
    class Loops:
        def __init__(self):
            self.calls = []

        def pause(self, swarm, **kw):
            self.calls.append(("pause", kw))

        def resume(self, swarm, **kw):
            self.calls.append(("resume", kw))

    rules = [{"name": "gone", "metric": "world.up", "op": "==", "value": 0, "do": "pause_type",
              "clear_do": "resume_type"},
             {"name": "talk", "metric": "world.up", "op": "==", "value": 0, "mode": "enforce",
              "do": {"prompt": {"text": "world is down"}}}]
    spec = spec_with({"rules": rules})
    store.put_swarm(spec, 0)
    p = Policies(store, secrets={})
    p.loops = loops = Loops()
    out = p.evaluate(spec, body({"up": False, "ts": 1}), 1, 30)
    assert [(k, d.get("action"), d.get("mode")) for k, d in out] == [
        ("alert", None, None), ("policy_action", "pause_type", "would_do"),
        ("alert", None, None), ("policy_action", "prompt", "would_do")]  # prompt: alert-only
    assert loops.calls == []
    spec = spec_with({"mode": "enforce", "rules": rules[:1]})
    p.evaluate(spec, body({"up": True, "ts": 2}), 2, 30)
    p.evaluate(spec, body({"up": False, "ts": 3}), 3, 30)
    p.evaluate(spec, body({"up": True, "ts": 4}), 4, 30)
    # the alert-mode episode's clear_do was not enforced
    assert loops.calls == [("pause", {"types": [], "reason": "gone", "by": "policy", "auto": True}),
                           ("resume", {"types": [], "by": "policy", "auto_only": True})]


def test_mute(store):
    spec = spec_with({"rules": [MEM]})
    store.put_swarm(spec, 0)
    p = Policies(store, secrets={})
    p.mute("demo", "mem", until=100)
    assert p.evaluate(spec, body(w=(950, 1)), 0, 60) == []
    assert p.view(spec, 0)["rules"][0]["firing"] == ["w"]  # state keeps updating
    p.evaluate(spec, body(w=(750, 2)), 50, 60)
    assert [k for k, _ in p.evaluate(spec, body(w=(950, 3)), 101, 60)] == ["alert", "policy_action"]


def test_builtin_rules_without_a_block():
    from flock.controller.policy import rules_of

    mode, rules = rules_of(make_spec())
    assert mode == "alert" and [r.name for r in rules] == ["worker-mem"]


def test_bad_rules_are_refused():
    for bad in ({"rules": [{**MEM, "metric": "worker.cpu"}]},
                {"rules": [{**MEM, "do": "reboot"}]},
                {"rules": [MEM, MEM]},
                {"sinks": ["file:/etc/x"]}):
        with pytest.raises(ValueError):
            spec_with(bad)


# ---------------------------------------------------------------- e2e: drains

def _workers(store, jobs, clock, tick, n=4):
    store.put_swarm(make_spec(researcher={"replicas": 2 * n, "agents_per_worker": 2}), 0)
    tick()
    return [f"demo-researcher-w{i:04d}" for i in range(n)]


def _beat(store, clock, wids, mb):
    for i, wid in enumerate(wids):
        h = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=24),
                            agents=[AgentHeartbeat(name=f"researcher-{j:04d}", state="idle")
                                    for j in (2 * i, 2 * i + 1)],
                            mem=WorkerMem(limit_mb=1000, nonreclaim_mb=mb))
        assert record_heartbeat(store, wid, h, clock())


def _drains(store, wid):
    return [c for c in store.commands_for(wid) if c["body"] == DrainCommand().model_dump_json()]


@pytest.mark.parametrize("mode", ["enforce", "alert"])
def test_rising_memory_drains_within_the_rotation_budget(store, jobs, clock, tick, mode):
    wids = _workers(store, jobs, clock, tick)
    rule = {**MEM, "do": "drain_worker", "max_concurrent": 3}
    sw = store.get_swarm("demo")["spec"].model_dump(mode="json")
    store.update_swarm_spec(SwarmSpec.model_validate({**sw, "policies": {"mode": mode,
                                                                         "rules": [rule]}}))
    h = Health(store, secrets={"FLOCK_MAX_ROTATING": "2"})
    for mb in (500, 700, 920, 950):  # rising; `for: 60s` is 2 samples of 30 s
        clock.advance(30)
        _beat(store, clock, wids, mb)
        h.tick(clock().timestamp())
    drained = [w for w in wids if _drains(store, w)]
    acts = events(store, "policy_action")
    if mode == "alert":
        assert drained == [] and {d["mode"] for _, d in acts} == {"would_do"} and len(acts) == 4
        return
    assert len(drained) == 2  # max_rotating 2, under max_concurrent 3
    rot = events(store, "memory_rotation")
    assert len(rot) == 2 and all(d["rule"] == "mem" for _, d in rot)
    assert [d["mode"] for _, d in acts] == ["done", "done"]
    clock.advance(30)
    _beat(store, clock, wids, 960)
    h.tick(clock().timestamp())
    assert len([w for w in wids if _drains(store, w)]) == 2  # no slot yet: deferred, not dropped


# ---------------------------------------------------------------- digest

def test_digest_text_and_report_sinks(store, clock, tick):
    b = {"world": {"up": True, "manifest_ms": 41.5, "metrics": {"merges_last_hour": 7,
                                                                "opens_last_hour": 30, "x": 1}},
         "workers": {"w1": {"eta_limit_s": 1800}, "w2": {"eta_limit_s": None, "restore": {"files_done": 1}}},
         "types": {"a": {"by_state": {"idle": 3, "working": 2}, "stuck": [{"agent": "a-1"}]}},
         "tokens": {"hour": "2026-10-14T09", "by_type": {"a": {"input": 1200, "output": 30}}},
         "alerts": [{"kind": "stuck", "msg": "1 agents stuck or silent"}]}
    text = digest("demo", b, [(50.0, "policy_action", {"rule": "mem", "target": "w1",
                                                        "action": "drain_worker",
                                                        "mode": "would_do"})], 100.0, 3600)
    assert "world: up, manifest 42 ms" in text
    assert "world/h: merges_last_hour 7, opens_last_hour 30" in text
    assert "workers: 2, restoring 1; nearest memory limit: w1 0.5h" in text
    assert "a: idle 3 working 2, stuck 1" in text and "a in 1,200 out 30" in text
    assert "alert: 1 agents stuck" in text and "mem w1 drain_worker would_do" in text
    long = digest("demo", {**b, "alerts": [{"msg": "x" * 100}] * 100}, [], 100.0, 3600)
    assert len(long.encode()) <= 4096

    store.put_swarm(spec_with(report={"every": "1h", "sinks": ["file:digest.txt"]}), 0)
    h = Health(store, secrets={})
    t = clock().timestamp()
    h.tick(t)
    h.tick(t + 1800)
    assert events(store, "digest") == []
    h.tick(t + 3600)
    [(_, d)] = events(store, "digest")
    assert d["text"].startswith("flock demo")
    assert (store.data_dir / "reports" / "digest.txt").read_text() == d["text"]


# ---------------------------------------------------------------- ramp

def test_replicas_ramp_steps_on_the_clock(store, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 0, "agents_per_worker": 2,
                                          "replicas_ramp": {"start": 2, "add": 3, "every": "10m",
                                                            "max": 7}}), 0)
    reps = lambda: store.get_swarm("demo")["spec"].types["researcher"].replicas
    tick()
    assert reps() == 2 and len(store.list_agents("demo")) == 2
    clock.advance(599)
    tick()
    assert reps() == 2
    clock.advance(1)
    tick()
    assert reps() == 5
    clock.advance(1200)
    tick()
    assert reps() == 7 and len(store.list_agents("demo")) == 7  # capped at max
    assert [d["replicas"] for _, d in events(store, "ramp_step")] == [2, 5, 7]


# ---------------------------------------------------------------- broadcasts, banner, max_working

@pytest.fixture
async def sim(store, clock, tick):
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        s = Sim(store, clock, tick, c)
        s.loops = app.state.loops
        yield s


async def test_tracked_broadcast_counts(sim):
    await sim.apply(replicas=3)
    await sim.heartbeat()
    r = await sim.c.post("/swarms/demo/prompts", json={"type": "researcher", "text": "wrap up",
                                                       "track": True})
    bid = r.json()["prompt_id"]
    assert bid.startswith("b-") and r.json()["queued"] == 2
    got = prompts(await sim.commands())
    assert {p["prompt_id"] for p in got.values()} == {bid}
    await sim.events(A0, ("turn_started", {"prompt_id": bid}), ("turn_ended", {"prompt_id": bid}))
    await sim.events("researcher-0001", ("turn_started", {"prompt_ids": ["x", bid]}))
    [b] = (await sim.c.get("/swarms/demo/broadcasts")).json()
    assert {k: b[k] for k in ("targets", "delivered", "started", "finished")} == {
        "targets": 3, "delivered": 2, "started": 2, "finished": 1}  # 0002 has no worker yet


async def test_banner_on_loop_prompts_until_cleared(sim):
    await sim.apply({"prompt": "round {round}", "every": "30s"})
    await sim.heartbeat()
    r = await sim.c.put("/swarms/demo/notice", json={"type": "researcher",
                                                     "text": "final round at 20:00"})
    assert r.json() == {"researcher": {"text": "final round at 20:00", "since": r.json()[
        "researcher"]["since"]}}
    sim.tick()
    got = prompts(await sim.commands())
    assert got[A0]["text"] == "final round at 20:00\n\nround {round}"
    await sim.c.delete("/swarms/demo/notice", params={"type": "researcher"})
    await sim.turn(A0, got[A0])
    sim.tick(31)
    assert prompts(await sim.commands())[A0]["text"] == "round {round}"


async def test_banner_reaches_worker_side_loops(sim):
    await sim.apply({"prompt": "go on", "every": "turn_end"})
    await sim.heartbeat()
    sim.tick()
    [start] = [c for c in await sim.commands() if c["kind"] == "loop_start" and c["agent"] == A0]
    assert start["text"] == "go on"
    await sim.c.put("/swarms/demo/notice", json={"text": "stop opening PRs"})  # every type
    sim.tick()
    [start] = [c for c in await sim.commands() if c["kind"] == "loop_start" and c["agent"] == A0]
    assert start["text"] == "stop opening PRs\n\ngo on"


async def test_type_max_working(store, clock, tick):
    spec = make_spec(researcher={"replicas": 2, "agents_per_worker": 2, "max_working": 1})
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        assert (await c.put("/swarms/demo", json=spec.model_dump(mode="json"))).status_code == 200
        tick()
        await c.get(f"/workers/{W0}/commands", params={"wait": 0})
        idle = [AgentHeartbeat(name=n, state="idle") for n in ("researcher-0000", "researcher-0001")]
        await c.post(f"/workers/{W0}/heartbeat", json=hb_body(clock, W0, idle))
        await c.post("/swarms/demo/prompts", json={"type": "researcher", "text": "go"})
        cmds = (await c.get(f"/workers/{W0}/commands", params={"wait": 0})).json()["commands"]
        assert [x["agent"] for x in cmds if x["kind"] == "prompt"] == ["researcher-0000"]


async def test_policy_routes_and_mute(sim):
    await sim.apply()
    out = (await sim.c.get("/swarms/demo/policies")).json()
    assert out["builtin"] and out["rules"][0]["name"] == "worker-mem"
    out = (await sim.c.post("/swarms/demo/policies/worker-mem/mute", json={"for": "2h"})).json()
    assert out["rules"][0]["muted"] and out["rules"][0]["muted_until"]
    out = (await sim.c.post("/swarms/demo/policies/worker-mem/unmute", json={})).json()
    assert not out["rules"][0]["muted"]
    assert (await sim.c.post("/swarms/demo/policies/nope/mute", json={})).status_code == 404


async def test_pause_type_uses_automatic_holds(store, clock):
    app = create_app(store, TOKEN, clock=clock, loop_interval=0)
    rule = {"name": "busy", "metric": "types.researcher.idle", "op": ">=", "value": 1,
            "do": {"pause_type": "researcher"}, "clear_do": {"resume_type": "researcher"}}
    spec = spec_with({"mode": "enforce", "rules": [rule]})
    store.put_swarm(spec, 0)
    p = app.state.health.policies
    assert p.loops is app.state.loops
    t = clock().timestamp()
    by_state = lambda n: {"types": {"researcher": {"by_state": {"idle": n}}}}
    p.evaluate(spec, by_state(2), t, 30)
    held = store.swarm_flags("demo")["holds"]["types"]["researcher"]
    assert held["auto"] and held["reason"] == "busy" and held["by"] == "policy"
    app.state.loops.pause("demo", types=["researcher"], reason="operator", by="me")
    p.evaluate(spec, by_state(0), t + 30, 30)  # clear_do: an operator's hold stays
    assert store.swarm_flags("demo")["holds"]["types"]["researcher"]["by"] == "me"
