"""`types.T.replicas_ramp` on the controller: phases, stagger, the paused clock, a lowered max
and the vLLM load guard (`hold_if`, was experiments/run4/load_guard.py)."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from flock.controller.health import parse_vllm
from flock.controller.reconcile import vllm_busy
from flock.models import RampHold, RampSpec

from .conftest import make_spec


def put(store, clock, **ramp):
    spec = make_spec(r={"replicas": 0, "agents_per_worker": 50, "replicas_ramp": ramp})
    store.put_swarm(spec, clock().timestamp())


def replicas(store) -> int:
    return store.get_swarm("demo")["spec"].types["r"].replicas


def events(store, kind):
    return [e for e in store.read_events("_swarm.demo", 0) if e["type"] == kind]


def test_phases_follow_on_and_the_old_form_is_one_phase():
    run4 = RampSpec(start=20, max=300, phases=[{"every": "5m", "add": 20, "until": 100},
                                               {"every": "10m", "add": 20}])
    s = [(t, n) for t, n, _ in run4.schedule()]
    assert s[:6] == [(0, 20), (300, 40), (600, 60), (900, 80), (1200, 100), (1800, 120)]
    assert s[-1] == (7200, 300)
    old = RampSpec(start=10, add=10, every="10m", max=50)  # stored before phases existed
    assert old.schedule() == [(0, 10, 10), (600, 20, 0), (1200, 30, 0), (1800, 40, 0),
                              (2400, 50, 0)]
    for bad in ({"start": 1, "max": 5}, {"start": 1, "max": 5, "add": 1, "every": "1m",
                                         "phases": [{"every": "1m", "add": 1}]},
                {"start": 1, "max": 5, "add": 1, "every": "1m", "stagger": "soon"}):
        with pytest.raises(ValidationError):
            RampSpec.model_validate(bad)
    for bad in (0, -1, "bad", float("nan"), float("inf")):  # load_guard: finite and positive
        with pytest.raises(ValidationError):
            RampHold(request_s=bad)


def test_steps_raise_to_the_schedule(store, clock, tick):
    put(store, clock, start=2, max=5, stagger="0s",
        phases=[{"every": "60s", "add": 2, "until": 4}, {"every": "120s", "add": 2}])
    seen = []
    for dt in (0, 59, 1, 60, 60, 600):
        clock.advance(dt)
        tick()
        seen.append(replicas(store))
    assert seen == [2, 2, 4, 4, 5, 5]
    assert [e["data"]["replicas"] for e in events(store, "ramp_step")] == [2, 4, 5]


def test_stagger_adds_one_agent_at_a_time(store, clock, tick):
    put(store, clock, start=3, add=2, every="10m", max=5)  # stagger: 10 s by default
    seen = []
    for dt in (0, 5, 5, 5, 5, 30):
        clock.advance(dt)
        tick()
        seen.append(replicas(store))
    assert seen == [1, 1, 2, 2, 3, 3]
    [step] = events(store, "ramp_step")
    assert step["data"]["replicas"] == 3 and step["data"]["stagger_s"] == 10
    clock.advance(600)  # later phases: stagger 0 unless they set one (waves.yaml semantics)
    tick()
    assert replicas(store) == 5


def test_the_clock_stops_while_the_type_is_held(store, clock, tick):
    put(store, clock, start=1, add=1, every="60s", max=10, stagger="0s")
    tick()
    store.put_swarm_flags("demo", {"holds": {"types": {"r": {"by": "op"}}}})
    tick()
    clock.advance(3600)
    tick()
    assert replicas(store) == 1
    store.put_swarm_flags("demo", {})
    tick()
    clock.advance(59)
    tick()
    assert replicas(store) == 1  # the hour held does not count
    clock.advance(1)
    tick()
    assert replicas(store) == 2


def test_a_lowered_max_caps_raises_and_never_deletes(store, clock, tick):
    put(store, clock, start=4, add=2, every="60s", max=10, stagger="0s")
    tick()
    spec = store.get_swarm("demo")["spec"]
    r = spec.types["r"]
    low = r.model_copy(update={"replicas_ramp": r.replicas_ramp.model_copy(update={"max": 2})})
    store.update_swarm_spec(spec.model_copy(update={"types": {"r": low}}))
    for _ in range(3):
        clock.advance(60)
        tick()
        assert replicas(store) == 4  # above the new max: kept
    spec = store.get_swarm("demo")["spec"]
    store.update_swarm_spec(spec.model_copy(update={"types": {"r": spec.types["r"].model_copy(
        update={"replicas": 1})}}))  # an operator scales down: no step left under max 2
    clock.advance(60)
    tick()
    assert replicas(store) == 1


# ----------------------------------------------------------------------------- hold_if

H = RampHold(waiting=1, request_s=15, above=100)


def samples(waiting=(0, 0, 0), running=10, completed=(0, 60, 120), t0=1000.0):
    return [[t0 + 60 * i, w, running, c] for i, (w, c) in enumerate(zip(waiting, completed))]


def test_guard_waiting_in_every_sample_holds():
    now = 1120.0
    assert vllm_busy({"a": samples()}, H, now) is None
    assert vllm_busy({"a": samples(), "b": samples(waiting=(1, 2, 1))}, H, now).startswith(
        "HOLD: server 1: waiting=0")
    assert vllm_busy({"a": samples(waiting=(1, 0, 1))}, H, now) is None  # not in every sample


def test_guard_seconds_per_request():
    slow = {"a": samples(completed=(0, 30, 60))}  # 0.5/s with 10 running: 20 s/request
    assert "~20.0s/request" in vllm_busy(slow, H, 1120.0)
    assert vllm_busy(slow, RampHold(request_s=25), 1120.0) is None
    stalled = {"a": samples(completed=(5, 5, 5))}
    assert vllm_busy(stalled, H, 1120.0)  # running, nothing completes: infinite
    assert vllm_busy({"a": samples(running=0, completed=(5, 5, 5))}, H, 1120.0) is None


def test_guard_holds_on_missing_short_stale_or_reset_metrics():
    assert "no vLLM metrics" in vllm_busy({}, H, 0)
    assert "unavailable" in vllm_busy({"a": samples()[:2]}, H, 1120.0)
    assert "unavailable" in vllm_busy({"a": samples()}, H, 1120.0 + 600)
    assert "reset" in vllm_busy({"a": samples(completed=(50, 60, 3))}, H, 1120.0)


def test_parse_vllm_counts_completions_but_not_aborts():
    text = ('vllm:num_requests_running{m="x"} 3\nvllm:num_requests_waiting 1\n'
            'vllm:request_success_total{finished_reason="stop"} 7\n'
            'vllm:request_success_total{finished_reason="length"} 2\n'
            'vllm:request_success_total{finished_reason="abort"} 50\n')
    assert parse_vllm(text) == {"running": 3, "waiting": 1, "completed": 9}


def test_hold_if_only_above_the_floor_and_warns_after_three(store, clock, tick):
    put(store, clock, start=2, add=2, every="60s", max=200, stagger="0s",
        hold_if={"above": 3})
    tick()
    assert replicas(store) == 2  # 2 agents: under the floor, no metrics needed
    clock.advance(60)
    tick()
    assert replicas(store) == 2  # 4 > 3: no metrics -> held
    for _ in range(3):
        clock.advance(30)
        tick()  # re-asked only every 2 minutes
    clock.advance(30)
    tick()
    clock.advance(120)
    tick()
    held = events(store, "ramp_held")
    assert [e["data"]["holds"] for e in held] == [1, 2, 3]
    assert "warning" in held[-1]["data"] and "warning" not in held[1]["data"]
    assert replicas(store) == 2
    clock.advance(120)
    store.put_control("demo", "vllm", {"http://a": samples(t0=clock().timestamp() - 120)})
    tick()
    assert replicas(store) > 2  # capacity back: the step that is due now
    assert "holds" not in store.get_control("demo", "ramp:r")
