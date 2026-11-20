"""`spawn:`, `workers:` and `limits:` in the spec: parsing, defaults and apply-time checks."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flock.models import (
    SwarmSpec,
    child_depth,
    child_name,
    initial_workers,
    max_workers,
    root_depth,
)

from .conftest import FAKE_INFERENCE, HB


def spec(spawn=None, limits=None, types=None) -> SwarmSpec:
    base = {**HB, "model": "m"}
    types = types or {
        "leader": {**HB, "model": "lead", "replicas": 1,
                   "workers": {"initial": 1, "max_extra": 0}},
        "analyst": {"replicas": 0, "workers": {"initial": 1, "max_extra": 3}},
        "helper": {"replicas": 0, "workers": {"initial": 0, "max_extra": 2}},
    }
    body = {"swarm": "s", "namespace": "ns", "bucket_id": "ns/b", "inference": FAKE_INFERENCE,
            "types": {k: {**base, **v} for k, v in types.items()}}
    if spawn is not None:
        body["spawn"] = spawn
    if limits is not None:
        body["limits"] = limits
    return SwarmSpec.model_validate(body)


AGREED = {
    "leader": {"types": {"analyst": 3, "helper": 2}, "depth": 2},
    "analyst": {"types": ["helper"], "max": 3, "depth": 1},
}


def test_agreed_example_parses():
    s = spec(AGREED, {"alive": 500, "spawned_total": 2000, "budget_usd": 300})
    leader, analyst = s.spawn["leader"], s.spawn["analyst"]
    assert leader.allowed == ["analyst", "helper"] and leader.cap_for("helper") == 2
    assert leader.max is None and leader.depth == 2
    assert analyst.allowed == ["helper"] and analyst.cap_for("helper") is None
    assert analyst.max == 3
    assert s.limits.alive == 500 and s.limits.budget_usd == 300


def test_defaults_keep_v1_specs_valid():
    s = spec(types={"researcher": {"replicas": 3}})
    assert s.spawn == {} and s.limits.alive is None
    assert s.types["researcher"].workers is None
    # round trip through the stored form
    assert SwarmSpec.model_validate_json(s.model_dump_json()) == s


def test_depth_rules():
    s = spec(AGREED)
    assert root_depth(s, "leader") == 2 and root_depth(s, "helper") == 0
    assert child_depth(s, 2, "analyst") == 1  # min(2 - 1, analyst's 1)
    assert child_depth(s, 1, "analyst") == 0  # min(1 - 1, 1)
    assert child_depth(s, 2, "helper") == 0  # helper is absent from spawn:
    assert child_name("leader-0000", "analyst", 7) == "leader-0000-analyst-0007"


def test_worker_defaults():
    s = spec(types={"researcher": {"replicas": 120, "agents_per_worker": 50},
                    "analyst": {"replicas": 0, "workers": {"max_extra": 2}}})
    assert initial_workers(s.types["researcher"]) == 3 and max_workers(s.types["researcher"]) == 3
    assert initial_workers(s.types["analyst"]) == 0 and max_workers(s.types["analyst"]) == 2


@pytest.mark.parametrize(
    "spawn, message",
    [
        ({"nope": {"types": ["helper"], "max": 1}}, "spawn.nope: unknown type"),
        ({"leader": {"types": ["nope"], "max": 1}}, "unknown type 'nope'"),
        ({"leader": {"types": {"nope": 1}}}, "unknown type 'nope'"),
        ({"leader": {"types": ["helper"]}}, "needs `max`"),
        ({"leader": {"types": ["helper"], "max": -1}}, "must be >= 0"),
        ({"leader": {"types": {"helper": -1}}}, "caps must be >= 0"),
        ({"leader": {"types": ["helper"], "max": 1, "depth": -1}}, "must be >= 0"),
        ({"leader": {"types": ["helper"], "max": 1, "warm": 2}}, "warm\n  Extra inputs are not permitted"),
        ({"leader": {"types": ["leader"], "max": 1}}, "cycle leader -> leader"),
        (
            {**AGREED, "helper": {"types": ["analyst"], "max": 1}},
            "cycle analyst -> helper -> analyst",
        ),
    ],
)
def test_apply_rejects(spawn, message):
    with pytest.raises(ValidationError, match=message):
        spec(spawn)


def test_spawnable_type_needs_workers():
    with pytest.raises(ValidationError, match="types.helper: spawnable by 'leader'"):
        spec({"leader": {"types": ["helper"], "max": 1}},
             types={"leader": {"replicas": 1}, "helper": {"replicas": 0}})
    # the spawning type itself needs no `workers`
    spec({"leader": {"types": ["helper"], "max": 1}},
         types={"leader": {"replicas": 1}, "helper": {"workers": {"max_extra": 1}}})


@pytest.mark.parametrize("limits", [{"alive": -1}, {"spawned_total": -1}, {"budget_usd": -0.5}])
def test_limits_non_negative(limits):
    with pytest.raises(ValidationError, match="must be >= 0"):
        spec(AGREED, limits)


@pytest.mark.parametrize("workers", [{"initial": -1}, {"max_extra": -1}])
def test_workers_non_negative(workers):
    with pytest.raises(ValidationError, match="must be >= 0"):
        spec(types={"a": {"workers": workers}})


def test_replicas_must_fit_declared_workers():
    """Review P2 (spawn_worker_cap.py): declared workers are a cap, replicas included."""
    over = {"replicas": 3, "agents_per_worker": 1, "workers": {"initial": 1, "max_extra": 0}}
    with pytest.raises(ValidationError, match=r"replicas 3 do not fit in workers.initial \+ "
                       r"workers.max_extra = 1 workers x agents_per_worker 1 = 1 agents"):
        spec(types={"analyst": over})
    spec(types={"analyst": {**over, "workers": {"initial": 1, "max_extra": 2}}})  # exactly fits
    spec(types={"analyst": {**over, "workers": {"max_extra": 0}}})  # initial defaults to 3
    spec(types={"analyst": {"replicas": 3, "agents_per_worker": 1}})  # no workers: v1


