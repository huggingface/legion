"""`naming: words` (docs/spawn.md "Names"): readable `<adjective>-<noun>` agent names, reserved
in the store, unique on the controller, never reused, kept for a recreated replica slot and
across a controller restart, a type's `name_prefix`, valid
board handles; numeric naming unchanged."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flock.controller import naming
from flock.controller.store import Store
from flock.models import SwarmSpec
from flock.words import ADJECTIVES, NOUNS

from .test_spawn_api import apply, spawn, spawn_spec


def words_spec(swarm: str = "demo", **kw) -> SwarmSpec:
    d = spawn_spec(**kw).model_dump(mode="json")
    d.update(swarm=swarm, naming="words")
    d["types"]["helper"]["name_prefix"] = "iv-"
    return SwarmSpec.model_validate(d)


def test_word_lists():
    assert 180 <= len(ADJECTIVES) <= 220 and 300 <= len(NOUNS) <= 360
    assert len(set(ADJECTIVES)) == len(ADJECTIVES) and len(set(NOUNS)) == len(NOUNS)
    assert all(w.isalpha() and w.islower() for w in ADJECTIVES + NOUNS)


def test_two_thousand_unique_handles_with_prefix(tmp_path):
    store, spec = Store(tmp_path / "s"), words_spec()
    reps = naming.replica_names(store, spec, "analyst", 1500, 0.0)
    kids = [naming.new_child_name(store, spec, "leader-x", "helper", n, 0.0) for n in range(500)]
    names = reps + kids
    assert len(set(names)) == 2000
    assert all(naming.HANDLE_RE.match(n) for n in names)
    assert all(k.startswith("iv-") and not r.startswith("iv-") for k, r in zip(kids, reps))
    assert store.reserved_names_count("demo") == 2000
    assert all(not store.reserve_name(n, "demo", "x", None, None, 0.0) for n in names[:50])
    # another swarm on the same controller never gets one of them either
    other = words_spec(swarm="other")
    assert not set(naming.replica_names(store, other, "analyst", 300, 0.0)) & set(names)
    store.close()


def test_names_persist_and_slots_keep_theirs(tmp_path):
    a = Store(tmp_path / "a")
    spec = words_spec()
    first = naming.replica_names(a, spec, "analyst", 5, 0.0)
    for slot, name in enumerate(first):
        a.insert_agent(name, spec.swarm, "analyst", slot, 0.0)
    assert naming.replica_names(a, spec, "analyst", 5, 1.0) == first  # stable while they live
    a.delete_agent(first[2])  # a removed replica slot is recreated under its reserved name
    assert naming.replica_names(a, spec, "analyst", 5, 2.0) == first
    a.close()
    a = Store(tmp_path / "a")  # a controller restart: the reservations are in the store
    assert naming.replica_names(a, spec, "analyst", 6, 3.0)[:5] == first
    a.close()


def test_numeric_naming_unchanged(tmp_path):
    s = Store(tmp_path / "s")
    assert naming.replica_names(s, spawn_spec(), "analyst", 3, 0.0) == [
        "analyst-0000", "analyst-0001", "analyst-0002"]
    assert naming.new_child_name(s, spawn_spec(), "leader-0000", "helper", 4, 0.0) == \
        "leader-0000-helper-0004"
    s.close()


def test_name_prefix_validation():
    d = spawn_spec().model_dump(mode="json")
    for bad in ("IV-", "iv", "iv_", "-iv-", "i1-"):
        d["types"]["helper"]["name_prefix"] = bad
        with pytest.raises(ValidationError, match="name_prefix"):
            SwarmSpec.model_validate(d)
    d["types"]["helper"]["name_prefix"] = "iv-x-"
    assert SwarmSpec.model_validate(d).types["helper"].name_prefix == "iv-x-"


async def test_words_through_the_controller(client, store, tick):
    """Replicas and spawned children get words names through the API; the spawn counter still
    counts (spawned_total); removal does not free a name."""
    await apply(client, words_spec())
    tick()
    (leader,) = [a["name"] for a in store.list_agents("demo", "leader")]
    assert naming.HANDLE_RE.match(leader) and not leader.startswith("leader")
    r = await spawn(client, leader, "helper")
    assert r.status_code == 201, r.text
    child = r.json()["name"]
    assert child.startswith("iv-") and naming.HANDLE_RE.match(child)
    assert (await client.get(f"/agents/{child}")).json()["parent"] == leader
    await client.delete(f"/agents/{child}")
    again = (await spawn(client, leader, "helper")).json()["name"]
    assert again != child and store.spawned_count("demo") == 2
    assert store.reserved_name(child)["type"] == "helper"  # kept after removal
    tick()
    assert [a["name"] for a in store.list_agents("demo", "leader")] == [leader]
