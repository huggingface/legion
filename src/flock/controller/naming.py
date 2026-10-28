"""Agent names (docs/spawn.md "Names"). `naming: numeric` (the default): `<type>-NNNN` for
replicas, `<parent>-<type>-NNNN` for spawned children. `naming: words`: `<prefix><adjective>-
<noun>` (the type's `name_prefix`, e.g. `iv-`), reserved in the store with the slot or the
parent, unique on the controller, never reused while the swarm exists, and kept when a replica
slot is recreated (also across controller restarts: the reservation is in the store). With
`naming_seed` the words are a function of (seed, swarm, reservation number), so the same spec
gives the same names; without it they are random."""

from __future__ import annotations

import random
import re

from flock.models import SwarmSpec, agent_name, child_name
from flock.words import ADJECTIVES, NOUNS

HANDLE_RE = re.compile(r"^[a-z]+(-[a-z]+)+$")  # a words name: letters and hyphens only
_SYSTEM = random.SystemRandom()


def _words(rng: random.Random, long: bool) -> str:
    adj, noun = rng.choice(ADJECTIVES), rng.choice(NOUNS)
    while noun == adj:  # a few words are both (lemon, rose, ...)
        noun = rng.choice(NOUNS)
    return f"{rng.choice(ADJECTIVES)}-{adj}-{noun}" if long else f"{adj}-{noun}"


def reserve_word_name(store, spec: SwarmSpec, type_name: str, slot: int | None,
                      parent: str | None, now: float) -> str:
    """A new words name for this agent, reserved in the store."""
    ts = spec.types.get(type_name)
    prefix = (ts.name_prefix if ts else None) or ""
    k = store.reserved_names_count(spec.swarm)
    for attempt in range(10_000):
        rng = (random.Random(f"{spec.naming_seed}:{spec.swarm}:{k}:{attempt}")
               if spec.naming_seed is not None else _SYSTEM)
        name = prefix + _words(rng, long=attempt >= 200)  # 3 words once 2 are crowded
        if store.reserve_name(name, spec.swarm, type_name, slot, parent, now):
            return name
    raise RuntimeError(f"no free name for {type_name} in swarm {spec.swarm}")


def replica_names(store, spec: SwarmSpec, type_name: str, replicas: int, now: float) -> list[str]:
    """The names of replica slots 0..replicas-1: an existing replica's name, else the slot's
    reserved name, else a new one (numeric naming: `<type>-NNNN`)."""
    have = {a["slot"]: a["name"] for a in store.list_agents(spec.swarm, type_name)
            if a["parent"] is None}
    if spec.naming != "words":
        return [have.get(s) or agent_name(type_name, s) for s in range(replicas)]
    reserved = store.reserved_slot_names(spec.swarm, type_name)
    return [have.get(s) or reserved.get(s) or reserve_word_name(store, spec, type_name, s, None, now)
            for s in range(replicas)]


def new_child_name(store, spec: SwarmSpec, parent: str, type_name: str, n: int,
                   now: float) -> str:
    """A spawned child's name: `<parent>-<type>-NNNN`, or a reserved words name."""
    if spec.naming == "words":
        return reserve_word_name(store, spec, type_name, None, parent, now)
    return child_name(parent, type_name, n)
