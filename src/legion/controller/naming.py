"""Agent names (docs/spawn.md "Names"). `naming: numeric` (the default): `<type>-NNNN` for
replicas, `<parent>-<type>-NNNN` for spawned children (`naming: consecutive-id`, Harbor types: the
replicas get their roster's names, consecutive `agent-001`, ...). `naming: words`: `<prefix><adjective>-
<noun>` (the type's `name_prefix`, e.g. `iv-`), reserved in the store with the slot or the
parent, unique on the controller, never reused while the swarm exists, and kept when a replica
slot is recreated (also across controller restarts: the reservation is in the store). The
words are random."""

from __future__ import annotations

import itertools
import random
import re

from legion.models import SwarmSpec, agent_name, child_name
from legion.words import ADJECTIVES, NOUNS

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
    for attempt in range(10_000):
        name = prefix + _words(_SYSTEM, long=attempt >= 200)  # 3 words once 2 are crowded
        if store.reserve_name(name, spec.swarm, type_name, slot, parent, now):
            return name
    raise RuntimeError(f"no free name for {type_name} in swarm {spec.swarm}")


def replica_names(store, spec: SwarmSpec, type_name: str, replicas: int, now: float) -> list[str]:
    """The names of replica slots 0..replicas-1: an existing replica's name, else the slot's
    reserved name, else a new one (numeric naming: `<type>-NNNN`)."""
    have = {a["slot"]: a["name"] for a in store.list_agents(spec.swarm, type_name)
            if a["parent"] is None}
    ts = spec.types.get(type_name)
    if spec.naming == "consecutive-id" and ts and ts.harbor:
        roster = ts.harbor.roster
        taken = set(roster) | {a["name"] for a in store.list_agents(spec.swarm)}
        # past the roster (replicas above the count): the next free `agent-NNN` names
        spare = (n for n in (f"agent-{i:03d}" for i in itertools.count(len(roster) + 1))
                 if n not in taken)
        named = ts.harbor.names()
        return [have.get(s) or (named[s] if s < len(named) else next(spare))
                for s in range(replicas)]
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
