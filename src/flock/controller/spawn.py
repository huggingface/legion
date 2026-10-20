"""Spawned agents (docs/spawn.md): admission, removal with cascade, per-agent tokens and
what an agent token may do.

A spawned agent is an ordinary agent row with `parent` set and no replica slot (its `slot`
holds the swarm's spawn counter value instead). Reconcile places it on its type's workers like
any other row; scaling `replicas` never touches it.

Per-agent tokens are HMAC-SHA256(key, "<agent>\\0<nonce>") under a key derived from the
controller token. The table keeps the nonce and a SHA-256 of the token, never the token, and
the controller can re-derive the same value for every later assign of the agent (a relaunched
worker, a re-sent assign). Removing the agent deletes the row, which revokes the token;
rotating FLOCK_TOKEN invalidates every agent token until the next assign re-issues it.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime

from flock.controller.store import Row, Store
from flock.models import (
    Event,
    PromptCommand,
    SpawnAck,
    SpawnRequest,
    SwarmSpec,
    TypeHealth,
    UnassignCommand,
    child_depth,
    child_name,
    max_workers,
    root_depth,
)

log = logging.getLogger("flock.spawn")

REASONS = (
    "type_not_allowed",
    "max_children",
    "depth",
    "workers",
    "alive",
    "spawned_total",
    "budget",
    "type_crashlooping",
    "parent_removing",
)
MAX_LINEAGE = 64  # parent-chain walk bound; depth is finite by construction, this is a guard


class Refused(Exception):
    """A spawn the policy does not admit: 409 with `reason` (one of REASONS)."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason, self.message = reason, message


def _ts(now: float) -> datetime:
    return datetime.fromtimestamp(now, UTC)


# --------------------------------------------------------------------------- tokens


def set_token_key(store: Store, controller_token: str) -> None:
    store.agent_token_key = (
        hashlib.sha256(b"flock-agent-token\0" + controller_token.encode()).digest()
        if controller_token
        else None
    )


def _mint(key: bytes, agent: str, nonce: str) -> str:
    return "flock_at_" + hmac.new(key, f"{agent}\0{nonce}".encode(), hashlib.sha256).hexdigest()


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def agent_token(store: Store, agent: str, swarm: str, now: float) -> str | None:
    """The agent's token: issued on first call, the same value on every later call. None for
    an agent that does not exist or is being removed (its token stays revoked)."""
    key = store.agent_token_key
    if key is None:
        return None
    a = store.get_agent(agent)
    if a is None or a["removing"]:
        return None
    row = store.get_agent_token(agent)
    nonce = row["nonce"] if row else secrets.token_hex(16)
    token = _mint(key, agent, nonce)
    if row is None or row["token_hash"] != token_hash(token):  # new, or FLOCK_TOKEN rotated
        store.put_agent_token(agent, swarm, token_hash(token), nonce, now)
    return token


def agent_for_token(store: Store, token: str) -> Row | None:
    """The live agent a presented token belongs to, or None (also while it is being removed)."""
    key = store.agent_token_key
    if key is None or not token.startswith("flock_at_"):
        return None
    name = store.agent_for_token_hash(token_hash(token))
    row = store.get_agent_token(name) if name else None
    if row is None or not hmac.compare_digest(_mint(key, name, row["nonce"]), token):
        return None  # minted under an older controller token
    a = store.get_agent(name)
    return None if a is None or a["removing"] else a


# --------------------------------------------------------------------------- lineage


def agent_depth(spec: SwarmSpec | None, a: Row) -> int:
    """Generations `a` may still spawn below itself. A replica follows the current spec; a
    spawned agent keeps the depth it was born with."""
    if a["parent"] is not None:
        return a["depth"]
    return root_depth(spec, a["type"]) if spec else 0


def is_descendant(store: Store, name: str | None, ancestor: str) -> bool:
    a = store.get_agent(name) if name else None
    for _ in range(MAX_LINEAGE):
        if a is None or a["parent"] is None:
            return False
        if a["parent"] == ancestor:
            return True
        a = store.get_agent(a["parent"])
    return False


def agent_may(
    store: Store,
    agent: Row,
    method: str,
    route: str,
    params: Mapping[str, str],
    query: Mapping[str, str],
) -> bool:
    """What an agent token allows: itself (status, events, whoami, prompting its own
    children in bulk), its subtree (status, events, spawn, prompt, bulk-prompt children,
    remove; it cannot remove itself) and prompting its parent."""
    me, target = agent["name"], params.get("name")

    def below(t: str | None) -> bool:
        return is_descendant(store, t, me)

    match method, route:
        case "GET", "/whoami":
            return True
        case "GET", "/agents/{name}" | "/agents/{name}/events":
            return target == me or below(target)
        case "POST", "/agents/{name}/spawn":
            return target == me or below(target)
        case "POST", "/agents/{name}/children/prompts":
            return target == me or below(target)
        case "POST", "/agents/{name}/prompt":
            return (target is not None and target == agent["parent"]) or below(target)
        case "DELETE", "/agents/{name}":
            return below(target)
        case "GET", "/swarms/{swarm}/agents":
            parent = query.get("parent")
            return params.get("swarm") == agent["swarm"] and (parent == me or below(parent))
    return False


# --------------------------------------------------------------------------- spawn


def spawn(store: Store, parent_name: str, req: SpawnRequest, now: float) -> tuple[SpawnAck, bool]:
    """Admit and create one child of `parent_name`. Returns (ack, created). Raises KeyError for
    an unknown parent and Refused for a policy refusal. A repeated `key` returns the child it
    created first, whatever the caps say now."""
    with store.transaction():
        parent = store.get_agent(parent_name)
        if parent is None:
            raise KeyError(parent_name)
        if parent["removing"]:
            raise Refused("parent_removing", f"{parent_name} is being removed")
        if req.key is not None and (child := store.spawn_key_child(parent_name, req.key)):
            row = store.get_agent(child)
            depth = row["depth"] if row else 0
            ack = SpawnAck(name=child, created=False, parent=parent_name, type=req.type, depth=depth)
            return ack, False
        sw = store.get_swarm(parent["swarm"])
        if sw is None or sw["removing"]:
            raise Refused("parent_removing", f"swarm {parent['swarm']!r} is being removed")
        spec: SwarmSpec = sw["spec"]
        _admit(store, spec, parent, req.type)
        swarm, depth = parent["swarm"], child_depth(spec, agent_depth(spec, parent), req.type)
        while True:  # a replica of an oddly named type could hold the name: skip it
            n = store.next_spawn_number(swarm)
            name = child_name(parent_name, req.type, n)
            if store.get_agent(name) is None:
                break
        store.insert_child(name, swarm, req.type, n, parent_name, depth, now)
        if req.key is not None:
            store.put_spawn_key(parent_name, req.key, name)
        agent_token(store, name, swarm, now)  # issued at creation
        if req.prompt:
            cmd = PromptCommand(agent=name, prompt_id=uuid.uuid4().hex, text=req.prompt)
            store.add_prompt(name, cmd.model_dump_json(), now)
    ev = Event(type="spawned", ts=_ts(now), agent=parent_name,
               data={"child": name, "type": req.type, "depth": depth, "key": req.key})
    store.append_events(swarm, parent_name, [ev])
    store.listener("reconcile", "")
    return SpawnAck(name=name, created=True, parent=parent_name, type=req.type, depth=depth), True


def _admit(store: Store, spec: SwarmSpec, parent: Row, type_name: str) -> None:
    rule = spec.spawn.get(parent["type"])
    if rule is None or type_name not in rule.allowed or type_name not in spec.types:
        raise Refused("type_not_allowed", f"type {parent['type']!r} may not spawn {type_name!r}")
    if agent_depth(spec, parent) < 1:
        raise Refused("depth", f"{parent['name']} may not spawn (depth 0)")
    kids = store.list_agents(parent=parent["name"])
    cap = rule.cap_for(type_name)
    if cap is not None and sum(k["type"] == type_name for k in kids) >= cap:
        raise Refused("max_children", f"{parent['name']} already has {cap} {type_name} children")
    if rule.max is not None and len(kids) >= rule.max:
        raise Refused("max_children", f"{parent['name']} already has {rule.max} children")
    if store.type_health(spec.swarm, type_name) == TypeHealth.crashlooping:
        raise Refused("type_crashlooping", f"type {type_name!r} is crashlooping")
    lim = spec.limits
    if lim.alive is not None and store.count_agents(spec.swarm) >= lim.alive:
        raise Refused("alive", f"swarm has {lim.alive} agents (limits.alive)")
    if lim.spawned_total is not None and store.spawned_count(spec.swarm) >= lim.spawned_total:
        raise Refused("spawned_total", f"swarm spawned {lim.spawned_total} agents")
    if lim.budget_usd is not None and store.swarm_cost(spec.swarm) >= lim.budget_usd:
        raise Refused("budget", f"swarm cost reached ${lim.budget_usd:g} (limits.budget_usd)")
    ts = spec.types[type_name]
    room = max_workers(ts) * max(1, ts.agents_per_worker)
    if store.count_agents(spec.swarm, type_name) >= room:
        raise Refused(
            "workers",
            f"type {type_name!r} is full: {max_workers(ts)} workers x {ts.agents_per_worker}",
        )


def worker_cap_conflicts(store: Store, spec: SwarmSpec) -> list[str]:
    """Types whose declared worker cap is below the workers their agents occupy now. Agents
    never move between live workers, so such an edit could not be honoured: apply refuses it."""
    out = []
    for name, ts in sorted(spec.types.items()):
        if ts.workers is None:
            continue
        in_use = len({a["worker"] for a in store.list_agents(spec.swarm, name) if a["worker"]})
        if in_use > max_workers(ts):
            out.append(
                f"type {name!r}: workers.initial + workers.max_extra = {max_workers(ts)} is below "
                f"the {in_use} workers its agents occupy; remove agents first"
            )
    return out


# --------------------------------------------------------------------------- removal


def _subtree(store: Store, name: str) -> list[Row]:
    """`name` and its descendants, children before parents."""
    out: list[Row] = []
    for child in store.list_agents(parent=name):
        out += _subtree(store, child["name"])
    a = store.get_agent(name)
    return out + ([a] if a else [])


def remove_subtree(store: Store, name: str, reason: str, now: float) -> list[Row]:
    """Remove `name` and everything it spawned, children first. Each placed agent gets an
    unassign (final sync; bucket state is kept), each token is revoked, and each spawned
    agent's parent gets `child_removed`. Returns the removed rows.

    Two transactions, each serialized with spawn admission (same store transaction): the
    first enumerates the subtree, marks every row `removing` and revokes every token, so no
    child can be added below it from then on (a spawn there gets 409 parent_removing); the
    second deletes the rows. A crash between them leaves marked rows that reconcile removes."""
    with store.transaction():
        rows = _subtree(store, name)
        for r in rows:
            store.mark_removing(r["name"])
    if not rows:
        return []
    events: list[Event] = []
    with store.transaction():
        for r in rows:
            w = store.get_worker(r["worker"]) if r["worker"] else None
            if w and w["job_id"]:
                store.enqueue_command(w["id"], UnassignCommand(agent=r["name"]).model_dump_json())
            store.delete_agent_rows(r["name"])
            if r["parent"] and store.get_agent(r["parent"]) is not None:
                why = reason if r["name"] == name else "parent_removed"
                events.append(Event(type="child_removed", ts=_ts(now), agent=r["parent"],
                                    data={"child": r["name"], "type": r["type"], "reason": why}))
    for ev in events:
        store.append_events(rows[0]["swarm"], ev.agent, [ev])
    return rows
