"""Swarm status as data, for `flock status` and the Python API (docs/health.md): one view of a
swarm from the controller's status, health snapshot and routes, its verdict, and the event
streams in one normalised vocabulary. Everything goes through `flock.client.Client`; the CLI
only formats what these return."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from flock.client import Client
from flock.models import AGENT_NAME_RE

STALE_S = 120.0  # a health snapshot older than this: the controller's prober is not running
FAILING = ("world_down", "provider_down")  # snapshot alerts that make the verdict `failing`

# Stored event type -> its dotted name where `<family>.<rest>` or `<stream>.<type>` is not it.
# Stored events are never rewritten (resume reads them); docs/health.md "Events" has the table.
RENAMED = {
    "stuck": "agent.stuck", "nudge": "agent.nudged", "spawned": "agent.spawned",
    "child_removed": "agent.child_removed", "final_round": "loop.final_round",
    "final_done": "loop.final_done", "paused": "swarm.stopped", "remove_swarm": "swarm.removed",
    "launch": "worker.launched", "launch_failed": "worker.launch_failed",
    "launch_refused": "type.launch_refused", "crash": "worker.crashed",
    "crashloop": "worker.crashloop", "planned_restart": "worker.restarted",
    "cancel": "worker.cancelled", "retire": "worker.retired", "remove_worker": "worker.removed",
    "release": "worker.released", "memory_rotation": "worker.rotated",
    "sync_failed": "worker.sync_failed",
}
FAMILIES = ("loop", "ramp", "world", "budget", "comms")  # `ramp_held` -> `ramp.held`


def event_name(kind: str, stream: str) -> str:
    """The dotted name of a stored event type; `stream` is `agent` or `swarm`."""
    if kind in RENAMED:
        return RENAMED[kind]
    head, _, rest = kind.partition("_")
    return f"{head}.{rest}" if head in FAMILIES and rest else f"{stream}.{kind}"


def normalise(ev: dict, swarm: str, agent: dict | None = None) -> dict:
    """One stored event as `{name, ts, swarm, agent?, type?, worker?, data}`: from the agent
    stream of `agent` (its status) when given, else from the swarm stream. `data` is the
    event's payload (a pi entry's `data`, else the whole entry), with `turn_id` when set."""
    d = ev.get("data") or {}
    data = d["data"] if isinstance(d.get("data"), dict) else d
    if ev.get("turn_id"):
        data = {**data, "turn_id": ev["turn_id"]}
    name = event_name(ev["type"], "agent" if agent else "swarm")
    target = d.get("target")
    out = {"name": name, "ts": ev["ts"], "swarm": swarm,
           "agent": (agent or {}).get("name") or ev.get("agent") or data.get("agent")
           or (target if name.startswith("agent.") else None),
           "type": (agent or {}).get("type") or (data.get("type") if isinstance(data.get("type"), str)
                                                  else None),
           "worker": data.get("worker") or (target if name.startswith("worker.") else None),
           "data": data}
    return {k: v for k, v in out.items() if v is not None or k == "data"}


def events(client: Client, name: str, *, after: int = 0, follow: bool = False,
           prefix: str = "", timeout: float | None = None) -> Iterator[dict]:
    """Pages `{"events": [normalised, ...], "next": cursor}` of a swarm's or an agent's stream
    after `after`, only the names starting with `prefix`; `follow` long-polls (until
    `timeout` s, if given)."""
    is_swarm = not (client.as_agent or AGENT_NAME_RE.match(name)) and client.is_swarm(name)
    agent = None if is_swarm else client.agent(name)
    swarm = name if is_swarm else agent["swarm"]
    for page in client.events(name, after=after, follow=follow, timeout=timeout, swarm=is_swarm):
        evs = (normalise(e, swarm, agent) for e in page["events"])
        yield {"events": [e for e in evs if e["name"].startswith(prefix)], "next": page["next"]}


def verdict(status: dict, health: dict | None, now: float | None = None) -> dict:
    """`{"level": ok | degraded | failing, "reasons": [...]}`. Failing: the world is down, a
    provider's every server is down under its agents, or a type's launches are refused.
    Degraded: any other snapshot alert (stuck agents, a dead server, memory, tool errors, ramp
    holds), a launch error being retried, unsynced agents, a budget stop, or no recent
    snapshot."""
    failing, degraded = [], []
    for t in status["types"].values():
        if t["health"] == "launch_refused":
            failing.append(f"type {t['name']}: launches refused: {t.get('launch_error') or '-'}")
        elif t.get("launch_error"):
            degraded.append(f"type {t['name']}: launch error, retrying: {t['launch_error']}")
    if unsynced := (status.get("released") or {}).get("unsynced"):
        degraded.append(f"{len(unsynced)} agents' final sync failed")
    if status.get("budget_stop"):
        degraded.append(f"budget stop ({status['budget_stop']})")
    if health is None:
        degraded.append("no health snapshot yet")
    else:
        if (age := (now or time.time()) - health["ts"]) > STALE_S:
            degraded.append(f"health snapshot {age:.0f} s old")
        for a in health["body"]["alerts"]:
            (failing if a["kind"] in FAILING else degraded).append(a["msg"])
    return {"level": "failing" if failing else "degraded" if degraded else "ok",
            "reasons": failing + degraded}


def dead_servers(health: dict | None) -> set[str]:
    """`host:port` of every model server the last probe found down."""
    return {h for servers in ((health or {}).get("body", {}).get("providers") or {}).values()
            for h, sv in servers.items() if not sv["up"]}


def view(client: Client, swarm: str, *, type: str | None = None, state: str | None = None,
         parent: str | None = None, now: float | None = None) -> dict[str, Any]:
    """Everything `flock status SWARM` shows: `status` (counts, holds, loops), `health` (the
    newest snapshot, or None), `workers` (every row), `verdict`; with a filter also `agents`
    (each with `on_dead_server`), and with `type` that type's `loop` (per-agent rows)."""
    st = client.swarm(swarm)["status"]
    h = client.health(swarm)
    out: dict[str, Any] = {"swarm": swarm, "verdict": verdict(st, h, now), "status": st,
                           "health": h, "workers": client.workers(swarm)}
    if type or state or parent:
        dead = dead_servers(h)
        out["agents"] = [{**a, "on_dead_server": a.get("endpoint") in dead}
                         for a in client.agents(swarm, type=type, state=state, parent=parent)]
    if type:
        out["loop"] = next((lp for lp in client.loops(swarm) if lp["type"] == type), None)
    return out
