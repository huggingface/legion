"""Spawn on the controller: admission and its 409 reasons, cascade removal, events, the
?parent= filter, per-agent tokens and their scope, persistence of spawned rows."""

from __future__ import annotations

import httpx
import pytest
from pydantic import TypeAdapter

from flock.controller import spawn as spawning
from flock.controller.store import Store
from flock.models import AssignCommand, Command, SwarmSpec, TypeHealth

from .conftest import FAKE_PROVIDERS, TOKEN

CMD = TypeAdapter(Command)
LEADER = "leader-0000"


def spawn_spec(spawn=None, limits=None, **overrides) -> SwarmSpec:
    base = {"harness": "llm", "model": "fake:m"}
    types = {
        "leader": {"replicas": 1, "workers": {"initial": 1}},
        "analyst": {"replicas": 0, "agents_per_worker": 2, "workers": {"initial": 1, "max_extra": 1}},
        "helper": {"replicas": 0, "agents_per_worker": 2, "workers": {"initial": 0, "max_extra": 1}},
    }
    for name, o in overrides.items():
        types[name] = {**types.get(name, {}), **o}
    body = {
        "swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh", "providers": FAKE_PROVIDERS,
        "types": {k: {**base, **v} for k, v in types.items()},
        "spawn": spawn if spawn is not None else {
            "leader": {"types": {"analyst": 3, "helper": 2}, "depth": 2},
            "analyst": {"types": ["helper"], "max": 3, "depth": 1},
        },
    }
    if limits is not None:
        body["limits"] = limits
    return SwarmSpec.model_validate(body)


async def apply(client, spec: SwarmSpec) -> None:
    r = await client.put(f"/swarms/{spec.swarm}", params={"replicas": "spec"}, json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text


async def spawn(client, parent, type_, **body) -> httpx.Response:
    return await client.post(f"/agents/{parent}/spawn", json={"type": type_, **body})


def assigns(store, wid) -> list[AssignCommand]:
    cmds = [CMD.validate_json(c["body"]) for c in store.commands_for(wid)]
    return [c for c in cmds if isinstance(c, AssignCommand)]


def as_agent(app, token) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                             headers={"Authorization": f"Bearer {token}"})


@pytest.fixture
async def swarm(client, tick):
    await apply(client, spawn_spec())
    tick()


async def test_spawn_creates_child_with_prompt_event_and_lineage(client, store, swarm):
    r = await spawn(client, LEADER, "analyst", prompt="look at x")
    assert r.status_code == 201, r.text
    ack = r.json()
    assert ack == {"name": "leader-0000-analyst-0001", "created": True, "parent": LEADER,
                   "type": "analyst", "depth": 1}
    child = (await client.get(f"/agents/{ack['name']}")).json()
    assert child["parent"] == LEADER and child["depth"] == 1 and child["pending"] == 1
    leader = (await client.get(f"/agents/{LEADER}")).json()
    assert leader["children"] == [ack["name"]] and leader["depth"] == 2
    assert leader["parent"] is None
    evs = (await client.get(f"/agents/{LEADER}/events")).json()["events"]
    assert evs[-1]["type"] == "spawned"
    assert evs[-1]["data"] == {"child": ack["name"], "type": "analyst", "depth": 1, "key": None}
    # names come from a per-swarm counter, never reused
    r2 = (await spawn(client, LEADER, "helper")).json()
    assert r2["name"] == "leader-0000-helper-0002" and r2["depth"] == 0
    await client.delete(f"/agents/{r2['name']}")
    assert (await spawn(client, LEADER, "helper")).json()["name"] == "leader-0000-helper-0003"


async def test_children_are_placed_on_their_types_workers(client, store, jobs, tick, swarm):
    names = [(await spawn(client, LEADER, "analyst")).json()["name"] for _ in range(3)]
    tick()
    rows = {n: store.get_agent(n) for n in names}
    assert {r["worker"] for r in rows.values()} == {"demo-analyst-w0000", "demo-analyst-w0001"}
    launched = {x["worker_id"] for x in jobs.launches}
    assert launched == {"demo-leader-w0000", "demo-analyst-w0000", "demo-analyst-w0001"}
    cmd = next(a for a in assigns(store, "demo-analyst-w0000") if a.agent == names[0])
    assert cmd.parent == LEADER and cmd.agent_token.startswith("flock_at_")
    # the replica gets a token too, and the same one on every assign
    lead = assigns(store, "demo-leader-w0000")[0]
    assert lead.parent is None
    assert lead.agent_token == spawning.agent_token(store, LEADER, "demo", 0)
    # scaling replicas never touches children
    spec = spawn_spec(analyst={"replicas": 0})
    await apply(client, spec)
    tick()
    assert all(store.get_agent(n) for n in names)


async def test_repeated_key_returns_the_first_child(client, swarm):
    a = await spawn(client, LEADER, "analyst", key="k1")
    b = await spawn(client, LEADER, "analyst", key="k1")
    assert a.status_code == 201 and b.status_code == 200
    assert b.json()["created"] is False and b.json()["name"] == a.json()["name"]
    leader = (await client.get(f"/agents/{LEADER}")).json()
    assert leader["children"] == [a.json()["name"]]


async def reason(r: httpx.Response) -> str:
    assert r.status_code == 409, r.text
    return r.json()["reason"]


async def test_refusals(client, store, swarm):
    assert await reason(await spawn(client, LEADER, "leader")) == "type_not_allowed"
    for _ in range(3):
        assert (await spawn(client, LEADER, "analyst")).status_code == 201
    assert await reason(await spawn(client, LEADER, "analyst")) == "max_children"  # per type
    analyst = "leader-0000-analyst-0001"
    helper = (await spawn(client, analyst, "helper")).json()["name"]
    # helper is absent from spawn: it cannot spawn at all
    assert await reason(await spawn(client, helper, "helper")) == "type_not_allowed"
    store.set_type_health("demo", "helper", TypeHealth.crashlooping.value)
    assert await reason(await spawn(client, LEADER, "helper")) == "type_crashlooping"
    store.set_type_health("demo", "helper", TypeHealth.ok.value)
    # helper has max 1 worker x 2 agents: the analyst's helper and one more
    assert (await spawn(client, LEADER, "helper")).status_code == 201
    assert await reason(await spawn(client, LEADER, "helper")) == "workers"
    assert (await spawn(client, "nobody-0001", "helper")).status_code == 404


async def test_total_max_for_list_form(client, tick):
    await apply(client, spawn_spec({"leader": {"types": ["analyst", "helper"], "max": 2}}))
    tick()
    assert (await spawn(client, LEADER, "analyst")).status_code == 201
    assert (await spawn(client, LEADER, "helper")).status_code == 201
    assert await reason(await spawn(client, LEADER, "analyst")) == "max_children"


async def test_depth_runs_out(client, tick):
    spec = spawn_spec({"leader": {"types": ["analyst"], "max": 5, "depth": 1},
                       "analyst": {"types": ["helper"], "max": 5, "depth": 3}})
    await apply(client, spec)
    tick()
    child = (await spawn(client, LEADER, "analyst")).json()
    assert child["depth"] == 0  # min(1 - 1, 3)
    assert await reason(await spawn(client, child["name"], "helper")) == "depth"


@pytest.mark.parametrize(
    "limits, why", [({"alive": 2}, "alive"), ({"spawned_total": 1}, "spawned_total")]
)
async def test_swarm_limits(client, tick, limits, why):
    await apply(client, spawn_spec(limits=limits))
    tick()
    first = (await spawn(client, LEADER, "analyst")).json()["name"]
    assert await reason(await spawn(client, LEADER, "analyst")) == why
    await client.delete(f"/agents/{first}")
    r = await spawn(client, LEADER, "analyst")
    if why == "alive":  # removal frees a place under alive, never under spawned_total
        assert r.status_code == 201
    else:
        assert await reason(r) == why


async def test_budget_counts_removed_agents(client, store, tick):
    await apply(client, spawn_spec(limits={"budget_usd": 1.0}))
    tick()
    child = (await spawn(client, LEADER, "analyst")).json()["name"]
    store.update_agent(child, cost=0.6)
    store.update_agent(LEADER, cost=0.3)
    assert (await spawn(client, LEADER, "analyst")).status_code == 201
    await client.delete(f"/agents/{child}")
    store.update_agent(LEADER, cost=0.45)  # 0.6 removed + 0.45 live
    assert await reason(await spawn(client, LEADER, "analyst")) == "budget"


async def test_parent_removing(client, swarm):
    await client.delete("/swarms/demo")
    assert await reason(await spawn(client, LEADER, "analyst")) == "parent_removing"


async def test_delete_cascades_and_revokes(client, app, store, tick, swarm):
    a = (await spawn(client, LEADER, "analyst")).json()["name"]
    h = (await spawn(client, a, "helper")).json()["name"]
    tick()
    store.clear_commands("demo-analyst-w0000")
    token_h = spawning.agent_token(store, h, "demo", 0)
    r = await client.delete(f"/agents/{a}")
    assert r.status_code == 200 and r.json() == {"removed": [h, a]}
    assert store.get_agent(a) is None and store.get_agent(h) is None
    unassigned = [CMD.validate_json(c["body"]) for c in store.commands_for("demo-analyst-w0000")]
    assert [(c.kind, c.agent) for c in unassigned] == [("unassign", a)]
    evs = (await client.get(f"/agents/{LEADER}/events")).json()["events"]
    assert evs[-1]["type"] == "child_removed"
    assert evs[-1]["data"] == {"child": a, "type": "analyst", "reason": "removed"}
    evs = (await client.get(f"/agents/{a}/events")).json()["events"]
    assert evs[-1]["data"] == {"child": h, "type": "helper", "reason": "parent_removed"}
    async with as_agent(app, token_h) as c:
        assert (await c.get("/whoami")).status_code == 401
    # replicas are removed by scaling, not by DELETE
    r = await client.delete(f"/agents/{LEADER}")
    assert r.status_code == 409 and r.json()["reason"] == "not_spawned"


async def test_parent_scaled_down_or_swarm_removed_takes_children(client, store, tick, swarm):
    a = (await spawn(client, LEADER, "analyst")).json()["name"]
    h = (await spawn(client, a, "helper")).json()["name"]
    await apply(client, spawn_spec(leader={"replicas": 0}))
    tick()
    assert store.get_agent(LEADER) is None and store.get_agent(a) is None
    assert store.get_agent(h) is None
    await apply(client, spawn_spec())
    tick()
    (await spawn(client, LEADER, "analyst")).json()
    await client.delete("/swarms/demo")
    tick()
    assert store.list_agents("demo") == []


async def test_parent_filter(client, swarm):
    a = (await spawn(client, LEADER, "analyst")).json()["name"]
    h = (await spawn(client, a, "helper")).json()["name"]
    page = (await client.get("/swarms/demo/agents", params={"parent": LEADER})).json()
    assert [x["name"] for x in page["agents"]] == [a]
    assert page["agents"][0]["children"] == [h]
    page = (await client.get("/swarms/demo/agents", params={"parent": a})).json()
    assert [x["name"] for x in page["agents"]] == [h]


async def test_agent_token_scope(client, app, store, swarm):
    a = (await spawn(client, LEADER, "analyst")).json()["name"]
    h = (await spawn(client, a, "helper")).json()["name"]
    other = (await spawn(client, LEADER, "analyst")).json()["name"]
    tok = {n: spawning.agent_token(store, n, "demo", 0) for n in (LEADER, a, h)}

    async with as_agent(app, tok[a]) as c:
        me = (await c.get("/whoami")).json()
        assert me == {"kind": "agent", "name": a, "swarm": "demo", "type": "analyst",
                      "parent": LEADER, "depth": 1, "children": [h]}
        for path in (f"/agents/{a}", f"/agents/{a}/events", f"/agents/{h}", f"/agents/{h}/events"):
            assert (await c.get(path)).status_code == 200, path
        page = await c.get("/swarms/demo/agents", params={"parent": a})
        assert page.status_code == 200 and [x["name"] for x in page.json()["agents"]] == [h]
        assert (await c.post(f"/agents/{h}/prompt", json={"text": "x"})).status_code == 200
        assert (await c.post(f"/agents/{LEADER}/prompt", json={"text": "done"})).status_code == 200
        forbidden = [
            c.get(f"/agents/{LEADER}"),  # its parent: prompt only
            c.get(f"/agents/{other}"),  # a sibling
            c.post(f"/agents/{other}/prompt", json={"text": "x"}),
            c.post(f"/agents/{a}/prompt", json={"text": "x"}),  # itself
            c.delete(f"/agents/{a}"),  # itself
            c.delete(f"/agents/{other}"),
            c.post(f"/agents/{LEADER}/spawn", json={"type": "analyst"}),
            c.get("/swarms/demo/agents"),  # whole swarm
            c.get("/swarms/demo/agents", params={"parent": LEADER}),
            c.get("/swarms"),
            c.get("/swarms/demo"),
            c.put("/swarms/demo", json=spawn_spec().model_dump(mode="json")),
            c.delete("/swarms/demo"),
            c.get("/workers"),
            c.get("/workers/demo-leader-w0000/commands", params={"wait": 0}),
        ]
        for req in forbidden:
            r = await req
            assert r.status_code == 403, (r.request.method, r.request.url, r.status_code)
        r = await c.post(f"/agents/{a}/spawn", json={"type": "helper"})
        assert r.status_code == 201
        assert (await c.post(f"/agents/{h}/spawn", json={"type": "helper"})).status_code == 409
        assert (await c.delete(f"/agents/{h}")).status_code == 200
    async with as_agent(app, tok[LEADER]) as c:  # the subtree reaches grandchildren
        assert (await c.get(f"/agents/{r.json()['name']}")).status_code == 200
        assert (await c.post(f"/agents/{LEADER}/prompt", json={"text": "x"})).status_code == 403
    async with as_agent(app, "flock_at_" + "0" * 64) as c:
        assert (await c.get("/whoami")).status_code == 401
    assert (await client.get("/whoami")).json() == {"kind": "controller"}


async def test_rotated_controller_token_invalidates_agent_tokens(app, store, swarm):
    old = spawning.agent_token(store, LEADER, "demo", 0)
    spawning.set_token_key(store, "rotated")
    async with as_agent(app, old) as c:
        assert (await c.get("/whoami")).status_code == 401
    new = spawning.agent_token(store, LEADER, "demo", 0)  # the next assign re-issues
    assert new != old
    async with as_agent(app, new) as c:
        assert (await c.get("/whoami")).status_code == 200
    spawning.set_token_key(store, TOKEN)


async def test_spawned_rows_survive_a_controller_restart(client, store, tmp_path, swarm):
    a = (await spawn(client, LEADER, "analyst", key="k")).json()["name"]
    tok = spawning.agent_token(store, a, "demo", 0)
    store.close()
    again = Store(tmp_path / "data")
    try:
        spawning.set_token_key(again, TOKEN)
        row = again.get_agent(a)
        assert row["parent"] == LEADER and row["depth"] == 1
        assert again.spawned_count("demo") == 1 and again.spawn_key_child(LEADER, "k") == a
        assert spawning.agent_for_token(again, tok)["name"] == a
    finally:
        again.close()


def test_store_migrates_agents_and_swarms(tmp_path):
    import sqlite3

    d = tmp_path / "old"
    d.mkdir()
    db = sqlite3.connect(d / "controller.db")
    db.execute("CREATE TABLE agents (name TEXT PRIMARY KEY, swarm TEXT NOT NULL, type TEXT NOT "
               "NULL, slot INTEGER NOT NULL, worker TEXT, created_at REAL NOT NULL)")
    db.execute("CREATE TABLE swarms (name TEXT PRIMARY KEY, spec TEXT NOT NULL, "
               "applied_at REAL NOT NULL, removing INTEGER NOT NULL DEFAULT 0)")
    db.commit()
    db.close()
    s = Store(d)
    try:
        cols = {r["name"] for r in s._q("PRAGMA table_info(agents)")}
        assert {"parent", "depth"} <= cols
        cols = {r["name"] for r in s._q("PRAGMA table_info(swarms)")}
        assert {"spawned", "removed_cost"} <= cols
    finally:
        s.close()


def test_spawn_during_cascade_removal_leaves_no_orphan(store, client, swarm):
    """Review P1 (spawn_remove_race.py): a spawn racing a removal must not leave a child of a
    removed parent, nor a live token. The remover pauses while enumerating the subtree; the
    spawn must wait for it and then be refused."""
    import threading

    from flock.models import SpawnRequest

    parent = spawning.spawn(store, LEADER, SpawnRequest(type="analyst"), 0)[0].name
    scanned, resume = threading.Event(), threading.Event()
    original = store.list_agents

    def list_agents(*args, **kwargs):
        result = original(*args, **kwargs)
        if threading.current_thread().name == "remover" and kwargs.get("parent") == parent:
            scanned.set()
            assert resume.wait(5)
        return result

    store.list_agents = list_agents
    removed: list = []
    outcome: dict = {}

    def spawner():
        try:
            outcome["ack"] = spawning.spawn(store, parent, SpawnRequest(type="helper"), 2)
        except spawning.Refused as r:
            outcome["refused"] = r.reason
        except KeyError:
            outcome["gone"] = True

    remover = threading.Thread(
        name="remover",
        target=lambda: removed.extend(spawning.remove_subtree(store, parent, "removed", 1)),
    )
    remover.start()
    assert scanned.wait(5)
    spawn_thread = threading.Thread(target=spawner)
    spawn_thread.start()
    spawn_thread.join(0.3)
    assert spawn_thread.is_alive()  # admission waits for the enumeration to finish
    resume.set()
    remover.join(5)
    spawn_thread.join(5)
    store.list_agents = original
    assert [r["name"] for r in removed] == [parent]
    assert "ack" not in outcome and (outcome.get("refused") == "parent_removing" or outcome["gone"])
    assert store.list_agents(parent=parent) == [] and store.get_agent(parent) is None
    assert store.list_agents("demo") == [store.get_agent(LEADER)]
    assert store._q("SELECT agent FROM agent_tokens WHERE agent LIKE ?", [f"{parent}%"]) == []


async def test_spawn_under_a_row_being_removed_is_refused(client, store, tick, swarm):
    a = (await spawn(client, LEADER, "analyst")).json()["name"]
    h = (await spawn(client, a, "helper")).json()["name"]
    with store.transaction():  # the first phase of a removal, as a crash would leave it
        for n in (h, a):
            store.mark_removing(n)
    assert await reason(await spawn(client, a, "helper")) == "parent_removing"
    assert await reason(await spawn(client, a, "helper", key="k")) == "parent_removing"
    assert store.get_agent_token(a) is None and store.get_agent_token(h) is None
    tick()  # reconcile finishes the removal
    assert store.get_agent(a) is None and store.get_agent(h) is None


def test_placement_never_opens_more_than_declared_workers(store, jobs, tick):
    """Review P2: reconcile keeps a type that declares `workers` within initial + max_extra
    slots; agents that do not fit wait unplaced. A type without `workers` keeps v1."""
    spec = spawn_spec(analyst={"agents_per_worker": 1, "workers": {"initial": 1, "max_extra": 1}},
                      researcher={"replicas": 3, "agents_per_worker": 1})
    store.put_swarm(spec, 0)
    for n in range(1, 5):  # past admission's check, e.g. agents_per_worker lowered on re-apply
        store.insert_child(f"{LEADER}-analyst-{n:04d}", "demo", "analyst", n, LEADER, 0, 0)
    actions = tick()
    assert len(store.list_workers("demo", "analyst")) == 2
    placed = [a for a in store.list_agents("demo", "analyst") if a["worker"]]
    assert len(placed) == 2
    assert any(a.kind == "workers_full" for a in actions)
    assert len(store.list_workers("demo", "researcher")) == 3  # no `workers`: v1


def test_agent_being_removed_gets_no_new_token(store, clock, tick, swarm):
    """Review round 2 (spawn_removing_token_reissue.py): after the first removal phase revoked
    a token, a heartbeat that omits the agent must not re-assign it with a fresh token, and a
    token of a row marked removing must not authenticate."""
    from datetime import timedelta

    from flock.controller.reconcile import record_heartbeat
    from flock.models import SpawnRequest, WorkerHeartbeat

    parent = spawning.spawn(store, LEADER, SpawnRequest(type="analyst"), 0)[0].name
    tick()
    wid = store.get_agent(parent)["worker"]
    old = spawning.agent_token(store, parent, "demo", 0)
    with store.transaction():
        store.mark_removing(parent)
    store.clear_commands(wid)
    hb = WorkerHeartbeat(worker=wid, agents=[], deadline=clock() + timedelta(hours=1))
    assert record_heartbeat(store, wid, hb, clock())
    assert assigns(store, wid) == []  # no re-assign, so no new token
    assert spawning.agent_token(store, parent, "demo", 0) is None
    assert spawning.agent_for_token(store, old) is None
    assert store.get_agent_token(parent) is None
    # even a token row that survived (e.g. written by an older controller) does not authenticate
    spawning.set_token_key(store, TOKEN)
    store.update_agent(parent, removing=0)
    tok = spawning.agent_token(store, parent, "demo", 0)
    store.update_agent(parent, removing=1)
    assert spawning.agent_for_token(store, tok) is None


async def test_apply_refuses_a_cap_below_workers_in_use(client, store, tick):
    """Review round 2: lowering initial + max_extra below the workers a type occupies is
    refused (agents never move, nothing is grandfathered); lowering to what is in use is fine."""
    def with_extra(n):
        return spawn_spec(analyst={"agents_per_worker": 1,
                                   "workers": {"initial": 1, "max_extra": n}})

    await apply(client, with_extra(2))
    tick()
    kids = [(await spawn(client, LEADER, "analyst")).json()["name"] for _ in range(3)]
    tick()
    assert len(store.list_workers("demo", "analyst")) == 3
    for extra in (0, 1):
        r = await client.put("/swarms/demo", json=with_extra(extra).model_dump(mode="json"))
        assert r.status_code == 409
        assert r.json()["detail"] == (
            f"type 'analyst': workers.initial + workers.max_extra = {1 + extra} is below the 3 "
            "workers its agents occupy; remove agents first")
    assert store.get_swarm("demo")["spec"].types["analyst"].workers.max_extra == 2  # unchanged
    await client.delete(f"/agents/{kids[2]}")
    r = await client.put("/swarms/demo", json=with_extra(1).model_dump(mode="json"))
    assert r.status_code == 200, r.text


async def test_ephemeral_children_are_removed_after_their_session(client, store, clock, tick):
    """docs/spawn.md "Ephemeral types": a heartbeat that reports an ephemeral child idle (or
    stuck: a turn that hit max_turns, its budget or an error ends the same way) after a turn,
    with nothing pending, removes it: unassign to its worker, child_removed (reason ephemeral)
    on the parent, its cost kept, its place under the parent's cap freed; the next spawn is a
    new agent."""
    from datetime import timedelta

    from flock.models import AgentHeartbeat, UnassignCommand, WorkerHeartbeat

    spec = spawn_spec(spawn={"leader": {"types": ["analyst"], "max": 1}},
                      analyst={"ephemeral": True})
    await apply(client, spec)
    tick()
    first = (await spawn(client, LEADER, "analyst", prompt="interview x")).json()["name"]
    assert (await reason(await spawn(client, LEADER, "analyst"))) == "max_children"
    tick()
    wid = store.get_agent(first)["worker"]

    async def hb(**a):
        body = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=1),
                               agents=[AgentHeartbeat(name=first, **a)])
        r = await client.post(f"/workers/{wid}/heartbeat", json=body.model_dump(mode="json"))
        assert r.status_code == 200, r.text

    await hb(state="idle")  # placed, prompt still pending, no turn yet: kept
    got = (await client.get(f"/workers/{wid}/commands", params={"wait": 0})).json()
    assert any(c["kind"] == "prompt" for c in got["commands"])  # delivered: none pending
    await hb(state="working", turns=0)
    await hb(state="working", turns=1)  # a second turn running: kept
    assert store.get_agent(first) is not None
    store.update_agent(first, cost=0.25)
    await hb(state="stuck", turns=1)  # its turn ended (here: killed at turn_timeout)
    assert store.get_agent(first) is None
    cmds = [CMD.validate_json(c["body"]) for c in store.commands_for(wid)]
    assert UnassignCommand(agent=first) in cmds
    evs = (await client.get(f"/agents/{LEADER}/events")).json()["events"]
    assert evs[-1]["type"] == "child_removed"
    assert evs[-1]["data"] == {"child": first, "type": "analyst", "reason": "ephemeral"}
    assert store.get_swarm("demo")["removed_cost"] == pytest.approx(0.25)
    second = await spawn(client, LEADER, "analyst")  # the cap is free again: a new agent
    assert second.status_code == 201 and second.json()["name"] != first


def test_ephemeral_is_for_spawned_types_only():
    with pytest.raises(ValueError, match="ephemeral: true is for spawned agents only"):
        spawn_spec(analyst={"ephemeral": True, "replicas": 1})
