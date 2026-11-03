"""Parent-scoped bulk prompt (`POST /agents/{name}/children/prompts`, `flock prompt
--children`): a leader prompts its 100 children in one request with its own token."""

from __future__ import annotations

from datetime import timedelta

from typer.testing import CliRunner

from flock import cli
from flock.client import Client
from flock.controller import spawn as spawning
from flock.models import AgentHeartbeat, PromptCommand, WorkerHeartbeat

from .test_spawn_api import LEADER, apply, as_agent, spawn, spawn_spec

runner = CliRunner()


async def _children(client, store, tick, clock, n=3):
    await apply(client, spawn_spec())  # leader may have 3 analysts and 2 helpers
    tick()
    kids = [(await spawn(client, LEADER, "analyst")).json()["name"] for _ in range(n)]
    helper = (await spawn(client, LEADER, "helper")).json()["name"]
    tick()  # place them
    by_worker: dict[str, list[str]] = {}
    for a in store.list_agents("demo"):
        if a["parent"] == LEADER and a["worker"]:
            by_worker.setdefault(a["worker"], []).append(a["name"])
    for wid, names in by_worker.items():  # every child reports idle
        body = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=1),
                               agents=[AgentHeartbeat(name=x, state="idle") for x in names])
        assert (await client.post(f"/workers/{wid}/heartbeat",
                                  json=body.model_dump(mode="json"))).status_code == 200
    return kids, helper


async def test_leader_prompts_its_idle_children_once(client, app, store, tick, clock):
    kids, helper = await _children(client, store, tick, clock)
    store.add_prompt(kids[0], PromptCommand(agent=kids[0], prompt_id="x", text="t").model_dump_json(), 0)
    tok = spawning.agent_token(store, LEADER, "demo", 0)
    body = {"type": "analyst", "state": "idle", "text": "go"}
    async with as_agent(app, tok) as c:
        ack = (await c.post(f"/agents/{LEADER}/children/prompts", json=body)).json()
        assert ack == {"swarm": "demo", "type": "analyst", "parent": LEADER, "matched": 3,
                       "queued": 2, "pending": 0, "skipped_state": 0, "skipped_removing": 0,
                       "skipped_pending": 1}
        again = (await c.post(f"/agents/{LEADER}/children/prompts", json=body)).json()
        assert (again["queued"], again["skipped_pending"]) == (0, 3)  # nothing stacks
        everyone = (await c.post(f"/agents/{LEADER}/children/prompts",
                                 json={"text": "all", "skip_pending": False})).json()
        assert everyone["matched"] == 4 and everyone["type"] is None  # helper included
    assert store.pending_counts("demo") == {kids[0]: 2, kids[1]: 2, kids[2]: 2, helper: 1}
    # the swarm token works too
    r = await client.post(f"/agents/{LEADER}/children/prompts", json={"text": "x", "state": "idle"})
    assert r.status_code == 200 and r.json()["skipped_pending"] == 4


async def test_children_prompt_refused_for_other_tokens(client, app, store, tick, clock):
    kids, _ = await _children(client, store, tick, clock, n=2)
    grandchild = (await spawn(client, kids[0], "helper")).json()["name"]
    for who in (kids[0], kids[1]):  # a child cannot bulk-prompt its parent's children
        async with as_agent(app, spawning.agent_token(store, who, "demo", 0)) as c:
            r = await c.post(f"/agents/{LEADER}/children/prompts", json={"text": "x"})
            assert r.status_code == 403
    async with as_agent(app, spawning.agent_token(store, kids[1], "demo", 0)) as c:  # a sibling's
        assert (await c.post(f"/agents/{kids[0]}/children/prompts", json={"text": "x"})).status_code == 403
    async with as_agent(app, spawning.agent_token(store, LEADER, "demo", 0)) as c:  # its subtree
        r = await c.post(f"/agents/{kids[0]}/children/prompts", json={"text": "x"})
        assert r.status_code == 200 and r.json()["matched"] == 1
    assert store.pending_counts("demo")[grandchild] == 1
    assert (await client.post("/agents/nobody-0000/children/prompts", json={"text": "x"})).status_code == 404


def test_cli_prompt_children(monkeypatch, app, store):
    sent = []

    def fake_call(self, method, path, **kw):
        sent.append((method, path, kw["json"]))
        return {"matched": 0}

    monkeypatch.setattr(Client, "_call", fake_call)
    monkeypatch.setenv("FLOCK_AGENT", LEADER)
    r = runner.invoke(cli.app, ["prompt", "--children", "--type", "analyst", "--state", "idle", "go"])
    assert r.exit_code == 0, r.output
    assert sent[-1] == ("POST", f"/agents/{LEADER}/children/prompts",
                        {"timeout": None, "reply_to": None, "type": "analyst", "state": "idle",
                         "text": "go"})
    r = runner.invoke(cli.app, ["prompt", "--children", "--parent", "x-0001", "hi"])
    assert r.exit_code == 0 and sent[-1][1] == "/agents/x-0001/children/prompts"
    monkeypatch.delenv("FLOCK_AGENT")
    assert runner.invoke(cli.app, ["prompt", "--children", "hi"]).exit_code == 2  # no parent
    assert runner.invoke(cli.app, ["prompt", "--children", "a-0001", "hi"]).exit_code == 2
