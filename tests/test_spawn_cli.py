"""The flock CLI as an agent runs it: FLOCK_AGENT_TOKEN, FLOCK_AGENT, spawn/status/events/
prompt/remove."""

from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli
from flock.controller import spawn as spawning

from .conftest import TOKEN
from .test_spawn_api import LEADER, spawn_spec

runner = CliRunner()


@pytest.fixture
def agent_cli(monkeypatch, app, store, tick):
    """Apply the spawn spec as the operator, then return a function that switches the CLI to
    one agent's identity the way the worker sets it up."""
    with TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}) as admin:
        r = admin.put("/swarms/demo", json=spawn_spec().model_dump(mode="json"))
        assert r.status_code == 200, r.text
    tick()

    def become(name: str | None) -> None:
        monkeypatch.delenv("FLOCK_TOKEN", raising=False)
        if name is None:
            monkeypatch.delenv("FLOCK_AGENT_TOKEN", raising=False)
            monkeypatch.delenv("FLOCK_AGENT", raising=False)
            monkeypatch.setenv("FLOCK_TOKEN", TOKEN)
        else:
            monkeypatch.setenv("FLOCK_AGENT_TOKEN", spawning.agent_token(store, name, "demo", 0))
            monkeypatch.setenv("FLOCK_AGENT", name)
            monkeypatch.setenv("FLOCK_TOKEN", "")  # what the worker leaves in the harness env
        monkeypatch.setattr(cli, "client", lambda: TestClient(app, headers=_auth()))

    def _auth() -> dict:
        tok = os.environ.get("FLOCK_AGENT_TOKEN") or os.environ.get("FLOCK_TOKEN", "")
        return {"Authorization": f"Bearer {tok}"}

    return become


def run(*args: str):
    return runner.invoke(cli.agent_app if args[0] in ("spawn", "remove") else cli.app, list(args))


def test_client_prefers_the_agent_token(monkeypatch):
    monkeypatch.setenv("FLOCK_TOKEN", "swarm")
    monkeypatch.setenv("FLOCK_AGENT_TOKEN", "agent")
    assert cli.client().headers["authorization"] == "Bearer agent"
    monkeypatch.setenv("FLOCK_AGENT_TOKEN", "")
    assert cli.client().headers["authorization"] == "Bearer swarm"


def test_agent_spawns_prompts_and_removes(agent_cli):
    agent_cli(LEADER)
    r = run("spawn", "analyst", "--prompt", "count the files", "--key", "a1")
    assert r.exit_code == 0, r.output
    child = r.output.strip()
    assert child == "leader-0000-analyst-0001"
    r = run("spawn", "analyst", "--key", "a1", "--json")
    assert json.loads(r.output) == {"name": child, "created": False, "parent": LEADER,
                                    "type": "analyst", "depth": 1}

    r = run("status")  # yourself
    assert json.loads(r.output)["children"] == [child]
    r = run("status", child)
    st = json.loads(r.output)
    assert st["parent"] == LEADER and st["pending"] == 1

    r = run("events", LEADER, "--wait", "--timeout", "0")
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout.splitlines()[-1])["type"] == "spawned"

    r = run("spawn", "leader")
    assert r.exit_code == 1 and "type_not_allowed" in r.output

    # the child reports back to its parent, and cannot touch its sibling or the swarm
    sib = run("spawn", "analyst").output.strip()
    agent_cli(child)
    assert run("prompt", LEADER, "done: 42 files").exit_code == 0
    assert run("status", sib).exit_code == 1
    assert run("remove", sib).exit_code == 1
    assert run("status", "demo").exit_code == 1  # names are agents here, and demo is not one

    agent_cli(LEADER)
    r = run("remove", child)
    assert r.exit_code == 0 and json.loads(r.output) == {"removed": [child]}


def test_old_flock_spawn_and_remove_still_work_inside_an_agent(agent_cli):
    """Agents of runs started before flock-agent call `flock spawn|remove` (hidden aliases)."""
    agent_cli(LEADER)
    r = runner.invoke(cli.app, ["spawn", "analyst", "--key", "old"])
    assert r.exit_code == 0, r.output
    child = r.output.strip()
    assert child.startswith("leader-0000-analyst-")
    r = runner.invoke(cli.app, ["remove", child])
    assert json.loads(r.output) == {"removed": [child]}
    assert "spawn" not in runner.invoke(cli.app, ["--help"]).output


def test_operator_names_swarms_or_agents(agent_cli):
    agent_cli(None)
    r = run("spawn", "analyst", "--parent", LEADER)
    child = r.output.strip()
    assert r.exit_code == 0 and child.startswith("leader-0000-analyst-")
    assert "leader" in run("status", "demo").output  # a swarm: the table
    assert json.loads(run("status", child).output)["parent"] == LEADER  # not a swarm: an agent
    r = run("agents", "demo", "--parent", LEADER)
    assert child in r.output and "leader-0000 " not in r.output
    assert run("spawn", "analyst").exit_code == 1  # no FLOCK_AGENT, no --parent
    assert json.loads(run("remove", child).output) == {"removed": [child]}
    r = runner.invoke(cli.app, ["stop", "demo", "--delete", "--yes"])
    assert json.loads(r.output) == {"swarm": "demo", "removing": True}
