"""A world end to end under `flock run` (docs/world.md): the real CLI in a subprocess runs
examples/world-demo/spec.yaml (pointed at tests/fake_openai.py and a copy of worlds/taskboard):
the controller starts the world as a child process, two `llm` agents call `claim_task` through
the worker, one gets the task, the other reads who holds it, and the world stops with the run."""

import json
import shutil
import sqlite3
import subprocess

import httpx
import pytest
import yaml

from . import fake_openai
from .test_local_run_e2e import _bundle_events, _env, _flock, _run_dir
from .test_world import ROOT, TASKBOARD

AGENTS = ["worker-0000", "worker-0001"]


@pytest.fixture
def fake_server(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    srv.url = f"http://127.0.0.1:{srv.server_port}/v1"
    yield srv
    srv.shutdown()


def _tool_results(run, agent: str) -> list[str]:
    return [json.dumps(e["data"]["message"]["content"]) for e in _bundle_events(run, agent)
            if e["type"] == "message" and e["data"]["message"].get("role") == "toolResult"]


def test_flock_run_starts_the_world_and_agents_share_its_claims(tmp_path, fake_server):
    board = tmp_path / "taskboard"
    shutil.copytree(TASKBOARD, board, ignore=shutil.ignore_patterns("*.db*", "__pycache__"))
    spec = yaml.safe_load((ROOT / "examples" / "world-demo" / "spec.yaml").read_text())
    spec["providers"]["fake"]["base_url"] = fake_server.url
    spec["world"]["dir"] = str(board)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(spec))
    p = subprocess.run(
        _flock("run", "spec.yaml", "--prompt", 'call: claim_task {"name": "greeting"}',
               "--until", "idle", "--skip-preflight"),
        cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True, timeout=90,
        check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "2 agents, 2 turns" in p.stdout, p.stdout
    run = _run_dir(tmp_path)

    results = {a: _tool_results(run, a) for a in AGENTS}
    assert all(len(r) == 1 for r in results.values()), results
    winners = [a for a, r in results.items() if '\\"status\\": \\"claimed\\"' in r[0]]
    assert len(winners) == 1, results
    (loser,) = set(AGENTS) - set(winners)
    assert f"'greeting' is claimed by {winners[0]}" in results[loser][0]

    # the world's state is in its dir, with flock's identity on each change
    db = sqlite3.connect(board / "taskboard.db")
    assert db.execute("SELECT owner FROM tasks WHERE name = 'greeting'").fetchone() == (winners[0],)
    (agent, turn, req), = db.execute(
        "SELECT agent, turn, request_id FROM events WHERE action = 'claim'").fetchall()
    assert agent == winners[0] and turn == "t-1" and len(req) == 32
    # the model saw the world's tools next to bash, and the World section
    reqs = [json.loads(x)["body"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    offered = {t["function"]["name"] for t in reqs[0]["tools"]}
    assert {"bash", "list_tasks", "claim_task", "finish_task"} <= offered
    assert "## World: taskboard (version 0.1.0)" in reqs[0]["messages"][0]["content"]

    log = (run / "logs" / "world-world-demo.log").read_text()
    assert "world taskboard 0.1.0 on http://127.0.0.1:" in log
    url = log.split(" on ", 1)[1].split()[0]
    with pytest.raises(httpx.ConnectError):  # stopped with the run
        httpx.get(f"{url}/manifest", timeout=2)
    doc = json.loads((run / "run.json").read_text())
    assert doc["spec"]["world"]["manifest"]["name"] == "taskboard"
