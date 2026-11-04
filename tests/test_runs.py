"""Tiny real runs (tests/tinyrun.py): a `compute: local` swarm through its whole lifecycle, and
a controller that crashes mid-loop and comes back."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path


def test_lifecycle(tiny):
    """launch -> turn -> status -> update -> stop (hold) -> launch -> stop --release -> launch
    -> stop --delete, each step checked through the CLI as an operator sees it."""
    spec = tiny.spec(replicas=1)
    out = tiny.flock("launch", spec, "--skip-preflight")
    assert "local run " in out and "tiny  agents=0" in out and "gen 1" in out, out
    assert "warning: fake:fake-model: unverified (preflight skipped)" in out
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])

    # a turn: the agent's own bash ran, the events say so, the cost is priced
    ack = json.loads(tiny.flock("prompt", "writer-0000", "bash: echo one"))
    assert ack["agent"] == "writer-0000" and ack["prompt_id"], ack
    tiny.flock("prompt", "nobody-0001", "x", code=1)
    a = tiny.turns("writer-0000", 1)
    session = a["session_id"]
    assert session and a["resumed"] is False and a["cost_usd"] > 0, a
    assert a["worker"] == "tiny-writer-w0000"
    evs = tiny.flock("status", "writer-0000", "--events")
    for line in ("agent.session_started", "user     bash: echo one", "tool     bash: one",
                 "agent.turn_ended completed fake-model"):
        assert line in evs, evs
    turn = tiny.events("writer-0000", "agent.turn")
    assert [(e["name"], e["data"]["turn_id"]) for e in turn] == [
        ("agent.turn_started", "t-1"), ("agent.turn_ended", "t-1")], turn
    st = tiny.wait(lambda: (s := json.loads(tiny.flock("status", "tiny", "--json")))
                   ["verdict"]["level"] == "ok" and s)
    t = st["status"]["types"]["writer"]
    assert t["agents_by_state"] == {"idle": 1} and t["workers_by_state"] == {"running": 1}, t
    assert st["status"]["cost_usd"] == a["cost_usd"]
    assert "OK" in tiny.flock("status", "tiny") and "tiny  agents=1" in tiny.flock("status")

    # update: one more replica joins the running worker (agents_per_worker 2)
    tiny.spec(replicas=2)
    assert tiny.flock("update", spec, "--replicas", "spec").strip() == \
        "types.writer.replicas: 1 -> 2"
    b = tiny.wait(lambda: (b := tiny.agent("writer-0001"))["state"] == "idle" and b)
    assert b["worker"] == "tiny-writer-w0000"
    out = tiny.flock("status", "tiny", "--type", "writer")
    assert "gen 2" in tiny.flock("status", "tiny") and out.count("writer-000") == 2, out

    # stop holds: a prompt waits, launch SWARM resumes and it is delivered
    tiny.flock("stop", "tiny", "--type", "nope", code=1)
    assert tiny.flock("stop", "tiny", "--reason", "lunch").strip() == "tiny: held: type writer"
    tiny.flock("prompt", "writer-0000", "bash: echo two")
    out = tiny.flock("status", "tiny")
    assert "HELD type writer: lunch (by operator" in out and "pending=1" in out, out
    assert tiny.agent("writer-0000")["turns"] == 1
    assert tiny.flock("launch", "tiny").strip() == "tiny: held: nothing"
    assert tiny.turns("writer-0000", 2)["pending"] == 0
    assert "HELD" not in tiny.flock("status", "tiny")

    # stop --release frees the workers and keeps everything; launch FOLDER resumes the
    # sessions, refused while the folder differs from the running spec (replicas aside)
    out = tiny.flock("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed; `flock launch tiny` resumes" in out, out
    assert "controller kept: a local run keeps its controller" in out
    out = tiny.flock("status", "tiny")
    assert "RELEASED since" in out and "workers: dead:1" in out, out
    tiny.spec(replicas=1, extra='    system_prompt: "new"\n')
    assert "run `flock update` first" in tiny.flock("launch", spec, "--skip-preflight", code=1)
    tiny.spec(replicas=1)
    assert "tiny: held: nothing" in tiny.flock("launch", spec, "--skip-preflight")
    tiny.flock("prompt", "writer-0000", "bash: echo three")
    a = tiny.turns("writer-0000", 3)
    assert (a["session_id"], a["resumed"]) == (session, True), a
    users = [m["content"] for m in tiny.requests("writer-0000")[-1]["messages"]
             if m["role"] == "user"]
    assert users == ["bash: echo one", "bash: echo two", "bash: echo three"]  # the same session
    assert [e["name"] for e in tiny.events("tiny")] == [
        "swarm.spec_applied", "worker.launched", "swarm.spec_applied",  # launch, update
        "swarm.stopped", "swarm.resumed",  # stop, launch
        "swarm.released", "swarm.stopped", "worker.released", "worker.ended",  # --release
        "swarm.unreleased", "swarm.resumed", "worker.launched"]  # launch

    # stop --delete ends the local run: the bundle
    out = tiny.flock("stop", "tiny", "--delete", "--yes")
    assert f"local run ended; bundle: {run}/run.json" in out, out
    assert not Path(".flock/current").exists()
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "interrupted" and doc["spec"]["compute"] == {"kind": "local"}
    assert {x["name"]: x["turns_completed"] for x in doc["agents"]} == {  # replicas kept: 2
        "writer-0000": 3, "writer-0001": 0}
    assert doc["totals"]["cost_usd"] == a["cost_usd"]


# shortcut: the workers outlive their controller, as on Slurm or HF Jobs (a local worker dies
# with it, and its restarted controller relaunches it only after UNKNOWN_AFTER, 90 s)
SURVIVE = 'FLOCK_LOCAL_PARENT_PID: ""'
LOOP = """\
    loop:
      prompt: "bash: echo round {round}"
      every: 3s
      while: {max_rounds: 5}
  helper:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "1", FLOCK_LOCAL_PARENT_PID: ""}
spawn:
  writer: {types: {helper: 1}, depth: 1}
"""
SPAWN = f"bash: {Path(sys.executable).parent}/flock-agent spawn helper --prompt 'bash: echo hi'"


def test_controller_restart(tiny, tmp_path):
    """A controller killed mid-loop (SIGKILL: no final flush) and started again on the same
    state keeps its swarm (turns, cost, spawn links, sessions), and its loop goes on: the next
    round reaches each agent."""
    data = tmp_path / "ctl"
    port = tiny.controller(data)
    tiny.flock("launch", tiny.spec(replicas=2, extra=LOOP, env=SURVIVE), "--skip-preflight")
    tiny.turns("writer-0000", 1)  # the loop's first round
    tiny.flock("prompt", "writer-0000", SPAWN)
    child = tiny.wait(lambda: tiny.agent("writer-0000")["children"])[0]
    tiny.turns(child, 1)
    writers = ("writer-0000", "writer-0001")
    before = {n: tiny.agent(n) for n in (*writers, child)}
    loop = json.loads(tiny.flock("status", "tiny", "--type", "writer", "--json"))["loop"]
    rounds = {n: loop["per_agent"][n]["rounds"] for n in writers}
    assert loop["state"] == "running" and max(rounds.values()) < 5, loop

    tiny.crash()
    restarted = datetime.now(UTC)
    tiny.controller(data, port)
    st = json.loads(tiny.flock("status", "tiny", "--json"))["status"]
    assert st["cost_usd"] >= sum(a["cost_usd"] for a in before.values()) > 0, st
    for n, a in before.items():
        now = tiny.agent(n)
        assert now["session_id"] == a["session_id"] and now["turns"] >= a["turns"] >= 1, now
        assert now["cost_usd"] >= a["cost_usd"] > 0, now
    assert tiny.agent(child)["parent"] == "writer-0000"
    assert tiny.agent("writer-0000")["children"] == [child]
    assert child in tiny.flock("status", "tiny", "--parent", "writer-0000")

    # the loop engine goes on: a round past the one before the crash reaches each writer and
    # its model, in the same session; no round is sent twice
    for n in writers:
        first = tiny.wait(lambda n=n: [e["data"]["round"] for e in tiny.events(n, "loop.prompt")
                                       if datetime.fromisoformat(e["ts"]) > restarted
                                       and e["data"]["round"] > rounds[n]])[0]
        users = tiny.wait(lambda n=n, k=first: (u := [
            m["content"] for m in tiny.requests(n)[-1]["messages"] if m["role"] == "user"])
            and u[-1] == f"bash: echo round {k}" and u)
        assert users[0] == "bash: echo round 1", users
        sent = [e["data"]["round"] for e in tiny.events(n, "loop.prompt")]
        assert sent == list(range(1, len(sent) + 1)), sent

    # stop --delete on a controller: the swarm is removed
    assert json.loads(tiny.flock("stop", "tiny", "--delete", "--yes")) == {
        "swarm": "tiny", "removing": True}
