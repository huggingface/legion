"""Tiny real runs (tests/tinyrun.py): a `compute: local` swarm through its whole lifecycle, and
a controller that crashes mid-loop and comes back."""

from __future__ import annotations

import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def test_lifecycle(tiny):
    """launch -> turn -> status -> update -> stop (hold) -> launch -> stop --release -> launch
    -> stop --delete, each step checked through the CLI as an operator sees it."""
    spec = tiny.spec(replicas=1)
    out = tiny.flock("launch", spec, "--skip-preflight")
    assert "local run " in out and "tiny  agents=0" in out and "gen 1" in out, out
    assert "writer  ok      1" in out, out
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
    evs = tiny.wait(lambda: "agent.turn_ended" in (e := tiny.flock(
        "status", "writer-0000", "--events")) and e)  # events can trail the heartbeat
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
    out = tiny.flock("stop", "tiny", "--type", "writer", "--reason", "lunch")
    assert out.strip() == "tiny: held: type writer", out
    tiny.flock("prompt", "writer-0000", "bash: echo two")
    out = tiny.flock("status", "tiny")
    assert "HELD type writer: lunch (by operator" in out and "pending=1" in out, out
    assert tiny.agent("writer-0000")["turns"] == 1
    assert tiny.flock("launch", "tiny").strip() == "tiny: held: nothing"
    assert tiny.turns("writer-0000", 2)["pending"] == 0
    assert "HELD" not in tiny.flock("status", "tiny")

    # stop --release frees the workers and keeps everything; launch FOLDER resumes the
    # sessions, refused while the folder differs from the running spec (replicas aside)
    out = tiny.flock("stop", "tiny", "--release", "--reason", "night")
    assert "releasing: 1 workers still running" in out, out  # it waited for the worker
    assert "released: every agent synced, compute freed; `flock launch tiny` resumes" in out
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
    assert tiny.said("writer-0000")[-1] == [  # the same session
        "bash: echo one", "bash: echo two", "bash: echo three"]
    assert [e["name"] for e in tiny.events("tiny")] == [
        "swarm.spec_applied", "worker.launched", "swarm.spec_applied",  # launch, update
        "swarm.stopped", "swarm.resumed",  # stop, launch
        "swarm.released", "swarm.stopped", "worker.released", "worker.ended",  # --release
        "swarm.unreleased", "swarm.resumed", "worker.launched"]  # launch
    assert tiny.events("tiny", "swarm.released")[0]["data"]["reason"] == "night"

    # stop --delete ends the local run: the bundle
    out = tiny.flock("stop", "tiny", "--delete", "--yes")
    assert f"local run ended; bundle: {run}/run.json" in out, out
    assert not Path(".flock/current").exists()
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "interrupted"
    assert doc["spec"]["compute"]["kind"] == "local"
    assert {x["name"]: x["turns_completed"] for x in doc["agents"]} == {  # replicas kept: 2
        "writer-0000": 3, "writer-0001": 0}
    assert doc["totals"]["cost_usd"] == round(a["cost_usd"], 6)


LOOP = """\
    loop:
      prompt: "bash: echo round {round}"
      every: 3s
      while: {max_rounds: 10}
  helper:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "1"}
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
    tiny.flock("launch", tiny.spec(replicas=2, extra=LOOP), "--skip-preflight")
    tiny.turns("writer-0000", 1)  # the loop's first round
    tiny.flock("prompt", "writer-0000", SPAWN)
    child = tiny.wait(lambda: tiny.agent("writer-0000")["children"])[0]
    tiny.turns(child, 1)
    writers = ("writer-0000", "writer-0001")
    before = {n: tiny.agent(n) for n in (*writers, child)}
    loop = json.loads(tiny.flock("status", "tiny", "--type", "writer", "--json"))["loop"]
    rounds = {n: loop["per_agent"][n]["rounds"] for n in writers}
    assert loop["state"] == "running" and max(rounds.values()) < 10, loop

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
    # its model, in the same session (the workers died with the controller and were relaunched);
    # no round is sent twice (rounds due while the swarm was down are skipped)
    for n in writers:
        first = tiny.wait(lambda n=n: [e["data"]["round"] for e in tiny.events(n, "loop.prompt")
                                       if datetime.fromisoformat(e["ts"]) > restarted
                                       and e["data"]["round"] > rounds[n]])[0]
        said = tiny.wait(lambda n=n, k=first: [u for u in tiny.said(n)
                                               if u[-1] == f"bash: echo round {k}"])
        assert said[0][0] == "bash: echo round 1", said
        sent = [e["data"]["round"] for e in tiny.events(n, "loop.prompt")]
        assert sent == sorted(set(sent)), sent

    # stop --delete on a controller: the swarm is removed
    assert json.loads(tiny.flock("stop", "tiny", "--delete", "--yes")) == {
        "swarm": "tiny", "removing": True}



LOOPS = """\
    loop:
      prompt: "bash: echo round {round}"
      while: {max_rounds: 3}
      final: {prompt: "final after {round} rounds; last: {last_summary}"}
  ticker:
    harness: llm
    model: fake:fake-model
    replicas: 1
    max_turns: 1
    env: {HEARTBEAT_INTERVAL: "1"}
    loop:
      prompt: "bash: echo tick {round}"
      every: 500ms
      on: [comms.mention]
      while: {until: 5s}
      final: {prompt: "bash: echo bye"}
limits: {budget_usd: 1000}
"""


def test_loops(tiny, monkeypatch):
    """Two loops, each with its final round, then idle: `every: turn_end` with `max_rounds`
    (re-prompted by the worker) and `every: 500ms` with `while.until` (by the controller),
    whose turns all stop at `max_turns: 1`. Then the run's bundle sums them up."""
    monkeypatch.setenv("FLOCK_LOOP_INTERVAL", "0.2")
    out = tiny.flock("launch", tiny.spec(replicas=2, extra=LOOPS), "--skip-preflight")
    # unquoted `on:` is the key "on" (YAML 1.2), and this type is not on the board
    assert "loop.on comms.mention: no comms feeder: the type is not on the board" in out, out
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])

    def loop(t: str) -> dict:
        return json.loads(tiny.flock("status", "tiny", "--type", t, "--json"))["loop"]

    w = tiny.wait(lambda: (x := loop("writer"))["state"] == "done" and x)
    t = tiny.wait(lambda: (x := loop("ticker"))["state"] == "done" and x)
    time.sleep(1)  # then idle: nothing more is sent
    assert (w["mode"], w["reason"], w["rounds_total"], w["final_sent"], w["final_done"]) == (
        "worker", "max_rounds", 6, 2, 2), w
    for n in ("writer-0000", "writer-0001"):
        assert tiny.said(n)[-1] == [  # rendered on the worker, {last_summary}: the last reply
            "bash: echo round 1", "bash: echo round 2", "bash: echo round 3",
            "final after 3 rounds; last: done: round 3\n[exit code 0]"]
        evs = tiny.events(n)
        assert [(e["data"]["kind"], e["data"]["round"], e["data"]["why"]) for e in evs
                if e["name"] == "loop.prompt"] == [("loop", 1, "start"), ("loop", 2, "turn_end"),
                                                   ("loop", 3, "turn_end"),
                                                   ("final", 3, "final: max_rounds")]
        assert [e["data"] for e in evs if e["name"] == "loop.stopped"] == [
            {"reason": "max_rounds", "round": 3, "active": True}]
        seq = [e["name"] for e in evs
               if e["name"] in ("loop.prompt", "agent.turn_started", "agent.turn_ended")]
        assert seq == ["loop.prompt", "agent.turn_started", "agent.turn_ended"] * 4  # one each

    # the controller's loop: a round every 500 ms until 5 s after the loop started, then the
    # final round; a turn stopped at max_turns is no loop error
    assert (t["mode"], t["reason"], t["final_done"]) == ("controller", "until", 1), t
    k = t["rounds_total"]
    assert k >= 2 and tiny.said("ticker-0000")[-1] == [
        *(f"bash: echo tick {i}" for i in range(1, k + 1)), "bash: echo bye"]
    assert [e["data"]["why"] for e in tiny.events("ticker-0000", "loop.prompt")] == [
        "start", *["every 500ms"] * (k - 1), "final: until"]

    swarm = [(e["name"], e["data"]) for e in tiny.events("tiny", "loop.")
             if e["name"] not in ("loop.tick", "loop.max_turns")]
    for typ, why, n in (("writer", "max_rounds", 2), ("ticker", "until", 1)):
        assert [x for x, d in swarm if d["type"] == typ] == [
            "loop.started", "loop.stopped", *["loop.final_round"] * n, "loop.final_done"], swarm
        assert next(d for x, d in swarm if x == "loop.stopped" and d["type"] == typ)[
            "reason"] == why

    # the bundle: events.jsonl merges the swarm stream with the agents' by time, each event
    # once; run.json sums up each loop and the budget
    tiny.flock("stop", "tiny", "--delete", "--yes")
    merged = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    agents = ("writer-0000", "writer-0001", "ticker-0000")
    per_agent = sum(len((run / "events" / f"{a}.jsonl").read_text().splitlines()) for a in agents)
    assert len(merged) == len([e for e in merged if e["agent"] == ""]) + per_agent
    stamps = [datetime.fromisoformat(e["ts"]) for e in merged]
    assert stamps == sorted(stamps)
    assert sorted(e["agent"] for e in merged if e["type"] == "final_round") == sorted(agents)
    assert [(e["agent"], e["data"]["kind"], e["data"]["round"]) for e in merged
            if e["type"] == "loop_max_turns"] == [
        *(("ticker-0000", "loop", i) for i in range(1, k + 1)), ("ticker-0000", "final", k)]
    doc = json.loads((run / "run.json").read_text())
    assert doc["spec"]["types"]["ticker"]["loop"]["on"] == ["comms.mention"]
    lp = doc["loops"]["writer"]
    assert (lp["state"], lp["reason"], lp["rounds_total"], lp["final_sent"], lp["final_done"],
            lp["wakes_total"]) == ("done", "max_rounds", 6, 2, 2, 0), lp
    assert (doc["loops"]["ticker"]["state"], doc["loops"]["ticker"]["reason"]) == ("done", "until")
    assert doc["budget"] == {"budget_usd": 1000.0, "final_reserve_usd": 0.0,
                             "cost_usd": doc["totals"]["cost_usd"], "stop": None}


SPAWNING = """\
  helper:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "1"}
  scout:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "1"}
spawn:
  writer: {types: {helper: 1}, depth: 2}
  helper: {types: [scout], max: 1}
"""
BIN = Path(sys.executable).parent  # flock and flock-agent, as an agent's bash finds them


def test_spawn(tiny):
    """writer-0000 spawns a helper with a first prompt (one helper at most); the helper runs it
    as itself and reports to its parent, then spawns a scout; removing the helper takes the
    scout with it."""
    tiny.flock("launch", tiny.spec(extra=SPAWNING), "--skip-preflight")
    helper, scout = "writer-0000-helper-0001", "writer-0000-helper-0001-scout-0002"
    tiny.flock("prompt", "writer-0000", f"bash: {BIN}/flock-agent spawn helper --prompt "
               f"""'bash: {BIN}/flock prompt $FLOCK_PARENT "from $FLOCK_AGENT [$FLOCK_TOKEN]"'""")
    tiny.turns("writer-0000", 1)
    assert tiny.turns(helper, 1)["parent"] == "writer-0000"
    assert tiny.events(helper, "agent.session_started")[0]["data"]["parent"] == "writer-0000"
    assert tiny.events("writer-0000", "agent.session_started")[0]["data"]["parent"] is None
    tiny.turns("writer-0000", 2)
    assert tiny.said("writer-0000")[-1][-1] == f"from {helper} []"  # never the worker's token
    tiny.flock("prompt", "writer-0000", f"bash: {BIN}/flock-agent spawn helper")
    tiny.turns("writer-0000", 3)
    assert tiny.results("writer-0000")[-1].startswith('error 409: {"reason":"max_children"')

    tiny.flock("prompt", helper, f"bash: {BIN}/flock-agent spawn scout --prompt 'bash: echo hi'")
    tiny.turns(helper, 2)
    tiny.turns(scout, 1)
    assert tiny.said(scout)[-1] == ["bash: echo hi"]
    w, h, s = (tiny.agent(n) for n in ("writer-0000", helper, scout))
    assert (w["children"], w["depth"], w["parent"]) == ([helper], 2, None), w
    assert (h["children"], h["depth"], h["parent"]) == ([scout], 1, "writer-0000"), h
    assert (s["children"], s["depth"], s["parent"]) == ([], 0, helper), s

    tiny.flock("prompt", "writer-0000", f"bash: {BIN}/flock-agent remove {helper}")
    tiny.turns("writer-0000", 4)
    assert json.loads(tiny.results("writer-0000")[-1].split("\n")[0]) == {
        "removed": [scout, helper]}
    tiny.flock("status", helper, code=1)
    tiny.flock("status", scout, code=1)
    assert tiny.agent("writer-0000")["children"] == []
    assert [e["data"] for e in tiny.events("writer-0000", "agent.child_removed")] == [
        {"child": helper, "type": "helper", "reason": "removed"}]
    assert [e["data"] for e in tiny.events(helper, "agent.child_removed")] == [
        {"child": scout, "type": "scout", "reason": "parent_removed"}]
    types = json.loads(tiny.flock("status", "tiny", "--json"))["status"]["types"]
    assert types["helper"]["agents_by_state"] == types["scout"]["agents_by_state"] == {}
