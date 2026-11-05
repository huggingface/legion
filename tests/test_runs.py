"""Tiny real runs (tests/tinyrun.py): a `compute: local` swarm through its whole lifecycle, a
controller that crashes mid-loop and comes back, one run per feature: loops, spawn, the tools
folder, comms on the fake board (tests/fake_msgboard.py), one turn per harness; and the remote
backends on fakes: Slurm (tests/fixtures/slurm_stubs) and HF Jobs (tests/fake_hub.py)."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from flock.controller.jobs import BOOTSTRAP_IMAGE, bootstrap_command
from flock.controller.slurm import read_env_file
from flock.tools.bundle import pack

from . import fake_hub, fake_msgboard
from .test_worker_adapters import FAKE_APP_SERVER


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


# appended under types.writer
RESTART_LOOP = """\
    loop:
      prompt: "bash: echo round {round}"
      every: 3s
      while: {max_rounds: 10}
  helper:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "0.5"}
spawn:
  writer: {types: {helper: 1}, depth: 1}
"""
BIN = Path(sys.executable).parent  # flock and flock-agent, as an agent's bash finds them
SPAWN = f"bash: {BIN}/flock-agent spawn helper --prompt 'bash: echo hi'"


def test_controller_restart(tiny, tmp_path):
    """A controller killed mid-loop (SIGKILL: no final flush) and started again on the same
    state keeps its swarm (turns, cost, spawn links, sessions), and its loop goes on: the next
    round reaches each agent."""
    data = tmp_path / "ctl"
    port = tiny.controller(data)
    tiny.flock("launch", tiny.spec(replicas=2, extra=RESTART_LOOP), "--skip-preflight")
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



# appended under types.writer
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
    env: {HEARTBEAT_INTERVAL: "0.5"}
    loop:
      prompt: "bash: echo tick {round}"
      every: 500ms
      on: [comms.mention]
      while: {until: 4s}
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

    # the controller's loop: a round every 500 ms until 4 s after the loop started, then the
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
    assert sorted((e["agent"], e["data"]["kind"], e["data"]["round"]) for e in merged
                  if e["type"] == "loop_max_turns") == [
        ("ticker-0000", "final", k), *(("ticker-0000", "loop", i) for i in range(1, k + 1))]
    doc = json.loads((run / "run.json").read_text())
    assert doc["spec"]["types"]["ticker"]["loop"]["on"] == ["comms.mention"]
    lp = doc["loops"]["writer"]
    assert (lp["state"], lp["reason"], lp["rounds_total"], lp["final_sent"], lp["final_done"],
            lp["wakes_total"]) == ("done", "max_rounds", 6, 2, 2, 0), lp
    assert (doc["loops"]["ticker"]["state"], doc["loops"]["ticker"]["reason"]) == ("done", "until")
    assert doc["budget"] == {"budget_usd": 1000.0, "final_reserve_usd": 0.0,
                             "cost_usd": doc["totals"]["cost_usd"], "stop": None}


# appended under types.writer
SPAWNING = """\
  helper:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "0.5"}
  scout:
    harness: llm
    model: fake:fake-model
    replicas: 0
    workers: {max_extra: 1}
    env: {HEARTBEAT_INTERVAL: "0.5"}
spawn:
  writer: {types: {helper: 1}, depth: 2}
  helper: {types: [scout], max: 1}
"""


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


ADD = '''
def add(a: int, b: int) -> int:
    """Add two integers and return the sum."""
    return a + b
'''
MUL = ADD + '''

def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
'''


def test_tools_folder(tiny, tmp_path):
    """`flock launch` ships the folder's tools/ (a turn calls `add`); the folder changes (a new
    `mul`, a new system prompt, every tool selected) and `flock update` takes it to the running
    worker, whose next turn uses it, with a note that the system prompt changed."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "prompts").mkdir()
    (tmp_path / "tools" / "calc.py").write_text(ADD)
    (tmp_path / "prompts" / "system.md").write_text("You are calc v1.")
    folder = "    system_prompt_file: prompts/system.md\n    tools: {}\n"
    v1 = pack(tmp_path / "tools").version
    out = tiny.flock("launch", tiny.spec(extra=folder.format("[add]")), "--skip-preflight")
    assert f"tools: version {v1} (" in out and "): add" in out, out
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])

    def turn(n: int) -> tuple[list[dict], list[dict]]:
        """Turn t-n: its turn_started/turn_ended data and its messages."""
        evs = [e for e in tiny.events("writer-0000") if e["data"].get("turn_id") == f"t-{n}"]
        return ([e["data"] for e in evs if e["name"].startswith("agent.turn_")],
                [e["data"]["message"] for e in evs if e["name"] == "agent.message"])

    tiny.flock("prompt", "writer-0000", 'call: add {"a": 2, "b": 40}')
    tiny.turns("writer-0000", 1)
    ends, msgs = turn(1)
    assert [d["tools_version"] for d in ends] == [v1, v1] and ends[1]["status"] == "completed"
    call, res, final = msgs[1:]
    assert call["content"][0] == {"type": "toolCall", "id": "call_0", "name": "add",
                                  "arguments": {"a": 2, "b": 40}}
    assert (res["toolName"], res["content"][0]["text"], res["toolsVersion"], res["isError"]) == (
        "add", "42", v1, False)
    assert final["content"] == [{"type": "text", "text": "done: 42"}]
    reqs = [json.loads(x)["body"] for x in tiny.log.read_text().splitlines()]
    offered = {t["function"]["name"]: t["function"] for t in reqs[0]["tools"]}
    assert set(offered) == {"bash", "add"}
    assert offered["add"]["description"] == "Add two integers and return the sum."
    assert offered["add"]["parameters"] == {
        "type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"], "additionalProperties": False}
    assert "You are calc v1." in reqs[0]["messages"][0]["content"]
    assert (reqs[1]["messages"][-1]["role"], reqs[1]["messages"][-1]["content"]) == (
        "tool", "42")  # the model got the result
    assert (run / "agents" / "_tools" / v1 / "calc.py").is_file()  # the worker's, not the agent's

    (tmp_path / "tools" / "calc.py").write_text(MUL)
    (tmp_path / "prompts" / "system.md").write_text("You are calc v2.")
    tiny.spec(extra=folder.format("all"))
    v2 = pack(tmp_path / "tools").version
    assert tiny.flock("update", "spec.yaml").splitlines() == [
        f"tools: {v1} -> {v2} (+mul)", "prompts changed: writer (system_prompt)",
        'types.writer.tools: ["add"] -> all']
    log = run / "logs" / "tiny-writer-w0000.log"  # the same heartbeat brings the new spec
    tiny.wait(lambda: f"tools version {v2} ready" in log.read_text())
    tiny.flock("prompt", "writer-0000", 'call: mul {"a": 6, "b": 7}')
    tiny.turns("writer-0000", 2)
    ends, msgs = turn(2)
    assert [d["tools_version"] for d in ends] == [v2, v2]
    res = msgs[2]
    assert (res["toolName"], res["content"][0]["text"], res["toolsVersion"]) == ("mul", "42", v2)
    req = [json.loads(x)["body"] for x in tiny.log.read_text().splitlines()][-2]  # the call
    assert "You are calc v2." in req["messages"][0]["content"]
    assert "You are calc v1." not in req["messages"][0]["content"]
    assert req["messages"][-1] == {"role": "user", "content": (
        '[flock] Since your last turn: your system prompt changed.\n\ncall: mul {"a": 6, "b": 7}')}
    assert {t["function"]["name"] for t in req["tools"]} == {"bash", "add", "mul"}
    assert [e["name"] for e in tiny.events("tiny", "worker.")] == ["worker.launched"]  # no restart


# appended under types.writer
COMMS = """\
    comms: true
    loop: {prompt: "bash: echo round {round} {why}", every: 1h, on: [comms.mention],
           while: {max_rounds: 5}}
  quiet:
    harness: llm
    model: fake:fake-model
    replicas: 1
    comms: true
    env: {HEARTBEAT_INTERVAL: "0.5"}
    loop: {prompt: "bash: echo round {round} {why}", every: 1h, while: {max_rounds: 5}}
comms: {kind: msgboard, url: "URL", invite_env: TINY_INVITE, channel: flock-test,
        kickoff_file: prompts/kickoff.md, protocols: [protocols/lead.md],
        rules: {min_post_gap: GAP, max_len: 500, threads: per_topic, no_group_mentions: true}}
"""


@pytest.fixture
def board(monkeypatch):
    """tests/fake_msgboard.py on a free port; its invite in TINY_INVITE (the workers' secret)."""
    b, srv, b.url = fake_msgboard.serve()
    monkeypatch.setenv("TINY_INVITE", b.invite)
    yield b
    srv.shutdown()


def test_comms(tiny, board, tmp_path):
    """Two writers and a quiet agent on the fake board: the kickoff once, joins at the first
    prompt, a post, a fast second post refused, read, a reply in the thread that mentions two
    agents (only the type with `loop.on: [comms.mention]` wakes), search; then new protocol text
    and no post gap reach the running workers."""
    a, b = "writer-0000", "writer-0001"
    (tmp_path / "prompts").mkdir()
    (tmp_path / "protocols").mkdir()
    (tmp_path / "prompts" / "kickoff.md").write_text("Kickoff: build the bot.")
    (tmp_path / "protocols" / "lead.md").write_text("PROTO-V1: one agent leads a topic.")

    def spec(gap: str) -> str:
        return tiny.spec(replicas=2, extra=COMMS.replace("URL", board.url).replace("GAP", gap),
                         env='FLOCK_COMMS_RATE: "50"')  # a worker's board calls, per second

    out = tiny.flock("launch", spec("30s"), "--skip-preflight")
    assert "no comms feeder" not in out, out
    [kick] = board.msgs
    assert (kick["sender"]["external_id"], kick["body"]) == ("tiny-host", "Kickoff: build the bot.")
    assert board.joins == ["tiny-host"]
    for n in (a, b, "quiet-0000"):  # the loop's first round: each joins at its first prompt
        tiny.turns(n, 1)
    assert sorted(board.joins) == sorted(["tiny-host", a, b, "quiet-0000"])
    assert [e["data"]["channel_id"] for e in tiny.events(a, "comms.joined")] == [12]

    def call(agent: str, n: int, tool: str, args: str) -> dict:
        """`agent` calls the comms tool in its n-th turn; the result, parsed."""
        tiny.flock("prompt", agent, f"call: {tool} {args}")
        tiny.turns(agent, n)
        return json.loads(tiny.results(agent)[-1])

    out = call(a, 2, "post", '{"text": "taking the fallback ladder", "thread": "new"}')
    root = out["posted"]
    assert out["thread"] == root and board.msgs[-1]["sender"]["external_id"] == a
    out = call(a, 3, "post", '{"text": "and another thing", "thread": "new"}')
    assert out["error"].startswith("refused: wait") and len(board.msgs) == 2  # never sent
    out = call(b, 2, "read", "{}")
    assert [(m["id"], m["from"]) for m in out["messages"]][1:] == [(root, a)]
    assert out["messages"][0]["id"] == kick["id"]
    out = call(b, 3, "post", f'{{"text": "@{a} @quiet-0000 I can help with the ladder", "thread": {root}}}')
    assert out["thread"] == root
    mention = {"msg_id": out["posted"], "thread": root, "from": b}
    # both are mentioned; only the writer's loop wakes on it, a wake is not a round
    tiny.turns(a, 4)
    assert tiny.said(a)[-1][-1] == "bash: echo round 1 wake: comms.mention"
    assert [e["data"] for e in tiny.events("quiet-0000", "comms.mention")] == [mention]
    assert [e["data"] for e in tiny.events(a, "comms.mention")] == [mention]
    out = call(a, 5, "search", '{"query": "help ladder zebra"}')
    assert out["dropped"] == ["zebra"] and [(h["author"], h["thread"]) for h in out["results"]] == [
        (b, root)]

    reqs = [json.loads(x)["body"] for x in tiny.log.read_text().splitlines()]
    system = reqs[0]["messages"][0]["content"]
    assert "PROTO-V1: one agent leads" in system and "#flock-test" in system
    assert {t["function"]["name"] for t in reqs[0]["tools"]} == {"bash", "post", "read", "search"}
    pollers = {who for _, p, q, who in board.requests if p == "/v1/activity" and "wait" in q}
    assert len(pollers) == 2  # one long-poll per worker
    tokens = list(board.tokens)
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])
    files = [p for p in (run / "agents").rglob("*") if p.is_file()]
    assert files and not any(t in p.read_text(errors="replace") for p in files for t in tokens)
    tiny.flock("prompt", b, "bash: env")  # nor in the agents' environment
    tiny.turns(b, 4)
    assert "FLOCK_AGENT=" in tiny.results(b)[-1]
    assert not any(t in tiny.results(b)[-1] for t in tokens)

    # live: new protocol text and no post gap reach the running workers
    (tmp_path / "protocols" / "lead.md").write_text("PROTO-V2: the lead posts a plan first.")
    spec("0s")
    updated = datetime.now(UTC)
    tiny.flock("update", "spec.yaml")

    def beat() -> datetime:  # the writers' worker's last heartbeat
        st = json.loads(tiny.flock("status", "tiny", "--json"))
        return datetime.fromisoformat(
            next(w["last_heartbeat"] for w in st["workers"] if w["type"] == "writer"))

    first = tiny.wait(lambda: (t := beat()) > updated and t)
    tiny.wait(lambda: beat() > first)  # so the answer to `first` (the new rules) was applied
    out = call(a, 6, "post", f'{{"text": "plan: probes first", "thread": {root}}}')
    assert out["thread"] == root, out
    system = json.loads(tiny.log.read_text().splitlines()[-1])["body"]["messages"][0]["content"]
    assert "PROTO-V2" in system and "PROTO-V1" not in system
    assert [m["sender"]["external_id"] for m in board.msgs].count("tiny-host") == 1  # kickoff once
    assert [e["name"] for e in tiny.events("tiny", "worker.")] == ["worker.launched"] * 2

    # the mention woke the writer once (its 2 s debounce is long past)
    assert [(e["data"]["round"], e["data"]["why"]) for e in tiny.events(a, "loop.prompt")] == [
        (1, "start"), (1, "wake: comms.mention")]
    assert [e["data"]["why"] for e in tiny.events("quiet-0000", "loop.prompt")] == ["start"]
    assert [e["data"]["reason"] for e in tiny.events(a, "loop.wake")] == ["comms.mention"]


# appended under types.writer
HARNESSES = """\
  coder:
    harness: codex
    model: fake:fake-model
    replicas: 1
    env: {HEARTBEAT_INTERVAL: "0.5"}
"""


def test_one_turn_per_harness(tiny, tmp_path, monkeypatch):
    """One turn on each harness the tests can run: `llm` on the fake model server, `codex` on
    the fake app-server of tests/test_worker_adapters.py (FLOCK_CODEX_CLI, which a local
    worker keeps). `pi` and `claude` need their real CLIs (test_worker_pi runs pi when it is
    installed)."""
    codex = tmp_path / "fake-codex"
    codex.write_text(f"#!{sys.executable}\n{FAKE_APP_SERVER}")
    codex.chmod(0o755)
    monkeypatch.setenv("FLOCK_CODEX_CLI", str(codex))
    tiny.flock("launch", tiny.spec(extra=HARNESSES), "--skip-preflight")
    for n in ("writer-0000", "coder-0000"):
        tiny.flock("prompt", n, "bash: echo hi")
    for n, harness, reply in (("writer-0000", "llm", "done: hi\n[exit code 0]"),
                              ("coder-0000", "codex", "done")):
        assert tiny.turns(n, 1)["cost_usd"] > 0
        evs = tiny.events(n)
        started = next(e["data"] for e in evs if e["name"] == "agent.session_started")
        assert (started["harness"], started["model_id"]) == (harness, "fake-model"), started
        ended = next(e["data"] for e in evs if e["name"] == "agent.turn_ended")
        assert ended["status"] == "completed" and ended["tokens"]["fake-model"]["output"] > 0
        said = [e["data"]["message"] for e in evs if e["name"] == "agent.message"]
        assert said[-1]["role"] == "assistant" and said[-1]["content"][-1]["text"] == reply


@pytest.fixture
def hub(tmp_path, monkeypatch) -> fake_hub.Hub:
    """tests/fake_hub.py for the processes the test starts (HF_ENDPOINT)."""
    h, srv, url = fake_hub.serve(tmp_path / "hub")
    monkeypatch.setenv("HF_ENDPOINT", url)
    yield h
    srv.shutdown()


# The cluster's worker script (scripts/slurm/worker.sbatch) cut down: the env file the controller
# wrote, then the secrets file it names. No BUCKET: the agents' state stays on this disk.
WORKER_SBATCH = """\
#!/bin/bash
set -a; . "$1"; . "$FLOCK_SECRETS_FILE"; set +a
unset BUCKET
exec env PYTHONPATH="{pythonpath}" {python} -m flock.worker.main
"""


def test_slurm(tiny, slurm, hub, tmp_path, monkeypatch):
    """compute: slurm on the stub cluster (tests/fixtures/slurm_stubs; a job runs at once):
    launch, a turn; the node's Spot notice (SIGUSR1 to every process of the job, twice) cuts the
    next turn; the slot is relaunched under the same job name, no crash, and the agent resumes
    its session with the resume prompt; stop --release ends the job."""
    monkeypatch.setenv("SLURM_STUB_RUN", "1")
    script, secrets, host = (tmp_path / n for n in ("worker.sbatch", "secrets.env", "ctl.host"))
    script.write_text(WORKER_SBATCH.format(python=sys.executable,
                                           pythonpath=os.environ.get("PYTHONPATH", "")))
    script.chmod(0o755)
    secrets.write_text("FLOCK_TOKEN=tiny-token\nFAKE_KEY=fake-key\n")
    secrets.chmod(0o600)
    host.write_text("127.0.0.1\n")
    port = tiny.controller(tmp_path / "ctl", compute="slurm")
    out = tiny.flock("launch", tiny.spec(), "--skip-preflight", code=1)  # compute: local
    assert "compute 'slurm'" in out, out
    compute = (f"{{kind: slurm, partition: cpu, account: acct, worker_script: {script}, "
               f"log_dir: {tmp_path}/logs, launch_dir: {tmp_path}/launch, host_file: {host}, "
               f"container_root: {tmp_path}/node, secrets_file: {secrets}}}")
    spec = tiny.spec(compute=compute, env='FLUSH_INTERVAL: "0.2"', extra=(
        '    resume_prompt: "bash: echo resumed"\nbucket: hf://buckets/ns/tiny\n'))
    tiny.flock("launch", spec, "--skip-preflight")
    assert hub.asked == ["/api/buckets/ns/tiny/tree/tiny%2F?recursive=true"]  # empty
    tiny.flock("prompt", "writer-0000", "bash: echo one")
    session = tiny.turns("writer-0000", 1)["session_id"]

    # the job: sbatch's line, an env file without secrets, sbatch's scrubbed environment
    w = "tiny-writer-w0000"
    (first,) = slurm.ids()
    argv = slurm.argv(first)
    env_file = Path(argv[-1])
    assert argv == ["--parsable", "-p", "cpu", "-A", "acct", "-c", "4", "--mem=7G",
                    "--time=29:00:00", "--no-requeue", "-J", w,
                    f"--comment=swarm=tiny,type=writer,worker={w}",
                    "-o", f"{tmp_path}/logs/{w}-%j.out", str(script), str(env_file)]
    assert env_file.parent == tmp_path / "launch" and env_file.stat().st_mode & 0o777 == 0o600
    env = read_env_file(env_file)
    node = f"{tmp_path}/node/flock/{w}"
    assert {k: env[k] for k in ("FLOCK_URL", "FLOCK_CTL_HOST_FILE", "WORKER_ID", "SWARM",
                                "TYPE", "BUCKET", "RUN", "WORK_ROOT", "HOME")} == {
        "FLOCK_URL": f"http://${{CTL_HOST}}:{port}", "FLOCK_CTL_HOST_FILE": str(host),
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET": "hf://buckets/ns/tiny",
        "RUN": tmp_path.name, "WORK_ROOT": f"{node}/work", "HOME": f"{node}/home"}
    assert env_file.read_text().splitlines()[1].endswith(": FLOCK_TOKEN HF_TOKEN FAKE_KEY")
    environ = (slurm.dir / "jobs" / f"{first}.environ").read_text().splitlines()
    assert any(x.startswith("PATH=") for x in environ)
    for secret in ("tiny-token", "fake-key"):
        assert not any(secret in x for x in [env_file.read_text(), *argv, *environ]), secret

    # the Spot notice mid-turn: the worker drains once and exits 75; the same job name again
    tiny.flock("prompt", "writer-0000", "bash: sleep 60")
    tiny.wait(lambda: tiny.agent("writer-0000")["state"] == "working")
    for _ in range(2):
        os.killpg(int((slurm.dir / "jobs" / f"{first}.pid").read_text()), signal.SIGUSR1)
    second = tiny.wait(lambda: [i for i in slurm.ids() if i != first])[0]
    argv2 = slurm.argv(second)
    assert argv2[:-1] == argv[:-1] and argv2[-1] != argv[-1]  # same name, new incarnation
    log = (tmp_path / "logs" / f"{w}-{first}.out").read_text()
    assert log.count("preempted (") == 1 and "preempted (SIGUSR1)" in log, log
    tiny.wait(lambda: tiny.results("writer-0000")[-1].startswith("resumed"))
    a = tiny.wait(lambda: (a := tiny.agent("writer-0000"))["resumed"] and a)
    assert (a["session_id"], a["worker"]) == (session, w), a
    assert tiny.said("writer-0000")[-1] == ["bash: echo one", "bash: sleep 60",
                                            "bash: echo resumed"]
    # a planned restart, from the worker's last heartbeat or the job's exit 75 (whichever is
    # first): the old job ends FAILED, or CANCELLED by the controller
    evs = [(e["name"], e["data"]["detail"]) for e in tiny.events("tiny", "worker.")]
    assert [n for n, _ in evs] == ["worker.launched", "worker.restarted", "worker.launched"]
    assert (evs[0][1], evs[2][1]) == (first, second)
    assert evs[1][1].startswith("ERROR: preempted: "), evs

    # release: the worker drains and its job ends; a controller started by hand stays
    out = tiny.flock("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed" in out, out
    assert "controller kept: no job of this run serves it" in out
    ended = [(slurm.dir / "jobs" / f"{i}.state").read_text().strip() for i in (first, second)]
    assert ended in (["FAILED", "COMPLETED"], [f"CANCELLED by {os.getuid()}", "COMPLETED"])
    [wk] = json.loads(tiny.flock("status", "tiny", "--json"))["workers"]
    assert (wk["state"], wk["crashes"]) == ("dead", 0), wk


def test_hf_jobs(tiny, hub, tmp_path, monkeypatch):
    """compute: hf-jobs on the fake Hub (tests/fake_hub.py; a job runs on this machine): the
    controller starts the worker job (bootstrap image, the job env, the secrets apart), a turn,
    then stop --release drains the worker and its job completes."""
    monkeypatch.setenv("FLOCK_WHEEL_URL", "hf://buckets/ns/tiny/flock.whl")  # bootstrap mode
    tiny.controller(tmp_path / "ctl", compute="hf-jobs")
    tiny.flock("launch", tiny.spec(compute="hf-jobs", env='FLUSH_INTERVAL: "0.2"',
                                   extra="namespace: ns\nbucket: hf://buckets/ns/tiny\n"),
               "--skip-preflight")
    assert hub.asked == ["/api/buckets/ns/tiny/tree/tiny%2F?recursive=true"]  # empty
    tiny.flock("prompt", "writer-0000", "bash: echo one")
    assert tiny.turns("writer-0000", 1)["cost_usd"] > 0

    w = "tiny-writer-w0000"
    [(job, sent)] = hub.jobs.items()
    assert (sent["namespace"], sent["dockerImage"], sent["command"]) == (
        "ns", BOOTSTRAP_IMAGE, bootstrap_command())
    env = sent["environment"]
    assert {k: env[k] for k in ("WORKER_ID", "SWARM", "TYPE", "BUCKET", "FLOCK_WHEEL_URL",
                                "FLOCK_HARNESS")} == {
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET": "hf://buckets/ns/tiny",
        "FLOCK_WHEEL_URL": "hf://buckets/ns/tiny/flock.whl", "FLOCK_HARNESS": "llm"}
    assert sent["secrets"] == {"FLOCK_TOKEN": "tiny-token", "FAKE_KEY": "fake-key"}
    assert not {"tiny-token", "fake-key"} & set(env.values())
    assert {k: sent["labels"][k] for k in ("swarm", "type", "worker")} == {
        "swarm": "tiny", "type": "writer", "worker": w}
    assert sent["timeoutSeconds"] > 24 * 3600

    out = tiny.flock("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed" in out, out
    assert hub.info(job)["status"]["stage"] == "COMPLETED"
    assert [(e["name"], e["data"]["detail"]) for e in tiny.events("tiny", "worker.")] == [
        ("worker.launched", job), ("worker.released", job), ("worker.ended", "COMPLETED")]
