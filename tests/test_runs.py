"""Tiny real runs (tests/tinyrun.py): a `compute: local` Harbor swarm through its whole
lifecycle, a controller that crashes mid-loop and comes back, loops, the run's result; and the
remote backends on fakes: Slurm (tests/fixtures/slurm_stubs) and HF Jobs (tests/fake_hub.py)."""

from __future__ import annotations

import json
import os
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from legion import results
from legion.controller.jobs import BOOTSTRAP_IMAGE, bootstrap_command
from legion.controller.slurm import read_env_file
from legion.results import Outcome

from . import fake_hub


def test_lifecycle(tiny):
    """launch -> turn -> status -> update -> stop (hold) -> launch -> stop --release -> launch
    -> stop --delete, each step checked through the CLI as an operator sees it."""
    spec = tiny.spec(replicas=1)
    out = tiny.legion("launch", spec, "--skip-preflight")
    # agents=0, or 1 when the controller's first reconcile placed it before the reply
    assert "local run " in out and "tiny  agents=" in out and "gen 1" in out, out
    assert "writer  ok      1" in out, out
    assert "warning: inference:fake-model: unverified (preflight skipped)" in out
    run = Path(json.loads(Path(".legion/current").read_text())["run_dir"])
    w0, = tiny.names("writer", 1)  # legion's random words

    # a turn: its Trial's Pi answered (tests/harbor/test_adapter.py), the events say so, the
    # cost is priced
    ack = json.loads(tiny.legion("prompt", w0, "one"))
    assert ack["agent"] == w0 and ack["prompt_id"], ack
    tiny.legion("prompt", "nobody-0001", "x", code=1)
    a = tiny.turns(w0, 1)
    session = a["session_id"]
    assert session and a["resumed"] is False and a["cost_usd"] > 0, a
    assert a["worker"] == "tiny-writer-w0000"
    evs = tiny.legion("status", w0, "--events")
    for line in ("agent.session_started", "assistant reply-1",
                 "agent.turn_ended completed fake-model"):
        assert line in evs, evs
    turn = tiny.events(w0, "agent.turn")
    assert [(e["name"], e["data"]["turn_id"]) for e in turn] == [
        ("agent.turn_started", "t-1"), ("agent.turn_ended", "t-1")], turn
    assert turn[1]["data"]["tokens"]["fake-model"]["output"] > 0, turn
    started = tiny.events(w0, "agent.session_started")[0]["data"]
    assert (started["harness"], started["model_id"]) == ("harbor", "fake-model"), started
    assert tiny.said(w0)[0].endswith("\n\none")  # after the task's
    st = tiny.wait(lambda: (s := json.loads(tiny.legion("status", "tiny", "--json")))
                   ["verdict"]["level"] == "ok" and s)
    t = st["status"]["types"]["writer"]
    assert t["agents_by_state"] == {"idle": 1} and t["workers_by_state"] == {"running": 1}, t
    assert st["status"]["cost_usd"] == a["cost_usd"]
    assert "OK" in tiny.legion("status", "tiny") and "tiny  agents=1" in tiny.legion("status")

    # update: one more replica joins the running worker (agents_per_worker 2)
    tiny.spec(replicas=2)
    assert tiny.legion("update", spec, "--replicas", "spec").splitlines() == [
        "types.writer.replicas: 1 -> 2"]
    (w1,) = set(tiny.names("writer", 2)) - {w0}  # the first keeps its name
    b = tiny.wait(lambda: (b := tiny.agent(w1))["state"] == "idle" and b)
    assert b["worker"] == "tiny-writer-w0000"
    out = tiny.legion("status", "tiny", "--type", "writer")
    assert "gen 2" in tiny.legion("status", "tiny") and w0 in out and w1 in out, out

    # stop holds: a prompt waits, launch SWARM resumes and it is delivered
    tiny.legion("stop", "tiny", "--type", "nope", code=1)
    out = tiny.legion("stop", "tiny", "--type", "writer", "--reason", "lunch")
    assert out.strip() == "tiny: held: type writer", out
    tiny.legion("prompt", w0, "two")
    out = tiny.legion("status", "tiny")
    assert "HELD type writer: lunch (by operator" in out and "pending=1" in out, out
    assert tiny.agent(w0)["turns"] == 1
    assert tiny.legion("launch", "tiny").strip() == "tiny: held: nothing"
    assert tiny.turns(w0, 2)["pending"] == 0
    assert "HELD" not in tiny.legion("status", "tiny")

    # stop --release frees the workers and keeps everything; launch FOLDER resumes the
    # sessions, refused while the folder differs from the running spec (replicas aside)
    out = tiny.legion("stop", "tiny", "--release", "--reason", "night")
    assert "releasing: 1 workers still running" in out, out  # it waited for the worker
    assert "released: every agent synced, compute freed; `legion launch tiny` resumes" in out
    assert "controller kept: a local run keeps its controller" in out
    out = tiny.legion("status", "tiny")
    assert "RELEASED since" in out and "workers: dead:1" in out, out
    tiny.spec(replicas=1, extra="limits: {agent_max_turns: 5}\n")
    assert "run `legion update` first" in tiny.legion("launch", spec, "--skip-preflight", code=1)
    tiny.spec(replicas=1)
    assert "tiny: held: nothing" in tiny.legion("launch", spec, "--skip-preflight")
    tiny.legion("prompt", w0, "three")
    a = tiny.turns(w0, 3)
    assert (a["session_id"], a["resumed"]) == (session, True), a
    assert tiny.said(w0)[1:] == ["two", "three"]  # the same
    ended = [e["data"]["turn_id"] for e in tiny.events(w0, "agent.turn_ended")]
    assert ended == ["t-1", "t-2", "t-3"], ended  # the new worker goes on counting
    # swarm.redrain: a resent drain, harmless, when the drain poll beats the next heartbeat
    assert [e["name"] for e in tiny.events("tiny") if e["name"] != "swarm.redrain"] == [
        "swarm.spec_applied", "worker.launched", "swarm.spec_applied",  # launch, update
        "swarm.stopped", "swarm.resumed",  # stop, launch
        "swarm.released", "swarm.stopped", "worker.released", "worker.ended",  # --release
        "swarm.unreleased", "swarm.resumed", "worker.launched"]  # launch
    assert tiny.events("tiny", "swarm.released")[0]["data"]["reason"] == "night"

    # stop --delete ends the local run: the bundle
    out = tiny.legion("stop", "tiny", "--delete", "--yes")
    assert f"local run ended; bundle: {run}/run.json" in out, out
    assert not Path(".legion/current").exists()
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "interrupted"
    assert doc["spec"]["compute"]["kind"] == "local"
    assert {x["name"]: x["turns_completed"] for x in doc["agents"]} == {  # replicas kept: 2
        w0: 3, w1: 0}
    assert doc["totals"]["cost_usd"] == round(a["cost_usd"], 6)


def worker_pid(run: Path, worker: str) -> int | None:
    """The pid of the live worker process `worker` of the local run in `run`, if any: tiny runs
    in parallel tests reuse worker ids, so its WORK_ROOT (the run's agents/) must match too."""
    want = {f"WORKER_ID={worker}".encode(), f"WORK_ROOT={run / 'agents'}".encode()}
    for d in Path("/proc").glob("[0-9]*"):
        try:
            if (b"legion.worker.main" in (d / "cmdline").read_bytes()
                    and want <= set((d / "environ").read_bytes().split(b"\0"))):
                return int(d.name)
        except OSError:
            continue
    return None


def test_results(tiny):
    """`legion results` on a local run (docs/results.md): a verifier's 0 is a completed result
    (s3's wrong answer), and an agent whose worker was killed and relaunched is worker_replaced,
    though its new Trial completed: the relaunch reuses its Trial names (huggingface/legion#121)."""
    spec = tiny.spec(replicas=3, env='TINY_FAKE_REWARD: "0"')  # 2 workers: 2 agents and 1
    tiny.legion("launch", spec, "--skip-preflight")
    run = Path(json.loads(Path(".legion/current").read_text())["run_dir"])
    names = tiny.names("writer", 3)
    tiny.legion("prompt", "--swarm", "tiny", "--type", "writer", "go")
    workers = {n: tiny.turns(n, 1)["worker"] for n in names}
    [lone] = [n for n in names if list(workers.values()).count(workers[n]) == 1]
    victim = workers[lone]
    pid = worker_pid(run, victim)
    assert pid is not None
    os.kill(pid, signal.SIGKILL)  # no drain, no exit record
    tiny.wait(lambda: worker_pid(run, victim) not in (None, pid))
    tiny.legion("prompt", lone, "again")
    tiny.turns(lone, 2)
    tiny.legion("stop", "tiny", "--delete", "--yes")  # drains: the Trials are verified

    out = tiny.legion("results", str(run))
    r = results.RunResult.model_validate_json(out)
    assert r == results.load(run) == results.load(Path.cwd())  # a folder holding one bundle
    assert (r.run_id, r.attempt_id, r.swarm, r.compute) == ("tiny", run.name, "tiny", "local")
    held = r.agent(lone).workers  # the relaunched slot, or another worker it was placed on
    assert victim in held, r
    assert (r.outcome, r.reasons) == (Outcome.WORKER_REPLACED,
                                      [f"{lone}: worker_replaced, worker {', '.join(held)}"]), r
    assert (r.planned_agents, r.stop_reason, r.limits["agent_max_turns"]) == (3, "interrupted", 30)
    assert r.task == {"repo": "https://huggingface.co/datasets/ns/tasks", "selector": "t",
                      "revision": "main"}
    for a in r.agents:
        assert a.reward == 0.0 and (a.name == lone or a.workers == [workers[a.name]]), a
        assert a.outcome is (Outcome.WORKER_REPLACED if a.name == lone else Outcome.COMPLETED), a
        t = a.trials[-1]  # the Trial itself says completed, with Harbor's files
        assert (t.outcome, t.reward, t.rewards) == (Outcome.COMPLETED, 0.0, {"reward": 0.0}), t
        assert t.result_path == Path("agents/_harbor", a.name, "trials", t.name, "result.json")
        assert (run / t.result_path).is_file() and (run / t.verifier_dir / "reward.txt").is_file()
    killed = next(w for w in r.workers if w.id == victim).incarnations[0]
    assert killed.exit is None and killed.agents == [lone], killed
    assert all(i.exit == "drained" for w in r.workers for i in w.incarnations if i is not killed)


# appended under role writer
RESTART_LOOP = """\
      loop:
        prompt: "round {round}"
        every: 1s
        while: {max_rounds: 30}
    helper:
      count: 0
      workers: {max_extra: 1}
spawn:
  writer: {types: {helper: 1}, depth: 1}
limits: {run_timeout_seconds: 900}
"""


def test_controller_restart(tiny, tmp_path):
    """A controller killed mid-loop (SIGKILL: no final flush) and started again on the same
    state keeps its swarm (turns, cost, spawn links, sessions), and its loop goes on: the next
    round reaches each agent."""
    data = tmp_path / "ctl"
    port = tiny.controller(data)
    tiny.legion("launch", tiny.spec(replicas=2, extra=RESTART_LOOP), "--skip-preflight")
    w0, w1 = tiny.names("writer", 2)
    tiny.turns(w0, 1)  # the loop's first round
    child = tiny.legion("spawn", "helper", "--parent", w0, "--prompt", "hi").strip()
    assert tiny.agent(w0)["children"] == [child]
    tiny.turns(child, 1)
    writers = (w0, w1)
    before = {n: tiny.agent(n) for n in (*writers, child)}
    loop = json.loads(tiny.legion("status", "tiny", "--type", "writer", "--json"))["loop"]
    rounds = {n: loop["per_agent"][n]["rounds"] for n in writers}
    assert loop["state"] == "running" and max(rounds.values()) < 30, loop

    tiny.crash()
    restarted = datetime.now(UTC)
    tiny.controller(data, port)
    st = json.loads(tiny.legion("status", "tiny", "--json"))["status"]
    assert st["cost_usd"] >= sum(a["cost_usd"] for a in before.values()) > 0, st
    for n, a in before.items():
        now = tiny.agent(n)
        assert now["session_id"] == a["session_id"] and now["turns"] >= a["turns"] >= 1, now
        assert now["cost_usd"] >= a["cost_usd"] > 0, now
    assert tiny.agent(child)["parent"] == w0
    assert tiny.agent(w0)["children"] == [child]
    assert child in tiny.legion("status", "tiny", "--parent", w0)

    # the loop engine goes on: a round past the one before the crash reaches each writer and
    # its model, in the same session (the workers died with the controller and were relaunched);
    # no round is sent twice (rounds due while the swarm was down are skipped)
    for n in writers:
        first = tiny.wait(lambda n=n: [e["data"]["round"] for e in tiny.events(n, "loop.prompt")
                                       if datetime.fromisoformat(e["ts"]) > restarted
                                       and e["data"]["round"] > rounds[n]])[0]
        said = tiny.wait(lambda n=n, k=first: f"round {k}" in tiny.said(n) and tiny.said(n))
        assert said[0].endswith("\n\nround 1"), said
        sent = [e["data"]["round"] for e in tiny.events(n, "loop.prompt")]
        assert sent == sorted(set(sent)), sent

    # stop --delete on a controller: the swarm is removed
    assert json.loads(tiny.legion("stop", "tiny", "--delete", "--yes")) == {
        "swarm": "tiny", "removing": True}



# appended under role writer
LOOPS = """\
      loop:
        prompt: "round {round}"
        while: {max_rounds: 3}
        final: {prompt: "final after {round} rounds; last: {last_summary}"}
    ticker:
      count: 1
      loop:
        prompt: "tick {round}"
        every: 500ms
        on: [message_board.mention]
        while: {until: 6s}
        final: {prompt: "bye"}
    held:
      count: 0  # no loop: the run stays up until the test stops it (else it ends with the loops)
limits: {budget_usd: 1000, run_timeout_seconds: 900}
"""


def test_loops(tiny):
    """Two loops, each with its final round, then idle: `every: turn_end` with `max_rounds`
    (re-prompted by the worker) and `every: 500ms` with `while.until` (by the controller).
    Then the run's bundle sums them up."""
    out = tiny.legion("launch", tiny.spec(replicas=2, extra=LOOPS), "--skip-preflight")
    # unquoted `on:` is the key "on" (YAML 1.2), and the run has no message_board resource
    assert "loop.on message_board.mention: no feeder: the run has no message_board resource" in out, out
    run = Path(json.loads(Path(".legion/current").read_text())["run_dir"])
    w0, w1 = tiny.names("writer", 2)
    t0, = tiny.names("ticker", 1)

    def loop(t: str) -> dict:
        return json.loads(tiny.legion("status", "tiny", "--type", t, "--json"))["loop"]

    w = tiny.wait(lambda: (x := loop("writer"))["state"] == "done" and x)
    t = tiny.wait(lambda: (x := loop("ticker"))["state"] == "done" and x)
    time.sleep(1)  # then idle: nothing more is sent
    assert (w["mode"], w["reason"], w["rounds_total"], w["final_sent"], w["final_done"]) == (
        "worker", "max_rounds", 6, 2, 2), w
    for n in (w0, w1):
        said = tiny.said(n)  # rendered on the worker, {last_summary}: the last reply
        assert said[0].endswith("\n\nround 1") and said[1:3] == ["round 2", "round 3"], said
        assert said[3].startswith("final after 3 rounds; last: reply-") and len(said) == 4, said
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

    # the controller's loop: a round every 500 ms until 6 s after the loop started, then the
    # final round
    assert (t["mode"], t["reason"], t["final_done"]) == ("controller", "until", 1), t
    k = t["rounds_total"]
    said = tiny.said(t0)
    assert k >= 2 and said[0].endswith("\n\ntick 1") and said[1:] == [
        *(f"tick {i}" for i in range(2, k + 1)), "bye"], said
    assert [e["data"]["why"] for e in tiny.events(t0, "loop.prompt")] == [
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
    tiny.legion("stop", "tiny", "--delete", "--yes")
    merged = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    agents = (w0, w1, t0)
    per_agent = sum(len((run / "events" / f"{a}.jsonl").read_text().splitlines()) for a in agents)
    assert len(merged) == len([e for e in merged if e["agent"] == ""]) + per_agent
    stamps = [datetime.fromisoformat(e["ts"]) for e in merged]
    assert stamps == sorted(stamps)
    assert sorted(e["agent"] for e in merged if e["type"] == "final_round") == sorted(agents)
    doc = json.loads((run / "run.json").read_text())
    assert doc["spec"]["types"]["ticker"]["loop"]["on"] == ["message_board.mention"]
    lp = doc["loops"]["writer"]
    assert (lp["state"], lp["reason"], lp["rounds_total"], lp["final_sent"], lp["final_done"],
            lp["wakes_total"]) == ("done", "max_rounds", 6, 2, 2, 0), lp
    assert (doc["loops"]["ticker"]["state"], doc["loops"]["ticker"]["reason"]) == ("done", "until")
    assert doc["budget"] == {"budget_usd": 1000.0, "final_reserve_usd": 0.0,
                             "cost_usd": doc["totals"]["cost_usd"], "stop": None}


@pytest.fixture
def hub(tmp_path, monkeypatch) -> fake_hub.Hub:
    """tests/fake_hub.py for the processes the test starts (HF_ENDPOINT)."""
    h, srv, url = fake_hub.serve(tmp_path / "hub")
    monkeypatch.setenv("HF_ENDPOINT", url)
    yield h
    srv.shutdown()


# The cluster's worker script (scripts/slurm/worker.sbatch) cut down: the env file the controller
# wrote, then the secrets file it names. No BUCKET_ID: the agents' state stays on this disk, in the
# test's folder (a harbor worker's WORK_ROOT is the node's /tmp/legion/<worker>/work).
WORKER_SBATCH = """\
#!/bin/bash
set -a; . "$1"; . "$LEGION_SECRETS_FILE"; set +a
unset BUCKET_ID
export WORK_ROOT="{node}/$WORKER_ID/work" LEGION_RESULTS_DIR="{results}"
exec env PYTHONPATH="{pythonpath}" {python} -m legion.worker.main
"""


# a resource whose export, as its worker exits, is one file (a board's board.jsonl)
NOTE = """\
import asyncio


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.metadata = {}

    async def start(self):
        pass

    def bind(self, name):
        return {"env": {}, "mounts": []}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        (directory / "board.jsonl").write_text("{}\\n")
        return {"board": "board.jsonl"}

    async def stop(self):
        pass
"""


def test_slurm(tiny, slurm, hub, tmp_path, monkeypatch):
    """compute: slurm on the stub cluster (tests/fixtures/slurm_stubs; a job runs at once):
    launch, the loop's one round; the node's Spot notice (SIGUSR1 to every process of the job,
    twice) cuts the next turn; the slot is relaunched under the same job name, no crash, and the
    agent resumes its session with the loop's resume prompt; stop --release ends the job."""
    monkeypatch.setenv("SLURM_STUB_RUN", "1")
    script, secrets, host = (tmp_path / n for n in ("worker.sbatch", "secrets.env", "ctl.host"))
    script.write_text(WORKER_SBATCH.format(python=sys.executable, node=tmp_path / "node",
                                           results=tmp_path / "results",
                                           pythonpath=os.environ.get("PYTHONPATH", "")))
    script.chmod(0o755)
    secrets.write_text("LEGION_TOKEN=tiny-token\nFAKE_KEY=fake-key\n")
    secrets.chmod(0o600)
    host.write_text("127.0.0.1\n")
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "note.py").write_text(NOTE)
    port = tiny.controller(tmp_path / "ctl", compute="slurm")
    out = tiny.legion("launch", tiny.spec(), "--skip-preflight", code=1)  # compute: local
    assert "compute 'slurm'" in out, out
    compute = (f"{{kind: slurm, partition: cpu, account: acct, worker_script: {script}, "
               f"log_dir: {tmp_path}/logs, launch_dir: {tmp_path}/launch, host_file: {host}, "
               f"secrets_file: {secrets}}}")
    spec = tiny.spec(compute=compute, backend="podman", env='FLUSH_INTERVAL: "0.2"', extra=(
        '      loop: {prompt: one, while: {max_rounds: 1}, resume_prompt: resumed}\n'
        'limits: {run_timeout_seconds: 900}\nbucket_id: ns/tiny\n'
        'resources_dir: resources\nresources: {note: {}}\n'))
    tiny.legion("launch", spec, "--skip-preflight")
    assert hub.asked == ["/api/buckets/ns/tiny/tree/tiny%2F?recursive=true"]  # empty
    w0, = tiny.names("writer", 1)
    session = tiny.turns(w0, 1)["session_id"]

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
    assert {k: env[k] for k in ("LEGION_URL", "LEGION_CTL_HOST_FILE", "WORKER_ID", "SWARM",
                                "TYPE", "BUCKET_ID", "RUN", "WORK_ROOT", "LEGION_HARNESS")} == {
        "LEGION_URL": f"http://${{CTL_HOST}}:{port}", "LEGION_CTL_HOST_FILE": str(host),
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET_ID": "ns/tiny",
        "RUN": tmp_path.name, "WORK_ROOT": f"/tmp/legion/{w}/work", "LEGION_HARNESS": "harbor"}
    assert "HOME" not in env  # harbor runs podman on the node, never in a container
    assert env_file.read_text().splitlines()[1].endswith(": LEGION_TOKEN HF_TOKEN FAKE_KEY")
    environ = (slurm.dir / "jobs" / f"{first}.environ").read_text().splitlines()
    assert any(x.startswith("PATH=") for x in environ)
    for secret in ("tiny-token", "fake-key"):
        assert not any(secret in x for x in [env_file.read_text(), *argv, *environ]), secret

    # the Spot notice mid-turn: the worker drains once and exits 75; the same job name again
    tiny.legion("prompt", w0, "work [hold]")  # the fake Pi waits until killed
    tiny.wait(lambda: tiny.agent(w0)["state"] == "working")
    for _ in range(2):
        os.killpg(int((slurm.dir / "jobs" / f"{first}.pid").read_text()), signal.SIGUSR1)
    second = tiny.wait(lambda: [i for i in slurm.ids() if i != first])[0]
    argv2 = slurm.argv(second)
    assert argv2[:-1] == argv[:-1] and argv2[-1] != argv[-1]  # same name, new incarnation
    log = (tmp_path / "logs" / f"{w}-{first}.out").read_text()
    assert log.count("preempted (") == 1 and "preempted (SIGUSR1)" in log, log
    a = tiny.wait(lambda: (a := tiny.turns(w0, 2))["resumed"] and a)
    assert (a["session_id"], a["worker"]) == (session, w), a
    said = tiny.said(w0)  # the cut turn left nothing in the session
    assert said[0].endswith("\n\none") and said[1:] == ["resumed"], said
    # a planned restart, from the worker's last heartbeat or the job's exit 75 (whichever is
    # first): the old job ends FAILED, or CANCELLED by the controller
    evs = [(e["name"], e["data"]["detail"]) for e in tiny.events("tiny", "worker.")]
    assert [n for n, _ in evs] == ["worker.launched", "worker.restarted", "worker.launched"]
    assert (evs[0][1], evs[2][1]) == (first, second)
    assert evs[1][1].startswith("ERROR: preempted: "), evs

    # release: the worker drains and its job ends; a controller started by hand stays
    out = tiny.legion("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed" in out, out
    assert "controller kept: no job of this run serves it" in out
    # the drained worker's last heartbeat ends its slot at once; the controller's scancel then
    # finds the job ended, or ends it while the worker exits
    ours = f"CANCELLED by {os.getuid()}"
    ended = [(slurm.dir / "jobs" / f"{i}.state").read_text().strip() for i in (first, second)]
    assert ended[0] in ("FAILED", ours) and ended[1] in ("COMPLETED", ours), ended
    [wk] = json.loads(tiny.legion("status", "tiny", "--json"))["workers"]
    assert (wk["state"], wk["crashes"]) == ("dead", 0), wk

    # the result (docs/results.md), from this folder as a run dir (tmp_path: logs/, launch/,
    # results/): the agent's Trial completed after the relaunch, under the name its cut Trial
    # had (huggingface/legion#121), so the agent counts as preempted. Only the drained worker
    # exported its resources (the preempted one exits within its budget)
    (tmp_path / "experiment").mkdir()
    (tmp_path / "experiment" / "spec.yaml").write_text(Path(spec).read_text())
    r = results.load(tmp_path)
    assert (r.compute, r.swarm, r.outcome, r.planned_agents) == (
        "slurm", "tiny", Outcome.PREEMPTED, 1)
    assert r.reasons == [f"{w0}: preempted, worker {w}"]
    [worker] = r.workers
    assert [(i.job_id, i.exit, i.agents) for i in worker.incarnations] == [
        (first, "preempted", [w0]), (second, "drained", [w0])]
    [agent] = r.agents
    assert (agent.name, agent.workers, agent.outcome, agent.reward) == (
        w0, [w], Outcome.PREEMPTED, 1.0)
    [trial] = agent.trials
    assert trial.outcome is Outcome.COMPLETED
    assert trial.result_path == Path("results/tiny", w0, "trials", trial.name, "result.json")
    assert r.exports == [results.ResourceExport(
        worker=w, resource="note", path=Path("results/tiny", w, "resources/note/board.jsonl"))]
    # a sweep attempt's receipt names the logical run; without one, `-rN` is part of the name
    assert (r.run_id, r.attempt_id, r.attempt) == ("tiny", tmp_path.name, 0)
    (tmp_path / "attempt.json").write_text('{"run_id": "cell-1", "attempt": 2}')
    assert (results.load(tmp_path).run_id, results.load(tmp_path).attempt) == ("cell-1", 2)

    # a sweep attempt (legion sweep) that kept no Trial: its receipt's failure is the evidence,
    # in its own words; each planned agent takes it rather than `unknown`
    failed = tmp_path / "sweep-ab12-cell-r0"
    (failed / "experiment").mkdir(parents=True)
    (failed / "experiment" / "spec.yaml").write_text(
        Path(spec).read_text().replace("name: tiny", "name: sweep-ab12-cell-r0")
        + "naming: consecutive-id\n")
    (failed / "attempt.json").write_text(json.dumps(
        {"phase": "infrastructure_failure", "failure": "worker_crash", "note": None}))
    f = results.load(failed)  # one launch: the crashed worker was lost, not replaced
    assert (f.run_id, f.attempt_id, f.attempt) == ("sweep-ab12-cell", failed.name, 0)
    assert (f.outcome, f.reasons) == (Outcome.WORKER_LOST, [
        "attempt.json: infrastructure_failure, worker_crash",
        "agent-001: worker_lost, no Trials; see attempt.json"]), f
    assert [(a.name, a.outcome, a.reward, a.trials) for a in f.agents] == [
        ("agent-001", Outcome.WORKER_LOST, None, [])]
    (failed / "launch").mkdir()  # a second launch of the worker: it was replaced
    for incarnation in ("a" * 32, "b" * 32):
        (failed / "launch" / f"{w}-{incarnation}.env").write_text(f"WORKER_ID={w}\n")
    f = results.load(failed)
    assert (f.outcome, f.reasons) == (Outcome.WORKER_REPLACED, [
        f"worker {w}: worker_replaced (2 launches), agents unknown",
        "attempt.json: infrastructure_failure, worker_crash",
        "agent-001: worker_replaced, no Trials; see attempt.json"]), f


def test_hf_jobs(tiny, hub, tmp_path, monkeypatch):
    """compute: hf-jobs on the fake Hub (tests/fake_hub.py; a job runs on this machine): the
    controller starts the worker job (bootstrap image, the job env, the secrets apart), a turn,
    then stop --release drains the worker and its job completes."""
    monkeypatch.setenv("LEGION_WHEEL_URL", "hf://buckets/ns/tiny/legion.whl")  # bootstrap mode
    tiny.controller(tmp_path / "ctl", compute="hf-jobs")
    tiny.legion("launch", tiny.spec(compute="hf-jobs", backend="hf-sandbox",
                                   env='FLUSH_INTERVAL: "0.2"',
                                   extra="namespace: ns\nbucket_id: ns/tiny\n"),
               "--skip-preflight")
    assert hub.asked == ["/api/buckets/ns/tiny/tree/tiny%2F?recursive=true"]  # empty
    w0, = tiny.names("writer", 1)
    tiny.legion("prompt", w0, "one")
    assert tiny.turns(w0, 1)["cost_usd"] > 0

    w = "tiny-writer-w0000"
    [(job, sent)] = hub.jobs.items()
    assert (sent["namespace"], sent["dockerImage"], sent["command"]) == (
        "ns", BOOTSTRAP_IMAGE, bootstrap_command())
    env = sent["environment"]
    assert {k: env[k] for k in ("WORKER_ID", "SWARM", "TYPE", "BUCKET_ID", "LEGION_WHEEL_URL",
                                "LEGION_HARNESS")} == {
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET_ID": "ns/tiny",
        "LEGION_WHEEL_URL": "hf://buckets/ns/tiny/legion.whl", "LEGION_HARNESS": "harbor"}
    assert sent["secrets"] == {"LEGION_TOKEN": "tiny-token", "FAKE_KEY": "fake-key",
                               "HF_TOKEN": "fake-hub-token"}  # hf-sandbox starts sandboxes with it
    assert not {"tiny-token", "fake-key", "fake-hub-token"} & set(env.values())
    assert {k: sent["labels"][k] for k in ("swarm", "type", "worker")} == {
        "swarm": "tiny", "type": "writer", "worker": w}
    assert sent["timeoutSeconds"] >= 24 * 3600  # lifetime + jitter, in whole seconds

    out = tiny.legion("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed" in out, out
    assert hub.info(job)["status"]["stage"] in ("COMPLETED", "CANCELED")  # see test_slurm
    assert [(e["name"], e["data"]["detail"]) for e in tiny.events("tiny", "worker.")] == [
        ("worker.launched", job), ("worker.released", job),
        ("worker.ended", "COMPLETED: drained: worker reported it")]


def test_stop_on_success_releases_workers_and_verifies(tiny):
    # Real controller + two real worker processes; only Harbor/container/model are fake.
    spec = tiny.spec(replicas=4, extra="""
stop_on_success: true
success_check:
  service: puzzle
  command: after-first-turn
""")
    tiny.legion("launch", spec, "--skip-preflight")
    names = tiny.names("writer", 4)
    tiny.legion("prompt", "--swarm", "tiny", "--type", "writer", "solve [hold]")
    def finished():
        status = json.loads(tiny.legion("status", "tiny", "--json"))["status"]
        return (status.get("success", {}).get("verification") == "passed"
                and all(t["workers_by_state"].get("dead", 0) == 2
                        for t in status["types"].values()) and status)
    status = tiny.wait(finished, timeout=60)
    assert status["released"] and not status["released"]["unsynced"]
    assert status["pending_prompts"] == 0
    assert "STOPPED ON SUCCESS" in tiny.legion("status", "tiny")
    assert "already succeeded" in tiny.legion("prompt", status["success"]["agent"], "again", code=1)
    turns = [e for name in names for e in tiny.events(name, "agent.turn")]
    assert any(e["name"] == "agent.turn_started" for e in turns)
    assert any(e["name"] == "agent.turn_interrupted" for e in turns)
    assert "launch a new run" in tiny.legion("launch", "tiny", code=1)
    released = tiny.legion("stop", "tiny", "--release")
    assert "this swarm succeeded; launch a new run" in released
    assert "resumes" not in released
