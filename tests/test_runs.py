"""Tiny real runs (tests/tinyrun.py): a `compute: local` Harbor swarm through its whole
lifecycle, a controller that crashes mid-loop and comes back, loops; and the remote backends on
fakes: Slurm (tests/fixtures/slurm_stubs) and HF Jobs (tests/fake_hub.py)."""

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

from . import fake_hub


def test_lifecycle(tiny):
    """launch -> turn -> status -> update -> stop (hold) -> launch -> stop --release -> launch
    -> stop --delete, each step checked through the CLI as an operator sees it."""
    spec = tiny.spec(replicas=1)
    out = tiny.flock("launch", spec, "--skip-preflight")
    assert "local run " in out and "tiny  agents=0" in out and "gen 1" in out, out
    assert "writer  ok      1" in out, out
    assert "warning: inference:fake-model: unverified (preflight skipped)" in out
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])
    w0, = tiny.names("writer", 1)  # flock's random words

    # a turn: its Trial's Pi answered (tests/harbor/test_adapter.py), the events say so, the
    # cost is priced
    ack = json.loads(tiny.flock("prompt", w0, "one"))
    assert ack["agent"] == w0 and ack["prompt_id"], ack
    tiny.flock("prompt", "nobody-0001", "x", code=1)
    a = tiny.turns(w0, 1)
    session = a["session_id"]
    assert session and a["resumed"] is False and a["cost_usd"] > 0, a
    assert a["worker"] == "tiny-writer-w0000"
    evs = tiny.flock("status", w0, "--events")
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
    st = tiny.wait(lambda: (s := json.loads(tiny.flock("status", "tiny", "--json")))
                   ["verdict"]["level"] == "ok" and s)
    t = st["status"]["types"]["writer"]
    assert t["agents_by_state"] == {"idle": 1} and t["workers_by_state"] == {"running": 1}, t
    assert st["status"]["cost_usd"] == a["cost_usd"]
    assert "OK" in tiny.flock("status", "tiny") and "tiny  agents=1" in tiny.flock("status")

    # update: one more replica joins the running worker (agents_per_worker 2)
    tiny.spec(replicas=2)
    assert tiny.flock("update", spec, "--replicas", "spec").splitlines() == [
        "types.writer.replicas: 1 -> 2"]
    (w1,) = set(tiny.names("writer", 2)) - {w0}  # the first keeps its name
    b = tiny.wait(lambda: (b := tiny.agent(w1))["state"] == "idle" and b)
    assert b["worker"] == "tiny-writer-w0000"
    out = tiny.flock("status", "tiny", "--type", "writer")
    assert "gen 2" in tiny.flock("status", "tiny") and w0 in out and w1 in out, out

    # stop holds: a prompt waits, launch SWARM resumes and it is delivered
    tiny.flock("stop", "tiny", "--type", "nope", code=1)
    out = tiny.flock("stop", "tiny", "--type", "writer", "--reason", "lunch")
    assert out.strip() == "tiny: held: type writer", out
    tiny.flock("prompt", w0, "two")
    out = tiny.flock("status", "tiny")
    assert "HELD type writer: lunch (by operator" in out and "pending=1" in out, out
    assert tiny.agent(w0)["turns"] == 1
    assert tiny.flock("launch", "tiny").strip() == "tiny: held: nothing"
    assert tiny.turns(w0, 2)["pending"] == 0
    assert "HELD" not in tiny.flock("status", "tiny")

    # stop --release frees the workers and keeps everything; launch FOLDER resumes the
    # sessions, refused while the folder differs from the running spec (replicas aside)
    out = tiny.flock("stop", "tiny", "--release", "--reason", "night")
    assert "releasing: 1 workers still running" in out, out  # it waited for the worker
    assert "released: every agent synced, compute freed; `flock launch tiny` resumes" in out
    assert "controller kept: a local run keeps its controller" in out
    out = tiny.flock("status", "tiny")
    assert "RELEASED since" in out and "workers: dead:1" in out, out
    tiny.spec(replicas=1, extra="limits: {agent_max_turns: 5}\n")
    assert "run `flock update` first" in tiny.flock("launch", spec, "--skip-preflight", code=1)
    tiny.spec(replicas=1)
    assert "tiny: held: nothing" in tiny.flock("launch", spec, "--skip-preflight")
    tiny.flock("prompt", w0, "three")
    a = tiny.turns(w0, 3)
    assert (a["session_id"], a["resumed"]) == (session, True), a
    assert tiny.said(w0)[1:] == ["two", "three"]  # the same
    # swarm.redrain: a resent drain, harmless, when the drain poll beats the next heartbeat
    assert [e["name"] for e in tiny.events("tiny") if e["name"] != "swarm.redrain"] == [
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
        w0: 3, w1: 0}
    assert doc["totals"]["cost_usd"] == round(a["cost_usd"], 6)


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
    tiny.flock("launch", tiny.spec(replicas=2, extra=RESTART_LOOP), "--skip-preflight")
    w0, w1 = tiny.names("writer", 2)
    tiny.turns(w0, 1)  # the loop's first round
    child = tiny.flock("spawn", "helper", "--parent", w0, "--prompt", "hi").strip()
    assert tiny.agent(w0)["children"] == [child]
    tiny.turns(child, 1)
    writers = (w0, w1)
    before = {n: tiny.agent(n) for n in (*writers, child)}
    loop = json.loads(tiny.flock("status", "tiny", "--type", "writer", "--json"))["loop"]
    rounds = {n: loop["per_agent"][n]["rounds"] for n in writers}
    assert loop["state"] == "running" and max(rounds.values()) < 30, loop

    tiny.crash()
    restarted = datetime.now(UTC)
    tiny.controller(data, port)
    st = json.loads(tiny.flock("status", "tiny", "--json"))["status"]
    assert st["cost_usd"] >= sum(a["cost_usd"] for a in before.values()) > 0, st
    for n, a in before.items():
        now = tiny.agent(n)
        assert now["session_id"] == a["session_id"] and now["turns"] >= a["turns"] >= 1, now
        assert now["cost_usd"] >= a["cost_usd"] > 0, now
    assert tiny.agent(child)["parent"] == w0
    assert tiny.agent(w0)["children"] == [child]
    assert child in tiny.flock("status", "tiny", "--parent", w0)

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
    assert json.loads(tiny.flock("stop", "tiny", "--delete", "--yes")) == {
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
    out = tiny.flock("launch", tiny.spec(replicas=2, extra=LOOPS), "--skip-preflight")
    # unquoted `on:` is the key "on" (YAML 1.2), and the run has no message_board resource
    assert "loop.on message_board.mention: no feeder: the run has no message_board resource" in out, out
    run = Path(json.loads(Path(".flock/current").read_text())["run_dir"])
    w0, w1 = tiny.names("writer", 2)
    t0, = tiny.names("ticker", 1)

    def loop(t: str) -> dict:
        return json.loads(tiny.flock("status", "tiny", "--type", t, "--json"))["loop"]

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
    tiny.flock("stop", "tiny", "--delete", "--yes")
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
# wrote, then the secrets file it names. No BUCKET: the agents' state stays on this disk, in the
# test's folder (a harbor worker's WORK_ROOT is the node's /tmp/flock/<worker>/work).
WORKER_SBATCH = """\
#!/bin/bash
set -a; . "$1"; . "$FLOCK_SECRETS_FILE"; set +a
unset BUCKET
export WORK_ROOT="{node}/$WORKER_ID/work"
exec env PYTHONPATH="{pythonpath}" {python} -m flock.worker.main
"""


def test_slurm(tiny, slurm, hub, tmp_path, monkeypatch):
    """compute: slurm on the stub cluster (tests/fixtures/slurm_stubs; a job runs at once):
    launch, the loop's one round; the node's Spot notice (SIGUSR1 to every process of the job,
    twice) cuts the next turn; the slot is relaunched under the same job name, no crash, and the
    agent resumes its session with the loop's resume prompt; stop --release ends the job."""
    monkeypatch.setenv("SLURM_STUB_RUN", "1")
    script, secrets, host = (tmp_path / n for n in ("worker.sbatch", "secrets.env", "ctl.host"))
    script.write_text(WORKER_SBATCH.format(python=sys.executable, node=tmp_path / "node",
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
               f"secrets_file: {secrets}}}")
    spec = tiny.spec(compute=compute, backend="podman", env='FLUSH_INTERVAL: "0.2"', extra=(
        '      loop: {prompt: one, while: {max_rounds: 1}, resume_prompt: resumed}\n'
        'limits: {run_timeout_seconds: 900}\nbucket: hf://buckets/ns/tiny\n'))
    tiny.flock("launch", spec, "--skip-preflight")
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
    assert {k: env[k] for k in ("FLOCK_URL", "FLOCK_CTL_HOST_FILE", "WORKER_ID", "SWARM",
                                "TYPE", "BUCKET", "RUN", "WORK_ROOT", "FLOCK_HARNESS")} == {
        "FLOCK_URL": f"http://${{CTL_HOST}}:{port}", "FLOCK_CTL_HOST_FILE": str(host),
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET": "hf://buckets/ns/tiny",
        "RUN": tmp_path.name, "WORK_ROOT": f"/tmp/flock/{w}/work", "FLOCK_HARNESS": "harbor"}
    assert "HOME" not in env  # harbor runs podman on the node, never in a container
    assert env_file.read_text().splitlines()[1].endswith(": FLOCK_TOKEN HF_TOKEN FAKE_KEY")
    environ = (slurm.dir / "jobs" / f"{first}.environ").read_text().splitlines()
    assert any(x.startswith("PATH=") for x in environ)
    for secret in ("tiny-token", "fake-key"):
        assert not any(secret in x for x in [env_file.read_text(), *argv, *environ]), secret

    # the Spot notice mid-turn: the worker drains once and exits 75; the same job name again
    tiny.flock("prompt", w0, "work [hold]")  # the fake Pi waits until killed
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
    out = tiny.flock("stop", "tiny", "--release")
    assert "released: every agent synced, compute freed" in out, out
    assert "controller kept: no job of this run serves it" in out
    # the drained worker's last heartbeat ends its slot at once; the controller's scancel then
    # finds the job ended, or ends it while the worker exits
    ours = f"CANCELLED by {os.getuid()}"
    ended = [(slurm.dir / "jobs" / f"{i}.state").read_text().strip() for i in (first, second)]
    assert ended[0] in ("FAILED", ours) and ended[1] in ("COMPLETED", ours), ended
    [wk] = json.loads(tiny.flock("status", "tiny", "--json"))["workers"]
    assert (wk["state"], wk["crashes"]) == ("dead", 0), wk


def test_hf_jobs(tiny, hub, tmp_path, monkeypatch):
    """compute: hf-jobs on the fake Hub (tests/fake_hub.py; a job runs on this machine): the
    controller starts the worker job (bootstrap image, the job env, the secrets apart), a turn,
    then stop --release drains the worker and its job completes."""
    monkeypatch.setenv("FLOCK_WHEEL_URL", "hf://buckets/ns/tiny/flock.whl")  # bootstrap mode
    tiny.controller(tmp_path / "ctl", compute="hf-jobs")
    tiny.flock("launch", tiny.spec(compute="hf-jobs", backend="hf-sandbox",
                                   env='FLUSH_INTERVAL: "0.2"',
                                   extra="namespace: ns\nbucket: hf://buckets/ns/tiny\n"),
               "--skip-preflight")
    assert hub.asked == ["/api/buckets/ns/tiny/tree/tiny%2F?recursive=true"]  # empty
    w0, = tiny.names("writer", 1)
    tiny.flock("prompt", w0, "one")
    assert tiny.turns(w0, 1)["cost_usd"] > 0

    w = "tiny-writer-w0000"
    [(job, sent)] = hub.jobs.items()
    assert (sent["namespace"], sent["dockerImage"], sent["command"]) == (
        "ns", BOOTSTRAP_IMAGE, bootstrap_command())
    env = sent["environment"]
    assert {k: env[k] for k in ("WORKER_ID", "SWARM", "TYPE", "BUCKET", "FLOCK_WHEEL_URL",
                                "FLOCK_HARNESS")} == {
        "WORKER_ID": w, "SWARM": "tiny", "TYPE": "writer", "BUCKET": "hf://buckets/ns/tiny",
        "FLOCK_WHEEL_URL": "hf://buckets/ns/tiny/flock.whl", "FLOCK_HARNESS": "harbor"}
    assert sent["secrets"] == {"FLOCK_TOKEN": "tiny-token", "FAKE_KEY": "fake-key",
                               "HF_TOKEN": "fake-hub-token"}  # hf-sandbox starts sandboxes with it
    assert not {"tiny-token", "fake-key", "fake-hub-token"} & set(env.values())
    assert {k: sent["labels"][k] for k in ("swarm", "type", "worker")} == {
        "swarm": "tiny", "type": "writer", "worker": w}
    assert sent["timeoutSeconds"] >= 24 * 3600  # lifetime + jitter, in whole seconds

    out = tiny.flock("stop", "tiny", "--release")
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
    tiny.flock("launch", spec, "--skip-preflight")
    names = tiny.names("writer", 4)
    tiny.flock("prompt", "--swarm", "tiny", "--type", "writer", "solve [hold]")
    def finished():
        status = json.loads(tiny.flock("status", "tiny", "--json"))["status"]
        return (status.get("success", {}).get("verification") == "passed"
                and all(t["workers_by_state"].get("dead", 0) == 2
                        for t in status["types"].values()) and status)
    status = tiny.wait(finished, timeout=60)
    assert status["released"] and not status["released"]["unsynced"]
    assert status["pending_prompts"] == 0
    assert "STOPPED ON SUCCESS" in tiny.flock("status", "tiny")
    assert "already succeeded" in tiny.flock("prompt", status["success"]["agent"], "again", code=1)
    turns = [e for name in names for e in tiny.events(name, "agent.turn")]
    assert any(e["name"] == "agent.turn_started" for e in turns)
    assert any(e["name"] == "agent.turn_interrupted" for e in turns)
    assert "launch a new run" in tiny.flock("launch", "tiny", code=1)
    released = tiny.flock("stop", "tiny", "--release")
    assert "this swarm succeeded; launch a new run" in released
    assert "resumes" not in released
