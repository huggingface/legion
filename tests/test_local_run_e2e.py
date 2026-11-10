"""A local run end to end (docs/local.md): the controller on 127.0.0.1, real worker subprocesses
running the `llm` harness against tests/fake_openai.py. No HF token, bucket or network.

1. `python -m flock.controller.localrun --prompt ... --until idle`: the tools folder ships,
   3 writers each run the prompt in one turn, a looping type finishes its loop, the run stops
   on its own, exits 0 and leaves the bundle (run.json, events.jsonl, events/<agent>.jsonl).
2. Ctrl-C (`--grace 0`) while a tool runs: nothing survives the run.
`flock launch` in the background, through its lifecycle: tests/test_runs.py."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from . import fake_openai

SPEC = """\
swarm: hello
compute: local
providers:
  fake: {{type: openai_compat, base_url: "{url}", api_key_env: FAKE_KEY, self_hosted: true}}
types:
  writer:
    harness: llm
    model: fake:fake-model
    replicas: 3
    agents_per_worker: 2
"""


@pytest.fixture
def fake_server(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    srv.url = f"http://127.0.0.1:{srv.server_port}/v1"
    yield srv
    srv.shutdown()


def _env(tmp_path: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
    return {**env, "FAKE_KEY": "fake-key", "HOME": str(tmp_path / "home")}


def _localrun(*args: str) -> list[str]:
    return [sys.executable, "-m", "flock.controller.localrun", *args]


def _run_dir(cwd: Path) -> Path:
    (run,) = [p for p in (cwd / ".flock").iterdir() if p.is_dir()]
    return run


def _bundle_events(run: Path, agent: str) -> list[dict]:
    return [json.loads(x) for x in (run / "events" / f"{agent}.jsonl").read_text().splitlines()]


WRITERS = ["writer-0000", "writer-0001", "writer-0002"]
LOOPERS = ["looper-0000", "looper-0001"]
# appended under types: a looping type next to the writers (its agents get --prompt too)
LOOPER = """\
    tools: [greet]
  looper:
    harness: llm
    model: fake:fake-model
    replicas: 2
    agents_per_worker: 2
    tools: [greet]
    loop:
      prompt: "bash: echo round {round}"
      every: 5s
      while: {max_rounds: 2}
      final: {prompt: "bash: echo final"}
"""
# each agent runs its own bash, then the folder tool, in one turn
PROMPT = 'script: [["bash", {"command": "echo hi-from-$FLOCK_AGENT"}], ["greet", {"name": "flock"}]]'


def test_flock_run_until_idle(tmp_path, fake_server):
    """`localrun --prompt ... --until idle`: the tools folder is uploaded as by `flock launch`,
    each of 3 writers runs the prompt in one turn (its own bash, then the folder tool); a
    looping type (`every: 5s`) is idle between rounds, so the run goes on until its loop is done
    (both rounds and the final round), then stops on idle, exits 0 and leaves the bundle
    (run.json, events.jsonl, events/<agent>.jsonl) and the health files."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "greet.py").write_text(GREET)
    (tmp_path / "spec.yaml").write_text(SPEC.format(url=fake_server.url) + LOOPER)
    t0 = time.monotonic()
    p = subprocess.run(_localrun("spec.yaml", "--prompt", PROMPT, "--until", "idle",
                                 "--skip-preflight"),
                       cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True,
                       timeout=120, check=False)
    took = time.monotonic() - t0
    assert p.returncode == 0, p.stdout + p.stderr
    assert "stopping (idle)" in p.stdout and "5 agents, " in p.stdout, p.stdout
    assert "tools: version " in p.stdout and ": greet" in p.stdout, p.stdout
    assert took < 60, took

    run = _run_dir(tmp_path)
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "idle" and doc["compute"] == "local"
    assert doc["totals"]["agents"] == 5 and doc["totals"]["workers"] == 3  # 2 + 1 per 2 agents
    assert doc["totals"]["tokens"]["fake-model"]["input"] > 0
    turns = {a["name"]: a["turns_completed"] for a in doc["agents"]}
    assert sorted(turns) == sorted(WRITERS + LOOPERS)
    assert [turns[n] for n in WRITERS] == [1, 1, 1], turns  # the prompt, one turn each
    for name in WRITERS:
        evs = _bundle_events(run, name)
        started = [e for e in evs if e["type"] == "turn_started"]
        assert started and started[0]["data"]["data"].get("tools_version"), started
        tool = [json.dumps(e) for e in evs if e["type"] == "message"
                and e["data"]["message"].get("role") == "toolResult"]
        assert f"hi-from-{name}" in tool[0] and "hello flock via folder tool" in tool[1], tool
    for name in LOOPERS:  # every round and the final one, whatever the prompt did meanwhile
        prompts = [(e["data"]["data"]["kind"], e["data"]["data"]["round"])
                   for e in _bundle_events(run, name) if e["type"] == "loop_prompt"]
        assert prompts == [("loop", 1), ("loop", 2), ("final", 2)], (name, prompts)
        assert turns[name] == 4, turns  # the prompt, 2 rounds, the final round
    merged = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    merged = [e for e in merged if e["agent"]]  # the swarm stream: control-plane events
    assert {e["agent"] for e in merged} == set(WRITERS + LOOPERS)
    assert len(merged) == sum(len(_bundle_events(run, a)) for a in WRITERS + LOOPERS)
    for log in (run / "logs").glob("hello-*-w*.log"):
        assert "drained; exiting" in log.read_text()  # drained, not killed
    assert not (tmp_path / ".flock" / "current").exists()
    assert "Authorization" not in (run / "run.json").read_text()
    # health (docs/health.md): the last snapshot, the hour's rollups and the viewer's run.json
    ctl = run / "controller"
    latest = json.loads((ctl / "metrics" / "hello" / "latest.json").read_text())
    assert latest["body"]["types"]["writer"]["by_state"], latest
    assert sum(t["input"] for t in latest["body"]["tokens"]["by_type"].values()) > 0
    rolled: dict[str, int] = {}  # summed: the run may straddle an hour
    for hour in (ctl / "rollups" / "hello").glob("*.jsonl"):
        for x in map(json.loads, hour.read_text().splitlines()):
            rolled[x["agent"]] = rolled.get(x["agent"], 0) + x["turns"]
    assert rolled == turns
    assert json.loads((ctl / "run.json").read_text())["status"] == "ended"


def test_flock_run_ctrl_c_kills_running_tools(tmp_path, fake_server):
    """--grace 0 and SIGINT while an agent's bash tool sleeps: the worker interrupts the turn,
    the adapter kills the tool's own session, and nothing survives the local run."""
    (tmp_path / "spec.yaml").write_text(SPEC.format(url=fake_server.url).replace(
        "replicas: 3", "replicas: 1"))
    pids = tmp_path / "pids"
    cmd = f"bash: sleep 60 & echo $$ $! > {pids}; wait"
    proc = subprocess.Popen(_localrun("spec.yaml", "--prompt", cmd, "--grace", "0",
                                   "--until", "50s", "--skip-preflight"),
                            cwd=tmp_path, env=_env(tmp_path), stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    alive: list[int] = []
    try:
        end = time.monotonic() + 30
        while not (pids.exists() and len(pids.read_text().split()) == 2):
            assert proc.poll() is None and time.monotonic() < end, proc.stdout.read()
            time.sleep(0.1)
        bash_pid, sleep_pid = (int(x) for x in pids.read_text().split())
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=40)
        from flock.controller.local import _alive

        end = time.monotonic() + 5
        while (alive := [p for p in (bash_pid, sleep_pid) if _alive(p)]) and \
                time.monotonic() < end:
            time.sleep(0.05)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for p in alive:
            os.kill(p, signal.SIGKILL)
    assert proc.returncode == 0, out
    assert alive == [], f"tool processes survived: {alive}"
    run = _run_dir(tmp_path)
    kinds = [e["type"] for e in _bundle_events(run, "writer-0000")]
    assert "turn_interrupted" in kinds, kinds  # the worker stopped the turn, not just died
    assert "terminating (SIGTERM)" in next((run / "logs").glob("hello-writer-w*.log")).read_text()


GREET = '''"""A folder tool for the local run tools test."""

from flock.tools import Context


def greet(name: str, ctx: Context) -> str:
    """Greet someone.

    Args:
        name: who to greet
    """
    return f"hello {name} via folder tool"
'''
