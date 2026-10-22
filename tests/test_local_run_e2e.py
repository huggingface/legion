"""`flock run` end to end (docs/local.md): the real CLI in a subprocess, the controller on
127.0.0.1, real worker subprocesses running the `llm` harness against tests/fake_openai.py.
No HF token, bucket or network.

1. `--prompt ... --until idle`: 3 agents each get the prompt, finish a turn, the run stops on
   its own, exits 0 and leaves the bundle (run.json, events.jsonl, events/<agent>.jsonl).
2. A second terminal: plain `flock prompt` / `flock events` find the run through
   `.flock/current`; Ctrl-C (SIGINT) stops it cleanly."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

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


def _flock(*args: str) -> list[str]:
    return [sys.executable, "-m", "flock.cli", *args]


def _run_dir(cwd: Path) -> Path:
    (run,) = [p for p in (cwd / ".flock").iterdir() if p.is_dir()]
    return run


def _bundle_events(run: Path, agent: str) -> list[dict]:
    return [json.loads(x) for x in (run / "events" / f"{agent}.jsonl").read_text().splitlines()]


AGENTS = ["writer-0000", "writer-0001", "writer-0002"]


def test_flock_run_three_llm_agents_until_idle(tmp_path, fake_server):
    (tmp_path / "spec.yaml").write_text(SPEC.format(url=fake_server.url))
    t0 = time.monotonic()
    p = subprocess.run(
        _flock("run", "spec.yaml", "--prompt", "bash: echo hi-from-$FLOCK_AGENT", "--until",
               "idle", "--skip-preflight"),
        cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True, timeout=90,
        check=False)
    took = time.monotonic() - t0
    assert p.returncode == 0, p.stdout + p.stderr
    assert "stopping (idle)" in p.stdout and "3 agents, 3 turns" in p.stdout, p.stdout
    assert took < 60, took

    run = _run_dir(tmp_path)
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "idle" and doc["compute"] == "local"
    assert doc["totals"]["agents"] == 3 and doc["totals"]["turns_completed"] == 3
    assert doc["totals"]["workers"] == 2  # agents_per_worker 2: two worker processes
    assert doc["totals"]["tokens"]["fake-model"]["input"] > 0
    assert sorted(a["name"] for a in doc["agents"]) == AGENTS
    for name in AGENTS:
        evs = _bundle_events(run, name)
        kinds = [e["type"] for e in evs]
        assert "turn_started" in kinds and "turn_ended" in kinds, kinds
        tool = [e for e in evs if e["type"] == "message"
                and e["data"]["message"].get("role") == "toolResult"]
        assert tool and f"hi-from-{name}" in json.dumps(tool[0]), tool  # its own bash ran
    merged = [json.loads(x) for x in (run / "events.jsonl").read_text().splitlines()]
    assert {e["agent"] for e in merged} == set(AGENTS)
    assert len(merged) == sum(len(_bundle_events(run, a)) for a in AGENTS)
    for log in (run / "logs").glob("hello-writer-w*.log"):
        assert "drained; exiting" in log.read_text()  # drained, not killed
    assert not (tmp_path / ".flock" / "current").exists()
    assert "Authorization" not in (run / "run.json").read_text()


def test_flock_run_second_terminal_and_ctrl_c(tmp_path, fake_server):
    (tmp_path / "spec.yaml").write_text(SPEC.format(url=fake_server.url))
    env = _env(tmp_path)
    proc = subprocess.Popen(_flock("run", "spec.yaml", "--until", "50s", "--skip-preflight"),
                            cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    try:
        current = tmp_path / ".flock" / "current"
        end = time.monotonic() + 30
        while not current.exists():
            assert proc.poll() is None and time.monotonic() < end, proc.stdout.read()
            time.sleep(0.1)

        def flock(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(_flock(*args), cwd=tmp_path, env=env, capture_output=True,
                                  check=False,
                                  text=True, timeout=60)

        r = flock("prompt", "writer-0001", "bash: echo second-terminal")
        assert r.returncode == 0, r.stdout + r.stderr
        end = time.monotonic() + 40
        while True:
            r = flock("events", "writer-0001", "--pretty")
            assert r.returncode == 0, r.stderr
            if "turn_ended" in r.stdout:
                break
            assert time.monotonic() < end, r.stdout
            time.sleep(0.5)
        assert "second-terminal" in r.stdout
        r = flock("status", "hello")
        assert r.returncode == 0 and "writer" in r.stdout, r.stdout + r.stderr
        proc.send_signal(signal.SIGINT)
        out, _ = proc.communicate(timeout=40)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    assert proc.returncode == 0, out
    assert "stopping (interrupted)" in out and "bundle:" in out, out
    run = _run_dir(tmp_path)
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "interrupted"
    one = next(a for a in doc["agents"] if a["name"] == "writer-0001")
    assert one["turns_completed"] == 1
    assert yaml.safe_load(yaml.safe_dump(doc["spec"]))["compute"] == {"kind": "local"}
    assert not (tmp_path / ".flock" / "current").exists()


def test_flock_run_ctrl_c_kills_running_tools(tmp_path, fake_server):
    """--grace 0 and SIGINT while an agent's bash tool sleeps: the worker interrupts the turn,
    the adapter kills the tool's own session, and nothing survives `flock run`."""
    (tmp_path / "spec.yaml").write_text(SPEC.format(url=fake_server.url).replace(
        "replicas: 3", "replicas: 1"))
    pids = tmp_path / "pids"
    cmd = f"bash: sleep 60 & echo $$ $! > {pids}; wait"
    proc = subprocess.Popen(_flock("run", "spec.yaml", "--prompt", cmd, "--grace", "0",
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


GREET = '''"""A folder tool for the `flock run` tools test."""

from flock.tools import Context


def greet(name: str, ctx: Context) -> str:
    """Greet someone.

    Args:
        name: who to greet
    """
    return f"hello {name} via folder tool"
'''


def test_flock_run_ships_the_tools_folder(tmp_path, fake_server):
    """`flock run` uploads tools/ like `flock apply` (docs/tools.md): the spec is applied with
    its tools_version, and an agent's turn calls the folder tool."""
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "greet.py").write_text(GREET)
    spec = SPEC.format(url=fake_server.url).replace("replicas: 3", "replicas: 1")
    (tmp_path / "spec.yaml").write_text(spec + "    tools: [greet]\n")
    p = subprocess.run(
        _flock("run", "spec.yaml", "--prompt", 'call: greet {"name": "flock"}', "--until",
               "idle", "--skip-preflight"),
        cwd=tmp_path, env=_env(tmp_path), capture_output=True, text=True, timeout=90,
        check=False)
    assert p.returncode == 0, p.stdout + p.stderr
    assert "tools: version " in p.stdout and ": greet" in p.stdout, p.stdout
    evs = _bundle_events(_run_dir(tmp_path), "writer-0000")
    started = [e for e in evs if e["type"] == "turn_started"]
    assert started and started[0]["data"]["data"].get("tools_version"), started
    tool = [e for e in evs if e["type"] == "message"
            and e["data"]["message"].get("role") == "toolResult"]
    assert tool and "hello flock via folder tool" in json.dumps(tool[0]), tool
