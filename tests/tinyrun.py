"""Tiny real runs (AGENTS.md "Tests"): a Harbor spec run in a temp folder, driven by the `flock`
CLI in-process (the same `flock.ops`/`flock.client` calls a Python API makes). The controller and
its workers are real processes running `harness: harbor`; only Harbor is faked, in each of them
(tests/harbor/site/sitecustomize.py): every agent's Trial is the real FlockPi on a fake
environment whose `pi` answers `reply-N` (tests/harbor/test_adapter.py), and the task pin and
download need no Hub. The `tiny` fixture (conftest.py) gives one per test, in its own working
directory, and kills whatever is left at the end."""

from __future__ import annotations

import contextlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from typer.testing import CliRunner

from flock import cli
from flock.client import Client

LOCAL = "{kind: local, worker_unknown_after: 10s, worker_dead_after: 11s, crash_backoff: 100ms}"
# A Harbor spec, one role `writer`. USD per 1M tokens: a fake turn (1500 tokens in, 100 out)
# costs $1.6. TINY_FAKE_HARBOR in the workers' env too: a fake Hub job gets only the spec's env.
SPEC = """\
name: tiny
model: fake-model
task: {{repo: "https://huggingface.co/datasets/ns/tasks", selector: t}}
inference: {{api_key_env: FAKE_KEY}}
environment: {{backend: {backend}}}
pricing: [1000, 1000, 1000]
compute: {compute}
workers: {{agents_per_worker: 2, env: {{TINY_FAKE_HARBOR: "1", HEARTBEAT_INTERVAL: "0.5"{env}}}}}
agents:
  roles:
    writer:
      count: {replicas}
"""


class TinyRun:
    def __init__(self, root: Path, monkeypatch) -> None:
        self.root, self.mp = root, monkeypatch
        self.procs: list[subprocess.Popen] = []

    def spec(self, replicas: int = 1, extra: str = "", env: str = "", compute: str = LOCAL,
             backend: str = "docker") -> str:
        """Write spec.yaml (SPEC, `env` added to the workers' env, plus `extra` YAML at the end
        of role `writer`) and return its name. `backend`: what compute allows (slurm: podman,
        hf-jobs: hf-sandbox); faked either way."""
        text = SPEC.format(replicas=replicas, env=env and f", {env}", compute=compute,
                           backend=backend) + extra
        (self.root / "spec.yaml").write_text(text)
        return "spec.yaml"

    def flock(self, *args: str, code: int = 0) -> str:
        """`flock ARGS` in-process; asserts the exit code, returns stdout and stderr."""
        r = CliRunner().invoke(cli.app, list(args))
        assert r.exit_code == code, (args, r.output, r.exception)
        return r.output

    def agent(self, name: str) -> dict:
        return json.loads(self.flock("status", name))

    def wait(self, what, timeout: float = 30.0):
        """Poll `what()` until it returns something truthy, and return that. A check that
        raises (status not there yet) is retried; the last error is re-raised at the timeout."""
        end = time.monotonic() + timeout
        while True:
            try:
                if got := what():
                    return got
                err = None
            except Exception as e:  # noqa: BLE001 - any failure of the check is retried
                err = e
            if time.monotonic() > end:
                if err:
                    raise err
                raise AssertionError(f"not met in {timeout:g} s")
            time.sleep(0.2)

    def turns(self, name: str, n: int) -> dict:
        """The agent's status once it has ended `n` turns, is idle again and its events show
        them (events can trail the heartbeat)."""
        return self.wait(lambda: (a := self.agent(name))["turns"] >= n and a["state"] == "idle"
                         and len(self.events(name, "agent.turn_ended")) >= n and a)

    def events(self, name: str, only: str = "") -> list[dict]:
        """`flock status NAME --events --json`: the normalised events whose name starts with
        `only`."""
        out = self.flock("status", name, "--events", "--json", "--only", only)
        return [json.loads(x) for x in out.splitlines()]

    def said(self, agent: str) -> list[str]:
        """The prompts of the agent's Pi session, in order: the fake environment's session file
        (tests/harbor/test_adapter.py FakeEnv), as its worker last kept it. A Trial's first
        prompt starts with the task's instruction."""
        kept = max(self.root.rglob(f"_harbor/{agent}/pi/sessions/s.jsonl"),
                   key=lambda p: p.stat().st_mtime)
        return [json.loads(x)["prompt"] for x in kept.read_text().splitlines()]

    def controller(self, data: Path, port: int = 0, compute: str = "local") -> int:
        """Start `flock-controller` for `compute` with its state in `data` (a controller that
        can crash, unlike the local run's) on `port` (0: a free one), point the CLI at it and
        wait until it answers. Returns the port."""
        if not port:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"  # also what its workers use (not SPACE_HOST)
        env = {**os.environ, "FLOCK_URL": url, "FLOCK_TOKEN": "tiny-token", "FLOCK_COMPUTE":
               compute, "FLOCK_DATA": str(data), "PORT": str(port),
               "FLOCK_RECONCILE_INTERVAL": "0.2"}
        with (data.parent / "controller.log").open("ab") as log:
            self.procs.append(subprocess.Popen(
                [sys.executable, "-m", "flock.controller.main"], env=env, stdout=log,
                stderr=log, stdin=subprocess.DEVNULL, start_new_session=True))
        self.mp.setenv("FLOCK_URL", url)
        self.mp.setenv("FLOCK_TOKEN", "tiny-token")
        self.wait(Client().answers)
        return port

    def crash(self) -> None:
        """SIGKILL the controller: no final flush, no drain."""
        p = self.procs.pop()
        p.kill()
        p.wait(10)

    def close(self) -> None:
        """Kill every process left working in the test's folder: controllers, workers, tools."""
        for d in Path("/proc").glob("[0-9]*"):
            try:
                inside = Path(os.readlink(d / "cwd")).is_relative_to(self.root)
            except OSError:
                continue
            if inside and int(d.name) != os.getpid():
                with contextlib.suppress(ProcessLookupError):
                    os.kill(int(d.name), signal.SIGKILL)
        for p in self.procs:
            p.wait(10)
