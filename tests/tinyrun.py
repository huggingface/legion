"""Tiny real runs (AGENTS.md "Tests"): a `compute: local` swarm in a temp folder, driven by the
`flock` CLI in-process (the same `flock.ops`/`flock.client` calls a Python API makes). The
controller and its workers are real processes running the `llm` harness against
tests/fake_openai.py; nothing of flock is faked. The `tiny` fixture (conftest.py) gives one per
test, in its own working directory, and kills whatever is left at the end."""

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

# USD per 1M tokens, so one fake turn (~100 tokens in, 7 out) costs about $0.1
SPEC = """\
swarm: tiny
compute: {{kind: local, worker_unknown_after: 10s, worker_dead_after: 11s, crash_backoff: 100ms}}
providers:
  fake: {{type: openai_compat, base_url: "{url}", api_key_env: FAKE_KEY,
          pricing: {{"*": [1000, 1000, 1000]}}}}
types:
  writer:
    harness: llm
    model: fake:fake-model
    replicas: {replicas}
    agents_per_worker: 2
    env: {{HEARTBEAT_INTERVAL: "1"{env}}}
"""


class TinyRun:
    def __init__(self, root: Path, url: str, monkeypatch) -> None:
        self.root, self.url, self.mp = root, url, monkeypatch
        self.log = root / "requests.jsonl"  # every model request (fake_openai)
        self.procs: list[subprocess.Popen] = []

    def spec(self, replicas: int = 1, extra: str = "", env: str = "") -> str:
        """Write spec.yaml (SPEC, `env` added to the type's env, plus `extra` YAML at the end
        of the type) and return its name."""
        text = SPEC.format(url=self.url, replicas=replicas, env=env and f", {env}") + extra
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
        """Poll `what()` until it returns something truthy, and return that."""
        end = time.monotonic() + timeout
        while not (got := what()):
            assert time.monotonic() < end, f"not met in {timeout:g} s"
            time.sleep(0.2)
        return got

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

    def results(self, name: str) -> list[str]:
        """The text of each tool result in the agent's events, in order."""
        return [d["message"]["content"][0]["text"] for e in self.events(name, "agent.message")
                if (d := e["data"])["message"]["role"] == "toolResult"]

    def said(self, agent: str) -> list[list[str]]:
        """The user messages of each model request made for `agent` (its system message names
        it), in order."""
        bodies = [json.loads(x)["body"] for x in self.log.read_text().splitlines()]
        return [[m["content"] for m in b["messages"] if m["role"] == "user"] for b in bodies
                if b["messages"][0]["content"].startswith(f"You are {agent},")]

    def controller(self, data: Path, port: int = 0) -> int:
        """Start `flock-controller` for `compute: local` with its state in `data` (a controller
        that can crash, unlike the local run's) on `port` (0: a free one), point the CLI at it
        and wait until it answers. Returns the port."""
        if not port:
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
        url = f"http://127.0.0.1:{port}"  # also what its workers use (not SPACE_HOST)
        env = {**os.environ, "FLOCK_URL": url, "FLOCK_TOKEN": "tiny-token", "FLOCK_COMPUTE":
               "local", "FLOCK_DATA": str(data), "PORT": str(port),
               "FLOCK_RECONCILE_INTERVAL": "0.5"}
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
