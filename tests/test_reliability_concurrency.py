"""Eight isolated local experiments against a synthetic provider; no paid model or Hub calls."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

import yaml

from . import fake_openai


def test_eight_concurrent_experiments(tmp_path):
    server = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    jobs = []
    try:
        for index in range(8):
            folder = tmp_path / str(index)
            folder.mkdir()
            n = 32 if index in {0, 1} else 4
            spec = {"swarm": f"isolation-{index}", "compute": "local",
                "providers": {"fake": {"type": "openai_compat",
                    "base_url": f"http://127.0.0.1:{server.server_port}/v1",
                    "api_key_env": "FAKE_KEY", "self_hosted": True}},
                "comms": {"managed": True, "wake": [], "poll_cycle_s": 5},
                "types": {"worker": {"harness": "llm", "model": "fake:fake-model",
                    "replicas": n, "agents_per_worker": 8, "comms": True,
                    "env": {"FLOCK_COMMS_RATE": "0", "FLOCK_COMMS_AGENT_RATE": "0"}}}}
            (folder / "spec.yaml").write_text(yaml.safe_dump(spec))
            prompt = 'script: ' + json.dumps([["post", {"text": f"only-run-{index}"}]])
            log = (folder / "launcher.log").open("w")
            process = subprocess.Popen([sys.executable, "-m", "flock.cli", "run",
                str(folder / "spec.yaml"), "--data-dir", str(folder / "runs"),
                "--skip-preflight", "--prompt", prompt, "--until", "idle"],
                env={**os.environ, "FAKE_KEY": "fake", "FLOCK_BOARD_LOCAL_ROOT": "/tmp"},
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((process, log, folder, n))
        deadline = time.monotonic() + 240
        for process, log, folder, n in jobs:
            process.wait(timeout=max(1, deadline - time.monotonic()))
            log.close()
            assert process.returncode == 0, (folder / "launcher.log").read_text()
            bundle, = (folder / "runs").glob("*/run.json")
            result = json.loads(bundle.read_text())
            assert result["totals"]["turns_completed"] == n
            assert result["totals"]["agents"] == n
            import sqlite3
            with sqlite3.connect(bundle.parent / "board/board.db") as db:
                bodies = [r[0] for r in db.execute("SELECT body FROM messages")]
            own = f"only-run-{folder.name}"
            assert sum(body == own for body in bodies) == n
            assert not any("only-run-" in body and body != own for body in bodies)
            assert not (folder / "runs/current").exists()
    finally:
        for process, log, _, _ in jobs:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            log.close()
        server.shutdown()
