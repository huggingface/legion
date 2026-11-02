"""SIGUSR1 (a Spot notice: the cluster's watcher signals every process of the job) stops the
controller and the world exactly like SIGTERM, once, also when signals repeat (docs/slurm.md).
The worker's side is in test_worker_preempt.py."""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import socket
import subprocess
import sys
import time

import httpx
import pytest

from .test_controller_job import FakeBucket, StateSync, _built, _free_port
from .test_run3_experiment import ROOT, _env

SEQUENCES = [("SIGUSR1", "SIGUSR1"), ("SIGUSR1", "SIGTERM"), ("SIGTERM",)]


@pytest.mark.parametrize("signals", SEQUENCES)
async def test_controller_stops_once(tmp_path, monkeypatch, caplog, signals):
    import flock.controller.main as main_mod

    caplog.set_level(logging.INFO)
    bucket = FakeBucket(tmp_path / "bucket")
    monkeypatch.setattr(main_mod, "state_sync_from_env", lambda env, d: StateSync(bucket, d))
    port = _free_port()
    b = _built(tmp_path, PORT=str(port), FLOCK_STATUS_PORT="0", FLOCK_STATE_URL="hf://x/y/z")
    task = asyncio.create_task(main_mod.serve(b, host="127.0.0.1"))
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}") as c:
        for _ in range(100):
            try:
                if (await c.get("/healthz")).status_code == 200:
                    break
            except httpx.TransportError:
                await asyncio.sleep(0.05)
    for sig in signals:
        os.kill(os.getpid(), getattr(signal, sig))
        await asyncio.sleep(0.05)
    await asyncio.wait_for(task, 20)
    lines = [r.getMessage() for r in caplog.records]
    assert sum(m.endswith(": shutting down") for m in lines) == 1
    assert f"{signals[0]}: shutting down" in lines
    assert sum(m.startswith("final push: ok") for m in lines) == 1
    b.store.close()


@pytest.mark.parametrize("signals", SEQUENCES)
def test_world_serve_stops_once(tmp_path, signals):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**_env(tmp_path), "FLOCK_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_WS": str(tmp_path / "ws")}
    proc = subprocess.Popen([sys.executable, "-m", "flock.cli", "world", "serve",
                             "chatbot3:Chatbot3World", "--port", str(port)],
                            cwd=ROOT / "worlds" / "chatbot3", env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                httpx.get(f"http://127.0.0.1:{port}/healthz", timeout=2)
                break
            except httpx.TransportError:
                assert time.monotonic() < deadline and proc.poll() is None
                time.sleep(0.2)
        for sig in signals:
            proc.send_signal(getattr(signal, sig))
            time.sleep(0.1)
        out, _ = proc.communicate(timeout=20)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    # uvicorn shuts down gracefully, then re-raises the SIGTERM it captured: -15, never -10
    # (a USR1 left at its default action would kill the process at once)
    assert proc.returncode in (0, -signal.SIGTERM), (proc.returncode, out)
    assert out.count("SIGUSR1: shutting down") == (1 if signals[0] == "SIGUSR1" else 0)
