"""What the board resource does around a Space that needs no Space: a deploy with the
Hub unreachable, the deploy worker's lifetime, and the export guard on a real local
board. The Space itself is tested in test_message_board_live.py."""

import os
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest
from msgboard.client import Client

from flock.config import BUILTIN
from flock.resources import load_module

RESOURCES = BUILTIN / "resources"
board = load_module(RESOURCES / "message_board.py")


async def test_a_failed_deploy_is_a_launch_error_and_keeps_the_staged_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # huggingface_hub refuses every Hub call
    monkeypatch.setenv("HF_TOKEN", "hf_never-sent")
    secrets = []
    resource = board.Resource(
        {"space_id": "user/board"},
        state_dir=tmp_path / "state",
        agents=[],
        secrets=secrets,
    )
    with pytest.raises(
        RuntimeError,
        match=r"^Could not deploy the board to Space user/board: OfflineModeIsEnabled",
    ):
        await resource.start()
    token_file = tmp_path / "state/operator.token"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    staged = token_file.read_text().strip()
    assert any(secret == staged for secret in secrets)  # redacted, not shown
    with pytest.raises(
        RuntimeError, match="Space user/board did not start, so there is nothing"
    ):
        await resource.export(tmp_path)
    await resource.stop()


def running(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    try:  # an orphan nobody has reaped yet has exited too
        return (
            Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
        )
    except FileNotFoundError:
        return True


def test_the_deploy_worker_exits_with_its_launcher():
    # Started as the resource starts it. Its stdin stays open here, so only the
    # launcher's death can end its wait for a request.
    launcher = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import os, subprocess, sys, time\n"
            "worker = subprocess.Popen([sys.executable, '-P', sys.argv[1], str(os.getpid())])\n"
            "print(worker.pid, flush=True)\n"
            "time.sleep(60)\n",
            str(RESOURCES / "message_board_deploy.py"),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    worker = int(launcher.stdout.readline())
    try:
        time.sleep(1)
        assert running(worker)
        launcher.send_signal(signal.SIGKILL)
        launcher.wait()
        deadline = time.monotonic() + 5
        while running(worker) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not running(worker)
    finally:
        if running(worker):
            os.kill(worker, signal.SIGKILL)
        launcher.kill()
        launcher.stdin.close()
        launcher.stdout.close()


async def test_export_fails_when_the_board_lost_the_runs_channels(tmp_path):
    resource = board.Resource({}, state_dir=tmp_path / "state", agents=[], secrets=[])
    await resource.start()
    try:
        token = (tmp_path / "state/data/operator.token").read_text().strip()
        # As a board restored from a snapshot older than the run would be.
        with Client(resource.metadata["url"], token) as operator:
            operator.delete_channel(operator.channel_id("general"), "general")
        with pytest.raises(RuntimeError, match="no longer has the run's channels"):
            await resource.export(tmp_path)
    finally:
        await resource.stop()
