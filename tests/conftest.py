import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from flock import resources

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

RESOURCES_DIR = Path(__file__).resolve().parents[1] / "resources"


@pytest.fixture
def message_board_module(monkeypatch):
    original = resources.load_module
    module = original(RESOURCES_DIR / "message_board.py")
    monkeypatch.setattr(
        resources,
        "load_module",
        lambda path: module if path.name == "message_board.py" else original(path),
    )
    return module


@pytest.fixture
def message_board_provider(message_board_module):
    return message_board_module.Resource


class FakeSpace:
    """msgboard.space's deploy() on a real local board. As on a Space, each deploy
    restarts the board, makes its operator token the admin credential and keeps the data."""

    def __init__(self, directory):
        self.directory = directory
        self.deploys = []
        self.process = None
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            self.port = reservation.getsockname()[1]
        self.url = f"http://127.0.0.1:{self.port}"

    def resolve_space_id(self, space_id, *, hf_token=None):
        return space_id if "/" in space_id else f"user/{space_id}"

    def deploy(self, space_id, *, operator_token, private=None, hf_token=None):
        from msgboard.space import Deployment
        from msgboard.space.app import ensure_board

        self.deploys.append(
            {"operator_token": operator_token, "private": private, "hf_token": hf_token}
        )
        self.stop()
        ensure_board(self.directory / "data", operator_token)
        self.start()  # it serves a little later, as a Space does
        return Deployment(
            space_id=space_id,
            url=self.url,
            bucket_id=f"{space_id}-bucket",
            created=len(self.deploys) == 1,
            private=bool(private),
            commit="c" * 40,
            msgboard_revision="r" * 40,
        )

    def start(self):
        sock = self.directory / "board.sock"
        sock.unlink(missing_ok=True)
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(RESOURCES_DIR / "message_board_server.py"),
                str(self.directory / "data"),
                "127.0.0.1",
                str(self.port),
                str(sock),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=15)


@pytest.fixture
def fake_space(message_board_module, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_test-launcher-secret")
    space = FakeSpace(tmp_path / "space")
    monkeypatch.setattr(message_board_module, "deploy", space.deploy)
    monkeypatch.setattr(
        message_board_module, "resolve_space_id", space.resolve_space_id
    )
    try:
        yield space
    finally:
        space.stop()
