"""Space-mode logic of the board resource that needs no Space: a deploy with the Hub
unreachable, and revocation and export against real local boards. The Space itself is
tested live in test_message_board_live.py."""

import asyncio
import stat
import time
from pathlib import Path

import pytest
from msgboard import auth
from msgboard.client import BoardError, Client

from flock.resources import load_module

board = load_module(Path(__file__).resolve().parents[1] / "resources/message_board.py")


async def test_a_failed_deploy_is_a_launch_error_and_keeps_the_staged_token(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")  # huggingface_hub refuses every Hub call
    monkeypatch.setenv("HF_TOKEN", "hf_never-sent")
    resource = board.Resource(
        {"space_id": "user/board"}, state_dir=tmp_path / "state", agents=[], secrets=[]
    )
    with pytest.raises(
        RuntimeError,
        match=r"^Could not deploy the board to Space user/board: OfflineModeIsEnabled",
    ):
        await resource.start()
    assert resource.deploy_process.returncode == 0
    token_file = tmp_path / "state/operator.token"
    assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
    staged = token_file.read_text().strip()
    assert any(secret == staged for secret in resource.secrets)  # redacted, not shown
    with pytest.raises(
        RuntimeError, match="Space user/board did not start, so there is nothing"
    ):
        await resource.export(tmp_path)
    await resource.stop()


async def test_revocation_tries_every_token_and_names_only_failures(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(board, "REVOKE_SECONDS", 2)
    agents = ["agent-001", "agent-002", "agent-003"]
    resource = board.Resource(
        {}, state_dir=tmp_path / "state", agents=agents, secrets=[]
    )
    await resource.start()
    try:
        tokens = {name: c["token"] for name, c in resource.credentials.items()}
        # A token that can't be parsed keeps failing.
        resource.credentials["agent-001"]["token"] = "not-a-board-token"
        # A well-formed token the board never issued: its 404 not_found means revoked.
        resource.credentials["agent-003"]["token"] = auth.new_token()[1]
        began = time.monotonic()
        with pytest.raises(
            RuntimeError, match="board tokens of agent-001, which may still work"
        ):
            await asyncio.to_thread(resource._revoke_agent_tokens)
        assert 2 <= time.monotonic() - began < 4
        with Client(resource.url, tokens["agent-002"]) as agent:
            with pytest.raises(BoardError, match="401"):
                agent.me()
        for name in ("agent-001", "agent-003"):  # their real tokens were never tried
            with Client(resource.url, tokens[name]) as agent:
                assert agent.me()["external_id"] == name
    finally:
        await resource.stop()


async def test_export_fails_when_the_board_lost_the_runs_channels(tmp_path):
    resource = board.Resource({}, state_dir=tmp_path / "state", agents=[], secrets=[])
    await resource.start()
    try:
        # As a board restored from a snapshot older than the run would be.
        resource.client.delete_channel(resource.client.channel_id("general"), "general")
        with pytest.raises(RuntimeError, match="no longer has the run's channels"):
            await resource.export(tmp_path)
    finally:
        await resource.stop()
