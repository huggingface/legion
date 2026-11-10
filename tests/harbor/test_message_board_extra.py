"""message_board_extra.* (flock's board tools beyond shepherd's) against a managed local board."""

from pathlib import Path

import pytest
from msgboard.client import BoardError, Client

import flock.harbor
from flock.harbor.tools.runner import load_module

TOOLS_DIR = Path(flock.harbor.__file__).with_name("builtin") / "tools"
message_board_extra = load_module(TOOLS_DIR / "message_board_extra.py")
BoardResource = load_module(TOOLS_DIR.parent / "resources" / "message_board.py").Resource


@pytest.fixture
async def board(tmp_path, monkeypatch):
    runtime = BoardResource({}, state_dir=tmp_path / "state", agents=[], secrets=[])
    try:
        await runtime.start()
        monkeypatch.setenv("MSGBOARD_URL", runtime.url)
        monkeypatch.delenv("MSGBOARD_SOCKET", raising=False)
        yield runtime
    finally:
        await runtime.stop()


async def test_reactions_and_channel_docs(board, monkeypatch):
    identity = board.provision(["agent-001"])[0]
    monkeypatch.setenv("MSGBOARD_TOKEN", identity["token"])
    with Client(board.url, identity["token"]) as client:
        posted = client.post(client.channel_id("general"), "who takes the plan?")
    message_id = posted.get("message", posted)["id"]
    assert message_board_extra.react(message_id, "👍")["reactions"][0]["count"] == 1
    assert message_board_extra.doc()["version"] == 0
    assert message_board_extra.edit_doc("# plan", 0)["version"] == 1
    assert message_board_extra.doc()["body"] == "# plan"
    with pytest.raises(BoardError, match="doc_conflict"):
        message_board_extra.edit_doc("# stale", 0)
