"""A board served under a path prefix (the multi-board Space: `https://host/boards/<id>`)
keeps that prefix on every request, worker and controller, and gets the token in
X-Msgboard-Token."""
import pytest

from flock.comms.board import SyncBoard

from . import fake_msgboard

PREFIX = "/boards/demo"


@pytest.fixture
def board():
    b, server, url = fake_msgboard.serve(prefix=PREFIX)
    b.url = url
    yield b
    server.shutdown()


def _all_prefixed(board) -> None:
    assert board.raw_paths and all(p.startswith(PREFIX + "/v1/") for p in board.raw_paths), \
        [p for p in board.raw_paths if not p.startswith(PREFIX + "/v1/")]
    assert board.token_headers and set(board.token_headers) == {"X-Msgboard-Token"}


def test_controller_board_keeps_the_prefix(board):
    sync = SyncBoard(board.url + "/")  # a trailing slash is not a different board
    try:
        token = sync.join(board.invite, "flock-sw")["token"]
        (channel,) = sync.channels(token, "flock-test")
        sync.post(token, channel["id"], "kickoff", key="k1")
    finally:
        sync.close()
    _all_prefixed(board)
