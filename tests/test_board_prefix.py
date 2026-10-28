"""A board served under a path prefix (the multi-board Space: `https://host/boards/<id>`)
keeps that prefix on every request, worker and controller, and gets the token in
X-Msgboard-Token."""
import pytest

from flock.comms.board import SyncBoard
from flock.models import ALL_COMMS_TOOLS

from . import fake_msgboard
from .test_comms import Team, custom
from .worker_fakes import until

PREFIX = "/boards/hierarchy-v6"


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


async def test_worker_comms_keep_the_prefix(tmp_path, board):
    team = Team(tmp_path, board, ["cedar", "birch"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()  # join + activity cursor
        out, error = await team.call("cedar", "post", text="@birch prefix check", thread="new")
        assert not error, out
        await until(lambda: custom(team.agents["birch"], "comms_mention"))  # activity long-poll
        page, error = await team.call("birch", "read")
        assert not error and page["messages"]
        _, error = await team.call("birch", "search", query="prefix")
        assert not error
        _, error = await team.call("birch", "react", msg_id=out["posted"], emoji="👍")
        assert not error
        votes, error = await team.call("cedar", "votes", msg_id=out["posted"])
        assert not error and votes["msg_id"] == out["posted"]
        doc, error = await team.call("cedar", "doc")
        assert not error, doc
    finally:
        await team.close()
    paths = {p.split("?")[0] for p in board.raw_paths}
    for route in ("/v1/join", "/v1/activity", "/v1/search", f"/v1/channels/{12}/doc"):
        assert PREFIX + route in paths, route
    _all_prefixed(board)


def test_controller_board_keeps_the_prefix(board):
    sync = SyncBoard(board.url + "/")  # a trailing slash is not a different board
    try:
        token = sync.join(board.invite, "flock-sw")["token"]
        (channel,) = sync.channels(token, "flock-test")
        sync.post(token, channel["id"], "kickoff", key="k1")
    finally:
        sync.close()
    _all_prefixed(board)
