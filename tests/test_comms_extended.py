"""Multi-channel and private DM delivery through one rotating worker poller."""
import asyncio

import pytest

from flock.models import ALL_COMMS_TOOLS, CommsRules

from . import fake_msgboard
from .test_comms import Team, custom
from .worker_fakes import until


@pytest.fixture
def board():
    b, server, url = fake_msgboard.serve()
    b.url = url
    yield b
    server.shutdown()


async def test_channels_dm_reaction_and_private_notices(tmp_path, board):
    team = Team(tmp_path, board, ["cedar", "birch", "pine"])
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        out, error = await team.call("cedar", "create_channel", name="memory", purpose="Memory work")
        assert not error, out
        cid = out["id"]
        await team.call("birch", "join_channel", name="memory")
        out, error = await team.call("cedar", "post", channel="memory", text="@birch please review", thread="new")
        assert not error, out
        await until(lambda: custom(team.agents["birch"], "comms_mention"))
        mention = custom(team.agents["birch"], "comms_mention")[0]
        assert mention["channel"] == "memory" and mention["channel_id"] == cid
        assert not custom(team.agents["pine"], "comms_mention")
        _, error = await team.call("birch", "post", channel="memory", text="Reviewing it", thread=out["posted"])
        assert not error
        await until(lambda: custom(team.agents["cedar"], "comms_thread_reply"))
        result, error = await team.call("birch", "react", msg_id=out["posted"], emoji="👍")
        assert not error, result
        assert next(m for m in board.msgs if m["id"] == out["posted"])["reactions"]
        result, error = await team.call("cedar", "dm", to="birch", text="Private review question")
        assert not error, result
        await until(lambda: custom(team.agents["birch"], "comms_dm"))
        assert not custom(team.agents["pine"], "comms_dm")
        read, error = await team.call("birch", "read_dms")
        assert not error and read["messages"][0]["text"] == "Private review question"
        read, error = await team.call("pine", "read_dms")
        assert not error and read["messages"] == []
        page, error = await team.call("birch", "read", channel="memory")
        assert not error and page["messages"][0]["text"] == "@birch please review"
        assert len([t for t in asyncio.all_tasks() if t.get_name() == "comms-poll"]) == 1
    finally:
        await team.close()


async def test_channel_creation_cap_survives_worker_state(tmp_path, board):
    team = Team(tmp_path, board, ["cedar"])
    team.spec.extended = True
    team.spec.rules = CommsRules(max_channels_per_agent=1)
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        _, error = await team.call("cedar", "create_channel", name="one", purpose="work")
        assert not error
        rt = team.agents["cedar"]
        assert rt.state.comms_created_channels
        result, error = await team.call("cedar", "create_channel", name="two", purpose="work")
        assert error and "limit" in result["error"]
    finally:
        await team.close()


async def test_external_dm_delivered_to_its_recipient_by_rotating_poll(tmp_path, board):
    team = Team(tmp_path, board, ["cedar", "birch"])
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        board.token_for("human", "user")
        human = board.principal("human", "user")
        birch = board.principal("birch")
        cid = max(board.channels) + 1
        board.channels[cid], board.seq[cid] = "dm-test", 0
        board.dms[cid] = frozenset((human["id"], birch["id"]))
        board.members[cid] = set(board.dms[cid])
        board.post("human", cid, "private hello")
        await until(lambda: custom(team.agents["birch"], "comms_dm"))
        assert not custom(team.agents["cedar"], "comms_dm")
        assert team.agents["birch"].state.comms_cursor
    finally:
        await team.close()
