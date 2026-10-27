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
    team.hub.poll_min_interval = 0.01  # functional delivery; production spacing has a virtual-time test
    team.hub.poll_cycle_s = 0.03
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
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
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
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        board.token_for("human", "user")
        human = board.principal("human", "user")
        birch = board.principal("birch")
        with board.lock:  # the board's server threads read these maps
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


async def test_poller_survives_a_non_board_exception(tmp_path, board, monkeypatch, caplog):
    """A bad page (ValueError, KeyError, a non-JSON proxy reply) is logged with the agent's
    name and backed off; the single worker poller keeps delivering notices."""
    from flock.comms.board import AsyncBoard

    real, failures = AsyncBoard.activity, []

    async def flaky(self, cred, channel, after, wait=0):
        if after is not None and len(failures) < 2:
            failures.append(cred)
            raise ValueError("invalid literal for int() with base 10: 'oops'")
        return await real(self, cred, channel, after, wait)

    monkeypatch.setattr(AsyncBoard, "activity", flaky)
    team = Team(tmp_path, board, ["cedar", "birch"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.hub.backoff_s = 0.01
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        await until(lambda: len(failures) == 2)
        poller = team.hub._poller
        out, error = await team.call("cedar", "post", text="@birch after the bad page", thread="new")
        assert not error, out
        await until(lambda: custom(team.agents["birch"], "comms_mention"))
        assert team.hub._poller is poller and not poller.done()
        assert any("ValueError" in r.getMessage() and ("cedar" in r.getMessage() or "birch" in r.getMessage())
                   for r in caplog.records)
    finally:
        await team.close()


async def test_group_mentions_wake_agents_only_when_a_human_posts_them(tmp_path, board):
    team = Team(tmp_path, board, ["cedar", "birch", "pine"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        out, error = await team.call("cedar", "post", text="@all status update", thread="new")
        assert not error, out
        out, error = await team.call("cedar", "post", text="@pine direct question", thread="new")
        assert not error, out
        await until(lambda: custom(team.agents["pine"], "comms_mention"))
        await asyncio.sleep(0.2)  # several rotations: the @all post was seen too
        assert [m["msg_id"] for m in custom(team.agents["pine"], "comms_mention")] == [out["posted"]]
        assert not custom(team.agents["birch"], "comms_mention")
        board.token_for("operator", "user")
        human = board.post("operator", team.spec.channel_id, "@all please wrap up")
        await until(lambda: all(any(m["msg_id"] == human["id"] for m in custom(team.agents[n], "comms_mention"))
                                for n in ("cedar", "birch", "pine")))
    finally:
        await team.close()


async def test_retried_channel_create_does_not_use_another_slot(tmp_path, board, monkeypatch):
    """The real board answers a repeated create by its creator with the existing channel."""
    from flock.comms.board import AsyncBoard

    real, made = AsyncBoard.request, {}

    async def request(self, route, cred, *a, **kw):
        if route[:2] == ("POST", "/v1/channels"):
            name = route[2]["json"]["name"]
            if name not in made:
                made[name] = await real(self, route, cred, *a, **kw)
            return made[name]
        return await real(self, route, cred, *a, **kw)

    monkeypatch.setattr(AsyncBoard, "request", request)
    team = Team(tmp_path, board, ["cedar"])
    team.spec.extended = True
    team.spec.rules = CommsRules(max_channels_per_agent=2)
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        first, error = await team.call("cedar", "create_channel", name="memory", purpose="work")
        assert not error, first
        again, error = await team.call("cedar", "create_channel", name="memory", purpose="work")
        assert not error and again["id"] == first["id"], again
        assert team.agents["cedar"].state.comms_created_channels == [first["id"]]
        _, error = await team.call("cedar", "create_channel", name="other", purpose="work")
        assert not error
    finally:
        await team.close()
