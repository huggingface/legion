"""Multi-channel and private DM delivery through one rotating worker poller."""
import asyncio
import itertools

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


def test_poll_backoff_is_full_jitter_and_honours_retry_after():
    from flock.worker.comms_extended import POLL_BACKOFF_MAX_S, poll_backoff

    top = lambda a, b: b
    assert [poll_backoff(n, 1.0, rng=top) for n in (1, 2, 3, 7, 8, 40)] == \
        [2.0, 4.0, 8.0, 120.0, 120.0, POLL_BACKOFF_MAX_S]
    assert poll_backoff(3, 1.0, rng=lambda a, b: a) == 0.0  # full jitter reaches zero
    assert poll_backoff(1, 1.0, retry_after=45.0, rng=top) == 45.0  # never shorter


async def test_rate_limited_poller_backs_off_for_the_whole_worker_then_resets(tmp_path, board,
                                                                               monkeypatch):
    """429s space the single poller by Retry-After (all identities wait), and a success
    resets the backoff."""
    from flock.comms.board import AsyncBoard, BoardError
    from flock.worker import comms_extended

    real, calls, delays = AsyncBoard.activity, [], []

    async def limited(self, cred, channel, after, wait=0):
        if after is not None:
            calls.append(cred)
            if len(calls) <= 2:
                raise BoardError("board: GET /v1/activity: HTTP 429", 429, "rate_limited", 0.05)
        return await real(self, cred, channel, after, wait)

    def spy(failures, base, retry_after=None, rng=None):
        delays.append((failures, retry_after))
        return retry_after or 0.0

    monkeypatch.setattr(AsyncBoard, "activity", limited)
    monkeypatch.setattr(comms_extended, "poll_backoff", spy)
    team = Team(tmp_path, board, ["cedar", "birch"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.02
    team.spec.extended = True
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        await until(lambda: len(calls) >= 3)
        assert delays == [(1, 0.05), (2, 0.05)]
        await until(lambda: len(calls) >= 6)
        assert len(delays) == 2  # reset after the success: no further backoff
    finally:
        await team.close()


async def test_poll_backoff_only_slows_the_rotation(monkeypatch):
    """Virtual time: a 429 with a short backoff keeps the normal spacing; one with a long
    Retry-After delays the next poll by at least that much."""
    from types import SimpleNamespace

    import flock.worker.comms_extended as module
    from flock.comms.board import BoardError
    from flock.models import CommsSpec
    from flock.worker.comms import CommsHub, Member
    from flock.worker.identity import MsgboardIdentity

    now, starts = 0.0, []
    hub = CommsHub(swarm="run4")
    hub.configure(CommsSpec(url="http://board", invite_env="INVITE", channel="general",
                            channel_id=12, extended=True), ["comms:read"])
    for i in range(4):
        hub.members[f"a{i}"] = Member(f"a{i}", ident=MsgboardIdentity("http://board", "t", f"a{i}", i),
                                      cursor=0)

    async def sleep(delay):
        nonlocal now
        now += delay

    async def activity(*args, **kwargs):
        starts.append(now)
        if len(starts) == 2:
            raise BoardError("HTTP 429", 429, "rate_limited", retry_after=1.0)
        if len(starts) == 3:
            raise BoardError("HTTP 429", 429, "rate_limited", retry_after=400.0)
        if len(starts) == 6:
            hub.entries = []
        return {"messages": [], "next_cursor": 1}

    monkeypatch.setattr(module, "monotonic", lambda: now, raising=False)
    monkeypatch.setattr(asyncio, "sleep", sleep)
    monkeypatch.setattr(module.random, "uniform", lambda a, b: 0.0)
    monkeypatch.setattr(hub, "_board", lambda _: SimpleNamespace(activity=activity))
    await hub._poll_extended()
    gaps = [b - a for a, b in itertools.pairwise(starts)]
    assert gaps[0] == 75.0 and gaps[1] == 75.0  # 300 s / 4; the 1 s Retry-After changes nothing
    assert gaps[2] == 400.0  # the long Retry-After wins over the spacing
    assert gaps[3] == 75.0  # and the next success is back to normal
