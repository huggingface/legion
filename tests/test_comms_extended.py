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


async def test_tool_fan_out_is_cached_and_limited_per_agent(tmp_path, board, monkeypatch):
    from flock.comms.board import AsyncBoard

    real, sent = AsyncBoard.request, []

    async def counting(self, route, cred=None, **kw):
        sent.append((route[0], route[1]))
        return await real(self, route, cred, **kw)

    monkeypatch.setattr(AsyncBoard, "request", counting)
    team = Team(tmp_path, board, ["cedar", "birch"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.rules = CommsRules(min_post_gap=None)
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        _, error = await team.call("birch", "create_channel", name="memory", purpose="work")
        assert not error
        sent.clear()
        for _ in range(3):
            out, error = await team.call("cedar", "channels")
            assert not error
        # a channel created after cedar's cached list is still found (one refresh)
        _, error = await team.call("birch", "create_channel", name="later", purpose="work")
        assert not error
        out, error = await team.call("cedar", "read", channel="later")
        assert not error, out
        cedar_lists = [p for p in sent if p == ("GET", "/v1/channels")]
        assert len(cedar_lists) == 2 + 1  # one cached list, birch's create check, one refresh

        sent.clear()
        for text in ("one", "two"):
            out, error = await team.call("cedar", "dm", to="birch", text=text)
            assert not error, out
        assert sent.count(("GET", "/v1/principals")) == 1
        read, error = await team.call("birch", "read_dms")
        assert not error and [m["text"] for m in read["messages"]] == ["one", "two"]
        sent.clear()
        read, error = await team.call("birch", "read_dms")  # nothing new: the list only
        assert not error and len(read["messages"]) == 2
        assert [p for p in sent if p[1] != "/v1/activity"] == [("GET", "/v1/channels")]

        team.hub.agent_rate, team.hub.agent_burst = 0.001, 2
        sent.clear()
        assert not (await team.call("cedar", "channels"))[1]  # (the dm dropped the cached list)
        assert not (await team.call("cedar", "channels"))[1]
        out, error = await team.call("cedar", "channels")
        assert error and "too many board calls from you; retry in" in out["error"]
        assert [p for p in sent if p[1] != "/v1/activity"] == [("GET", "/v1/channels")]  # 1st only
    finally:
        await team.close()


async def gateway(hub, agent, tool, **args):
    """One flock-tool call through the hub's gateway, as the CLI sends it."""
    import json
    host, port = hub.address.rsplit(":", 1)
    reader, writer = await asyncio.open_connection(host, int(port))
    writer.write(json.dumps({"agent": agent, "key": hub.key(agent), "tool": tool,
                             "args": args}).encode() + b"\n")
    await writer.drain()
    r = json.loads(await reader.readline())
    writer.close()
    return json.loads(r["output"]), r["error"]


async def test_board_news_rides_on_the_next_successful_flock_tool_call(tmp_path, board):
    from flock.tools.runner import Result

    team = Team(tmp_path, board, ["cedar", "birch"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    worlds = []

    async def world(agent, entry, args):
        worlds.append(entry)
        if entry == "world:list_prs":
            return Result("[1, 2]", False)
        if entry == "world:fail":
            return Result('{"error": "nope"}', True)
        return Result('{"ok": 1, "inbox": ["PR #3 approved"]}', False)

    team.hub.routes["world"] = world
    try:
        await team.start()
        assert "board_news" in team.hub.section("birch")
        out, error = await team.call("cedar", "dm", to="birch", text="Private  review\nquestion " + "x" * 200)
        assert not error, out
        dm_id = out["posted"]
        await until(lambda: custom(team.agents["birch"], "comms_dm"))
        rt = team.agents["birch"]
        assert rt.state.comms_news and rt.state.comms_news[0]["id"] == dm_id  # persisted state
        # errors and refusals keep the news
        out, error = await gateway(team.hub, "birch", "comms:post", text=3)
        assert error and "board_news" not in out
        out, error = await gateway(team.hub, "birch", "world:fail")
        assert error and "board_news" not in out
        # a world dict result: board_news next to the world's own inbox
        out, error = await gateway(team.hub, "birch", "world:status")
        assert not error and out["inbox"] == ["PR #3 approved"] and out["ok"] == 1
        [line] = out["board_news"]
        assert line.startswith("DM from cedar (channel ") and f"(msg {dm_id}): Private review question x" in line
        assert line.endswith("...") and len(line.split(": ", 1)[1]) == 120
        # only once
        out, error = await gateway(team.hub, "birch", "world:status")
        assert not error and "board_news" not in out
        assert rt.state.comms_news == []
        # a list result is wrapped, only when there is news
        out, error = await gateway(team.hub, "birch", "world:list_prs")
        assert out == [1, 2]
        _, error = await team.call("cedar", "post", text="@birch please look", thread="new")
        assert not error
        await until(lambda: rt.state.comms_news)
        out, error = await gateway(team.hub, "birch", "world:list_prs")
        assert not error and out["result"] == [1, 2]
        assert out["board_news"][0].startswith("cedar mentioned you in #flock-test, thread ")
        # read_dms drops what it returned itself; the other news still comes along
        out, error = await team.call("cedar", "dm", to="birch", text="second dm")
        assert not error
        _, error = await team.call("cedar", "post", text="@birch and this", thread="new")
        assert not error
        await until(lambda: len(rt.state.comms_news) == 2)
        out, error = await gateway(team.hub, "birch", "comms:read_dms")
        assert not error and [m["text"] for m in out["messages"]][-1] == "second dm"
        assert len(out["board_news"]) == 1 and "mentioned you" in out["board_news"][0]
        assert rt.state.comms_news == []
        # the comms result itself is unchanged when there is no news
        out, error = await gateway(team.hub, "birch", "comms:read_dms")
        assert not error and "board_news" not in out and "result" not in out
    finally:
        await team.close()


async def test_tool_cache_is_bounded_per_agent_and_dropped_with_the_agent():
    from types import SimpleNamespace

    from flock.worker import comms_extended
    from flock.worker.comms import CommsHub

    hub = CommsHub(swarm="sw", env={})
    cedar, birch = SimpleNamespace(name="cedar"), SimpleNamespace(name="birch")
    calls = []

    async def fetch():
        calls.append(1)
        return ["page"]

    assert await hub._cached(cedar, "channels", fetch) == (["page"], False)
    assert await hub._cached(cedar, "channels", fetch) == (["page"], True)  # within the TTL
    for i in range(comms_extended.TOOL_CACHE_MAX + 20):  # read_dms pages: one per channel and n
        hub._remember("cedar", ("dm", i, 20), (i, {"messages": []}))
    hub._remember("birch", ("dm", 1, 20), (1, {"messages": []}))
    assert len(hub._tool_cache["cedar"]) == comms_extended.TOOL_CACHE_MAX
    assert hub._recall("cedar", ("dm", comms_extended.TOOL_CACHE_MAX + 19, 20)) is not None
    assert hub._recall("cedar", "channels") is None  # the oldest went first
    assert hub._recall("cedar", ("dm", 0, 20)) is None
    hub._forget(birch, ("dm", 1, 20))
    assert hub._tool_cache["birch"] == {}
    hub.forget("cedar")  # unassigned: nothing of it stays
    assert "cedar" not in hub._tool_cache and len(calls) == 1
    await hub.aclose()
