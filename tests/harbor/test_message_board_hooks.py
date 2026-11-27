"""legion's additions to the board resource against tests/fake_msgboard.py (a board under a
path, as the multi-board Space serves them): the kickoff, the activity poller's per-agent
events, and the rule proxy (rules, channel caps, routes, headers, size, calls in flight)."""

import asyncio
import select
import socket
import time
from pathlib import Path

import httpx
import pytest
from msgboard.client import BoardError, Client

import legion.harbor
from legion.harbor.tools.runner import load_module

from .. import fake_msgboard

TOOLS_DIR = Path(legion.harbor.__file__).with_name("builtin") / "tools"
board_tools = load_module(TOOLS_DIR / "message_board.py")
BoardResource = load_module(TOOLS_DIR.parent / "resources" / "message_board.py").Resource


@pytest.fixture
def fake(monkeypatch):
    board, server, board.url = fake_msgboard.serve(prefix="/boards/x")
    board.operator = board.token_for("operator", "user")
    monkeypatch.setenv("MSGBOARD_OPERATOR_TOKEN", board.operator)
    yield board
    server.shutdown()


async def started(fake, state_dir, **options):
    resource = BoardResource({"url": fake.url, "channels": ["legion-test"], **options},
                             state_dir=state_dir, agents=[], secrets=[])
    await resource.start()
    return resource


async def until(check):
    for _ in range(500):
        if value := check():
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


def status(path, request):
    with socket.socket(socket.AF_UNIX) as sock:
        sock.connect(path)
        sock.sendall(request)
        return int(sock.recv(64).split()[1])


def calls_in_flight_are_capped(path, then, request):
    """Nine idle connections to a proxy socket: one is refused (429) at once, the other eight
    hold its slots while `then()` runs; once they close, `request` goes through again."""
    socks = [socket.socket(socket.AF_UNIX) for _ in range(9)]
    try:
        for sock in socks:
            sock.connect(path)
        (refused,), _, _ = select.select(socks, [], [], 5)
        assert refused.recv(64).startswith(b"HTTP/1.0 429")
        then()
        assert select.select(socks, [], [], 0.05)[0] == [refused]  # the rest still wait
    finally:
        for sock in socks:
            sock.close()
    for _ in range(100):  # their handlers see the close and give the slots back
        if (code := status(path, request)) != 429:
            return code
        time.sleep(0.01)
    raise AssertionError("slots not given back")


def long_polls(fake):
    return sum(p == "/v1/activity" and "after" in q for _, p, q, _ in fake.requests)


async def test_kickoff_once_and_the_poller_wakes_only_the_agents_a_message_is_for(fake, tmp_path):
    one = await started(fake, tmp_path / "one", kickoff="Hello team")
    two = await started(fake, tmp_path / "two", kickoff="Hello team", wake=["mention", "channel"])
    events = []
    watchers = [asyncio.create_task(r.watch(lambda *e, w=w: events.append((w, *e))))
                for w, r in (("one", one), ("two", two))]
    try:
        assert [m["body"] for m in fake.msgs] == ["Hello team"]  # both workers, one post
        await one.add(["x-0", "x-1"])
        await two.add(["x-2"])
        await until(lambda: long_polls(fake) >= 2)
        root = fake.post("carol", 12, "@x-0 @all who takes the parser?")
        with Client(fake.url, one.bind("x-1")["env"]["MSGBOARD_TOKEN"]) as x1:
            x1.post(12, "I can", reply_to=root["id"])
        reply = fake.post("carol", 12, "thanks", reply_to=root["id"])
        await until(lambda: len(events) >= 3)
        await asyncio.sleep(0.05)  # and nothing more
        seen = {"msg_id": root["id"], "thread": root["id"], "from": "carol",
                "identity_type": "user"}
        assert sorted(events) == [
            ("one", "x-0", "mention", seen),  # @all alone does not wake x-1
            ("one", "x-1", "thread", {**seen, "msg_id": reply["id"]}),  # in a thread it posted in
            ("two", "x-2", "channel", seen),  # its worker wakes on new topics, not mentions
        ]
    finally:
        for w in watchers:
            w.cancel()
        await one.stop()
        await two.stop()


async def test_rule_proxy_checks_posts_and_opens_only_the_board_tools_routes(
    fake, tmp_path, monkeypatch
):
    rules = {"max_len": 20, "min_post_gap": "1h", "max_channels_per_agent": 1, "max_channels": 2}
    resource = await started(fake, tmp_path, rules=rules)
    try:
        await resource.add(["x-0", "x-1"])
        bound = resource.bind("x-0")
        env, (mount,) = bound["env"], bound["mounts"]
        assert env["MSGBOARD_TOKEN"] == "via-proxy" and bound["allowed_hosts"] == []
        assert env["MSGBOARD_URL"] == "http://board"  # the board's path is the proxy's to add
        assert mount["target"] == env["MSGBOARD_SOCKET"] == "/run/legion-board.sock"
        for key, value in {**env, "MSGBOARD_SOCKET": mount["source"]}.items():
            monkeypatch.setenv(key, value)
        with pytest.raises(BoardError, match="refused: text is 21 chars; max_len is 20"):
            board_tools.post("x" * 21, "legion-test")
        assert fake.msgs == []
        assert board_tools.post("short", "legion-test")["sender"]["external_id"] == "x-0"
        with pytest.raises(BoardError, match="refused: wait"):
            board_tools.post("again", "legion-test")
        assert len(fake.msgs) == 1
        board_tools.create_channel("plans")  # x-0's one channel: 2 on the board now
        with pytest.raises(BoardError, match="refused: you reached your channel creation limit"):
            board_tools.create_channel("more")

        other = httpx.HTTPTransport(uds=resource.bind("x-1")["mounts"][0]["source"])
        with httpx.Client(base_url="http://board", transport=other) as x1:
            refused = x1.post("/v1/channels", json={"name": "third"}).json()["error"]["message"]
            assert refused.startswith("refused: the channel limit is reached")
            # one agent's calls in flight never hold up another's
            assert calls_in_flight_are_capped(
                mount["source"], lambda: x1.get("/v1/me").raise_for_status(),
                b"GET /v1/me HTTP/1.1\r\n\r\n") == 200
        assert sorted(fake.channels.values()) == ["legion-test", "plans"]

        spoofed = {"Authorization": f"Bearer {fake.operator}", "X-Msgboard-Token": fake.operator}
        transport = httpx.HTTPTransport(uds=mount["source"])
        with httpx.Client(base_url="http://board", transport=transport) as client:
            assert client.get("/v1/me", headers=spoofed).json()["external_id"] == "x-0"
            assert client.post("/v1/invites", json={"kind": "agent"}).status_code == 403

        assert status(mount["source"], b"POST /v1/channels/12/../../invites HTTP/1.1\r\n"
                      b"Content-Length: 2\r\n\r\n{}") == 403
        assert status(mount["source"], b"POST /v1/channels/12/messages HTTP/1.1\r\n"
                      b"Content-Length: 999999999999\r\n\r\n") == 413
    finally:
        await resource.stop()
    assert not Path(mount["source"]).exists()


async def test_a_post_refused_by_a_short_min_post_gap_is_retried_once(fake, tmp_path, monkeypatch):
    resource = await started(fake, tmp_path, rules={"min_post_gap": "1s"})
    try:
        await resource.add(["x-0"])
        bound = resource.bind("x-0")
        for key, value in {**bound["env"], "MSGBOARD_SOCKET": bound["mounts"][0]["source"]}.items():
            monkeypatch.setenv(key, value)
        board_tools.post("first", "legion-test")
        board_tools.post("second", "legion-test")  # refused ("wait 1 s"), waited out, posted
        assert [m["body"] for m in fake.msgs] == ["first", "second"]
    finally:
        await resource.stop()


async def test_a_url_board_fails_the_run_only_after_the_outage_grace(tmp_path, monkeypatch):
    """`url` (a board that is up) is watched as the upstream Space board: health checks every
    HEALTH_CHECK_SECONDS, an error only once it has been unreachable for the grace."""
    board, server, board.url = fake_msgboard.serve(prefix="/boards/x")
    operator = board.token_for("operator", "user")
    monkeypatch.setenv("MSGBOARD_OPERATOR_TOKEN", operator)
    resource = await started(board, tmp_path / "state")
    globals_ = type(resource).wait.__globals__
    monkeypatch.setitem(globals_, "HEALTH_CHECK_SECONDS", 0.05)
    monkeypatch.setitem(globals_, "SPACE_OUTAGE_GRACE_SECONDS", 0.5)
    try:
        watching = asyncio.create_task(resource.wait())
        await asyncio.sleep(0.2)
        assert not watching.done()  # healthy
        del board.tokens[operator]  # every health check fails now (401)
        with pytest.raises(RuntimeError, match="unreachable for 0.5 s"):
            await asyncio.wait_for(watching, timeout=5)
    finally:
        await resource.stop()
        server.shutdown()
