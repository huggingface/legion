"""flock's additions to the board resource against tests/fake_msgboard.py: the kickoff, the
activity poller's per-agent events, and the rule proxy (rules, routes, headers, size)."""

import asyncio
import socket
from pathlib import Path

import httpx
import pytest

pytest.importorskip("harbor")

from msgboard.client import BoardError, Client

import flock.harbor
from flock.harbor.tools.runner import load_module

from .. import fake_msgboard

TOOLS_DIR = Path(flock.harbor.__file__).with_name("plugins") / "tools"
board_tools = load_module(TOOLS_DIR / "board.py")
BoardResource = load_module(TOOLS_DIR.parent / "resources" / "board.py").Resource


@pytest.fixture
def fake(monkeypatch):
    board, server, board.url = fake_msgboard.serve()
    board.operator = board.token_for("operator", "user")
    monkeypatch.setenv("MSGBOARD_OPERATOR_TOKEN", board.operator)
    yield board
    server.shutdown()


async def started(fake, state_dir, **options):
    resource = BoardResource({"url": fake.url, "channels": ["flock-test"], **options},
                             state_dir=state_dir, agents=[], secrets=[])
    await resource.start()
    return resource


async def until(check):
    for _ in range(500):
        if value := check():
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("timed out")


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
    resource = await started(fake, tmp_path, rules={"max_len": 20, "min_post_gap": "1h"})
    try:
        await resource.add(["x-0"])
        bound = resource.bind("x-0")
        env, (mount,) = bound["env"], bound["mounts"]
        assert env["MSGBOARD_TOKEN"] == "via-proxy" and bound["allowed_hosts"] == []
        assert mount["target"] == env["MSGBOARD_SOCKET"] == "/run/flock-board.sock"
        for key, value in {**env, "MSGBOARD_SOCKET": mount["source"]}.items():
            monkeypatch.setenv(key, value)
        with pytest.raises(BoardError, match="refused: text is 21 chars; max_len is 20"):
            board_tools.post("x" * 21, "flock-test")
        assert fake.msgs == []
        assert board_tools.post("short", "flock-test")["sender"]["external_id"] == "x-0"
        with pytest.raises(BoardError, match="refused: wait"):
            board_tools.post("again", "flock-test")
        assert len(fake.msgs) == 1

        spoofed = {"Authorization": f"Bearer {fake.operator}", "X-Msgboard-Token": fake.operator}
        transport = httpx.HTTPTransport(uds=mount["source"])
        with httpx.Client(base_url="http://board", transport=transport) as client:
            assert client.get("/v1/me", headers=spoofed).json()["external_id"] == "x-0"
            assert client.post("/v1/invites", json={"kind": "agent"}).status_code == 403

        def status(request):
            with socket.socket(socket.AF_UNIX) as sock:
                sock.connect(mount["source"])
                sock.sendall(request)
                return int(sock.recv(64).split()[1])

        assert status(b"POST /v1/channels/12/../../invites HTTP/1.1\r\n"
                      b"Content-Length: 2\r\n\r\n{}") == 403
        assert status(b"POST /v1/channels/12/messages HTTP/1.1\r\n"
                      b"Content-Length: 999999999999\r\n\r\n") == 413
    finally:
        await resource.stop()
    assert not Path(mount["source"]).exists()
