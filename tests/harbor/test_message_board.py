import json
import os
import subprocess
import sys
import tarfile

import pytest
from msgboard.client import BoardError, Client

from legion.harbor.config import BUILTIN
from legion.harbor.tools.bundle import build_tool_bundle
from legion.harbor.tools.runner import load_module

TOOLS_DIR = BUILTIN / "tools"
board_tools = load_module(TOOLS_DIR / "message_board.py")
BoardResource = load_module(BUILTIN / "resources" / "message_board.py").Resource


def BoardRuntime(state_dir):
    return BoardResource({}, state_dir=state_dir, agents=[], secrets=[])


@pytest.fixture
async def board(tmp_path, monkeypatch):
    runtime = BoardRuntime(tmp_path / "state")
    try:
        await runtime.start()
        monkeypatch.setenv("MSGBOARD_URL", runtime.url)
        monkeypatch.delenv("MSGBOARD_SOCKET", raising=False)
        yield runtime
    finally:
        await runtime.stop()


async def test_agent_created_channels_permissions_retry_and_export(
    board, tmp_path, monkeypatch
):
    a, b = board.provision(["agent-001", "agent-002"])
    monkeypatch.setenv("MSGBOARD_TOKEN", a["token"])
    created = board_tools.create_channel("findings", private=True)
    channel_id = created.get("channel", created)["id"]
    with Client(board.url, b["token"]) as peer:
        with pytest.raises(BoardError):
            peer.channel_id("findings")
    board_tools.invite("findings", "agent-002")
    first = board_tools.post("answer=42", "findings", request_id="logical-post")
    again = board_tools.post("answer=42", "findings", request_id="logical-post")
    assert first == again
    assert len(board_tools.agents()["agents"]) == 2
    monkeypatch.setenv("MSGBOARD_TOKEN", b["token"])
    assert len(board_tools.read("findings")["messages"]) == 1
    with Client(board.url, b["token"]) as peer:
        assert peer.channel_id("findings") == channel_id
    await board.export(tmp_path)
    exported = [
        json.loads(line) for line in (tmp_path / "board.jsonl").read_text().splitlines()
    ]
    assert any(row["message"]["body"] == "answer=42" for row in exported)
    serialized = (tmp_path / "channels.json").read_text() + (
        tmp_path / "board.jsonl"
    ).read_text()
    assert all(secret not in serialized for secret in board.secrets)


async def test_post_from_a_file_and_unread_compact_reads(board, tmp_path, monkeypatch):
    monkeypatch.setattr(board_tools, "UNREAD_DIR", tmp_path / "unread")
    a, b = board.provision(["agent-001", "agent-002"])
    monkeypatch.setenv("MSGBOARD_TOKEN", a["token"])
    message = tmp_path / "msg.txt"
    message.write_text("x = $a'b$ and \\frac{1}{2}, \"quoted\"\n")
    first = board_tools.post(path=str(message))
    assert first["body"] == "x = $a'b$ and \\frac{1}{2}, \"quoted\""
    with pytest.raises(ValueError, match="exactly one of text or path"):
        board_tools.post("both", path=str(message))

    monkeypatch.setenv("MSGBOARD_TOKEN", b["token"])
    assert board_tools.read(unread=True, compact=True) == {
        "messages": [f"#{first['id']} agent-001: {first['body']}"]}
    assert board_tools.read(unread=True)["messages"] == []  # its place is kept
    mine = board_tools.post("I get 7", reply_to=first["id"])
    assert board_tools.wait(timeout=0, unread=True, compact=True) == {
        "messages": [f"#{mine['id']} agent-002 (you) (reply to #{first['id']}): I get 7"]}
    assert len(board_tools.read()["messages"]) == 2  # the default read is unchanged


def extract_bundle(tmp_path, names):
    bundle = build_tool_bundle(TOOLS_DIR, names, tmp_path / "tools.tar.gz")
    directory = tmp_path / "bundle"
    with tarfile.open(bundle.archive) as archive:
        archive.extractall(directory, filter="data")
    return directory


def invoke_bundle(directory, name, arguments, env):
    return subprocess.run(
        [
            sys.executable,
            "-S",
            str(directory / "bin/legion-tool"),
            name,
            "--json",
            json.dumps(arguments),
        ],
        env={
            **os.environ,
            **env,
            "LEGION_TOOLS_MANIFEST": str(directory / "manifest.json"),
        },
        capture_output=True,
        text=True,
        timeout=15,
    )


async def test_bundle_runs_without_host_environment_and_rejects_unselected_tools(
    board, tmp_path
):
    directory = extract_bundle(tmp_path, ["message_board.post"])
    identity = board.provision(["bundled-agent"])[0]
    env = {"MSGBOARD_URL": board.url, "MSGBOARD_TOKEN": identity["token"]}
    result = invoke_bundle(
        directory, "message_board.post", {"text": "from bundle"}, env
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "from bundle" in result.stdout
    result = invoke_bundle(
        directory, "message_board.create_channel", {"name": "not-selected"}, env
    )
    assert result.returncode == 1
    assert "not selected" in result.stdout
    assert identity["token"] not in result.stdout + result.stderr


async def test_cli_uses_authenticated_socket_and_removes_it(board, tmp_path):
    runtime = BoardRuntime(tmp_path / "other-state")
    directory = extract_bundle(tmp_path, ["message_board.post"])
    try:
        await runtime.start()
        socket_path = runtime.socket_path
        identity = runtime.provision(["socket-agent"])[0]
        result = invoke_bundle(
            directory,
            "message_board.post",
            {"text": "via socket"},
            {
                "MSGBOARD_URL": "http://unreachable.invalid",
                "MSGBOARD_TOKEN": identity["token"],
                "MSGBOARD_SOCKET": str(socket_path),
            },
        )
        assert result.returncode == 0, result.stdout + result.stderr
        with Client(runtime.url, identity["token"]) as client:
            assert (
                client.read(client.channel_id("general"))["messages"][0]["body"]
                == "via socket"
            )
        same = board.provision(["socket-agent"])[0]  # another board: isolated, same name or not
        with Client(board.url, same["token"]) as client:
            assert client.read(client.channel_id("general"))["messages"] == []
        with Client(board.url, identity["token"]) as client, pytest.raises(BoardError):
            client.me()
    finally:
        await runtime.stop()
    assert not socket_path.exists()
