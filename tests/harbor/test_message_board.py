import json
import os
import shlex
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor

import pytest
from msgboard.client import BoardError, Client

from legion.harbor.config import BUILTIN
from legion.harbor.instructions import build_agent_instructions
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
    assert board_tools.read(unread=True, compact=True)["messages"] == [
        f"#{first['id']} agent-001: {first['body']}"]
    assert board_tools.read(unread=True)["messages"] == []  # its place is kept
    mine = board_tools.post("I get 7", reply_to=first["id"])
    assert board_tools.wait(timeout=0, unread=True, compact=True)["messages"] == [
        f"#{mine['id']} agent-002 (you) (reply to #{first['id']}): I get 7"]
    assert len(board_tools.read()["messages"]) == 2  # the default read is unchanged


async def test_failed_compact_unread_call_leaves_its_page_unread(board, tmp_path, monkeypatch):
    monkeypatch.setattr(board_tools, "UNREAD_DIR", tmp_path / "unread")
    a, b = board.provision(["agent-001", "agent-002"])
    monkeypatch.setenv("MSGBOARD_TOKEN", a["token"])
    posted = [board_tools.post(f"m{i}")["id"] for i in range(3)]
    monkeypatch.setenv("MSGBOARD_TOKEN", b["token"])

    def me_fails(self):  # the board answers /me with an error after the page was fetched
        raise BoardError(500, "internal", "boom")

    for call in (lambda: board_tools.read(unread=True, compact=True),
                 lambda: board_tools.wait(timeout=0, unread=True, compact=True)):
        with monkeypatch.context() as broken:
            broken.setattr(Client, "me", me_fails)
            with pytest.raises(BoardError):
                call()
        shown = call()["messages"]
        assert [int(line.split()[0][1:]) for line in shown] == posted
        for place in board_tools.UNREAD_DIR.glob("*.seq"):  # unread again for the next call
            place.unlink()


async def test_concurrent_unread_reads_show_each_message_once(board, tmp_path, monkeypatch):
    monkeypatch.setattr(board_tools, "UNREAD_DIR", tmp_path / "unread")
    a, b = board.provision(["agent-001", "agent-002"])
    monkeypatch.setenv("MSGBOARD_TOKEN", a["token"])
    posted = [board_tools.post(f"m{i}")["id"] for i in range(24)]
    monkeypatch.setenv("MSGBOARD_TOKEN", b["token"])

    def drain():
        ids = []
        while messages := board_tools.read(unread=True, limit=1)["messages"]:
            ids += [m["id"] for m in messages]
        return ids

    with ThreadPoolExecutor(6) as pool:
        shown = [i for ids in pool.map(lambda _: drain(), range(6)) for i in ids]
    assert sorted(shown) == posted  # none twice, none skipped
    assert not list(board_tools.UNREAD_DIR.glob("*.tmp"))


async def test_compact_read_keeps_pagination(board, tmp_path, monkeypatch):
    a = board.provision(["agent-001"])[0]
    monkeypatch.setenv("MSGBOARD_TOKEN", a["token"])
    board_tools.create_channel("other")
    board_tools.post("elsewhere", "other")  # message ids and general's seqs now differ
    first, second = board_tools.post("one")["id"], board_tools.post("two")["id"]
    page = board_tools.read(limit=1, compact=True)
    full = board_tools.read(limit=1)
    assert (page["next_cursor"], page["has_more"]) == (full["next_cursor"], True)
    assert page["messages"][0].startswith(f"#{first} ")
    after = board_tools.read(after=page["next_cursor"], limit=1, compact=True)
    assert after["messages"][0].startswith(f"#{second} ") and not after["has_more"]


async def test_instruction_examples_run_as_written(board, tmp_path):
    directory = extract_bundle(tmp_path, ["message_board.*"])
    manifest = json.loads((directory / "manifest.json").read_text())
    text = build_agent_instructions("agent-001", {"agent-001": None}, manifest)
    example = next(line for line in text.splitlines()
                   if "Example: `legion-tool message_board.post " in line)
    command = shlex.split(example.split("`")[1])
    identity = board.provision(["agent-001"])[0]
    env = {"MSGBOARD_URL": board.url, "MSGBOARD_TOKEN": identity["token"]}
    result = invoke_bundle(directory, command[1], json.loads(command[3]), env)
    assert result.returncode == 0, result.stdout + result.stderr


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
