"""Space mode of the board resource, against a fake msgboard.space on a real local board."""

import asyncio
import json
import shutil
import stat
import threading
from contextlib import contextmanager, nullcontext
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import httpx
import msgboard.client
import pytest
from huggingface_hub.errors import HFValidationError
from msgboard import auth
from msgboard.client import BoardError, Client
from msgboard.space import DeployError

from flock.tools.runner import load_module

HF_TOKEN = "hf_test-launcher-secret"
HTML = {"Content-Type": "text/html"}
# What the board answers to writes while it hands its bucket to a new container.
HANDING_OVER = (
    503,
    {"Content-Type": "application/json", "Retry-After": "2"},
    json.dumps(
        {"error": {"code": "shutting_down", "message": "server is shutting down"}}
    ).encode(),
)


class FakeTime:
    """Stands in for the board module's `time`: only sleeping moves the clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


@contextmanager
def gate(board_url, answer):
    """The Space's gate in front of the board, at the yielded URL. It passes each
    request on, unless answer(method) returns (status, headers, body) instead."""

    class Gate(BaseHTTPRequestHandler):
        def relay(self):
            reply = answer(self.command)
            if reply is None:
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                headers = {
                    name: value
                    for name, value in self.headers.items()
                    if name.lower() not in ("host", "content-length", "connection")
                }
                passed = httpx.request(
                    self.command, board_url + self.path, headers=headers, content=body
                )
                reply = (
                    passed.status_code,
                    {"Content-Type": passed.headers["content-type"]},
                    passed.content,
                )
            status, headers, body = reply
            self.send_response(status)
            for name, value in {**headers, "Content-Length": len(body)}.items():
                self.send_header(name, str(value))
            self.end_headers()
            self.wfile.write(body)

        do_GET = do_POST = do_DELETE = relay

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Gate)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()


def gated(url, token):
    """A client as agents build it for a private Space (Client.from_env)."""
    return Client(
        url,
        token,
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        token_header="X-Msgboard-Token",
    )


async def test_space_board_run_then_relaunch(
    fake_space, message_board_provider, tmp_path, capsys
):
    options = {
        "space_id": "board",
        "private": True,
        "channels": ["general", "findings"],
    }
    first = message_board_provider(
        options, state_dir=tmp_path / "run-1", agents=["agent-001"], secrets=[]
    )
    try:
        await first.start()
        operator_file = tmp_path / "run-1/operator.token"
        assert stat.S_IMODE(operator_file.stat().st_mode) == 0o600
        assert fake_space.deploys == [
            {
                "operator_token": operator_file.read_text().strip(),
                "private": True,
                "hf_token": HF_TOKEN,
            }
        ]
        space = first.metadata["space"]
        assert first.metadata["mode"] == "space"
        assert space["id"] == "user/board" and space["bucket"] == "user/board-bucket"
        assert space["admin_invite_created"]
        output = capsys.readouterr().out
        invite = output.split('"Join with Hugging Face"): ', 1)[1].split()[0]
        assert invite.startswith(fake_space.url + "/?invite=")
        binding = first.bind("agent-001")
        assert binding["env"]["MSGBOARD_HF_TOKEN"] == HF_TOKEN
        assert binding["allowed_hosts"] == ["127.0.0.1"] and binding["mounts"] == []
        agent_token = binding["env"]["MSGBOARD_TOKEN"]
        with gated(fake_space.url, agent_token) as agent:
            agent.post(agent.channel_id("findings"), "from the first run")
            agent_id = agent.me()["id"]
        # A person takes the admin invite, as "Join with Hugging Face" does.
        with Client.join(
            fake_space.url, invite.split("=", 1)[1], name="lewis"
        ) as person:
            person_token = person.token
        await first.export(tmp_path)
    finally:
        await first.stop()
    assert fake_space.process.poll() is None  # the Space keeps running
    with gated(fake_space.url, agent_token) as agent, pytest.raises(BoardError) as exc:
        agent.me()
    assert exc.value.status == 401

    # A relaunch with the same Space ID: same board, new operator token, no new invite.
    second = message_board_provider(
        options, state_dir=tmp_path / "run-2", agents=["agent-001"], secrets=[]
    )
    try:
        await second.start()
        assert len(fake_space.deploys) == 2
        assert fake_space.deploys[1]["operator_token"] != first.operator_token
        assert not second.metadata["space"]["admin_invite_created"]
        # The re-key didn't use up its invite, so it was revoked.
        assert all(i["status"] != "active" for i in second.client.invites())
        assert "invite" not in capsys.readouterr().out
        with (
            gated(fake_space.url, first.operator_token) as old,
            pytest.raises(BoardError),
        ):
            old.me()
        with Client(fake_space.url, person_token) as person:
            assert person.me()["role"] == "admin"
        new_token = second.bind("agent-001")["env"]["MSGBOARD_TOKEN"]
        assert new_token != agent_token
        with gated(fake_space.url, new_token) as agent:
            assert agent.me()["id"] == agent_id
            findings = agent.channel_id("findings")
            assert agent.read(findings)["messages"][0]["body"] == "from the first run"
            agent.post(findings, "from the relaunch")
        await second.export(tmp_path)
    finally:
        await second.stop()
    bodies = [
        json.loads(line)["message"]["body"]
        for line in (tmp_path / "board.jsonl").read_text().splitlines()
    ]
    assert bodies == ["from the first run", "from the relaunch"]


@pytest.mark.parametrize(
    "error",
    [
        DeployError("user/board exists but is not a msgboard board"),
        HFValidationError("Repo id must use alphanumeric chars"),
        httpx.ConnectError("[Errno -3] Temporary failure in name resolution"),
    ],
)
async def test_space_deploy_failures_are_launch_errors(
    fake_space,
    message_board_provider,
    tmp_path,
    monkeypatch,
    message_board_module,
    error,
):
    def refuse(*args, **kwargs):
        raise error

    monkeypatch.setattr(message_board_module, "deploy", refuse)
    resource = message_board_provider(
        {"space_id": "board"}, state_dir=tmp_path / "state", agents=["a"], secrets=[]
    )
    try:
        with pytest.raises(RuntimeError) as exc:
            await resource.start()
    finally:
        await resource.stop()
    assert str(exc.value).startswith(
        f"Could not deploy the board to Space board: {type(error).__name__}: "
    )
    assert (tmp_path / "state/operator.token").exists()


async def test_space_outage_fails_the_run_only_after_the_grace_period(
    message_board_provider, message_board_module, tmp_path, monkeypatch
):
    monkeypatch.setenv("HF_TOKEN", HF_TOKEN)
    clock = FakeTime()
    # A crash that a new container takes over once the old lease is 5 minutes
    # stale, then gone for good.
    answers = ["up"] + ["down"] * 65 + ["up"] + ["down"] * 100
    checks = []

    class Probe:
        """The Space's health as the watcher sees it, one check every 5 s."""

        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def me(self):
            clock.sleep(5)
            checks.append(clock.now)
            if answers[len(checks) - 1] == "down":
                raise httpx.ConnectError("Space restarting")
            return {"role": "admin"}

    monkeypatch.setattr(message_board_module, "time", clock)
    monkeypatch.setattr(message_board_module, "Client", Probe)
    monkeypatch.setattr(message_board_module, "HEALTH_CHECK_SECONDS", 0)
    resource = message_board_provider(
        {"space_id": "board"}, state_dir=tmp_path / "state", agents=[], secrets=[]
    )
    with pytest.raises(RuntimeError, match="unreachable for 420 s"):
        await asyncio.wait_for(resource.wait(), timeout=30)
    # The 330 s outage passed; the run failed at the first check more than 420 s
    # after the last good one (at 335 s).
    assert checks[-1] == 760


async def test_space_revocation_tries_every_token_and_names_only_failures(
    fake_space, message_board_provider, message_board_module, tmp_path, monkeypatch
):
    resource = message_board_provider(
        {"space_id": "board"},
        state_dir=tmp_path / "state",
        agents=["agent-001", "agent-002"],
        secrets=[],
    )
    await resource.start()
    tokens = {name: c["token"] for name, c in resource.credentials.items()}
    resource.credentials["agent-001"]["token"] = "not-a-board-token"  # keeps failing
    clock = FakeTime()
    attempts = []
    revoke = Client.revoke_token

    def restarting(self, token_id):  # the board is restarting at the first attempt
        attempts.append(clock.now)
        if len(attempts) == 1:
            raise httpx.ConnectError("Space restarting")
        return revoke(self, token_id)

    monkeypatch.setattr(message_board_module, "time", clock)
    monkeypatch.setattr(Client, "revoke_token", restarting)
    with pytest.raises(RuntimeError, match="board tokens of agent-001, which may"):
        await resource.stop()
    assert attempts == [0, 1]  # agent-002, retried a second later
    assert clock.now == message_board_module.REVOKE_SECONDS  # agent-001, until then
    with Client(fake_space.url, tokens["agent-002"]) as agent:
        with pytest.raises(BoardError, match="401"):
            agent.me()
    with Client(fake_space.url, tokens["agent-001"]) as agent:
        assert agent.me()["external_id"] == "agent-001"


async def test_space_export_fails_when_the_board_lost_the_runs_channels(
    fake_space, message_board_provider, tmp_path
):
    resource = message_board_provider(
        {"space_id": "board"}, state_dir=tmp_path / "state", agents=[], secrets=[]
    )
    try:
        await resource.start()
        # As after a restart from a snapshot older than the run.
        resource.client.delete_channel(resource.client.channel_id("general"), "general")
        with pytest.raises(RuntimeError, match="no longer has the run's channels"):
            await resource.export(tmp_path)
    finally:
        await resource.stop()


@pytest.mark.parametrize("answer", ["board", "proxy"])
async def test_space_revocation_trusts_only_the_boards_own_404(
    fake_space,
    message_board_provider,
    message_board_module,
    tmp_path,
    monkeypatch,
    answer,
):
    resource = message_board_provider(
        {"space_id": "board"},
        state_dir=tmp_path / "state",
        agents=["agent-001", "agent-002"],
        secrets=[],
    )
    await resource.start()
    tokens = {name: c["token"] for name, c in resource.credentials.items()}
    # A well-formed token the board never issued: the board answers 404 not_found.
    resource.credentials["agent-001"]["token"] = auth.new_token()[1]
    clock = FakeTime()
    monkeypatch.setattr(message_board_module, "time", clock)
    html_404 = (404, HTML, b"<html><body><h1>404</h1></body></html>")
    with (
        gate(fake_space.url, lambda method: html_404)
        if answer == "proxy"
        else nullcontext(fake_space.url)
    ) as url:
        resource.url = url
        if answer == "board":
            await resource.stop()
            assert clock.now == 0  # nothing left to retry
            with Client(fake_space.url, tokens["agent-002"]) as agent:
                with pytest.raises(BoardError, match="401"):
                    agent.me()
        else:
            with pytest.raises(
                RuntimeError,
                match="agent-001, agent-002, which may still work: BoardError: 404 "
                "http_error",
            ):
                await resource.stop()
            assert clock.now == message_board_module.REVOKE_SECONDS
            with Client(fake_space.url, tokens["agent-002"]) as agent:
                assert agent.me()["external_id"] == "agent-002"


@pytest.mark.parametrize("answer", ["hand-over", "paused"])
async def test_space_revocation_waits_out_a_hand_over_but_not_a_paused_space(
    fake_space,
    message_board_provider,
    message_board_module,
    tmp_path,
    monkeypatch,
    answer,
):
    resource = message_board_provider(
        {"space_id": "board"},
        state_dir=tmp_path / "state",
        agents=["agent-001", "agent-002"],
        secrets=[],
    )
    await resource.start()
    tokens = {name: c["token"] for name, c in resource.credentials.items()}
    clock = FakeTime()
    monkeypatch.setattr(message_board_module, "time", clock)
    # Writes are refused for 75 s: a hand-over while bucket uploads are slow, or
    # the gate's own 503 for a paused Space.
    refusal = HANDING_OVER if answer == "hand-over" else (503, HTML, b"paused")
    with gate(
        fake_space.url, lambda method: refusal if clock.now < 75 else None
    ) as url:
        resource.url = url
        if answer == "hand-over":
            await resource.stop()
            assert clock.now == 75
        else:
            with pytest.raises(
                RuntimeError,
                match="agent-001, agent-002, which may still work: BoardError: 503 "
                "http_error",
            ):
                await resource.stop()
            assert clock.now == message_board_module.REVOKE_SECONDS
    for name, token in tokens.items():
        with Client(fake_space.url, token) as agent:
            if answer == "hand-over":
                with pytest.raises(BoardError, match="401"):
                    agent.me()
            else:
                assert agent.me()["external_id"] == name


async def test_agent_post_outlasts_a_slow_hand_over(
    fake_space, message_board_provider, tmp_path, monkeypatch
):
    board_tools = load_module(
        Path(__file__).resolve().parents[1] / "tools/message_board.py"
    )
    resource = message_board_provider(
        {"space_id": "board", "private": True},
        state_dir=tmp_path / "state",
        agents=["agent-001"],
        secrets=[],
    )
    await resource.start()
    try:
        for name, value in resource.bind("agent-001")["env"].items():
            monkeypatch.setenv(name, value)
        clock = FakeTime()
        monkeypatch.setattr(msgboard.client, "time", clock)
        # The shortest backoff the client's jitter allows.
        monkeypatch.setattr(
            msgboard.client, "random", SimpleNamespace(random=lambda: 0.0)
        )
        with gate(
            fake_space.url,
            lambda method: (
                HANDING_OVER if method == "POST" and clock.now < 75 else None
            ),
        ) as url:
            monkeypatch.setenv("MSGBOARD_URL", url)
            board_tools.post("found it")
        assert clock.now > 75
        general = resource.client.channel_id("general")
        messages = resource.client.read(general)["messages"]
        assert [message["body"] for message in messages] == ["found it"]
    finally:
        await resource.stop()


async def test_space_export_fails_after_a_rollback_to_before_the_relaunch(
    fake_space, message_board_provider, tmp_path
):
    from msgboard.space.app import ensure_board

    options = {"space_id": "board"}
    first = message_board_provider(
        options, state_dir=tmp_path / "run-1", agents=["agent-001"], secrets=[]
    )
    try:
        await first.start()
    finally:
        await first.stop()
    data = fake_space.directory / "data"
    fake_space.stop()
    shutil.copytree(data, tmp_path / "snapshot")  # the bucket's snapshot at this point
    fake_space.start()
    second = message_board_provider(
        options, state_dir=tmp_path / "run-2", agents=["agent-001"], secrets=[]
    )
    try:
        await second.start()
        # A restart that restores the older snapshot (msgboard#26). The channels are
        # still there; this run's agent invites are not.
        fake_space.stop()
        shutil.rmtree(data)
        shutil.copytree(tmp_path / "snapshot", data)
        ensure_board(data, second.operator_token)  # the Space app does this on boot
        fake_space.start()
        with Client(fake_space.url, second.operator_token) as board:
            await asyncio.to_thread(board.wait_ready, 60)
        with pytest.raises(
            RuntimeError, match="no longer has this run's agent invites"
        ):
            await second.export(tmp_path)
    finally:
        await second.stop()
