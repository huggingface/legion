"""Space mode of the board resource, against a fake msgboard.space on a real local board."""

import asyncio
import json
import stat

import httpx
import pytest
from huggingface_hub.errors import HFValidationError
from msgboard.client import BoardError, Client
from msgboard.space import DeployError

HF_TOKEN = "hf_test-launcher-secret"


class FakeTime:
    """Stands in for the board module's `time`: only sleeping moves the clock."""

    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


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
    answers = ["up", "down", "down", "up"] + ["down"] * 100  # a restart, then gone
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
    with pytest.raises(RuntimeError, match="unreachable for 180 s"):
        await asyncio.wait_for(resource.wait(), timeout=30)
    # The 10 s outage passed; the run failed at the first check more than 180 s
    # after the last good one (at 20 s).
    assert checks[-1] == 205


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
    assert 20 <= clock.now <= 21  # agent-001 was tried until the 20 s deadline
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
