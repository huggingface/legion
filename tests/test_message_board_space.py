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
        assert space["admin_invite_created"] and space["msgboard_revision"] == "r" * 40
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
        f"Could not deploy the board to Space user/board: {type(error).__name__}: "
    )
    assert (tmp_path / "state/operator.token").exists()


async def test_space_outage_fails_the_run_only_after_the_grace_period(
    fake_space, message_board_provider, message_board_module, tmp_path, monkeypatch
):
    resource = message_board_provider(
        {"space_id": "board"}, state_dir=tmp_path / "state", agents=[], secrets=[]
    )
    try:
        await resource.start()
        monkeypatch.setattr(message_board_module, "HEALTH_CHECK_SECONDS", 0.2)
        monkeypatch.setattr(message_board_module, "SPACE_OUTAGE_GRACE_SECONDS", 4)
        watcher = asyncio.create_task(resource.wait())
        fake_space.stop()  # a restart: down for a moment, then back
        await asyncio.sleep(0.5)
        fake_space.start()
        await asyncio.sleep(5)
        assert not watcher.done()
        fake_space.stop()
        with pytest.raises(RuntimeError, match="unreachable for 4 s"):
            await asyncio.wait_for(watcher, timeout=10)
    finally:
        await resource.stop()
