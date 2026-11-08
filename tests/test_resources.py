import asyncio
import json

import httpx
import pytest
from msgboard.client import Client

from flock.resources import Resources


@pytest.fixture
async def server(message_board_provider, tmp_path):
    resource = message_board_provider(
        {}, state_dir=tmp_path / "server", agents=[], secrets=[]
    )
    try:
        await resource.start()
        yield resource
    finally:
        await resource.stop()


async def test_external_board_scoped_export_and_memberships(
    server, message_board_provider, tmp_path, monkeypatch
):
    unrelated = server.client.create_channel("unrelated")
    server.client.post(unrelated["id"], "Keep this data on the existing board")
    monkeypatch.setenv("BOARD_ADMIN", server.operator_token)
    resource = message_board_provider(
        {
            "url": server.url,
            "token_env": "BOARD_ADMIN",
            "channels": ["general", "findings"],
        },
        state_dir=tmp_path / "external",
        agents=["agent-001", "agent-002"],
        secrets=[],
    )
    try:
        await resource.start()
        assert resource.process is None and resource.socket_path is None
        a, b = [resource.bind(name) for name in resource.agents]
        assert a["env"]["MSGBOARD_TOKEN"] != b["env"]["MSGBOARD_TOKEN"]
        for binding in (a, b):
            assert binding["mounts"] == []
            assert "MSGBOARD_SOCKET" not in binding["env"]
            assert server.operator_token not in binding["env"].values()
            with Client(server.url, binding["env"]["MSGBOARD_TOKEN"]) as client:
                assert {c["name"] for c in client.list_channels() if c["member"]} == {
                    "general",
                    "findings",
                }
        with Client(server.url, a["env"]["MSGBOARD_TOKEN"]) as client:
            private = client.create_channel("private-findings", private=True)
            client.post(private["id"], "Agent-created private data")
        output = tmp_path / "export"
        output.mkdir()
        await resource.export(output)
        channels = json.loads((output / "channels.json").read_text())
        assert {c["name"] for c in channels} == {
            "general",
            "findings",
            "private-findings",
        }
        assert "Agent-created private data" in (output / "board.jsonl").read_text()
    finally:
        await resource.stop()
    assert server.process.returncode is None
    assert (
        server.client.read(unrelated["id"])["messages"][0]["body"]
        == "Keep this data on the existing board"
    )


async def test_external_name_conflict_does_not_rotate_existing_token(
    server, message_board_provider, tmp_path, monkeypatch
):
    original = server.provision(["agent-001"])[0]
    monkeypatch.setenv("BOARD_ADMIN", server.operator_token)
    resource = message_board_provider(
        {"url": server.url, "token_env": "BOARD_ADMIN", "channels": ["do-not-create"]},
        state_dir=tmp_path / "external",
        agents=["agent-001"],
        secrets=[],
    )
    try:
        with pytest.raises(ValueError, match="already has participant identities"):
            await resource.start()
        with Client(server.url, original["token"]) as client:
            assert client.me()["external_id"] == "agent-001"
        assert "do-not-create" not in {c["name"] for c in server.client.list_channels()}
    finally:
        await resource.stop()


async def test_external_member_token_rejected_before_provisioning(
    server, message_board_provider, tmp_path, monkeypatch
):
    member = server.provision(["member"])[0]
    monkeypatch.setenv("BOARD_ADMIN", member["token"])
    resource = message_board_provider(
        {"url": server.url, "token_env": "BOARD_ADMIN"},
        state_dir=tmp_path / "external",
        agents=["new-agent"],
        secrets=[],
    )
    try:
        with pytest.raises(ValueError, match="operator/admin token"):
            await resource.start()
        assert "new-agent" not in {p["external_id"] for p in server.client.people()}
    finally:
        await resource.stop()


@pytest.mark.parametrize(
    "options",
    [
        {"url": "https://user:secret@example.com"},
        {"url": "https://example.com?token=secret"},
        {"url": "file:///tmp/board"},
        {"channels": []},
        {"channels": ["general", "general"]},
        {"token": "secret"},
    ],
)
def test_board_options_reject_invalid_config_without_echoing_values(
    message_board_provider, tmp_path, options
):
    with pytest.raises(ValueError) as exc:
        message_board_provider(options, state_dir=tmp_path, agents=[], secrets=[])
    assert "secret" not in str(exc.value)


def test_missing_external_credential_fails_during_preflight(
    message_board_provider, tmp_path, monkeypatch
):
    monkeypatch.delenv("ABSENT_BOARD_TOKEN", raising=False)
    with pytest.raises(ValueError, match="Set ABSENT_BOARD_TOKEN"):
        message_board_provider(
            {"url": "https://example.com", "token_env": "ABSENT_BOARD_TOKEN"},
            state_dir=tmp_path,
            agents=[],
            secrets=[],
        )


async def test_generic_resources_cleanup_after_partial_startup(tmp_path):
    module = """
class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        self.state_dir = state_dir
        self.fail = options.get("fail", False)
        self.metadata = {"provider": "sample"}
    async def start(self):
        self.state_dir.mkdir(parents=True)
        if self.fail:
            raise RuntimeError("startup failed")
    async def wait(self):
        import asyncio
        await asyncio.Future()
    async def export(self, directory):
        (directory / "result.txt").write_text("result")
        return {"result": "result.txt"}
    async def stop(self):
        (self.state_dir / "stopped").touch()
"""
    for name in ("first", "second"):
        (tmp_path / f"{name}.py").write_text(module)
    resources = Resources(
        tmp_path,
        {"first": {}, "second": {"fail": True}},
        run_dir=tmp_path / "run",
        agents=[],
        secrets=[],
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        await resources.start()
    assert await resources.finish() is False
    for name in ("first", "second"):
        assert (tmp_path / "run/state/resources" / name / "stopped").exists()
        assert (
            resources.records[name]["artifacts"]["result"]
            == f"resources/{name}/result.txt"
        )


async def test_external_health_monitor_detects_board_loss(
    server, message_board_provider, tmp_path, monkeypatch
):
    monkeypatch.setenv("BOARD_ADMIN", server.operator_token)
    resource = message_board_provider(
        {"url": server.url, "token_env": "BOARD_ADMIN"},
        state_dir=tmp_path / "external",
        agents=[],
        secrets=[],
    )
    try:
        await resource.start()
        await server.stop()
        with pytest.raises(httpx.TransportError):
            await asyncio.wait_for(resource.wait(), timeout=8)
    finally:
        await resource.stop()
