"""Live tests: the board resource on real Hugging Face Spaces.

Skipped unless FLOCK_LIVE_SPACES says where test Spaces go, as <namespace>/<name-prefix>
(for example your-name/flock-test), and a Hugging Face token is available:

    FLOCK_LIVE_SPACES=your-name/flock-test uv run pytest -m live

The tests deploy real CPU Basic Spaces, take several minutes, and delete the Spaces and
their buckets afterwards. Tokens and invite links stay out of the output: what the
resource prints is captured, and assertions compare secrets without showing them.
"""

import asyncio
import json
import os
import re
import stat
import time
import uuid
from pathlib import Path

import pytest
from huggingface_hub import HfApi, get_token
from msgboard.client import BoardError, Client

from flock.resources import Resources, load_module

ROOT = Path(__file__).resolve().parents[1]
board = load_module(ROOT / "resources/message_board.py")
board_tools = load_module(ROOT / "tools/message_board.py")
PREFIX = os.environ.get("FLOCK_LIVE_SPACES")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (PREFIX and get_token()),
        reason="set FLOCK_LIVE_SPACES=<namespace>/<name-prefix> and log in to the Hub",
    ),
]


@pytest.fixture
def new_space():
    """Names for new test Spaces; deletes the Spaces and their buckets afterwards."""
    api = HfApi()
    names = []

    def name(purpose):
        names.append(f"{PREFIX}-{purpose}-{uuid.uuid4().hex[:6]}")
        return names[-1]

    yield name
    for space_id in names:
        api.delete_repo(space_id, repo_type="space", missing_ok=True)
        for bucket in api.list_buckets(namespace=space_id.split("/")[0]):
            if re.fullmatch(re.escape(space_id) + r"-bucket(-\d+)?", bucket.id):
                api.delete_bucket(bucket.id)


def as_agent(monkeypatch, resource, name):
    """Give the bundled tool functions the environment Flock gives this agent."""
    for key, value in resource.bind(name)["env"].items():
        monkeypatch.setenv(key, value)


def refusal(url, token, **gate):
    """The board's status for a token it refuses, or None if it accepts it."""
    with Client(url, token, max_retries=3, **gate) as client:
        try:
            client.me()
        except BoardError as exc:
            return exc.status


def identity_id(credential):
    return credential["identity"].get("principal", credential["identity"])["id"]


def exported(directory):
    lines = (directory / "board.jsonl").read_text().splitlines()
    return sorted(json.loads(line)["message"]["body"] for line in lines)


def post_through_a_restart(space_id, seconds=45):
    """Restart the Space and post through the agent's tool once a second meanwhile.
    Returns the acknowledged posts and the stages the Space went through."""
    api = HfApi()
    api.restart_space(space_id)
    posted, stages = [], []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        stage = api.get_space_runtime(space_id).stage
        if stage not in stages[-1:]:
            stages.append(stage)
        text = f"during the restart {len(posted) + 1}"
        board_tools.post(text)
        posted.append(text)
        time.sleep(1)
    return posted, stages


async def test_a_space_board_through_restart_relaunch_and_outage(
    new_space, tmp_path, capsys, monkeypatch
):
    api = HfApi()
    space_id = new_space("board")
    options = {"space_id": space_id, "private": True}
    agents = ["agent-001", "agent-002"]

    # Run 1 deploys a new private board.
    first = board.Resource(
        options, state_dir=tmp_path / "run-1", agents=agents, secrets=[]
    )
    await first.start()
    invite = re.search(r"\?invite=([\w-]+)", capsys.readouterr().out).group(1)
    assert stat.S_IMODE((tmp_path / "run-1/operator.token").stat().st_mode) == 0o600
    space = first.metadata["space"]
    assert space["id"] == space_id and space["bucket"] == f"{space_id}-bucket"
    assert space["created"] and space["private"] and space["admin_invite_created"]
    assert api.space_info(space_id).private and api.bucket_info(space["bucket"]).private
    gate = first._gate()
    binding = first.bind("agent-001")
    gets_gate_token = binding["env"].get("MSGBOARD_HF_TOKEN") == first.hf_token
    assert gets_gate_token
    assert binding["allowed_hosts"] == [first.url.removeprefix("https://")]
    # Without the HF token, the Space's gate refuses even a valid board token.
    assert refusal(first.url, binding["env"]["MSGBOARD_TOKEN"]) in (401, 404)
    as_agent(monkeypatch, first, "agent-001")
    posted = ["hello from agent-001"]
    board_tools.post(posted[0])

    # A restart while the agent posts and the watcher runs loses nothing.
    watcher = asyncio.create_task(first.wait())
    during, stages = await asyncio.to_thread(post_through_a_restart, space_id)
    posted += during
    assert len(stages) > 1 and stages[-1] == "RUNNING", stages
    assert not watcher.done()
    watcher.cancel()
    await first.export(tmp_path / "run-1")
    assert exported(tmp_path / "run-1") == sorted(posted)

    # Revocation trusts only the board: a gate refusing the HF token says nothing.
    monkeypatch.setattr(board, "REVOKE_SECONDS", 3)
    gate_token, first.hf_token = first.hf_token, "hf_" + "0" * 34
    with pytest.raises(
        RuntimeError,
        match=r"agent-001, agent-002, which may still work: BoardError: 4\d\d",
    ):
        await asyncio.to_thread(first._revoke_agent_tokens)
    first.hf_token = gate_token
    assert refusal(first.url, first.credentials["agent-002"]["token"], **gate) is None
    await first.stop()
    for credential in first.credentials.values():
        assert refusal(first.url, credential["token"], **gate) == 401
    # A person takes the admin invite, as "Join with Hugging Face" does.
    with Client.join(
        first.url, invite, name="flock-test-admin", max_retries=5, **gate
    ) as person:
        person_token = person.token

    # Run 2, a relaunch, joins the same board with a new operator token.
    second = board.Resource(
        options, state_dir=tmp_path / "run-2", agents=agents, secrets=[]
    )
    await second.start()
    assert "?invite=" not in capsys.readouterr().out
    space = second.metadata["space"]
    assert not space["created"] and space["bucket"] == f"{space_id}-bucket"
    assert not space["admin_invite_created"]
    assert all(invite["status"] != "active" for invite in second.client.invites())
    assert refusal(second.url, first.operator_token, **gate) == 401
    with Client(second.url, person_token, max_retries=3, **gate) as person:
        assert person.me()["role"] == "admin"
    for name in agents:
        assert identity_id(second.credentials[name]) == identity_id(
            first.credentials[name]
        )
    as_agent(monkeypatch, second, "agent-002")
    earlier = board_tools.read("general", limit=100)["messages"]
    assert sorted(message["body"] for message in earlier) == sorted(posted)
    posted.append("hello again from agent-002")
    board_tools.post(posted[-1])
    await second.export(tmp_path / "run-2")
    assert exported(tmp_path / "run-2") == sorted(posted)

    # An outage fails the run once the board is unreachable for the grace period.
    monkeypatch.setattr(board, "SPACE_OUTAGE_GRACE_SECONDS", 20)
    watcher = asyncio.create_task(second.wait())
    await asyncio.to_thread(api.pause_space, space_id)
    with pytest.raises(RuntimeError, match="Board Space unreachable for 20 s"):
        await asyncio.wait_for(watcher, 300)
    # A paused Space's gate answers 503 itself, so revocation doesn't wait for a
    # hand-over: it gives up after REVOKE_SECONDS and names the agents.
    began = time.monotonic()
    with pytest.raises(
        RuntimeError,
        match="agent-001, agent-002, which may still work: BoardError: 503 http_error",
    ):
        await second.stop()
    assert time.monotonic() - began < 15


async def test_a_space_that_is_not_a_board_is_refused_unchanged(new_space, tmp_path):
    api = HfApi()
    space_id = new_space("static")
    api.create_repo(space_id, repo_type="space", space_sdk="static", private=True)
    before = api.space_info(space_id).sha
    resource = board.Resource(
        {"space_id": space_id}, state_dir=tmp_path / "state", agents=[], secrets=[]
    )
    with pytest.raises(
        RuntimeError,
        match=rf"Space {re.escape(space_id)}: DeployError: .*not a msgboard board",
    ):
        await resource.start()
    assert (tmp_path / "state/operator.token").exists()
    assert api.space_info(space_id).sha == before
    namespace = space_id.split("/")[0]
    assert not any(
        bucket.id.startswith(f"{space_id}-bucket")
        for bucket in api.list_buckets(namespace=namespace)
    )
    with pytest.raises(RuntimeError, match="did not start, so there is nothing"):
        await resource.export(tmp_path)
    await resource.stop()


async def test_the_startup_limit_stops_a_deploy_in_progress(new_space, tmp_path):
    options = {"space_id": new_space("limit"), "startup_timeout_seconds": 0.5}
    resources = Resources(
        ROOT / "resources",
        {"message_board": options},
        run_dir=tmp_path / "run",
        agents=[],
        secrets=[],
    )
    with pytest.raises(RuntimeError, match="message_board did not start within 0.5 s"):
        await resources.start()
    deploy_process = resources.providers["message_board"].deploy_process
    assert deploy_process.returncode < 0  # killed before it finished
    assert (tmp_path / "run/state/resources/message_board/operator.token").exists()
    assert await resources.finish() is True
    record = resources.records["message_board"]
    assert "did not start, so there is nothing to export" in record["export_error"]
    assert "cleanup_error" not in record
