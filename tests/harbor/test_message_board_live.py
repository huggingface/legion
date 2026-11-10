# vendored from eabb46f:tests/test_message_board_live.py
"""One end-to-end test of the board resource on a real Hugging Face Space.

Skipped unless FLOCK_LIVE_SPACES says where test Spaces go, as <namespace>/<name-prefix>
(for example your-name/flock-test), and a Hugging Face token is available:

    FLOCK_LIVE_SPACES=your-name/flock-test uv run pytest -m live

It deploys a CPU Basic Space, takes a few minutes, and deletes the Space and its
bucket afterwards. Its output never shows a secret. pytest's report shows the
arguments of every call in a traceback, the values in a failed assertion and the
captured output, so assertions are on computed values only, and `sanitized` reports
any failure as one line with secrets redacted, no traceback and no captured output.
Only --full-trace with --showlocals would still show the test's local variables.
"""

import asyncio
import json
import os
import re
import stat
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

import pytest
from huggingface_hub import HfApi, get_token
from msgboard.client import BoardError, Client

from flock.harbor.config import BUILTIN
from flock.harbor.resources import load_module

board = load_module(BUILTIN / "resources/message_board.py")
board_tools = load_module(BUILTIN / "tools/message_board.py")
PREFIX = os.environ.get("FLOCK_LIVE_SPACES")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        not (PREFIX and get_token()),
        reason="set FLOCK_LIVE_SPACES=<namespace>/<name-prefix> and log in to the Hub",
    ),
]


class Secrets(list):
    """Secrets to redact; pytest shows a function's arguments, but not these."""

    def __repr__(self):
        return f"<{len(self)} secrets>"


@contextmanager
def sanitized(secrets, capfd=None):
    """Turn a failure into the test's line number and the error, with secrets redacted."""
    try:
        yield
    except (Exception, pytest.fail.Exception) as exc:
        if capfd:
            capfd.readouterr()  # what the resource printed, the admin invite included
        lines = [
            str(frame.lineno)
            for frame in traceback.extract_tb(exc.__traceback__)
            if frame.filename == __file__ and frame.name != "sanitized"
        ]
        text = str(exc)
        for secret in sorted(filter(None, secrets), key=len, reverse=True):
            text = text.replace(secret, "<secret>")
        exc.args = (text,)  # --full-trace shows it as an argument of __exit__
        where = f"{Path(__file__).name}:{', '.join(lines) or '?'}"
        # from None: pytest would show the original error as this one's context.
        raise pytest.fail.Exception(
            f"{where}: {type(exc).__name__}: {text}", pytrace=False
        ) from None


@pytest.fixture
def new_space():
    """Names for new test Spaces; deletes the Spaces and their buckets afterwards."""
    api = HfApi()
    names = []

    def name(purpose):
        names.append(f"{PREFIX}-{purpose}-{uuid.uuid4().hex[:6]}")
        return names[-1]

    yield name
    with sanitized(Secrets([get_token()])):
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


def identity_id(binding):
    identity = binding["metadata"]["identity"]
    return identity.get("principal", identity)["id"]


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
    new_space, tmp_path, capfd, monkeypatch
):
    api = HfApi()
    hf_token = get_token()
    # The resources add the tokens and invite links they make.
    secrets = Secrets([hf_token])
    gate = {
        "headers": {"Authorization": f"Bearer {hf_token}"},
        "token_header": "X-Msgboard-Token",
    }
    space_id = new_space("board")
    options = {"space_id": space_id, "private": True}
    agents = ["agent-001", "agent-002"]
    with sanitized(secrets, capfd):
        # Run 1 deploys a new private board.
        first = board.Resource(
            options, state_dir=tmp_path / "run-1", agents=agents, secrets=secrets
        )
        await first.start()
        printed = re.search(r"\?invite=([\w-]+)", capfd.readouterr().out)
        assert printed is not None
        invite = printed.group(1)
        token_file = tmp_path / "run-1/operator.token"
        assert stat.S_IMODE(token_file.stat().st_mode) == 0o600
        url, space = first.metadata["url"], first.metadata["space"]
        assert space["id"] == space_id and space["bucket"] == f"{space_id}-bucket"
        assert space["created"] and space["private"] and space["admin_invite_created"]
        assert api.space_info(space_id).private
        assert api.bucket_info(space["bucket"]).private
        bindings = {name: first.bind(name) for name in agents}
        gets_hf_token = (
            bindings["agent-001"]["env"].get("MSGBOARD_HF_TOKEN") == hf_token
        )
        assert gets_hf_token
        assert bindings["agent-001"]["allowed_hosts"] == [url.removeprefix("https://")]
        # Without the HF token, the Space's gate refuses even a valid board token.
        status = refusal(url, bindings["agent-001"]["env"]["MSGBOARD_TOKEN"])
        assert status in (401, 404)
        as_agent(monkeypatch, first, "agent-001")
        posted = ["hello from agent-001"]
        board_tools.post(posted[0])

        # Every post through a restart is acknowledged and exported once, and the
        # watcher doesn't fail the run.
        watcher = asyncio.create_task(first.wait())
        during, stages = await asyncio.to_thread(post_through_a_restart, space_id)
        posted += during
        assert len(stages) > 1 and stages[-1] == "RUNNING", stages
        assert not watcher.done()
        watcher.cancel()
        await first.export(tmp_path / "run-1")
        assert exported(tmp_path / "run-1") == sorted(posted)

        # stop() revokes the agents' tokens and leaves the Space running.
        await first.stop()
        statuses = [
            refusal(url, binding["env"]["MSGBOARD_TOKEN"], **gate)
            for binding in bindings.values()
        ]
        assert statuses == [401, 401]
        # A person takes the admin invite, as "Join with Hugging Face" does.
        with Client.join(
            url, invite, name="flock-test-admin", max_retries=5, **gate
        ) as person:
            secrets.append(person.token)
            person_token = person.token

        # Run 2, a relaunch, joins the same board with a new operator token.
        second = board.Resource(
            options, state_dir=tmp_path / "run-2", agents=agents, secrets=secrets
        )
        await second.start()
        printed_again = "?invite=" in capfd.readouterr().out
        assert not printed_again
        space = second.metadata["space"]
        assert not space["created"] and space["bucket"] == f"{space_id}-bucket"
        assert not space["admin_invite_created"]
        status = refusal(url, token_file.read_text().strip(), **gate)
        assert status == 401
        operator_token = (tmp_path / "run-2/operator.token").read_text().strip()
        with Client(url, operator_token, max_retries=3, **gate) as operator:
            active = [i for i in operator.invites() if i["status"] == "active"]
        assert not active
        with Client(url, person_token, max_retries=3, **gate) as person:
            role = person.me()["role"]
        assert role == "admin"
        for name in agents:
            assert identity_id(second.bind(name)) == identity_id(bindings[name])
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
        # A paused Space's gate answers 503, not the board, so stop() doesn't wait
        # as for a restart: it gives up after REVOKE_SECONDS and names the agents.
        began = time.monotonic()
        with pytest.raises(
            RuntimeError,
            match="agent-001, agent-002, which may still work: BoardError: 503 http_error",
        ):
            await second.stop()
        assert time.monotonic() - began < 30
