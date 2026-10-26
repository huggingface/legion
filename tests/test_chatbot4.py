"""Offline native-git PR workflow, frozen interface and sandbox contract tests."""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from flock.world import Agent, ToolError
from flock.world.server import create_app
from flock.world.workspace import capture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worlds/chatbot4"))
from chatbot4 import Chatbot4World
from repository import Repository
from sandbox import run


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


class LocalHub:
    def __init__(self, remote):
        self.remote, self.n, self.comments = remote, 0, []

    def open(self, title, description):
        self.n += 1
        git(self.remote, "update-ref", f"refs/pr/{self.n}", "main")
        return self.n

    def ready(self, pr):
        pass

    def comment(self, pr, text):
        self.comments.append((pr, text))

    def merge(self, pr, comment):
        main, head = git(self.remote, "rev-parse", "main"), git(self.remote, "rev-parse", f"refs/pr/{pr}")
        sha = git(self.remote, "-c", "user.name=world", "-c", "user.email=world@flock.invalid",
                  "commit-tree", f"{head}^{{tree}}", "-p", main, "-p", head, "-m", comment)
        git(self.remote, "update-ref", "refs/heads/main", sha, main)


@pytest.fixture
def setup(tmp_path):
    seed = tmp_path / "seed"
    shutil.copytree(ROOT / "worlds/chatbot4/seed", seed, ignore=shutil.ignore_patterns("__pycache__"))
    git(seed, "init", "-b", "main")
    git(seed, "add", ".")
    git(seed, "-c", "user.name=seed", "-c", "user.email=seed@flock.invalid", "commit", "-m", "seed")
    remote = tmp_path / "remote.git"
    git(tmp_path, "clone", "--bare", str(seed), str(remote))
    repo = Repository(tmp_path / "world.git", str(remote))
    repo.hub = LocalHub(remote)
    world = Chatbot4World(tmp_path / "world-state", repo)
    clone = tmp_path / "agent" / "chatbot-swarm"
    clone.parent.mkdir()
    git(tmp_path, "clone", str(remote), str(clone))
    return world, clone, remote


def submit(world, clone, author="cedar", branch="greeting", pr=None):
    return world.call_workspace("open_pr", {"branch": branch, "title": "Improve greetings",
        "description": "Tested a conversation and the interface.", "pr": pr},
        Agent(author), capture(clone.parent, clone.name))


def approve(world, rec, who="birch"):
    return world.review_pr(Agent(who), rec["pr"], rec["head_sha"], "approve", "Checked the code and bot")


def test_independent_review_merge_and_restart(setup):
    world, clone, remote = setup
    rec = submit(world, clone)
    with pytest.raises(ToolError, match="self-review"):
        approve(world, rec, "cedar")
    with pytest.raises(ToolError, match="approve"):
        world.merge_pr(Agent("cedar"), rec["pr"])
    approve(world, rec)
    restarted = Chatbot4World(world.ws, world.repo)
    result = restarted.merge_pr(Agent("cedar"), rec["pr"])
    assert git(remote, "rev-parse", "main") == result["merged_sha"]
    assert len(git(remote, "rev-list", "--parents", "-n", "1", "main").split()) == 3
    assert json.loads((world.ws / "eval-queue.jsonl").read_text())["merged_sha"] == result["merged_sha"]


def test_update_invalidates_approval(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    new = submit(world, clone, pr=rec["pr"])
    assert new["head_sha"] != rec["head_sha"]
    with pytest.raises(ToolError, match="approve"):
        world.merge_pr(Agent("cedar"), rec["pr"])
    with pytest.raises(ToolError, match="head changed"):
        approve(world, rec)
    approve(world, new)
    assert world.merge_pr(Agent("birch"), new["pr"])["merged_sha"]


def test_two_prs_stale_base_and_rebase(setup):
    world, clone, _ = setup
    one, two = submit(world, clone, branch="first"), submit(world, clone, "birch", "second")
    approve(world, one)
    approve(world, two, "cedar")
    world.merge_pr(Agent("birch"), one["pr"])
    with pytest.raises(ToolError, match="stale"):
        world.merge_pr(Agent("cedar"), two["pr"])
    git(clone, "fetch", "origin")
    git(clone, "reset", "--soft", "origin/main")
    updated = submit(world, clone, "birch", "second", two["pr"])
    approve(world, updated, "cedar")
    assert world.merge_pr(Agent("cedar"), two["pr"])


def test_requested_changes_block_and_can_be_resolved(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    world.review_pr(Agent("pine"), rec["pr"], rec["head_sha"], "request_changes", "Need a test")
    with pytest.raises(ToolError, match="requests changes"):
        world.merge_pr(Agent("cedar"), rec["pr"])
    approve(world, rec, "pine")
    assert world.merge_pr(Agent("cedar"), rec["pr"])


def test_limits_interface_and_failed_contract(setup):
    world, clone, _ = setup
    submit(world, clone, branch="first")
    submit(world, clone, branch="second")
    with pytest.raises(ToolError, match="two open"):
        submit(world, clone, branch="third")
    (clone / "bot/__main__.py").write_text('print("new interface")')
    with pytest.raises(ToolError, match="fixed"):
        submit(world, clone, author="pine")
    shutil.copy(ROOT / "worlds/chatbot4/seed/bot/__main__.py", clone / "bot/__main__.py")
    (clone / "bot/modules/noise.py").write_text('print("bad extra output")\n')
    rec = submit(world, clone, author="pine")
    approve(world, rec)
    with pytest.raises(ToolError, match="contract failed"):
        world.merge_pr(Agent("pine"), rec["pr"])


def test_upload_rejects_escape_and_symlink(setup):
    _, clone, _ = setup
    with pytest.raises(ValueError, match="inside"):
        capture(clone, "../..")
    (clone / "leak").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        capture(clone.parent, clone.name)


def test_seed_module_loading_and_sandbox(setup):
    world, clone, _ = setup
    assert run(clone, ["hi", "", "bye"])["ok"]
    (clone / "bot/modules/topic.py").write_text('PRIORITY=100\ndef respond(text, memory):\n return "Topic!"\n')
    assert run(clone, ["hi"])["replies"] == ["Topic!"]
    (clone / "bot/modules/topic.py").write_text('import socket\nsocket.socket()\n')
    assert run(clone, ["hi"])["replies"] == ["Hello!"]
    assert not run(clone, ["one\ntwo"])["ok"]
    assert world.run_bot("main", ["hi"])["timeout_s"] == 300


async def test_http_upload_is_outside_model_args(setup):
    world, clone, _ = setup
    app = create_app(world, "secret")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app), base_url="http://world") as client:
        payload = {"args": {"branch": "hello", "title": "Hello bot", "description": "tested"},
                   "workspace": capture(clone.parent, clone.name)}
        headers = {"X-World-Token": "secret", "X-Flock-Agent": "cedar"}
        r = await client.post("/tools/open_pr", json=payload, headers=headers)
        assert r.json()["ok"], r.text
        assert "secret" not in r.text
        payload["workspace"]["files"]["../leak"] = base64.b64encode(b"leak").decode()
        payload["args"]["branch"] = "bad"
        r = await client.post("/tools/open_pr", json=payload, headers=headers)
        assert not r.json()["ok"]


async def test_world_hub_captures_local_workdir_and_preserves_identity(setup):
    from types import SimpleNamespace

    from .test_world import Served, hub_for

    world, clone, _ = setup
    served = Served(world)
    try:
        hub = hub_for(served, runtime=lambda _: SimpleNamespace(workdir=clone.parent, turn_id="t-1"))
        try:
            result = await hub.call("cedar", "world:open_pr", {"branch": "hello",
                "title": "New greeting", "description": "tested"})
            assert not result.error, result.output
            assert json.loads(result.output)["author"] == "cedar"
            result = await hub.call("cedar", "world:run_bot", {"ref_or_path": clone.name, "lines": ["hi"]})
            assert not result.error and json.loads(result.output)["replies"] == ["Hello!"]
        finally:
            await hub.aclose()
    finally:
        served.close()


def test_bot_cannot_read_world_credentials_or_write_files(setup, monkeypatch):
    _, clone, _ = setup
    monkeypatch.setenv("HF_TOKEN", "private-world-token")
    (clone / "bot/modules/check.py").write_text(
        'import os\nPRIORITY=100\ndef respond(text, memory):\n'
        ' return os.environ.get("HF_TOKEN", "no token")\n')
    result = run(clone, ["hi"])
    assert result["replies"] == ["no token"]
    assert "private-world-token" not in json.dumps(result)
    target = clone / "leaked.txt"
    (clone / "bot/modules/check.py").write_text(
        f'open({str(target)!r}, "w").write("bad")\n')
    assert run(clone, ["hi"])["ok"]
    assert not target.exists()


def test_world_archive_restores_git_reviews_and_merge_hook(setup, tmp_path, monkeypatch):
    import importlib.util

    world, clone, remote = setup
    rec = submit(world, clone)
    approve(world, rec)
    shutil.copytree(world.repo.root, world.ws / "repo.git")
    from .test_controller_job import FakeBucket

    spec = importlib.util.spec_from_file_location("chatbot4_snapshot", ROOT / "worlds/chatbot4/snapshot.py")
    snapshot = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(snapshot)
    store = FakeBucket(tmp_path / "bucket")
    snapshot.transfer("push", world.ws, "hf://buckets/fake/run4", store)
    restored = tmp_path / "restored"
    assert snapshot.transfer("restore", restored, "hf://buckets/fake/run4", store)
    assert not snapshot.transfer("restore", restored, "hf://buckets/fake/absent",
                                 FakeBucket(tmp_path / "absent"))
    repo = Repository(restored / "repo.git", str(remote))
    repo.hub = LocalHub(remote)
    restarted = Chatbot4World(restored, repo)
    assert restarted.list_prs()[0]["reviews"]["birch"]["head_sha"] == rec["head_sha"]
    result = restarted.merge_pr(Agent("cedar"), rec["pr"])
    assert json.loads((restored / "eval-queue.jsonl").read_text())["merged_sha"] == result["merged_sha"]
