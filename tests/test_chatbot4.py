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

    def close(self, pr, comment):
        self.closed = [*getattr(self, "closed", []), pr]

    def merge(self, pr, comment):
        main, head = git(self.remote, "rev-parse", "main"), git(self.remote, "rev-parse", f"refs/pr/{pr}")
        sha = git(self.remote, "-c", "user.name=world", "-c", "user.email=world@flock.invalid",
                  "commit-tree", f"{head}^{{tree}}", "-p", main, "-p", head, "-m", comment)
        git(self.remote, "update-ref", "refs/heads/main", sha, main)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    # Workflow/contract unit tests. Real kernel isolation is checked separately in
    # test_run4_isolation.py and required by the prepared cluster dry run.
    monkeypatch.setattr("sandbox.isolate_command", lambda argv, *args, **kwargs: argv)
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


def test_bot_cannot_send_signals_to_other_processes(setup):
    _, clone, _ = setup
    (clone / "bot/modules/signals.py").write_text(
        'import os, signal\nPRIORITY=100\n'
        'def attempt(fn):\n'
        '    try:\n'
        '        fn()\n'
        '        return "sent"\n'
        '    except Exception as exc:\n'
        '        return type(exc).__name__\n'
        'def respond(text, memory):\n'
        '    group = os.getpgid(os.getppid())\n'
        '    probes = [lambda: os.killpg(group, 0), lambda: os.kill(os.getppid(), 0),\n'
        '              lambda: __import__("posix").killpg(group, 0),\n'
        '              lambda: signal.pthread_kill(__import__("threading").get_ident(), 0),\n'
        '              lambda: __import__("_signal").pidfd_send_signal(0, 0)]\n'
        '    return " ".join(attempt(p) for p in probes)\n')
    result = run(clone, ["hi"])
    assert result["ok"], result
    assert "sent" not in result["replies"][0], result["replies"]


def test_upload_respects_ignores_size_limit_and_compiles_only_changes(setup, monkeypatch):
    world, clone, remote = setup
    (clone / "bot/modules/topic.py").write_text('def respond(text, memory):\n    return None\n')
    snapshot = capture(clone.parent, clone.name)
    junk = {".venv/lib/broken.py": b"def (:\n", "bot/__pycache__/x.cpython-312.pyc": b"\0junk",
            ".pytest_cache/v/cache/lastfailed": b"{}"}
    snapshot["files"].update({k: base64.b64encode(v).decode() for k, v in junk.items()})
    snapshot["files"].pop(".gitignore")  # the world's default excludes still apply
    checked = []
    real = world._contract
    monkeypatch.setattr(world, "_contract", lambda tree, names=None: (checked.append(names),
                                                                       real(tree, names)))
    rec = world.call_workspace("open_pr", {"branch": "topic", "title": "Topic module",
                                           "description": "tested"}, Agent("cedar"), snapshot)
    committed = git(remote, "ls-tree", "-r", "--name-only", rec["head_sha"]).splitlines()
    assert "bot/modules/topic.py" in committed and ".gitignore" not in committed
    assert not [name for name in committed if name in junk]
    assert checked == [["bot/modules/topic.py"]]
    big = clone / "bot/modules/data.py"
    big.write_text("X = 1\n" + "#" * (2 << 20))
    with pytest.raises(ToolError, match="file too large: bot/modules/data.py"):
        submit(world, clone, branch="big")
    with pytest.raises(ToolError, match="file too large"):
        world.call_workspace("run_bot", {"ref_or_path": "chatbot-swarm", "lines": ["hi"]},
                             Agent("cedar"), capture(clone.parent, clone.name))
    assert world.repo.hub.n == 1  # the refused upload created no Hub PR


def test_open_pr_from_an_older_main_rebases_or_refuses_before_the_hub(setup):
    world, clone, remote = setup
    old_base = git(clone, "rev-parse", "HEAD")
    other = clone.parent / "other"
    git(clone.parent, "clone", str(remote), str(other))
    (other / "bot/modules/weather.py").write_text('def respond(text, memory):\n    return None\n')
    first = submit(world, other, author="pine", branch="weather")
    approve(world, first)
    merged = world.merge_pr(Agent("pine"), first["pr"])["merged_sha"]
    # birch still has the old main: a non-conflicting change is rebased onto main
    (clone / "bot/modules/time.py").write_text('def respond(text, memory):\n    return None\n')
    rec = submit(world, clone, author="birch", branch="time")
    assert rec["base_sha"] == merged and rec["rebased_from"] == old_base
    assert git(remote, "rev-parse", f"{rec['head_sha']}^") == merged
    files = git(remote, "ls-tree", "-r", "--name-only", rec["head_sha"]).splitlines()
    assert {"bot/modules/weather.py", "bot/modules/time.py"} <= set(files)
    approve(world, rec, "cedar")
    assert world.merge_pr(Agent("cedar"), rec["pr"])["merged_sha"]
    # a conflicting change from the old base is refused, and no Hub PR is created
    opened = world.repo.hub.n
    (clone / "bot/modules/weather.py").write_text('def respond(text, memory):\n    return "Sunny"\n')
    with pytest.raises(ToolError, match=f"based on {old_base}, main is .*pull, rebase"):
        submit(world, clone, author="oak", branch="sunny")
    assert world.repo.hub.n == opened and not getattr(world.repo.hub, "closed", [])


def test_a_failed_push_after_creating_the_hub_pr_closes_it(setup, monkeypatch):
    world, clone, _ = setup

    def push(sha, ref, expected=None):
        raise ToolError("repository operation failed; refresh your clone and retry", "git_error")

    monkeypatch.setattr(world.repo, "push", push)
    with pytest.raises(ToolError, match="repository operation failed"):
        submit(world, clone)
    assert world.repo.hub.closed == [world.repo.hub.n] and world.list_prs() == []


def test_hub_merge_that_fails_after_merging_is_reconciled(setup, monkeypatch):
    world, clone, remote = setup
    rec = submit(world, clone, branch="first")
    (clone / "bot/modules/other.py").write_text('def respond(text, memory):\n    return None\n')
    submit(world, clone, branch="second")  # a different head, which stays open
    approve(world, rec)
    real = world.repo.hub.merge

    def merge_then_fail(pr, comment):
        real(pr, comment)
        raise ToolError("Hub request failed; retry later", "hub_error")

    monkeypatch.setattr(world.repo.hub, "merge", merge_then_fail)
    result = world.merge_pr(Agent("cedar"), rec["pr"])
    assert result["merged_sha"] == git(remote, "rev-parse", "main")
    assert [r["pr"] for r in world.list_prs("merged")] == [rec["pr"]]
    assert submit(world, clone, branch="third")["state"] == "open"  # the cap counts it no more
    queue = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert queue[-1]["reconciled"] and queue[-1]["merged_sha"] == result["merged_sha"]


def test_restore_from_an_archive_older_than_a_merge_reconciles(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    before = (world.ws / "prs.json").read_text()
    merged = world.merge_pr(Agent("birch"), rec["pr"])["merged_sha"]
    (world.ws / "prs.json").write_text(before)  # restored from an archive taken before the merge
    restarted = Chatbot4World(world.ws, world.repo)
    assert restarted.list_prs() == []
    assert restarted.list_prs("merged")[0]["merged_sha"] == merged
    with pytest.raises(ToolError, match="no open PR"):
        restarted.merge_pr(Agent("birch"), rec["pr"])


def test_close_pr_frees_the_cap_and_is_limited_to_author_or_operator(setup):
    world, clone, _ = setup
    one, two = submit(world, clone, branch="first"), submit(world, clone, branch="second")
    with pytest.raises(ToolError, match=f"update one with pr=<n> or close one with close_pr .*{one['pr']}"):
        submit(world, clone, branch="third")
    with pytest.raises(ToolError, match="only the author or the run operator"):
        world.close_pr(Agent("birch"), one["pr"], "duplicate")
    closed = world.close_pr(Agent("cedar"), one["pr"], "superseded by a smaller PR")
    assert closed["state"] == "closed" and world.repo.hub.closed == [one["pr"]]
    assert submit(world, clone, branch="third")["state"] == "open"
    world.operators = {"operator"}
    world.close_pr(Agent("operator"), two["pr"], "abandoned")
    assert {r["pr"] for r in world.list_prs("closed")} == {one["pr"], two["pr"]}
    with pytest.raises(ToolError, match="no open PR"):
        world.review_pr(Agent("birch"), one["pr"], one["head_sha"], "approve", "late review")
    events = [json.loads(line) for line in (world.ws / "prs.jsonl").read_text().splitlines()]
    assert [e["agent"] for e in events if e["event"] == "pr_closed"] == ["cedar", "operator"]
