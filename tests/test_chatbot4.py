"""Offline native-git PR workflow, frozen interface and sandbox contract tests."""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest

from flock.world import Agent, ToolError
from flock.world.server import create_app
from flock.world.workspace import capture, capture_v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "worlds/chatbot4"))
import chatbot4
from chatbot4 import Chatbot4World
from repository import Hub, Repository
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
        self.close_comments = {**getattr(self, "close_comments", {}), pr: comment}

    def merge(self, pr, comment):  # like the Hub: a three-way merge of refs/pr/N into main
        main, head = git(self.remote, "rev-parse", "main"), git(self.remote, "rev-parse", f"refs/pr/{pr}")
        tree = git(self.remote, "merge-tree", "--write-tree", main, head).split()[0]
        sha = git(self.remote, "-c", "user.name=world", "-c", "user.email=world@flock.invalid",
                  "commit-tree", tree, "-p", main, "-p", head, "-m", comment)
        git(self.remote, "update-ref", "refs/heads/main", sha, main)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    return _setup(tmp_path, monkeypatch, "seed")


@pytest.fixture
def setup_v8(tmp_path, monkeypatch):
    """A world on the v8 seed (CHATBOT_SEED_DIR=seed-v8): the runtime with an arbiter."""
    monkeypatch.setenv("CHATBOT_SEED_DIR", "seed-v8")
    return _setup(tmp_path, monkeypatch, "seed-v8")


def _setup(tmp_path, monkeypatch, seed_dir):
    # Workflow/contract unit tests. Real kernel isolation is checked separately in
    # test_isolation.py and required by the prepared cluster dry run.
    monkeypatch.setattr("sandbox.isolate_command", lambda argv, *args, **kwargs: argv)
    monkeypatch.setenv("CHATBOT_PR_CREATE_GAP_S", "0")  # the gap has its own test
    monkeypatch.setenv("CHATBOT_HUB_OUTBOX_GAP_S", "0")
    monkeypatch.setattr(Chatbot4World, "MERGER", False)  # merge() runs the queue in the test's thread
    monkeypatch.setenv("CHATBOT_MERGE_BUNDLE", "0")  # the single-PR path; bundle tests set world.bundle
    seed = tmp_path / "seed"
    shutil.copytree(ROOT / "worlds/chatbot4" / seed_dir, seed, ignore=shutil.ignore_patterns("__pycache__"))
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


def _write_archive(world, state):
    """Leave the workspace as an archive from before state.db restores it: prs.json only."""
    world.db.write_text(json.dumps(state))
    for name in ("state.db", "state.db-wal", "state.db-shm"):
        (world.ws / name).unlink(missing_ok=True)


def submit(world, clone, author="cedar", branch="greeting", pr=None):
    return world.call_workspace("open_pr", {"branch": branch, "title": "Improve greetings",
        "description": "Tested a conversation and the interface.", "pr": pr},
        Agent(author), capture(clone.parent, clone.name))


def merge(world, agent, pr):
    """merge_pr, then the merge queue's run of its first PR (this one in these tests): the
    merge, or the refusal raised."""
    out = world.merge_pr(agent, pr)
    if not out.get("queued"):
        return out
    for entry in world.state["merge_queue"]:
        if entry["pr"] == pr:
            entry["next_at"] = 0.0  # a retry's time has come
    return world._merge_next()


def approve(world, rec, who="birch"):
    return world.review_pr(Agent(who), rec["pr"], rec["head_sha"], "approve", "Checked the code and bot")


def test_independent_review_merge_and_restart(setup):
    world, clone, remote = setup
    rec = submit(world, clone)
    with pytest.raises(ToolError, match="self-review"):
        approve(world, rec, "cedar")
    with pytest.raises(ToolError, match="approve"):
        merge(world, Agent("cedar"), rec["pr"])
    approve(world, rec)
    restarted = Chatbot4World(world.ws, world.repo)
    result = merge(restarted, Agent("cedar"), rec["pr"])
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
        merge(world, Agent("cedar"), rec["pr"])
    with pytest.raises(ToolError, match="head changed"):
        approve(world, rec)
    approve(world, new)
    assert merge(world, Agent("birch"), new["pr"])["merged_sha"]


def test_requested_changes_block_and_can_be_resolved(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    world.review_pr(Agent("pine"), rec["pr"], rec["head_sha"], "request_changes", "Need a test")
    with pytest.raises(ToolError, match="requests changes"):
        merge(world, Agent("cedar"), rec["pr"])
    approve(world, rec, "pine")
    assert merge(world, Agent("cedar"), rec["pr"])


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
        merge(world, Agent("pine"), rec["pr"])


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
    world.flush()  # the periodic archive sees prs.json at most CHATBOT_SAVE_INTERVAL_S old
    snapshot.transfer("push", world.ws, "hf://buckets/fake/run4", store)
    restored = tmp_path / "restored"
    assert snapshot.transfer("restore", restored, "hf://buckets/fake/run4", store)
    assert not snapshot.transfer("restore", restored, "hf://buckets/fake/absent",
                                 FakeBucket(tmp_path / "absent"))
    (restored / "prs.json").unlink()  # the state comes from the state.db backup in the archive
    repo = Repository(restored / "repo.git", str(remote))
    repo.hub = LocalHub(remote)
    restarted = Chatbot4World(restored, repo)
    assert restarted.list_prs()[0]["reviews"]["birch"]["head_sha"] == rec["head_sha"]
    result = merge(restarted, Agent("cedar"), rec["pr"])
    assert json.loads((restored / "eval-queue.jsonl").read_text())["merged_sha"] == result["merged_sha"]


def test_bucket_restores_an_empty_workspace_and_pushes_at_close(setup, tmp_path, monkeypatch):
    """CHATBOT_BUCKET: an empty workspace is restored from world.tar.gz before the world reads
    it, close() pushes it back, and a workspace of another bucket is refused."""
    from .test_controller_job import FakeBucket

    world, clone, remote = setup
    world.close()
    bucket, hub = FakeBucket(tmp_path / "bucket"), LocalHub(remote)
    monkeypatch.setattr("snapshot.HubBucketStore", lambda url, token=None: bucket)
    monkeypatch.setattr(chatbot4, "Hub", lambda repo_id: hub)
    monkeypatch.setattr(chatbot4, "Repository", lambda root, url, hub: Repository(root, str(remote), hub))
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/fake/v8")
    first = Chatbot4World(tmp_path / "ws1")  # nothing in the bucket yet: a first start
    rec = submit(first, clone)
    approve(first, rec)
    first.close()
    assert bucket.list("world.tar.gz") == ["world.tar.gz"]
    restored = Chatbot4World(tmp_path / "ws2")
    try:
        assert restored.list_prs()[0]["reviews"]["birch"]["head_sha"] == rec["head_sha"]
        assert merge(restored, Agent("cedar"), rec["pr"])["merged_sha"] == git(remote, "rev-parse", "main")
    finally:
        restored.close()
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/fake/other")
    with pytest.raises(RuntimeError, match="holds the world of hf://buckets/fake/v8"):
        Chatbot4World(tmp_path / "ws2")


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
    monkeypatch.setattr("chatbot4.MAX_DELETED_FILES", 1)  # dropping .gitignore is 1 deletion: allowed
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
    big.unlink()
    (clone / "README.md").unlink()
    (clone / "AGENTS.md").unlink()
    for snap in (capture, capture_v2):  # an incomplete clone, sent as v1 files or a v2 patch
        with pytest.raises(ToolError, match="deletes 2 files; your clone looks incomplete") as err:
            world.call_workspace("open_pr", {"branch": "gone", "title": "Tidy", "description": "tested"},
                                 Agent("cedar"), snap(clone.parent, clone.name))
        assert err.value.code == "mass_delete"
    assert world.repo.hub.n == 1  # the refused uploads created no Hub PR


def test_open_pr_from_an_older_main_rebases_or_refuses_before_the_hub(setup):
    world, clone, remote = setup
    old_base = git(clone, "rev-parse", "HEAD")
    other = clone.parent / "other"
    git(clone.parent, "clone", str(remote), str(other))
    (other / "bot/modules/weather.py").write_text('def respond(text, memory):\n    return None\n')
    first = submit(world, other, author="pine", branch="weather")
    approve(world, first)
    merged = merge(world, Agent("pine"), first["pr"])["merged_sha"]
    # birch still has the old main: a non-conflicting change is rebased onto main
    (clone / "bot/modules/time.py").write_text('def respond(text, memory):\n    return None\n')
    rec = submit(world, clone, author="birch", branch="time")
    assert rec["base_sha"] == merged and rec["rebased_from"] == old_base
    assert git(remote, "rev-parse", f"{rec['head_sha']}^") == merged
    files = git(remote, "ls-tree", "-r", "--name-only", rec["head_sha"]).splitlines()
    assert {"bot/modules/weather.py", "bot/modules/time.py"} <= set(files)
    approve(world, rec, "cedar")
    assert merge(world, Agent("cedar"), rec["pr"])["merged_sha"]
    # a conflicting change from the old base is refused, and no Hub PR is created
    opened = world.repo.hub.n
    (clone / "bot/modules/weather.py").write_text('def respond(text, memory):\n    return "Sunny"\n')
    with pytest.raises(ToolError, match=f"based on {old_base}, main is .*pull, rebase"):
        submit(world, clone, author="oak", branch="sunny")
    assert world.repo.hub.n == opened and not getattr(world.repo.hub, "closed", [])


def failing_pr_push(world, monkeypatch, failures):
    """Make the next `failures` pushes to refs/pr/* fail; returns the list of attempted refs."""
    import chatbot4

    monkeypatch.setattr(chatbot4, "PUSH_RETRY_DELAYS", (0.01, 0.01))
    real, attempts = world.repo.push, []

    def push(sha, ref, expected=None, force=False):
        attempts.append(ref)
        if ref.startswith("refs/pr/") and failures[0] > 0:
            failures[0] -= 1
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return real(sha, ref, expected, force)

    monkeypatch.setattr(world.repo, "push", push)
    return attempts


def test_a_failed_push_after_creating_the_hub_pr_parks_a_draft_and_reuses_it(setup, monkeypatch, caplog):
    world, clone, _ = setup
    failures = [3]  # the push and both retries
    attempts = failing_pr_push(world, monkeypatch, failures)
    with caplog.at_level("WARNING"), pytest.raises(ToolError) as exc:
        submit(world, clone)
    number = world.repo.hub.n
    assert exc.value.code == "hub_busy" and exc.value.message == (
        f"the Hub did not accept the push for your PR #{number}; retry open_pr with pr={number} in a minute")
    assert attempts == [f"refs/pr/{number}"] * 3
    assert number == 1 and not getattr(world.repo.hub, "closed", [])  # never closed
    assert f"parked draft PR #{number} for agent=cedar" in caplog.text
    [draft] = world.list_prs()
    assert draft["pr"] == number and draft["draft"] is True and draft["state"] == "open"
    assert draft["head_sha"] is None and world.list_prs(needs_review=True) == []
    with pytest.raises(ToolError, match=f"PR #{number} is a draft"):
        world.review_pr(Agent("birch"), number, "0" * 40, "approve", "looks fine")
    # the draft is one of cedar's two open PRs, and its slug stays cedar's
    (clone / "bot/modules/other.py").write_text('def respond(text, memory):\n    return None\n')
    other = submit(world, clone, branch="other")
    with pytest.raises(ToolError, match=f"at most two open PRs.*yours: {other['pr']}, {number}\\)"):
        submit(world, clone, branch="third")
    with pytest.raises(ToolError, match="only the author"):
        submit(world, clone, author="birch", pr=number)
    # the next open_pr for that branch (with or without pr=) reuses the PR number
    opened = world.repo.hub.n
    rec = submit(world, clone)
    assert rec["pr"] == number and world.repo.hub.n == opened
    assert world.state["drafts"] == {} and "draft" not in world.list_prs()[0]
    approve(world, rec)
    assert merge(world, Agent("birch"), number)["merged_sha"]


def test_a_parked_draft_survives_a_world_restart(setup, monkeypatch):
    world, clone, _ = setup
    failing_pr_push(world, monkeypatch, [3])
    with pytest.raises(ToolError, match="did not accept the push"):
        submit(world, clone)
    number = world.repo.hub.n
    restarted = Chatbot4World(world.ws, world.repo)
    assert [(r["pr"], r.get("draft")) for r in restarted.list_prs()] == [(number, True)]
    rec = submit(restarted, clone, pr=number)
    assert rec["pr"] == number and restarted.repo.hub.n == number
    assert Chatbot4World(world.ws, world.repo).list_prs()[0]["head_sha"] == rec["head_sha"]


def test_a_push_retry_after_main_moved_rebases_within_the_same_pr(setup, monkeypatch):
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", str(remote), str(other))
    (other / "bot/modules/first.py").write_text('def respond(text, memory):\n    return None\n')
    first = submit(world, other, author="pine", branch="first")
    real_push = world.repo.push

    def push_then_merge(sha, ref, expected=None, force=False):
        if ref.startswith("refs/pr/") and ref != f"refs/pr/{first['pr']}":
            world.repo.hub.merge(first["pr"], "merged outside the world")  # main moves
            monkeypatch.setattr(world.repo, "push", real_push)
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return real_push(sha, ref, expected, force)

    import chatbot4
    monkeypatch.setattr(chatbot4, "PUSH_RETRY_DELAYS", (0.01, 0.01))
    monkeypatch.setattr(world.repo, "push", push_then_merge)
    (clone / "bot/modules/second.py").write_text('def respond(text, memory):\n    return None\n')
    rec = submit(world, clone, author="birch", branch="second")
    assert rec["pr"] == world.repo.hub.n == first["pr"] + 1  # one Hub PR, none closed
    assert rec["base_sha"] == git(remote, "rev-parse", "main") != first["base_sha"]
    assert not getattr(world.repo.hub, "closed", [])


def test_concurrent_open_prs_never_run_two_hub_pushes_at_once(setup, monkeypatch):
    import threading
    import time

    world, clone, _ = setup
    rec = submit(world, clone, branch="seed")
    active, peak, guard = [0], [0], threading.Lock()

    def tracked(fn):
        def wrapper(*args, **kwargs):
            with guard:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                time.sleep(0.05)
                return fn(*args, **kwargs)
            finally:
                with guard:
                    active[0] -= 1
        return wrapper

    hub = world.repo.hub
    for name in ("open", "ready", "close"):  # the push lane; comments and merges have their own
        monkeypatch.setattr(hub, name, tracked(getattr(hub, name)))
    monkeypatch.setattr(world.repo, "push", tracked(world.repo.push))
    snapshot, errors = capture(clone.parent, clone.name), []

    def call(fn):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=call, args=(lambda who=who: world.call_workspace(
        "open_pr", {"branch": "topic", "title": "Topic", "description": "tested"},
        Agent(who), snapshot),)) for who in ("birch", "pine", "oak", "elm", "ash")]
    threads.append(threading.Thread(target=call, args=(lambda: approve(world, rec, "fir"),)))
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert errors == [] and peak[0] == 1
    assert hub.n == 6 and len(world.list_prs()) == 6


def test_pr_creation_gap_and_busy_hub_refuse_before_creating(setup, monkeypatch, caplog):
    import time

    world, clone, _ = setup
    submit(world, clone, branch="first")
    world.pr_gap, world.hub_wait = 60, 0.5
    with caplog.at_level("WARNING"), pytest.raises(ToolError) as exc:
        submit(world, clone, author="birch", branch="second")
    assert (exc.value.code, exc.value.message) == ("hub_busy", "the Hub is busy; retry in about a minute")
    assert exc.value.retry_after_s == 60
    assert world.repo.hub.n == 1 and "reason=pr_create_gap" in caplog.text
    world.pr_gap, world.last_create = 0.4, time.monotonic()
    started = time.monotonic()
    assert submit(world, clone, author="birch", branch="second")["pr"] == 2  # waits the gap
    assert time.monotonic() - started >= 0.3
    world.pr_gap = 0
    with world.lanes.hold("push"), pytest.raises(ToolError, match="the Hub is busy") as exc:
        submit(world, clone, author="pine", branch="third")  # another Hub write in progress
    assert exc.value.code == "hub_busy" and world.repo.hub.n == 2


def test_pause_flag_refuses_open_and_merge_but_not_reviews_or_run_bot(setup, tmp_path, monkeypatch):
    world, clone, _ = setup
    rec = submit(world, clone)
    flag = tmp_path / "shared" / "prs-paused"
    flag.parent.mkdir()
    monkeypatch.setenv("CHATBOT_PRS_PAUSED_FILE", str(flag))
    world = Chatbot4World(world.ws, world.repo)
    flag.touch()  # read per call: no restart needed
    paused = "PRs are paused by the operators; keep working locally, run_bot and reviews still work"
    for attempt in (lambda: submit(world, clone, pr=rec["pr"]),
                    lambda: merge(world, Agent("birch"), rec["pr"])):
        with pytest.raises(ToolError) as exc:
            attempt()
        assert (exc.value.code, exc.value.message) == ("paused", paused)
    assert approve(world, rec)["verdict"] == "approve"
    assert world.run_bot("main", ["hi"])["ok"]
    assert world.list_prs()[0]["pr"] == rec["pr"]
    flag.unlink()
    assert merge(world, Agent("birch"), rec["pr"])["merged_sha"]
    other = submit(world, clone, branch="other")
    monkeypatch.setenv("CHATBOT_PRS_PAUSED", "1")  # start-time override
    world = Chatbot4World(world.ws, world.repo)
    with pytest.raises(ToolError, match="paused by the operators"):
        submit(world, clone, branch="third")
    assert world.close_pr(Agent("cedar"), other["pr"], "paused")["state"] == "closed"


def test_admin_lanes_and_knobs(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    assert world.knobs == {"save.interval_s": world.save_interval, "save.legacy_json": True,
                           "merge.bundle": 0, "review.max_reviewers": 0, "pr.open_floor": 20, "hub.push.slots": 1, "hub.merge.slots": 1, "hub.comment.slots": 1,
                           "hub.comment.gap_s": 0.0}
    world.admin.set({"hub.push.slots": 2, "hub.comment.gap_s": 0.5})
    assert world.lanes["push"].slots == 2 and world.lanes["comment"].gap_s == 0.5
    world.admin.pause(["comment"])
    assert world.lanes.metrics()["comment"]["paused"] == 1
    world.admin.resume(["comment"])
    world.admin.pause(["open"], "hub 429s")
    with pytest.raises(ToolError, match="paused by the operators"):
        submit(world, clone, branch="second")  # a new PR: the open lane
    updated = submit(world, clone, pr=rec["pr"])  # an update is a push only
    assert updated["pr"] == rec["pr"]
    rec = updated  # its head: a new commit once the clock passed a second (commit date)
    world.admin.pause(["merge"])
    with pytest.raises(ToolError, match="paused by the operators"):
        merge(world, Agent("birch"), rec["pr"])
    world.admin.set({"pr.open_floor": 3})
    assert world.budget_floor == 3
    world = Chatbot4World(world.ws, world.repo)  # restart: knobs.json in the workspace
    assert world.budget_floor == 3 and world.paused("open") and world.paused("merge")
    assert world.lanes["push"].slots == 2 and world.lanes.metrics()["merge"]["paused"] == 1
    world.admin.resume(["merge", "open"])
    approve(world, rec)
    assert merge(world, Agent("birch"), rec["pr"])["merged_sha"]


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
    result = merge(world, Agent("cedar"), rec["pr"])
    assert result["merged_sha"] == git(remote, "rev-parse", "main")
    assert [r["pr"] for r in world.list_prs("merged")] == [rec["pr"]]
    assert submit(world, clone, branch="third")["state"] == "open"  # the cap counts it no more
    queue = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert queue[-1]["reconciled"] and queue[-1]["merged_sha"] == result["merged_sha"]


def test_restore_from_an_archive_older_than_a_merge_reconciles(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    world.flush()
    before = (world.ws / "prs.json").read_text()
    merged = merge(world, Agent("birch"), rec["pr"])["merged_sha"]
    _write_archive(world, json.loads(before))  # restored from an archive taken before the merge
    restarted = Chatbot4World(world.ws, world.repo)
    assert restarted.list_prs() == []
    assert restarted.list_prs("merged")[0]["merged_sha"] == merged
    again = merge(restarted, Agent("birch"), rec["pr"])  # a merged PR answers with its merge
    assert again["already_merged"] and again["merged_sha"] == merged


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
    with pytest.raises(ToolError, match="was closed by cedar: superseded by a smaller PR"):
        world.review_pr(Agent("birch"), one["pr"], one["head_sha"], "approve", "late review")
    events = [json.loads(line) for line in (world.ws / "prs.jsonl").read_text().splitlines()]
    assert [e["agent"] for e in events if e["event"] == "pr_closed"] == ["cedar", "operator"]


def test_open_pr_work_runs_outside_the_world_lock(setup, monkeypatch):
    import threading

    world, clone, _ = setup
    entered, release, results = threading.Event(), threading.Event(), {}
    real = world._files

    def slow_files(snapshot, tree):
        if threading.current_thread().name == "slow":
            entered.set()
            assert release.wait(10)
        return real(snapshot, tree)

    monkeypatch.setattr(world, "_files", slow_files)
    snapshot = capture(clone.parent, clone.name)
    slow = threading.Thread(name="slow", target=lambda: results.update(slow=world.call_workspace(
        "open_pr", {"branch": "slow", "title": "Slow upload", "description": "tested"},
        Agent("cedar"), snapshot)))
    slow.start()
    try:
        assert entered.wait(10)
        assert world.lock.acquire(timeout=2), "the lock is held during the upload/compile"
        world.lock.release()
        fast = submit(world, clone, author="birch", branch="fast")  # proceeds meanwhile
        approve(world, fast, "pine")
        assert merge(world, Agent("pine"), fast["pr"])["merged_sha"]
    finally:
        release.set()
        slow.join(10)
    # main moved while the slow open was in flight: it was rebased onto the new main
    assert results["slow"]["base_sha"] == world.repo.sha()


def test_merge_queue_runs_in_order_each_on_the_last_main(setup):
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", "-q", str(remote), str(other))
    _module(other, "one", None)
    one = submit(world, other, "birch", "one")
    _module(clone, "two", None)
    two = submit(world, clone, "cedar", "two")
    approve(world, one, "pine")
    approve(world, two, "pine")
    with world.lanes.hold("merge"):  # a merge in progress: requests never wait for the lane
        started = time.monotonic()
        assert world.merge_pr(Agent("oak"), two["pr"])["position"] == 1
        assert world.merge_pr(Agent("pine"), one["pr"])["position"] == 2
        assert world.merge_pr(Agent("cedar"), two["pr"])["position"] == 1  # a re-call keeps its place
        assert time.monotonic() - started < 5
    assert [e["pr"] for e in world.state["merge_queue"]] == [two["pr"], one["pr"]]
    a, b = world._merge_next(), world._merge_next()
    assert (a["pr"], b["pr"]) == (two["pr"], one["pr"]) and not world.state["merge_queue"]
    assert "rebased_from" not in a and b["rebased_from"] == one["base_sha"]
    assert b["parent_sha"] == a["merged_sha"] and git(remote, "rev-parse", "main") == b["merged_sha"]
    assert len(world.list_prs("merged")) == 2
    queued = [json.loads(line) for line in (world.ws / "prs.jsonl").read_text().splitlines()]
    assert [(e["agent"], e["pr"]) for e in queued if e["event"] == "merge_queued"] == [
        ("oak", two["pr"]), ("pine", one["pr"])]
    merged = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert [e["pr"] for e in merged] == [two["pr"], one["pr"]]


def test_main_moving_while_the_hub_pr_is_created_keeps_one_pr(setup, monkeypatch):
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", str(remote), str(other))
    (other / "bot/modules/first.py").write_text('def respond(text, memory):\n    return None\n')
    first = submit(world, other, author="pine", branch="first")
    real, calls = world.repo.hub.open, []

    def open_after_a_merge(title, description):
        if not calls:  # main moves outside the world after its check, before the Hub PR exists
            calls.append(world.repo.hub.merge(first["pr"], "merged outside the world"))
        return real(title, description)

    monkeypatch.setattr(world.repo.hub, "open", open_after_a_merge)
    (clone / "bot/modules/second.py").write_text('def respond(text, memory):\n    return None\n')
    rec = submit(world, clone, author="birch", branch="second")
    assert rec["pr"] == world.repo.hub.n == first["pr"] + 1 and calls
    assert not getattr(world.repo.hub, "closed", [])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == rec["head_sha"]


def test_refusals_are_logged_and_request_keys_are_bounded(setup, monkeypatch):
    import chatbot4

    world, clone, _ = setup
    rec = submit(world, clone)
    with pytest.raises(ToolError, match="self-review"):
        world.call("review_pr", {"pr": rec["pr"], "head_sha": rec["head_sha"], "verdict": "approve",
                                 "comment": "mine"}, Agent("cedar"))
    with pytest.raises(ToolError, match="approve"):
        merge(world, Agent("birch"), rec["pr"])
    events = [json.loads(line) for line in (world.ws / "prs.jsonl").read_text().splitlines()]
    refused = [(e["event"], e["agent"], e["pr"], e["code"]) for e in events if e["event"].endswith("_refused")]
    assert refused == [("review_pr_refused", "cedar", rec["pr"], "tool_error"),
                       ("merge_pr_refused", "birch", rec["pr"], "tool_error")]
    from flock.world import tool_schema

    schema = tool_schema(world.tools()["merge_pr"])
    assert list(schema["parameters"]["properties"]) == ["pr"]  # the wrapper keeps the schema
    monkeypatch.setattr(chatbot4, "MAX_REQUESTS", 2)
    for i in range(3):
        world.call_workspace("open_pr", {"branch": "greeting", "title": "Bounded keys",
                                         "description": "x", "pr": rec["pr"]},
                             Agent("cedar", request_id=f"req-{i}"), capture(clone.parent, clone.name))
    assert list(world.state["requests"]) == ["req-1", "req-2"]


def test_hub_failure_logs_status_and_headers_but_no_secrets(monkeypatch, caplog):
    from huggingface_hub.errors import HfHubHTTPError
    from repository import Hub

    secret = "hf_SECRETTOKEN1234567890"
    monkeypatch.setenv("HF_TOKEN", secret)
    request = httpx.Request("POST", "https://huggingface.co/api/datasets/o/r/discussions?token=" + secret,
                            headers={"Authorization": f"Bearer {secret}"})
    response = httpx.Response(403, request=request, text=f"body with {secret}",
                              headers={"x-error-message": "Forbidden: no write access",
                                       "x-request-id": "req-1"})
    hub = Hub("o/r")

    def fail(*args, **kwargs):
        raise HfHubHTTPError(f"403 {secret}", response=response)

    monkeypatch.setattr(hub.api, "create_pull_request", fail)
    with caplog.at_level("WARNING"), pytest.raises(ToolError, match="Hub request failed"):
        hub.open("t", "d")
    text = caplog.text
    assert "403" in text and "Forbidden: no write access" in text and "req-1" in text
    assert "create_pull_request" in text and "o/r" in text
    assert "/api/datasets/o/r/discussions" in text and "?" not in text.split("request=")[1]
    assert secret not in text and "SECRETTOKEN" not in text


def test_open_pr_branch_prefix_and_specific_errors(setup):
    world, clone, _ = setup
    rec = submit(world, clone, branch=" agents/cedar/greeting ")
    assert rec["branch"] == "agents/cedar/greeting"
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    for form in ("greeting", "agents/cedar/greeting", "refs/heads/agents/cedar/greeting"):
        assert submit(world, clone, branch=form, pr=rec["pr"])["branch"] == rec["branch"]
    with pytest.raises(ToolError, match="prefix must be your own"):
        submit(world, clone, branch="agents/birch/greeting", pr=rec["pr"])
    with pytest.raises(ToolError, match="prefix must be your own"):
        submit(world, clone, branch="refs/heads/agents/birch/other")
    with pytest.raises(ToolError, match="branch slug already used"):
        submit(world, clone, branch="agents/cedar/greeting")
    with pytest.raises(ToolError, match=r"branch must be a short slug matching .*got \"Bad_Slug\""):
        submit(world, clone, branch="Bad_Slug")
    with pytest.raises(ToolError, match="got \"" + "x" * 80 + "\"$"):
        submit(world, clone, branch="x" * 200 + "!")
    for bad, msg in (("hi", r"title must be 3–200 characters \(got 2\)"),
                     ("t" * 201, r"\(got 201\)")):
        with pytest.raises(ToolError, match=msg):
            world.call_workspace("open_pr", {"branch": "other", "title": bad, "description": "d"},
                                 Agent("cedar"), capture(clone.parent, clone.name))
    with pytest.raises(ToolError, match="description must not be empty"):
        world.call_workspace("open_pr", {"branch": "other", "title": "Fine title", "description": "  "},
                             Agent("cedar"), capture(clone.parent, clone.name))


def _module(clone, name, reply):
    (clone / f"bot/modules/{name}.py").write_text(f'def respond(text, memory):\n    return {reply!r}\n')


def _commit(clone, message="change"):
    git(clone, "add", "-A")
    git(clone, "-c", "user.name=a", "-c", "user.email=a@flock.invalid", "commit", "-qm", message)


def _rebase(clone):
    git(clone, "fetch", "-q", "origin")
    git(clone, "-c", "user.name=a", "-c", "user.email=a@flock.invalid", "rebase", "-q", "origin/main")


def _two_prs(setup, edit_one, edit_two):
    """pine's PR merges first; cedar's PR (from the fixture clone) is approved, on the old main."""
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", "-q", str(remote), str(other))
    edit_one(other)
    first = submit(world, other, author="pine", branch="one")
    edit_two(clone)
    _commit(clone)
    second = submit(world, clone, branch="two")
    approve(world, first)
    world.review_pr(Agent("birch"), second["pr"], second["head_sha"], "approve", "Looks right")
    merge(world, Agent("pine"), first["pr"])
    return world, clone, second


def test_approval_carries_over_a_clean_rebase(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == []
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert updated["head_sha"] != second["head_sha"]
    review = world.state["prs"][str(second["pr"])]["reviews"]["birch"]
    assert review["head_sha"] == second["head_sha"] and review["patch_id"]
    assert world.list_prs(needs_review=True) == []  # the approval carried over
    result = merge(world, Agent("cedar"), second["pr"])
    assert result["reviewers"] == ["birch"] and result["head_sha"] == updated["head_sha"]


def test_rebase_that_changes_the_diff_needs_a_fresh_approval(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "greeting", "Hey!"),
                                    lambda c: _module(c, "greeting", "Hi!"))
    git(clone, "fetch", "-q", "origin")
    git(clone, "reset", "-q", "--hard", "origin/main")  # resolve the conflict by redoing the edit
    _module(clone, "greeting", "Hi!")
    _commit(clone, "resolved")
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated)
    assert merge(world, Agent("cedar"), second["pr"])["head_sha"] == updated["head_sha"]


def test_change_request_still_blocks_after_a_clean_rebase(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    world.review_pr(Agent("oak"), second["pr"], second["head_sha"], "request_changes", "Add a test")
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    with pytest.raises(ToolError, match="requests changes"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated, "oak")
    assert merge(world, Agent("cedar"), second["pr"])["reviewers"] == ["birch", "oak"]


def test_reviews_without_patch_id_only_count_for_their_exact_head(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    # State archived before patch ids existed, restored into a restarted world.
    world.flush()
    state = json.loads(world.db.read_text())
    del state["prs"][str(second["pr"])]["reviews"]["birch"]["patch_id"]
    del state["prs"][str(second["pr"])]["reviews"]["birch"]["patch_id0"]
    _write_archive(world, state)
    world = Chatbot4World(world.ws, world.repo)
    assert world.list_prs(needs_review=True) == []  # still approved at its exact head
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated)
    assert merge(world, Agent("cedar"), second["pr"])["merged_sha"]


HOTSPOT = [f"LINE_{i} = {i}\n" for i in range(1, 10)]


def _hotspot(setup):
    """Put a nine-line module on main, as a file many PRs edit."""
    world, clone, _ = setup
    (clone / "bot/modules/hotspot.py").write_text("".join(HOTSPOT))
    _commit(clone, "hotspot")
    git(clone, "push", "-q", "origin", "HEAD:main")
    world.repo.fetch()


def _edit_line(clone, number, value):
    lines = (clone / "bot/modules/hotspot.py").read_text().splitlines(keepends=True)
    lines[number - 1] = f"LINE_{number} = {value!r}\n"
    (clone / "bot/modules/hotspot.py").write_text("".join(lines))


def _adjacent_prs(setup):
    """main changes line 3 while cedar's approved PR changes line 5 (inside the diff context)."""
    _hotspot(setup)
    return _two_prs(setup, lambda c: _edit_line(c, 3, "main"),
                    lambda c: _edit_line(c, 5, "pr"))


def test_approval_carries_over_a_rebase_next_to_a_change_of_main(setup):
    world, clone, second = _adjacent_prs(setup)
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    rec = world.state["prs"][str(second["pr"])]
    review = rec["reviews"]["birch"]
    assert review["head_sha"] == second["head_sha"] and review["patch_id0"]
    # The diff with context changed (main's line 3 shows in it); the change itself did not.
    assert review["patch_id"] != world.repo.patch_id(rec["base_sha"], rec["head_sha"])
    assert review["patch_id0"] == world.repo.patch_id0(rec["base_sha"], rec["head_sha"])
    assert world.list_prs(needs_review=True) == []
    result = merge(world, Agent("cedar"), second["pr"])
    assert result["reviewers"] == ["birch"] and result["head_sha"] == updated["head_sha"]


def test_change_request_carries_over_a_rebase_next_to_a_change_of_main(setup):
    world, clone, second = _adjacent_prs(setup)
    world.review_pr(Agent("oak"), second["pr"], second["head_sha"], "request_changes", "Add a test")
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    with pytest.raises(ToolError, match="requests changes"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated, "oak")
    assert merge(world, Agent("cedar"), second["pr"])["reviewers"] == ["birch", "oak"]


def test_a_content_change_after_an_adjacent_rebase_needs_a_fresh_approval(setup):
    world, clone, second = _adjacent_prs(setup)
    _rebase(clone)
    _edit_line(clone, 5, "changed")
    _commit(clone, "different change")
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated)
    assert merge(world, Agent("cedar"), second["pr"])["head_sha"] == updated["head_sha"]


def _drop_patch_id0(world, pr):
    """A review stored before patch_id0 existed, restored into a restarted world."""
    world.flush()
    state = json.loads(world.db.read_text())
    del state["prs"][str(pr)]["reviews"]["birch"]["patch_id0"]
    _write_archive(world, state)
    return Chatbot4World(world.ws, world.repo)


def test_old_reviews_carry_by_their_patch_id_only(setup):
    # Away from main's change, the old formula still matches and the approval carries.
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    world = _drop_patch_id0(world, second["pr"])
    _rebase(clone)
    submit(world, clone, branch="two", pr=second["pr"])
    assert world.list_prs(needs_review=True) == []
    assert merge(world, Agent("cedar"), second["pr"])["reviewers"] == ["birch"]


def test_old_reviews_do_not_carry_a_rebase_next_to_a_change_of_main(setup):
    # The base a review was made on is not stored, so patch_id0 is not backfilled.
    world, clone, second = _adjacent_prs(setup)
    world = _drop_patch_id0(world, second["pr"])
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        merge(world, Agent("cedar"), second["pr"])
    approve(world, updated)
    assert merge(world, Agent("cedar"), second["pr"])["merged_sha"]


def test_merge_pr_rebases_a_stale_pr_onto_main_keeping_its_approvals(setup):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    remote, main = setup[2], world.repo.sha()
    result = merge(world, Agent("cedar"), second["pr"])
    head = result["head_sha"]
    assert result["rebased_from"] == second["base_sha"] and result["reviewers"] == ["birch"]
    assert result["parent_sha"] == main == git(remote, "rev-parse", f"{head}^")
    assert git(remote, "rev-parse", f"refs/pr/{second['pr']}") == head
    assert git(remote, "log", "-1", "--format=%an|%s", head) == "cedar|Improve greetings"
    assert {"time.py", "weather.py"} <= set(git(remote, "ls-tree", "--name-only", "main:bot/modules").split())
    rec = world.state["prs"][str(second["pr"])]
    assert (rec["state"], rec["base_sha"], rec["head_sha"]) == ("merged", main, head)
    queue = json.loads((world.ws / "eval-queue.jsonl").read_text().splitlines()[-1])
    assert (queue["parent_sha"], queue["head_sha"], queue["merged_sha"]) == (main, head, result["merged_sha"])


def test_merge_pr_refuses_a_stale_pr_that_conflicts_with_main(setup):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "greeting", "Hey!"),
                                    lambda c: _module(c, "greeting", "Hi!"))
    remote, main, pr = setup[2], world.repo.sha(), second["pr"]
    with pytest.raises(ToolError, match=r"conflicts with main \w+ in bot/modules/greeting.py:") as err:
        merge(world, Agent("oak"), pr)
    assert err.value.code == "conflict" and git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", f"refs/pr/{pr}") == second["head_sha"]
    assert world.state["prs"][str(pr)]["head_sha"] == second["head_sha"]
    assert not world.state["merge_queue"]  # refused: off the queue, the author and requester told
    reason = err.value.message
    assert world.call("list_prs", {}, Agent("cedar"))["inbox"][-1] == (
        f"your PR #{pr} was not merged: {reason} (just now)")
    assert world.call("list_prs", {}, Agent("oak"))["inbox"] == [
        f"PR #{pr} you queued was not merged: {reason} (just now)"]
    with pytest.raises(ToolError, match="conflicts with main") as again:
        world.merge_pr(Agent("pine"), pr)  # the same head on the same main: answered at once
    assert again.value.code == "conflict" and not world.state["merge_queue"]
    assert not world.state["inbox"].get("cedar")
    third = setup[1].parent / "third"  # main moves: queued and checked again
    git(third.parent, "clone", "-q", str(remote), str(third))
    _module(third, "extra", None)
    _commit(third)
    git(third, "push", "-q", "origin", "HEAD:main")
    world.repo.fetch()
    with pytest.raises(ToolError, match="conflicts with main"):
        merge(world, Agent("pine"), pr)
    assert [e["kind"] for e in world.state["inbox"]["cedar"]] == ["merge"]


def test_a_pr_fixed_during_its_refused_checks_is_queued_again(setup, monkeypatch):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "greeting", "Hey!"),
                                    lambda c: _module(c, "greeting", "Hi!"))
    pr, real, fixed = second["pr"], world._check_merge, []

    def check_while_the_author_fixes_it(*args):  # the author resolves it on main meanwhile
        git(clone, "fetch", "-q", "origin")
        git(clone, "reset", "-q", "--hard", "origin/main")
        _module(clone, "greeting", "Hi!")
        _commit(clone, "fixed")
        fixed.append(submit(world, clone, branch="two", pr=pr))
        return real(*args)

    monkeypatch.setattr(world, "_check_merge", check_while_the_author_fixes_it)
    world.merge_pr(Agent("oak"), pr)
    with pytest.raises(ToolError, match="conflicts with main"):
        world._merge_next()  # the old head conflicts
    monkeypatch.setattr(world, "_check_merge", real)
    approve(world, fixed[0])
    assert world.merge_pr(Agent("oak"), pr)["queued"]  # the refusal was about the old head
    assert world._merge_next()["head_sha"] == fixed[0]["head_sha"]


def test_merge_pr_refuses_a_rebase_that_changes_the_diff(setup):
    _hotspot(setup)

    def both(clone):  # main makes the same edit of line 3: the rebased diff is line 5 only
        _edit_line(clone, 3, "same")
        _edit_line(clone, 5, "pr")

    world, _clone, second = _two_prs(setup, lambda c: _edit_line(c, 3, "same"), both)
    remote, main = setup[2], world.repo.sha()
    with pytest.raises(ToolError, match="changes its diff, so it needs a fresh review") as err:
        merge(world, Agent("cedar"), second["pr"])
    assert err.value.code == "rebase_changed" and git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", f"refs/pr/{second['pr']}") == second["head_sha"]
    assert not world.state["merge_queue"]  # dropped, and its author (who queued it) told why
    assert world.state["inbox"]["cedar"][-1]["text"] == f"your PR #{second['pr']} was not merged: {err.value.message}"


def _deck_file(clone, name, deck_id):
    (clone / f"deck/{name}.jsonl").write_text(
        json.dumps({"id": deck_id, "area": "test", "turns": [f"hello from {name}"]}) + "\n")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_merge_pr_refuses_a_rebase_that_breaks_the_shared_tests(setup_v8):
    # Each PR adds a deck file with the same new id: each passes alone, together they fail.
    world, clone, second = _two_prs(setup_v8, lambda c: _deck_file(c, "one", "dup-1"),
                                    lambda c: _deck_file(c, "two", "dup-1"))
    remote, main = setup_v8[2], world.repo.sha()
    with pytest.raises(ToolError, match=r"shared tests \(tests/test_deck.py\).*'dup-1'") as err:
        merge(world, Agent("cedar"), second["pr"])
    assert err.value.code == "shared_tests" and git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", f"refs/pr/{second['pr']}") == second["head_sha"]
    assert not world.state["merge_queue"]
    assert world.state["inbox"]["cedar"][-1]["text"].startswith(f"your PR #{second['pr']} was not merged: the shared tests")
    # Once main itself fails them (a push outside the world), they no longer refuse a PR.
    third = clone.parent / "third"
    git(clone.parent, "clone", "-q", str(remote), str(third))
    _deck_file(third, "three", "dup-1")
    _commit(third, "red main")
    git(third, "push", "-q", "origin", "HEAD:main")
    world.repo.fetch()
    assert merge(world, Agent("cedar"), second["pr"])["merged_sha"] == git(remote, "rev-parse", "main")


def test_a_pr_updated_while_queued_is_checked_again_at_its_turn(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    assert world.merge_pr(Agent("birch"), rec["pr"])["position"] == 1
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hey!"\n')
    updated = submit(world, clone, pr=rec["pr"])  # a new change: birch's approval does not carry
    with pytest.raises(ToolError, match="another agent must approve the current PR head"):
        world._merge_next()
    assert not world.state["merge_queue"]
    assert world.state["inbox"]["cedar"][-1]["text"].startswith(
        f"your PR #{rec['pr']} was not merged: another agent must approve")
    approve(world, updated)
    assert merge(world, Agent("birch"), rec["pr"])["head_sha"] == updated["head_sha"]


def test_the_merge_queue_survives_a_restart_and_its_thread_merges_it(setup, monkeypatch):
    world, clone, remote = setup
    rec = submit(world, clone)
    approve(world, rec)
    assert world.merge_pr(Agent("birch"), rec["pr"])["queued"]
    world.admin.pause(["merge"])  # kept across the restart (knobs.json)
    world.close()
    real_merge, calls = world.repo.hub.merge, []

    def broken_once(pr, comment):  # not a ToolError: the thread logs it and retries
        calls.append(pr)
        if len(calls) == 1:
            raise RuntimeError("a bug")
        return real_merge(pr, comment)

    monkeypatch.setattr(world.repo.hub, "merge", broken_once)
    monkeypatch.setattr(chatbot4, "MERGE_BACKOFF", (0.2, 1.0))
    monkeypatch.setattr(Chatbot4World, "MERGER", True)
    world = Chatbot4World(world.ws, world.repo)  # the queue restored; its thread runs it
    try:
        time.sleep(1)
        assert not calls and world.merger.is_alive()  # paused: it waits
        world.admin.resume(["merge"])
        world.merge_wake.set()
        deadline = time.monotonic() + 30
        while world.state["merge_queue"] and time.monotonic() < deadline:
            time.sleep(0.1)
        assert calls == [rec["pr"]] * 2 and world.merger.is_alive()
        out = world.call("list_prs", {"state": "merged"}, Agent("birch"))
        assert [r["pr"] for r in out["prs"]] == [rec["pr"]]
        assert out["prs"][0]["merged_sha"] == git(remote, "rev-parse", "main")
        assert f"PR #{rec['pr']} you queued is merged (just now)" in out["inbox"]
    finally:
        world.close()
    world.merger.join(10)
    assert not world.merger.is_alive()  # close() stops it


def test_a_hub_merge_failure_after_the_rebase_keeps_the_record_on_the_rebased_head(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    remote, main, real = setup[2], world.repo.sha(), world.repo.hub.merge

    def fail(pr, comment):
        raise ToolError("Hub request failed; retry later", "hub_error")

    monkeypatch.setattr(world.repo.hub, "merge", fail)
    with pytest.raises(ToolError, match="Hub request failed"):
        merge(world, Agent("cedar"), second["pr"])
    head = git(remote, "rev-parse", f"refs/pr/{second['pr']}")
    rec = world.state["prs"][str(second["pr"])]
    assert head != second["head_sha"] and git(remote, "rev-parse", f"{head}^") == main
    assert (rec["state"], rec["base_sha"], rec["head_sha"]) == ("open", main, head)
    assert git(remote, "rev-parse", "main") == main
    monkeypatch.setattr(world.repo.hub, "merge", real)
    world = Chatbot4World(world.ws, world.repo)  # restart: the saved record has the rebased head
    assert world.list_prs(needs_review=True) == []  # the approval still counts
    result = merge(world, Agent("cedar"), second["pr"])
    assert "rebased_from" not in result and (result["parent_sha"], result["head_sha"]) == (main, head)


def _push_to_main(setup, name):
    """A change of main from outside the world (another clone pushes it)."""
    _world, clone, remote = setup
    outside = clone.parent / f"outside-{name}"
    git(clone.parent, "clone", "-q", str(remote), str(outside))
    _module(outside, name, None)
    _commit(outside)
    git(outside, "push", "-q", "origin", "HEAD:main")


def test_main_moving_during_the_checks_refuses_and_touches_nothing(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                     lambda c: _module(c, "time", None))
    remote, pr, real = setup[2], second["pr"], world._shared_tests

    def main_moves(*args):
        _push_to_main(setup, "extra")
        world.repo.fetch()  # another tool's refresh
        return real(*args)

    monkeypatch.setattr(world, "_shared_tests", main_moves)
    with pytest.raises(ToolError, match="changed during checks"):
        merge(world, Agent("oak"), pr)
    assert git(remote, "rev-parse", f"refs/pr/{pr}") == world.state["prs"][str(pr)]["head_sha"] == second["head_sha"]


def test_a_pr_updated_during_the_checks_refuses_and_keeps_the_update(setup, monkeypatch):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    remote, pr, real, updated = setup[2], second["pr"], world._shared_tests, {}
    main = world.repo.sha()

    def author_updates(*args):
        _module(clone, "time", "Noon")
        updated.update(submit(world, clone, branch="two", pr=pr))
        return real(*args)

    monkeypatch.setattr(world, "_shared_tests", author_updates)
    with pytest.raises(ToolError, match="changed during checks"):
        merge(world, Agent("oak"), pr)
    assert git(remote, "rev-parse", f"refs/pr/{pr}") == world.state["prs"][str(pr)]["head_sha"] == updated["head_sha"]
    assert git(remote, "rev-parse", "main") == main


def test_a_stale_lease_after_a_restart_adopts_the_hub_head(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                     lambda c: _module(c, "time", None))
    remote, pr, real = setup[2], second["pr"], world.repo.hub.merge

    def fail(pr, comment):
        raise ToolError("Hub request failed; retry later", "hub_error")

    monkeypatch.setattr(world.repo.hub, "merge", fail)
    with pytest.raises(ToolError):
        merge(world, Agent("oak"), pr)
    assert git(remote, "rev-parse", f"refs/pr/{pr}") != second["head_sha"]
    # A crash after the force-push, before the record was saved: record and local ref are older.
    with world.lock:
        world.state["prs"][str(pr)].update(base_sha=second["base_sha"], head_sha=second["head_sha"])
        world._dirty("prs", pr)
    world.flush()
    world.repo.update_refs((f"refs/pr/{pr}", second["head_sha"]))
    monkeypatch.setattr(world.repo.hub, "merge", real)
    world = Chatbot4World(world.ws, world.repo)
    time.sleep(1.1)  # the next replay gets a new committer date, so a new sha
    result = merge(world, Agent("oak"), pr)
    assert result["head_sha"] == git(remote, "rev-parse", f"refs/pr/{pr}")
    assert result["merged_sha"] == git(remote, "rev-parse", "main")
    assert world.metrics()["stale_lease_adoptions_total"] == 1


def test_a_hub_merge_that_lands_then_errors_is_seen_by_the_next_merge(setup, monkeypatch):
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", "-q", str(remote), str(other))
    _module(other, "weather", None)
    a = submit(world, other, "pine", "a")
    _module(clone, "time", None)
    b = submit(world, clone, "cedar", "b")
    approve(world, a)
    approve(world, b)
    real_merge = world.repo.hub.merge

    def merge_then_fail(pr, comment):
        real_merge(pr, comment)
        if pr == a["pr"]:  # landed on the Hub, but the call fails (not a ToolError)
            raise ConnectionError("reset by peer")

    monkeypatch.setattr(world.repo.hub, "merge", merge_then_fail)
    world.merge_pr(Agent("oak"), a["pr"])
    world.merge_pr(Agent("oak"), b["pr"])
    out = world._merge_next()  # the failed call's fetch finds A in main
    assert out["already_merged"] and out["pr"] == a["pr"]
    out = world._merge_next()
    landed = git(remote, "rev-parse", "main^1")  # A's merge on the Hub, B's parent
    assert out["parent_sha"] == landed and "tree_mismatch" not in out
    assert git(remote, "rev-parse", "main^{tree}") == git(remote, "rev-parse", f"{out['head_sha']}^{{tree}}")


def test_a_transient_merge_failure_keeps_the_pr_queued_while_the_next_goes_ahead(setup, monkeypatch):
    world, clone, remote = setup
    other = clone.parent / "other"
    git(clone.parent, "clone", "-q", str(remote), str(other))
    _module(other, "weather", None)
    a = submit(world, other, "pine", "a")
    _module(clone, "time", None)
    b = submit(world, clone, "cedar", "b")
    approve(world, a)
    approve(world, b)
    real_merge, fails = world.repo.hub.merge, [1]

    def flaky(pr, comment):
        if fails:
            fails.pop()
            raise ToolError("Hub request failed; retry later", "hub_error")
        return real_merge(pr, comment)

    monkeypatch.setattr(world.repo.hub, "merge", flaky)
    world.merge_pr(Agent("oak"), a["pr"])
    world.merge_pr(Agent("oak"), b["pr"])
    queue = world.state["merge_queue"]
    queue[0]["tries"] = 1024  # a long outage: the backoff stays capped (no float overflow)
    with pytest.raises(ToolError, match="Hub request failed"):
        world._merge_next()
    assert [(e["pr"], e["tries"]) for e in queue] == [(a["pr"], 1025), (b["pr"], 0)]
    assert 590 < queue[0]["next_at"] - time.time() <= 600  # A keeps its place and waits
    assert world._merge_next()["pr"] == b["pr"]  # B goes ahead meanwhile; the Hub merges again,
    assert queue[0]["next_at"] == 0.0  # so A's retry is due at once: it lands on B's main
    assert world._merge_next()["rebased_from"] == a["base_sha"] and not queue
    assert not any(e["kind"] == "merge" for box in world.state["inbox"].values() for e in box)


def test_a_retry_after_a_landed_merge_and_a_failed_fetch_finds_it_merged(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                     lambda c: _module(c, "time", None))
    pr = second["pr"]
    real_merge, real_fetch, hiccup = world.repo.hub.merge, world.repo.fetch, []

    def merge_then_fail(pr_, comment):  # landed, but the call fails, and so does the next fetch
        real_merge(pr_, comment)
        hiccup.append(1)
        raise ToolError("Hub request failed; retry later", "hub_error")

    def fetch():
        if hiccup and hiccup.pop():
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return real_fetch()

    monkeypatch.setattr(world.repo.hub, "merge", merge_then_fail)
    monkeypatch.setattr(world.repo, "fetch", fetch)
    world.merge_pr(Agent("oak"), pr)
    with pytest.raises(ToolError, match="Hub request failed"):
        world._merge_next()
    monkeypatch.setattr(world.repo.hub, "merge", real_merge)
    world.state["merge_queue"][0]["next_at"] = 0.0  # the retry
    assert world._merge_next()["already_merged"] and not world.state["merge_queue"]
    assert world.state["prs"][str(pr)]["state"] == "merged"
    assert [e["kind"] for e in world.state["inbox"]["cedar"]][-1] == "merged"  # no refusal sent
    lines = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert lines[-1]["pr"] == pr and lines[-1]["reconciled"]


def test_merge_queue_logs_one_timing_line_per_run(setup, caplog):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                     lambda c: _module(c, "time", None))
    caplog.set_level("INFO", logger="chatbot4")
    merge(world, Agent("oak"), second["pr"])
    assert merge(world, Agent("oak"), second["pr"])["already_merged"]  # answered, never queued
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("merge_pr ")]
    assert len(lines) == 1 and lines[0].startswith(f"merge_pr pr={second['pr']} outcome=merged total=")
    steps = dict(item.split("=") for item in lines[0].split()[3:])
    assert set(steps) == {"total", "lane_wait", "refresh", "checkout",
                          "replay", "approvals", "contract", "shared_tests", "main_lock_wait",
                          "push", "update_ref", "hub_merge", "fetch", "tree_check"}
    assert all(float(v) >= 0 for v in steps.values())


def _bundle(setup, *replies):
    """Bundle mode; one approved PR per (module, reply), each from its own clone and author,
    queued in that order."""
    world, clone, remote = setup
    world.bundle = 5
    recs = []
    for i, (name, reply) in enumerate(replies):
        work = clone.parent / f"bundle-{i}"
        git(clone.parent, "clone", "-q", str(remote), str(work))
        _module(work, name, reply)
        recs.append(submit(world, work, author=f"author{i}", branch=f"b{i}"))
        approve(world, recs[-1])
        world.merge_pr(Agent("oak"), recs[-1]["pr"])
    return world, remote, recs


def test_a_bundle_lands_its_prs_as_one_commit_each_in_one_push(setup, monkeypatch, caplog):
    world, remote, recs = _bundle(setup, ("one", None), ("two", None), ("three", None))
    main, pushes, real = world.repo.sha(), [], world.repo.push
    monkeypatch.setattr(world.repo, "push", lambda sha, ref, **kw: pushes.append(ref) or real(sha, ref, **kw))
    caplog.set_level("INFO", logger="chatbot4")
    first = world._merge_next()
    assert pushes == ["refs/heads/main"] and not world.state["merge_queue"]
    shas = git(remote, "rev-list", "--reverse", f"{main}..main").split()
    parents = [main, *shas[:2]]
    assert len(shas) == 3 and first["merged_sha"] == shas[0] and world.repo.sha() == shas[-1]
    for rec, sha in zip(recs, shas):
        assert world.state["prs"][str(rec["pr"])]["state"] == "merged"
        assert git(remote, "log", "-1", "--format=%an|%ce|%s", sha) == (
            f"{rec['author']}|{chatbot4.LANDER[1]}|PR #{rec['pr']} by {rec['author']}; approved by birch")
        assert any(n["kind"] == "merged" for n in world.state["inbox"][rec["author"]])
    queue = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert [(q["pr"], q["parent_sha"], q["merged_sha"], q["bundle"], q["bundle_tip"]) for q in queue] == [
        (r["pr"], parent, sha, 3, shas[-1]) for r, parent, sha in zip(recs, parents, shas)]
    [line] = [r.getMessage() for r in caplog.records if r.getMessage().startswith("merge_pr ")]
    assert line.startswith(f"merge_pr bundle=3 prs={','.join(str(r['pr']) for r in recs)} outcome=merged,merged,merged total=")
    hub = world.repo.hub
    assert _wait(lambda: len(getattr(hub, "closed", [])) == 3)
    assert hub.close_comments == {r["pr"]: f"Landed in {s} (bundle of 3)" for r, s in zip(recs, shas)}


def test_a_bundle_refuses_a_conflicting_pr_as_today_and_lands_the_others(setup):
    world, remote, recs = _bundle(setup, ("same", "a"), ("same", "b"), ("other", None))
    world._merge_next()
    states = [world.state["prs"][str(r["pr"])]["state"] for r in recs]
    assert states == ["merged", "open", "merged"] and not world.state["merge_queue"]
    landed = world.state["prs"][str(recs[0]["pr"])]["merged_sha"]
    assert world.state["prs"][str(recs[2]["pr"])]["merged_sha"] == git(remote, "rev-parse", "main")
    [news] = [n["text"] for n in world.state["inbox"]["author1"] if n["kind"] == "merge"]
    assert news.startswith(f"your PR #{recs[1]['pr']} was not merged: PR #{recs[1]['pr']} conflicts "
                           f"with main {landed[:12]} in bot/modules/same.py")
    assert world.merge_refusals[-1]["code"] == "conflict"


def test_a_bundle_failing_its_checks_falls_back_to_one_merge_per_pr(setup, monkeypatch):
    world, remote, recs = _bundle(setup, ("one", None), ("bad", None), ("three", None))
    real = world._shared_tests

    def shared_tests(pr, tree, main):
        if (tree / "bot/modules/bad.py").exists():
            raise ToolError("the shared tests fail", "shared_tests")
        return real(pr, tree, main)

    monkeypatch.setattr(world, "_shared_tests", shared_tests)
    world._merge_next()
    states = [world.state["prs"][str(r["pr"])]["state"] for r in recs]
    assert states == ["merged", "open", "merged"] and not world.state["merge_queue"]
    assert world.merge_refusals[-1] == {**world.merge_refusals[-1], "pr": recs[1]["pr"], "code": "shared_tests"}
    assert len(git(remote, "rev-list", "--parents", "-n", "1", "main").split()) == 3  # Hub merges


def test_a_bundle_lands_nothing_when_main_moves_before_its_push(setup, monkeypatch):
    world, remote, recs = _bundle(setup, ("one", None), ("two", None))
    real, moved = world._shared_tests, []

    def main_moves(pr, tree, main):
        if not moved:
            _push_to_main(setup, "extra")
            moved.append(git(remote, "rev-parse", "main"))
        return real(pr, tree, main)

    monkeypatch.setattr(world, "_shared_tests", main_moves)
    with pytest.raises(ToolError) as refused:
        world._merge_next()
    assert refused.value.code == "stale_lease" and git(remote, "rev-parse", "main") == moved[0]
    assert [world.state["prs"][str(r["pr"])]["state"] for r in recs] == ["open", "open"]
    assert [e["tries"] for e in world.state["merge_queue"]] == [1, 1]
    for entry in world.state["merge_queue"]:
        entry["next_at"] = 0.0
    world._merge_next()  # retried from the start on the new main
    assert git(remote, "rev-list", "--count", f"{moved[0]}..main") == "2" and not world.state["merge_queue"]


def test_a_bundle_of_one_failing_its_checks_keeps_no_refusal_against_its_own_commit(setup, monkeypatch):
    world, _remote, recs = _bundle(setup, ("same", "a"), ("same", "b"))
    real = world._shared_tests

    def shared_tests(pr, tree, main):
        if "'a'" in (tree / "bot/modules/same.py").read_text():
            raise ToolError("the shared tests fail", "shared_tests")
        return real(pr, tree, main)

    monkeypatch.setattr(world, "_shared_tests", shared_tests)
    with pytest.raises(ToolError, match="the shared tests fail"):
        world._merge_next()  # A refused; B, which conflicted only with A, goes the single-PR path
    assert [world.state["prs"][str(r["pr"])]["state"] for r in recs] == ["open", "merged"]
    assert [(r["pr"], r["code"]) for r in world.merge_refusals] == [(recs[0]["pr"], "shared_tests")]


@pytest.mark.parametrize("change", ["request_changes", "close"])
def test_a_bundle_lands_nothing_when_a_pr_changes_before_its_push(setup, monkeypatch, change):
    world, remote, recs = _bundle(setup, ("one", None), ("two", None))
    main, real = world.repo.sha(), world._shared_tests

    def changes(pr, tree, main):
        if change == "close":
            world.close_pr(Agent(recs[1]["author"]), recs[1]["pr"], "obsolete")
        else:
            world.review_pr(Agent("pine"), recs[1]["pr"], recs[1]["head_sha"], "request_changes", "Wrong reply")
        return real(pr, tree, main)

    monkeypatch.setattr(world, "_shared_tests", changes)
    with pytest.raises(ToolError, match="changed during checks"):
        world._merge_next()
    assert git(remote, "rev-parse", "main") == main
    assert world.state["prs"][str(recs[0]["pr"])]["state"] == "open"
    assert [e["tries"] for e in world.state["merge_queue"]] == [1, 1]


def test_a_bundle_push_that_lands_then_fails_is_reconciled(setup, monkeypatch):
    world, remote, recs = _bundle(setup, ("one", None), ("two", None))
    real_push, real_fetch, hiccup = world.repo.push, world.repo.fetch, []

    def push_then_time_out(sha, ref, **kw):
        real_push(sha, ref, **kw)
        hiccup.append(1)
        raise subprocess.TimeoutExpired("git push", 120)

    def fetch():  # the "did it land?" fetch fails too
        if hiccup and hiccup.pop():
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return real_fetch()

    monkeypatch.setattr(world.repo, "push", push_then_time_out)
    monkeypatch.setattr(world.repo, "fetch", fetch)
    with pytest.raises(subprocess.TimeoutExpired):
        world._merge_next()
    assert [world.state["prs"][str(r["pr"])]["state"] for r in recs] == ["open", "open"]
    for entry in world.state["merge_queue"]:
        entry["next_at"] = 0.0
    assert world._merge_next()["already_merged"] and not world.state["merge_queue"]
    merged = [world.state["prs"][str(r["pr"])]["merged_sha"] for r in recs]
    assert [world.state["prs"][str(r["pr"])]["state"] for r in recs] == ["merged", "merged"]
    assert merged[1] == git(remote, "rev-parse", "main") and merged[0] == git(remote, "rev-parse", "main^")
    queue = [json.loads(line) for line in (world.ws / "eval-queue.jsonl").read_text().splitlines()]
    assert [(q["pr"], q["merged_sha"], q["reconciled"]) for q in queue] == [
        (r["pr"], sha, True) for r, sha in zip(recs, merged)]


def test_bundle_merges_stay_merged_through_hub_closed_states_and_a_lost_state_write(setup):
    world, _remote, recs = _bundle(setup, ("one", None), ("two", None))
    world.flush()
    before = (world.ws / "prs.json").read_text()  # the state as a crash after the push leaves it
    world._merge_next()
    prs = [r["pr"] for r in recs]
    merged = {pr: world.state["prs"][str(pr)]["merged_sha"] for pr in prs}
    world.repo.hub.pr_states = lambda: dict.fromkeys(prs, "closed")
    world.hub_states_at = float("-inf")
    world._sync_hub_states()
    assert {pr: world.state["prs"][str(pr)]["state"] for pr in prs} == dict.fromkeys(prs, "merged")
    world.close()
    _write_archive(world, json.loads(before))
    restarted = Chatbot4World(world.ws, world.repo)
    assert len(restarted.state["merge_queue"]) == 2  # the old state, restored
    assert {pr: (restarted.state["prs"][str(pr)]["state"], restarted.state["prs"][str(pr)]["merged_sha"])
            for pr in prs} == {pr: ("merged", sha) for pr, sha in merged.items()}
    assert restarted._merge_next()["already_merged"]  # the queue left in the old state is answered


def test_temporary_checkouts_go_to_chatbot_tmp_dir(setup, tmp_path, monkeypatch):
    world, clone, remote = setup
    monkeypatch.setenv("CHATBOT_TMP_DIR", str(tmp_path / "local-tmp"))
    world = Chatbot4World(world.ws, world.repo)
    real, dests = world.repo.checkout, []
    monkeypatch.setattr(world.repo, "checkout", lambda ref, dest: (dests.append(dest), real(ref, dest))[1])
    world, _clone, second = _two_prs((world, clone, remote), lambda c: _module(c, "weather", None),
                                     lambda c: _module(c, "time", None))
    assert merge(world, Agent("oak"), second["pr"])["rebased_from"] == second["base_sha"]
    assert world.run_bot("main", ["hi"])["ok"]
    assert len(dests) >= 5 and all((tmp_path / "local-tmp") in d.parents for d in dests)
    assert list((tmp_path / "local-tmp").iterdir()) == []  # each one removed after use


def test_waiting_fetches_reuse_a_fetch_that_started_after_them(setup, monkeypatch):
    import threading

    world, _clone, _remote = setup
    repo, real, calls = world.repo, world.repo.hub_fetch, []
    first_in, release = threading.Event(), threading.Event()

    def hub_fetch(*args, **kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            first_in.set()
            assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(repo, "hub_fetch", hub_fetch)
    threads = [threading.Thread(target=repo.fetch, name=f"t{i}") for i in range(6)]
    threads[0].start()
    assert first_in.wait(10)
    for t in threads[1:]:  # they ask while the first fetch runs: it may predate their question
        t.start()
    time.sleep(0.3)
    release.set()
    for t in threads:
        t.join(10)
    assert len(calls) == 2  # the running one, then one for all who waited
    repo.fetch()  # a later call fetches again
    assert len(calls) == 3


def test_ttl_refreshes_skip_a_running_fetch_and_a_merge_fetch_waits_for_its_own(setup, monkeypatch):
    import threading

    world, _clone, remote = setup
    repo, real, calls = world.repo, world.repo.hub_fetch, []
    first_in, release = threading.Event(), threading.Event()

    def hub_fetch(*args, **kwargs):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            first_in.set()
            assert release.wait(10)
        return real(*args, **kwargs)

    monkeypatch.setattr(repo, "hub_fetch", hub_fetch)
    repo.fetched -= 1000  # every TTL is past
    slow = threading.Thread(target=repo.fetch, name="slow")
    slow.start()
    assert first_in.wait(10)
    _push_to_main(setup, "outside")  # main moves after the running fetch started
    ttl = [threading.Thread(target=repo.refresh, args=(60,), name=f"ttl{i}") for i in range(5)]
    for t in ttl:
        t.start()
    for t in ttl:
        t.join(5)
    assert not any(t.is_alive() for t in ttl)  # they returned while the slow fetch ran
    merge = threading.Thread(target=repo.refresh, args=(0,), name="merge")
    merge.start()
    time.sleep(0.2)
    release.set()
    slow.join(10)
    merge.join(10)
    assert calls == ["slow", "merge"] and repo.sha() == git(remote, "rev-parse", "main")


def test_update_refs_writes_both_refs_at_once(setup):
    world, clone, _remote = setup
    _module(clone, "weather", None)
    rec = submit(world, clone)
    pr, head = rec["pr"], rec["head_sha"]
    assert world.repo.sha(f"pr:{pr}") == head
    assert world.repo.git("rev-parse", f"refs/flock/pr/{pr}/{head}") == head
    main = world.repo.sha()
    world.repo.update_refs((f"refs/pr/{pr}", main), (f"refs/flock/pr/{pr}/{main}", main))
    assert world.repo.sha(f"pr:{pr}") == main == world.repo.git("rev-parse", f"refs/flock/pr/{pr}/{main}")


def _deck_line(clone, deck_id, *turns):
    with open(clone / "deck/multi.jsonl", "a") as out:
        out.write(json.dumps({"id": deck_id, "area": "test", "turns": list(turns)}) + "\n")


@pytest.fixture
def setup_v8_union(tmp_path, monkeypatch):
    """The v8 world with CHATBOT_UNION_PATHS=deck/*.jsonl: deck replays keep both sides' lines."""
    monkeypatch.setenv("CHATBOT_SEED_DIR", "seed-v8")
    monkeypatch.setenv("CHATBOT_UNION_PATHS", "deck/*.jsonl")
    return _setup(tmp_path, monkeypatch, "seed-v8")


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_union_paths_let_two_deck_appends_merge_one_after_the_other(setup_v8_union):
    world, _clone, second = _two_prs(setup_v8_union, lambda c: _deck_line(c, "multi-new-1", "hi", "one"),
                                     lambda c: _deck_line(c, "multi-new-2", "hi", "two"))
    result = merge(world, Agent("oak"), second["pr"])
    assert result["rebased_from"] == second["base_sha"] and result["reviewers"] == ["birch"]
    deck = git(setup_v8_union[2], "show", "main:deck/multi.jsonl").splitlines()
    assert [json.loads(line)["id"] for line in deck[-2:]] == ["multi-new-1", "multi-new-2"]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_union_paths_still_refuse_the_same_deck_id_twice(setup_v8_union):
    world, _clone, second = _two_prs(setup_v8_union, lambda c: _deck_line(c, "multi-dup", "hi", "one"),
                                     lambda c: _deck_line(c, "multi-dup", "hi", "two"))
    remote, main = setup_v8_union[2], world.repo.sha()
    with pytest.raises(ToolError, match="'multi-dup'") as err:
        merge(world, Agent("oak"), second["pr"])
    assert err.value.code == "shared_tests" and git(remote, "rev-parse", "main") == main
    assert git(remote, "rev-parse", f"refs/pr/{second['pr']}") == second["head_sha"]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_union_paths_leave_code_conflicts_refused(setup_v8_union):
    world, _clone, second = _two_prs(setup_v8_union, lambda c: _module(c, "greeting", "Hey!"),
                                     lambda c: _module(c, "greeting", "Hi!"))
    with pytest.raises(ToolError, match="in bot/modules/greeting.py") as err:
        merge(world, Agent("oak"), second["pr"])
    assert err.value.code == "conflict"


def _drop_last_deck_line(clone):
    deck = clone / "deck/multi.jsonl"
    deck.write_text("".join(deck.read_text().splitlines(keepends=True)[:-1]))


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_union_paths_only_at_merge_where_a_kept_deleted_line_is_refused(setup_v8_union):
    # main deletes the deck's last line; the PR, on the older main, appends after it.
    world, clone, second = _two_prs(setup_v8_union, _drop_last_deck_line,
                                    lambda c: _deck_line(c, "multi-new-2", "hi", "two"))
    with pytest.raises(ToolError, match="changes its diff") as err:  # union kept the deleted line
        merge(world, Agent("oak"), second["pr"])
    assert err.value.code == "rebase_changed"
    _deck_line(clone, "multi-new-3", "hi", "three")  # an update still on the older main
    with pytest.raises(ToolError, match="conflict with main") as err:
        submit(world, clone, branch="two", pr=second["pr"])
    assert err.value.code == "stale_base"


def test_a_pr_may_not_add_or_change_gitattributes_but_may_delete_one(setup):
    world, clone, remote = setup
    for path in (".gitattributes", "bot/modules/.gitattributes"):
        (clone / path).write_text("* merge=union\n")
        with pytest.raises(ToolError, match=r"\.gitattributes is not accepted") as err:
            submit(world, clone, branch="attributes")
        assert err.value.code == "fixed_interface"
        (clone / path).unlink()
    outside = clone.parent / "outside"  # main already has one: a PR may delete it
    git(clone.parent, "clone", "-q", str(remote), str(outside))
    (outside / ".gitattributes").write_text("*.txt text\n")
    _commit(outside)
    git(outside, "push", "-q", "origin", "HEAD:main")
    world.repo.fetch()
    git(clone, "pull", "-q", "origin", "main")
    (clone / ".gitattributes").write_text("* merge=union\n")
    with pytest.raises(ToolError, match="is not accepted"):
        submit(world, clone, branch="attributes")  # changing it is refused too
    (clone / ".gitattributes").unlink()
    _module(clone, "weather", None)
    assert submit(world, clone, branch="attributes")["state"] == "open"


@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_without_union_paths_deck_appends_still_conflict(setup_v8):
    world, _clone, second = _two_prs(setup_v8, lambda c: _deck_line(c, "multi-new-1", "hi", "one"),
                                     lambda c: _deck_line(c, "multi-new-2", "hi", "two"))
    assert world.repo.attributes is None
    with pytest.raises(ToolError, match="in deck/multi.jsonl") as err:
        merge(world, Agent("oak"), second["pr"])
    assert err.value.code == "conflict"


def test_git_maintenance_packs_refs_and_all_loose_objects_without_rewriting_packs(setup, caplog):
    world, clone, _remote = setup
    repo = world.repo
    rec = submit(world, clone)
    _module(clone, "weather", "sunny")
    rec = submit(world, clone, pr=rec["pr"])  # a superseded head keeps its refs/flock/pr ref
    assert repo.git("config", "maintenance.auto") == "false"  # fetches start no background gc
    assert repo.git("config", "gc.auto") == "0"  # no git command starts gc --auto either
    orphans = [repo.git("hash-object", "-w", "--stdin", data=f"orphan {i}".encode())
               for i in range(30)]  # unreachable loose objects, as refused rebases leave behind
    pack_dir = repo.root / "objects" / "pack"
    packs = set(pack_dir.glob("pack-*.pack"))
    stale, fresh = pack_dir / "tmp_pack_killed", pack_dir / "tmp_pack_fetching"
    stale.write_bytes(b"x")
    fresh.write_bytes(b"x")
    os.utime(stale, (time.time() - 7200,) * 2)  # left by a pass killed at its timeout
    (pack_dir / "tmp_pack_renamed").symlink_to(pack_dir / "gone")  # its stat() raises, as after a rename
    caplog.set_level("INFO", logger="chatbot4")
    world._maintain_repo()
    line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("git maintenance"))
    info = dict(item.split("=") for item in line.split(": ", 1)[1].split())
    assert line.startswith("git maintenance done: ") and int(info["loose_before"]) >= 30
    assert set(info) == {"loose_before", "pack_refs_s", "pack_loose_s", "packs", "loose_after"}
    assert info["loose_after"] == "0"
    assert not [f for f in (repo.root / "refs").rglob("*") if f.is_file()]  # all in packed-refs
    assert len(repo.git("for-each-ref", "--format=%(refname)", f"refs/flock/pr/{rec['pr']}/").split()) == 2
    assert re.search(r"^count: 0$", repo.git("count-objects", "-v"), re.MULTILINE)
    repo.git("cat-file", "-e", orphans[0])  # packed, not pruned
    after = set(pack_dir.glob("pack-*.pack"))
    assert packs < after and len(after) == len(packs) + 1 == int(info["packs"])  # old packs kept as they are
    assert (pack_dir / "multi-pack-index").exists()
    assert not stale.exists() and fresh.exists() and not (repo.root / "gc.pid").exists()
    world._maintain_repo()  # nothing loose: no new pack
    assert set(pack_dir.glob("pack-*.pack")) == after


def test_git_maintenance_packs_a_bounded_batch_per_pass(setup, monkeypatch, caplog):
    repo = setup[0].repo
    world = setup[0]
    for i in range(25):
        repo.git("hash-object", "-w", "--stdin", data=f"orphan {i}".encode())
    monkeypatch.setattr("repository.PACK_LOOSE_MAX", 10)
    caplog.set_level("INFO", logger="chatbot4")
    left = []
    for _ in range(3):
        caplog.clear()
        world._maintain_repo()
        line = next(r.getMessage() for r in caplog.records if r.getMessage().startswith("git maintenance"))
        assert line.startswith("git maintenance done: ")
        left.append(int(re.search(r"loose_after=(\d+)", line)[1]))
    assert left[0] >= 15 and left[1] == left[0] - 10 and left[2] == max(0, left[1] - 10)


def test_merged_and_closed_prs_answer_instead_of_no_open_pr(setup):
    world, clone, _remote = setup
    _module(clone, "weather", None)
    merged = submit(world, clone, branch="merged")
    approve(world, merged)
    sha = merge(world, Agent("birch"), merged["pr"])["merged_sha"]
    _module(clone, "time", None)
    closed = submit(world, clone, branch="closed")
    world.close_pr(Agent("cedar"), closed["pr"], "superseded")

    def check():
        for call in (lambda: merge(world, Agent("oak"), merged["pr"]),
                     lambda: world.close_pr(Agent("cedar"), merged["pr"], "done")):
            result = call()
            assert result["already_merged"] and result["merged_sha"] == sha
        with pytest.raises(ToolError, match=f"PR #{merged['pr']} is already merged") as err:
            approve(world, merged, "oak")
        assert err.value.code == "pr_merged"
        for call in (lambda: merge(world, Agent("oak"), closed["pr"]),
                     lambda: approve(world, closed, "oak")):
            with pytest.raises(ToolError, match=f"PR #{closed['pr']} was closed by cedar: superseded") as err:
                call()
            assert err.value.code == "pr_closed"
        with pytest.raises(ToolError, match="no open PR with that number"):
            merge(world, Agent("oak"), 999)

    check()
    world.legacy_json = False
    with world.lock:  # ended long ago: both move to the cold table
        world.ended.update({str(merged["pr"]): 0.0, str(closed["pr"]): 0.0})
        assert world._move_cold(time.time()) and not world.state["prs"]
    check()


def test_a_timed_out_grouped_git_kills_its_children_too(setup, tmp_path):
    repo, pidfile = setup[0].repo, tmp_path / "child.pid"
    slow = f"alias.slow=!sh -c 'sleep 60 & echo $! > {pidfile}; wait'"
    with pytest.raises(subprocess.TimeoutExpired):
        repo.git("-c", slow, "slow", timeout=1, group=True)
    child = Path(f"/proc/{pidfile.read_text().strip()}/stat")
    for _ in range(50):  # killed: gone, or a zombie until reaped
        if not child.exists() or child.read_text().split(") ")[1][0] == "Z":
            break
        time.sleep(0.1)
    else:
        raise AssertionError("the grouped git's child outlived the timeout")


def test_a_stale_gc_pid_is_removed_at_start(setup):
    world, _clone, _remote = setup
    pid = world.repo.root / "gc.pid"
    pid.write_text("12345 another-node")  # a gc stopped with a world that ran on another node
    lock = world.repo.root / "objects" / "pack" / "multi-pack-index.lock"
    lock.write_text("")  # a multi-pack-index write killed with the world
    restarted = Chatbot4World(world.ws, Repository(world.repo.root, world.repo.remote))
    assert not pid.exists() and not lock.exists() and restarted.repo.loose() >= 0


def test_patch_id_is_cached_and_none_for_an_empty_diff(setup):
    world, clone, _ = setup
    main = world.repo.sha()
    assert world.repo.patch_id(main, main) is None
    _module(clone, "time", None)
    rec = submit(world, clone)
    first = world.repo.patch_id(rec["base_sha"], rec["head_sha"])
    assert first and world.repo.patch_ids[(rec["base_sha"], rec["head_sha"])] == first
    assert world.repo.patch_id0(main, main) is None
    zero = world.repo.patch_id0(rec["base_sha"], rec["head_sha"])
    assert zero and world.repo.patch_ids[(rec["base_sha"], rec["head_sha"], 0)] == zero


def _review(world, rec, who, verdict, comment="Looked at the diff"):
    return world.call("review_pr", {"pr": rec["pr"], "head_sha": rec["head_sha"],
                                    "verdict": verdict, "comment": comment}, Agent(who))


def test_review_news_reaches_the_author_once_and_never_on_a_refusal(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    result = _review(world, rec, "birch", "request_changes", "Please add a test. " + "x" * 300)
    assert "inbox" not in result  # the reviewer acted; no news for it
    with pytest.raises(ToolError):
        world.call("list_prs", {"state": "bogus"}, Agent("cedar"))
    with pytest.raises(ToolError):
        world.call("merge_pr", {"pr": rec["pr"]}, Agent("cedar"))
    got = world.call("list_prs", {"author": "cedar"}, Agent("cedar"))
    assert set(got) == {"prs", "inbox", "pr_budget"} and got["prs"][0]["pr"] == rec["pr"]
    [line] = got["inbox"]
    assert line.startswith(f"birch reviewed your PR #{rec['pr']}: request_changes: Please add a test.")
    assert line.endswith("(just now)") and len(line) < 220
    again = world.call("list_prs", {"author": "cedar"}, Agent("cedar"))
    assert isinstance(again, list) and again[0]["pr"] == rec["pr"]
    assert world.state["inbox"].get("cedar") is None


def test_update_after_request_changes_tells_the_reviewer_and_approval_tells_mergeable(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    _review(world, rec, "birch", "request_changes")
    _review(world, rec, "aspen", "approve")
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    new = world.call_workspace("open_pr", {"branch": "greeting", "title": "Improve greetings",
                               "description": "Added a test.", "pr": rec["pr"]},
                               Agent("cedar"), capture(clone.parent, clone.name))
    assert len(new["inbox"]) == 2  # birch's and aspen's reviews, no mergeable (changes requested)
    assert not any("ready to merge_pr" in line for line in new["inbox"])
    assert "cedar" not in world.state["inbox"]
    assert "aspen" not in world.state["inbox"]  # aspen approved: no re-review notice
    [notice] = world.call("list_prs", {}, Agent("birch"))["inbox"]
    assert notice.startswith(f"PR #{rec['pr']} was updated by cedar after your request_changes")
    _review(world, new, "birch", "approve", "Thanks")
    inbox = world.call("run_bot", {"ref_or_path": "main", "lines": ["Hello"]}, Agent("cedar"))["inbox"]
    assert inbox[0].startswith(f"birch reviewed your PR #{rec['pr']}: approve: Thanks")
    assert inbox[1].startswith(f"PR #{rec['pr']} approved by birch; ready to merge_pr")
    world.call("merge_pr", {"pr": rec["pr"]}, Agent("cedar"))
    world._merge_next()  # the merge queue's run: the news is for the author who queued it too
    assert [e["text"] for e in world.state["inbox"]["cedar"]] == [f"your PR #{rec['pr']} was merged"]
    assert world.state["inbox"]["birch"][-1]["text"] == f"PR #{rec['pr']} you approved was merged by cedar"


def test_merge_and_close_by_others_are_news_for_the_author(setup):
    world, clone, _ = setup
    world.operators = {"oak"}
    one = submit(world, clone, branch="first")
    _module(clone, "second", "Second")
    two = submit(world, clone, branch="second")
    _review(world, one, "birch", "approve")
    assert world.call("merge_pr", {"pr": one["pr"]}, Agent("birch"))["queued"]
    world._merge_next()
    assert [e["text"] for e in world.state["inbox"]["birch"]] == [f"PR #{one['pr']} you queued is merged"]
    world.call("close_pr", {"pr": two["pr"], "reason": "duplicate of #1"}, Agent("oak"))
    inbox = world.call("list_prs", {"state": "all"}, Agent("cedar"))["inbox"]
    assert inbox[-2].startswith(f"your PR #{one['pr']} was merged by birch")
    assert inbox[-1].startswith(f"your PR #{two['pr']} was closed by oak: duplicate of #1")
    assert not any("ready to merge_pr" in line for line in inbox)  # superseded by the merge


def test_inbox_is_capped_deduped_and_survives_a_restart(setup):
    world, clone, _ = setup
    with world.lock:
        for n in range(30):
            world._notify("cedar", n, "merged", f"news {n}")
        world._notify("cedar", 29, "merged", "news 29 again")
        world._notify("cedar", 29, "closed", "closed 29")
        world._save()
    box = world.state["inbox"]["cedar"]
    assert len(box) == 20 and [e["text"] for e in box[-2:]] == ["news 29 again", "closed 29"]
    assert sum(e["pr"] == 29 for e in box) == 2
    restarted = Chatbot4World(world.ws, world.repo)
    inbox = restarted.call("list_prs", {}, Agent("cedar"))["inbox"]
    assert len(inbox) == 20 and inbox[-1] == "closed 29 (just now)"
    assert Chatbot4World(world.ws, world.repo).state["inbox"] == {}  # delivery was saved
    state = json.loads(world.db.read_text())
    del state["inbox"]  # an archive from before the inbox existed
    _write_archive(world, state)
    old = Chatbot4World(world.ws, world.repo)
    assert old.call("list_prs", {}, Agent("cedar")) == []
    rec = submit(old, clone)
    _review(old, rec, "birch", "approve")
    assert len(old.call("list_prs", {}, Agent("cedar"))["inbox"]) == 2


def test_inbox_age_and_idempotent_retry_keep_the_cached_result(setup):
    world, clone, _ = setup
    assert [Chatbot4World._age(s) for s in (5, 180, 7200, 200000)] == [
        "just now", "3 min ago", "2 h ago", "2 d ago"]
    first = world.call_workspace("open_pr", {"branch": "greeting", "title": "Improve greetings",
                                 "description": "Tested."}, Agent("cedar", request_id="k1"),
                                 capture(clone.parent, clone.name))
    with world.lock:
        world._notify("cedar", 99, "merged", "news")
    retry = world.call_workspace("open_pr", {"branch": "greeting", "title": "Improve greetings",
                                 "description": "Tested."}, Agent("cedar", request_id="k1"),
                                 capture(clone.parent, clone.name))
    assert retry["inbox"] == ["news (just now)"] and retry["pr"] == first["pr"]
    assert "inbox" not in world.state["requests"]["k1"]


def _wait(cond, timeout=10):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.02)
    return cond()


def test_new_pr_budget_floor_refuses_before_the_hub_and_logs(setup, monkeypatch, caplog):
    monkeypatch.setenv("CHATBOT_PR_OPEN_FLOOR", "2")
    world, clone, _ = setup
    world = Chatbot4World(world.ws, world.repo)
    assert submit(world, clone, "cedar", "one")["pr"] == 1
    second = world.call_workspace("open_pr", {"branch": "two", "title": "Two", "description": "t"},
                                  Agent("birch"), capture(clone.parent, clone.name))
    assert second["pr_budget"] == {"mode": "merges", "opened_last_hour": 2, "merged_last_hour": 0,
                                   "limit": 2, "available": 0, "next_slot_min": 60}
    with caplog.at_level("WARNING"), pytest.raises(ToolError) as exc:
        submit(world, clone, "pine", "three")
    assert exc.value.code == "open_budget" and world.repo.hub.n == 2
    assert exc.value.retry_after_s == 3600  # the structured form of "next slot in ~60 min"
    assert world.metrics()["budget_refusals_last_hour"] == 1
    assert exc.value.message == (
        "the swarm opened 2 PRs in the last hour and merged 0; new PRs wait until more merge "
        "(next slot in ~60 min). Meanwhile review others' PRs (list_prs needs_review=true) or "
        "improve your open PR with pr=<n>.")
    assert "open_budget: refused a new PR for agent=pine opened=2 merged=0 limit=2" in caplog.text
    events = [json.loads(line) for line in (world.ws / "prs.jsonl").read_text().splitlines()]
    assert events[-1]["event"] == "open_pr_refused" and events[-1]["code"] == "open_budget"


def test_merges_raise_the_budget_and_updates_are_not_counted(setup, monkeypatch):
    import time

    monkeypatch.setenv("CHATBOT_PR_OPEN_FLOOR", "1")
    world, clone, _ = setup
    world = Chatbot4World(world.ws, world.repo)
    rec = submit(world, clone, "cedar", "one")
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    rec = submit(world, clone, "cedar", "one", pr=rec["pr"])  # an update: never counted
    assert len(world.state["pr_opens"]) == 1
    with pytest.raises(ToolError, match="opened 1 PRs in the last hour and merged 0"):
        submit(world, clone, "birch", "two")
    approve(world, rec, "pine")
    merge(world, Agent("cedar"), rec["pr"])
    assert len(world.state["pr_merges"]) == 1
    with pytest.raises(ToolError) as exc:  # 1 merge: the limit is still max(1, 1)
        submit(world, clone, "birch", "two")
    assert exc.value.code == "open_budget"
    git(clone, "fetch", "origin")
    git(clone, "reset", "--hard", "origin/main")
    world.state["pr_merges"] += [time.time()] * 2  # 3 merges this hour: 3 new PRs
    assert submit(world, clone, "birch", "two")["pr"] == 2
    assert submit(world, clone, "pine", "three")["pr"] == 3
    with pytest.raises(ToolError, match="opened 3 PRs in the last hour and merged 3"):
        submit(world, clone, "oak", "four")


def test_budget_off_and_budget_survives_a_restart(setup, monkeypatch):
    import time

    monkeypatch.setenv("CHATBOT_PR_OPEN_FLOOR", "1")
    world, clone, _ = setup
    world = Chatbot4World(world.ws, world.repo)
    submit(world, clone, "cedar", "one")
    restarted = Chatbot4World(world.ws, world.repo)
    with pytest.raises(ToolError) as exc:
        submit(restarted, clone, "birch", "two")
    assert exc.value.code == "open_budget"
    restarted.state["pr_opens"] = [time.time() - 3590]  # leaves the window in about a minute
    with pytest.raises(ToolError, match=r"next slot in ~1 min"):
        submit(restarted, clone, "birch", "two")
    restarted.state["pr_opens"] = [time.time() - 3601]
    assert submit(restarted, clone, "birch", "two")["pr"] == 2
    monkeypatch.setenv("CHATBOT_PR_OPEN_BUDGET", "off")
    off = Chatbot4World(world.ws, world.repo)
    assert submit(off, clone, "pine", "three")["pr_budget"]["mode"] == "off"
    assert submit(off, clone, "oak", "four")["pr"] == 4


def test_review_makes_no_hub_call_and_is_never_hub_busy(setup, monkeypatch):
    import threading
    import time

    world, clone, _ = setup
    rec = submit(world, clone)
    hub, posted_by = world.repo.hub, []
    real = hub.comment
    monkeypatch.setattr(hub, "comment", lambda pr, text: (posted_by.append(
        threading.current_thread().name), real(pr, text)))
    world.hub_wait = 0.1
    with world.lanes.hold("comment"):  # another comment is being posted for a long time
        started = time.monotonic()
        for who in ("birch", "pine", "oak", "elm"):
            assert approve(world, rec, who)["verdict"] == "approve"
        assert time.monotonic() - started < 2 and hub.comments == []
        assert world.outbox.depth()[0] == 4
    assert _wait(lambda: world.outbox.depth()[0] == 0)
    assert posted_by == ["lane-comment-outbox"]  # the backlog of one PR is one post
    texts = [f"approve by {who} on {rec['head_sha']}\n\nChecked the code and bot"
             for who in ("birch", "pine", "oak", "elm")]
    assert hub.comments == [(rec["pr"], chatbot4.COMMENT_SEPARATOR.join(texts))]


def test_outbox_retries_persists_across_restarts_and_merge_needs_no_comment(setup, monkeypatch):
    monkeypatch.setattr("chatbot4.OUTBOX_BACKOFF", (0.05, 0.1))
    world, clone, remote = setup
    rec = submit(world, clone)
    hub, fail = world.repo.hub, [True]
    real = hub.comment

    def flaky(pr, text):
        if fail[0]:
            raise ToolError("Hub 503", "hub_error")
        real(pr, text)
    monkeypatch.setattr(hub, "comment", flaky)
    world.outbox.backoff = chatbot4.OUTBOX_BACKOFF
    approve(world, rec)
    assert _wait(lambda: world.outbox.entries.get(rec["pr"], {}).get("tries", 0) >= 2)
    result = merge(world, Agent("cedar"), rec["pr"])  # the comment is still unposted
    assert git(remote, "rev-parse", "main") == result["merged_sha"] and hub.comments == []
    restarted = Chatbot4World(world.ws, world.repo)  # stops the old outbox, keeps the comment
    assert list(restarted.outbox.entries) == [rec["pr"]]
    fail[0] = False
    assert _wait(lambda: len(hub.comments) == 1) and hub.comments[0][0] == rec["pr"]
    assert _wait(lambda: restarted.outbox.depth()[0] == 0)
    restarted.flush()
    assert "hub_outbox" not in json.loads((world.ws / "prs.json").read_text())


def test_outbox_never_drops_and_close_leaves_it_for_the_next_start(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    with world.lanes.hold("comment"):
        for who in ("birch", "pine", "oak", "elm", "ash"):
            approve(world, rec, who)
        world.close(timeout=5)
    assert world.repo.hub.comments == [] and not world.outbox.thread.is_alive()
    restarted = Chatbot4World(world.ws, world.repo)
    assert _wait(lambda: restarted.outbox.depth()[0] == 0)
    [(pr, text)] = restarted.repo.hub.comments
    assert pr == rec["pr"] and text.count(chatbot4.COMMENT_SEPARATOR) == 4


def test_legacy_hub_outbox_in_prs_json_moves_to_the_outbox_file(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    world.close()
    state = json.loads((world.ws / "prs.json").read_text())
    state["hub_outbox"] = [{"id": "x", "pr": rec["pr"], "agent": "birch", "text": "old review",
                            "tries": 3, "next_at": 0.0, "queued_at": 1.0}]
    _write_archive(world, state)
    restarted = Chatbot4World(world.ws, world.repo)
    assert _wait(lambda: restarted.repo.hub.comments == [(rec["pr"], "old review")])
    restarted.flush()
    assert "hub_outbox" not in json.loads((world.ws / "prs.json").read_text())


def _hub_moved_past_the_record(world, clone, remote):
    """Open a PR, then push a newer head to the Hub's refs/pr/N behind the world's
    back (a push in the gap before a restart from an older archive)."""
    rec = submit(world, clone)
    _module(clone, "gap", "from the gap")
    _commit(clone, "pushed in the gap")
    gap = git(clone, "rev-parse", "HEAD")
    git(clone, "push", "-q", str(remote), f"+{gap}:refs/pr/{rec['pr']}")
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == rec["head_sha"] != gap
    return rec, gap


def test_stale_lease_adopts_the_hub_head_and_retries_once(setup, caplog):
    world, clone, remote = setup
    rec, gap = _hub_moved_past_the_record(world, clone, remote)
    caplog.set_level("WARNING")
    _module(clone, "later", "a later change")
    new = submit(world, clone, pr=rec["pr"])
    assert new["head_sha"] not in (rec["head_sha"], gap)
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]
    assert f"adopted Hub head for PR #{rec['pr']} after a stale lease" in caplog.text
    assert world.metrics()["stale_lease_adoptions_total"] == 1


def test_stale_lease_with_a_failed_fetch_refuses_cleanly(setup, monkeypatch):
    world, clone, remote = setup
    rec, gap = _hub_moved_past_the_record(world, clone, remote)
    calls = []
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda numbers, **kw: calls.append(numbers) or list(numbers))
    _module(clone, "later", "a later change")
    with pytest.raises(ToolError, match=r"newer than the world's record.*retry open_pr with pr=") as err:
        submit(world, clone, pr=rec["pr"])
    assert err.value.code == "stale_lease" and calls == [[rec["pr"]]]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == rec["head_sha"]
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == gap
    monkeypatch.delattr(world.repo, "fetch_pr_refs")  # the Hub answers again: adopt and land
    new = submit(world, clone, pr=rec["pr"])
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]


def test_stale_lease_rejected_again_after_adopting_refuses_without_a_loop(setup, monkeypatch):
    world, clone, remote = setup
    rec, gap = _hub_moved_past_the_record(world, clone, remote)
    real, pushes = world.repo.push, []

    def push(sha, ref, expected=None, force=False):
        pushes.append(ref)
        if ref.startswith("refs/pr/"):
            raise ToolError("repository operation failed; refresh your clone and retry", "stale_lease")
        return real(sha, ref, expected, force)

    monkeypatch.setattr(world.repo, "push", push)
    _module(clone, "later", "a later change")
    with pytest.raises(ToolError, match="newer than the world's record") as err:
        submit(world, clone, pr=rec["pr"])
    assert err.value.code == "stale_lease" and pushes == [f"refs/pr/{rec['pr']}"] * 2
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == gap  # adopted for the next call


def test_a_normal_update_never_fetches_pr_refs(setup, monkeypatch):
    world, clone, remote = setup
    rec = submit(world, clone)
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda *a, **k: pytest.fail("fetched"))
    _module(clone, "later", "a later change")
    new = submit(world, clone, pr=rec["pr"])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]


def test_open_and_update_push_no_agent_branch(setup):
    world, clone, remote = setup
    rec = submit(world, clone)
    _module(clone, "later", "a later change")
    submit(world, clone, pr=rec["pr"])
    assert git(remote, "for-each-ref", "refs/heads/agents") == ""
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}")


def test_a_lease_rejection_has_its_own_code(setup):
    world, clone, remote = setup
    rec = submit(world, clone)
    with pytest.raises(ToolError) as err:
        world.repo.push(git(remote, "rev-parse", "main"), f"refs/pr/{rec['pr']}", expected="0" * 40)
    assert err.value.code == "stale_lease"
    with pytest.raises(ToolError) as err:
        world.repo.push("1" * 40, "refs/pr/999")
    assert err.value.code == "git_error"


def test_stale_lease_adopts_the_hub_head_from_the_refs_api(setup, monkeypatch, caplog):
    world, clone, remote = setup
    rec, gap = _hub_moved_past_the_record(world, clone, remote)
    assert not world.repo.has_commit(gap)  # the lease needs the sha only, not the object
    calls = []
    world.repo.hub.refs = lambda: calls.append(1) or {f"refs/pr/{rec['pr']}": gap}
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda *a, **k: pytest.fail("fetched"))
    caplog.set_level("WARNING")
    _module(clone, "later", "a later change")
    new = submit(world, clone, pr=rec["pr"])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]
    assert f"adopted Hub head for PR #{rec['pr']} after a stale lease (via API)" in caplog.text
    assert calls == [1]


@pytest.mark.parametrize("api", ["empty", "raises"])
def test_refs_api_failure_falls_back_to_git_fetch_then_refuses(setup, monkeypatch, api):
    world, clone, remote = setup
    rec, _gap = _hub_moved_past_the_record(world, clone, remote)

    def refs():
        if api == "raises":
            raise RuntimeError("down")
        return {}

    world.repo.hub.refs = refs
    calls, real = [], world.repo.fetch_pr_refs
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda numbers, **kw: calls.append(numbers) or list(numbers))
    _module(clone, "later", "a later change")
    with pytest.raises(ToolError, match="newer than the world's record") as err:
        submit(world, clone, pr=rec["pr"])
    assert err.value.code == "stale_lease" and calls == [[rec["pr"]]]
    monkeypatch.setattr(world.repo, "fetch_pr_refs", real)  # the git fetch works: adopt and land
    new = submit(world, clone, pr=rec["pr"])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]


class _Ref:
    def __init__(self, ref, sha):
        self.ref, self.target_commit, self.name = ref, sha, ref.rsplit("/", 1)[-1]


class _Refs:
    def __init__(self):
        self.branches = [_Ref("refs/heads/main", "a" * 40), _Ref("refs/heads/greeting", "b" * 40)]
        self.pull_requests = [_Ref("refs/pr/7", "c" * 40)]
        self.converts, self.tags = [_Ref("refs/convert/parquet", "d" * 40)], []


def test_hub_refs_map_is_cached_per_burst_and_survives_failures(monkeypatch, caplog):
    import threading
    monkeypatch.setenv("HF_TOKEN", "hf_secret_token")
    monkeypatch.setenv("CHATBOT_HUB_REFS_TTL_S", "60")
    hub = Hub("org/data")
    calls, fail = [], []

    class Api:
        def list_repo_refs(self, repo_id, repo_type, include_pull_requests):
            calls.append((repo_id, repo_type, include_pull_requests))
            if fail:
                raise RuntimeError("down")
            return _Refs()

    hub.api = Api()
    caplog.set_level("WARNING")
    results = []
    threads = [threading.Thread(target=lambda: results.append(hub.refs())) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    expected = {"refs/heads/main": "a" * 40, "refs/heads/greeting": "b" * 40, "refs/pr/7": "c" * 40}
    assert results == [expected] * 8 and calls == [("org/data", "dataset", True)]
    hub.refs_at -= 61  # the TTL expired: one new request; a failure keeps the stale map
    fail.append(1)
    assert hub.refs() == expected and len(calls) == 2
    assert "Hub refs not listed for org/data" in caplog.text and "hf_secret_token" not in caplog.text
    hub.refs_cache = None
    assert hub.refs() == {} and len(calls) == 3


def test_pr_refs_are_fetched_anonymously_first(setup, monkeypatch):
    world, clone, remote = setup
    rec, gap = _hub_moved_past_the_record(world, clone, remote)
    repo, real, seen = world.repo, world.repo.git, []

    def spy(*args, **kw):
        if "fetch" in args:
            seen.append((str(remote) in args, "GIT_ASKPASS" in (kw.get("env") or repo.env), "credential.helper=" in args))
        return real(*args, **kw)

    monkeypatch.setattr(repo, "git", spy)
    repo.env["GIT_ASKPASS"] = "/nonexistent-askpass"  # the authenticated path's helper
    repo.anonymous = True
    assert repo.fetch_pr_refs([rec["pr"]]) == [] and repo.has_commit(gap)
    assert seen == [(True, False, True)]  # the plain URL, no askpass, no credential helper
    seen.clear()
    repo.remote = str(remote.parent / "missing.git")  # the anonymous fetch fails: use the token
    git(remote, "update-ref", "refs/pr/99", gap)
    assert repo.fetch_pr_refs([99], delays=()) == []
    assert seen == [(False, False, True), (False, True, False)]


def test_hub_fetches_offer_only_mains_history(setup, monkeypatch):
    world, clone, remote = setup
    rec, _gap = _hub_moved_past_the_record(world, clone, remote)
    repo, real, fetches = world.repo, world.repo.git, []

    def spy(*args, **kw):
        if "fetch" in args:
            fetches.append(args)
        return real(*args, **kw)

    monkeypatch.setattr(repo, "git", spy)
    repo.fetch()
    repo.anonymous = True
    assert repo.fetch_pr_refs([rec["pr"]]) == []  # anonymous
    repo.anonymous = False
    assert repo.fetch_pr_refs([rec["pr"]]) == []  # with the token
    assert len(fetches) == 3
    for args in fetches:  # git -c protocol.version=2 [...] fetch ... --negotiation-tip=refs/heads/main
        at = args.index("fetch")
        assert args[:2] == ("-c", "protocol.version=2")
        assert "--negotiation-tip=refs/heads/main" in args[at:]
        assert not any("*" in a for a in args)  # explicit refspecs only


def test_the_first_fetch_of_an_empty_repository_skips_the_negotiation_tip(tmp_path):
    seed = tmp_path / "seed"
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    git(seed, "-c", "user.name=s", "-c", "user.email=s@x", "commit", "-q", "--allow-empty", "-m", "s")
    repo = Repository(tmp_path / "w" / "repo.git", str(seed))
    assert repo.negotiation() == ["--negotiation-tip=refs/heads/main"]
    assert repo.sha() == git(seed, "rev-parse", "main")


def test_startup_reconciles_prs_the_hub_closed_or_merged(setup, caplog):
    world, clone, _ = setup
    recs = []
    for name, who in (("one", "cedar"), ("two", "cedar"), ("three", "fern"), ("four", "fern")):
        (clone / "bot/modules" / f"{name}.py").write_text(f'def respond(text, memory):\n    return "{name}"\n')
        recs.append(submit(world, clone, author=who, branch=name))
    n = [r["pr"] for r in recs]
    world.repo.hub.pr_states = lambda: {n[0]: "merged", n[1]: "closed", n[2]: "open", 999: "closed"}
    with caplog.at_level("WARNING"):
        world.hub_states_at = float("-inf")
        world._sync_hub_states()
    by = {r["pr"]: r for r in world.state["prs"].values()}
    assert (by[n[0]]["state"], by[n[0]]["merged_sha"], by[n[0]]["reconciled_from_hub"]) == ("merged", None, True)
    assert (by[n[1]]["state"], by[n[1]]["reconciled_from_hub"]) == ("closed", True)
    assert by[n[2]]["state"] == "open" and "reconciled_from_hub" not in by[n[2]]  # Hub: open
    assert by[n[3]]["state"] == "open" and "reconciled_from_hub" not in by[n[3]]  # not listed
    assert sum("on the Hub but was open here" in r.message for r in caplog.records) == 2
    # throttled: a second call within the period does not read the Hub again
    calls = []
    world.repo.hub.pr_states = lambda: calls.append(1) or {}
    world._sync_hub_states(every=600)
    assert calls == []
    # a failing read keeps the records and never raises
    world.repo.hub.pr_states = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    world._sync_hub_states()
    assert by[n[3]]["state"] == "open"


def _http_400(message):
    import httpx
    from huggingface_hub.errors import HfHubHTTPError
    req = httpx.Request("POST", "https://hub.invalid/api/datasets/x/y/discussions/7/status")
    resp = httpx.Response(400, headers={"x-error-message": message}, request=req)
    return HfHubHTTPError("bad", response=resp)


def test_hub_request_names_an_already_closed_or_merged_pr(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "t")
    hub = Hub("x/y")
    calls = []

    def api(repo_id, **kw):
        calls.append(kw)
        raise _http_400("Cannot change the status of a merged discussion")

    hub.api.change_discussion_status = api
    with pytest.raises(ToolError) as exc:
        hub.close(7, "bye")
    assert exc.value.code == "pr_merged" and exc.value.message == "PR #7 is already merged on the Hub"
    assert len(calls) == 1
    hub.api.change_discussion_status = lambda repo_id, **kw: (_ for _ in ()).throw(_http_400("bad title"))
    with pytest.raises(ToolError) as exc:
        hub.ready(7)
    assert exc.value.code == "hub_error"


def test_status_change_on_a_closed_pr_is_not_retried(setup, monkeypatch):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    calls = []

    def closed(pr, *args):
        calls.append(pr)
        raise ToolError(f"PR #{pr} is already closed on the Hub", "pr_closed")

    monkeypatch.setattr(world.repo.hub, "close", closed)
    with pytest.raises(ToolError, match=f"PR #{rec['pr']} is already closed on the Hub"):
        world.close_pr(Agent("cedar"), rec["pr"], "obsolete")
    assert calls == [rec["pr"]] and world.state["prs"][str(rec["pr"])]["state"] == "closed"
    assert world.state["prs"][str(rec["pr"])]["reconciled_from_hub"] is True

    # merge on a PR the Hub already merged elsewhere
    (clone / "bot/modules/two.py").write_text('def respond(text, memory):\n    return None\n')
    two = submit(world, clone, branch="two")
    approve(world, two)
    merges = []

    def merged(pr, comment):
        merges.append(pr)
        raise ToolError(f"PR #{pr} is already merged on the Hub", "pr_merged")

    monkeypatch.setattr(world.repo.hub, "merge", merged)
    result = merge(world, Agent("birch"), two["pr"])  # answered with the (Hub's) merge
    assert result["already_merged"] and result["merged_sha"] is None
    assert merges == [two["pr"]] and world.state["prs"][str(two["pr"])]["state"] == "merged"

    # a queued comment on a closed PR is dropped, not retried 12 times
    three = submit(world, clone, branch="three")
    posts = []

    def refused(pr, text):
        posts.append(pr)
        raise ToolError(f"PR #{pr} is already closed on the Hub", "pr_closed")

    monkeypatch.setattr(world.repo.hub, "comment", refused)
    world._queue_comment(Agent("birch"), three["pr"], "looks fine")
    assert _wait(lambda: world.outbox.depth()[0] == 0)
    assert posts == [three["pr"]] and world.lanes.metrics()["comment"]["final"] == 1
    assert world.state["prs"][str(three["pr"])]["state"] == "closed"


# -- debounced prs.json saves and the slim open_pr request cache ------------------------------

def _on_disk(world):
    return json.loads(world.db.read_text())


def test_save_snapshot_encodes_exactly_like_json_dumps(setup):
    import io

    world, _, _ = setup
    with world.lock:
        world.state["odd"] = {"\ud800 lone": ["é", 1.5, None, True, {"n": [1, 2]}], "": {}}
        world.state["scalar"] = 3
        world.state["inbox"]["x"] = [{"pr": 1, "kind": "k", "text": "t\n\"q\"", "ts": 1.25}]
        out = io.StringIO()
        world._encode(world._snapshot(), out)
        assert out.getvalue() == json.dumps(world.state)


def test_debounced_save_writes_within_the_interval(setup):
    world, clone, _ = setup
    world.save_interval = 0.5
    submit(world, clone)
    writes = []
    real = world.flush

    def counting():
        before = world.written_gen
        real()
        if world.written_gen != before:
            writes.append(world.written_gen)
    world.flush = counting
    time.sleep(0.6)  # the interval after the open's write has passed
    started = time.monotonic()
    for n in range(50):  # a burst of changes: one write now, one at most an interval later
        with world.lock:
            world._notify("cedar", n, "merged", f"news {n}")
            world._save()
        time.sleep(0.01)
    assert _wait(lambda: any(e["pr"] == 49 for e in _on_disk(world)["inbox"].get("cedar", [])), 3)
    assert time.monotonic() - started < 0.5 + 0.5 + 1.0  # the burst, one interval, slack
    assert 1 <= len(writes) <= 3
    assert _on_disk(world) == json.loads(json.dumps(world.state))


def test_close_and_exit_flush_the_last_change(setup):
    import chatbot4

    world, clone, _ = setup
    world.save_interval = 3600
    rec = submit(world, clone)
    world.flush()
    assert _wait(lambda: str(rec["pr"]) in _on_disk(world)["prs"])
    approve(world, rec)  # inside the interval: not on disk yet
    assert _on_disk(world)["prs"][str(rec["pr"])]["reviews"] == {}
    chatbot4._close_all()  # the atexit handler
    assert "birch" in _on_disk(world)["prs"][str(rec["pr"])]["reviews"]
    assert not world.saver.is_alive()
    with world.lock:  # a straggling change after close is written at once
        world._notify("cedar", 1, "merged", "late")
        world._save()
    assert _on_disk(world)["inbox"]["cedar"][-1]["text"] == "late"
    restarted = Chatbot4World(world.ws, world.repo)
    assert "birch" in restarted.state["prs"][str(rec["pr"])]["reviews"]
    assert not list(world.ws.glob("prs.json.*.tmp"))


def test_saves_are_never_torn_under_concurrent_changes(setup):
    import threading

    world, _, _ = setup
    world.save_interval = 0.005
    with world.lock:
        world.state["stress"] = {}
        world._save()
    world.flush()
    stop = threading.Event()
    errors = []

    def mutate(name):
        n = 0
        while not stop.is_set():
            n += 1
            with world.lock:  # two entries that must always agree on disk
                world.state["stress"][f"{name}-a"] = n
                world.state["inbox"][name] = [{"pr": n, "kind": "k", "text": "x" * 500, "ts": n}]
                world.state["stress"][f"{name}-b"] = n
                world._save()

    def read():
        while not stop.is_set():
            try:
                state = _on_disk(world)
            except ValueError as exc:  # a torn or partial file
                errors.append(repr(exc))
                continue
            for name in ("t0", "t1", "t2", "t3"):
                a, b = state["stress"].get(f"{name}-a"), state["stress"].get(f"{name}-b")
                box = state["inbox"].get(name)
                if a != b or (a is not None and box[0]["pr"] != a):
                    errors.append(f"inconsistent {name}: {a} {b}")

    def flusher():
        while not stop.is_set():
            world.flush()

    threads = [threading.Thread(target=mutate, args=(f"t{i}",)) for i in range(4)]
    threads += [threading.Thread(target=read), threading.Thread(target=read),
                threading.Thread(target=flusher)]
    for t in threads:
        t.start()
    time.sleep(3)
    stop.set()
    for t in threads:
        t.join()
    assert errors == []
    world.flush()
    assert world.written_gen == world.save_gen > 50
    assert world.db.read_text() == json.dumps(world.state)


def _open(world, clone, key, description="Tested a conversation.", pr=None, author="cedar"):
    result = world.call_workspace("open_pr", {"branch": "greeting", "title": "Improve greetings",
                                              "description": description, "pr": pr},
                                  Agent(author, request_id=key), capture(clone.parent, clone.name))
    return {k: v for k, v in result.items() if k not in ("pr_budget", "inbox")}


def test_idempotent_replay_returns_the_same_result_from_a_slim_cache(setup):
    world, clone, _ = setup
    long = "Evidence:\n" + "transcript line\n" * 2000
    first = _open(world, clone, "k1", long)
    assert first["description"] == long
    entry = world.state["requests"]["k1"]
    assert entry["description"] is None and len(json.dumps(entry)) < 1500
    assert _open(world, clone, "k1", long) == first
    assert list(_open(world, clone, "k1", long)) == list(first)  # same keys, same order
    # The PR is updated with another description; a late retry of k1 still gets its own.
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    second = _open(world, clone, "k2", "Second version.", pr=first["pr"])
    assert second["description"] == "Second version."
    assert _open(world, clone, "k1", long) == first
    restarted = Chatbot4World(world.ws, world.repo)  # replays from prs.json
    assert _open(restarted, clone, "k1", long) == first
    assert _open(restarted, clone, "k2", "Second version.", pr=first["pr"]) == second


def test_old_request_cache_is_slimmed_on_load_and_expires(setup, monkeypatch):
    import chatbot4

    world, clone, _ = setup
    first = _open(world, clone, "k1", "Original long description " * 200)
    world.close()
    state = _on_disk(world)
    # An archive from before slimming: the full result, no timestamp.
    state["requests"]["k1"] = dict(first)
    state["requests"]["old"] = {**first, "description": "y" * 10000}
    _write_archive(world, state)
    restarted = Chatbot4World(world.ws, world.repo)
    restarted.close()
    on_disk = _on_disk(restarted)["requests"]
    assert all(e["description"] is None and "_ts" in e for e in on_disk.values())
    assert len(json.dumps(on_disk)) < 2000
    again = Chatbot4World(world.ws, world.repo)
    assert _open(again, clone, "k1", "Original long description " * 200) == first
    # The rebuilt description falls back to the PR record when the retry's differs.
    assert _open(again, clone, "k1", "a different text")["description"] == first["description"]
    monkeypatch.setattr(chatbot4, "REQUEST_TTL", 60)
    with again.lock:
        again.state["requests"]["k1"]["_ts"] -= 120
        again._cache_request("new", {"pr": 9, "description": "z"})
    assert list(again.state["requests"]) == ["old", "new"]


def test_merge_has_its_own_hub_slot(setup):
    world, clone, remote = setup
    rec = submit(world, clone)
    approve(world, rec)
    other = submit(world, clone, author="pine", branch="other")
    world.hub_wait = 0.2
    with world.lanes.hold("push"):  # every push lane slot is held
        with pytest.raises(ToolError, match="the Hub is busy"):  # pushes still respect the cap
            (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hey!"\n')
            submit(world, clone, author="pine", branch="other", pr=other["pr"])
        result = merge(world, Agent("cedar"), rec["pr"])  # does not wait for them
    assert git(remote, "rev-parse", "main") == result["merged_sha"]
    m = world.lanes.metrics()["merge"]
    assert (m["in_flight"], m["ok"]) == (0, 1)  # released after the merge


def test_merge_waits_for_its_own_slot_only(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    world.hub_wait = 0.2
    with world.lanes.hold("merge"):  # another merge in progress
        with pytest.raises(ToolError, match="the Hub is busy") as exc:
            merge(world, Agent("cedar"), rec["pr"])
        assert exc.value.code == "hub_busy" and world.state["merge_queue"][0]["tries"] == 1  # kept
        (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hey!"\n')
        assert submit(world, clone, pr=rec["pr"])["head_sha"]  # pushes don't wait for merges


def test_merge_hub_slots_env(setup, monkeypatch):
    world, _, _ = setup
    monkeypatch.setenv("CHATBOT_MERGE_HUB_SLOTS", "3")
    restarted = Chatbot4World(world.ws, world.repo)
    assert restarted.lanes["merge"].slots == 3 and restarted.lanes["push"].slots == 1


def _pr_with_changes(world, clone, author="cedar", branch="greeting"):
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hi!"\n')
    (clone / "bot/modules/weather.py").write_text(
        'PRIORITY = 5\n\n\ndef respond(text, memory):\n    return None\n')
    return submit(world, clone, author=author, branch=branch)


def test_get_pr_shows_files_and_diff_from_the_world_copy(setup):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    got = world.get_pr(rec["pr"])
    assert (got["pr"], got["author"], got["state"], got["title"]) == (
        rec["pr"], "cedar", "open", "Improve greetings")
    assert (got["head_sha"], got["base_sha"]) == (rec["head_sha"], rec["base_sha"])
    assert got["base_is_main"] and got["truncated"] is False
    files = {f["path"]: f for f in got["files"]}
    assert set(files) == {"bot/modules/greeting.py", "bot/modules/weather.py"}
    assert (files["bot/modules/weather.py"]["added"], files["bot/modules/weather.py"]["deleted"]) == (5, 0)
    assert files["bot/modules/greeting.py"]["added"] >= 1
    assert '+    return "Hi!"' in got["diff"] and "+++ b/bot/modules/weather.py" in got["diff"]
    assert got["diff"] == git(world.repo.root, "diff", rec["base_sha"], rec["head_sha"]) + "\n"
    # Another merge moves main: the PR's base is no longer current; merged PRs stay readable.
    approve(world, rec)
    merge(world, Agent("cedar"), rec["pr"])
    merged = world.get_pr(rec["pr"])
    assert merged["state"] == "merged" and merged["diff"] == got["diff"]
    git(clone, "fetch", "origin")
    git(clone, "reset", "--hard", "origin/main")
    (clone / "bot/modules/other.py").write_text("X = 1\n")
    other = submit(world, clone, author="pine", branch="other")
    (clone / "bot/modules/more.py").write_text("Y = 1\n")
    third = submit(world, clone, author="oak", branch="third")
    approve(world, other)
    merge(world, Agent("pine"), other["pr"])
    assert not world.get_pr(third["pr"])["base_is_main"]


def test_get_pr_path_mode(setup):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    got = world.get_pr(rec["pr"], "bot/modules/weather.py")
    assert got["content"] == (clone / "bot/modules/weather.py").read_text()
    assert got["content_truncated"] is False and got["size"] == len(got["content"])
    assert [f["path"] for f in got["files"]] == ["bot/modules/weather.py"]
    assert "weather.py" in got["diff"] and "greeting.py" not in got["diff"]
    unchanged = world.get_pr(rec["pr"], "bot/runtime.py")  # in the head, not in the diff
    assert unchanged["files"] == [] and unchanged["diff"] == ""
    assert unchanged["content"] == (clone / "bot/runtime.py").read_text()
    with pytest.raises(ToolError, match="neither"):
        world.get_pr(rec["pr"], "bot/modules/nope.py")
    for bad in ("../etc/passwd", "/etc/passwd", ""):
        with pytest.raises(ToolError, match="relative file path"):
            world.get_pr(rec["pr"], bad)


def test_get_pr_truncates_large_diffs(setup, monkeypatch):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    monkeypatch.setattr("chatbot4.MAX_DIFF_BYTES", 120)
    got = world.get_pr(rec["pr"])
    assert got["truncated"] is True and got["diff_bytes"] > 120
    assert 0 < len(got["diff"].encode()) <= 120 and got["diff"].endswith("\n")
    assert "path=" in got["hint"] and len(got["files"]) == 2  # the file list is complete
    monkeypatch.setattr("chatbot4.MAX_FILE_BYTES", 10)
    one = world.get_pr(rec["pr"], "bot/modules/weather.py")
    assert one["content_truncated"] is True and one["content"] == "PRIORITY ="


def test_get_pr_unknown_and_missing_head(setup, monkeypatch):
    world, clone, _ = setup
    with pytest.raises(ToolError, match="no PR with that number"):
        world.get_pr(999)
    rec = _pr_with_changes(world, clone)
    world.state["prs"][str(rec["pr"])]["head_sha"] = "f" * 40
    calls = []
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda numbers, **kw: calls.append(numbers) or numbers)
    with pytest.raises(ToolError, match="not in the world's repository") as exc:
        world.get_pr(rec["pr"])
    assert exc.value.code == "missing_head" and calls == [[rec["pr"]]]


def test_get_pr_fetches_a_missing_head_once(setup, monkeypatch):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    real_has = world.repo.has_commit
    fetched = []
    monkeypatch.setattr(world.repo, "has_commit",
                        lambda sha: bool(fetched) and real_has(sha))
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda numbers, **kw: fetched.append(numbers) or [])
    assert world.get_pr(rec["pr"])["head_sha"] == rec["head_sha"] and fetched == [[rec["pr"]]]


def test_get_pr_runs_git_without_the_world_lock(setup, monkeypatch):
    import threading

    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    real, free = world.repo.git_bytes, []

    def probe(*args, **kwargs):
        def other():  # another tool call needing the lock while get_pr runs git
            free.append(world.lock.acquire(timeout=2))
            if free[-1]:
                world.lock.release()
        t = threading.Thread(target=other)
        t.start()
        t.join()
        return real(*args, **kwargs)
    monkeypatch.setattr(world.repo, "git_bytes", probe)
    world.get_pr(rec["pr"], "bot/modules/weather.py")
    assert free and all(free)


def test_get_pr_is_read_only(setup, monkeypatch):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    saves = []
    monkeypatch.setattr(world, "_save", lambda: saves.append(1))
    world.get_pr(rec["pr"])
    assert saves == []


def test_get_pr_is_a_registered_tool(setup):
    world, clone, _ = setup
    rec = _pr_with_changes(world, clone)
    schema = next(t for t in world.manifest()["tools"] if t["name"] == "get_pr")
    assert "pr" in json.dumps(schema) and "path" in json.dumps(schema)
    got = world.call("get_pr", {"pr": rec["pr"], "path": "bot/modules/weather.py"}, Agent("birch"))
    assert got["content"].startswith("PRIORITY")


def test_manifest_declares_workspace_uploads_and_timeouts(setup):
    from flock.world.protocol import WorldManifest

    tools = {t.name: t for t in WorldManifest.model_validate(setup[0].manifest()).tools}
    assert tools["open_pr"].workspace.arg == "path" and tools["open_pr"].workspace.default == "chatbot-swarm"
    run = tools["run_bot"].workspace
    assert (run.arg, run.default, run.skip, run.skip_prefix) == ("ref_or_path", "main", ["main"], "pr:")
    assert tools["review_pr"].workspace is None
    assert {t.timeout_s for t in tools.values()} == {330}


def test_a_slow_hub_merge_does_not_block_reads(setup, monkeypatch):
    import threading

    world, clone, remote = setup
    rec = submit(world, clone)
    approve(world, rec)
    real, entered, release = world.repo.hub.merge, threading.Event(), threading.Event()

    def slow_merge(pr, comment):  # a Hub merge sleeping through 429 retries
        entered.set()
        release.wait(20)
        real(pr, comment)

    monkeypatch.setattr(world.repo.hub, "merge", slow_merge)
    result = {}
    merging = threading.Thread(target=lambda: result.update(merge(world, Agent("birch"), rec["pr"])))
    merging.start()
    try:
        assert entered.wait(20)
        started = time.monotonic()
        assert world.get_pr(rec["pr"])["head_sha"] == rec["head_sha"]
        world.list_prs("all")
        assert time.monotonic() - started < 2, "reads waited for the Hub merge"
        assert world.merge_pr(Agent("pine"), rec["pr"])["position"] == 1  # no second merge
        with pytest.raises(ToolError, match="being merged; retry shortly"):
            approve(world, rec, "pine")
        assert [(r["pr"], r["state"]) for r in world.list_prs()] == [(rec["pr"], "merging")]
    finally:
        release.set()
        merging.join(30)
    assert result["merged_sha"] == git(remote, "rev-parse", "main")


def test_reconcile_runs_git_only_when_main_moved(setup, monkeypatch):
    world, clone, _ = setup
    one = submit(world, clone, branch="one")
    (clone / "bot/modules/two.py").write_text('def respond(text, memory):\n    return None\n')
    submit(world, clone, "birch", "two")
    real, calls = world.repo.ancestor, []
    monkeypatch.setattr(world.repo, "ancestor", lambda *a: (calls.append(a), real(*a))[1])
    with world.lock:
        world._reconcile()  # fills the cache for any PR not seen on this main yet
    calls.clear()
    with world.lock:
        assert not world._reconcile()
    assert not calls  # main unchanged: no git
    approve(world, one)
    merge(world, Agent("birch"), one["pr"])  # main moved: the other PR is checked again
    calls.clear()
    with world.lock:
        world._reconcile()
    assert len(calls) == 1


def test_a_hub_merge_that_raises_puts_the_pr_back_to_open(setup, monkeypatch):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)

    def broken(pr, comment):
        assert world.state["prs"][str(pr)]["state"] == "merging"
        raise RuntimeError("connection reset")

    monkeypatch.setattr(world.repo.hub, "merge", broken)
    with pytest.raises(RuntimeError):
        merge(world, Agent("birch"), rec["pr"])
    assert world.state["prs"][str(rec["pr"])]["state"] == "open"


def test_metrics_report_prs_budget_outbox_and_saves(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    approve(world, rec)
    world.flush()
    m = world.metrics()
    lanes = {f"lane_{lane}_{k}" for lane in ("push", "merge", "comment")
             for k in ("slots", "effective_slots", "in_flight", "waits_total", "wait_s_total",
                       "ok", "transient", "final", "tripped", "paused", "outbox_depth",
                       "outbox_oldest_age_s")}
    assert set(m) == {"open_prs", "opens_last_hour", "merges_last_hour", "open_budget_limit",
                      "budget_refusals_last_hour", "parked_drafts", "merge_queue", "stale_lease_adoptions_total",
                      "hub_busy_total", "hub_comments_rejected_total", "save_duration_s",
                      "save_age_s", "state_bytes", "dirty_rows", "verdicts_total",
                      *(f"main_verdicts_last_hour_{v}" for v in ("human", "agent", "script")),
                      *lanes}
    assert all(isinstance(v, int | float) for v in m.values())
    assert (m["open_prs"], m["opens_last_hour"], m["parked_drafts"]) == (1, 1, 0)
    assert m["state_bytes"] >= (world.ws / "state.db").stat().st_size > 0 and m["save_age_s"] < 60
    assert m["dirty_rows"] == 0


def test_vote_failures_trip_the_push_lane_and_comments_survive_a_restart(setup, monkeypatch):
    world, clone, _ = setup
    monkeypatch.setenv("CHATBOT_HUB_WRITE_SLOTS", "2")
    world = Chatbot4World(world.ws, world.repo)
    world.lanes["push"].trip_gap_s = 0.2  # tripped: one push at a time, 0.2 s apart
    failing_pr_push(world, monkeypatch, [3])  # "vote failed: transaction was aborted", 3 times
    with pytest.raises(ToolError) as exc:
        submit(world, clone)
    assert exc.value.code == "hub_busy" and exc.value.retry_after_s == 60
    m = world.lanes.metrics()["push"]
    assert (m["transient"], m["tripped"], m["slots"], m["effective_slots"]) == (3, 1, 2, 1)
    assert world.lanes.metrics()["merge"]["tripped"] == 1  # push and merge share one breaker
    assert world.lanes.metrics()["comment"]["tripped"] == 0
    rec = submit(world, clone, pr=1)  # the parked draft lands on the tripped lane
    assert world.lanes.metrics()["push"]["ok"] == 1  # the parked attempt counts no outcome

    hub, fail = world.repo.hub, [True]
    real = hub.comment

    def flaky(pr, text):
        if fail[0]:
            raise ToolError("Hub request failed; retry later", "hub_error")
        real(pr, text)

    monkeypatch.setattr(hub, "comment", flaky)
    approve(world, rec)
    assert _wait(lambda: world.lanes.metrics()["comment"]["transient"] >= 1)
    assert world.metrics()["lane_comment_outbox_depth"] == 1
    monkeypatch.setattr(chatbot4, "OUTBOX_BACKOFF", (0.05, 0.1))
    restarted = Chatbot4World(world.ws, world.repo)  # the breaker resets, the outbox does not
    assert not restarted.lanes.metrics()["push"]["tripped"]
    fail[0] = False
    assert _wait(lambda: restarted.outbox.depth()[0] == 0)
    assert hub.comments == [(rec["pr"], f"approve by birch on {rec['head_sha']}\n\nChecked the code and bot")]

    lane = world.lanes["push"]  # the first world's lane recovers after 10 quiet minutes
    lane.clock = lambda: time.monotonic() + 601
    m = world.lanes.metrics()["push"]
    assert (m["tripped"], m["effective_slots"]) == (0, 2)


def test_a_comment_the_hub_refuses_is_dropped_and_joined_ones_are_split(setup, monkeypatch):
    world, clone, _ = setup
    rec = submit(world, clone)
    hub = world.repo.hub
    real = hub.comment

    def picky(pr, text):
        if chatbot4.COMMENT_SEPARATOR in text or "spam" in text:
            error = ToolError("Hub request failed; retry later", "hub_error")
            error.status = 400
            raise error
        real(pr, text)

    monkeypatch.setattr(hub, "comment", picky)
    with world.lanes.hold("comment"):  # three comments pile up and are joined
        for who, text in (("birch", "fine"), ("pine", "spam"), ("oak", "good")):
            world.review_pr(Agent(who), rec["pr"], rec["head_sha"], "approve", text)
    assert _wait(lambda: world.outbox.depth()[0] == 0)
    assert [text.split("\n\n")[1] for _, text in hub.comments] == ["fine", "good"]
    assert world.metrics()["hub_comments_rejected_total"] == 1
    assert world.lanes.metrics()["comment"]["final"] == 1


# -- state.db: dirty-row saves, migration from prs.json, cold PR records ---------------------

def _reload(world, tmp_path, name, keep, flush=True):
    """The state a new world loads from a copy of the workspace holding only prs.json
    (keep="json": the migration path) or only a backup of state.db (keep="db")."""
    if flush:
        world.flush()
    copy = tmp_path / name
    copy.mkdir()
    if keep == "json":
        shutil.copy(world.db, copy / "prs.json")
    else:
        world.store.backup_to(copy / "state.db")
    loaded = Chatbot4World(copy, world.repo)
    loaded.close()
    return loaded.state


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_random_operations_give_the_same_state_through_json_and_sqlite(setup, tmp_path, seed):
    import random

    world, clone, _ = setup
    rng = random.Random(seed)
    names = ["cedar", "birch", "ash"]
    for step in range(30):
        op, who = rng.choice(["open", "open", "update", "review", "review", "review", "merge",
                              "close", "inbox"]), rng.choice(names)
        mine = [r for r in world.state["prs"].values() if r["state"] == "open" and r["author"] == who]
        others = [r for r in world.state["prs"].values() if r["state"] == "open" and r["author"] != who]
        try:
            if op in ("open", "update"):
                _module(clone, f"m{step}", f"reply {seed} {step}")
                pr = rng.choice(mine)["pr"] if op == "update" and mine else None
                branch = next((r["branch"].rsplit("/", 1)[1] for r in mine if r["pr"] == pr), f"b{step}")
                submit(world, clone, who, branch, pr=pr)
            elif op == "review" and others:
                rec = rng.choice(others)
                world.call("review_pr", {"pr": rec["pr"], "head_sha": rec["head_sha"],
                                         "verdict": rng.choice(["approve", "approve", "request_changes"]),
                                         "comment": "checked"}, Agent(who))
            elif op == "merge" and (mine or others):
                world.call("merge_pr", {"pr": rng.choice(mine + others)["pr"]}, Agent(who))
            elif op == "close" and mine:
                world.call("close_pr", {"pr": rng.choice(mine)["pr"], "reason": "superseded"}, Agent(who))
            else:
                world.call("list_prs", {}, Agent(who))  # delivers the inbox
        except ToolError:
            pass
        world.flush()  # after every operation: the rows state.db holds are the memory's
        rows = {kind: dict(world.store.iter(kind)) for kind in chatbot4.KINDS}
        assert {**world.state, **rows, **dict(world.store.iter("_"))} == world.state, (step, op)
    memory = json.loads(json.dumps(world.state))
    assert memory["prs"]
    assert _reload(world, tmp_path, "from-json", "json") == memory
    assert _reload(world, tmp_path, "from-db", "db") == memory


def test_a_save_cut_mid_transaction_keeps_the_saved_state_and_is_retried(setup, tmp_path):
    import sqlite3

    world, clone, _ = setup
    world.save_stop.set()  # no background saver: this test runs every save
    world.save_wake.set()
    world.saver.join()
    rec = submit(world, clone)
    world.flush()
    saved = json.loads(json.dumps(world.state))
    approve(world, rec)  # the PR's review and the author's inbox: two dirty rows
    real, calls = world.store._db, []

    class Cut:  # dies after BEGIN and the first upsert, as a killed saver
        def execute(self, *args):
            calls.append(args[0])
            if len(calls) == 3:
                raise sqlite3.OperationalError("disk I/O error (injected)")
            return real.execute(*args)

    world.store._db = Cut()
    with pytest.raises(sqlite3.OperationalError):
        world.flush()
    world.store._db = real
    assert calls[0] == "BEGIN IMMEDIATE" and world.store.pending == 2
    assert _reload(world, tmp_path, "cut", "db", flush=False) == saved
    world.flush()
    memory = json.loads(json.dumps(world.state))
    assert memory != saved and _reload(world, tmp_path, "retried", "db") == memory


def test_a_run4_prs_json_is_migrated_on_the_first_start(setup, caplog):
    world, clone, _ = setup
    rec = submit(world, clone)
    world.close()
    state = json.loads(world.db.read_text())  # what a run-4 archive carries, plus older records
    state["prs"]["90"] = {**state["prs"][str(rec["pr"])], "pr": 90, "state": "merged",
                          "branch": "agents/ash/old", "merged_sha": rec["head_sha"]}
    state["drafts"]["agents/ash/parked"] = {"pr": 91, "author": "ash", "branch": "agents/ash/parked",
                                            "base_sha": rec["base_sha"], "sha": rec["head_sha"],
                                            "title": "t", "parked_at": 1.0, "url": "u"}
    state["inbox"]["ash"] = [{"pr": 90, "kind": "merged", "text": "merged", "ts": 1.0}]
    state["pr_merges"] = [time.time()]
    state["hub_outbox"] = []
    _write_archive(world, state)
    caplog.set_level("WARNING")
    migrated = Chatbot4World(world.ws, world.repo)
    assert "migrated prs.json to state.db (2 PRs)" in caplog.text
    del state["hub_outbox"]
    assert migrated.state == state
    migrated.close()
    world.db.unlink()  # from state.db alone
    assert Chatbot4World(world.ws, world.repo).state == state
    assert "differ" not in caplog.text


def test_an_import_cut_before_its_commit_is_redone(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    world.close()
    world.store.close()
    for f in world.ws.glob("state.db*"):
        f.unlink()
    from flock.world.store import Store

    Store(world.ws / "state.db").close()  # a kill during the first import: schema, no rows
    again = Chatbot4World(world.ws, world.repo)
    assert again.state["prs"][str(rec["pr"])]["head_sha"] == rec["head_sha"]


def test_old_merged_and_closed_prs_move_to_the_cold_table(setup):
    world, clone, _ = setup
    one, two = submit(world, clone, branch="one"), submit(world, clone, branch="two")
    world.close_pr(Agent("cedar"), one["pr"], "superseded")
    approve(world, two)
    merge(world, Agent("birch"), two["pr"])
    world.close()
    ended = world.state["prs"][str(one["pr"])]["ended_at"]
    world = Chatbot4World(world.ws, world.repo)
    assert world.ended[str(one["pr"])] == ended  # the hour counts across restarts
    with world.lock:
        for key in world.ended:
            world.ended[key] -= chatbot4.COLD_AFTER + 1
    world.flush()
    assert len(world.state["prs"]) == 2  # save.legacy_json on: prs.json keeps every PR
    world.legacy_json = False
    world.flush()
    assert world.state["prs"] == {} and world.store.get("prs", str(one["pr"]))["state"] == "closed"
    for w in (world, Chatbot4World(world.ws, world.repo)):
        assert w.state["prs"] == {}
        assert [r["pr"] for r in w.list_prs("closed")] == [one["pr"]]
        assert [r["pr"] for r in w.list_prs("all", author="cedar")] == [one["pr"], two["pr"]]
        assert w.get_pr(two["pr"])["state"] == "merged"
        with pytest.raises(ToolError, match="slug already used"):
            submit(w, clone, branch="one")


def submit_v2(world, clone, author="cedar", branch="greeting", pr=None):
    from flock.world.workspace import capture_v2
    snapshot = capture_v2(clone.parent, clone.name)
    assert "patch" in snapshot
    return world.call_workspace("open_pr", {"branch": branch, "title": "Improve greetings",
        "description": "Tested a conversation and the interface.", "pr": pr}, Agent(author), snapshot)


def test_open_pr_and_run_bot_through_a_v2_patch(setup):
    world, clone, remote = setup
    assert "workspace-upload-v2" in world.manifest()["capabilities"]
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hey!"\n')
    (clone / "bot/modules/topic.py").write_text('def respond(text, memory):\n    return None\n')
    (clone / "bot/__pycache__").mkdir(exist_ok=True)
    (clone / "bot/__pycache__/x.pyc").write_bytes(b"\0junk")
    rec = submit_v2(world, clone)
    committed = git(remote, "ls-tree", "-r", "--name-only", rec["head_sha"]).splitlines()
    assert "bot/modules/topic.py" in committed and not [n for n in committed if "__pycache__" in n]
    assert git(remote, "show", f"{rec['head_sha']}:bot/modules/greeting.py").endswith('"Hey!"')
    from flock.world.workspace import capture_v2
    out = world.call_workspace("run_bot", {"ref_or_path": "chatbot-swarm", "lines": ["hi"]},
                               Agent("cedar"), capture_v2(clone.parent, clone.name))
    assert out["replies"] == ["Hey!"]
    (clone / "bot/modules/data.py").write_text("X = 1\n" + "#" * (2 << 20))
    with pytest.raises(ToolError, match="file too large"):
        submit_v2(world, clone, branch="big")


def test_v2_with_an_unknown_base_asks_for_the_full_tree(setup, monkeypatch):
    world, clone, _ = setup
    (clone / "bot/modules/topic.py").write_text('def respond(text, memory):\n    return None\n')
    from flock.world.workspace import capture_v2
    v2 = {**capture_v2(clone.parent, clone.name), "base_sha": "1" * 40}
    args = {"branch": "topic", "title": "Topic module", "description": "tested"}
    with pytest.raises(ToolError, match="base is not a main commit"):  # a v1 resend can't help
        world.call_workspace("open_pr", args, Agent("cedar"), v2)
    with pytest.raises(ToolError) as e:
        world.call_workspace("run_bot", {"ref_or_path": "chatbot-swarm", "lines": ["hi"]},
                             Agent("cedar"), v2)
    assert e.value.code == "need_full"
    v1 = {**capture(clone.parent, clone.name), "base_sha": "1" * 40}
    with pytest.raises(ToolError, match="base is not a main commit"):
        world.call_workspace("open_pr", args, Agent("cedar"), v1)
    # the budget is checked before any apply
    monkeypatch.setattr(world, "_check_budget", lambda agent: (_ for _ in ()).throw(
        ToolError("budget", "open_budget")))
    monkeypatch.setattr(chatbot4, "apply_patch", lambda *a: pytest.fail("applied"))
    with pytest.raises(ToolError, match="budget"):
        world.call_workspace("open_pr", args, Agent("cedar"), capture_v2(clone.parent, clone.name))


# ---------------------------------------------------------------- v8 settings (all off by default)

def test_reviewer_cap_refuses_one_more_reviewer_and_hides_full_prs(setup, monkeypatch):
    monkeypatch.setenv("CHATBOT_MAX_REVIEWERS", "2")
    world, clone, _ = setup
    world = Chatbot4World(world.ws, world.repo)
    _module(clone, "weather", "Sunny!")
    rec = submit(world, clone, branch="weather")
    for who in ("birch", "pine"):
        world.review_pr(Agent(who), rec["pr"], rec["head_sha"], "request_changes", "Off-topic reply")
    assert world.list_prs(needs_review=True) == []
    with pytest.raises(ToolError) as exc:
        world.review_pr(Agent("oak"), rec["pr"], rec["head_sha"], "approve", "Looks good")
    assert exc.value.code == "review_full" and "already has 2 reviewers" in exc.value.message
    approve(world, rec, "birch")  # a reviewer of this change may review it again
    _module(clone, "weather", "Sunny today!")  # a new change: only its reviews count
    rec = submit(world, clone, branch="weather", pr=rec["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [rec["pr"]]
    approve(world, rec, "oak")
    merge(world, Agent("cedar"), rec["pr"])


def test_adds_module_reads_v1_and_v2_uploads(setup):
    from flock.world.workspace import capture_v2

    world, clone, _ = setup
    base = git(clone, "rev-parse", "HEAD")

    def adds():
        return (world._adds_module(capture(clone.parent, clone.name), base),
                world._adds_module(capture_v2(clone.parent, clone.name), base))

    assert adds() == (False, False)
    (clone / "tests/test_more.py").write_text("x = 1\n")
    (clone / "bot/modules/greeting.py").write_text('PRIORITY = -1000\n\ndef respond(text, memory):\n    return "Hi!"\n')
    assert adds() == (False, False)  # a test and an edit of an existing module
    (clone / "bot/modules/helpers").mkdir()
    (clone / "bot/modules/helpers/__init__.py").write_text("")
    assert adds() == (True, True)  # a new package
    shutil.rmtree(clone / "bot/modules/helpers")
    git(clone, "mv", "bot/modules/greeting.py", "bot/modules/hello.py")
    assert adds() == (True, True)  # renamed to a new module name
    assert world._adds_module({"files": None}, base) and world._adds_module({"patch": "%%"}, base)


def test_new_modules_scope_rations_only_prs_that_add_a_module(setup, monkeypatch):
    monkeypatch.setenv("CHATBOT_PR_OPEN_FLOOR", "1")
    monkeypatch.setenv("CHATBOT_PR_OPEN_BUDGET_SCOPE", "new_modules")
    world, clone, _ = setup
    world = Chatbot4World(world.ws, world.repo)
    _module(clone, "weather", None)
    first = submit(world, clone, "cedar", "weather")
    assert first["pr_budget"]["scope"] == "new_modules" and "budget_exempt" not in first
    with pytest.raises(ToolError) as exc:  # a second new module this hour: refused
        submit(world, clone, "birch", "weather-two")
    assert exc.value.code == "open_budget"
    (clone / "bot/modules/weather.py").unlink()
    _module(clone, "greeting", "Hi!")  # an edit of an existing module: never counted
    fix = submit(world, clone, "birch", "greeting-fix")
    assert fix["budget_exempt"] is True and len(world.state["pr_opens"]) == 1
    _module(clone, "news", None)
    with pytest.raises(ToolError, match="open a new PR to add one") as exc:
        submit(world, clone, "birch", "greeting-fix", pr=fix["pr"])
    assert exc.value.code == "open_budget"
    with pytest.raises(ValueError, match="all or new_modules"):
        monkeypatch.setenv("CHATBOT_PR_OPEN_BUDGET_SCOPE", "some")
        Chatbot4World(world.ws, world.repo)


# 3.11's copy.py tries `from org.python.core import ...`, which the sandbox's import audit
# refuses with PermissionError (not ImportError): the v8 runtime needs a world on 3.12+.
@pytest.mark.skipif(sys.version_info < (3, 12), reason="seed-v8 imports copy: Python 3.12+")
def test_v8_seed_runtime_passes_the_contract_and_merges(setup_v8):
    world, clone, _ = setup_v8
    assert world.fixed["bot/runtime.py"] == (ROOT / "worlds/chatbot4/seed-v8/bot/runtime.py").read_bytes()
    (clone / "bot/modules/names.py").write_text(
        'def observe(text, memory):\n'
        '    if text.lower().startswith("my name is "):\n'
        '        memory["name"] = text.split()[-1]\n\n'
        'def respond(text, memory):\n'
        '    if "my name" in text.lower() and "?" in text and "name" in memory:\n'
        '        return f"You are {memory[\'name\']}."\n'
        '    return None\n')
    rec = submit(world, clone, branch="names")
    approve(world, rec)
    merge(world, Agent("cedar"), rec["pr"])
    replies = world.run_bot("main", ["Hello", "My name is Sam", "What is my name?"])["replies"]
    assert replies == ["Hello!", "Hello!", "You are Sam."]
    (clone / "bot/runtime.py").write_text((ROOT / "worlds/chatbot4/seed/bot/runtime.py").read_text())
    with pytest.raises(ToolError) as exc:  # the v4 runtime is not this seed's
        submit(world, clone, branch="old-runtime")
    assert exc.value.code == "fixed_interface"


def test_record_verdict_round_trip_inbox_and_metrics(setup):
    world, clone, _ = setup
    rec = submit(world, clone)
    target = f"pr:{rec['pr']}"
    with pytest.raises(ToolError, match="main or pr"):
        world.record_verdict(Agent("iv-owl"), "best", "script", 0.5, "x")
    with pytest.raises(ToolError, match="no PR"):
        world.record_verdict(Agent("iv-owl"), "pr:999", "script", 0.5, "x")
    with pytest.raises(ToolError, match="confidence"):
        world.record_verdict(Agent("iv-owl"), "main", "agent", 1.5, "x")
    with pytest.raises(ToolError, match="confidence"):
        world.record_verdict(Agent("iv-owl"), "main", "agent", True, "x")
    with pytest.raises(ToolError, match="limit"):
        world.verdicts(limit="many")
    world.record_verdict(Agent("iv-owl"), "main", "agent", 0.7, "remembered my dog's name")
    out = world.call("record_verdict", {"target": target, "verdict": "script", "confidence": 0.8,
                                        "notes": "same reply twice"}, Agent("iv-fox"))
    assert out["verdict"] == "script" and out["sha"] == rec["head_sha"]
    inbox = world.call("list_prs", {}, Agent("cedar"))["inbox"]
    assert any(f"interviewer iv-fox judged PR #{rec['pr']}: script (0.8)" in line for line in inbox)
    restarted = Chatbot4World(world.ws, world.repo)
    assert [v["by"] for v in restarted.verdicts()["verdicts"]] == ["iv-fox", "iv-owl"]
    assert [v["verdict"] for v in restarted.verdicts("main")["verdicts"]] == ["agent"]
    assert len(restarted.verdicts(limit=1)["verdicts"]) == 1
    m = restarted.metrics()
    assert m["verdicts_total"] == 2 and m["main_verdicts_last_hour_agent"] == 1
    assert m["main_verdicts_last_hour_script"] == 0
    names = {t["name"]: t for t in world.manifest()["tools"]}
    assert {"record_verdict", "verdicts"} <= set(names)


def test_world_status_queue_refusals_shared_tests_and_counts(setup, monkeypatch):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "greeting", "Hey!"),
                                    lambda c: _module(c, "greeting", "Hi!"))
    with pytest.raises(ToolError, match="conflicts with main"):
        merge(world, Agent("oak"), second["pr"])
    third = clone.parent / "third"
    git(clone.parent, "clone", "-q", str(setup[2]), str(third))
    _module(third, "farewell", "Bye!")
    _commit(third)
    rec = submit(world, third, author="maple", branch="third")
    approve(world, rec)
    world.merge_pr(Agent("oak"), rec["pr"])
    world.state["merge_queue"][0]["ts"] -= 150  # queued 2.5 minutes ago
    runs = iter([{"ok": True, "status": "ok", "output": ""},
                 {"ok": False, "status": "timeout", "output": ""}])
    monkeypatch.setattr(chatbot4, "run_tests", lambda *a: next(runs))
    main = world.repo.sha()
    world._shared_tests(1, clone, main)
    world._shared_tests(1, clone, main)  # times out: merged without them (+1 run: the first merge)
    out = world.call("world_status", {}, Agent("steward"))
    assert out["merge_queue"] == [{"pr": rec["pr"], "position": 1, "queued_by": "oak",
                                   "minutes_waiting": 2.5, "tries": 0}]
    assert out["merge_queue_total"] == 1 and out["merges_last_hour"] == 1
    assert out["refusals_last_hour"] == {"by_code": {"conflict": 1}, "last": [
        {"pr": second["pr"], "code": "conflict", "minutes_ago": 0.0}]}
    tests = out["shared_tests"]
    assert (tests["main_sha"], tests["result"], tests["minutes_ago"]) == (main[:12], "timeout", 0.0)
    assert tests["last_hour"]["runs"] == 3 and tests["last_hour"]["timeouts"] == 1
    assert isinstance(tests["seconds"], float)
    # open: the refused PR (approved, refused on this head) and the queued one (approved)
    assert (out["open_prs"], out["approved_not_queued"], out["approved_refused"],
            out["unreviewed"], out["changes_requested"]) == (2, 0, 1, 0, 0)
    assert out["approved_refused_prs"] == [{"pr": second["pr"], "code": "conflict"}]
    prs = {}
    for name in ("four", "five", "six"):  # unreviewed, approved, changes requested
        _module(third, name, f"{name}!")
        _commit(third)
        prs[name] = submit(world, third, author=f"ash-{name}", branch=name)
    approve(world, prs["five"])
    _review(world, prs["six"], "birch", "request_changes")
    out = world.world_status()
    assert out["approved_not_queued_prs"] == [prs["five"]["pr"]]
    assert out["unreviewed_prs"] == [prs["four"]["pr"]]
    assert out["changes_requested_prs"] == [prs["six"]["pr"]]
    assert (out["open_prs"], out["approved_not_queued"], out["unreviewed"],
            out["changes_requested"]) == (5, 1, 1, 1)
    assert {"opens_last_hour", "new_pr_limit"} <= set(out)
    assert "world_status" in {t["name"] for t in world.manifest()["tools"]}


def test_agent_checkouts_are_capped_and_the_merger_takes_no_slot(setup, monkeypatch):
    world, clone, remote = setup
    rec = submit(world, clone)
    approve(world, rec)
    monkeypatch.setattr(chatbot4, "CHECKOUT_WAIT_S", 0.1)
    held = 0
    while world.checkouts.acquire(blocking=False):  # every checkout slot is taken by agents
        held += 1
    assert held == int(os.environ.get("CHATBOT_CHECKOUT_SLOTS", "8"))
    try:
        for call in (lambda: world.run_bot("main", ["hi"]),
                     lambda: submit(world, clone, author="pine", branch="other")):
            with pytest.raises(ToolError, match="busy checking out") as exc:
                call()
            assert exc.value.code == "busy"
        result = merge(world, Agent("cedar"), rec["pr"])  # the merger does not wait for them
        assert git(remote, "rev-parse", "main") == result["merged_sha"]
    finally:
        for _ in range(held):
            world.checkouts.release()
    world.checkouts = threading.BoundedSemaphore(1)  # a leaked permit would show at once
    with pytest.raises(ToolError) as exc:
        world.run_bot("pr:999", ["hi"])  # fails while holding the slot
    assert exc.value.code != "busy"
    assert world.run_bot("main", ["hi"])["ok"]
    assert world.run_bot("main", ["hi"])["ok"]


def test_a_create_whose_gap_wait_sees_main_move_opens_on_its_build(setup, monkeypatch):
    world, clone, remote = setup
    main, gaps, real_gap = world.repo.sha(), [], world._create_gap

    def gap_then_main_moves(agent, deadline):
        gaps.append(deadline)
        real_gap(agent, deadline)
        tree = git(remote, "rev-parse", "main^{tree}")
        moved = git(remote, "-c", "user.name=x", "-c", "user.email=x@flock.invalid",
                    "commit-tree", tree, "-p", "main", "-m", "a merge elsewhere")
        git(remote, "update-ref", "refs/heads/main", moved)
        world.repo.fetch()

    monkeypatch.setattr(world, "_create_gap", gap_then_main_moves)
    (clone / "bot/modules/greeting.py").write_text('def respond(text, memory):\n    return "Hey!"\n')
    rec = submit(world, clone)
    assert world.repo.hub.n == 1 and rec["pr"] == 1 and len(gaps) == 1  # one gap slot, one PR
    assert rec["base_sha"] == main != world.repo.sha()  # opened on the commit already built
    approve(world, rec)
    assert merge(world, Agent("cedar"), rec["pr"])["merged_sha"]  # the merge replays the stale base


def test_a_stale_pr_that_conflicts_is_refused_before_its_replay(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "greeting", "Hey!"),
                                    lambda c: _module(c, "greeting", "Hi!"))

    def no_replay(*args, **kwargs):
        raise AssertionError("the merge-tree pre-check should have refused it")

    real_commit = world.repo.commit
    monkeypatch.setattr(world.repo, "commit", no_replay)
    with pytest.raises(ToolError, match=r"conflicts with main \w+ in bot/modules/greeting.py:") as err:
        merge(world, Agent("oak"), second["pr"])
    assert err.value.code == "conflict"
    # A .gitattributes in the tree (the replay's checkout honours it, merge-tree here does
    # not): the pre-check stays silent and the replay decides.
    monkeypatch.setattr(world.repo, "commit", real_commit)
    third = setup[1].parent / "third"
    git(third.parent, "clone", "-q", str(setup[2]), str(third))
    (third / "bot/modules/.gitattributes").write_text("greeting.py merge=union\n")  # in a subfolder
    _commit(third)
    git(third, "push", "-q", "origin", "HEAD:main")
    world.repo.fetch()
    assert world.repo.precheck(second["base_sha"], second["head_sha"], world.repo.sha()) is None
    with pytest.raises(ToolError) as replayed:  # the replay's union merge is clean, its diff new
        merge(world, Agent("oak"), second["pr"])
    assert replayed.value.code == "rebase_changed"


def test_shared_tests_run_beside_the_bot_contract_which_still_refuses_first(setup, monkeypatch):
    world, _clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    monkeypatch.setattr(chatbot4, "MERGE_TESTS", "test_*.py")
    tests_started, trees = threading.Event(), []

    def fake_tests(tree, pattern, timeout):
        trees.append(tree)
        tests_started.set()
        return {"ok": False, "status": "exit 1", "output": "FAIL: test_x (t.T)"}

    def fake_bot(tree, lines, timeout=300):
        assert tests_started.wait(5), "the shared tests did not start beside the contract"
        trees.append(tree)
        return {"ok": False}

    monkeypatch.setattr(chatbot4, "run_tests", fake_tests)
    monkeypatch.setattr(chatbot4, "run", fake_bot)
    with pytest.raises(ToolError) as err:
        merge(world, Agent("cedar"), second["pr"])
    assert err.value.code == "contract" and trees[0] == trees[1]  # one tree, the contract's refusal
