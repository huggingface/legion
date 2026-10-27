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
    monkeypatch.setenv("CHATBOT_PR_CREATE_GAP_S", "0")  # the gap has its own test
    monkeypatch.setenv("CHATBOT_HUB_OUTBOX_GAP_S", "0")
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
    assert world.merge_pr(Agent("birch"), number)["merged_sha"]


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


def test_concurrent_open_prs_never_run_two_hub_writes_at_once(setup, monkeypatch):
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
    for name in ("open", "ready", "comment", "close", "merge"):
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
    assert world.repo.hub.n == 1 and "reason=pr_create_gap" in caplog.text
    world.pr_gap, world.last_create = 0.4, time.monotonic()
    started = time.monotonic()
    assert submit(world, clone, author="birch", branch="second")["pr"] == 2  # waits the gap
    assert time.monotonic() - started >= 0.3
    world.pr_gap = 0
    assert world.hub_slots.acquire(timeout=1)  # another Hub write in progress
    try:
        with pytest.raises(ToolError, match="the Hub is busy") as exc:
            submit(world, clone, author="pine", branch="third")
    finally:
        world.hub_slots.release()
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
                    lambda: world.merge_pr(Agent("birch"), rec["pr"])):
        with pytest.raises(ToolError) as exc:
            attempt()
        assert (exc.value.code, exc.value.message) == ("paused", paused)
    assert approve(world, rec)["verdict"] == "approve"
    assert world.run_bot("main", ["hi"])["ok"]
    assert world.list_prs()[0]["pr"] == rec["pr"]
    flag.unlink()
    assert world.merge_pr(Agent("birch"), rec["pr"])["merged_sha"]
    other = submit(world, clone, branch="other")
    monkeypatch.setenv("CHATBOT_PRS_PAUSED", "1")  # start-time override
    world = Chatbot4World(world.ws, world.repo)
    with pytest.raises(ToolError, match="paused by the operators"):
        submit(world, clone, branch="third")
    assert world.close_pr(Agent("cedar"), other["pr"], "paused")["state"] == "closed"


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
        assert world.merge_pr(Agent("pine"), fast["pr"])["merged_sha"]
    finally:
        release.set()
        slow.join(10)
    # main moved while the slow open was in flight: it was rebased onto the new main
    assert results["slow"]["base_sha"] == world.repo.sha()


def test_racing_merges_never_both_land(setup, monkeypatch):
    import threading

    world, clone, remote = setup
    one = submit(world, clone, branch="one")
    (clone / "bot/modules/two.py").write_text('def respond(text, memory):\n    return None\n')
    two = submit(world, clone, "birch", "two")
    approve(world, one, "pine")
    approve(world, two, "pine")
    barrier, real, outcomes = threading.Barrier(2, timeout=20), world._run, []
    monkeypatch.setattr(world, "_run", lambda *a, **k: (barrier.wait(), real(*a, **k))[1])

    def merge(rec):
        try:
            outcomes.append(world.merge_pr(Agent("oak"), rec["pr"])["merged_sha"])
        except ToolError as exc:
            outcomes.append(exc)

    threads = [threading.Thread(target=merge, args=(rec,)) for rec in (one, two)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    merged = [o for o in outcomes if isinstance(o, str)]
    refused = [o for o in outcomes if isinstance(o, ToolError)]
    assert len(merged) == 1 and len(refused) == 1 and "changed" in str(refused[0])
    assert git(remote, "rev-parse", "main") == merged[0]
    assert len(world.list_prs("merged")) == 1


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
        world.merge_pr(Agent("birch"), rec["pr"])
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
    """pine's PR merges first; cedar's PR (from the fixture clone) is approved and goes stale."""
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
    world.merge_pr(Agent("pine"), first["pr"])
    with pytest.raises(ToolError, match="stale PR: .*approvals carry over"):
        world.merge_pr(Agent("cedar"), second["pr"])
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
    result = world.merge_pr(Agent("cedar"), second["pr"])
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
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated)
    assert world.merge_pr(Agent("cedar"), second["pr"])["head_sha"] == updated["head_sha"]


def test_change_request_still_blocks_after_a_clean_rebase(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    world.review_pr(Agent("oak"), second["pr"], second["head_sha"], "request_changes", "Add a test")
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    with pytest.raises(ToolError, match="requests changes"):
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated, "oak")
    assert world.merge_pr(Agent("cedar"), second["pr"])["reviewers"] == ["birch", "oak"]


def test_reviews_without_patch_id_only_count_for_their_exact_head(setup):
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    # State archived before patch ids existed, restored into a restarted world.
    state = json.loads(world.db.read_text())
    del state["prs"][str(second["pr"])]["reviews"]["birch"]["patch_id"]
    del state["prs"][str(second["pr"])]["reviews"]["birch"]["patch_id0"]
    world.db.write_text(json.dumps(state))
    world = Chatbot4World(world.ws, world.repo)
    assert world.list_prs(needs_review=True) == []  # still approved at its exact head
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated)
    assert world.merge_pr(Agent("cedar"), second["pr"])["merged_sha"]


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
    result = world.merge_pr(Agent("cedar"), second["pr"])
    assert result["reviewers"] == ["birch"] and result["head_sha"] == updated["head_sha"]


def test_change_request_carries_over_a_rebase_next_to_a_change_of_main(setup):
    world, clone, second = _adjacent_prs(setup)
    world.review_pr(Agent("oak"), second["pr"], second["head_sha"], "request_changes", "Add a test")
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    with pytest.raises(ToolError, match="requests changes"):
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated, "oak")
    assert world.merge_pr(Agent("cedar"), second["pr"])["reviewers"] == ["birch", "oak"]


def test_a_content_change_after_an_adjacent_rebase_needs_a_fresh_approval(setup):
    world, clone, second = _adjacent_prs(setup)
    _rebase(clone)
    _edit_line(clone, 5, "changed")
    _commit(clone, "different change")
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated)
    assert world.merge_pr(Agent("cedar"), second["pr"])["head_sha"] == updated["head_sha"]


def _drop_patch_id0(world, pr):
    """A review stored before patch_id0 existed, restored into a restarted world."""
    state = json.loads(world.db.read_text())
    del state["prs"][str(pr)]["reviews"]["birch"]["patch_id0"]
    world.db.write_text(json.dumps(state))
    return Chatbot4World(world.ws, world.repo)


def test_old_reviews_carry_by_their_patch_id_only(setup):
    # Away from main's change, the old formula still matches and the approval carries.
    world, clone, second = _two_prs(setup, lambda c: _module(c, "weather", None),
                                    lambda c: _module(c, "time", None))
    world = _drop_patch_id0(world, second["pr"])
    _rebase(clone)
    submit(world, clone, branch="two", pr=second["pr"])
    assert world.list_prs(needs_review=True) == []
    assert world.merge_pr(Agent("cedar"), second["pr"])["reviewers"] == ["birch"]


def test_old_reviews_do_not_carry_a_rebase_next_to_a_change_of_main(setup):
    # The base a review was made on is not stored, so patch_id0 is not backfilled.
    world, clone, second = _adjacent_prs(setup)
    world = _drop_patch_id0(world, second["pr"])
    _rebase(clone)
    updated = submit(world, clone, branch="two", pr=second["pr"])
    assert [r["pr"] for r in world.list_prs(needs_review=True)] == [second["pr"]]
    with pytest.raises(ToolError, match="must approve"):
        world.merge_pr(Agent("cedar"), second["pr"])
    approve(world, updated)
    assert world.merge_pr(Agent("cedar"), second["pr"])["merged_sha"]


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
    world.call("merge_pr", {"pr": rec["pr"]}, Agent("cedar"))  # own merge: no news
    assert "cedar" not in world.state["inbox"]


def test_merge_and_close_by_others_are_news_for_the_author(setup):
    world, clone, _ = setup
    world.operators = {"oak"}
    one = submit(world, clone, branch="first")
    _module(clone, "second", "Second")
    two = submit(world, clone, branch="second")
    _review(world, one, "birch", "approve")
    world.call("merge_pr", {"pr": one["pr"]}, Agent("birch"))
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
    world.db.write_text(json.dumps(state))
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
    world.merge_pr(Agent("cedar"), rec["pr"])
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
    assert world.hub_slots.acquire(timeout=1)  # another Hub write runs for a long time
    try:
        started = time.monotonic()
        for who in ("birch", "pine", "oak", "elm"):
            assert approve(world, rec, who)["verdict"] == "approve"
        assert time.monotonic() - started < 2 and hub.comments == []
        assert len(world.state["hub_outbox"]) == 4
    finally:
        world.hub_slots.release()
    assert _wait(lambda: len(hub.comments) == 4)
    assert set(posted_by) == {"chatbot4-hub-outbox"} and world.state["hub_outbox"] == []
    text = f"approve by birch on {rec['head_sha']}\n\nChecked the code and bot"
    assert hub.comments[0] == (rec["pr"], text)


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
    approve(world, rec)
    assert _wait(lambda: world.state["hub_outbox"] and world.state["hub_outbox"][0]["tries"] >= 2)
    result = world.merge_pr(Agent("cedar"), rec["pr"])  # the comment is still unposted
    assert git(remote, "rev-parse", "main") == result["merged_sha"] and hub.comments == []
    restarted = Chatbot4World(world.ws, world.repo)  # stops the old outbox, keeps the comment
    assert [e["pr"] for e in restarted.state["hub_outbox"]] == [rec["pr"]]
    fail[0] = False
    assert _wait(lambda: len(hub.comments) == 1) and hub.comments[0][0] == rec["pr"]
    assert _wait(lambda: restarted.state["hub_outbox"] == [])
    assert json.loads((world.ws / "prs.json").read_text())["hub_outbox"] == []


def test_outbox_is_bounded_and_close_flushes_within_the_timeout(setup, monkeypatch, caplog):
    import threading

    world, clone, _ = setup
    rec = submit(world, clone)
    world.outbox_max = 3
    assert world.hub_slots.acquire(timeout=1)
    try:
        with caplog.at_level("WARNING"):
            for who in ("birch", "pine", "oak", "elm", "ash"):
                approve(world, rec, who)
        assert [e["agent"] for e in world.state["hub_outbox"]] == ["oak", "elm", "ash"]
        assert "hub_outbox: over 3 comments, dropped the 1 oldest" in caplog.text
        threading.Timer(0.3, world.hub_slots.release).start()
        world.close(timeout=5)
    except BaseException:
        world.hub_slots.release()
        raise
    assert len(world.repo.hub.comments) == 3 and world.state["hub_outbox"] == []
    assert not world.drainer.is_alive()
    assert json.loads((world.ws / "prs.json").read_text())["hub_outbox"] == []


def _hub_moved_past_the_record(world, clone, remote):
    """Open a PR, then push a newer head to the Hub's refs/pr/N and branch behind the world's
    back (a push in the gap before a restart from an older archive)."""
    rec = submit(world, clone)
    _module(clone, "gap", "from the gap")
    _commit(clone, "pushed in the gap")
    gap = git(clone, "rev-parse", "HEAD")
    git(clone, "push", "-q", str(remote), f"+{gap}:refs/pr/{rec['pr']}", f"+{gap}:refs/heads/{rec['branch']}")
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
    assert git(remote, "rev-parse", f"refs/heads/{rec['branch']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]
    assert f"adopted Hub head for PR #{rec['pr']} after a stale lease" in caplog.text


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
    monkeypatch.setattr(world.repo, "fetch_ref", lambda *a, **k: pytest.fail("fetched"))
    _module(clone, "later", "a later change")
    new = submit(world, clone, pr=rec["pr"])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]


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
    world.repo.hub.refs = lambda: calls.append(1) or {f"refs/pr/{rec['pr']}": gap,
                                                      f"refs/heads/{rec['branch']}": gap}
    monkeypatch.setattr(world.repo, "fetch_pr_refs", lambda *a, **k: pytest.fail("fetched"))
    monkeypatch.setattr(world.repo, "fetch_ref", lambda *a, **k: pytest.fail("fetched"))
    caplog.set_level("WARNING")
    _module(clone, "later", "a later change")
    new = submit(world, clone, pr=rec["pr"])
    assert git(remote, "rev-parse", f"refs/pr/{rec['pr']}") == new["head_sha"]
    assert git(remote, "rev-parse", f"refs/heads/{rec['branch']}") == new["head_sha"]
    assert world.state["prs"][str(rec["pr"])]["head_sha"] == new["head_sha"]
    assert f"adopted Hub head for PR #{rec['pr']} after a stale lease (via API)" in caplog.text
    assert calls == [1]  # the branch push then leases on the adopted head: no second lookup


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
    repo.anonymous = False  # disabled (a local or private remote): straight to the token
    seen.clear()
    assert repo.fetch_ref(f"refs/heads/{rec['branch']}", "refs/flock/hub/x")
    assert seen == [(False, True, False)]


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
    assert repo.fetch_ref(f"refs/heads/{rec['branch']}", "refs/flock/hub/x")
    assert len(fetches) == 4
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
