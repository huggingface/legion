"""workspace-upload-v2: capture a clone as a binary patch, apply it on the world's checkout."""
from __future__ import annotations

import base64
import os
import subprocess

import pytest

from flock.world.workspace import apply_patch, capture, capture_v2


def git(path, *args):
    return subprocess.run(["git", "-C", str(path), "-c", "user.name=t", "-c", "user.email=t@t",
                           *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repos(tmp_path):
    """(agent workdir, agent clone, world's bare repository) sharing one seed commit."""
    seed = tmp_path / "seed"
    (seed / "src").mkdir(parents=True)
    (seed / "src/a.py").write_text("a = 1\n")
    (seed / "src/old.py").write_text("old = 1\n" * 50)
    (seed / "gone.txt").write_text("bye\n")
    (seed / "run.sh").write_text("echo hi\n")
    (seed / "blob.bin").write_bytes(bytes(range(256)) * 40)
    (seed / ".gitignore").write_text("*.log\n")
    git(tmp_path, "init", "-q", "-b", "main", str(seed))
    git(seed, "add", ".")
    git(seed, "commit", "-qm", "seed")
    bare = tmp_path / "world.git"
    git(tmp_path, "clone", "-q", "--bare", str(seed), str(bare))
    work = tmp_path / "work"
    work.mkdir()
    git(tmp_path, "clone", "-q", str(bare), str(work / "clone"))
    return work, work / "clone", bare


def materialise(bare, snapshot, tree):
    git(bare.parent, "clone", "-q", "--shared", "--no-checkout", str(bare), str(tree))
    git(tree, "checkout", "-q", "--detach", snapshot["base_sha"])
    return apply_patch(snapshot, tree)


def files(root):
    return {p.relative_to(root).as_posix(): (p.read_bytes(), os.access(p, os.X_OK))
            for p in root.rglob("*") if p.is_file() and ".git" not in p.relative_to(root).parts}


def test_round_trip_matches_the_clone(repos, tmp_path):
    work, clone, bare = repos
    (clone / "src/a.py").write_text("a = 2\n")  # modified
    (clone / "gone.txt").unlink()  # deleted
    (clone / "run.sh").chmod(0o755)  # mode change
    (clone / "blob.bin").write_bytes(bytes(range(255, -1, -1)) * 41)  # binary
    (clone / "img.png").write_bytes(os.urandom(3000))  # new binary, untracked
    (clone / "notes.md").write_text("untracked\n")  # untracked, not ignored
    (clone / "debug.log").write_text("ignored\n")  # ignored: never sent
    git(clone, "mv", "src/old.py", "src/new.py")  # rename
    git(clone, "commit", "-qam", "local")
    (clone / "src/a.py").write_text("a = 3\n")  # uncommitted edit on top
    snap = capture_v2(work, "clone")
    assert set(snap) == {"base_sha", "patch"}
    assert b"rename from src/old.py" in base64.b64decode(snap["patch"])
    changed = materialise(bare, snap, tmp_path / "tree")
    want = {k: v for k, v in files(clone).items() if k != "debug.log"}
    assert files(tmp_path / "tree") == want
    assert set(changed) == {"src/a.py", "gone.txt", "run.sh", "blob.bin", "img.png", "notes.md",
                            "src/old.py", "src/new.py"}
    # v1 is unchanged: the same files, as base64
    v1 = capture(work, "clone")
    assert v1["base_sha"] == snap["base_sha"]
    assert {k: base64.b64decode(v) for k, v in v1["files"].items()} == {k: v[0] for k, v in want.items()}


def test_unchanged_clone_is_an_empty_patch(repos, tmp_path):
    work, _, bare = repos
    snap = capture_v2(work, "clone")
    assert snap["patch"] == ""
    assert materialise(bare, snap, tmp_path / "tree") == []


def test_symlinks_are_refused_both_ways(repos, tmp_path):
    work, clone, bare = repos
    (clone / "link").symlink_to("src/a.py")
    with pytest.raises(ValueError, match="symlinks"):
        capture_v2(work, "clone")
    (clone / "link").unlink()
    # a hand-made patch adding a symlink is refused by the world
    other = tmp_path / "other"
    git(tmp_path, "clone", "-q", str(bare), str(other))
    (other / "link").symlink_to("/etc/passwd")
    git(other, "add", "link")
    patch = subprocess.run(["git", "-C", str(other), "diff", "--cached", "--binary", "HEAD"],
                           capture_output=True, check=True).stdout
    snap = {"base_sha": git(other, "rev-parse", "HEAD"), "patch": base64.b64encode(patch).decode()}
    with pytest.raises(ValueError, match="symlinks"):
        materialise(bare, snap, tmp_path / "tree")


def test_world_refuses_bad_patches(repos, tmp_path):
    work, clone, bare = repos
    (clone / "src/a.py").write_text("a = 2\n")
    snap = capture_v2(work, "clone")
    tree = tmp_path / "tree"
    git(bare.parent, "clone", "-q", "--shared", str(bare), str(tree))
    escape = (b"diff --git a/../x b/../x\nnew file mode 100644\n--- /dev/null\n+++ b/../x\n"
              b"@@ -0,0 +1 @@\n+x\n")
    dotgit = escape.replace(b"../x", b".git/hooks/post-commit")
    link = (b"diff --git a/l b/l\nnew file mode %s\n--- /dev/null\n+++ b/l\n@@ -0,0 +1 @@\n"
            b"+/etc/passwd\n\\ No newline at end of file\n")
    sub = (b"diff --git a/sub b/sub\nnew file mode 0160000\n--- /dev/null\n+++ b/sub\n@@ -0,0 +1 @@\n"
           b"+Subproject commit " + b"1" * 40 + b"\n")
    links = [link % m for m in (b"0120000", b"120755", b"120000\r")]
    for patch in (escape, dotgit, b"not a patch\n", *links, sub):
        with pytest.raises(ValueError):
            apply_patch({**snap, "patch": base64.b64encode(patch).decode()}, tree)
    with pytest.raises(ValueError, match="base"):
        apply_patch({**snap, "base_sha": "0" * 40}, tree)
    assert not (tmp_path / "x").exists() and not (tree / ".git/hooks/post-commit").exists()
    assert not (tree / "l").is_symlink() and not (tree / "sub").exists()


def test_size_cap_falls_back(repos):
    work, clone, _ = repos
    (clone / "big.bin").write_bytes(os.urandom(4096))
    assert capture_v2(work, "clone", cap=1000) is None
    assert capture_v2(work, "clone") is not None


def test_clone_filters_never_run(repos, tmp_path):
    work, clone, _ = repos
    marker = tmp_path / "ran"
    git(clone, "config", "filter.x.clean", f"touch {marker}; cat")
    (clone / ".gitattributes").write_text("* filter=x\n")
    (clone / "src/a.py").write_text("a = 2\n")
    loose = len(list((clone / ".git/objects").rglob("*")))
    assert capture_v2(work, "clone") is not None
    assert not marker.exists()
    assert len(list((clone / ".git/objects").rglob("*"))) == loose  # no objects left in the clone


OK = {"ok": True, "result": "done"}
NEED_FULL = {"ok": False, "error": {"code": "need_full", "message": "no base"}}


def test_server_counts_uploads_and_answers_need_full_without_v2():
    from fastapi.testclient import TestClient

    from flock.world import World, tool
    from flock.world.server import create_app

    class Up(World):
        name, version = "up", "1"
        capabilities = ("workspace-upload-v1",)

        @tool
        def put(self) -> str:
            """Upload."""
            return "plain"

        def call_workspace(self, name, args, agent, snapshot):
            return sorted(snapshot)

    world = Up()
    c = TestClient(create_app(world, None))
    h = {"X-Flock-Agent": "a"}
    v1 = {"args": {}, "workspace": {"base_sha": "b", "files": {}}}
    v2 = {"args": {}, "workspace": {"base_sha": "b", "patch": ""}}
    assert c.post("/tools/put", json=v1, headers=h).json()["ok"]
    assert c.post("/tools/put", json=v2, headers=h).json()["error"]["code"] == "need_full"
    world.capabilities = ["workspace-upload-v1", "workspace-upload-v2"]
    assert c.post("/tools/put", json=v2, headers=h).json()["result"] == ["base_sha", "patch"]
    m = c.get("/metrics").json()
    assert (m["upload_v1_total"], m["upload_v2_total"]) == (1, 1)


def test_upload_stays_inside_the_workspace(repos):
    work, clone, _ = repos
    with pytest.raises(ValueError, match="inside"):
        capture_v2(work, "..")
    (clone / "leak").symlink_to("/etc/passwd")
    with pytest.raises(ValueError, match="symlinks"):
        capture_v2(work, "clone")


def test_a_git_that_times_out_is_a_tool_error(repos, monkeypatch):
    def slow(argv, **_):
        raise subprocess.TimeoutExpired(argv, 30)

    monkeypatch.setattr(subprocess, "run", slow)
    with pytest.raises(ValueError, match="timed out after 30 s"):
        capture_v2(repos[0], "clone")
