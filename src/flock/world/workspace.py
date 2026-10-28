"""Bounded local git snapshots for worlds advertising workspace-upload-v1 and -v2.

The worker captures files itself; an agent supplies a path within its own workspace,
never a server path or a repository base SHA. Only git-tracked/nonignored files are
included, and symlinks and git metadata are refused. Transport lives outside tool args.
v1 sends every file (base64); v2 sends a binary git patch against the base (`capture_v2`),
which the world applies on its own checkout of that base (`apply_patch`).
"""
from __future__ import annotations

import base64
import os
import re
import resource
import subprocess
import tempfile
from pathlib import Path, PurePosixPath

MAX_BYTES = 128 << 20
MAX_FILES = 10000
PATCH_MAX_BYTES = 20 << 20  # a larger v2 patch is sent as a v1 snapshot instead
# The agent owns its clone's config: never run a command it configured (fsmonitor, filters,
# external diff, textconv) or let its diff settings change the patch format.
SAFE = ("-c", "core.fsmonitor=false", "-c", "diff.noprefix=false", "-c", "diff.mnemonicPrefix=false",
        "-c", "diff.relative=false")


def git(root: Path, *args: str, data: bytes | None = None, env: dict | None = None) -> str:
    return git_bytes(root, *args, data=data, env=env).decode().strip()


def git_bytes(root: Path, *args: str, data: bytes | None = None, env: dict | None = None) -> bytes:
    p = subprocess.run(["git", *SAFE, "-C", str(root), *args], input=data, capture_output=True,
                       timeout=30, check=False, env=env)
    if p.returncode:
        raise ValueError("expected a local git clone with a committed base")
    return p.stdout


def _clone(workdir: Path, path: str) -> tuple[Path, str, list[str]]:
    """The clone's root, its base (merge base with origin/main) and the files to upload."""
    root = (workdir / path).resolve()
    if not root.is_relative_to(workdir.resolve()) or not root.is_dir():
        raise ValueError("path must be a git clone inside your workspace")
    top = Path(git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise ValueError("path must name the root of your clone")
    # The world verifies this commit against its own repository, including ancestry.
    base = git(root, "merge-base", "HEAD", "refs/remotes/origin/main")
    names = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    files, total = [], 0
    for name in dict.fromkeys(names):
        if not name:
            continue
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts:
            raise ValueError("unsafe file path")
        f = root / name
        if f.is_symlink() or not f.resolve().is_relative_to(root):
            raise ValueError("symlinks are not supported")
        if not f.exists():  # tracked deletion
            continue
        if not f.is_file() or len(files) >= MAX_FILES:
            raise ValueError("unsupported file or too many files")
        total += f.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("workspace snapshot exceeds 128 MiB")
        files.append(name)
    return root, base, files


def capture(workdir: Path, path: str) -> dict:
    root, base, names = _clone(workdir, path)
    return {"base_sha": base,
            "files": {n: base64.b64encode((root / n).read_bytes()).decode("ascii") for n in names}}


def capture_v2(workdir: Path, path: str, cap: int = PATCH_MAX_BYTES) -> dict | None:
    """The clone as `{"base_sha", "patch"}`: the same files `capture` would send, staged into a
    temporary index, as a base64 `git diff --binary` against the base. None when the patch is
    over `cap` bytes (send `capture` instead)."""
    root, base, names = _clone(workdir, path)
    if any("\n" in n for n in names):
        return None  # hash-object --stdin-paths reads lines
    # hash-object --no-filters instead of `git add`: add would run the clone's clean filters.
    shas = git(root, "hash-object", "-w", "--no-filters", "--stdin-paths",
               data="".join(n + "\n" for n in names).encode()).split()
    info = "".join(f"{'100755' if (root / n).stat().st_mode & 0o100 else '100644'} {sha}\t{n}\0"
                   for n, sha in zip(names, shas, strict=True))
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "GIT_INDEX_FILE": str(Path(tmp) / "index")}
        git(root, "update-index", "-z", "--index-info", data=info.encode(), env=env)
        patch = git_bytes(root, "diff", "--cached", "--binary", "--full-index", "--find-renames",
                          "--no-ext-diff", "--no-textconv", "--no-color", base, env=env)
    if len(patch) > cap:
        return None
    return {"base_sha": base, "patch": base64.b64encode(patch).decode("ascii")}


_LINK = re.compile(rb"^(?:new file mode|new mode|index \S+) 1[26]0000$", re.MULTILINE)
_SIZES = re.compile(rb"^(?:literal|delta) (\d+)$", re.MULTILINE)


def apply_patch(snapshot: dict, tree: Path) -> list[str]:
    """Apply a v2 upload onto `tree`, a git checkout of its `base_sha` (the world's own), in the
    worktree (not the index), so the world's excludes apply when it stages the tree. Returns
    the paths that differ from the base (added, modified, deleted), as `unpack` callers expect.
    ValueError for an invalid patch; symlinks and submodules are refused as in v1."""
    try:
        patch = base64.b64decode(snapshot.get("patch") or "", validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid workspace patch") from exc
    if git(tree, "rev-parse", "HEAD") != snapshot.get("base_sha"):
        raise ValueError("the tree is not a checkout of the patch's base")
    if _LINK.search(patch):
        raise ValueError("symlinks are not supported")
    # shortcut: a binary delta can expand past its declared size, up to RLIMIT_FSIZE per file;
    # decode delta headers if a world ever has to take uploads from untrusted callers.
    if (patch.count(b"\ndiff --git ") + patch.startswith(b"diff --git ") > MAX_FILES
            or sum(int(n) for n in _SIZES.findall(patch)) > MAX_BYTES):
        raise ValueError("workspace snapshot exceeds 128 MiB or 10,000 files")

    def limit() -> None:  # in the child: no file git apply writes exceeds the snapshot cap
        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_BYTES, MAX_BYTES))

    if patch:
        p = subprocess.run(["git", "-C", str(tree), "apply", "--binary", "--whitespace=nowarn", "-"],
                           input=patch, capture_output=True, timeout=120, check=False,
                           preexec_fn=limit)
        if p.returncode:
            raise ValueError("the workspace patch does not apply to its base")
    names = git(tree, "ls-files", "-z", "--modified", "--deleted", "--others", "--exclude-standard")
    return sorted({n for n in names.split("\0") if n})


def unpack(snapshot: dict, root: Path) -> None:
    files = snapshot.get("files")
    if not isinstance(files, dict) or len(files) > MAX_FILES:
        raise ValueError("invalid workspace snapshot")
    total = 0
    for name, encoded in files.items():
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts or "\\" in name:
            raise ValueError("unsafe file path")
        data = base64.b64decode(encoded, validate=True)
        total += len(data)
        if total > MAX_BYTES:
            raise ValueError("workspace snapshot exceeds 128 MiB")
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
