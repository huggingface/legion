"""Bounded local git snapshots for worlds advertising workspace-upload-v1.

The worker captures files itself; an agent supplies a path within its own workspace,
never a server path or a repository base SHA. Only git-tracked/nonignored files are
included, and symlinks and git metadata are refused. Transport lives outside tool args.
"""

import base64
import subprocess
from pathlib import Path, PurePosixPath

MAX_BYTES = 128 << 20
MAX_FILES = 10000


def git(root: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True,
                       timeout=30, check=False)
    if p.returncode:
        raise ValueError("expected a local git clone with a committed base")
    return p.stdout.decode().strip()


def capture(workdir: Path, path: str) -> dict:
    root = (workdir / path).resolve()
    if not root.is_relative_to(workdir.resolve()) or not root.is_dir():
        raise ValueError("path must be a git clone inside your workspace")
    top = Path(git(root, "rev-parse", "--show-toplevel")).resolve()
    if top != root:
        raise ValueError("path must name the root of your clone")
    # The world verifies this commit against its own repository, including ancestry.
    base = git(root, "merge-base", "HEAD", "refs/remotes/origin/main")
    names = git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    files, total = {}, 0
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
        data = f.read_bytes()
        files[name] = base64.b64encode(data).decode("ascii")
    return {"base_sha": base, "files": files}


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
