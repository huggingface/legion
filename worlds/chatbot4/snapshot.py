"""One archive for world state; agent clones live in their own worker snapshots."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath

from flock.worker.sync import HubBucketStore

FILES = ("repo.bundle", "prs.json", "prs.jsonl", "eval-queue.jsonl")


def transfer(mode, workspace, bucket, store=None):
    store = store or HubBucketStore(bucket, token=os.environ.get("HF_TOKEN"))
    if mode == "restore" and "world.tar.gz" not in store.list("world.tar.gz"):
        return False  # a first start; permission/network/invalid-archive failures still raise
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=workspace.parent) as tmp:
        archive = Path(tmp) / "world.tar.gz"
        if mode == "push":
            bundle = Path(tmp) / "repo.bundle"
            subprocess.run(["git", "-C", str(workspace / "repo.git"), "bundle", "create",
                            str(bundle), "--all"], check=True, capture_output=True)
            with tarfile.open(archive, "w:gz", compresslevel=1) as out:
                out.add(bundle, arcname="repo.bundle")
                for name in FILES[1:]:
                    if (workspace / name).exists():
                        out.add(workspace / name, arcname=name)
            store.upload([(archive, "world.tar.gz")])
        else:
            store.download([("world.tar.gz", archive)])
        if mode == "restore":
            with tarfile.open(archive, "r:gz") as source:
                for member in source.getmembers():
                    path = PurePosixPath(member.name)
                    if (path.is_absolute() or ".." in path.parts or not path.parts
                            or path.parts[0] not in FILES or not (member.isfile() or member.isdir())):
                        raise ValueError("unexpected world archive entry")
                    target = workspace / path
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    else:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as src, target.open("wb") as dst:
                            shutil.copyfileobj(src, dst)

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["push", "restore"])
    parser.add_argument("workspace")
    parser.add_argument("bucket")
    args = parser.parse_args()
    if not transfer(args.mode, args.workspace, args.bucket):
        raise SystemExit(3)
