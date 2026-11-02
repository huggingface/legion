"""Replace the dataset scaffold with the greeting seed, preserving git history.

Default: print a read-only change plan. --apply requires the expected main SHA;
Hub parent_commit prevents replacing concurrent changes. No service or job starts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

SEED = Path(__file__).resolve().parents[2] / "worlds/chatbot4/seed"


def seed_files():
    return {p.relative_to(SEED).as_posix(): p for p in SEED.rglob("*") if p.is_file()
            and "__pycache__" not in p.parts and not p.name.endswith(".pyc")}


def publish(repo_id, expected=None, apply=False):
    api = HfApi()
    before = api.repo_info(repo_id, repo_type="dataset").sha
    if expected is not None and before != expected:
        raise SystemExit("main changed since preparation; inspect the new revision before replacing it")
    files = seed_files()
    existing = api.list_repo_files(repo_id, repo_type="dataset", revision=before)
    deleted = sorted(set(existing) - set(files))
    plan = {"repo": repo_id, "before": before, "files": sorted(files), "delete": deleted}
    if not apply:
        return plan
    if not expected:
        raise SystemExit("--apply requires --expected-sha")
    operations = [CommitOperationDelete(name) for name in deleted]
    operations.extend(CommitOperationAdd(name, path) for name, path in files.items())
    commit = api.create_commit(repo_id, repo_type="dataset", revision="main", parent_commit=expected,
        operations=operations, commit_message="Replace knowledge-base scaffold with run4 chatbot seed")
    actual = set(api.list_repo_files(repo_id, repo_type="dataset", revision=commit.oid))
    if actual != set(files):
        raise RuntimeError("seed replacement has an unexpected file list")
    for name, path in files.items():
        remote = Path(hf_hub_download(repo_id, name, repo_type="dataset", revision=commit.oid))
        if hashlib.sha256(path.read_bytes()).digest() != hashlib.sha256(remote.read_bytes()).digest():
            raise RuntimeError(f"seed verification failed: {name}")
    return {**plan, "commit": commit.oid, "url": commit.commit_url, "verified": True}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="cmpatino/chatbot-swarm")
    parser.add_argument("--expected-sha")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(publish(args.repo, args.expected_sha, args.apply), indent=2))
