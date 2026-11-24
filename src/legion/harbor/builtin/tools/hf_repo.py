"""Git checkouts and native HF dataset PRs; reviews are advisory, not access controls."""

import importlib.metadata
import json
import os
import random
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import islice
from pathlib import Path
from typing import Literal

import httpx
from huggingface_hub import HfApi, set_client_factory
from huggingface_hub.community import deserialize_event

# PR APIs need no native Xet transfer client. Package SDK/YAML Python files and
# metadata, rather than their native optional dependencies. Metadata is retained
# for SDK version checks. RECORD is omitted by the portable bundler, so files()
# can return None when this declaration is evaluated inside an agent.
DEPENDENCIES = [
    "httpx",
    "filelock",
    "fsspec",
    "packaging",
    "tqdm",
    "typing-extensions",
]
BUNDLE_FILES = {
    name: [
        str(p)
        for p in (importlib.metadata.files(name) or [])
        if str(p).endswith(".py") or (
            ".dist-info/" in str(p) and (p.name == "METADATA" or "licenses" in p.parts)
        )
    ]
    for name in ("huggingface_hub", "PyYAML", "tomli")
}
REQUIRES = ["hf_repo"]
LIMIT = 24000


def _settings():
    repo = os.environ["LEGION_HF_REPO"]
    agent = os.environ["LEGION_HF_REPO_AGENT"]
    token = os.environ["LEGION_HF_REPO_TOKEN"]
    if not re.fullmatch(r"[\w-]+/[\w.-]+", repo) or not re.fullmatch(r"[\w-]+", agent):
        raise ValueError("Invalid repository binding")
    return repo, agent, token


def _api():
    set_client_factory(lambda: httpx.Client(timeout=30, follow_redirects=True))
    return HfApi(endpoint="https://huggingface.co", token=_settings()[2])


def _retry(operation, label):
    for attempt in range(4):
        try:
            return operation()
        except Exception as exc:
            response = getattr(exc, "response", None)
            code = getattr(response, "status_code", None)
            if code != 429 or attempt == 3:
                raise RuntimeError(
                    f"HF {label} failed (HTTP {code or 'unknown'}); inspect state before retrying"
                ) from None
            delay = 2 ** attempt
            header = response.headers.get("Retry-After")
            if header:
                try:
                    delay = max(delay, float(header))
                except ValueError:
                    try:
                        delay = max(delay, (parsedate_to_datetime(header)
                                            - datetime.now(timezone.utc)).total_seconds())
                    except (TypeError, ValueError, OverflowError):
                        pass
            # Do not tie up an agent indefinitely or retry sooner than requested.
            if delay > 60:
                raise RuntimeError(f"HF {label} rate limited; retry after {delay:.0f}s") from None
            time.sleep(delay + random.uniform(0, 1))


def _call(method, **kwargs):
    return _retry(
        lambda: getattr(_api(), method)(_settings()[0], repo_type="dataset", **kwargs),
        method,
    )


def _url():
    return f"https://huggingface.co/datasets/{_settings()[0]}"


def _git(path, *args):
    _, agent, token = _settings()
    with tempfile.TemporaryDirectory(prefix="legion-git-") as temp:
        askpass = Path(temp) / "askpass"
        askpass.write_text(
            '#!/bin/sh\ncase "$1" in *Username*) echo hf_user;; *) printf "%s\\n" "$LEGION_HF_REPO_TOKEN";; esac\n'
        )
        askpass.chmod(0o700)
        env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
        env.update(
            GIT_TERMINAL_PROMPT="0",
            GIT_CONFIG_NOSYSTEM="1",
            GIT_CONFIG_GLOBAL="/dev/null",
            GIT_LFS_SKIP_SMUDGE="1",
            GIT_ASKPASS=str(askpass),
            LEGION_HF_REPO_TOKEN=token,
        )
        try:
            proc = subprocess.run(
                [
                    "git",
                    "-c",
                    "credential.helper=",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "protocol.file.allow=never",
                    "-c",
                    f"user.name={agent}",
                    "-c",
                    f"user.email={agent}@legion.invalid",
                    "-C",
                    str(path),
                    *args,
                ],
                env=env,
                check=False,
                capture_output=True,
                timeout=120,
            )
        except FileNotFoundError:
            raise RuntimeError(
                "Git is required in the task image; install Git before using hf_repo"
            ) from None
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                "Git timed out after 120 seconds; inspect state before retrying"
            ) from None
    if proc.returncode:
        raise RuntimeError(
            "Git operation failed: " + proc.stderr.decode(errors="replace").replace(token, "[REDACTED]")[-800:]
        )
    return proc.stdout.decode(errors="replace").strip()


def _checkout(path, clean=False):
    root = Path(path).resolve()
    if _git(root, "rev-parse", "--show-toplevel") != str(root):
        raise ValueError("path must be the root of a collaboration checkout")
    if (
        _git(root, "remote", "get-url", "--all", "origin") != _url()
        or _git(root, "remote", "get-url", "--push", "--all", "origin") != _url()
    ):
        raise ValueError("Checkout origin does not match this run's repository")
    if clean and _git(root, "status", "--porcelain"):
        raise ValueError("Checkout has local changes; commit or resolve them first")
    return root


def _bounded(text):
    return {"text": text[:LIMIT], "truncated": len(text) > LIMIT}


def _pr(pr):
    if pr < 1:
        raise ValueError("PR number must be positive")

    def read():
        # The SDK always requests diff=1. Read metadata directly to avoid full Hub diffs.
        with httpx.Client(timeout=30) as client:
            response = client.get(
                f"https://huggingface.co/api/datasets/{_settings()[0]}/discussions/{pr}",
                headers={"Authorization": f"Bearer {_settings()[2]}"},
            )
            response.raise_for_status()
            return response.json()

    detail = _retry(read, "get_discussion_details")
    if not detail["isPullRequest"]:
        raise ValueError("Discussion is not a pull request")
    return detail


def _head(pr):
    # Target just this PR rather than enumerating every repository ref.
    with tempfile.TemporaryDirectory(prefix="legion-ref-") as temp:
        result = _git(temp, "ls-remote", "--", _url(), f"refs/pr/{pr}")
    return result.split()[0] if result else None


def clone(path: str) -> dict:
    """Clone into a new or empty directory, separate from the task workspace. Create a working branch before editing; keep main for sync."""
    target = Path(path).resolve()
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ValueError("Clone destination must be new or empty")
    target.parent.mkdir(parents=True, exist_ok=True)
    _git(target.parent, "clone", "--", _url(), str(target))
    return {
        "path": str(target),
        "repo_id": _settings()[0],
        "head_sha": _git(target, "rev-parse", "HEAD"),
    }


def status(path: str) -> dict:
    """Show local changes and HEAD in this run's collaboration checkout."""
    root = _checkout(path)
    return {
        "head_sha": _git(root, "rev-parse", "HEAD"),
        **_bounded(_git(root, "status", "--short", "--branch")),
    }


def diff(path: str) -> dict:
    """Show a bounded diff of tracked local changes against HEAD; untracked files appear in status."""
    return _bounded(
        _git(_checkout(path), "diff", "--no-ext-diff", "--no-textconv", "HEAD", "--")
    )


def commit(path: str, files: list[str], message: str) -> dict:
    """Commit explicitly named original source files (including deletions). Never submit task binaries, secrets or hidden tests."""
    root = _checkout(path)
    if not files or not message.strip():
        raise ValueError("Provide explicit files and a commit message")
    if _git(root, "diff", "--cached", "--name-only"):
        raise ValueError(
            "Index already contains staged changes; resolve them before calling commit"
        )
    for name in files:
        p = Path(name)
        if p.is_absolute() or ".." in p.parts or ".git" in p.parts or name in ("", "."):
            raise ValueError("Provide relative file paths inside the checkout")
        dest = root / p
        if (
            dest.is_symlink()
            or not dest.resolve().is_relative_to(root)
            or dest.is_dir()
        ):
            raise ValueError("Only individual files inside the checkout can be staged")
        if p.name == "executable" or p.name == ".env" or p.name.startswith(".env."):
            raise ValueError(
                "Do not publish reference executables or environment credentials"
            )
    _git(root, "--literal-pathspecs", "add", "--", *files)
    _git(root, "commit", "-m", message)
    return {"head_sha": _git(root, "rev-parse", "HEAD")}


def open_pr(path: str, title: str, description: str) -> dict:
    """Open a PR from committed changes. If publication fails, recover with update_pr and the returned PR number; do not open a duplicate."""
    root = _checkout(path, clean=True)
    if not 3 <= len(title.strip()) <= 200 or not description.strip():
        raise ValueError("Provide a 3–200 character title and nonempty description")
    _git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    if _git(root, "merge-base", "origin/main", "HEAD") != _git(root, "rev-parse", "origin/main"):
        raise ValueError("Branch is behind main; merge origin/main first")
    if not _git(root, "diff", "--name-only", "origin/main...HEAD"):
        raise ValueError("No changes relative to main")
    pr = _call(
        "create_pull_request",
        title=title,
        description=f"Legion agent: {_settings()[1]}\n\n{description}",
    )
    # Record the recovery handle before touching the remote ref.
    try:
        _git(root, "config", "legion.lastPr", str(pr.num))
        return update_pr(str(root), pr.num)
    except Exception as exc:
        return {
            "pr": pr.num,
            "url": f"{_url()}/discussions/{pr.num}",
            "published": False,
            "error": str(exc),
            "next_action": "update_pr",
        }


def update_pr(path: str, pr: int) -> dict:
    """Publish committed changes to a PR. Uses an observed-head lease; rejects non-fast-forward updates."""
    root = _checkout(path, clean=True)
    detail = _pr(pr)
    if detail["status"] not in ("open", "draft"):
        raise ValueError("PR is already closed or merged")
    ref = f"refs/pr/{pr}"
    observed = f"refs/legion/pr/{pr}"
    _git(root, "fetch", "origin", f"+{ref}:{observed}")
    previous = _git(root, "rev-parse", observed)
    _git(root, "merge-base", "--is-ancestor", previous, "HEAD")
    _git(root, "push", f"--force-with-lease={ref}:{previous}", "origin", f"HEAD:{ref}")
    if detail["status"] == "draft":
        _call("change_discussion_status", discussion_num=pr, new_status="open")
    return {
        "pr": pr,
        "url": f"{_url()}/discussions/{pr}",
        "published": True,
        "head_sha": _git(root, "rev-parse", "HEAD"),
    }


def list_prs(
    state: Literal["all", "open", "draft", "closed", "merged"] = "open",
    limit: int = 100,
) -> dict:
    """List at most limit native HF PRs, filtered on the Hub. Author is the shared account."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")

    def read():
        return list(islice(_api().get_repo_discussions(
            _settings()[0], repo_type="dataset", discussion_type="pull_request",
            discussion_status=state if state in ("open", "closed") else "all",
        ), limit + 1))

    prs = _retry(read, "get_repo_discussions")
    return {
        "pull_requests": [
            {"pr": p.num, "title": p.title, "author": p.author, "state": p.status,
             "url": f"{_url()}/discussions/{p.num}"}
            for p in prs[:limit] if state == "all" or p.status == state
        ],
        "truncated": len(prs) > limit,
    }


def inspect_pr(path: str, pr: int, paths: list[str] | None = None) -> dict:
    """Inspect metadata, file statistics and a bounded local merge-base diff for one exact SHA."""
    root = _checkout(path)
    p = _pr(pr)
    _git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    _git(root, "fetch", "origin", f"refs/pr/{pr}")
    # Resolve the fetched ref once: metadata and concurrent remote updates cannot relabel this diff.
    sha = _git(root, "rev-parse", "FETCH_HEAD")
    comments = [
        {"author": e.author, **_bounded(e.content)}
        for e in (deserialize_event(event) for event in p["events"])
        if e.type == "comment"
    ]
    args = ("--literal-pathspecs", "diff", "--no-ext-diff", "--no-textconv")
    return {
        "pr": pr, "title": p["title"], "state": p["status"], "head_sha": sha,
        "comments": comments[-30:], "comments_truncated": len(comments) > 30,
        "stat": _bounded(_git(root, *args, "--stat", f"origin/main...{sha}", "--", *(paths or []))),
        "diff": _bounded(_git(root, *args, f"origin/main...{sha}", "--", *(paths or []))),
        "url": f"{_url()}/discussions/{pr}",
    }


def close_pr(pr: int, comment: str) -> dict:
    """Close an abandoned PR with an explanation; never deletes its history."""
    p = _pr(pr)
    if not comment.strip():
        raise ValueError("Provide a reason for closing the PR")
    if p["status"] == "merged":
        raise ValueError("Cannot close a merged PR")
    if p["status"] != "closed":
        _call("change_discussion_status", discussion_num=pr, new_status="closed", comment=comment)
    return {"pr": pr, "state": "closed"}


def checkout_pr(path: str, pr: int) -> dict:
    """Fetch a PR into a new local review branch. Refuses dirty worktrees and never resets an existing branch."""
    root = _checkout(path, clean=True)
    _pr(pr)
    _git(root, "fetch", "origin", f"refs/pr/{pr}")
    sha = _git(root, "rev-parse", "FETCH_HEAD")
    _git(root, "switch", "-c", f"review/pr-{pr}-{sha[:12]}", sha)
    return {"pr": pr, "head_sha": sha}


def review_pr(
    pr: int,
    head_sha: str,
    verdict: Literal["approve", "request_changes", "comment"],
    comment: str,
) -> dict:
    """Post an advisory review tied to the observed head SHA. This does not enforce approval before merge."""
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha) or not comment.strip():
        raise ValueError("Provide the reviewed full commit SHA and a comment")
    _pr(pr)
    if _head(pr) != head_sha:
        raise ValueError("PR head changed; inspect and review the new commit")
    review = {"agent": _settings()[1], "head_sha": head_sha, "verdict": verdict}
    _call(
        "comment_discussion",
        discussion_num=pr,
        comment=f"Legion review\n```json\n{json.dumps(review)}\n```\n\n{comment}",
    )
    return {"pr": pr, **review}


def merge_pr(pr: int) -> dict:
    """Merge through HF. Reviews are advisory; inspect current changes before invoking. Conflicts are returned, not auto-resolved."""
    p = _pr(pr)
    if p["status"] == "merged":
        return {"pr": pr, "state": "merged"}
    if p["status"] != "open":
        raise ValueError("Only open PRs can be merged")
    _call(
        "merge_pull_request",
        discussion_num=pr,
        comment=f"Merged by Legion agent {_settings()[1]}",
    )
    return {"pr": pr, "state": "merged", "url": f"{_url()}/discussions/{pr}"}


def sync(path: str) -> dict:
    """Refresh origin/main from any clean branch; fast-forward local main only when checked out."""
    root = _checkout(path, clean=True)
    _git(root, "fetch", "origin", "refs/heads/main:refs/remotes/origin/main")
    if _git(root, "branch", "--show-current") == "main":
        _git(root, "merge", "--ff-only", "origin/main")
    return {"head_sha": _git(root, "rev-parse", "HEAD")}


TOOLS = {
    f.__name__: f
    for f in (
        clone,
        status,
        diff,
        commit,
        open_pr,
        update_pr,
        list_prs,
        inspect_pr,
        checkout_pr,
        review_pr,
        merge_pr,
        close_pr,
        sync,
    )
}
