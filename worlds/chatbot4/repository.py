"""Git transport and native Hub PRs. The world uses the shared experiment HF_TOKEN for writes."""
from __future__ import annotations

import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from flock.world import ToolError
from flock.world.server import redact

log = logging.getLogger(__name__)

_URL_CREDENTIALS = re.compile(r"(?<=://)[^/@\s]+@")
_HUB_HEADERS = ("x-error-code", "x-error-message", "x-request-id", "retry-after")


def _safe_text(text, limit):
    return _URL_CREDENTIALS.sub("[redacted]@", redact(text))[:limit]


def _log_hub_failure(method, repo_id, exc):
    """One WARNING with the status and diagnostic headers; never the exception text, body or
    request headers, which can hold secrets."""
    response = getattr(exc, "response", None)
    request = getattr(response, "request", None)
    where = ""
    if request is not None:
        where = f"{request.method} {urlsplit(str(request.url)).path}"
    headers = {name: _safe_text(str(response.headers[name]), 300 if name == "x-error-message" else 100)
               for name in _HUB_HEADERS if response is not None and name in response.headers}
    log.warning("Hub call %s failed for %s: status=%s request=%s headers=%s", method, repo_id,
                getattr(response, "status_code", None), where or "unknown", headers)


class Repository:
    def __init__(self, root: Path, remote: str, hub=None):
        self.root, self.remote, self.hub = root, remote, hub
        # Ref writes in the shared bare repository (fetch, update-ref) are serialized here,
        # not under the world's state lock; reads, clones and object writes run concurrently.
        self.refs_lock = threading.Lock()
        self.fetched = 0.0
        self.patch_ids = {}  # (base, head) -> stable patch-id or None
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_LFS_SKIP_SMUDGE": "1"}
        if hub is not None:
            # The URL, argv and repository config contain no token.
            askpass = root.parent / "git-askpass.py"
            askpass.write_text("#!/usr/bin/env python3\nimport os,sys\n"
                               "print('hf_user' if 'Username' in sys.argv[1] else os.environ['HF_TOKEN'])\n")
            askpass.chmod(0o700)
            self.env["GIT_ASKPASS"] = str(askpass)
        if not (root / "HEAD").exists():
            root.mkdir(parents=True, exist_ok=True)
            self.git("init", "--bare", str(root), at=root.parent)
            self.git("remote", "add", "origin", remote)
            bundle = root.parent / "repo.bundle"
            if bundle.exists():
                self.git("fetch", str(bundle), "+refs/*:refs/*")
                bundle.unlink()
        self.fetch()

    def git(self, *args, at=None, data=None):
        p = subprocess.run(["git", "-C", str(at or self.root), *args], input=data,
                           capture_output=True, env=self.env, timeout=120, check=False)
        if p.returncode:
            # Git's stderr can contain remote data or credentials: never expose it to agents;
            # the world's log gets it redacted.
            command = next((a for a in args if not a.startswith("-") and "=" not in a), "?")
            log.warning("git %s failed: exit code %s, stderr=%s", command, p.returncode,
                        _safe_text(p.stderr.decode(errors="replace").strip(), 500))
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return p.stdout.decode().strip()

    def fetch(self):
        with self.refs_lock:
            started = time.monotonic()
            self.git("fetch", "--no-write-fetch-head", "origin", "+refs/heads/main:refs/heads/main",
                     "+refs/pr/*:refs/pr/*")
            self.fetched = started

    def refresh(self, max_age):
        """Fetch unless a fetch started within max_age seconds. The world updates its own
        refs after every push and merge, so this only catches up with outside changes."""
        if time.monotonic() - self.fetched >= max_age:
            self.fetch()

    def update_ref(self, ref, sha):
        with self.refs_lock:
            self.git("update-ref", ref, sha)

    def sha(self, ref="main"):
        if ref.startswith("pr:") and ref[3:].isdigit():
            ref = f"refs/pr/{int(ref[3:])}"
        if ref != "main" and not ref.startswith("refs/pr/"):
            raise ToolError("ref must be main or pr:<number>")
        return self.git("rev-parse", "--verify", f"{ref}^{{commit}}")

    def ancestor(self, sha, target):
        if len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            return False
        p = subprocess.run(["git", "-C", str(self.root), "merge-base", "--is-ancestor", sha, target],
                           env=self.env, capture_output=True, timeout=10, check=False)
        return p.returncode == 0

    def merged_by(self, head, target="main"):
        """The first commit on target's first-parent chain that contains head."""
        out = self.git("rev-list", "--first-parent", "--ancestry-path", "--reverse",
                       f"{head}..{target}")
        return out.split("\n", 1)[0] or self.git("rev-parse", head)  # fast-forward: head is main

    def checkout(self, ref, dest):
        # --shared borrows the object store instead of copying it; objects are never pruned
        # while reachable, and new commits are fetched back into the bare repository.
        self.git("clone", "-q", "--shared", "--no-checkout", str(self.root), str(dest), at=dest.parent)
        self.git("checkout", "-q", "--detach", ref, at=dest)

    def staged(self, tree):
        """Stage the tree (honouring .gitignore) and list added/changed paths."""
        self.git("add", "-A", at=tree)
        names = self.git("diff", "--cached", "--name-only", "-z", "--no-renames",
                         "--diff-filter=AM", "HEAD", at=tree)
        return [name for name in names.split("\0") if name]

    def changed(self, base, head):
        """Paths added or modified between two commits."""
        out = self.git("diff", "--name-only", "-z", "--no-renames", "--diff-filter=AM", base, head)
        return [name for name in out.split("\0") if name]

    def patch_id(self, base, head):
        """Stable patch-id of the diff base..head (None for an empty diff), cached per pair."""
        key = (base, head)
        if key not in self.patch_ids:
            diff = subprocess.run(["git", "-C", str(self.root), "diff", base, head],
                                  capture_output=True, env=self.env, timeout=120, check=False)
            if diff.returncode:
                log.warning("git diff failed: exit code %s", diff.returncode)
                raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
            out = subprocess.run(["git", "-C", str(self.root), "patch-id", "--stable"], input=diff.stdout,
                                 capture_output=True, env=self.env, timeout=120, check=False)
            if out.returncode:
                log.warning("git patch-id failed: exit code %s", out.returncode)
                raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
            if len(self.patch_ids) >= 10000:
                self.patch_ids.clear()
            self.patch_ids[key] = out.stdout.decode().split()[0] if out.stdout.strip() else None
        return self.patch_ids[key]

    def commit(self, tree, parent, author, title, onto=None):
        """Commit tree on parent; with onto != parent, replay that commit onto it."""
        who = ("-c", f"user.name={author}", "-c", f"user.email={author}@flock.invalid")
        self.git("add", "-A", at=tree)
        self.git(*who, "commit", "--allow-empty", "-m", title, at=tree)
        sha = self.git("rev-parse", "HEAD", at=tree)
        if onto and onto != parent:
            self.git("checkout", "-q", "--detach", onto, at=tree)
            try:
                self.git(*who, "cherry-pick", "--allow-empty", "--keep-redundant-commits", sha, at=tree)
            except ToolError:
                subprocess.run(["git", "-C", str(tree), "cherry-pick", "--abort"], env=self.env,
                               capture_output=True, timeout=60, check=False)
                raise ToolError(f"your branch is based on {parent}, main is {onto}, and your "
                                "changes conflict with main: pull, rebase onto origin/main, "
                                "then open again", "stale_base") from None
            sha = self.git("rev-parse", "HEAD", at=tree)
        self.git("fetch", "--no-write-fetch-head", str(tree), sha)
        return sha

    def push(self, sha, ref, expected=None):
        options = [f"--force-with-lease={ref}:{expected}"] if expected else []
        self.git("push", *options, "origin", f"{sha}:{ref}")


class Hub:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ToolError("HF_TOKEN is required for the bucket and dataset", "configuration")
        self.api = HfApi(token=token)

    def request(self, method, **kwargs):
        # Retry 429 only (only merges call this under the world's state lock); other
        # failures are surfaced without exception bodies, which may include secrets.
        for attempt in range(5):
            try:
                return getattr(self.api, method)(self.repo_id, repo_type="dataset", **kwargs)
            except HfHubHTTPError as exc:
                if exc.response.status_code != 429 or attempt == 4:
                    _log_hub_failure(method, self.repo_id, exc)
                    raise ToolError("Hub request failed; retry later", "hub_error") from None
                delay = exc.response.headers.get("Retry-After", "1")
                try:
                    delay = float(delay)
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(max(0, min(delay, 30)))
            except Exception as exc:
                # The world reports this as an internal error with an id; name the class here.
                log.warning("Hub call %s failed for %s: %s", method, self.repo_id, type(exc).__name__)
                raise

    def open(self, title, description):
        pr = self.request("create_pull_request", title=title, description=description)
        return pr.num

    def ready(self, pr):
        self.request("change_discussion_status", discussion_num=pr, new_status="open")

    def comment(self, pr, text):
        self.request("comment_discussion", discussion_num=pr, comment=text)

    def close(self, pr, comment):
        self.request("change_discussion_status", discussion_num=pr, new_status="closed",
                     comment=comment)

    def status(self, pr):
        return self.request("get_discussion_details", discussion_num=pr).status

    def merge(self, pr, comment):
        self.request("merge_pull_request", discussion_num=pr, comment=comment)
