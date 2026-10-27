"""Git transport and native Hub PRs. The world uses the shared experiment HF_TOKEN for writes."""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from flock.world import ToolError


class Repository:
    def __init__(self, root: Path, remote: str, hub=None):
        self.root, self.remote, self.hub = root, remote, hub
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
            # Git's stderr can contain remote data or credentials: never expose it.
            raise ToolError("repository operation failed; refresh your clone and retry", "git_error")
        return p.stdout.decode().strip()

    def fetch(self):
        self.git("fetch", "origin", "+refs/heads/main:refs/heads/main",
                 "+refs/pr/*:refs/pr/*")

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

    def checkout(self, ref, dest):
        self.git("clone", "--no-hardlinks", str(self.root), str(dest), at=dest.parent)
        self.git("checkout", "--detach", ref, at=dest)

    def staged(self, tree):
        """Stage the tree (honouring .gitignore) and list added/changed paths."""
        self.git("add", "-A", at=tree)
        names = self.git("diff", "--cached", "--name-only", "-z", "--no-renames",
                         "--diff-filter=AM", "HEAD", at=tree)
        return [name for name in names.split("\0") if name]

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
        self.git("fetch", str(tree), sha)
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
        # Calls are serialized by the world's mutation lock. Retry 429 only; other
        # failures are surfaced without exception bodies, which may include secrets.
        for attempt in range(5):
            try:
                return getattr(self.api, method)(self.repo_id, repo_type="dataset", **kwargs)
            except HfHubHTTPError as exc:
                if exc.response.status_code != 429 or attempt == 4:
                    raise ToolError("Hub request failed; retry later", "hub_error") from None
                delay = exc.response.headers.get("Retry-After", "1")
                try:
                    delay = float(delay)
                except ValueError:
                    delay = 2 ** attempt
                time.sleep(max(0, min(delay, 30)))

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

    def merge(self, pr, comment):
        self.request("merge_pull_request", discussion_num=pr, comment=comment)
