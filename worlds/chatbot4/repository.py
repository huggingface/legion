"""Git transport and native Hub PRs. The world uses the shared experiment HF_TOKEN for writes."""
from __future__ import annotations

import logging
import os
import re
import signal
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
        # (base, head) -> stable patch-id of the default diff, (base, head, 0) -> of the -U0 diff
        self.patch_ids = {}
        self.env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1",
                    "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_LFS_SKIP_SMUDGE": "1"}
        # The dataset is public: PR refs are fetched anonymously first (the Hub throttles git
        # fetches per token), then with the token. CHATBOT_ANON_FETCH=0 disables it.
        self.anonymous = (remote.startswith("https://")
                          and os.environ.get("CHATBOT_ANON_FETCH", "1").strip() != "0")
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
        # CHATBOT_UNION_PATHS: gitattributes patterns (e.g. deck/*.jsonl) whose replays onto main
        # keep both sides' lines (merge=union) instead of conflicting. The attributes file is the
        # world's own, next to its repository; the dataset is unchanged.
        union = [p for p in re.split(r"[\s,]+", os.environ.get("CHATBOT_UNION_PATHS", "")) if p]
        self.attributes = root.parent / "union.gitattributes" if union else None
        if self.attributes:
            self.attributes.write_text("".join(f"{p} merge=union\n" for p in union))
        # Fetches start no background gc: it packed refs outside refs_lock. maintain() runs gc.
        self.git("config", "maintenance.auto", "false")
        # Only this world runs gc here: a gc.pid is a gc stopped with a world that ran elsewhere,
        # and git would honour it for 12 h.
        (root / "gc.pid").unlink(missing_ok=True)
        self.fetch()

    def git(self, *args, at=None, data=None, env=None, timeout=120, group=False):
        return self.git_bytes(*args, at=at, data=data, env=env, timeout=timeout,
                              group=group).decode().strip()

    def git_bytes(self, *args, at=None, data=None, env=None, timeout=120, group=False):
        """Run git and return its raw stdout (for diffs and file contents, which need not be
        UTF-8). Failures raise like git(). group: git runs in its own process group, and a
        timeout kills the whole group (gc's repack and pack-objects too), then raises."""
        argv = ["git", "-C", str(at or self.root), *args]
        if group:
            with subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, env=env or self.env,
                                  start_new_session=True) as proc:
                try:
                    out, err = proc.communicate(data, timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    raise
            p = subprocess.CompletedProcess(argv, proc.returncode, out, err)
        else:
            p = subprocess.run(argv, input=data, capture_output=True, env=env or self.env,
                               timeout=timeout, check=False)
        if p.returncode:
            # Git's stderr can contain remote data or credentials: never expose it to agents;
            # the world's log gets it redacted.
            command = next((a for a in args if not a.startswith("-") and "=" not in a), "?")
            stderr = p.stderr.decode(errors="replace")
            log.warning("git %s failed: exit code %s, stderr=%s", command, p.returncode,
                        _safe_text(stderr.strip(), 500))
            # A --force-with-lease push whose expected value no longer matches the remote ref.
            code = "stale_lease" if command == "push" and "stale info" in stderr else "git_error"
            raise ToolError("repository operation failed; refresh your clone and retry", code)
        return p.stdout

    def negotiation(self):
        """Offer only main's history as haves: with ~2,000 local refs the default negotiation
        makes the request large enough for the Hub to drop it. Git refuses a tip that doesn't
        exist yet (before the first fetch), so it's passed only when main is there."""
        p = subprocess.run(["git", "-C", str(self.root), "rev-parse", "--verify", "-q", "refs/heads/main"],
                           env=self.env, capture_output=True, timeout=10, check=False)
        return ["--negotiation-tip=refs/heads/main"] if p.returncode == 0 else []

    def hub_fetch(self, remote, *specs, options=(), env=None):
        """Every fetch from the Hub: protocol v2 (explicit; v0 floods haves) with main as the
        only negotiation tip, explicit refspecs only (the Hub drops wildcard PR-ref fetches)."""
        self.git("-c", "protocol.version=2", *options, "fetch", "--no-write-fetch-head",
                 *self.negotiation(), remote, *specs, env=env)

    def fetch(self):
        """Fetch main only. The world writes refs/pr/N itself after every push (update_refs), so
        PR refs never come from the Hub here; a wildcard over refs/pr/* is also refused by the
        Hub's git backend once there are many PRs (hang-up / 'expected acknowledgments').
        A fetch that started after this call is as fresh as this one: waiters behind it return."""
        asked = time.monotonic()
        with self.refs_lock:
            if self.fetched > asked:
                return
            started = time.monotonic()
            self.hub_fetch("origin", "+refs/heads/main:refs/heads/main")
            self.fetched = started

    def has_commit(self, sha):
        if not sha or len(sha) != 40 or any(c not in "0123456789abcdef" for c in sha):
            return False
        p = subprocess.run(["git", "-C", str(self.root), "cat-file", "-e", f"{sha}^{{commit}}"],
                           env=self.env, capture_output=True, timeout=10, check=False)
        return p.returncode == 0

    def fetch_anonymous(self, *specs):
        """Fetch specs from the plain remote URL without credentials (no askpass, no token, no
        credential helper). Never raises: returns False on failure (or when disabled)."""
        if not self.anonymous:
            return False
        env = {k: v for k, v in self.env.items()
               if k not in ("GIT_ASKPASS", "SSH_ASKPASS", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")}
        try:
            with self.refs_lock:
                self.hub_fetch(self.remote, *specs, options=("-c", "credential.helper="), env=env)
            return True
        except Exception:  # noqa: BLE001 - logged by git() (the URL holds no credentials)
            return False

    def fetch_pr_refs(self, numbers, batch=8, delays=(1, 3, 9)):
        """Fetch refs/pr/N for the given PR numbers from the Hub, explicitly, in small batches
        (the Hub hangs up on large or wildcard PR-ref fetches): anonymously first, then with the
        token. Never raises: a batch that still fails after the retries is logged, and the list
        of PR numbers left unfetched is returned."""
        missing = []
        numbers = [int(n) for n in numbers]
        for start in range(0, len(numbers), batch):
            chunk = numbers[start:start + batch]
            specs = [f"+refs/pr/{n}:refs/pr/{n}" for n in chunk]
            if self.fetch_anonymous(*specs):
                continue
            for attempt, delay in enumerate(delays + (None,)):
                try:
                    with self.refs_lock:
                        self.hub_fetch("origin", *specs)
                    break
                except ToolError:
                    if delay is None:
                        log.warning("PR refs %s not fetched after %d attempts", chunk, attempt + 1)
                        missing.extend(chunk)
                    else:
                        time.sleep(delay)
        return missing

    def fetch_ref(self, remote_ref, local_ref):
        """Fetch one ref from the Hub into local_ref, anonymously first, then with the token.
        Never raises: returns False on failure."""
        if self.fetch_anonymous(f"+{remote_ref}:{local_ref}"):
            return True
        try:
            with self.refs_lock:
                self.hub_fetch("origin", f"+{remote_ref}:{local_ref}")
            return True
        except Exception:  # noqa: BLE001 - logged by git(); the caller refuses cleanly
            return False

    def refresh(self, max_age):
        """Fetch unless a fetch started within max_age seconds. The world updates its own
        refs after every push and merge, so this only catches up with outside changes; with
        max_age > 0 it also returns at once while refs_lock is busy (a fetch is running)
        instead of queueing another fetch. max_age 0 always gets a main fetched after the call."""
        if time.monotonic() - self.fetched < max_age or (max_age > 0 and self.refs_lock.locked()):
            return
        self.fetch()

    def update_refs(self, *refs):
        """Point each (ref, sha) at its sha in one transaction: one refs_lock wait."""
        with self.refs_lock:
            self.git("update-ref", "--stdin", data="".join(f"update {ref} {sha}\n" for ref, sha in refs).encode())

    def maintain(self, info):
        """Keep the bare repository fast: pack every ref, then gc --auto in the foreground (it
        ignores and removes a gc.log). gc --auto never packs unreachable loose objects (rebased
        and refused commits), so past gc.auto of them a full gc --cruft packs them (what fixed
        the live v8 world: 120k loose objects, merges at 3/h). Fills info with the loose-object
        counts and each step's seconds, also when a step fails."""
        info["loose_before"] = self.loose()
        start = time.monotonic()
        with self.refs_lock:
            self.git("pack-refs", "--all")
        info["pack_refs_s"], start = time.monotonic() - start, time.monotonic()
        # gc.packRefs off: refs were packed above under refs_lock, never while the world writes one
        # pack.threads=2: a full repack leaves the world's other git work some cores
        gc = ("-c", "gc.autoDetach=false", "-c", "gc.cruftPacks=true", "-c", "gc.packRefs=false",
              "-c", "pack.threads=2", "gc", "--quiet")
        try:
            self.git(*gc, "--auto", timeout=1800, group=True)
        finally:
            info["gc_auto_s"], start = time.monotonic() - start, time.monotonic()
        loose = self.loose()
        if loose > int(self.git("config", "--get", "--default", "6700", "gc.auto")):
            try:
                self.git(*gc, timeout=1800, group=True)
            finally:
                info["gc_full_s"] = time.monotonic() - start
            loose = self.loose()
        info["loose_after"] = loose

    def loose(self):
        """Loose objects in the repository (git count-objects)."""
        return int(re.search(r"^count: (\d+)", self.git("count-objects", "-v"), re.MULTILINE)[1])

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
        return self._patch_id((base, head), ["diff", base, head])

    def patch_id0(self, base, head):
        """Stable patch-id of the context-free diff base..head (None for an empty diff).

        Without context lines, a change of main next to the PR's hunks leaves it unchanged,
        so it identifies the PR's own change across a clean rebase. Cached per pair."""
        return self._patch_id((base, head, 0), ["diff", "-U0", "--no-color", base, head])

    def _patch_id(self, key, args):
        if key not in self.patch_ids:
            diff = subprocess.run(["git", "-C", str(self.root), *args],
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

    def commit(self, tree, parent, author, title, onto=None, union=False):
        """Commit tree on parent; with onto != parent, replay that commit onto it (with union,
        CHATBOT_UNION_PATHS merge as merge=union: only where a changed diff is refused)."""
        who = ("-c", f"user.name={author}", "-c", f"user.email={author}@flock.invalid")
        self.git("add", "-A", at=tree)
        self.git(*who, "commit", "--allow-empty", "-m", title, at=tree)
        sha = self.git("rev-parse", "HEAD", at=tree)
        if onto and onto != parent:
            self.git("checkout", "-q", "--detach", onto, at=tree)
            attributes = ("-c", f"core.attributesFile={self.attributes}") if union and self.attributes else ()
            try:
                self.git(*who, *attributes, "cherry-pick", "--allow-empty", "--keep-redundant-commits", sha,
                         at=tree)
            except ToolError:
                unmerged = subprocess.run(["git", "-C", str(tree), "diff", "--name-only", "-z",
                                           "--diff-filter=U"], env=self.env, capture_output=True,
                                          timeout=60, check=False).stdout.decode(errors="replace")
                subprocess.run(["git", "-C", str(tree), "cherry-pick", "--abort"], env=self.env,
                               capture_output=True, timeout=60, check=False)
                error = ToolError(f"your branch is based on {parent}, main is {onto}, and your "
                                  "changes conflict with main: pull, rebase onto origin/main, "
                                  "then open again", "stale_base")
                error.files = [name for name in unmerged.split("\0") if name]  # the conflicting paths
                raise error from None
            sha = self.git("rev-parse", "HEAD", at=tree)
        self.git("fetch", "--no-write-fetch-head", str(tree), sha)
        return sha

    def push(self, sha, ref, expected=None, force=False):
        """Push sha to ref; expected replaces only that head; force overwrites a ref the
        world owns outright (a PR or branch it created and nobody else writes)."""
        options = [f"--force-with-lease={ref}:{expected}"] if expected else []
        self.git("push", *options, "origin", f"{'+' if force and not expected else ''}{sha}:{ref}")


FINAL_CODES = {"closed": "pr_closed", "merged": "pr_merged"}
_FINAL_WORDS = re.compile(r"\b(merged|closed)\b")


def _final_state(exc):
    """"closed" or "merged" when an HTTP 400 says the PR is already closed or merged (matched on
    the x-error-message header, else the response text; neither is ever logged), else None."""
    response = getattr(exc, "response", None)
    if getattr(response, "status_code", None) != 400:
        return None
    try:
        text = str(response.headers.get("x-error-message") or "") or str(response.text or "")
    except Exception:  # noqa: BLE001 - an unreadable body is just "unknown"
        return None
    found = _FINAL_WORDS.search(text[:2000].lower())
    return found.group(1) if found else None


class Hub:
    def __init__(self, repo_id):
        self.repo_id = repo_id
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise ToolError("HF_TOKEN is required for the bucket and dataset", "configuration")
        self.api = HfApi(token=token)
        self.refs_ttl = float(os.environ.get("CHATBOT_HUB_REFS_TTL_S", "5"))
        self.refs_lock = threading.Lock()  # one refresh at a time; waiters reuse its result
        self.refs_cache, self.refs_at = None, 0.0

    def request(self, method, **kwargs):
        # Retry 429 only (only merges call this under the world's state lock); other
        # failures are surfaced without exception bodies, which may include secrets.
        for attempt in range(5):
            try:
                return getattr(self.api, method)(self.repo_id, repo_type="dataset", **kwargs)
            except HfHubHTTPError as exc:
                if exc.response.status_code != 429 or attempt == 4:
                    _log_hub_failure(method, self.repo_id, exc)
                    final = _final_state(exc)
                    if final and isinstance(kwargs.get("discussion_num"), int):
                        num = kwargs["discussion_num"]
                        raise ToolError(f"PR #{num} is already {final} on the Hub",
                                        FINAL_CODES[final]) from None
                    error = ToolError("Hub request failed; retry later", "hub_error")
                    error.status = exc.response.status_code  # for the world, never sent to agents
                    raise error from None
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

    def pr_states(self):
        """{PR number: status} (open, closed, merged or draft) of the Hub's pull requests, a read
        (no Hub-write slot). Raises on failure."""
        return {d.num: d.status for d in self.request("get_repo_discussions") if d.is_pull_request}

    def refs(self):
        """{ref: sha} for the Hub's refs/pr/N and refs/heads/<branch>, from the refs API (a
        read: no Hub-write slot). Cached for CHATBOT_HUB_REFS_TTL_S so a burst costs one
        request; on failure, a WARNING and the stale cache (or {})."""
        with self.refs_lock:
            if self.refs_cache is not None and time.monotonic() - self.refs_at < self.refs_ttl:
                return self.refs_cache
            try:
                found = self.request("list_repo_refs", include_pull_requests=True)
                refs = {r.ref: r.target_commit for r in [*(found.branches or []),
                                                         *(found.pull_requests or [])]
                        if r.ref and r.target_commit}
            except Exception as exc:  # noqa: BLE001 - request() logged the details, never secrets
                log.warning("Hub refs not listed for %s (%s); using %s", self.repo_id,
                            type(exc).__name__, "the stale cache" if self.refs_cache else "nothing")
                return self.refs_cache or {}
            self.refs_cache, self.refs_at = refs, time.monotonic()
            return refs

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
