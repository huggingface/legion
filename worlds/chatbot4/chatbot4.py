"""Run-4 world: local snapshots, native Hub PRs and independent reviews."""
from __future__ import annotations

import contextvars
import json
import os
import re
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import ClassVar, Literal

from repository import Hub, Repository
from sandbox import run

from flock.world import Agent, ToolError, World, tool
from flock.world.workspace import unpack

HERE = Path(__file__).resolve().parent
UPLOAD = contextvars.ContextVar("workspace", default=None)
FIXED = ("bot/__init__.py", "bot/__main__.py", "bot/runtime.py", "bot/modules/__init__.py")
NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")


class Chatbot4World(World):
    name, version = "chatbot4", "0.4.0"
    concurrent = True
    capabilities: ClassVar[list[str]] = ["workspace-upload-v1"]
    description = ("Edit your local git clone, then submit with open_pr. Another agent must "
                   "approve the exact head before merge. Bot input/output and module loading "
                   "are fixed; edit bot/modules/. run_bot allows 300 seconds per conversation.")

    def __init__(self, ws=None, repository=None):
        self.ws = Path(ws or os.environ.get("CHATBOT_WS", HERE / "ws"))
        self.ws.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = self.ws / "prs.json"
        self.state = json.loads(self.db.read_text()) if self.db.exists() else {"prs": {}, "requests": {}}
        self.repo_id = repo_id = os.environ.get("CHATBOT_REPO", "cmpatino/chatbot-swarm")
        self.repo = repository or Repository(self.ws / "repo.git",
                                            f"https://huggingface.co/datasets/{repo_id}", Hub(repo_id))
        self.fixed = {name: (HERE / "seed" / name).read_bytes() for name in FIXED}
        self.slots = threading.BoundedSemaphore(int(os.environ.get("CHATBOT_RUN_SLOTS", "4")))

    def _save(self):
        tmp = self.db.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state))
        os.replace(tmp, self.db)

    def _event(self, action, agent, **data):
        with open(self.ws / "prs.jsonl", "a") as out:
            out.write(json.dumps({"ts": time.time(), "event": action, "agent": agent.name,
                                  **data}) + "\n")

    def _pr(self, pr):
        rec = self.state["prs"].get(str(pr))
        if rec is None or rec["state"] != "open":
            raise ToolError("no open PR with that number")
        return rec

    def _identity(self, agent):
        if not NAME.fullmatch(agent.name):
            raise ToolError("invalid agent identity")

    def call_workspace(self, name, args, agent, snapshot):
        if name not in ("open_pr", "run_bot"):
            raise ToolError("this tool does not accept workspace uploads")
        token = UPLOAD.set(snapshot)
        try:
            return self.call(name, args, agent)
        finally:
            UPLOAD.reset(token)

    def _files(self, snapshot, tree):
        for child in tree.iterdir():
            if child.name == ".git":
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        try:
            unpack(snapshot, tree)
        except (ValueError, TypeError, OSError) as exc:
            raise ToolError("invalid workspace upload", "bad_workspace") from exc
        self._contract(tree)

    def _contract(self, tree):
        for name, expected in self.fixed.items():
            f = tree / name
            if not f.is_file() or f.read_bytes() != expected:
                raise ToolError(f"keep the seed interface fixed: {name}; edit bot/modules/ instead",
                                "fixed_interface")
        for f in tree.rglob("*.py"):
            if ".git" not in f.relative_to(tree).parts:
                try:
                    compile(f.read_bytes(), str(f.relative_to(tree)), "exec")
                except (SyntaxError, ValueError):
                    raise ToolError(f"Python does not compile: {f.relative_to(tree)}", "compile") from None

    @tool
    def open_pr(self, agent: Agent, branch: str, title: str, description: str,
                path: str = "chatbot-swarm", pr: int | None = None) -> dict:
        """Submit your local clone, or update your existing PR with pr:<number>.

        The worker uploads path from your private workspace. Include tests and run_bot
        evidence in description. At most two open PRs per agent. Read CONTRIBUTING.md.
        """
        self._identity(agent)
        if not NAME.fullmatch(branch) or not 3 <= len(title) <= 200 or not description.strip():
            raise ToolError("use a short branch slug, a title (3–200 chars), and a description")
        snapshot = UPLOAD.get()
        if not isinstance(snapshot, dict):
            raise ToolError("submit a local clone through flock-tool; the worker uploads it")
        with self.lock:
            key = agent.request_id
            if key and key in self.state["requests"]:
                return self.state["requests"][key]
            self.repo.fetch()
            base = snapshot.get("base_sha", "")
            main = self.repo.sha()
            if not self.repo.ancestor(base, main):
                raise ToolError("base is not a main commit; fetch main and rebase your branch")
            old = self._pr(pr) if pr is not None else None
            if old and old["author"] != agent.name:
                raise ToolError("only the author can update this PR")
            if old and old["branch"] != f"agents/{agent.name}/{branch}":
                raise ToolError("use the original branch slug when updating a PR")
            if not old and sum(r["state"] == "open" and r["author"] == agent.name
                               for r in self.state["prs"].values()) >= 2:
                raise ToolError("at most two open PRs per agent")
            with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
                tree = Path(tmp) / "tree"
                self.repo.checkout(base, tree)
                self._files(snapshot, tree)
                sha = self.repo.commit(tree, base, agent.name, title)
            branch_ref = f"agents/{agent.name}/{branch}"
            if not old and any(r["branch"] == branch_ref for r in self.state["prs"].values()):
                raise ToolError("branch slug already used; choose a new topic slug")
            number = pr if old else self.repo.hub.open(title, f"Agent: {agent.name}\nBase: {base}\n\n{description}")
            # Only agent/PR refs may be replaced, with an exact expected old head.
            self.repo.push(sha, f"refs/pr/{number}", expected=old["head_sha"] if old else None)
            self.repo.push(sha, f"refs/heads/{branch_ref}", expected=old["head_sha"] if old else None)
            if not old:
                self.repo.hub.ready(number)
            self.repo.git("update-ref", f"refs/flock/pr/{number}/{sha}", sha)
            record = {"pr": number, "state": "open", "author": agent.name,
                      "branch": branch_ref, "base_sha": base, "head_sha": sha,
                      "url": f"https://huggingface.co/datasets/{self.repo_id}/discussions/{number}",
                      "title": title, "description": description, "reviews": {}}
            self.state["prs"][str(number)] = record
            result = {k: v for k, v in record.items() if k != "reviews"}
            if key:
                self.state["requests"][key] = result
            self._save()
            self._event("updated" if old else "opened", agent, **result)
            return result

    @tool
    def review_pr(self, agent: Agent, pr: int, head_sha: str,
                  verdict: Literal["approve", "request_changes"], comment: str) -> dict:
        """Review the exact head SHA you inspected or tested, independently of the author."""
        self._identity(agent)
        if verdict not in ("approve", "request_changes") or not comment.strip():
            raise ToolError("give a verdict and a review comment")
        with self.lock:
            rec = self._pr(pr)
            if rec["author"] == agent.name:
                raise ToolError("another agent must review your PR; self-review is refused")
            self.repo.fetch()
            current = self.repo.sha(f"pr:{pr}")
            if head_sha != current or current != rec["head_sha"]:
                raise ToolError("the PR head changed; inspect it again before reviewing")
            review = {"head_sha": head_sha, "verdict": verdict, "comment": comment}
            self.repo.hub.comment(pr, f"{verdict} by {agent.name} on {head_sha}\n\n{comment}")
            rec["reviews"][agent.name] = review
            self._save()
            self._event("reviewed", agent, pr=pr, **review)
            return {"pr": pr, "reviewer": agent.name, **review}

    def _approved(self, rec):
        reviews = {name: r for name, r in rec["reviews"].items()
                   if name != rec["author"] and r["head_sha"] == rec["head_sha"]}
        if any(r["verdict"] == "request_changes" for r in reviews.values()):
            raise ToolError("a reviewer still requests changes on this head")
        names = [name for name, r in reviews.items() if r["verdict"] == "approve"]
        if not names:
            raise ToolError("another agent must approve the current PR head")
        return names

    @tool
    def merge_pr(self, agent: Agent, pr: int) -> dict:
        """Merge an independently approved PR after checking its base and bot contract."""
        self._identity(agent)
        with self.lock:
            rec = self._pr(pr)
            self._approved(rec)
            self.repo.fetch()
            main, head = self.repo.sha(), self.repo.sha(f"pr:{pr}")
            if head != rec["head_sha"] or main != rec["base_sha"]:
                raise ToolError("stale PR: fetch main, rebase, update the PR and get a fresh review")
        with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
            tree = Path(tmp) / "tree"
            with self.lock:
                self.repo.checkout(head, tree)
            self._contract(tree)
            verdict = self._run(tree, ["Hello", "My name is Sam", "", "What is my name?", "Bye"])
            if not verdict["ok"]:
                raise ToolError("bot contract failed: one nonempty reply per input, no crash, 300s total")
        with self.lock:
            rec = self._pr(pr)
            reviewers = self._approved(rec)
            self.repo.fetch()
            if self.repo.sha() != main or self.repo.sha(f"pr:{pr}") != head or rec["head_sha"] != head:
                raise ToolError("main or the PR changed during checks; update and review again")
            self.repo.hub.merge(pr, f"Approved by {', '.join(reviewers)}; head {head}; base {main}; contract passed")
            self.repo.fetch()
            merged = self.repo.sha()
            rec.update(state="merged", merged_sha=merged)
            self._save()
            event = {"pr": pr, "author": rec["author"], "reviewers": reviewers,
                     "parent_sha": main, "head_sha": head, "merged_sha": merged}
            self._event("merged", agent, **event)
            # Append-only integration hook, consumed by the separate evaluation workstream.
            with open(self.ws / "eval-queue.jsonl", "a") as out:
                out.write(json.dumps({"ts": time.time(), **event}) + "\n")
            return event

    @tool
    def list_prs(self, state: str = "open", author: str = "", needs_review: bool = False,
                 offset: int = 0) -> list[dict]:
        """Find PRs, their head SHA and reviews. state is open, merged or all."""
        if offset < 0:
            raise ToolError("offset must be nonnegative")
        if state not in ("open", "merged", "all"):
            raise ToolError("state must be open, merged or all")
        with self.lock:
            records = []
            for rec in self.state["prs"].values():
                if (state != "all" and rec["state"] != state) or (author and rec["author"] != author):
                    continue
                if needs_review and any(r["verdict"] == "approve" and r["head_sha"] == rec["head_sha"]
                                        for r in rec["reviews"].values()):
                    continue
                records.append(json.loads(json.dumps(rec)))
            return records[offset:offset + 100]

    @tool
    def run_bot(self, ref_or_path: str, lines: list[str]) -> dict:
        """Run main, pr:<number>, or a local clone path. 300 seconds for the whole conversation."""
        with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
            tree = Path(tmp) / "tree"
            snapshot = UPLOAD.get()
            if ref_or_path == "main" or ref_or_path.startswith("pr:"):
                with self.lock:
                    self.repo.fetch()
                    sha = self.repo.sha(ref_or_path)
                    self.repo.checkout(sha, tree)
            elif isinstance(snapshot, dict):
                tree.mkdir()
                self._files(snapshot, tree)
                sha = snapshot.get("base_sha")
            else:
                raise ToolError("use main, pr:<number>, or a clone path inside your workspace")
            return {"ref": ref_or_path, "sha": sha, **self._run(tree, lines)}

    def _run(self, tree, lines):
        if not self.slots.acquire(timeout=10):
            raise ToolError("bot runners are busy; retry later", "busy")
        try:
            return run(tree, lines)
        finally:
            self.slots.release()


if __name__ == "__main__":
    from flock.world.server import serve
    serve("chatbot4:Chatbot4World", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
