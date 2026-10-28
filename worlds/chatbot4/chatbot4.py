"""Run-4 world: local snapshots, native Hub PRs and independent reviews."""
from __future__ import annotations

import atexit
import base64
import contextlib
import contextvars
import functools
import hashlib
import inspect
import json
import logging
import marshal
import math
import os
import re
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import ClassVar, Literal

from repository import Hub, Repository
from sandbox import run

from flock.isolation import protect_supervisor
from flock.world import Agent, ToolError, World, tool
from flock.world.workspace import unpack

HERE = Path(__file__).resolve().parent
UPLOAD = contextvars.ContextVar("workspace", default=None)
FIXED = ("bot/__init__.py", "bot/__main__.py", "bot/runtime.py", "bot/modules/__init__.py")
MAX_FILE_BYTES = int(os.environ.get("CHATBOT_MAX_FILE_BYTES", str(1 << 20)))
EXCLUDE = ("__pycache__/", "*.pyc", ".venv/", ".pytest_cache/", ".mypy_cache/", ".ruff_cache/",
           "*.egg-info/", ".DS_Store")
MAX_REQUESTS = 5000  # open_pr idempotency keys kept for worker retries (oldest dropped)
# ... and for at most this long (a worker retries within minutes).
REQUEST_TTL = float(os.environ.get("CHATBOT_REQUEST_TTL_S", "21600"))
# prs.json is written by a background thread at most once per this many seconds while the
# state changed (0: write synchronously on every change). close() always writes it.
SAVE_INTERVAL = float(os.environ.get("CHATBOT_SAVE_INTERVAL_S", "2.0"))
FETCH_TTL = float(os.environ.get("CHATBOT_FETCH_TTL_S", "60"))
NAME = re.compile(r"^[a-z][a-z0-9-]{0,63}$")
# A world tool call must answer within about 55 s (the worker's deadline); Hub waits and push
# retries stay inside this budget, counted from the start of the call.
CALL_BUDGET = float(os.environ.get("CHATBOT_CALL_BUDGET_S", "50"))
PUSH_RETRY_DELAYS = (2, 5)
PUSH_ESTIMATE_S = 8.0  # time kept free for one more push attempt
BUSY = "the Hub is busy; retry in about a minute"
INBOX_MAX = 20  # unread inbox entries kept per agent (oldest dropped)
INBOX_SNIPPET = 160  # review comment characters quoted in an inbox entry
PAUSED = "PRs are paused by the operators; keep working locally, run_bot and reviews still work"
WINDOW = 3600.0  # the new-PR budget counts opens and merges in this rolling window (seconds)
WINDOW_MAX = 10000  # timestamps kept per list (oldest dropped)
OUTBOX_TRIES = 12  # attempts per queued Hub comment before it is dropped
OUTBOX_BACKOFF = (5.0, 600.0)  # first retry delay, longest retry delay (doubling)
HUB_SYNC_EVERY = float(os.environ.get("CHATBOT_HUB_STATES_EVERY_S", "600"))  # PR state check
FINAL_CODES = ("pr_closed", "pr_merged")
# get_pr: diff bytes returned (cut at a line end, with truncated: true beyond this).
MAX_DIFF_BYTES = int(os.environ.get("CHATBOT_GET_PR_DIFF_BYTES", str(200 << 10)))
DESCRIPTION_CHARS = 4000  # PR description characters returned by get_pr  # Hub 400: the PR is already closed / merged
log = logging.getLogger(__name__)
_LIVE: dict[str, Chatbot4World] = {}  # one world per state directory: a restart replaces it


def _key(key):
    """A dict key as json.dumps writes it."""
    return json.dumps(key) if isinstance(key, str) else json.dumps({key: 0})[1:-4]


@atexit.register
def _close_all():
    for world in list(_LIVE.values()):
        world.close()


def logs_refusals(fn):
    """Log a refused PR operation to prs.jsonl (code and message only), then re-raise."""
    signature = inspect.signature(fn)

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        try:
            return fn(self, *args, **kwargs)
        except ToolError as exc:
            bound = signature.bind_partial(self, *args, **kwargs).arguments
            agent = bound.get("agent")
            if isinstance(agent, Agent):
                self._event(f"{fn.__name__}_refused", agent, pr=bound.get("pr"),
                            code=exc.code, reason=exc.message[:500])
            raise
    return wrapper


class Chatbot4World(World):
    name, version = "chatbot4", "0.4.0"
    concurrent = True
    capabilities: ClassVar[list[str]] = ["workspace-upload-v1"]
    description = ("Edit your local git clone, then submit with open_pr. Another agent must "
                   "approve the PR's current change (a clean rebase keeps the approval) before merge. Bot input/output and module loading "
                   "are fixed; edit bot/modules/. run_bot allows 300 seconds per conversation. "
                   "Tool results may carry an `inbox` with news about your PRs (reviews, "
                   "merges, closes, updates); handle it first. New PRs are limited swarm-wide to "
                   "about the merge rate (pr_budget); reviews are unlimited.")

    # The worker uploads the caller's local clone for these calls (workspace-upload-v1).
    WORKSPACE: ClassVar[dict] = {
        "open_pr": {"arg": "path", "default": "chatbot-swarm"},
        "run_bot": {"arg": "ref_or_path", "default": "main", "skip": ["main"], "skip_prefix": "pr:"},
    }

    def manifest(self) -> dict:
        m = super().manifest()
        for t in m["tools"]:
            t["x-flock-timeout_s"] = 330  # a 300 s run_bot conversation plus the upload
            if t["name"] in self.WORKSPACE:
                t["x-flock-workspace"] = self.WORKSPACE[t["name"]]
        return m

    def __init__(self, ws=None, repository=None):
        protect_supervisor()
        self.ws = Path(ws or os.environ.get("CHATBOT_WS", HERE / "ws"))
        self.ws.mkdir(parents=True, exist_ok=True)
        previous = _LIVE.get(str(self.ws.resolve()))
        if previous is not None:  # flush and stop its outbox before reading the saved state
            previous.close()
        # Lock order: Hub write slot, then main_lock, then self.lock (the state lock).
        self.lock = threading.RLock()
        # The compare-and-swap on main: held by merges across the Hub merge and the fetch that
        # moves local main, and by the "did main move" check before a Hub PR is created. The
        # state lock is taken only around record reads and updates, never across a Hub call.
        self.main_lock = threading.Lock()
        self.ancestry = {}  # (head, main sha) -> head is in main; read and written under self.lock
        self.db = self.ws / "prs.json"
        for stale in self.ws.glob("prs.json.*.tmp"):  # a write cut by a crash
            stale.unlink(missing_ok=True)
        self.state = json.loads(self.db.read_text()) if self.db.exists() else {"prs": {}, "requests": {}}
        # Saving (see _save): every change bumps save_gen; written_gen is the generation on disk.
        self.save_interval = float(os.environ.get("CHATBOT_SAVE_INTERVAL_S", SAVE_INTERVAL))
        self.save_gen = self.written_gen = 0
        self.save_sync = self.save_interval <= 0  # also set by close(): later saves write at once
        self.replace_lock = threading.Lock()  # a leaf lock: never take self.lock while holding it
        self.save_wake = threading.Event()
        self.save_stop = threading.Event()
        self.saver = None
        # For metrics(): the last prs.json write and in-memory counters since start.
        self.save_s, self.saved_at = 0.0, self.db.stat().st_mtime if self.db.exists() else time.time()
        self.state_bytes = self.db.stat().st_size if self.db.exists() else 0
        self.budget_refusals = []  # unix times of open_budget refusals (last WINDOW seconds)
        self.stale_adoptions = self.hub_busy_total = 0
        self.hub_slot_wait_s = 0.0
        self.repo_id = repo_id = os.environ.get("CHATBOT_REPO", "cmpatino/chatbot-swarm")
        self.repo = repository or Repository(self.ws / "repo.git",
                                            f"https://huggingface.co/datasets/{repo_id}", Hub(repo_id))
        self.fixed = {name: (HERE / "seed" / name).read_bytes() for name in FIXED}
        self.slots = threading.BoundedSemaphore(int(os.environ.get("CHATBOT_RUN_SLOTS", "26")))
        # Merge contract checks get their own runners, so run_bot load never blocks merges.
        self.merge_slots = threading.BoundedSemaphore(int(os.environ.get("CHATBOT_MERGE_SLOTS", "4")))
        self.pending = {}  # open_pr reservations: token -> (author, branch ref, pr or None, key)
        self.operators = {n.strip() for n in os.environ.get("CHATBOT_OPERATORS", "").split(",") if n.strip()}
        # Every Hub write (create PR, push, ready, merge, close, comment) holds a slot: one IP
        # and one token cannot push for 140 agents at once. Lock order: see self.lock.
        self.hub_slots = threading.BoundedSemaphore(int(os.environ.get("CHATBOT_HUB_WRITE_SLOTS", "1")))
        # Merges hold their own slots instead: a merge never queues behind the rebase pushes,
        # opens and closes, whose wait would let main or the PR move under its checks. At most
        # CHATBOT_HUB_WRITE_SLOTS + CHATBOT_MERGE_HUB_SLOTS Hub writes run at once.
        self.merge_hub_slots = threading.BoundedSemaphore(
            int(os.environ.get("CHATBOT_MERGE_HUB_SLOTS", "1")))
        self.hub_wait = float(os.environ.get("CHATBOT_HUB_WAIT_S", "20"))
        self.pr_gap = float(os.environ.get("CHATBOT_PR_CREATE_GAP_S", "10"))
        self.gap_lock = threading.Lock()
        self.last_create = float("-inf")  # monotonic time of the last (reserved) PR creation
        # Pause open_pr/merge_pr: CHATBOT_PRS_PAUSED=1 at start, or the flag file, checked per call.
        self.paused_env = os.environ.get("CHATBOT_PRS_PAUSED", "").strip() not in ("", "0")
        self.pause_file = Path(os.environ.get("CHATBOT_PRS_PAUSED_FILE") or self.ws / "prs-paused")
        # Hub PRs whose head could not be pushed: branch ref -> draft, reused by the next open_pr.
        self.state.setdefault("drafts", {})
        # Unread news per agent: name -> [{"pr", "kind", "text", "ts"}], newest last.
        self.state.setdefault("inbox", {})
        # New-PR budget: Hub PR creations and merges (unix times, last WINDOW seconds).
        self.state.setdefault("pr_opens", [])
        self.state.setdefault("pr_merges", [])
        budget = os.environ.get("CHATBOT_PR_OPEN_BUDGET", "merges").strip().lower() or "merges"
        if budget not in ("merges", "off"):
            raise ValueError("CHATBOT_PR_OPEN_BUDGET must be merges or off")
        self.budget_on = budget == "merges"
        self.budget_floor = int(os.environ.get("CHATBOT_PR_OPEN_FLOOR", "20"))
        self.open_reserved = set()  # new-PR calls past the budget check, Hub PR not yet created
        # Review comments wait here for the Hub; a background thread posts them.
        self.state.setdefault("hub_outbox", [])
        self.outbox_max = int(os.environ.get("CHATBOT_HUB_OUTBOX_MAX", "2000"))
        self.outbox_gap = float(os.environ.get("CHATBOT_HUB_OUTBOX_GAP_S", "1"))
        self.stopping = threading.Event()
        self.wake = threading.Event()
        with self.lock:  # cached results from before slimming: the first save is already small
            if self._slim_requests():
                self._save()
        with self.lock:  # the archive may predate merges that already landed on the Hub
            for rec in self.state["prs"].values():
                if rec["state"] in ("closing", "merging"):  # interrupted: the agent may retry
                    rec["state"] = "open"
            self._fetch_missing_heads()
            self._reconcile()
        self.hub_states_at = float("-inf")
        self._sync_hub_states()  # outside the lock: a Hub read
        self.drainer = threading.Thread(target=self._drain, name="chatbot4-hub-outbox", daemon=True)
        self.drainer.start()
        if not self.save_sync:
            self.saver = threading.Thread(target=self._saver, name="chatbot4-save", daemon=True)
            self.saver.start()
        _LIVE[str(self.ws.resolve())] = self

    # -- review comment outbox -------------------------------------------------------------

    def _queue_comment(self, agent, pr, text):
        """Queue a Hub comment (call with the lock held; the caller saves)."""
        box = self.state["hub_outbox"]
        box.append({"id": uuid.uuid4().hex, "pr": pr, "agent": agent.name, "text": text,
                    "tries": 0, "next_at": 0.0, "queued_at": time.time()})
        if len(box) > self.outbox_max:
            dropped = box[:len(box) - self.outbox_max]
            del box[:len(dropped)]
            log.warning("hub_outbox: over %d comments, dropped the %d oldest (PRs %s)",
                        self.outbox_max, len(dropped), sorted({d["pr"] for d in dropped}))
        self.wake.set()

    def _post(self, item, wait):
        """Post one queued comment holding the Hub slot (waiting at most wait seconds), then
        update the outbox. Returns False if the slot stayed busy."""
        if not self.hub_slots.acquire(timeout=max(0.0, wait)):
            return False
        try:
            with self.lock:  # dropped while waiting for the slot: nothing to post
                if not any(e["id"] == item["id"] for e in self.state["hub_outbox"]):
                    return True
            self.repo.hub.comment(item["pr"], item["text"])
            failure = None
        except Exception as exc:  # noqa: BLE001 -- retried later, never raised to a tool call
            failure = getattr(exc, "code", type(exc).__name__)
        finally:
            self.hub_slots.release()
        if failure in FINAL_CODES:  # the PR is closed or merged: no point retrying
            with self.lock:
                self._mark_hub_final(item["pr"], failure)
                self.state["hub_outbox"] = [e for e in self.state["hub_outbox"] if e["id"] != item["id"]]
                self._save()
            log.warning("hub_outbox: dropped the comment on PR #%s: already %s on the Hub",
                        item["pr"], failure[3:])
            return True
        with self.lock:
            box = self.state["hub_outbox"]
            entry = next((e for e in box if e["id"] == item["id"]), None)
            if entry is None:  # dropped meanwhile
                return True
            if failure is None:
                box.remove(entry)
            else:
                entry["tries"] += 1
                if entry["tries"] >= OUTBOX_TRIES:
                    box.remove(entry)
                    log.warning("hub_outbox: dropped the comment on PR #%s by %s after %d "
                                "attempts: %s", entry["pr"], entry["agent"], entry["tries"], failure)
                else:
                    delay = min(OUTBOX_BACKOFF[1], OUTBOX_BACKOFF[0] * 2 ** (entry["tries"] - 1))
                    entry["next_at"] = time.time() + delay
                    log.warning("hub_outbox: comment on PR #%s failed (%s); retry %d in %.0f s",
                                entry["pr"], failure, entry["tries"], delay)
            self._save()
        return True

    def _drain(self):
        """Background thread: post due comments one at a time, never blocking a tool call."""
        while not self.stopping.is_set():
            self._sync_hub_states(every=HUB_SYNC_EVERY)
            with self.lock:
                now = time.time()
                box = self.state["hub_outbox"]
                item = next((dict(e) for e in box if e["next_at"] <= now), None)
                idle = min((e["next_at"] - now for e in box), default=5.0)
            if item is None:
                self.wake.wait(min(5.0, max(0.05, idle)))
                self.wake.clear()
                continue
            try:
                posted = self._post(item, 1.0)
            except Exception:  # keep the thread alive
                log.exception("hub_outbox: drain failed")
                posted = False
            self.stopping.wait(self.outbox_gap if posted else 0.2)

    def close(self, timeout=5.0):
        """Write prs.json now, stop the outbox thread and try each queued comment once within
        timeout seconds; what is left stays in prs.json for the next start. Saves after this
        write synchronously."""
        deadline = time.monotonic() + timeout
        self.save_sync = True
        self.save_stop.set()
        self.save_wake.set()
        self.flush()  # first: the outbox below may use the whole timeout
        self.stopping.set()
        self.wake.set()
        if self.drainer.is_alive() and self.drainer is not threading.current_thread():
            self.drainer.join(max(0.0, deadline - time.monotonic()))
            if self.drainer.is_alive():
                self.flush()
                return
        with self.lock:
            items = [dict(e) for e in self.state["hub_outbox"]]
        for item in items:
            left = deadline - time.monotonic()
            if left <= 0 or not self._post(item, left):
                break
        self.flush()
        if _LIVE.get(str(self.ws.resolve())) is self:
            del _LIVE[str(self.ws.resolve())]

    # -- new-PR budget ---------------------------------------------------------------------

    @staticmethod
    def _recent(stamps, now):
        """Drop timestamps older than WINDOW (in place); return the list."""
        cut = now - WINDOW
        stamps[:] = [t for t in stamps if t > cut][-WINDOW_MAX:]
        return stamps

    def _budget(self):
        """Current new-PR budget (call with the lock held)."""
        now = time.time()
        opens = self._recent(self.state["pr_opens"], now)
        merges = self._recent(self.state["pr_merges"], now)
        limit = max(len(merges), self.budget_floor)
        used = len(opens) + len(self.open_reserved)
        info = {"mode": "merges" if self.budget_on else "off", "opened_last_hour": used,
                "merged_last_hour": len(merges), "limit": limit,
                "available": max(0, limit - used) if self.budget_on else None}
        if self.budget_on and used >= limit:
            # The next slot frees when enough of the oldest opens leave the window.
            index = used - limit
            info["next_slot_min"] = (math.ceil(max(0.0, opens[index] + WINDOW - now) / 60)
                                     if index < len(opens) else 1)
        return info

    def _check_budget(self, agent):
        """Refuse a new PR when the swarm's opens in the last hour reached
        max(merges in the last hour, CHATBOT_PR_OPEN_FLOOR) (call with the lock held)."""
        if not self.budget_on:
            return
        info = self._budget()
        if info["opened_last_hour"] < info["limit"]:
            return
        n, m, k = info["opened_last_hour"], info["merged_last_hour"], info["next_slot_min"]
        now = time.time()
        self._recent(self.budget_refusals, now).append(now)
        log.warning("open_budget: refused a new PR for agent=%s opened=%d merged=%d limit=%d",
                    agent.name, n, m, info["limit"])
        raise ToolError(f"the swarm opened {n} PRs in the last hour and merged {m}; new PRs wait "
                        f"until more merge (next slot in ~{k} min). Meanwhile review others' PRs "
                        "(list_prs needs_review=true) or improve your open PR with pr=<n>.",
                        "open_budget", retry_after_s=k * 60)

    def _note_merge(self):
        """Count a merge for the new-PR budget (call with the lock held; the caller saves)."""
        now = time.time()
        self._recent(self.state["pr_merges"], now).append(now)

    def metrics(self):
        """World numbers for GET /metrics (docs/world.md)."""
        now = time.time()
        with self.lock:
            budget = self._budget()
            box = self.state["hub_outbox"]
            out = {"open_prs": sum(r["state"] == "open" for r in self.state["prs"].values()),
                   "opens_last_hour": budget["opened_last_hour"],
                   "merges_last_hour": budget["merged_last_hour"], "open_budget_limit": budget["limit"],
                   "budget_refusals_last_hour": len(self._recent(self.budget_refusals, now)),
                   "parked_drafts": len(self.state["drafts"]), "outbox_depth": len(box),
                   "outbox_oldest_age_s": now - min((e.get("queued_at", now) for e in box), default=now)}
        return {**out, "stale_lease_adoptions_total": self.stale_adoptions,
                "hub_busy_total": self.hub_busy_total, "hub_slot_wait_s_total": self.hub_slot_wait_s,
                "save_duration_s": self.save_s, "save_age_s": now - self.saved_at,
                "state_bytes": self.state_bytes}

    # -- saving prs.json -------------------------------------------------------------------

    def _save(self):
        """Record that the state changed (call with the lock held). The saver thread writes
        prs.json at most once per CHATBOT_SAVE_INTERVAL_S; flush() writes it at once."""
        self.save_gen += 1
        if self.save_sync:
            self.flush()
        else:
            self.save_wake.set()

    def _saver(self):
        """Background thread: write prs.json when the state changed, at most once per
        save_interval. close() stops it and writes the last change itself."""
        last = float("-inf")
        while True:
            self.save_wake.wait()
            self.save_wake.clear()
            if self.save_stop.wait(max(0.0, last + self.save_interval - time.monotonic())):
                return
            last = time.monotonic()
            try:
                self.flush()
            except Exception:  # keep the thread alive; the next change or close() retries
                log.exception("prs.json: save failed; retrying")
                self.save_wake.set()

    def _snapshot(self):
        """A consistent copy of the state as marshal blobs (call with the lock held).

        marshal copies the JSON types several times faster than json.dumps or deepcopy, and the
        copy is encoded and written after the lock is released. Each entry of a top-level dict
        or list gets its own blob, so the encoding outside the lock also yields the GIL between
        entries instead of holding it for one long json.dumps call."""
        parts = []
        for key, value in self.state.items():
            if isinstance(value, dict):
                parts.append((key, dict, [(k, marshal.dumps(v)) for k, v in value.items()]))
            elif isinstance(value, list):
                parts.append((key, list, [marshal.dumps(v) for v in value]))
            else:
                parts.append((key, None, marshal.dumps(value)))
        return parts

    @staticmethod
    def _encode(parts, out):
        """Write a snapshot as JSON, byte for byte what json.dumps(state) gives."""
        out.write("{")
        for i, (key, kind, body) in enumerate(parts):
            out.write(f"{', ' if i else ''}{_key(key)}: ")
            if kind is dict:
                out.write("{")
                for j, (k, blob) in enumerate(body):
                    out.write(f"{', ' if j else ''}{_key(k)}: {json.dumps(marshal.loads(blob))}")
                out.write("}")
            elif kind is list:
                out.write("[" + ", ".join(json.dumps(marshal.loads(b)) for b in body) + "]")
            else:
                out.write(json.dumps(marshal.loads(body)))
        out.write("}")

    def flush(self):
        """Write prs.json now if the state changed since the last write. The lock is held
        only for the snapshot; a write never replaces a newer one (tmp file + os.replace)."""
        if self.save_gen == self.written_gen:
            return
        with self.lock:
            gen = self.save_gen
            if gen == self.written_gen:
                return
            parts = self._snapshot()
        tmp = self.db.with_name(f"prs.json.{uuid.uuid4().hex}.tmp")
        started = time.monotonic()
        try:
            with open(tmp, "w") as out:
                self._encode(parts, out)
                size = out.tell()
            with self.replace_lock:
                if gen > self.written_gen:
                    os.replace(tmp, self.db)
                    self.written_gen = gen
                    self.save_s, self.saved_at, self.state_bytes = (
                        time.monotonic() - started, time.time(), size)
        finally:
            tmp.unlink(missing_ok=True)

    # -- open_pr idempotency cache ----------------------------------------------------------

    @staticmethod
    def _digest(text):
        return hashlib.sha256(text.encode("utf-8", "surrogatepass")).hexdigest()

    def _slim(self, entry, now):
        """Strip the description from a cached open_pr result (a replay rebuilds it, see
        _replay) and stamp it; returns True if the entry changed."""
        changed = False
        if not isinstance(entry.get("_ts"), (int, float)):
            entry["_ts"] = now
            changed = True
        if isinstance(entry.get("description"), str):
            entry["_description_sha256"] = self._digest(entry["description"])
            entry["description"] = None  # keeps the key's place in the result
            changed = True
        return changed

    def _cache_request(self, key, result):
        """Cache an open_pr result for retries of the same request (call with the lock held)."""
        now = time.time()
        entry = dict(result)
        self._slim(entry, now)
        requests = self.state["requests"]
        requests.pop(key, None)
        requests[key] = entry
        self._prune_requests(now)

    def _prune_requests(self, now):
        """Drop cached results older than REQUEST_TTL, then the oldest past MAX_REQUESTS
        (entries are in insertion order, so oldest first). Returns True if any was dropped."""
        requests = self.state["requests"]
        cut = now - REQUEST_TTL
        stale = []
        for key, entry in requests.items():
            if entry["_ts"] > cut:
                break
            stale.append(key)
        stale += list(requests)[len(stale):max(len(stale), len(requests) - MAX_REQUESTS)]
        for key in stale:
            del requests[key]
        return bool(stale)

    def _slim_requests(self):
        """Startup: slim every cached result (entries cached before slimming existed get the
        current time) and prune. Returns True if the state changed."""
        now = time.time()
        changed = False
        for entry in self.state["requests"].values():
            changed |= self._slim(entry, now)
        return self._prune_requests(now) or changed

    def _replay(self, key, description):
        """The result an open_pr request already returned (call with the lock held). The
        description comes from the retried call (the same request carries the same one) or
        else from the PR record, whichever matches the digest stored with the result."""
        entry = self.state["requests"][key]
        result = {k: v for k, v in entry.items() if not k.startswith("_")}
        digest = entry.get("_description_sha256")
        if digest is not None:
            rec = self.state["prs"].get(str(entry.get("pr"))) or {}
            candidates = [description, rec.get("description")]
            result["description"] = next(
                (c for c in candidates if isinstance(c, str) and self._digest(c) == digest),
                description)
        return result

    def _event(self, action, agent, **data):
        with open(self.ws / "prs.jsonl", "a") as out:
            out.write(json.dumps({"ts": time.time(), "event": action, "agent": agent.name,
                                  **data}) + "\n")

    def _pr(self, pr):
        rec = self.state["prs"].get(str(pr))
        if rec is not None and rec["state"] == "merging":
            raise ToolError(f"PR #{pr} is being merged; retry shortly", "busy")
        if rec is None or rec["state"] != "open":
            if self._draft(pr):
                raise ToolError(f"PR #{pr} is a draft whose head is not on the Hub yet; its author "
                                f"must retry open_pr with pr={pr}")
            raise ToolError("no open PR with that number")
        return rec

    def _draft(self, pr):
        return next((d for d in self.state["drafts"].values() if d["pr"] == pr), None)

    def _check_paused(self):
        if self.paused_env or os.path.exists(self.pause_file):
            raise ToolError(PAUSED, "paused")

    def _busy(self, agent, what, pr=None, reason="slot_wait"):
        log.warning("hub_busy: refused %s for agent=%s pr=%s reason=%s", what, agent.name, pr, reason)
        self.hub_busy_total += 1
        raise ToolError(BUSY, "hub_busy", retry_after_s=60)

    @contextlib.contextmanager
    def _hub_write(self, agent, what, pr=None, deadline=None, slots=None):
        """Hold a Hub write slot (from slots, default self.hub_slots), waiting at most until
        deadline (default: CHATBOT_HUB_WAIT_S).

        Never acquire this with main_lock or self.lock held."""
        slots = slots or self.hub_slots
        if deadline is None:
            deadline = time.monotonic() + self.hub_wait
        started = time.monotonic()
        acquired = slots.acquire(timeout=max(0.0, deadline - started))
        self.hub_slot_wait_s += time.monotonic() - started
        if not acquired:
            self._busy(agent, what, pr)
        try:
            yield
        finally:
            slots.release()

    def _create_gap(self, agent, deadline):
        """Reserve the next create_pull_request time, CHATBOT_PR_CREATE_GAP_S after the last
        one world-wide, and wait for it; refuse if that is after deadline."""
        with self.gap_lock:
            now = time.monotonic()
            at = max(now, self.last_create + self.pr_gap)
            if at > deadline:
                self._busy(agent, "create_pull_request", reason="pr_create_gap")
            self.last_create = at
        time.sleep(at - now)

    def _fetch_missing_heads(self):
        """Startup only: the bundle restore carries the PR refs the world knew at archive time;
        fetch, explicitly and in small batches, the open PRs whose recorded head isn't in the
        repository (pushes after the archive). Never fails the start: what stays missing is
        logged and will error on that PR only (its author's next update re-pushes it)."""
        try:
            missing = [rec["pr"] for rec in self.state["prs"].values()
                       if rec["state"] == "open" and rec.get("head_sha")
                       and not self.repo.has_commit(rec["head_sha"])]
            if not missing:
                return
            left = self.repo.fetch_pr_refs(missing)
            log.warning("startup: fetched %d PR heads missing after the restore; still missing: %s",
                        len(missing) - len(left), left or "none")
        except Exception as e:  # noqa: BLE001 - a fake or failing repository must not stop the start
            log.warning("startup: PR head check skipped: %s", e)

    def _sync_hub_states(self, every=0.0):
        """Reconcile records the world has as open with the Hub's PR states (a restore from a
        stale archive keeps PRs open that the Hub already closed or merged). At most once per
        `every` seconds; a Hub read taken outside the world lock and the Hub-write slot. Never
        raises."""
        now = time.monotonic()
        if now - self.hub_states_at < every:
            return
        self.hub_states_at = now
        states = getattr(self.repo.hub, "pr_states", None)
        if states is None:
            return
        try:
            seen = states()
        except Exception as exc:  # noqa: BLE001 - retried at the next period
            log.warning("Hub PR states not listed (%s); keeping the local records", type(exc).__name__)
            return
        with self.lock:
            changed = False
            for rec in self.state["prs"].values():
                status = seen.get(rec["pr"])
                if rec["state"] == "open" and status in ("closed", "merged"):
                    self._apply_hub_final(rec, status)
                    changed = True
            if changed:
                self._save()

    def _apply_hub_final(self, rec, status):
        """Record that the Hub closed or merged an open PR (lock held; the caller saves)."""
        if status == "merged":
            rec.update(state="merged", merged_sha=None, reconciled_from_hub=True)
        else:
            rec.update(state="closed", reconciled_from_hub=True)
        self._unnotify(rec["author"], rec["pr"], "mergeable")
        log.warning("PR #%s is %s on the Hub but was open here (author=%s); marked %s",
                    rec["pr"], status, rec["author"], rec["state"])

    def _mark_hub_final(self, pr, code):
        """A Hub write said PR `pr` is already closed or merged: update the record (or drop a
        parked draft) and return the ToolError for the agent. Lock held or not; caller saves
        if it needs to."""
        status = code[3:]
        with self.lock:
            rec = self.state["prs"].get(str(pr))
            if rec is not None and rec["state"] in ("open", "closing"):
                self._apply_hub_final(rec, status)
                self._save()
            draft = self._draft(pr)
            if draft:
                self.state["drafts"].pop(draft["branch"], None)
                self._save()
        return ToolError(f"PR #{pr} is already {status} on the Hub", code)

    def _reconcile(self, author=None):
        """Mark open PRs whose head is already in main as merged (call with the lock held).

        This recovers a Hub merge whose local bookkeeping failed, and state restored
        from an archive that predates merges. Uses the local main ref.
        """
        changed, main = False, self.repo.sha()
        for rec in self.state["prs"].values():
            if rec["state"] != "open" or (author and rec["author"] != author):
                continue
            key = (rec["head_sha"], main)  # a git subprocess per open PR only when main moved
            if key not in self.ancestry:
                if len(self.ancestry) >= 10000:
                    self.ancestry.clear()
                self.ancestry[key] = self.repo.ancestor(rec["head_sha"], main)
            if not self.ancestry[key]:
                continue
            merged = self.repo.merged_by(rec["head_sha"])
            reviewers = sorted(name for name, r in self._current_reviews(rec).items()
                               if r["verdict"] == "approve")
            rec.update(state="merged", merged_sha=merged)
            self._note_merge()
            event = {"pr": rec["pr"], "author": rec["author"], "reviewers": reviewers,
                     "parent_sha": self.repo.git("rev-parse", f"{merged}^1") if merged else None,
                     "head_sha": rec["head_sha"], "merged_sha": merged, "reconciled": True}
            self._event("merged", Agent("world"), **event)
            with open(self.ws / "eval-queue.jsonl", "a") as out:
                out.write(json.dumps({"ts": time.time(), **event}) + "\n")
            changed = True
        if changed:
            self._save()
        return changed

    def _notify(self, name, pr, kind, text):
        """Queue news for agent name (call with the lock held; the caller saves). One entry
        per (pr, kind): a repeated event replaces the older one. Keeps the newest INBOX_MAX."""
        box = [e for e in self.state["inbox"].get(name, ()) if (e["pr"], e["kind"]) != (pr, kind)]
        box.append({"pr": pr, "kind": kind, "text": text, "ts": time.time()})
        self.state["inbox"][name] = box[-INBOX_MAX:]

    def _unnotify(self, name, pr, kind):
        box = self.state["inbox"].get(name)
        if box:
            self.state["inbox"][name] = [e for e in box if (e["pr"], e["kind"]) != (pr, kind)]

    @staticmethod
    def _age(seconds):
        seconds = max(0, int(seconds))
        if seconds < 60:
            return "just now"
        if seconds < 3600:
            return f"{seconds // 60} min ago"
        if seconds < 86400:
            return f"{seconds // 3600} h ago"
        return f"{seconds // 86400} d ago"

    def call(self, tool, args, agent):
        """Run the tool; on success, attach and clear the caller's unread inbox.

        A dict result gets a top-level "inbox" list of strings (newest last). A list result
        (list_prs) becomes {"prs": [...], "inbox": [...], "pr_budget": {...}}, only when the
        inbox is non-empty. open_pr, review_pr, merge_pr and close_pr results carry pr_budget.
        A ToolError leaves the inbox untouched."""
        result = super().call(tool, args, agent)
        if inspect.isawaitable(result):
            return result
        if isinstance(result, dict) and tool in ("open_pr", "review_pr", "merge_pr", "close_pr"):
            with self.lock:
                result = {**result, "pr_budget": self._budget()}
        if not self.state["inbox"].get(agent.name):
            return result
        with self.lock:
            box = self.state["inbox"].pop(agent.name, None)
            if not box:
                return result
            self._save()
            budget = self._budget()
        now = time.time()
        inbox = [f"{e['text']} ({self._age(now - e['ts'])})" for e in box]
        if isinstance(result, dict):
            return {**result, "inbox": inbox}
        if isinstance(result, list):
            return {"prs": result, "inbox": inbox, "pr_budget": budget}
        return {"result": result, "inbox": inbox}

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
        """Apply an upload over tree (a checkout of its base, or an empty directory).

        Byte-identical files are not rewritten; tracked files missing from the upload are
        deleted. Returns the paths that differ from the tree, for the compile check.
        """
        files = snapshot.get("files")
        if not isinstance(files, dict):
            raise ToolError("invalid workspace upload", "bad_workspace")
        changed = {}
        for name, encoded in files.items():
            rel = PurePosixPath(name) if isinstance(name, str) else None
            if (rel is None or rel.is_absolute() or ".." in rel.parts or ".git" in rel.parts
                    or "\\" in name or not isinstance(encoded, str)):
                raise ToolError("invalid workspace upload", "bad_workspace")
            if len(encoded) > (MAX_FILE_BYTES + 2) // 3 * 4:
                raise ToolError(f"file too large: {name[:200]} is over {MAX_FILE_BYTES} bytes; "
                                "remove it from the clone or add it to .gitignore", "file_too_large")
            target = tree / rel
            try:
                data = base64.b64decode(encoded, validate=True)
            except ValueError as exc:
                raise ToolError("invalid workspace upload", "bad_workspace") from exc
            if len(data) > MAX_FILE_BYTES:
                raise ToolError(f"file too large: {name[:200]} is over {MAX_FILE_BYTES} bytes; "
                                "remove it from the clone or add it to .gitignore", "file_too_large")
            if target.is_file() and not target.is_symlink() and target.read_bytes() == data:
                continue
            changed[name] = encoded
        if (tree / ".git").is_dir():
            # Default excludes even if an agent drops .gitignore: caches are never committed.
            (tree / ".git/info").mkdir(exist_ok=True)
            (tree / ".git/info/exclude").write_text("\n".join(EXCLUDE) + "\n")
            tracked = self.repo.git("ls-files", "-z", at=tree).split("\0")
            for name in tracked:
                if name and name not in files:
                    (tree / name).unlink(missing_ok=True)
                    changed.setdefault(name, None)
        try:
            unpack({"files": {k: v for k, v in changed.items() if v is not None}}, tree)
        except (ValueError, TypeError, OSError) as exc:
            raise ToolError("invalid workspace upload", "bad_workspace") from exc
        return sorted(changed)

    def _contract(self, tree, names=None):
        """Check the fixed interface, and that names (default: every .py file) compile."""
        for name, expected in self.fixed.items():
            f = tree / name
            if not f.is_file() or f.read_bytes() != expected:
                raise ToolError(f"keep the seed interface fixed: {name}; edit bot/modules/ instead",
                                "fixed_interface")
        if names is None:
            names = [f.relative_to(tree).as_posix() for f in tree.rglob("*.py")
                     if ".git" not in f.relative_to(tree).parts]
        for name in names:
            f = tree / name
            if not name.endswith(".py") or not f.is_file():
                continue
            try:
                compile(f.read_bytes(), name, "exec")
            except (SyntaxError, ValueError):
                raise ToolError(f"Python does not compile: {name}", "compile") from None

    @staticmethod
    def _slug(agent: Agent, branch) -> str:
        """Normalize a branch argument to the bare slug, accepting only this agent's prefix."""
        text = branch.strip() if isinstance(branch, str) else ""
        own = f"agents/{agent.name}/"
        for prefix in ("refs/heads/" + own, own):
            if text.startswith(prefix):
                text = text[len(prefix):]
                break
        else:
            bare = text.removeprefix("refs/heads/")
            if bare.startswith("agents/"):
                raise ToolError(f"branch prefix must be your own (agents/{agent.name}/) or omitted; "
                                f"got \"{text[:80]}\"")
        if not NAME.fullmatch(text):
            raise ToolError("branch must be a short slug matching [a-z][a-z0-9-]{0,63} "
                            f"(you may omit the agents/{agent.name}/ prefix); got \"{str(branch)[:80]}\"")
        return text

    @tool(pool="slow")
    @logs_refusals
    def open_pr(self, agent: Agent, branch: str, title: str, description: str,
                path: str = "chatbot-swarm", pr: int | None = None) -> dict:
        """Submit your local clone, or update your existing PR with pr:<number>.

        The worker uploads path from your private workspace. Include tests and run_bot
        evidence in description. At most two open PRs per agent. Read CONTRIBUTING.md.
        branch is a short slug like my-topic; the world stores it as agents/<you>/<slug>.
        New PRs are budgeted swarm-wide: one is refused (code open_budget) while the swarm
        opened max(merges in the last hour, 20) new PRs in the last hour; updates with
        pr=<n> are never counted. Results carry pr_budget with the current counts.
        """
        started = time.monotonic()
        self._identity(agent)
        self._check_paused()
        branch = self._slug(agent, branch)
        if not isinstance(title, str) or not 3 <= len(title) <= 200:
            raise ToolError(f"title must be 3–200 characters (got {len(title) if isinstance(title, str) else 0})")
        if not isinstance(description, str) or not description.strip():
            raise ToolError("description must not be empty")
        snapshot = UPLOAD.get()
        if not isinstance(snapshot, dict):
            raise ToolError("submit a local clone through flock-tool; the worker uploads it")
        key = agent.request_id
        branch_ref = f"agents/{agent.name}/{branch}"
        with self.lock:
            if key and key in self.state["requests"]:
                return self._replay(key, description)
            if pr is None and branch_ref not in self.state["drafts"]:
                self._check_budget(agent)  # before any fetch, build or Hub call
        # The lock is held only for state and the main check; fetches, copies, compiles,
        # Hub calls and pushes run outside it. A reservation keeps the cap and slugs exact.
        self.repo.refresh(FETCH_TTL)
        base = snapshot.get("base_sha", "")
        token = object()
        with self.lock:
            if key and key in self.state["requests"]:
                return self._replay(key, description)
            if not self.repo.ancestor(base, "main"):
                raise ToolError("base is not a main commit; fetch main and rebase your branch")
            # A parked draft (its Hub PR exists, its head never landed) is reused, never recreated.
            draft = self._draft(pr) if pr is not None else self.state["drafts"].get(branch_ref)
            old = self._pr(pr) if pr is not None and not draft else None
            for existing in (old, draft):
                if existing and existing["author"] != agent.name:
                    raise ToolError("only the author can update this PR")
                if existing and existing["branch"] != branch_ref:
                    raise ToolError("use the original branch slug when updating a PR")
            number = draft["pr"] if draft else (pr if old else None)
            pending = self.pending.values()
            if key and any(p[3] == key for p in pending):
                raise ToolError("this request is already in progress", "busy")
            if number is not None and any(p[2] == number for p in pending):
                raise ToolError("an update of this PR is already in progress; retry when it finishes")
            if not old and not draft:
                self._reconcile(agent.name)
                mine = [r["pr"] for r in self.state["prs"].values()
                        if r["state"] in ("open", "merging") and r["author"] == agent.name]
                mine += [d["pr"] for d in self.state["drafts"].values() if d["author"] == agent.name]
                if len(mine) + sum(p[0] == agent.name and p[2] is None for p in pending) >= 2:
                    raise ToolError(f"at most two open PRs per agent: update one with pr=<n> or "
                                    f"close one with close_pr (yours: {', '.join(map(str, mine))})")
                if (any(r["branch"] == branch_ref for r in self.state["prs"].values())
                        or branch_ref in self.state["drafts"]
                        or any(p[1] == branch_ref for p in pending)):
                    raise ToolError("branch slug already used; choose a new topic slug")
                self._check_budget(agent)
                self.open_reserved.add(token)
            old_head = old["head_sha"] if old else None
            self.pending[token] = (agent.name, branch_ref, number, key)
        try:
            number, sha, main, old_head = self._publish(agent, snapshot, base, title, description,
                                                        branch_ref, number, old_head, started, token)
            with self.lock:
                if old and (old["state"] != "open" or old["head_sha"] != old_head):
                    raise ToolError("the PR was merged or closed during the update")
                record = {"pr": number, "state": "open", "author": agent.name,
                          "branch": branch_ref, "base_sha": main, "head_sha": sha,
                          "url": f"https://huggingface.co/datasets/{self.repo_id}/discussions/{number}",
                          "title": title, "description": description,
                          # Kept so reviews of an identical diff carry over a clean rebase.
                          "reviews": dict(old["reviews"]) if old else {}}
                self.state["prs"][str(number)] = record
                self.state["drafts"].pop(branch_ref, None)
                for name, r in (old["reviews"] if old else {}).items():
                    if r["verdict"] == "request_changes" and name != agent.name:
                        self._notify(name, number, "updated",
                                     f"PR #{number} was updated by {agent.name} after your "
                                     "request_changes; re-review the new head")
                result = {k: v for k, v in record.items() if k != "reviews"}
                if base != main:
                    result["rebased_from"] = base
                    result["note"] = (f"rebased onto main {main[:12]}; run "
                                      f"git -c protocol.version=0 fetch origin refs/pr/{number} "
                                      "and continue from FETCH_HEAD")
                if key:
                    self._cache_request(key, result)
                self._save()
                self._event("updated" if old else "opened", agent, **result)
                return result
        finally:
            with self.lock:
                self.pending.pop(token, None)
                self.open_reserved.discard(token)

    def _build(self, agent, snapshot, base, title, main):
        """Commit the upload on base, replayed onto main (outside the world lock)."""
        with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
            tree = Path(tmp) / "tree"
            self.repo.checkout(base, tree)
            self._files(snapshot, tree)
            self._contract(tree, self.repo.staged(tree))
            # An older base is rebased onto main here, before anything exists on the Hub.
            return self.repo.commit(tree, base, agent.name, title, onto=main)

    def _publish(self, agent, snapshot, base, title, description, branch_ref, pr, old_head, started,
                 token=None):
        """Publish the upload: update open PR pr (old_head set), finish parked draft pr, or
        create one Hub PR. At most one create_pull_request per call, and a created PR is
        never closed by the world: if its head cannot be pushed it is parked as a draft."""
        deadline = None
        for _attempt in range(3):
            main = self.repo.sha()
            sha = self._build(agent, snapshot, base, title, main)
            if deadline is None:
                deadline = min(time.monotonic() + self.hub_wait,
                               started + CALL_BUDGET - 2 * PUSH_ESTIMATE_S)
            if old_head is not None:  # an open PR: replace only the expected head
                with self._hub_write(agent, "push", pr, deadline):
                    old_head = self._lease_push(agent, pr, sha, f"refs/pr/{pr}", old_head, started)
                    self._lease_push(agent, pr, sha, f"refs/heads/{branch_ref}", old_head, started)
            else:
                with self._hub_write(agent, "open_pr", pr, deadline):
                    if pr is None:
                        self._create_gap(agent, deadline)
                        with self.main_lock:  # nothing exists on the Hub yet: redo on a moved main
                            if self.repo.sha() != main:
                                continue
                        pr = self.repo.hub.open(title, f"Agent: {agent.name}\nBase: {main}\n\n{description}")
                        with self.lock:  # the reservation becomes a counted open
                            self.open_reserved.discard(token)
                            now = time.time()
                            self._recent(self.state["pr_opens"], now).append(now)
                            self._save()
                        with self.gap_lock:
                            self.last_create = max(self.last_create, time.monotonic())
                    sha, main = self._land(agent, snapshot, base, title, branch_ref, pr, sha, main,
                                           started)
            self.repo.update_ref(f"refs/pr/{pr}", sha)
            self.repo.update_ref(f"refs/flock/pr/{pr}/{sha}", sha)
            return pr, sha, main, old_head
        raise ToolError("main kept moving while opening the PR; open it again", "busy")

    def _lease_push(self, agent, pr, sha, ref, expected, started):
        """Push sha to ref (refs/pr/N or the PR's branch) with a lease on expected; the caller
        holds a Hub slot, never the world lock. On a stale lease (the Hub's ref moved past the
        world's record, e.g. after a restart from an older archive), read the Hub's value from
        the refs API (or, failing that, fetch it with git), adopt it (for refs/pr/N: into the record's head_sha) and retry the push once.
        Returns the expected value the push landed with."""
        try:
            self.repo.push(sha, ref, expected=expected)
            return expected
        except ToolError as exc:
            if exc.code != "stale_lease":
                raise
        refusal = ToolError(f"your PR #{pr}'s head on the Hub is newer than the world's record; "
                            f"the world is catching up, retry open_pr with pr={pr} in a minute",
                            "stale_lease")
        # The refs API gives the Hub's value (the lease needs only the sha, not the object);
        # a git fetch of the ref is the fallback when the API gives nothing.
        hub, via = None, ""
        refs = getattr(self.repo.hub, "refs", None)
        if refs is not None:
            try:
                hub = refs().get(ref)
            except Exception as e:  # noqa: BLE001 - fall back to the git fetch
                log.warning("Hub refs lookup failed for PR #%s: %s", pr, type(e).__name__)
            via = " (via API)" if hub else ""
        if not hub:
            room = started + CALL_BUDGET - time.monotonic()
            if ref == f"refs/pr/{pr}":
                local = ref
                fetched = not self.repo.fetch_pr_refs([pr], delays=(1,) if room > 1 + 2 * PUSH_ESTIMATE_S else ())
            else:
                local = f"refs/flock/hub/{ref.removeprefix('refs/')}"
                fetched = self.repo.fetch_ref(ref, local)
            if fetched:
                try:
                    hub = self.repo.git("rev-parse", "--verify", f"{local}^{{commit}}")
                except ToolError:
                    hub = None
        if not hub or hub == expected:
            log.warning("stale lease on %s for PR #%s (agent=%s): Hub head not fetched; refused",
                        ref, pr, agent.name)
            raise refusal
        if ref == f"refs/pr/{pr}":
            with self.lock:
                rec = self.state["prs"].get(str(pr))
                if rec is None or rec["state"] != "open" or rec["head_sha"] != expected:
                    raise ToolError("the PR was merged or closed during the update")
                rec["head_sha"] = hub
                self._save()
            log.warning("adopted Hub head for PR #%s after a stale lease%s: %s -> %s",
                        pr, via, expected[:12], hub[:12])
        else:
            log.warning("adopted Hub value of %s for PR #%s after a stale lease%s", ref, pr, via)
        self.stale_adoptions += 1
        if time.monotonic() + PUSH_ESTIMATE_S > started + CALL_BUDGET:
            raise refusal
        try:
            self.repo.push(sha, ref, expected=hub)
        except ToolError as exc:
            if exc.code != "stale_lease":
                raise
            log.warning("stale lease again on %s for PR #%s (agent=%s); refused", ref, pr, agent.name)
            raise refusal from None
        return hub if ref == f"refs/pr/{pr}" else expected

    def _land(self, agent, snapshot, base, title, branch_ref, number, sha, main, started):
        """Push the head of the world's own (draft) Hub PR and mark it ready; the caller holds
        a Hub slot. Retries briefly within the call budget, then parks the draft."""
        delays = list(PUSH_RETRY_DELAYS)
        while True:
            try:
                # refs/pr/N and the agent's branch are written only by the world: overwrite.
                self.repo.push(sha, f"refs/pr/{number}", force=True)
                self.repo.push(sha, f"refs/heads/{branch_ref}", force=True)
                self.repo.hub.ready(number)
                return sha, main
            except ToolError as exc:
                if exc.code in FINAL_CODES:  # closed or merged on the Hub: no retry loop
                    raise self._mark_hub_final(number, exc.code) from None
                failure = exc
            except Exception as exc:  # noqa: BLE001 -- the Hub PR exists: never orphan it
                failure = exc
            delay = delays.pop(0) if delays else None
            if delay is None or time.monotonic() + delay + PUSH_ESTIMATE_S > started + CALL_BUDGET:
                break
            time.sleep(delay)
            try:
                self.repo.fetch()
                if self.repo.sha() != main:  # main moved: rebase within the same PR number
                    main = self.repo.sha()
                    sha = self._build(agent, snapshot, base, title, main)
            except Exception as exc:  # noqa: BLE001
                failure = exc
                break
        code = getattr(failure, "code", type(failure).__name__)
        with self.lock:
            self.state["drafts"][branch_ref] = {
                "pr": number, "author": agent.name, "branch": branch_ref, "base_sha": main,
                "sha": sha, "title": title, "parked_at": time.time(),
                "url": f"https://huggingface.co/datasets/{self.repo_id}/discussions/{number}"}
            self._save()
            self._event("parked", agent, pr=number, branch=branch_ref, code=code)
        log.warning("parked draft PR #%s for agent=%s: %s", number, agent.name, code)
        if code == "stale_base":
            raise ToolError(f"{failure.message}; your PR #{number} is kept: retry open_pr with "
                            f"pr={number}", "stale_base")
        raise ToolError(f"the Hub did not accept the push for your PR #{number}; retry open_pr "
                        f"with pr={number} in a minute", "hub_busy", retry_after_s=60)

    @tool
    @logs_refusals
    def review_pr(self, agent: Agent, pr: int, head_sha: str,
                  verdict: Literal["approve", "request_changes"], comment: str) -> dict:
        """Review the exact head SHA you inspected or tested, independently of the author.

        Reviews are unlimited: the review counts at once, and its Hub comment is posted
        in the background."""
        self._identity(agent)
        if verdict not in ("approve", "request_changes") or not comment.strip():
            raise ToolError("give a verdict and a review comment")

        def check():
            rec = self._pr(pr)
            if rec["author"] == agent.name:
                raise ToolError("another agent must review your PR; self-review is refused")
            # Only the world writes PR refs, so its record is the PR head.
            if head_sha != rec["head_sha"]:
                raise ToolError("the PR head changed; inspect it again before reviewing")
            return rec

        with self.lock:
            base = check()["base_sha"]
        review = {"head_sha": head_sha, "patch_id": self._patch_id(base, head_sha),
                  "patch_id0": self._patch_id(base, head_sha, context=False),
                  "verdict": verdict, "comment": comment}
        with self.lock:
            rec = check()
            rec["reviews"][agent.name] = review
            # No Hub call here: the outbox thread posts the comment when the Hub slot is free.
            self._queue_comment(agent, pr, f"{verdict} by {agent.name} on {head_sha}\n\n{comment}")
            snippet = " ".join(comment.split())
            if len(snippet) > INBOX_SNIPPET:
                snippet = snippet[:INBOX_SNIPPET - 3] + "..."
            author = rec["author"]
            self._notify(author, pr, f"review:{agent.name}",
                         f"{agent.name} reviewed your PR #{pr}: {verdict}: {snippet}")
            try:
                approvers = self._approved(rec)
            except ToolError:
                self._unnotify(author, pr, "mergeable")
            else:
                self._notify(author, pr, "mergeable", f"PR #{pr} approved by "
                             f"{', '.join(sorted(approvers))}; ready to merge_pr")
            self._save()
            self._event("reviewed", agent, pr=pr, **review)
            return {"pr": pr, "reviewer": agent.name, **review}

    def _patch_id(self, base, head, context=True):
        try:
            return (self.repo.patch_id if context else self.repo.patch_id0)(base, head)
        except ToolError:
            return None  # unknown change identity: only exact-head reviews count

    def _current_reviews(self, rec):
        """Reviews of the current head, or of the same change (a clean rebase).

        A review carries to a new head when its context-free patch-id (patch_id0) equals the
        current one, so main changing lines next to the PR's hunks does not void it; or when
        its patch_id (the diff with context, the older formula) equals the current one.
        Reviews stored before patch_id0 existed keep only the patch_id match: reviews do not
        record the base they were made on, so patch_id0 cannot be backfilled for them.
        Reviews stored without either only count for their exact head.
        """
        current, ids = {}, {}

        def ident(field):
            if field not in ids:
                ids[field] = self._patch_id(rec["base_sha"], rec["head_sha"],
                                            context=field == "patch_id")
            return ids[field]

        for name, r in rec["reviews"].items():
            if r["head_sha"] != rec["head_sha"] and not any(
                    r.get(field) is not None and r[field] == ident(field)
                    for field in ("patch_id0", "patch_id")):
                continue
            current[name] = r
        return current

    def _approved(self, rec):
        reviews = {name: r for name, r in self._current_reviews(rec).items()
                   if name != rec["author"]}
        if any(r["verdict"] == "request_changes" for r in reviews.values()):
            raise ToolError("a reviewer still requests changes on this head")
        names = [name for name, r in reviews.items() if r["verdict"] == "approve"]
        if not names:
            raise ToolError("another agent must approve the current PR head")
        return names

    @tool(pool="slow")
    @logs_refusals
    def merge_pr(self, agent: Agent, pr: int) -> dict:
        """Merge an independently approved PR after checking its base and bot contract."""
        self._identity(agent)
        self._check_paused()
        self.repo.refresh(FETCH_TTL)
        with self.lock:
            rec = self._pr(pr)
            self._approved(rec)
            if self._reconcile(rec["author"]) and rec["state"] == "merged":
                return self._merged_event(rec)
            main, head = self.repo.sha(), rec["head_sha"]
            if main != rec["base_sha"]:
                raise ToolError("stale PR: fetch main, rebase and update the PR; approvals carry "
                                "over if your diff is unchanged, otherwise ask for a fresh review")
        with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
            tree = Path(tmp) / "tree"
            self.repo.checkout(head, tree)
            self._contract(tree, self.repo.changed(main, head))
            verdict = self._run(tree, ["Hello", "My name is Sam", "", "What is my name?", "Bye"],
                                self.merge_slots)
            if not verdict["ok"]:
                raise ToolError("bot contract failed: one nonempty reply per input, no crash, 300s total")
        self.repo.refresh(0)  # outside the lock; the compare-and-swap below uses local refs
        # The merge Hub slot is taken before the lock (see _hub_write); it is separate from
        # the slots of pushes, opens and closes, so a merge never waits behind them.
        with self._hub_write(agent, "merge", pr, slots=self.merge_hub_slots), self.main_lock:
            with self.lock:
                rec = self._pr(pr)
                reviewers = self._approved(rec)
                if (self.repo.sha() != main or rec["head_sha"] != head
                        or any(p[2] == pr for p in self.pending.values())):
                    raise ToolError("main or the PR changed during checks; update and review again")
                rec["state"] = "merging"  # no update, review, close or merge while the Hub call runs
            try:
                try:
                    self.repo.hub.merge(pr, f"Approved by {', '.join(reviewers)}; head {head}; base {main}; contract passed")
                except ToolError as exc:
                    # The Hub may have merged before failing (a 5xx or timeout after commit).
                    try:
                        self.repo.fetch()
                    except ToolError:
                        pass
                    with self.lock:
                        rec["state"] = "open"
                        if self._reconcile(rec["author"]) and rec["state"] == "merged":
                            return self._merged_event(rec)
                    if exc.code in FINAL_CODES:  # already closed or merged on the Hub
                        raise self._mark_hub_final(pr, exc.code) from None
                    raise
                # Local main must move before main_lock is released: opens compare against it.
                self.repo.fetch()
                merged = self.repo.sha()
                with self.lock:
                    rec.update(state="merged", merged_sha=merged)
                    self._note_merge()
                    if rec["author"] != agent.name:
                        self._unnotify(rec["author"], pr, "mergeable")
                        self._notify(rec["author"], pr, "merged", f"your PR #{pr} was merged by {agent.name}")
                    self._save()
                    event = {"pr": pr, "author": rec["author"], "reviewers": reviewers,
                             "parent_sha": main, "head_sha": head, "merged_sha": merged}
                    self._event("merged", agent, **event)
                    # Append-only integration hook, consumed by the separate evaluation workstream.
                    with open(self.ws / "eval-queue.jsonl", "a") as out:
                        out.write(json.dumps({"ts": time.time(), **event}) + "\n")
                    return event
            finally:
                with self.lock:
                    if rec["state"] == "merging":  # failed before the record was settled
                        rec["state"] = "open"

    def _merged_event(self, rec):
        return {"pr": rec["pr"], "author": rec["author"], "head_sha": rec["head_sha"],
                "merged_sha": rec.get("merged_sha"), "already_merged": True}

    @tool(pool="slow")
    @logs_refusals
    def close_pr(self, agent: Agent, pr: int, reason: str) -> dict:
        """Close your open PR (frees one of your two open-PR slots). Give a short reason."""
        self._identity(agent)
        if not reason.strip() or len(reason) > 2000:
            raise ToolError("give a short reason (at most 2000 chars)")
        with self.lock:
            draft = self._draft(pr)
        if draft:
            return self._close_draft(agent, draft, reason)
        with self.lock:
            rec = self.state["prs"].get(str(pr))
            if rec is not None and rec["state"] == "open":
                self._reconcile(rec["author"])
            rec = self._pr(pr)
            if rec["author"] != agent.name and agent.name not in self.operators:
                raise ToolError("only the author or the run operator can close this PR")
            if any(p[2] == pr for p in self.pending.values()):
                raise ToolError("an update of this PR is in progress; retry when it finishes", "busy")
            rec["state"] = "closing"  # no update, review or merge while the Hub call runs
        try:
            try:
                with self._hub_write(agent, "close", pr):
                    self.repo.hub.close(pr, f"Closed by {agent.name}: {reason}")
            except ToolError as exc:
                if exc.code == "hub_busy":
                    raise
                if exc.code in FINAL_CODES:
                    raise self._mark_hub_final(pr, exc.code) from None
                status = getattr(self.repo.hub, "status", None)
                if status is None or status(pr) != "closed":
                    raise
        except BaseException:
            with self.lock:
                if rec["state"] == "closing":  # not already settled from the Hub's answer
                    rec["state"] = "open"
            raise
        with self.lock:
            rec.update(state="closed", closed_by=agent.name, close_reason=reason)
            self._notify_closed(agent, rec["author"], pr, reason)
            self._save()
            event = {"pr": pr, "author": rec["author"], "head_sha": rec["head_sha"], "reason": reason}
            self._event("pr_closed", agent, **event)
            return {**event, "state": "closed", "closed_by": agent.name}

    def _notify_closed(self, agent, author, pr, reason):
        if author != agent.name:
            self._unnotify(author, pr, "mergeable")
            self._notify(author, pr, "closed",
                         f"your PR #{pr} was closed by {agent.name}: {reason[:INBOX_SNIPPET]}")

    def _close_draft(self, agent, draft, reason):
        pr = draft["pr"]
        with self.lock:
            if draft["author"] != agent.name and agent.name not in self.operators:
                raise ToolError("only the author or the run operator can close this PR")
            if any(p[2] == pr for p in self.pending.values()):
                raise ToolError("an update of this PR is in progress; retry when it finishes", "busy")
        try:
            with self._hub_write(agent, "close", pr):
                self.repo.hub.close(pr, f"Closed by {agent.name}: {reason}")
        except ToolError as exc:
            if exc.code in FINAL_CODES:
                raise self._mark_hub_final(pr, exc.code) from None
            raise
        with self.lock:
            self.state["drafts"].pop(draft["branch"], None)
            self._notify_closed(agent, draft["author"], pr, reason)
            self._save()
            event = {"pr": pr, "author": draft["author"], "head_sha": None, "reason": reason}
            self._event("pr_closed", agent, draft=True, **event)
            return {**event, "state": "closed", "closed_by": agent.name}

    @tool(read_only=True)
    def list_prs(self, state: str = "open", author: str = "", needs_review: bool = False,
                 offset: int = 0) -> list[dict]:
        """Find PRs, their head SHA and reviews. state is open, merged, closed or all.

        A PR with "draft": true is open on the Hub but its head is not pushed yet (head_sha is
        null); it counts toward its author's open PRs and cannot be reviewed or merged until
        its author retries open_pr with pr=<number>. state open also lists PRs whose merge
        is running ("state": "merging").
        Returns a list of PRs; if you have unread inbox news it is
        {"prs": [...], "inbox": [...], "pr_budget": {...}} (the swarm's new-PR budget).
        """
        if offset < 0:
            raise ToolError("offset must be nonnegative")
        if state not in ("open", "merged", "closed", "all"):
            raise ToolError("state must be open, merged, closed or all")
        # Shallow copies under the lock (reviews are replaced, never edited in place); the
        # review filter's git and the deep copy run outside it.
        with self.lock:
            records = [{**rec, "reviews": dict(rec["reviews"])} for rec in self.state["prs"].values()
                       if (state == "all" or rec["state"] == state
                           or (state == "open" and rec["state"] == "merging"))
                       and (not author or rec["author"] == author)]
            drafts = [] if state not in ("open", "all") or needs_review else [
                {"pr": d["pr"], "state": "open", "draft": True, "author": d["author"],
                 "branch": d["branch"], "base_sha": d["base_sha"], "head_sha": None,
                 "url": d["url"], "title": d["title"], "reviews": {}}
                for d in self.state["drafts"].values() if not author or d["author"] == author]
        if needs_review:
            records = [rec for rec in records if not any(
                r["verdict"] == "approve" for r in self._current_reviews(rec).values())]
        return json.loads(json.dumps(records + drafts))[offset:offset + 100]

    @tool(read_only=True)
    def get_pr(self, pr: int, path: str | None = None) -> dict:
        """Show a PR from the world's copy, no git fetch needed: its head and base SHAs, the
        changed files with +/- line counts and the diff against its base. With path: that
        file's full content at the PR head and its diff only."""
        if path is not None:
            rel = PurePosixPath(path) if isinstance(path, str) else None
            if (not path or rel is None or rel.is_absolute() or ".." in rel.parts
                    or "\\" in path or "\0" in path):
                raise ToolError("path must be a relative file path inside the repository")
            path = rel.as_posix()
        with self.lock:  # a quick state read; git runs below without the lock
            rec = self.state["prs"].get(str(pr))
            if rec is None:
                if self._draft(pr):
                    raise ToolError(f"PR #{pr} is a draft whose head is not on the Hub yet; "
                                    "there is nothing to show")
                raise ToolError("no PR with that number")
            rec = {k: rec.get(k) for k in ("pr", "title", "author", "state", "branch", "url",
                                           "description", "head_sha", "base_sha")}
        head, base = rec["head_sha"], rec["base_sha"]
        with contextlib.suppress(ToolError):  # main moves rarely without the world: cheap
            self.repo.refresh(FETCH_TTL)
        if not (self.repo.has_commit(head) and self.repo.has_commit(base)):
            self.repo.fetch_pr_refs([pr], delays=(1,))  # once, from the Hub
            if not (self.repo.has_commit(head) and self.repo.has_commit(base)):
                raise ToolError(f"PR #{pr}'s head {str(head)[:12]} is not in the world's "
                                "repository (the Hub did not return it); retry later or fetch "
                                "it with git", "missing_head")
        try:
            main = self.repo.sha()
        except ToolError:
            main = None
        description = rec["description"] or ""
        if len(description) > DESCRIPTION_CHARS:
            description = description[:DESCRIPTION_CHARS - 3] + "..."
        result = {**rec, "description": description, "base_is_main": base == main,
                  "main_sha": main}
        spec = ("--", path) if path else ()
        files = []
        stat = self.repo.git_bytes("diff", "--numstat", "-z", "--no-renames", base, head, *spec)
        for entry in stat.decode(errors="replace").split("\0"):
            if not entry:
                continue
            added, deleted, name = entry.split("\t", 2)
            binary = added == "-"
            files.append({"path": name, "added": None if binary else int(added),
                          "deleted": None if binary else int(deleted)}
                         | ({"binary": True} if binary else {}))
        result["files"] = files
        diff = self.repo.git_bytes("diff", "--no-color", "--no-ext-diff", "--no-renames", base,
                                   head, *spec)
        result["diff_bytes"] = len(diff)
        result["truncated"] = len(diff) > MAX_DIFF_BYTES
        if result["truncated"]:
            cut = diff[:MAX_DIFF_BYTES]
            diff = cut[:cut.rfind(b"\n") + 1] or cut
            result["hint"] = (f"the diff is {result['diff_bytes']} bytes; showing the first "
                              f"{len(diff)}. Pass path=<file> for one file's diff and content")
        result["diff"] = diff.decode(errors="replace")
        if path:
            result.update(self._file_at(head, path))
            if not files and result["content"] is None:
                raise ToolError(f"{path[:200]} is neither in PR #{pr}'s head nor in its diff")
        return result

    def _file_at(self, sha, path):
        """{"content", "size", ...} of path at commit sha (content None when absent)."""
        listing = self.repo.git_bytes("ls-tree", "-z", "--long", sha, "--", path)
        entry = listing.split(b"\0", 1)[0].decode(errors="replace")
        if not entry:
            return {"content": None, "note": "the file does not exist at the PR head"}
        meta, _, _ = entry.partition("\t")
        kind, size = meta.split()[1], meta.split()[3]
        if kind != "blob":
            return {"content": None, "note": f"{path[:200]} is a {kind}, not a file"}
        data = self.repo.git_bytes("cat-file", "blob", f"{sha}:{path}")
        out = {"size": int(size)}
        if b"\0" in data[:8192]:
            return {**out, "content": None, "binary": True}
        out["content_truncated"] = len(data) > MAX_FILE_BYTES
        out["content"] = data[:MAX_FILE_BYTES].decode(errors="replace")
        return out

    @tool(pool="slow")
    def run_bot(self, ref_or_path: str, lines: list[str]) -> dict:
        """Run main, pr:<number>, or a local clone path. 300 seconds for the whole conversation."""
        with tempfile.TemporaryDirectory(dir=self.ws) as tmp:
            tree = Path(tmp) / "tree"
            snapshot = UPLOAD.get()
            if ref_or_path == "main" or ref_or_path.startswith("pr:"):
                self.repo.refresh(FETCH_TTL)
                sha = self.repo.sha(ref_or_path)
                self.repo.checkout(sha, tree)
            elif isinstance(snapshot, dict):
                sha = snapshot.get("base_sha")
                if isinstance(sha, str) and self.repo.ancestor(sha, "main"):
                    self.repo.checkout(sha, tree)  # then only changed files are written/compiled
                    self._contract(tree, self._files(snapshot, tree))
                else:
                    tree.mkdir()
                    self._files(snapshot, tree)
                    self._contract(tree)
            else:
                raise ToolError("use main, pr:<number>, or a clone path inside your workspace")
            return {"ref": ref_or_path, "sha": sha, **self._run(tree, lines)}

    def _run(self, tree, lines, slots=None):
        slots = slots or self.slots
        if not slots.acquire(timeout=10):
            raise ToolError("bot runners are busy; retry later", "busy")
        try:
            return run(tree, lines)
        finally:
            slots.release()


if __name__ == "__main__":
    from flock.world.server import serve
    serve("chatbot4:Chatbot4World", host="0.0.0.0", port=int(os.environ.get("PORT", "7860")))
