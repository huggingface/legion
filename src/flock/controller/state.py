"""Controller state in a bucket, for deployments whose disk does not survive a restart (job mode,
docs/deploy-job.md). Layout under the state URL (`hf://buckets/<ns>/<bucket>/controller/<group>`):

    state.db             SQLite snapshot (online backup API: consistent while the app writes)
    events/<swarm>/<agent>.jsonl   per-agent event streams, mirrored after the snapshot

`push` takes the snapshot first and mirrors the event files second, so a restored event file is
never shorter than the restored index by more than what the index never saw; `Store.repair_events`
trims the rest after a restore. A push is skipped when the store has not changed since the last
one. Anything written after the last push is lost on a hard kill: the loss window is the sync
interval (60 s by default), and a SIGTERM triggers a final push.

Event files are append-only, so the bucket keeps what was uploaded of each one: a periodic push
re-uploads a changed file whole, but the final push (SIGTERM, a short deadline) uploads only the
bytes appended since, as `events/<swarm>/<agent>.jsonl.tail/<offset>`. `restore` appends those
tails to their file (in offset order, stopping at a gap), and the next periodic push uploads the
whole file again and deletes them. After a restore the restored files count as uploaded, so the
first push after a restart is incremental too.

Health files (docs/health.md) are mirrored too: `metrics/<swarm>/latest.json` and
`<date>.jsonl`, `rollups/<swarm>/<hour>.jsonl` and `run.json`, each uploaded whole when it
changed, and restored with the snapshot so the day and hour files keep growing.

A snapshot with no swarms is never restored: once the last swarm is removed, the next push
writes an empty snapshot, so a later controller in the same group starts clean.
`scripts/deploy-job.sh --down` also deletes the whole prefix (`clear`).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import tempfile
import threading
from pathlib import Path

from flock.controller.store import DB_NAME, Store
from flock.worker.sync import BucketStore

log = logging.getLogger("flock.state")

STATE_DB = "state.db"
EVENTS = "events"
DEFAULT_INTERVAL = 60.0
TAIL = ".tail"  # events/<swarm>/<agent>.jsonl.tail/<offset>: bytes appended after <offset>
MIRRORED = ("metrics", "rollups", "run.json")  # health files, uploaded whole when changed
_CHECK = 4096  # bytes before the uploaded end that must still match for a tail upload


def _end_hash(path: Path, end: int) -> str:
    """Hash of the `_CHECK` bytes before `end`: the uploaded prefix is still the same file."""
    with path.open("rb") as f:
        f.seek(max(0, end - _CHECK))
        return hashlib.sha256(f.read(end - max(0, end - _CHECK))).hexdigest()


def _tail_name(offset: int) -> str:
    return f"{offset:016d}"


def merge_tails(events_dir: Path) -> tuple[dict[str, list[str]], set[str]]:
    """Append every `<file>.tail/<offset>` part to its file, in offset order, and remove the
    parts locally. Returns the remote tail paths per file, and the files left behind the
    bucket by a gap (a missing part) that must be uploaded whole."""
    tails: dict[str, list[str]] = {}
    gaps: set[str] = set()
    if not events_dir.is_dir():
        return tails, gaps
    for tail_dir in sorted(p for p in events_dir.rglob("*" + TAIL) if p.is_dir()):
        base = tail_dir.with_name(tail_dir.name[: -len(TAIL)])
        rel = f"{EVENTS}/{base.relative_to(events_dir).as_posix()}"
        parts = sorted((p for p in tail_dir.iterdir() if p.is_file() and p.name.isdigit()),
                       key=lambda p: int(p.name))
        tails[rel] = [f"{rel}{TAIL}/{p.name}" for p in parts]
        size = base.stat().st_size if base.exists() else 0
        with base.open("ab") as out:
            for part in parts:
                offset, data = int(part.name), part.read_bytes()
                if offset > size:
                    gaps.add(rel)
                    break
                if offset + len(data) > size:
                    out.write(data[size - offset:])
                    size = offset + len(data)
        for part in tail_dir.iterdir():
            part.unlink()
        tail_dir.rmdir()
    return tails, gaps


class StateSync:
    def __init__(self, bucket: BucketStore, data_dir: str | Path, *, events: bool = True) -> None:
        self.bucket = bucket
        self.data_dir = Path(data_dir)
        self.events = events
        self._pushed_at_change: int | None = None  # store.changes() at the last push
        self._lock = threading.Lock()  # the periodic push and the final push never overlap
        self.pushes = 0
        # What the bucket holds of each event file: (bytes, mtime_ns, _end_hash at those bytes).
        # None until the first push (or a restore) establishes it.
        self._uploaded: dict[str, tuple[int, int, str]] | None = None
        self._remote_tails: dict[str, list[str]] = {}  # tail parts in the bucket, per file
        self._mirrored: dict[str, tuple[int, int]] = {}  # health file -> (bytes, mtime_ns) sent

    def restore(self) -> bool:
        """Before the Store opens: replace the local DB (and event files) with the bucket's copy.
        Returns False when the bucket has no snapshot, or one without swarms (the last swarm
        was removed): the controller then starts empty."""
        if STATE_DB not in self.bucket.list(STATE_DB):
            log.info("state: no snapshot in the bucket; starting empty")
            return False
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.data_dir / (STATE_DB + ".restore")
        self.bucket.download([(STATE_DB, tmp)])
        summary = snapshot_summary(tmp)
        if not summary["swarms"]:
            tmp.unlink(missing_ok=True)
            log.info("state: the bucket snapshot has no swarms (all removed); starting empty")
            return False
        db = self.data_dir / DB_NAME
        for suffix in ("-wal", "-shm"):  # a stale WAL must not be replayed onto the snapshot
            Path(str(db) + suffix).unlink(missing_ok=True)
        os.replace(tmp, db)
        if self.events:
            events_dir = self.data_dir / EVENTS
            self.bucket.sync_down(EVENTS, events_dir)
            self._remote_tails, gaps = merge_tails(events_dir)
            # The restored files are what the bucket holds: the next push sends changes only.
            self._uploaded = {rel: rec for rel, (path, rec) in self._scan(events_dir).items()
                              if rel not in gaps}
        for name in MIRRORED:
            if name.endswith(".json"):
                if name in self.bucket.list(name):
                    self.bucket.download([(name, self.data_dir / name)])
            else:
                self.bucket.sync_down(name, self.data_dir / name)
        self._mirrored = dict(self._scan_mirrored())
        log.info("state: restored %d swarms (%s), %d agents, %d workers from the bucket",
                 len(summary["swarms"]), ", ".join(summary["swarms"]), summary["agents"],
                 summary["workers"])
        return True

    def clear(self) -> int:
        """Delete everything under the state prefix (`deploy-job.sh --down`). Returns the count."""
        files = self.bucket.list("")
        if files:
            self.bucket.delete(files)
        return len(files)

    def _scan_mirrored(self) -> list[tuple[str, tuple[int, int]]]:
        out = []
        for name in MIRRORED:
            root = self.data_dir / name
            for path in [root] if root.is_file() else root.rglob("*") if root.is_dir() else []:
                if path.is_file() and not path.name.endswith(".tmp"):
                    st = path.stat()
                    out.append((path.relative_to(self.data_dir).as_posix(),
                                (st.st_size, st.st_mtime_ns)))
        return out

    @staticmethod
    def _scan(events_dir: Path) -> dict[str, tuple[Path, tuple[int, int, str]]]:
        out = {}
        for path in events_dir.rglob("*"):
            if path.is_file():
                st = path.stat()
                rel = f"{EVENTS}/{path.relative_to(events_dir).as_posix()}"
                out[rel] = (path, (st.st_size, st.st_mtime_ns, _end_hash(path, st.st_size)))
        return out

    def push(self, store: Store, *, force: bool = False, final: bool = False) -> int:
        """Snapshot the DB and mirror the event files. Blocking; run it in a thread. Returns
        the number of files pushed (the DB plus the event files), 0 when nothing changed.
        `final` (the shutdown push): upload only the bytes appended to each event file."""
        with self._lock:
            changes = store.changes()
            if not force and changes == self._pushed_at_change:
                return 0
            snap = self.data_dir / (STATE_DB + ".snapshot")
            store.backup_to(snap)
            try:
                self.bucket.upload([(snap, STATE_DB)])
            finally:
                snap.unlink(missing_ok=True)
            files = 1
            events_dir = self.data_dir / EVENTS
            if self.events and events_dir.is_dir():
                files += self._push_events(events_dir, final)
            changed = [(rel, rec) for rel, rec in self._scan_mirrored()
                       if self._mirrored.get(rel) != rec]
            if changed:
                self.bucket.upload([(self.data_dir / rel, rel) for rel, _ in changed])
                self._mirrored.update(changed)
                files += len(changed)
            self._pushed_at_change = changes
            self.pushes += 1
            return files

    def _push_events(self, events_dir: Path, final: bool) -> int:
        if self._uploaded is None:
            self.bucket.sync_up(events_dir, EVENTS)  # also drops remote files missing here
            self._uploaded = {rel: rec for rel, (_, rec) in self._scan(events_dir).items()}
            self._remote_tails = {}
            return len(self._uploaded)
        whole: list[tuple[Path, str]] = []
        n = 0
        with tempfile.TemporaryDirectory(prefix="flock-tails-", dir=self.data_dir) as tmp:
            parts: list[tuple[Path, str]] = []
            records: dict[str, tuple[int, int, str]] = {}
            for path in events_dir.rglob("*"):
                if not path.is_file():
                    continue
                rel = f"{EVENTS}/{path.relative_to(events_dir).as_posix()}"
                st = path.stat()
                old = self._uploaded.get(rel)
                if old is not None and (st.st_size, st.st_mtime_ns) == old[:2]:
                    continue
                if final and old is not None and st.st_size > old[0] and _end_hash(path, old[0]) == old[2]:
                    with path.open("rb") as f:
                        f.seek(old[0])
                        data = f.read(st.st_size - old[0])
                    local = Path(tmp) / f"{len(parts)}"
                    local.write_bytes(data)
                    name = f"{rel}{TAIL}/{_tail_name(old[0])}"
                    parts.append((local, name))
                    end = old[0] + len(data)
                    records[rel] = (end, st.st_mtime_ns, _end_hash(path, end))
                    self._remote_tails.setdefault(rel, []).append(name)
                else:
                    whole.append((path, rel))
                    records[rel] = (st.st_size, st.st_mtime_ns, _end_hash(path, st.st_size))
            if parts:
                self.bucket.upload(parts)
                n += len(parts)
        if whole:
            self.bucket.upload(whole)
            n += len(whole)
            stale = [t for _, rel in whole for t in self._remote_tails.pop(rel, [])]
            if stale:
                self.bucket.delete(stale)
        self._uploaded.update(records)
        return n


def snapshot_summary(path: str | Path) -> dict:
    """Swarm names and agent/worker counts of a snapshot file (read-only)."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        swarms = [r[0] for r in db.execute("SELECT name FROM swarms ORDER BY name")]
        agents = db.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
        workers = db.execute("SELECT COUNT(*) FROM workers").fetchone()[0]
    finally:
        db.close()
    return {"swarms": swarms, "agents": agents, "workers": workers}


def state_sync_from_env(env, data_dir: str | Path) -> StateSync | None:
    """FLOCK_STATE_URL (hf://buckets/<ns>/<bucket>/<prefix>) turns state sync on;
    FLOCK_STATE_EVENTS=0 keeps event files local."""
    url = env.get("FLOCK_STATE_URL")
    if not url:
        return None
    from flock.worker.sync import HubBucketStore

    bucket = HubBucketStore(url, token=env.get("HF_TOKEN") or None)
    return StateSync(bucket, data_dir, events=env.get("FLOCK_STATE_EVENTS", "1") != "0")
