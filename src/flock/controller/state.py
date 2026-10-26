"""Controller state in a bucket, for deployments whose disk does not survive a restart (job mode,
docs/deploy-job.md). Layout under the state URL (`hf://buckets/<ns>/<bucket>/controller/<group>`):

    state.db             SQLite snapshot (online backup API: consistent while the app writes)
    events/<swarm>/<agent>.jsonl   per-agent event streams, mirrored after the snapshot

`push` takes the snapshot first and mirrors the event files second, so a restored event file is
never shorter than the restored index by more than what the index never saw; `Store.repair_events`
trims the rest after a restore. A push is skipped when the store has not changed since the last
one. Anything written after the last push is lost on a hard kill: the loss window is the sync
interval (60 s by default), and a SIGTERM triggers a final push.

A snapshot with no swarms is never restored: once the last swarm is removed, the next push
writes an empty snapshot, so a later controller in the same group starts clean.
`scripts/deploy-job.sh --down` also deletes the whole prefix (`clear`).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
from pathlib import Path

from flock.controller.store import DB_NAME, Store
from flock.worker.sync import BucketStore

log = logging.getLogger("flock.state")

STATE_DB = "state.db"
EVENTS = "events"
DEFAULT_INTERVAL = 60.0


class StateSync:
    def __init__(self, bucket: BucketStore, data_dir: str | Path, *, events: bool = True) -> None:
        self.bucket = bucket
        self.data_dir = Path(data_dir)
        self.events = events
        self._pushed_at_change: int | None = None  # store.changes() at the last push
        self._lock = threading.Lock()  # the periodic push and the final push never overlap
        self.pushes = 0
        self._event_versions: dict[str, tuple[int, int]] | None = None

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
            self.bucket.sync_down(EVENTS, self.data_dir / EVENTS)
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

    def push(self, store: Store, *, force: bool = False) -> int:
        """Snapshot the DB and mirror the event files. Blocking; run it in a thread. Returns
        the number of files pushed (the DB plus the event files), 0 when nothing changed."""
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
                versions = {}
                paths = {}
                for path in events_dir.rglob("*"):
                    if path.is_file():
                        stat = path.stat()
                        rel = f"{EVENTS}/{path.relative_to(events_dir).as_posix()}"
                        versions[rel] = (stat.st_size, stat.st_mtime_ns)
                        paths[rel] = path
                if self._event_versions is None:
                    self.bucket.sync_up(events_dir, EVENTS)
                    files += len(paths)
                else:
                    changed = [(paths[rel], rel) for rel, version in versions.items()
                               if self._event_versions.get(rel) != version]
                    if changed:
                        self.bucket.upload(changed)
                        files += len(changed)
                self._event_versions = versions
            self._pushed_at_change = changes
            self.pushes += 1
            return files


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
