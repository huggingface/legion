"""Controller state in a bucket, for deployments whose disk does not survive a restart (job mode,
docs/deploy-job.md). Layout under the state URL (`hf://buckets/<ns>/<bucket>/controller/<group>`):

    state.db             SQLite snapshot (online backup API: consistent while the app writes)
    events/<swarm>/<agent>.jsonl   per-agent event streams, mirrored after the snapshot

`push` takes the snapshot first and mirrors the event files second, so a restored event file is
never shorter than the restored index by more than what the index never saw; `Store.repair_events`
trims the rest after a restore. A push is skipped when the store has not changed since the last
one. Anything written after the last push is lost on a hard kill: the loss window is the sync
interval (60 s by default), and a SIGTERM triggers a final push.
"""

from __future__ import annotations

import logging
import os
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
        self.pushes = 0

    def restore(self) -> bool:
        """Before the Store opens: replace the local DB (and event files) with the bucket's copy.
        Returns False when the bucket has no snapshot (a fresh controller)."""
        if STATE_DB not in self.bucket.list(STATE_DB):
            log.info("state: no snapshot in the bucket; starting empty")
            return False
        self.data_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.data_dir / (STATE_DB + ".restore")
        self.bucket.download([(STATE_DB, tmp)])
        db = self.data_dir / DB_NAME
        for suffix in ("-wal", "-shm"):  # a stale WAL must not be replayed onto the snapshot
            Path(str(db) + suffix).unlink(missing_ok=True)
        os.replace(tmp, db)
        if self.events:
            self.bucket.sync_down(EVENTS, self.data_dir / EVENTS)
        log.info("state: restored %s from the bucket", STATE_DB)
        return True

    def push(self, store: Store, *, force: bool = False) -> bool:
        """Snapshot the DB and mirror the event files. Blocking; run it in a thread."""
        changes = store.changes()
        if not force and changes == self._pushed_at_change:
            return False
        snap = self.data_dir / (STATE_DB + ".snapshot")
        store.backup_to(snap)
        try:
            self.bucket.upload([(snap, STATE_DB)])
        finally:
            snap.unlink(missing_ok=True)
        events_dir = self.data_dir / EVENTS
        if self.events and events_dir.is_dir():
            self.bucket.sync_up(events_dir, EVENTS)
        self._pushed_at_change = changes
        self.pushes += 1
        return True


def state_sync_from_env(env, data_dir: str | Path) -> StateSync | None:
    """FLOCK_STATE_URL (hf://buckets/<ns>/<bucket>/<prefix>) turns state sync on;
    FLOCK_STATE_EVENTS=0 keeps event files local."""
    url = env.get("FLOCK_STATE_URL")
    if not url:
        return None
    from flock.worker.sync import HubBucketStore

    bucket = HubBucketStore(url, token=env.get("HF_TOKEN") or None)
    return StateSync(bucket, data_dir, events=env.get("FLOCK_STATE_EVENTS", "1") != "0")
