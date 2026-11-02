"""`flock.world.store`: a small `(kind, key) -> JSON` table on SQLite in WAL mode (docs/world.md).

For worlds that keep their state in memory and want saves that cost O(changes): the world reads
its own dicts, marks the rows it changed with `dirty(kind, key)`, and a saver calls
`flush(read)`, which upserts (or deletes) only those rows in one transaction. `backup_to(path)`
writes a consistent copy while others write (SQLite online backup API), for archives.

Rows moved out of the hot set (`put(..., cold=True)`) stay readable with `get` and
`iter(kind, cold=True)`; `iter(kind)` (the hot rows) is what a world loads at start.

    store = Store(ws / "state.db")
    state = {key: value for key, value in store.iter("prs")}
    state["7"] = {...}; store.dirty("prs", "7")
    store.flush(lambda kind, key: state.get(key))   # None deletes the row
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS rows (
    kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
    cold INTEGER NOT NULL DEFAULT 0, ts REAL NOT NULL,
    PRIMARY KEY (kind, key));
"""
UPSERT = ("INSERT INTO rows (kind, key, value, cold, ts) VALUES (?, ?, ?, ?, ?) "
          "ON CONFLICT (kind, key) DO UPDATE SET value = excluded.value, cold = excluded.cold, "
          "ts = excluded.ts")


class Store:
    """Thread-safe: one connection behind a lock (writes are short: dirty rows only)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        # (kind, key) -> generation of its newest change; _written: generation last committed,
        # so a flush that lost a race never commits an older value over a newer one.
        self._dirty: dict[tuple[str, str], int] = {}
        self._written: dict[tuple[str, str], int] = {}
        self._gen = 0
        self._flushing = 0  # flushes between their read and their commit
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")  # WAL: durable at checkpoints, never corrupt
        self._db.execute("PRAGMA journal_size_limit=4194304")  # truncate the WAL after big writes
        self._db.executescript(SCHEMA)

    def get(self, kind: str, key: str) -> Any:
        """The row's value (hot or cold), or None."""
        with self._lock:
            row = self._db.execute("SELECT value FROM rows WHERE kind = ? AND key = ?",
                                   (kind, key)).fetchone()
        return None if row is None else json.loads(row[0])

    def iter(self, kind: str, cold: bool = False) -> Iterator[tuple[str, Any]]:
        """(key, value) of the hot rows of `kind` (cold=True: the cold rows), in first-write
        order."""
        with self._lock:
            rows = self._db.execute("SELECT key, value FROM rows WHERE kind = ? AND cold = ? "
                                    "ORDER BY rowid", (kind, int(cold))).fetchall()
        for key, value in rows:
            yield key, json.loads(value)

    def put(self, kind: str, key: str, value: Any, cold: bool = False) -> None:
        """Write one row now (and drop its dirty mark: this is its newest value)."""
        self.write([(kind, key, value)], cold=cold)

    def delete(self, kind: str, key: str) -> None:
        self.write([(kind, key, None)])

    def write(self, rows: list[tuple[str, str, Any]], cold: bool = False) -> None:
        """Upsert (value None: delete) several rows in one transaction."""
        with self._lock:
            self._gen += 1
            encoded = [(kind, key, self._gen, None if value is None else json.dumps(value))
                       for kind, key, value in rows]
            self._commit(encoded, cold)

    def dirty(self, kind: str, key: str) -> None:
        """Mark a row changed; the next flush writes it."""
        with self._lock:
            self._gen += 1
            self._dirty.pop((kind, key), None)  # re-insert: keeps the newest change last
            self._dirty[(kind, key)] = self._gen

    @property
    def pending(self) -> int:
        """Rows marked dirty and not yet flushed."""
        return len(self._dirty)

    def flush(self, read: Callable[[str, str], Any], lock=None) -> int:
        """Write every dirty row in one transaction: `read(kind, key)` gives its current value
        (None: delete it). `lock` guards what `read` reads: it is held only while the rows are
        read and encoded, not during the commit. A row changed again meanwhile stays dirty. On a
        failed commit nothing is written and the rows stay dirty. Returns the rows written."""
        with lock or contextlib.nullcontext(), self._lock:
            rows = []
            for (kind, key), gen in self._dirty.items():
                value = read(kind, key)
                rows.append((kind, key, gen, None if value is None else json.dumps(value)))
            if not rows:
                return 0
            self._flushing += 1
        with self._lock:
            try:
                self._commit(rows, False)
            finally:
                self._flushing -= 1
                if not self._flushing:  # no older read left in flight: nothing to guard against
                    self._written.clear()
        return len(rows)

    def _commit(self, rows, cold):
        """Write rows (kind, key, generation, encoded value or None) in one transaction,
        skipping any already written at a newer generation; then drop their dirty marks."""
        rows = [r for r in rows if self._written.get(r[:2], 0) < r[2]]
        now = time.time()
        self._db.execute("BEGIN IMMEDIATE")
        try:
            for kind, key, _, value in rows:
                if value is None:
                    self._db.execute("DELETE FROM rows WHERE kind = ? AND key = ?", (kind, key))
                else:
                    self._db.execute(UPSERT, (kind, key, value, int(cold), now))
            self._db.execute("COMMIT")
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        for kind, key, gen, _ in rows:
            self._written[(kind, key)] = gen
            if self._dirty.get((kind, key), gen + 1) <= gen:
                del self._dirty[(kind, key)]

    def backup_to(self, path: str | Path) -> None:
        """Consistent copy of the database, WAL included (SQLite online backup API). Holds the
        store's lock for the copy: writers wait."""
        backup_to(self._db, path, self._lock)

    def size(self) -> int:
        """Bytes on disk (database plus WAL)."""
        return sum(p.stat().st_size for p in (self.path, self.path.with_name(self.path.name + "-wal"))
                   if p.exists())

    def checkpoint(self) -> None:
        """Fold the WAL into the database file and truncate it."""
        with self._lock:
            self._db.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def close(self) -> None:
        self.checkpoint()
        with self._lock:
            self._db.close()


def backup_to(db: sqlite3.Connection | str | Path, path: str | Path, lock=None) -> None:
    """Copy database `db` (a connection, or a file another process may be writing) to `path`."""
    path = Path(path)
    path.unlink(missing_ok=True)
    src = db if isinstance(db, sqlite3.Connection) else sqlite3.connect(db)
    dst = sqlite3.connect(path)
    try:
        if lock is None:
            src.backup(dst)
        else:
            with lock:
                src.backup(dst)
    finally:
        dst.close()
        if src is not db:
            src.close()
