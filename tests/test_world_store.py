"""flock.world.store: dirty-row flushes, atomic commits, cold rows, backup under writes."""
from __future__ import annotations

import sqlite3
import threading

import pytest

from flock.world.store import Store


def test_flush_writes_only_dirty_rows_and_deletes_gone_ones(tmp_path):
    store = Store(tmp_path / "s.db")
    state = {"1": {"a": 1}, "2": {"b": [1, 2]}}
    store.write([("prs", k, v) for k, v in state.items()])
    state["1"]["a"] = 5
    state["3"] = {"c": None}
    del state["2"]
    state["4"] = {"untouched": True}  # never marked: not written
    for key in ("1", "2", "3"):
        store.dirty("prs", key)
    assert store.pending == 3
    assert store.flush(lambda kind, key: state.get(key)) == 3
    assert store.pending == 0 and store.flush(lambda kind, key: 1 / 0) == 0
    store.close()
    again = Store(tmp_path / "s.db")
    assert dict(again.iter("prs")) == {"1": {"a": 5}, "3": {"c": None}}
    assert again.get("prs", "2") is None and again.get("other", "1") is None


def test_cold_rows_leave_the_hot_set_but_stay_readable(tmp_path):
    store = Store(tmp_path / "s.db")
    store.put("prs", "1", {"state": "open"})
    store.dirty("prs", "2")
    store.put("prs", "2", {"state": "merged"}, cold=True)
    assert store.pending == 0  # put wrote its newest value
    assert dict(store.iter("prs")) == {"1": {"state": "open"}}
    assert dict(store.iter("prs", cold=True)) == {"2": {"state": "merged"}}
    assert store.get("prs", "2") == {"state": "merged"}
    store.delete("prs", "2")
    assert store.get("prs", "2") is None


class FailingDb:
    """A connection whose nth execute raises, as a saver killed mid-transaction."""

    def __init__(self, db, fail_at):
        self.db, self.n, self.fail_at = db, 0, fail_at

    def execute(self, *args):
        self.n += 1
        if self.n == self.fail_at:
            raise sqlite3.OperationalError("disk I/O error (injected)")
        return self.db.execute(*args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_a_flush_cut_mid_transaction_writes_nothing_and_retries(tmp_path):
    store = Store(tmp_path / "s.db")
    store.write([("k", str(i), i) for i in range(5)])
    state = {str(i): i * 10 for i in range(5)}
    for key in state:
        store.dirty("k", key)
    real = store._db
    store._db = FailingDb(real, fail_at=4)  # BEGIN, two upserts, then the third fails
    with pytest.raises(sqlite3.OperationalError):
        store.flush(lambda kind, key: state[key])
    store._db = real
    reader = Store(tmp_path / "s.db")
    assert dict(reader.iter("k")) == {str(i): i for i in range(5)}  # all or nothing
    assert store.pending == 5
    store.flush(lambda kind, key: state[key])
    assert dict(reader.iter("k")) == state


def test_backup_while_50_threads_write(tmp_path):
    store = Store(tmp_path / "s.db")
    stop = threading.Event()

    def writer(n):
        i = 0
        while not stop.is_set():
            store.write([("t", f"{n}", {"i": i}), ("u", f"{n}", {"i": i})])  # one transaction
            i += 1

    threads = [threading.Thread(target=writer, args=(n,)) for n in range(50)]
    for t in threads:
        t.start()
    try:
        for b in range(5):
            store.backup_to(tmp_path / f"copy{b}.db")
            copy = Store(tmp_path / f"copy{b}.db")
            # Each writer's two rows commit together: a consistent copy has them equal.
            assert dict(copy.iter("t")) == dict(copy.iter("u"))
            copy.close()
    finally:
        stop.set()
        for t in threads:
            t.join()
    assert len(dict(store.iter("t"))) == 50
