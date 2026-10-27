"""BucketSync: layout under <swarm>/<type>/<agent>/, incremental push, restore to the same paths."""

from __future__ import annotations

import json
import os

import pytest

from flock.worker.sync import MODES_FILE, BucketSync, apply_modes, collect_modes, parse_bucket

from .worker_fakes import MemoryStore, runtime, until


def test_parse_bucket():
    assert parse_bucket("hf://buckets/cmpatino/flock") == ("cmpatino/flock", "")
    assert parse_bucket("hf://buckets/ns/b/runs/x/") == ("ns/b", "runs/x/")


async def test_push_and_restore_round_trip(tmp_path):
    store = MemoryStore()
    sync = BucketSync(store, "sw")
    rt, _ = runtime(tmp_path / "a", sync=sync)
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    (rt.workdir / "sub").mkdir(parents=True, exist_ok=True)
    (rt.workdir / "sub" / "f.txt").write_text("data")
    await rt.sync_now("test")
    pre = "sw/researcher/researcher-0001/"
    assert set(store.files) == {
        pre + "workdir/sub/f.txt", pre + "sessions/researcher-0001.jsonl",
        pre + "sessions/researcher-0001.state.json", pre + "harness/s-1.jsonl"}
    # second push uploads only what changed (the session log gained a sync event)
    n = await sync.push(rt)
    assert n == 0 or store.uploads[-1] == [pre + "sessions/researcher-0001.jsonl"]
    # the bucket mirrors workdir deletions
    (rt.workdir / "sub" / "f.txt").unlink()
    (rt.workdir / "g.txt").write_text("g")
    await sync.push(rt)
    assert pre + "workdir/sub/f.txt" not in store.files and pre + "workdir/g.txt" in store.files
    await rt.stop()

    # a fresh machine: same paths, empty disk
    rt2, adapter2 = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt2.sync.restore(rt2)
    assert (res.workdir, res.sessions, res.harness) == (1, 2, 1)
    assert (rt2.workdir / "g.txt").read_text() == "g"
    assert (adapter2.home / "s-1.jsonl").exists()
    rt2.load_state()
    assert rt2.state.session_id == "s-1" and rt2.state.turns == 1


async def test_persist_workdir_false_skips_workdir(tmp_path):
    store = MemoryStore()
    rt, _ = runtime(tmp_path, sync=BucketSync(store, "sw"), persist_workdir=False)
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    (rt.workdir / "big.bin").write_text("x")
    await rt.sync_now("test")
    assert not any("/workdir/" in k for k in store.files)


def test_http_route_names_bucket_calls():
    from flock.worker.sync import http_route

    assert http_route("GET", "huggingface.co", "/api/buckets/ns/b/tree/sw/x") == "GET buckets/tree"
    assert http_route("POST", "huggingface.co", "/api/buckets/ns/b/batch") == "POST buckets/batch"
    assert http_route("GET", "huggingface.co", "/api/buckets/ns/b") == "GET buckets"
    assert http_route("GET", "cas-server.xethub.hf.co", "/v1/reconstructions/abc") == (
        "GET cas-server.xethub.hf.co/v1")


async def test_sync_counts_push_and_restore_failures(tmp_path):
    class Broken(MemoryStore):
        def upload(self, files):
            raise RuntimeError("429")

        def list(self, prefix):
            raise RuntimeError("429")

    sync = BucketSync(Broken(), "sw")
    rt, _ = runtime(tmp_path, sync=sync)
    await rt.setup()  # restore fails: fresh start, counted
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    await rt.sync_now("test")  # push fails: logged, counted, agent stays dirty
    assert sync.stats.snapshot()["errors"] == {"restore_failed": 1, "push_failed": 1}
    assert rt.dirty
    await rt.stop()


async def test_file_modes_survive_a_bucket_round_trip(tmp_path):
    """The bucket keeps bytes only (MemoryStore, like the Hub, writes 0644 on download); the
    manifest pushed with the workdir brings executable bits back on restore."""
    old_umask = os.umask(0o022)  # restored files get 0644/0755 whatever the runner's umask
    try:
        await _file_modes_round_trip(tmp_path)
    finally:
        os.umask(old_umask)


async def _file_modes_round_trip(tmp_path):
    store = MemoryStore()
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"))
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    (rt.workdir / "canvas").write_text("#!/bin/sh\necho ok\n")
    os.chmod(rt.workdir / "canvas", 0o755)
    (rt.workdir / "bin").mkdir()
    (rt.workdir / "bin" / "tool").write_text("x")
    os.chmod(rt.workdir / "bin" / "tool", 0o700)
    (rt.workdir / "data.txt").write_text("d")
    os.chmod(rt.workdir / "data.txt", 0o644)
    await rt.sync_now("test")
    pre = "sw/researcher/researcher-0001/workdir/"
    assert json.loads(store.files[pre + MODES_FILE]) == {"bin/tool": "0o700", "canvas": "0o755"}
    mtime = (rt.workdir / MODES_FILE).stat().st_mtime_ns
    await rt.sync_now("again")  # unchanged modes: the manifest is not rewritten
    assert (rt.workdir / MODES_FILE).stat().st_mtime_ns == mtime
    await rt.stop()

    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    await rt2.sync.restore(rt2)
    mode = lambda p: (rt2.workdir / p).stat().st_mode & 0o777
    assert (mode("canvas"), mode("bin/tool"), mode("data.txt")) == (0o755, 0o700, 0o644)

    # nothing executable any more: the manifest goes away (and from the bucket)
    for f in ("canvas", "bin/tool"):
        os.chmod(rt2.workdir / f, 0o644)
    await rt2.sync.push(rt2)
    assert not (rt2.workdir / MODES_FILE).exists() and pre + MODES_FILE not in store.files


def test_apply_modes_ignores_paths_outside_the_workdir(tmp_path):
    wd = tmp_path / "wd"
    wd.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("s")
    os.chmod(outside, 0o600)
    (wd / "link").symlink_to(outside)
    (wd / "ok").write_text("x")
    (wd / MODES_FILE).write_text(json.dumps({"../secret": "0o777", "link": "0o777", "ok": "0o750",
                                             "missing": "0o755", "bad": "nope"}))
    assert apply_modes(wd) == 1
    assert outside.stat().st_mode & 0o777 == 0o600 and (wd / "ok").stat().st_mode & 0o777 == 0o750
    assert collect_modes(wd) == {"ok": "0o750"}  # the symlink is not followed
    (wd / MODES_FILE).write_text("not json")
    assert apply_modes(wd) == 0


class FlakyStore(MemoryStore):
    """Download fails for any batch holding a path in `bad`; `transient` paths fail their
    first `fails` attempts only (xet 404 on reconstruction, as in production)."""

    def __init__(self, bad=(), transient=(), fails=1):
        super().__init__()
        self.bad, self.transient, self.fails = set(bad), set(transient), fails
        self.calls: list[list[str]] = []
        self.list_error: Exception | None = None

    def list(self, prefix):
        if self.list_error is not None:
            raise self.list_error
        return super().list(prefix)

    def download(self, files):
        self.calls.append([r for r, _ in files])
        for r, _ in files:
            if r in self.bad or (r in self.transient and
                                 sum(r in c for c in self.calls) <= self.fails):
                raise RuntimeError("File reconstruction error: CAS Client Error: 404 Not Found")
        super().download(files)


async def _seeded(tmp_path, store, n_files=3):
    """An agent with one turn, `n_files` workdir files and its session pushed to `store`."""
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"))
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    for i in range(n_files):
        (rt.workdir / f"f{i}.txt").write_text(f"data{i}")
    await rt.sync_now("test")
    await rt.stop()
    return "sw/researcher/researcher-0001/"


async def test_restore_skips_an_unreadable_file_and_restores_the_rest(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.bad = {pre + "workdir/f1.txt"}
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt2.sync.restore(rt2)  # no exception
    assert res.failed == [pre + "workdir/f1.txt"]
    assert (res.workdir, res.sessions, res.harness) == (2, 2, 1)
    assert (rt2.workdir / "f0.txt").read_text() == "data0"
    assert (rt2.workdir / "f2.txt").read_text() == "data2"
    assert not (rt2.workdir / "f1.txt").exists()
    assert not list(rt2.workdir.rglob("*.flock-part"))  # no half-written leftovers
    # the batch, then the file alone, then 3 retries
    assert sum(pre + "workdir/f1.txt" in c for c in store.calls) == 5
    assert rt2.sync.stats.snapshot()["errors"] == {"restore_file_failed": 1}


async def test_setup_emits_restore_partial(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.bad = {pre + "workdir/f1.txt"}
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    await rt2.setup()
    ev = next(e for e in rt2.outbox._q if e.type == "restore_partial")
    assert ev.data["data"] == {"failed": 1, "paths": [pre + "workdir/f1.txt"]}
    assert rt2.state.session_id == "s-1"  # the rest of the state came back
    assert not any(e.type == "error" for e in rt2.outbox._q)


async def test_restore_retries_a_transient_failure(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("flock.worker.sync.time.sleep", slept.append)
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.transient, store.fails = {pre + "workdir/f2.txt"}, 2  # batch and solo attempt fail
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt2.sync.restore(rt2)
    assert res.failed == [] and res.workdir == 3
    assert (rt2.workdir / "f2.txt").read_text() == "data2"
    assert slept == [1.0]  # first backoff round succeeded


async def test_failed_session_file_still_starts_with_session_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.bad = {pre + "harness/s-1.jsonl"}  # the harness session transcript
    rt2, adapter2 = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    await rt2.setup()
    assert (rt2.workdir / "f0.txt").read_text() == "data0"  # workdir restored
    assert rt2.state.session_id == "s-1"
    await rt2.prompt("hello")
    await until(lambda: len(adapter2.sent) == 2)
    assert adapter2.starts == ["s-1", None]
    assert "could not be resumed" in adapter2.sent[0]
    restarted = next(e for e in rt2.outbox._q if e.type == "session_restarted")
    assert restarted.data["data"]["previous_session_id"] == "s-1"
    partial = next(e for e in rt2.outbox._q if e.type == "restore_partial")
    assert partial.data["data"]["paths"] == [pre + "harness/s-1.jsonl"]
    await rt2.stop()


async def test_restore_listing_failure_still_raises(tmp_path):
    store = FlakyStore()
    await _seeded(tmp_path, store)
    store.list_error = RuntimeError("500 listing")
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    with pytest.raises(RuntimeError, match="500 listing"):
        await rt2.sync.restore(rt2)
    assert rt2.sync.stats.snapshot()["errors"] == {"restore_failed": 1}
