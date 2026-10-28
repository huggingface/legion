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
        pre + "sessions/researcher-0001.state.json", pre + "harness/s-1.jsonl",
        pre + "meta.json"}
    assert json.loads(store.files[pre + "meta.json"])["files"] == 1
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
    """Download fails for any batch holding a path in `bad` (xet 404 on reconstruction, as in
    production: permanent); `transient` paths fail their first `fails` attempts only, with a
    retryable error."""

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
            if r in self.bad:
                raise RuntimeError("File reconstruction error: CAS Client Error: 404 Not Found")
            if r in self.transient and sum(r in c for c in self.calls) <= self.fails:
                raise RuntimeError("CAS Client Error: 503 Service Unavailable")
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


async def test_restore_reports_progress_while_in_flight(tmp_path):
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    sync = BucketSync(store, "sw")
    seen = []
    plain = store.download

    def download(files):
        seen.append(list(sync.restore_progress[pre][:2]))
        plain(files)

    store.download = download
    rt2, _ = runtime(tmp_path / "b", sync=sync)
    await sync.restore(rt2)
    # 3 workdir files, then meta.json, 2 session files and 1 harness file in a second batch
    assert seen == [[0, 7], [3, 7]] and sync.restore_progress == {}


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
    # the batch, then the file alone; a 404 is permanent: no retry rounds
    assert sum(pre + "workdir/f1.txt" in c for c in store.calls) == 2
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


async def test_push_after_partial_restore_never_deletes_remote_files(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.bad = {pre + "workdir/f1.txt"}
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    await rt2.setup()
    assert not (rt2.workdir / "f1.txt").exists() and not rt2.sync.mirror_deletes(rt2)
    (rt2.workdir / "f0.txt").unlink()  # even a real local delete is not mirrored now
    (rt2.workdir / "new.txt").write_text("n")
    await rt2.sync_now("periodic")
    assert store.files[pre + "workdir/f1.txt"] == b"data1"  # not restored, still in the bucket
    assert store.files[pre + "workdir/f0.txt"] == b"data0"
    assert store.files[pre + "workdir/new.txt"] == b"n"  # changes still go up


async def test_push_after_failed_listing_never_deletes_remote_files(tmp_path):
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.list_error = RuntimeError("500 listing")
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    await rt2.setup()  # fresh start
    store.list_error = None
    rt2.workdir.mkdir(parents=True, exist_ok=True)
    (rt2.workdir / "x.txt").write_text("x")
    await rt2.sync_now("periodic")
    assert {pre + f"workdir/f{i}.txt" for i in range(3)} <= set(store.files)
    assert pre + "workdir/x.txt" in store.files


async def test_flock_part_files_are_never_pushed_and_restore_cleans_them(tmp_path):
    store = MemoryStore()
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"))
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    (rt.workdir / "f.txt").write_text("f")
    (rt.workdir / "sub").mkdir()
    (rt.workdir / "f.txt.flock-part").write_text("")
    (rt.workdir / "sub" / "g.bin.flock-part").write_text("half")
    await rt.sync_now("test")
    assert not any(k.endswith(".flock-part") for k in store.files)
    assert "sw/researcher/researcher-0001/workdir/f.txt" in store.files
    # a remote .flock-part (pushed by an older worker) is not deleted by the mirror either
    store.files["sw/researcher/researcher-0001/workdir/old.flock-part"] = b""
    await rt.sync.push(rt)
    assert "sw/researcher/researcher-0001/workdir/old.flock-part" in store.files
    await rt.stop()
    del store.files["sw/researcher/researcher-0001/workdir/old.flock-part"]

    # a crash left temporaries on the next machine's disk: restore removes them
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    (rt2.workdir / "sub").mkdir(parents=True)
    (rt2.workdir / "sub" / "stray.flock-part").write_text("x")
    await rt2.sync.restore(rt2)
    assert not list(rt2.workdir.rglob("*.flock-part"))
    assert (rt2.workdir / "f.txt").read_text() == "f"


async def test_first_push_after_full_restore_uploads_nothing_unchanged(tmp_path):
    store = MemoryStore()
    pre = await _seeded(tmp_path, store, n_files=5)
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt2.sync.restore(rt2)
    assert res.failed == [] and res.workdir == 5 and rt2.sync.mirror_deletes(rt2)
    for i in range(5):  # the bucket's mtime, not the download time
        r = pre + f"workdir/f{i}.txt"
        assert (rt2.workdir / f"f{i}.txt").stat().st_mtime == pytest.approx(store.mtimes[r], abs=1e-3)
    uploads = len(store.uploads)
    assert await rt2.sync.push(rt2) == 0  # sessions/harness: marked as pushed by the restore
    assert store.synced[-1] == [] and len(store.uploads) == uploads
    # after a complete restore the mirror deletes again, and real changes go up
    (rt2.workdir / "f0.txt").unlink()
    (rt2.workdir / "f1.txt").write_text("changed!")
    await rt2.sync.push(rt2)
    assert store.synced[-1] == [pre + "workdir/f1.txt"]
    assert pre + "workdir/f0.txt" not in store.files
    meta = json.loads(store.files[pre + "meta.json"])  # one file fewer, 3 bytes more
    assert (meta["files"], meta["bytes"]) == (4, 4 * 5 + 3) and rt2.workdir_files == 4


def test_not_found_is_skipped_at_once_other_errors_are_retried(tmp_path, caplog):
    from flock.worker.sync import download_resilient

    class Store(MemoryStore):
        def __init__(self):
            super().__init__()
            self.calls: list[list[str]] = []

        def download(self, files):
            self.calls.append([r for r, _ in files])
            for r, _ in files:
                if r.startswith("gone"):
                    raise RuntimeError("File reconstruction error: CAS Client Error: "
                                       "Request failed: 404 Not Found")
                if r == "missing":
                    raise FileNotFoundError("missing not found in bucket")
                if r == "flaky":
                    raise RuntimeError("CAS Client Error: 503 Service Unavailable")
            super().download(files)

    store = Store()
    names = [f"gone{i}" for i in range(30)] + ["missing", "flaky", "ok"]
    store.files = {n: b"x" for n in names}
    slept: list[float] = []
    files = [(n, tmp_path / n) for n in names]
    with caplog.at_level("INFO", logger="flock.worker.sync"):
        done, failed = download_resilient(store, files, backoff=(1.0, 4.0, 10.0), sleep=slept.append)
    assert [r for r, _ in done] == ["ok"]
    assert sorted(failed) == sorted(n for n in names if n != "ok")
    count = lambda n: sum(n in c for c in store.calls)
    assert count("gone0") == 2 and count("missing") == 2  # batch + solo, no retry rounds
    assert count("flaky") == 5 and slept == [1.0, 4.0, 10.0]  # retried with backoff
    assert not [r for r in caplog.records if r.levelname == "WARNING"]  # no line per file


async def test_restore_logs_skipped_files_as_one_summary_line(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("flock.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store, n_files=15)
    store.bad = {pre + f"workdir/f{i}.txt" for i in range(12)}
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    caplog.clear()
    with caplog.at_level("INFO", logger="flock.worker.sync"):
        res = await rt2.sync.restore(rt2)
    assert len(res.failed) == 12
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "12 file(s) skipped" in warnings[0]
    assert warnings[0].count(pre + "workdir/") == 10


def test_restore_batch_size_from_env(monkeypatch):
    from flock.worker.sync import restore_batch
    monkeypatch.delenv("FLOCK_RESTORE_BATCH", raising=False)
    assert restore_batch() == 1000
    for raw, want in (("7", 7), ("1", 1), ("0", 1000), ("-3", 1000), ("abc", 1000), ("", 1000)):
        monkeypatch.setenv("FLOCK_RESTORE_BATCH", raw)
        assert restore_batch() == want


def test_failed_big_batch_falls_back_per_file(tmp_path, monkeypatch):
    from flock.worker.sync import download_resilient
    monkeypatch.setenv("FLOCK_RESTORE_BATCH", "4")
    store = FlakyStore(bad={"r/f2"})
    for i in range(6):
        store.files[f"r/f{i}"] = b"x"
    files = [(f"r/f{i}", tmp_path / f"f{i}") for i in range(6)]
    got = []
    done, failed = download_resilient(store, files, backoff=(), sleep=lambda s: None,
                                      progress=got.append)
    assert failed == ["r/f2"] and len(done) == 5 and sum(got) == 5
    assert [len(c) for c in store.calls[:2]] == [4, 2]  # batches of 4 from the env
    assert ["r/f2"] in store.calls  # the failed batch retried file by file


def test_default_workdir_excludes_match_at_any_depth(monkeypatch):
    from flock.worker.sync import is_excluded, workdir_excludes
    monkeypatch.delenv("FLOCK_SYNC_EXCLUDE", raising=False)
    monkeypatch.delenv("FLOCK_SYNC_EXCLUDE_DEFAULTS", raising=False)
    pats = workdir_excludes()
    for rel in ["__pycache__/a.cpython-311.pyc", "a/b/__pycache__/x.txt", "m.pyc", "a/b/m.pyc",
                ".pytest_cache/v/x", "p/.pytest_cache/v/x", ".mypy_cache/x", "p/q/.mypy_cache/x",
                ".ruff_cache/x", "p/.ruff_cache/x", ".hypothesis/x", "p/.hypothesis/d/x",
                "pkg.egg-info/PKG-INFO", "src/pkg.egg-info/PKG-INFO", "f.txt.flock-part", ".cache/pip/x",
                "web/node_modules/a/b.js", ".venv/bin/python", "p/venv/lib/x", ".tox/py/x",
                ".nox/s/x", "n/.ipynb_checkpoints/a.ipynb", ".gradle/x", "app/.next/x", "a/b.o"]:
        assert is_excluded(rel, pats), rel
    for rel in ["src/main.py", "my__pycache__/x", ".git/config", "wt/.git", "wt/.git/HEAD",
                "wt/src/a.py", "pycache/x", "notes.pyc.md", "egg-info/x", "build/x", "dist/x",
                "target/x", "a.ox", "myvenv/x"]:
        assert not is_excluded(rel, pats), rel


def test_workdir_excludes_env_extends_and_disables(monkeypatch):
    from flock.worker.sync import is_excluded, workdir_excludes
    monkeypatch.setenv("FLOCK_SYNC_EXCLUDE", "*.tmp, node_modules/")
    pats = workdir_excludes()
    assert is_excluded("a/b.tmp", pats) and is_excluded("web/node_modules/x/y.js", pats)
    assert is_excluded("a/__pycache__/x", pats)
    monkeypatch.setenv("FLOCK_SYNC_EXCLUDE_DEFAULTS", "0")
    pats = workdir_excludes()
    assert not is_excluded("a/__pycache__/x", pats) and not is_excluded("m.pyc", pats)
    assert is_excluded("a/b.tmp", pats) and is_excluded("f.flock-part", pats)


async def test_mirror_skips_caches_and_never_deletes_remote_ones(tmp_path):
    store = MemoryStore()
    pre = "sw/researcher/researcher-0001/workdir/"
    store.files[pre + "pkg/__pycache__/old.pyc"] = b"remote"
    store.files[pre + "gone.txt"] = b"stale"
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"))
    await rt.setup(restore=False)
    store.files[pre + "pkg/__pycache__/old.pyc"] = b"remote"
    store.files[pre + "gone.txt"] = b"stale"
    rt.workdir.mkdir(parents=True, exist_ok=True)
    (rt.workdir / "pkg" / "__pycache__").mkdir(parents=True)
    (rt.workdir / "pkg" / "__pycache__" / "new.pyc").write_text("c")
    (rt.workdir / "pkg" / "a.py").write_text("a")
    (rt.workdir / "wt").mkdir()
    (rt.workdir / "wt" / ".git").write_text("gitdir: x")
    await rt.sync.push(rt)
    assert pre + "pkg/a.py" in store.files and pre + "wt/.git" in store.files
    assert pre + "pkg/__pycache__/new.pyc" not in store.files  # not uploaded
    assert store.files[pre + "pkg/__pycache__/old.pyc"] == b"remote"  # not deleted
    assert pre + "gone.txt" not in store.files  # the mirror still deletes ordinary files
    await rt.stop()


async def test_restore_skips_excluded_remote_files(tmp_path):
    store = MemoryStore()
    pre = "sw/researcher/researcher-0001/workdir/"
    for k in ["keep.txt", "a/b/__pycache__/x.pyc", "m.pyc", ".mypy_cache/x", "wt/.git"]:
        store.files[pre + k] = b"x"
    rt, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt.sync.restore(rt)
    assert res.workdir == 2 and not res.failed
    assert sorted(p.relative_to(rt.workdir).as_posix() for p in rt.workdir.rglob("*")
                  if p.is_file()) == ["keep.txt", "wt/.git"]
