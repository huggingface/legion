"""BucketSync: layout under <swarm>/<type>/<agent>/, incremental push, restore to the same paths."""

from __future__ import annotations

import json
import os

import pytest

from legion.worker.sync import MODES_FILE, BucketSync, apply_modes, collect_modes, parse_bucket

from .worker_fakes import MemoryStore, runtime, until


def test_parse_bucket():
    assert parse_bucket("hf://buckets/ns/legion") == ("ns/legion", "")
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
    monkeypatch.setattr("legion.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
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
    assert not list(rt2.workdir.rglob("*.legion-part"))  # no half-written leftovers
    # the batch, then the file alone; a 404 is permanent: no retry rounds
    assert sum(pre + "workdir/f1.txt" in c for c in store.calls) == 2
    assert rt2.sync.stats.snapshot()["errors"] == {"restore_file_failed": 1}


async def test_restore_retries_a_transient_failure(tmp_path, monkeypatch):
    slept: list[float] = []
    monkeypatch.setattr("legion.worker.sync.time.sleep", slept.append)
    store = FlakyStore()
    pre = await _seeded(tmp_path, store)
    store.transient, store.fails = {pre + "workdir/f2.txt"}, 2  # batch and solo attempt fail
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    res = await rt2.sync.restore(rt2)
    assert res.failed == [] and res.workdir == 3
    assert (rt2.workdir / "f2.txt").read_text() == "data2"
    assert slept == [1.0]  # first backoff round succeeded


async def test_failed_session_file_still_starts_with_session_restarted(tmp_path, monkeypatch):
    monkeypatch.setattr("legion.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
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
    assert partial.data["data"] == {"failed": 1, "paths": [pre + "harness/s-1.jsonl"]}
    assert not any(e.type == "error" for e in rt2.outbox._q)
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
    monkeypatch.setattr("legion.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
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


async def test_legion_part_files_are_never_pushed_and_restore_cleans_them(tmp_path):
    store = MemoryStore()
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"))
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    (rt.workdir / "f.txt").write_text("f")
    (rt.workdir / "sub").mkdir()
    (rt.workdir / "f.txt.legion-part").write_text("")
    (rt.workdir / "sub" / "g.bin.legion-part").write_text("half")
    await rt.sync_now("test")
    assert not any(k.endswith(".legion-part") for k in store.files)
    assert "sw/researcher/researcher-0001/workdir/f.txt" in store.files
    # a .legion-part already in the bucket is not deleted by the mirror either
    store.files["sw/researcher/researcher-0001/workdir/old.legion-part"] = b""
    await rt.sync.push(rt)
    assert "sw/researcher/researcher-0001/workdir/old.legion-part" in store.files
    await rt.stop()
    del store.files["sw/researcher/researcher-0001/workdir/old.legion-part"]

    # a crash left temporaries on the next machine's disk: restore removes them
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    (rt2.workdir / "sub").mkdir(parents=True)
    (rt2.workdir / "sub" / "stray.legion-part").write_text("x")
    await rt2.sync.restore(rt2)
    assert not list(rt2.workdir.rglob("*.legion-part"))
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


def test_workdir_stats_skip_excluded_paths_and_rank_top_dirs(tmp_path):
    from legion.worker.sync import WorkdirStats, workdir_excludes
    wd = tmp_path / "wd"
    for rel, data in [("a.txt", "12"), ("snaps/1/x", "1"), ("snaps/2/y", "1"), ("snaps/z", "1"),
                      ("src/m.py", "123"), ("src/node_modules/p/i.js", "big" * 100),
                      ("src/m.o", "bin"), (".venv/bin/python", "b")]:
        (wd / rel).parent.mkdir(parents=True, exist_ok=True)
        (wd / rel).write_text(data)
    os.chmod(wd / "src/m.py", 0o755)
    (wd / ".venv/bin/python").chmod(0o755)
    ws = WorkdirStats()
    assert collect_modes(wd, ws, workdir_excludes()) == {"src/m.py": "0o755"}
    assert (ws.files, ws.bytes) == (5, 2 + 3 + 3)
    assert ws.top_dirs == [("snaps", 3), ("src", 1)]


def test_not_found_is_skipped_at_once_other_errors_are_retried(tmp_path, caplog):
    from legion.worker.sync import download_resilient

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
    with caplog.at_level("INFO", logger="legion.worker.sync"):
        done, failed = download_resilient(store, files, backoff=(1.0, 4.0, 10.0), sleep=slept.append)
    assert [r for r, _ in done] == ["ok"]
    assert sorted(failed) == sorted(n for n in names if n != "ok")
    count = lambda n: sum(n in c for c in store.calls)
    assert count("gone0") == 2 and count("missing") == 2  # batch + solo, no retry rounds
    assert count("flaky") == 5 and slept == [1.0, 4.0, 10.0]  # retried with backoff
    assert not [r for r in caplog.records if r.levelname == "WARNING"]  # no line per file


async def test_restore_logs_skipped_files_as_one_summary_line(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr("legion.worker.sync.RESTORE_BACKOFF", (0, 0, 0))
    store = FlakyStore()
    pre = await _seeded(tmp_path, store, n_files=15)
    store.bad = {pre + f"workdir/f{i}.txt" for i in range(12)}
    rt2, _ = runtime(tmp_path / "b", sync=BucketSync(store, "sw"))
    caplog.clear()
    with caplog.at_level("INFO", logger="legion.worker.sync"):
        res = await rt2.sync.restore(rt2)
    assert len(res.failed) == 12
    warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1 and "12 file(s) skipped" in warnings[0]
    assert warnings[0].count(pre + "workdir/") == 10


def test_restore_batch_size_from_env(monkeypatch):
    from legion.worker.sync import restore_batch
    monkeypatch.delenv("LEGION_RESTORE_BATCH", raising=False)
    assert restore_batch() == 1000
    for raw, want in (("7", 7), ("1", 1), ("0", 1000), ("-3", 1000), ("abc", 1000), ("", 1000)):
        monkeypatch.setenv("LEGION_RESTORE_BATCH", raw)
        assert restore_batch() == want


def test_failed_big_batch_falls_back_per_file(tmp_path, monkeypatch):
    from legion.worker.sync import download_resilient
    monkeypatch.setenv("LEGION_RESTORE_BATCH", "4")
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
    from legion.worker.sync import is_excluded, workdir_excludes
    monkeypatch.delenv("LEGION_SYNC_EXCLUDE", raising=False)
    monkeypatch.delenv("LEGION_SYNC_EXCLUDE_DEFAULTS", raising=False)
    pats = workdir_excludes()
    for rel in ["__pycache__/a.cpython-311.pyc", "a/b/__pycache__/x.txt", "m.pyc", "a/b/m.pyc",
                ".pytest_cache/v/x", "p/.pytest_cache/v/x", ".mypy_cache/x", "p/q/.mypy_cache/x",
                ".ruff_cache/x", "p/.ruff_cache/x", ".hypothesis/x", "p/.hypothesis/d/x",
                "pkg.egg-info/PKG-INFO", "src/pkg.egg-info/PKG-INFO", "f.txt.legion-part", ".cache/pip/x",
                "web/node_modules/a/b.js", ".venv/bin/python", "p/venv/lib/x", ".tox/py/x",
                ".nox/s/x", "n/.ipynb_checkpoints/a.ipynb", ".gradle/x", "app/.next/x", "a/b.o"]:
        assert is_excluded(rel, pats), rel
    for rel in ["src/main.py", "my__pycache__/x", ".git/config", "wt/.git", "wt/.git/HEAD",
                "wt/src/a.py", "pycache/x", "notes.pyc.md", "egg-info/x", "build/x", "dist/x",
                "target/x", "a.ox", "myvenv/x"]:
        assert not is_excluded(rel, pats), rel


def test_workdir_excludes_env_extends_and_disables(monkeypatch):
    from legion.worker.sync import is_excluded, workdir_excludes
    monkeypatch.setenv("LEGION_SYNC_EXCLUDE", "*.tmp, node_modules/")
    pats = workdir_excludes()
    assert is_excluded("a/b.tmp", pats) and is_excluded("web/node_modules/x/y.js", pats)
    assert is_excluded("a/__pycache__/x", pats)
    monkeypatch.setenv("LEGION_SYNC_EXCLUDE_DEFAULTS", "0")
    pats = workdir_excludes()
    assert not is_excluded("a/__pycache__/x", pats) and not is_excluded("m.pyc", pats)
    assert is_excluded("a/b.tmp", pats) and is_excluded("f.legion-part", pats)


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


async def test_over_soft_limit_notes_hourly_and_hard_limit_emits_event(tmp_path):
    from legion.worker.sync import QUOTA_NOTE_EVERY
    store = MemoryStore()
    sync = BucketSync(store, "sw")
    rt, _ = runtime(tmp_path, sync=sync, workdir={"soft_files": 2, "max_files": 5})
    await rt.setup(restore=False)
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    for rel in ["snaps/1/a", "snaps/2/b", "snaps/c", "src/d"]:
        (rt.workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (rt.workdir / rel).write_text("x")
    await sync.push(rt)
    assert list(rt._notes) == [("your workspace has 4 files, 0 MB; largest: snaps/ 3, src/ 1 "
                                "(soft limit 2 files, 2048 MB); delete clones you no longer need")]
    assert not over_quota(rt)  # 4 <= max_files
    assert pre_files(store) == 4  # never refused
    rt._notes.clear()
    for i in range(3):
        (rt.workdir / f"e{i}").write_text("x")
    await sync.push(rt)
    assert not rt._notes  # at most hourly
    sync._quota_at[sync.prefix(rt)] -= QUOTA_NOTE_EVERY
    await sync.push(rt)
    assert "7 files" in next(iter(rt._notes))
    ev = over_quota(rt)
    assert len(ev) == 1 and ev[0]["files"] == 7 and ev[0]["top_dirs"][0] == ["snaps", 3]
    assert pre_files(store) == 7
    await rt.stop()


async def test_quota_note_replaced_and_not_timestamped_when_dropped(tmp_path):
    sync = BucketSync(MemoryStore(), "sw")
    rt, _ = runtime(tmp_path, sync=sync, workdir={"soft_files": 2, "max_files": 5})
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    for i in range(4):
        (rt.workdir / f"f{i}").write_text("x")
    await sync.push(rt)  # no turn yet: the note is dropped, so nothing is timestamped
    assert not rt._notes and not sync._quota_at
    await rt.prompt("hi")
    await until(lambda: rt.state.turns == 1)
    await sync.push(rt)
    assert len(rt._notes) == 1 and sync._quota_at
    (rt.workdir / "g").write_text("x")
    sync._quota_at[sync.prefix(rt)] -= 3600.0
    await sync.push(rt)  # an unread note is replaced, not stacked
    assert len(rt._notes) == 1 and "5 files" in next(iter(rt._notes))
    await rt.stop()


def over_quota(rt):
    return [e.data["data"] for e in list(rt.outbox._q)
            if e.data.get("customType") == "workdir_over_quota"]


def pre_files(store):
    return sum(1 for k in store.files if "/workdir/" in k)


async def test_ephemeral_dir_is_never_synced_and_wiped_on_a_move(tmp_path):
    store = MemoryStore()
    eph = {"ephemeral": ["scratch"]}
    rt, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"), workdir=eph)
    await rt.setup(restore=False)
    assert (rt.workdir / "scratch").is_dir()  # created at setup
    (rt.workdir / "scratch/clone").mkdir()
    (rt.workdir / "scratch/clone/x").write_text("x")
    (rt.workdir / "keep.txt").write_text("k")
    await rt.sync_now("test")
    pre = "sw/researcher/researcher-0001/workdir/"
    assert pre + "keep.txt" in store.files and not [k for k in store.files if "scratch" in k]
    assert rt.workdir_stats.files == 1  # scratch/ is not counted either
    store.files[pre + "scratch/old"] = b"from before the setting"
    await rt.stop()
    # the next worker (same disk here: a stale local scratch/ too) restores everything else
    rt2, _ = runtime(tmp_path / "a", sync=BucketSync(store, "sw"), workdir=eph)
    await rt2.setup(restore=True)
    assert (rt2.workdir / "keep.txt").read_text() == "k"
    assert (rt2.workdir / "scratch").is_dir() and not any((rt2.workdir / "scratch").iterdir())


async def test_parallel_restore_gives_the_same_tree_as_sequential(tmp_path, monkeypatch):
    import threading
    import time

    from legion.worker import sync as sync_mod

    class SlowStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.active = self.peak = 0
            self.lock = threading.Lock()

        def download(self, files):
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            time.sleep(0.02)
            try:
                if any(r.endswith("/bad") for r, _ in files):
                    raise RuntimeError("503")
                super().download(files)
            finally:
                with self.lock:
                    self.active -= 1

    store = SlowStore()
    pre = await _seeded(tmp_path, store, n_files=40)
    store.files[pre + "workdir/d/bad"] = b"x"  # fails in its batch and alone: retries keep going
    trees = {}
    for threads in (1, 4):
        monkeypatch.setattr(sync_mod, "SPEC_KNOBS", {"restore_batch": 5, "restore_threads": threads})
        store.peak = 0
        rt, _ = runtime(tmp_path / f"t{threads}", sync=BucketSync(store, "sw"))
        sync = rt.sync
        monkeypatch.setattr(sync_mod, "RESTORE_BACKOFF", ())
        res = await sync.restore(rt)
        assert res.failed == [pre + "workdir/d/bad"] and res.workdir == 40
        assert store.peak == threads
        trees[threads] = {p.relative_to(rt.workdir).as_posix(): p.read_bytes()
                          for p in rt.workdir.rglob("*") if p.is_file()}
        assert not sync.mirror_deletes(rt)  # incomplete: pushes never delete
    assert trees[1] == trees[4] and len(trees[4]) == 40


# --- base tar plus per-file delta (workdir.base) -----------------------------------------------

BASE = {"workdir": {"base": True, "base_interval_s": 3600, "ephemeral": ["scratch"]}}
PRE = "sw/researcher/researcher-0001/"


def _tree(d):
    """{rel: ("link", target) | (bytes, mode, mtime)} of a workdir, symlinks not followed."""
    out = {}
    for root, dirs, files in os.walk(d):
        for f in files + [x for x in dirs if os.path.islink(os.path.join(root, x))]:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d)
            if os.path.islink(p):
                out[rel] = ("link", os.readlink(p))
            else:
                st = os.stat(p)
                with open(p, "rb") as f:
                    out[rel] = (f.read(), st.st_mode & 0o777, round(st.st_mtime, 2))
    return out


def _bytes(tree):
    """Regular files' bytes only: what the plain per-file mirror keeps (no modes, no links)."""
    return {k: v[0] for k, v in tree.items() if v[0] != "link"}


def _kept(tree):
    """What the bucket must keep: excluded caches and the ephemeral dir left out."""
    return {k: v for k, v in tree.items()
            if not k.startswith(("scratch/", "node_modules/")) and not k.endswith(".pyc")}


def _fill(wd):
    (wd / "src" / "pkg").mkdir(parents=True)
    (wd / "src" / "pkg" / "a.py").write_text("a")
    (wd / "src" / "pkg" / "a.pyc").write_text("cache")
    (wd / "run.sh").write_text("#!/bin/sh\n")
    os.chmod(wd / "run.sh", 0o755)
    (wd / "ro").write_text("read-only, like a git object")
    os.chmod(wd / "ro", 0o444)
    (wd / "link").symlink_to("src/pkg/a.py")
    (wd / "dirlink").symlink_to("src")
    (wd / "dangling").symlink_to("/nonexistent/x")
    (wd / "node_modules").mkdir()
    (wd / "node_modules" / "m.js").write_text("m")
    (wd / "scratch").mkdir(exist_ok=True)
    (wd / "scratch" / "clone").write_text("throwaway")
    os.utime(wd / "src" / "pkg" / "a.py", (1_600_000_000.25, 1_600_000_000.25))


async def _agent(path, store, **kw):
    rt, _ = runtime(path, sync=BucketSync(store, "sw"), **{**BASE, **kw})
    await rt.setup(restore=False)
    rt.workdir.mkdir(parents=True, exist_ok=True)
    return rt


async def _restored(path, store, **kw):
    rt, _ = runtime(path, sync=BucketSync(store, "sw"), **{**BASE, **kw})
    res = await rt.sync.restore(rt)
    return rt, res


@pytest.fixture
def umask022():
    old = os.umask(0o022)
    yield
    os.umask(old)


async def test_base_tar_round_trip_keeps_modes_links_mtimes_and_skips_excludes(tmp_path, umask022):
    from legion.worker.sync import BASE_JSON, BASE_TAR

    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    assert {PRE + BASE_TAR, PRE + BASE_JSON} <= set(store.files)
    assert not [k for k in store.files if k.startswith(PRE + "workdir/")]  # all in the base
    base = json.loads(store.files[PRE + BASE_JSON])
    assert base["files"] == 7 and base["deleted"] == [] and base["links"] == {}
    import io
    import tarfile
    with tarfile.open(fileobj=io.BytesIO(store.files[PRE + BASE_TAR])) as tf:
        names = tf.getnames()
    assert names == sorted(names) and not [n for n in names if "scratch" in n or "node_mod" in n]

    rt2, res = await _restored(tmp_path / "b", store)
    assert _tree(rt2.workdir) == _kept(_tree(rt.workdir)) and res.failed == []
    assert rt2.sync.mirror_deletes(rt2)
    await rt2.sync.push(rt2)  # nothing changed: no delta, no manifest rewrite
    assert not [k for k in store.files if k.startswith(PRE + "workdir/")]


async def test_delta_after_base_sends_only_changes_and_tombstones_delete(tmp_path, umask022):
    from legion.worker.sync import BASE_JSON

    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    wd = rt.workdir
    (wd / "run.sh").write_text("#!/bin/sh\necho 2\n")  # changed
    (wd / "new.txt").write_text("new")  # added
    (wd / "ro").unlink()  # deleted: a tombstone
    (wd / "old.txt").write_text("copied with cp -p")  # an mtime older than the base
    os.utime(wd / "old.txt", (1_500_000_000, 1_500_000_000))
    (wd / "link").unlink()
    (wd / "link").symlink_to("new.txt")  # a changed symlink: in the manifest
    store.uploads.clear()
    await rt.sync.push(rt)  # a drain/preempt push: never a new base
    deltas = {k for k in store.files if k.startswith(PRE + "workdir/")}
    assert deltas == {PRE + "workdir/" + f for f in ("run.sh", "new.txt", "old.txt")}
    base = json.loads(store.files[PRE + BASE_JSON])
    assert base["deleted"] == ["ro"] and base["links"] == {"link": "new.txt"}
    assert sum(len(u) for u in store.uploads) <= 5  # 3 deltas + manifest (+ session files)
    store.uploads.clear()
    await rt.sync.push(rt)
    assert not [r for u in store.uploads for r in u if "/workdir" in r]  # nothing new

    rt2, res = await _restored(tmp_path / "b", store)
    assert _tree(rt2.workdir) == _kept(_tree(wd)) and res.failed == []
    # deleting a delta file: the delta entry goes, a tombstone is not needed
    (wd / "new.txt").unlink()
    await rt.sync.push(rt)
    assert PRE + "workdir/new.txt" not in store.files
    rt3, _ = await _restored(tmp_path / "c", store)
    assert _tree(rt3.workdir) == _kept(_tree(wd))


class CrashStore(MemoryStore):
    """Raises on the upload of a path ending in `crash`, or on any delete when crash="delete"."""

    crash = ""

    def upload(self, files):
        if self.crash and any(r.endswith(self.crash) for _, r in files):
            raise RuntimeError("crash")
        super().upload(files)

    def delete(self, paths):
        if self.crash == "delete":
            raise RuntimeError("crash")
        super().delete(paths)


@pytest.mark.parametrize("crash", ["workdir.base.tar", "workdir.base.json", "delete"])
@pytest.mark.parametrize("first", [True, False], ids=["first-base", "next-base"])
async def test_restore_after_a_crash_while_making_a_base_equals_the_source(
        tmp_path, umask022, crash, first):
    store = CrashStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    wd = rt.workdir
    if first:  # the plain per-file mirror, no base yet
        await rt.sync.push(rt)
    else:
        await rt.sync.push(rt, rebase=True)
        (wd / "ro").unlink()
        (wd / "run.sh").write_text("#!/bin/sh\necho 2\n")
        (wd / "link").unlink()
        (wd / "link").symlink_to("run.sh")
        await rt.sync.push(rt)
    (wd / "late.txt").write_text("written before the base")
    os.utime(wd / "late.txt", (1_600_000_000, 1_600_000_000))
    (wd / "src" / "pkg" / "a.py").unlink()  # deleted before the base
    await rt.sync.push(rt)
    store.crash = crash
    rt.spec.workdir.base_interval_s = 0.001  # the base is due
    with pytest.raises(RuntimeError):
        await rt.sync.push(rt, rebase=True)
    rt2, res = await _restored(tmp_path / "b", store)
    if first and crash != "delete":  # no base committed: the plain mirror, file bytes only
        assert _bytes(_tree(rt2.workdir)) == _bytes(_kept(_tree(wd))) and res.failed == []
    else:
        assert _tree(rt2.workdir) == _kept(_tree(wd)) and res.failed == []
    # the next push heals a partial commit: stale deltas go, a restore still matches
    store.crash = ""
    await rt.sync.push(rt, rebase=True)
    rt3, _ = await _restored(tmp_path / "c", store)
    assert _tree(rt3.workdir) == _kept(_tree(wd))
    assert not [k for k in store.files if k.startswith(PRE + "workdir/")]


async def test_plain_mirror_bucket_restores_then_gets_a_base_and_can_go_back(tmp_path, umask022):
    from legion.worker.sync import BASE_JSON, BASE_TAR

    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store, workdir={"ephemeral": ["scratch"]})  # base off
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    assert PRE + BASE_JSON not in store.files and PRE + "workdir/run.sh" in store.files

    rt2, _ = await _restored(tmp_path / "b", store)  # base on, a bucket without one
    # the plain mirror brings symlinked files back as copies
    assert _bytes(_kept(_tree(rt.workdir))).items() <= _bytes(_tree(rt2.workdir)).items()
    await rt2.sync.push(rt2)  # not periodic: still the plain mirror
    assert PRE + BASE_JSON not in store.files
    await rt2.sync.push(rt2, rebase=True)  # the first periodic push makes the base
    assert PRE + BASE_JSON in store.files
    assert not [k for k in store.files if k.startswith(PRE + "workdir/")]

    rt2.spec.workdir.base = False  # turned off: the full mirror again, the base removed
    await rt2.sync.push(rt2)
    assert PRE + BASE_JSON not in store.files and PRE + BASE_TAR not in store.files
    rt3, _ = await _restored(tmp_path / "c", store, workdir={"ephemeral": ["scratch"]})
    assert _bytes(_tree(rt3.workdir)) == _bytes(_tree(rt2.workdir))


async def test_unreadable_base_is_an_incomplete_restore_and_blocks_deletes(tmp_path):
    from legion.worker.sync import BASE_TAR

    store = FlakyStore(bad=[PRE + BASE_TAR])
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    rt2, res = await _restored(tmp_path / "b", store)
    assert res.failed == [PRE + BASE_TAR] and not rt2.sync.mirror_deletes(rt2)
    rt2.workdir.mkdir(parents=True, exist_ok=True)
    (rt2.workdir / "x.txt").write_text("x")
    before = dict(store.files)
    await rt2.sync.push(rt2, rebase=True)  # uploads, but no new base and no deletes
    assert set(before) <= set(store.files) and store.files[PRE + BASE_TAR] == before[PRE + BASE_TAR]
    assert PRE + "workdir/x.txt" in store.files


async def test_a_push_during_a_push_keeps_the_delta_it_uploaded(tmp_path, umask022):
    import asyncio

    loop = asyncio.get_running_loop()

    class Interleaved(MemoryStore):
        """On the first workdir listing, another push (a preempt) runs with a new file."""
        hook = False

        def list_info(self, prefix):
            if self.hook and prefix.endswith("workdir/"):
                self.hook = False
                (rt.workdir / "x.txt").write_text("written after the periodic walk")
                asyncio.run_coroutine_threadsafe(rt.sync.push(rt), loop).result()
            return super().list_info(prefix)

    store = Interleaved()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    (rt.workdir / "run.sh").write_text("#!/bin/sh\necho 2\n")
    store.hook = True
    await rt.sync.push(rt, rebase=True)
    assert PRE + "workdir/x.txt" in store.files
    rt2, res = await _restored(tmp_path / "b", store)
    assert _tree(rt2.workdir) == _kept(_tree(rt.workdir)) and res.failed == []


async def test_a_delta_replacing_a_base_symlink_is_not_written_through_it(tmp_path, umask022):
    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    await rt.sync.push(rt, rebase=True)
    (rt.workdir / "link").unlink()  # in the tar: link -> src/pkg/a.py
    (rt.workdir / "link").write_text("now a regular file")
    (rt.workdir / "dirlink").unlink()  # in the tar: dirlink -> src
    (rt.workdir / "dirlink" / "pkg").mkdir(parents=True)
    (rt.workdir / "dirlink" / "pkg" / "a.py").write_text("not src/pkg/a.py")
    rt.sync._mirror[PRE] = False  # as after an incomplete restore: deltas, no tombstones
    await rt.sync.push(rt)
    rt2, res = await _restored(tmp_path / "b", store)
    assert not (rt2.workdir / "link").is_symlink() and not (rt2.workdir / "dirlink").is_symlink()
    assert _tree(rt2.workdir) == _kept(_tree(rt.workdir)) and res.failed == []


async def test_drain_stops_a_base_build_and_nothing_is_committed(tmp_path, umask022):
    from legion.worker.sync import BASE_JSON, BASE_TAR

    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    rt.sync.stop_rebase = True
    await rt.sync.push(rt, rebase=True)
    assert PRE + BASE_JSON not in store.files and PRE + BASE_TAR not in store.files
    assert PRE + "workdir/run.sh" in store.files  # the plain mirror instead
    assert not list(rt.sessions_dir.glob("*.base*"))


async def test_no_base_and_no_restore_without_free_disk(tmp_path, umask022, monkeypatch):
    import shutil

    from legion.worker.sync import BASE_JSON, BASE_TAR

    store = MemoryStore()
    rt = await _agent(tmp_path / "a", store)
    _fill(rt.workdir)
    full = shutil.disk_usage(tmp_path)._replace(free=0)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: full)
    await rt.sync.push(rt, rebase=True)
    assert PRE + BASE_JSON not in store.files and PRE + "workdir/run.sh" in store.files
    monkeypatch.undo()
    await rt.sync.push(rt, rebase=True)
    assert PRE + BASE_JSON in store.files
    monkeypatch.setattr(shutil, "disk_usage", lambda p: full)
    rt2, res = await _restored(tmp_path / "b", store)
    assert res.failed == [PRE + BASE_TAR] and not rt2.sync.mirror_deletes(rt2)
