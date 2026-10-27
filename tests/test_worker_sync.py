"""BucketSync: layout under <swarm>/<type>/<agent>/, incremental push, restore to the same paths."""

from __future__ import annotations

import json
import os

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
