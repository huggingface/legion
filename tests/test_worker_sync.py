"""BucketSync: layout under <swarm>/<type>/<agent>/, incremental push, restore to the same paths."""

from __future__ import annotations

from swarm_harness.worker.sync import BucketSync, parse_bucket

from .worker_fakes import MemoryStore, runtime, until


def test_parse_bucket():
    assert parse_bucket("hf://buckets/cmpatino/swarm-harness") == ("cmpatino/swarm-harness", "")
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
