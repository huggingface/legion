"""Bucket sync cost for N synthetic agents, outside a job: serial push (as the worker's periodic
sync loop does), then a concurrent restore into a fresh directory (as a relaunch does).

    python sync_bench.py hf://buckets/cmpatino/flock/load-1k-bench --agents 200

Each agent: workdir/CLAUDE.md + notes.md, sessions/<name>.jsonl + .state.json, one ~80 KB
harness transcript. Prints seconds and bucket calls per phase (BucketStats).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

from flock.worker.sync import BucketStats, BucketSync, HubBucketStore, count_hub_http


def fake_agent(root: Path, name: str, sid: str | None, write: bool):
    work = root / "agents" / name
    sessions = root / "agents" / "_sessions"
    home = root / "claude"
    proj = home / "projects" / f"-work-agents-{name}"
    if write:
        work.mkdir(parents=True, exist_ok=True)
        sessions.mkdir(parents=True, exist_ok=True)
        proj.mkdir(parents=True, exist_ok=True)
        (work / "CLAUDE.md").write_text("Keep notes.md.\n" * 20)
        (work / "notes.md").write_text(f"- Codeword: {name}-abc123\n")
        (sessions / f"{name}.jsonl").write_text('{"type":"session"}\n' * 40)
        (sessions / f"{name}.state.json").write_text(json.dumps({"session_id": sid}))
        (proj / f"{sid}.jsonl").write_text(("x" * 800 + "\n") * 100)
    files = lambda: [proj / f"{sid}.jsonl"] if sid and (proj / f"{sid}.jsonl").exists() else []
    return SimpleNamespace(
        name=name, type_name=name.rsplit("-", 1)[0], workdir=work, sessions_dir=sessions,
        state_path=sessions / f"{name}.state.json", log=SimpleNamespace(path=sessions / f"{name}.jsonl"),
        adapter=SimpleNamespace(home=home), harness_files=files,
        spec=SimpleNamespace(persist_workdir=True))


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bucket")
    ap.add_argument("--agents", type=int, default=200)
    ap.add_argument("--swarm", default="bench")
    a = ap.parse_args()
    stats = BucketStats()
    count_hub_http(stats)
    store = HubBucketStore(a.bucket, stats=stats)
    out = {}
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as dst:
        sids = {f"scribe-{i:04d}": str(uuid.uuid4()) for i in range(a.agents)}
        sync = BucketSync(store, a.swarm)
        rts = [fake_agent(Path(src), n, s, True) for n, s in sids.items()]
        t0, snap = time.monotonic(), stats.snapshot()
        per = []
        for rt in rts:
            t = time.monotonic()
            await sync.push(rt)
            per.append(time.monotonic() - t)
        per.sort()
        out["push_serial"] = {"agents": len(rts), "s": round(time.monotonic() - t0, 1),
                              "per_agent_p50": round(per[len(per) // 2], 2),
                              "per_agent_max": round(per[-1], 2), "bucket": stats.since(snap)}
        print(json.dumps(out["push_serial"]), flush=True)
        # second cycle with nothing changed but the session log (the steady-state case)
        for rt in rts:
            with rt.log.path.open("a") as f:
                f.write('{"type":"sync"}\n')
        t0, snap = time.monotonic(), stats.snapshot()
        for rt in rts:
            await sync.push(rt)
        out["push_serial_one_file_changed"] = {"s": round(time.monotonic() - t0, 1),
                                               "bucket": stats.since(snap)}
        print(json.dumps(out["push_serial_one_file_changed"]), flush=True)
        sync2 = BucketSync(store, a.swarm)
        fresh = [fake_agent(Path(dst), n, s, False) for n, s in sids.items()]
        t0, snap = time.monotonic(), stats.snapshot()
        res = await asyncio.gather(*(sync2.restore(rt) for rt in fresh), return_exceptions=True)
        ok = sum(1 for r in res if not isinstance(r, Exception) and r.harness == 1 and r.workdir == 2)
        out["restore_concurrent"] = {"agents": len(fresh), "complete": ok,
                                     "errors": [repr(r)[:200] for r in res if isinstance(r, Exception)][:3],
                                     "s": round(time.monotonic() - t0, 1), "bucket": stats.since(snap)}
        print(json.dumps(out["restore_concurrent"]), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
