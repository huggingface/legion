"""Capture native evidence and Pi usage before job-local storage is deleted."""

import json
import os
import shutil
import sqlite3
import sys
import time
from pathlib import Path

r = Path(sys.argv[1])
expected = int(sys.argv[2])
src = Path("/tmp/legion") / (r.name + "-agent-w0000") / "work/_harbor"
end = time.time() + 10800
last_board = 0
while time.time() < end:
    complete = 0
    ready = src / "_worker/state/resources/start_barrier/ready.json"
    if ready.exists():
        try:
            value = json.loads(ready.read_text())
            tmp = r / "barrier-ready.tmp"
            tmp.write_text(json.dumps(value))
            tmp.replace(r / "barrier-ready.json")
        except (OSError, ValueError):
            pass
    barrier = src / "_worker/state/resources/start_barrier/released.json"
    if barrier.exists() and not (r / "barrier.json").exists():
        shutil.copy2(barrier, r / "barrier.json")
    board = src / "_worker/state/resources/message_board/data/board.db"
    if board.exists() and time.time() - last_board > 5:
        try:
            with (
                sqlite3.connect("file:" + str(board) + "?mode=ro", uri=True) as source,
                sqlite3.connect(r / "board.db") as dest,
            ):
                source.backup(dest)
            last_board = time.time()
        except sqlite3.Error:
            pass
    for trial in src.glob("agent-*/trials/*"):
        dst = r / "native" / trial.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in trial.rglob("*"):
            if not f.is_file() or f.is_symlink():
                continue
            rel = f.relative_to(trial)
            if f.name not in [
                "result.json",
                "final.json",
                "trace.jsonl",
                "reward.json",
                "reward.txt",
                "pi.txt",
                "trajectory.json",
                "request-settings.jsonl",
                "sft-capture.jsonl",
            ] and not ("sessions" in rel.parts and f.suffix == ".jsonl"):
                continue
            target = dst / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                if f.name == "result.json":
                    a = json.loads(f.read_text())
                    safe = {
                        k: a.get(k)
                        for k in [
                            "trial_name",
                            "task_name",
                            "started_at",
                            "finished_at",
                            "verifier_result",
                            "agent_result",
                        ]
                    }
                    safe["exception_type"] = (a.get("exception_info") or {}).get("exception_type")
                    tmp = target.with_name(target.name + ".tmp")
                    tmp.write_text(json.dumps(safe))
                    os.replace(tmp, target)
                    complete += bool(a.get("finished_at"))
                elif not target.exists() or f.stat().st_mtime_ns != target.stat().st_mtime_ns:
                    shutil.copy2(f, target)
            except (OSError, ValueError):
                pass
    if complete >= expected:
        if board.exists():
            try:
                with (
                    sqlite3.connect("file:" + str(board) + "?mode=ro", uri=True) as source,
                    sqlite3.connect(r / "board.db") as dest,
                ):
                    source.backup(dest)
            except sqlite3.Error:
                pass
        break
    time.sleep(0.1)
print(json.dumps({"native_complete": complete, "expected": expected}), flush=True)
