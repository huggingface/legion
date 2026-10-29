"""Opt-in live compatibility gate; uses a real private bucket, no models or HF Jobs.

Run from the repository: uv run --no-sync python scripts/check_bucket_board.py --help
Snapshot retention keeps three verified generations in the printed unique prefix.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from flock.comms.managed import ManagedBoard, ManagedBoardError
from flock.models import SwarmSpec


def client(board: ManagedBoard, token: str) -> httpx.Client:
    return httpx.Client(base_url=board.url, timeout=20, trust_env=False,
                        headers={"Authorization": f"Bearer {token}"})


def assert_history(board: ManagedBoard, token: str, channel: int, expected: set[str]) -> None:
    with client(board, token) as http:
        response = http.get(f"/v1/channels/{channel}/messages", params={"limit": 200, "before": 9223372036854775807})
        response.raise_for_status()
        actual = {m["body"] for m in response.json()["messages"]}
        if not expected <= actual:
            raise RuntimeError("Restored snapshot lost previously verified history.")


def post(board: ManagedBoard, token: str, channel: int, body: str) -> None:
    with client(board, token) as http:
        response = http.post(f"/v1/channels/{channel}/messages", json={"body": body},
                             headers={"Idempotency-Key": uuid.uuid4().hex})
        response.raise_for_status()


def reopen(run_dir: Path, timeout: float = 300) -> ManagedBoard:
    board = ManagedBoard(run_dir, startup_timeout=timeout)
    try:
        board.start(restore=True)
        return board
    except BaseException:
        board.stop()
        raise


def faults(run_dir: Path, token: str, channel: int, baseline: set[str]) -> None:
    # A killed board must checkpoint/recover through the still-live mount on stop.
    board = reopen(run_dir)
    try:
        post(board, token, channel, "before-board-crash")
        os.killpg(board.proc.pid, signal.SIGKILL)
        board.proc.wait(timeout=10)
    finally:
        board.stop()
    baseline.add("before-board-crash")
    board = reopen(run_dir)
    try:
        assert_history(board, token, channel, baseline)
    finally:
        board.stop()
    print("PASS: abrupt msgboard exit, verified upload and fresh-cache recovery", flush=True)

    board = reopen(run_dir)
    original = board.bucket._api.batch_bucket_files
    def unavailable(*args, **kwargs):
        raise OSError("simulated upload outage")
    board.bucket._api.batch_bucket_files = unavailable
    post(board, token, channel, "during-upload-outage")
    try:
        board.stop()
    except ManagedBoardError:
        if not board.bucket.mount.exists() or board.bucket.verified_at is not None:
            raise RuntimeError("Failed upload was incorrectly marked persisted.") from None
    else:
        raise RuntimeError("Upload-outage test unexpectedly reported persistence.")
    board.bucket._api.batch_bucket_files = original
    board = reopen(run_dir)
    try:
        assert_history(board, token, channel, baseline)
    finally:
        board.stop()
    print("PASS: upload outage retains local data and last verified remote snapshot", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bucket", required=True, help="existing private namespace/bucket")
    parser.add_argument("--mount-root", default="/scratch/flock-boards")
    parser.add_argument("--data-dir", type=Path, default=Path(".flock"))
    parser.add_argument("--startup-timeout", type=float, default=300,
                        help="seconds for initial mount/upload readiness (default: 300)")
    parser.add_argument("--faults", action="store_true",
                        help="also kill the owned board and inject an upload failure")
    parser.add_argument("--duration", type=float, default=0)
    args = parser.parse_args()
    run_dir = args.data_dir.resolve() / f"bucket-check-{uuid.uuid4().hex}"
    spec = SwarmSpec.model_validate({
        "swarm": "bucket-check", "compute": "local", "types": {},
        "comms": {"managed": True, "storage": {
            "kind": "hf-bucket", "bucket": args.bucket, "mount_root": args.mount_root,
        }},
    })
    board = ManagedBoard(run_dir, startup_timeout=args.startup_timeout)
    print(f"run directory: {run_dir}", flush=True)
    expected = {f"concurrent-message-{i}" for i in range(24)}
    try:
        board.start(spec)
        print(f"bucket prefix: {board.bucket.prefix}", flush=True)
        token = board._operator
        with client(board, token) as http:
            channel = http.get("/v1/channels").json()["channels"][0]["id"]
        with client(board, token) as http:
            response = http.put(f"/v1/channels/{channel}/attachments/{uuid.uuid4().hex}",
                content=b"snapshot attachment", headers={"X-Filename": "gate.txt"})
            response.raise_for_status()
            attachment = response.json()["id"]
            http.post(f"/v1/channels/{channel}/messages",
                json={"body": "attachment gate", "attachments": [attachment]},
                headers={"Idempotency-Key": uuid.uuid4().hex}).raise_for_status()
        with ThreadPoolExecutor(max_workers=6) as pool:
            writes = [pool.submit(post, board, token, channel, text) for text in expected]
            reads = [pool.submit(assert_history, board, token, channel, set()) for _ in range(8)]
            for future in [*writes, *reads]:
                future.result()
        deadline = time.monotonic() + args.duration
        cycles = 0
        while time.monotonic() < deadline:
            with ThreadPoolExecutor(max_workers=16) as pool:
                futures = [pool.submit(post, board, token, channel, f"stress-{cycles}-{i}")
                           for i in range(8)]
                futures += [pool.submit(assert_history, board, token, channel, set())
                            for _ in range(8)]
                for future in futures:
                    future.result()
            cycles += 1
            if cycles % 10 == 0:
                print(f"Storage stress cycles={cycles}; snapshot={board.bucket.snapshot_at}", flush=True)
            time.sleep(1)
        # Recovery assertions use recent messages within the API's pagination window.
        if args.duration:
            expected = {f"recovery-{uuid.uuid4().hex}" for _ in range(24)}
            for text in expected:
                post(board, token, channel, text)
        with contextlib.closing(sqlite3.connect(board.local / "board.db")) as db:
            if db.execute("PRAGMA journal_mode").fetchone() != ("wal",):
                raise RuntimeError("msgboard is not running in WAL mode.")
            if db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0]:
                raise RuntimeError("Concurrent WAL checkpoint remained busy.")
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise RuntimeError("Live SQLite integrity check failed.")
        assert_history(board, token, channel, expected)
    finally:
        board.stop()
    previous_cache = board.bucket.cache
    board = reopen(run_dir, args.startup_timeout)
    try:
        if previous_cache == board.bucket.cache:
            raise RuntimeError("Reopen reused its old cache.")
        assert_history(board, token, channel, expected)
        with client(board, token) as http:
            response = http.get(f"/v1/attachments/{attachment}")
            response.raise_for_status()
            assert response.content == b"snapshot attachment"
    finally:
        board.stop()
    print("PASS: concurrent reads/writes, WAL checkpoint, remote verification, cold reopen", flush=True)
    if args.faults:
        faults(run_dir, token, channel, expected)
    else:
        print("Fault tests not run. Pass --faults before qualifying this backend.", flush=True)
    report = json.loads((run_dir / "board" / "metadata.json").read_text())
    print(f"last verified upload: {report['storage']['verified_at']}", flush=True)


if __name__ == "__main__":
    main()
