"""Explicit import of a verified, stopped bucket board; never starts msgboard."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from .export import capture, publish


def import_board(run_dir: Path, api=None):
    archive = run_dir.resolve() / "board"
    # Share the managed-board lock: a local reopen cannot race this import.
    with (archive / ".lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        metadata = json.loads((archive / "metadata.json").read_text())
        storage = metadata["storage"]
        if (metadata.get("running") or storage.get("kind") != "hf-bucket"
                or not storage.get("verified_at")):
            raise ValueError("Import requires a stopped, verified bucket board")
        api = api or HfApi(token=os.environ[storage.get("token_env", "HF_TOKEN")])
        prefix, bucket = storage["prefix"], storage["bucket"]
        with tempfile.TemporaryDirectory(prefix="flock-viewer-import-") as tmp:
            root = Path(tmp)
            data, bundle = root / "data", root / "export"
            data.mkdir()
            wal = api.get_bucket_paths_info(bucket, [f"{prefix}/board.db-wal"])
            if any(p.size for p in wal):
                raise ValueError("Board has a nonempty remote WAL; cannot import")
            api.download_bucket_files(bucket, [(f"{prefix}/board.db", data / "board.db")],
                                       raise_on_missing_files=True)
            # Download only ready, published attachment blobs referenced by this DB.
            import contextlib
            import sqlite3
            with contextlib.closing(sqlite3.connect(data / "board.db")) as conn:
                refs = {a for (encoded,) in conn.execute("SELECT attachments FROM messages")
                        for a in json.loads(encoded or "[]")}
                files = []
                for aid, key in conn.execute("SELECT id,storage_key FROM attachments "
                                             "WHERE state='ready' AND published=1"):
                    if aid in refs:
                        dest = (data / "blobs" / "objects" / key).resolve()
                        dest.relative_to((data / "blobs" / "objects").resolve())
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        files.append((f"{prefix}/blobs/objects/{key}", dest))
                if files:
                    api.download_bucket_files(bucket, files, raise_on_missing_files=True)
            capture(data, bundle)
            return publish(api, bucket, prefix, bundle, completed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    result = import_board(args.run_dir)
    print(f"Published {result['swarm']}/{result['run_id']}")


if __name__ == "__main__":
    main()
