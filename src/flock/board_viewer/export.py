"""Publish allow-listed conversation data from an owner-side SQLite backup."""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import sqlite3
import tempfile
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

log = logging.getLogger(__name__)
SCHEMA = 1
FIELDS = {
    "principals": "id,display_name,identity_type",
    "channels": "id,name,description,kind,archived,created_at",
    "channel_members": "channel_id,principal_id",
    "messages": "id,channel_id,seq,sender_id,body,reply_to,thread_id,attachments,created_at,edited_at",
    "message_reactions": "message_id,principal_id,emoji,created_at",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def capture(data_dir: Path, target: Path) -> dict:
    """Never copy a live DB/WAL pair. SQLite backup pins a consistent database view."""
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    backup = target / "private.db"
    deadline = time.monotonic() + 20

    def progress(*_):
        if time.monotonic() > deadline:
            raise TimeoutError("Board snapshot exceeded its deadline")

    try:
        with contextlib.closing(sqlite3.connect(
            (data_dir / "board.db").resolve().as_uri() + "?mode=ro", uri=True, timeout=5
        )) as source, contextlib.closing(sqlite3.connect(backup)) as dest:
            source.backup(dest, pages=256, progress=progress)
            dest.row_factory = sqlite3.Row
            if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Board failed integrity check")
            doc = {"schema_version": SCHEMA}
            for table, fields in FIELDS.items():
                doc[table] = [dict(row) for row in dest.execute(f"SELECT {fields} FROM {table}")]
            referenced = set()
            for msg in doc["messages"]:
                msg["attachments"] = json.loads(msg["attachments"] or "[]")
                referenced.update(msg["attachments"])
            doc["attachments"] = []
            for row in dest.execute("SELECT id,storage_key,filename,content_type,size,sha256 "
                                    "FROM attachments WHERE state='ready' AND published=1"):
                row = dict(row)
                if row["id"] not in referenced:
                    continue
                source_path = (data_dir / "blobs" / "objects" / row.pop("storage_key")).resolve()
                source_path.relative_to((data_dir / "blobs" / "objects").resolve())
                if not source_path.is_file():
                    raise ValueError("Referenced attachment missing")
                sha = digest(source_path)
                if sha != row["sha256"]:
                    raise ValueError("Attachment checksum mismatch")
                blob = target / sha
                if not blob.exists():
                    import shutil
                    shutil.copyfile(source_path, blob)
                row["file"] = sha
                doc["attachments"].append(row)
        (target / "board.json").write_text(json.dumps(doc), encoding="utf-8")
        return doc
    finally:
        backup.unlink(missing_ok=True)


def publish(api, bucket: str, prefix: str, bundle: Path, *, completed: bool = False) -> dict:
    parts = prefix.split("/")
    if len(parts) != 5 or parts[:2] != ["flock", "runs"] or parts[-1] != "board":
        raise ValueError("Invalid board prefix")
    viewer = "/".join(parts[:-1]) + "/viewer"
    generation = uuid.uuid4().hex
    files = {p.name: digest(p) for p in bundle.iterdir() if p.is_file()}
    if "board.json" not in files or any(
        name != "board.json" and (len(name) != 64 or any(c not in "0123456789abcdef" for c in name))
        for name in files
    ):
        raise ValueError("Unexpected export files")
    manifest = {"schema_version": SCHEMA, "swarm": parts[2], "run_id": parts[3],
                "generation": generation, "published_at": datetime.now(UTC).isoformat(),
                "completed": completed, "files": files}
    # Manifest last: readers can only discover complete immutable generations.
    api.batch_bucket_files(bucket, add=[(bundle / name, f"{viewer}/{generation}/{name}")
                                        for name in files])
    api.batch_bucket_files(bucket, add=[(json.dumps(manifest).encode(), f"{viewer}/manifest.json")])
    return manifest


class Publisher:
    """Independent uploads: failures never fail a run or replace its last good export."""
    def __init__(self, mount, data_dir: Path):
        import os

        from huggingface_hub import HfApi
        self.api = HfApi(token=os.environ.get(mount.config.token_env))
        self.bucket, self.prefix, self.data_dir = mount.config.bucket, mount.prefix, data_dir
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="board-viewer-export")

    def once(self, *, completed=False):
        with tempfile.TemporaryDirectory(prefix="flock-viewer-") as tmp:
            bundle = Path(tmp)
            capture(self.data_dir, bundle)
            publish(self.api, self.bucket, self.prefix, bundle, completed=completed)

    def _loop(self):
        while not self.done.is_set():
            started = time.monotonic()
            try:
                self.once()
            except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                log.warning("Board viewer export failed; retaining previous generation")
            self.done.wait(max(0, 30 - (time.monotonic() - started)))

    def start(self):
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            log.warning("Viewer upload still pending; skipping final export to avoid racing manifests")
            return False
        return True
