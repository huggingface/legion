"""Node-local SQLite with immutable, verified private Hub bucket snapshots.

BucketMount is retained as an import-compatible name; no FUSE mount is created.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from flock.models import BucketBoardStorage, parse_duration

log = logging.getLogger(__name__)


class BucketStorageError(ValueError):
    """A credential-free lifecycle error."""


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(p in {".", ".."} for p in path.parts):
        raise BucketStorageError("Invalid snapshot path")
    return name


class BucketMount:
    def __init__(self, config: BucketBoardStorage, swarm: str, log_path: Path,
                 *, prefix: str | None = None) -> None:
        self.config = config
        self.id = uuid.uuid4().hex
        self.root = Path(config.mount_root).resolve() / self.id
        self.mount = self.root / "data"
        self.cache = self.root / "snapshots"
        self.prefix = prefix or f"flock/runs/{swarm}/{self.id}/board"
        if not re.fullmatch(r"flock/runs/[a-zA-Z0-9_-]+/[0-9a-f]{32}/board", self.prefix):
            raise BucketStorageError("Invalid retained bucket board prefix.")
        self.log_path = log_path
        self._api = None
        self.verified_at = None
        self.snapshot_at = None
        self.last_error = None
        self._stopped = False
        self._done = threading.Event()
        self._thread = None
        self._lock = threading.Lock()
        self._generations = []

    @property
    def shutdown_timeout(self):
        return parse_duration(self.config.flush_shutdown_timeout).total_seconds()

    def start(self, cancel: threading.Event, timeout: float, *, restore=False) -> Path:
        from huggingface_hub import HfApi

        from flock.comms.managed import ManagedBoard

        if cancel.is_set():
            raise BucketStorageError("Bucket startup interrupted.")
        token = os.environ.get(self.config.token_env)
        if not token:
            raise BucketStorageError(f"Set {self.config.token_env} for the private board bucket.")
        self._api = HfApi(token=token)
        try:
            if not self._api.bucket_info(self.config.bucket).private:
                raise BucketStorageError("Managed boards require a private bucket.")
            self.root.mkdir(parents=True, mode=0o700)
            self.root.chmod(0o700)
            self.mount.mkdir(mode=0o700)
            self.cache.mkdir(mode=0o700)
            ManagedBoard._check_local(self.mount)
            probe = f"{self.prefix}/.write-check-{self.id}"
            self._api.batch_bucket_files(self.config.bucket, add=[(self.id.encode(), probe)])
            dest = self.root / "probe"
            self._api.download_bucket_files(self.config.bucket, [(probe, dest)],
                                            raise_on_missing_files=True)
            if dest.read_bytes() != self.id.encode():
                raise BucketStorageError("Bucket write-access check failed.")
            self._api.batch_bucket_files(self.config.bucket, delete=[probe])
            dest.unlink()
            if restore:
                self._restore()
            elif list(self._api.get_bucket_paths_info(self.config.bucket,
                    [f"{self.prefix}/manifest.json", f"{self.prefix}/board.db"])):
                raise BucketStorageError("Refusing to initialize an existing bucket board.")
        except BucketStorageError:
            raise
        except Exception:  # noqa: BLE001 — redact credentials from SDK failures
            raise BucketStorageError("Bucket startup failed; local recovery files retained.") from None
        return self.mount

    def check(self):
        if not self.mount.is_dir() or self._stopped:
            raise BucketStorageError("Local board storage is unavailable.")

    def metadata(self):
        return {**self.config.model_dump(), "prefix": self.prefix, "backend": "snapshot",
                "schema_version": 1, "data_dir": str(self.mount), "recovery_dir": str(self.root),
                "verified_at": self.verified_at, "snapshot_at": self.snapshot_at,
                "snapshot_error": self.last_error}

    @staticmethod
    def _integrity(path):
        with contextlib.closing(sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)) as conn:
            if conn.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise BucketStorageError("Snapshot failed integrity_check.")

    def _download(self, generation, dest):
        files = generation["files"]
        if "board.db" not in files or "operator.token" not in files:
            raise BucketStorageError("Incomplete board snapshot.")
        for name in files:
            _safe(name)
        remote = f"{self.prefix}/generations/{_safe(generation['id'])}"
        self._api.download_bucket_files(self.config.bucket,
            [(f"{remote}/{name}", dest / name) for name in files], raise_on_missing_files=True)
        for name, digest in files.items():
            path = dest / name
            path.chmod(0o600)
            if _digest(path) != digest:
                raise BucketStorageError("Snapshot checksum mismatch.")
        self._integrity(dest / "board.db")

    def _restore(self):
        manifest = f"{self.prefix}/manifest.json"
        if list(self._api.get_bucket_paths_info(self.config.bucket, [manifest])):
            target = self.root / "manifest.json"
            self._api.download_bucket_files(self.config.bucket, [(manifest, target)],
                                            raise_on_missing_files=True)
            doc = json.loads(target.read_text())
            if doc.get("schema_version") != 1:
                raise BucketStorageError("Unsupported board snapshot version.")
            self._generations = doc["generations"]
            for generation in self._generations:
                try:
                    self._download(generation, self.mount)
                    return
                except Exception:  # noqa: BLE001 — redact credentials from SDK failures
                    shutil.rmtree(self.mount)
                    self.mount.mkdir(mode=0o700)
            raise BucketStorageError("No valid retained snapshot; originals preserved.")
        # Legacy flat backups: recover DB and sidecars together on local disk only.
        files = [p.path for p in self._api.list_bucket_tree(
            self.config.bucket, prefix=self.prefix + "/", recursive=True) if hasattr(p, "size")]
        names = [(p, _safe(p.removeprefix(self.prefix + "/"))) for p in files]
        self._api.download_bucket_files(self.config.bucket,
            [(p, self.mount / name) for p, name in names], raise_on_missing_files=True)
        if not (self.mount / "operator.token").is_file():
            raise BucketStorageError("Legacy operator token missing.")
        self._integrity(self.mount / "board.db")
        for p in self.mount.rglob("*"):
            if p.is_file():
                p.chmod(0o600)

    def snapshot(self):
        """Serialize backups/uploads; publish only after download verification."""
        with self._lock:
            generation = uuid.uuid4().hex
            bundle = self.cache / generation
            bundle.mkdir(mode=0o700)
            deadline = time.monotonic() + 20

            def progress(*_):
                if time.monotonic() > deadline:
                    raise BucketStorageError("SQLite backup deadline exceeded.")

            try:
                with contextlib.closing(sqlite3.connect(
                    (self.mount / "board.db").as_uri() + "?mode=ro", uri=True, timeout=5
                )) as src, contextlib.closing(sqlite3.connect(bundle / "board.db")) as dst:
                    src.backup(dst, pages=256, progress=progress)
                for source in self.mount.rglob("*"):
                    if not source.is_file() or source.is_symlink():
                        continue
                    name = source.relative_to(self.mount).as_posix()
                    if source.name.startswith("board.db") or source.name.endswith(".lock"):
                        continue
                    dest = bundle / name
                    dest.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                    shutil.copyfile(source, dest)
                # A backup referencing a missing/changed ready attachment is not recoverable.
                with contextlib.closing(sqlite3.connect(bundle / "board.db")) as db:
                    has_attachments = db.execute(
                        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='attachments'"
                    ).fetchone()
                    if has_attachments:
                        for key, digest in db.execute(
                            "SELECT storage_key,sha256 FROM attachments WHERE state='ready'"
                        ):
                            blob = bundle / "blobs/objects" / _safe(key)
                            if not blob.is_file() or _digest(blob) != digest:
                                raise BucketStorageError("Referenced attachment missing or changed.")
                files = {p.relative_to(bundle).as_posix(): _digest(p)
                         for p in bundle.rglob("*") if p.is_file()}
                for p in bundle.rglob("*"):
                    if p.is_file():
                        p.chmod(0o600)
                remote = f"{self.prefix}/generations/{generation}"
                self._api.batch_bucket_files(self.config.bucket,
                    add=[(bundle / name, f"{remote}/{name}") for name in files])
                entry = {"id": generation, "files": files}
                self._verify(entry)
                retained = [entry, *self._generations][:3]
                doc = {"schema_version": 1, "generations": retained}
                self._api.batch_bucket_files(self.config.bucket,
                    add=[(json.dumps(doc).encode(), f"{self.prefix}/manifest.json")])
                obsolete = self._generations[2:]
                self._generations = retained
                self.snapshot_at = datetime.now(UTC).isoformat()
                self.last_error = None
                # Cleanup cannot invalidate a published, verified generation.
                for old in obsolete:
                    try:
                        self._api.batch_bucket_files(self.config.bucket, delete=[
                            f"{self.prefix}/generations/{old['id']}/{name}" for name in old["files"]])
                    except Exception:  # noqa: BLE001 — redact credentials from SDK failures
                        log.warning("Could not prune old board snapshot")
            except Exception:  # noqa: BLE001 — redact credentials from SDK failures
                self.last_error = "snapshot_upload_failed"
                raise BucketStorageError("Snapshot upload/verification failed; local data retained.") from None
            finally:
                shutil.rmtree(bundle, ignore_errors=True)

    def _verify(self, entry):
        target = self.cache / (entry["id"] + "-verify")
        target.mkdir(mode=0o700)
        try:
            self._download(entry, target)
        finally:
            shutil.rmtree(target, ignore_errors=True)

    def start_snapshots(self):
        if self._thread:
            return
        interval = parse_duration(self.config.flush_max_batch_window).total_seconds()

        def loop():
            while not self._done.wait(interval):
                try:
                    self.snapshot()
                except BucketStorageError:
                    log.warning("Board snapshot failed; retrying with local board still available")
        self._thread = threading.Thread(target=loop, daemon=True, name="board-snapshots")
        self._thread.start()

    def stop(self, *, verify):
        if self._stopped:
            return
        self._done.set()
        if self._thread:
            self._thread.join(timeout=self.shutdown_timeout)
            if self._thread.is_alive():
                raise BucketStorageError("Snapshot uploader did not stop; local data retained.")
        if verify:
            errors = []
            def final_snapshot():
                try:
                    self.snapshot()
                except BucketStorageError as exc:
                    errors.append(exc)
            final = threading.Thread(target=final_snapshot, daemon=True, name="board-final-snapshot")
            final.start()
            final.join(timeout=self.shutdown_timeout)
            if final.is_alive():
                raise BucketStorageError("Final snapshot timed out; local data retained.")
            if errors:
                raise errors[0]
            self.verified_at = self.snapshot_at
        self._stopped = True
        if verify:
            shutil.rmtree(self.root)
        self._api = None
