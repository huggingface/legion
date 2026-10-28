"""Owned hf-mount FUSE sidecar. The mount, rather than Flock, uploads live board files.

A successful daemon exit is insufficient: verify stopped files through the Hub API.
This is a single-client experimental backend, not distributed SQLite storage.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from flock.models import BucketBoardStorage, parse_duration

MOUNT_VERSION = "0.13.2"


class BucketStorageError(ValueError):
    """A credential-free lifecycle error."""


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mount_entry(path: Path) -> tuple[str, str, str] | None:
    """Read kernel state without touching a potentially dead FUSE server."""
    target = str(path)
    found = None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        mounted = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4])
        if mounted == target:
            if found is not None:
                raise BucketStorageError("Multiple mounts cover the owned board mountpoint.")
            found = fields[0], fs[0], fs[1]
    return found


class BucketMount:
    def __init__(self, config: BucketBoardStorage, swarm: str, log_path: Path,
                 *, prefix: str | None = None) -> None:
        self.config = config
        self.id = uuid.uuid4().hex
        self.root = Path(config.mount_root).resolve() / self.id
        self.mount = self.root / "mount"
        self.cache = self.root / "cache"
        self.token_file = self.root / "hf.token"
        # Mount the unique subfolder, never the shared bucket root.
        self.prefix = prefix or f"flock/runs/{swarm}/{self.id}/board"
        parts = PurePosixPath(self.prefix).parts
        if (len(parts) != 5 or parts[:2] != ("flock", "runs") or parts[-1] != "board"
                or not re.fullmatch(r"[0-9a-f]{32}", parts[-2])
                or not re.fullmatch(r"[a-zA-Z0-9_-]+", parts[2])):
            raise BucketStorageError("Invalid retained bucket board prefix.")
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self._mount_id: str | None = None
        self._token: str | None = None
        self._api = None
        self.verified_at: str | None = None
        self._stopped = False

    @staticmethod
    def _env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items()
                if k in {"PATH", "HOME", "LANG", "LANGUAGE", "TZ"} or k.startswith("LC_")}

    def start(self, cancel: threading.Event, timeout: float, *, restore: bool = False) -> Path:
        from huggingface_hub import HfApi

        if sys.platform != "linux":
            raise BucketStorageError("Bucket boards currently require Linux FUSE mount support.")
        binary = shutil.which("hf-mount-fuse")
        if binary is None:
            raise BucketStorageError(f"Install hf-mount-fuse v{MOUNT_VERSION} on PATH first.")
        result = subprocess.run([binary, "--version"], capture_output=True, text=True,
                                timeout=5, env=self._env(), check=False)
        if result.returncode or not re.fullmatch(r"hf-mount(?:-fuse)? 0\.13\.2\s*", result.stdout):
            raise BucketStorageError(f"Bucket boards require hf-mount-fuse v{MOUNT_VERSION}.")
        self._token = os.environ.get(self.config.token_env)
        if not self._token:
            raise BucketStorageError(f"Set {self.config.token_env} for the private board bucket.")
        self._api = HfApi(token=self._token)
        if not hasattr(self._api, "download_bucket_files"):
            raise BucketStorageError("Upgrade huggingface_hub: bucket APIs are required.")
        try:
            info = self._api.bucket_info(self.config.bucket)
        except Exception:  # noqa: BLE001 — redact upstream errors that may contain credentials
            raise BucketStorageError("Cannot access the existing board bucket.") from None
        if not info.private:
            raise BucketStorageError("Managed boards require a private bucket (it stores credentials).")
        self.root.mkdir(parents=True, mode=0o700)
        self.root.chmod(0o700)
        self.mount.mkdir(mode=0o700)
        self.cache.mkdir(mode=0o700)
        fd = os.open(self.token_file, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as stream:
            stream.write(self._token)
        args = [binary, "--token-file", str(self.token_file), "--cache-dir", str(self.cache),
                "--dir-mode", "0700", "--file-mode", "0600",
                "--fuse-owner-only", "--advanced-writes"]
        for field in ("flush_debounce", "flush_max_batch_window", "flush_shutdown_timeout"):
            ms = round(parse_duration(getattr(self.config, field)).total_seconds() * 1000)
            args += ["--" + field.replace("_", "-") + "-ms", str(ms)]
        args += ["bucket", f"{self.config.bucket}/{self.prefix}", str(self.mount)]
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("ab") as out:
            os.chmod(self.log_path, 0o600)
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "flock.comms.bucket", str(os.getpid()),
                 str(self.token_file), str(self.shutdown_timeout), *args],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True, env=self._env())
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cancel.is_set():
                raise BucketStorageError("Bucket mount startup interrupted.")
            if self.proc.poll() is not None:
                raise BucketStorageError(f"Bucket mount failed; see {self.log_path}")
            entry = _mount_entry(self.mount)
            if entry:
                if entry[1] not in {"fuse", "fuse.hf-mount"} or entry[2] != "hf-mount":
                    raise BucketStorageError("Unexpected filesystem at the owned board mount.")
                self._mount_id = entry[0]
                if restore and not (self.mount / "board.db").is_file():
                    raise BucketStorageError("Retained bucket board.db is missing.")
                if not restore and any(self.mount.iterdir()):
                    raise BucketStorageError("Refusing to initialize a nonempty bucket prefix.")
                self._check_writable(cancel, deadline)
                return self.mount
            time.sleep(0.05)
        raise BucketStorageError(f"Bucket mount readiness timed out; see {self.log_path}")

    def _check_writable(self, cancel: threading.Event, deadline: float) -> None:
        # Exercise this mount's upload path before starting a database or any agents.
        name = f".flock-write-check-{self.id}"
        probe = self.mount / name
        expected = self.id.encode()
        probe.write_bytes(expected)
        downloaded = self.root / "write-check"
        verified = False
        try:
            while time.monotonic() < deadline:
                if cancel.is_set():
                    raise BucketStorageError("Bucket write-access check interrupted.")
                self.check()
                try:
                    found = self._api.get_bucket_paths_info(
                        self.config.bucket, [f"{self.prefix}/{name}"])
                    if found:
                        self._api.download_bucket_files(
                            self.config.bucket, [(f"{self.prefix}/{name}", downloaded)],
                            raise_on_missing_files=True)
                        if downloaded.read_bytes() == expected:
                            verified = True
                            return
                except Exception:  # noqa: BLE001, S110 — redact SDK errors, retry until deadline
                    pass
                cancel.wait(0.5)
            raise BucketStorageError("Bucket write-access check failed; no agents were started.")
        finally:
            # Do not touch a dead FUSE filesystem during failure cleanup.
            if verified:
                probe.unlink(missing_ok=True)
            downloaded.unlink(missing_ok=True)

    @property
    def shutdown_timeout(self) -> float:
        return parse_duration(self.config.flush_shutdown_timeout).total_seconds() + 15

    def check(self) -> None:
        entry = _mount_entry(self.mount)
        if (self.proc is None or self.proc.poll() is not None or entry is None
                or entry[0] != self._mount_id):
            raise BucketStorageError(f"Owned bucket mount exited or disappeared; see {self.log_path}")

    def metadata(self) -> dict:
        return {**self.config.model_dump(), "prefix": self.prefix, "mount_version": MOUNT_VERSION,
                "backend": "fuse", "mount_dir": str(self.mount), "recovery_dir": str(self.cache),
                "stopped_manifest": str(self.root / "stopped-files.json"),
                "pid": self.proc.pid if self.proc else None, "verified_at": self.verified_at}

    def _manifest(self) -> dict[str, str]:
        self.check()
        db = self.mount / "board.db"
        if not db.is_file():
            raise BucketStorageError("Stopped board.db is missing; cache retained.")
        with db.open("rb") as stream:
            if stream.read(16) != b"SQLite format 3\x00":
                raise BucketStorageError("Stopped board.db has an invalid SQLite header.")
        with contextlib.closing(sqlite3.connect(db, timeout=10)) as conn:
            busy, _, _ = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy or conn.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise BucketStorageError("Stopped board failed checkpoint or integrity_check.")
        # All SQLite connections are closed and the WAL is checkpointed. hf-mount 0.13.2
        # can leave uploaded sidecars behind after local unlink; publish empty sidecars
        # through the mount so a cold reopen cannot replay an obsolete remote WAL/index.
        for suffix in ("-wal", "-shm"):
            with (self.mount / ("board.db" + suffix)).open("wb") as stream:
                os.fsync(stream.fileno())
        return {p.relative_to(self.mount).as_posix(): _digest(p)
                for p in self.mount.rglob("*") if p.is_file()
                and not (p.name.endswith((".lock", "-shm", "-wal"))
                         or p.name.startswith(".flock-write-check-"))}

    def _detach_failed_mount(self) -> None:
        entry = _mount_entry(self.mount)
        if entry is None:
            return
        if (entry[0] != self._mount_id or entry[1] not in {"fuse", "fuse.hf-mount"}
                or entry[2] != "hf-mount"):
            raise BucketStorageError("Refusing to detach an unrecognized or replaced mount.")
        result = subprocess.run(["fusermount3", "-uz", str(self.mount)],
                                stdin=subprocess.DEVNULL, capture_output=True,
                                env=self._env(), timeout=10, check=False)
        if result.returncode or _mount_entry(self.mount) is not None:
            raise BucketStorageError(f"Could not detach the failed owned mount at {self.mount}.")

    def _drain(self) -> None:
        try:
            if self.proc is not None:
                if self.proc.poll() is not None:
                    raise BucketStorageError("Bucket mount exited before its shutdown drain.")
                self.proc.terminate()
                try:
                    rc = self.proc.wait(timeout=self.shutdown_timeout + 5)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
                    raise BucketStorageError("Bucket mount shutdown drain timed out.") from None
                if rc:
                    raise BucketStorageError(f"Bucket mount shutdown failed; see {self.log_path}")
            if _mount_entry(self.mount) is not None:
                raise BucketStorageError(f"Owned mount is still active at {self.mount}; cache retained.")
        finally:
            # A dead FUSE server can leave a disconnected mount; detach only our mount ID.
            if self.proc is not None and self.proc.poll() is not None:
                self._detach_failed_mount()

    def _verify(self, manifest: dict[str, str]) -> None:
        # A fresh download through Hub/Xet, outside the mount's metadata and block caches.
        dest = self.root / "verify"
        dest.mkdir(mode=0o700)
        files = [(f"{self.prefix}/{name}", dest / name) for name in manifest]
        try:
            self._api.download_bucket_files(self.config.bucket, files,
                                            raise_on_missing_files=True)
            for name in manifest:
                (dest / name).chmod(0o600)
            if any(_digest(dest / name) != digest for name, digest in manifest.items()):
                raise BucketStorageError("Remote board bytes do not match the stopped board.")
            wal = self._api.get_bucket_paths_info(
                self.config.bucket, [f"{self.prefix}/board.db-wal"])
            if any(getattr(p, "size", 0) for p in wal):
                raise BucketStorageError("Remote board still has a nonempty WAL; persistence unverified.")
            with contextlib.closing(sqlite3.connect(
                (dest / "board.db").as_uri() + "?mode=ro", uri=True
            )) as conn:
                if conn.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                    raise BucketStorageError("Downloaded board failed integrity_check.")
        except BucketStorageError:
            raise
        except Exception:  # noqa: BLE001 — redact upstream errors that may contain credentials
            raise BucketStorageError("Could not verify stopped board through the Hub API.") from None
        self.verified_at = datetime.now(UTC).isoformat()

    def stop(self, *, verify: bool) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            try:
                manifest = self._manifest() if verify else None
                if manifest is not None:
                    path = self.root / "stopped-files.json"
                    path.write_text(json.dumps(manifest, indent=2) + "\n")
                    path.chmod(0o600)
            finally:
                # A failed startup may already have exited without ever mounting.
                if verify or (self.proc and self.proc.poll() is None) or _mount_entry(self.mount):
                    self._drain()
            if manifest is not None:
                self._verify(manifest)
                # Never traverse/remove an active or replaced mount.
                if _mount_entry(self.mount) is None:
                    shutil.rmtree(self.root)
        except sqlite3.Error:
            raise BucketStorageError("Could not checkpoint the stopped board; cache retained.") from None
        finally:
            # Also remove the temporary credential on startup/verification failure.
            self.token_file.unlink(missing_ok=True)
            self._token = None
            self._api = None


def _supervise(parent: int, token_file: Path, timeout: float, args: list[str]) -> int:
    """Forward shutdown and drain even if the Flock controller disappears."""
    done = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: done.set())
    try:
        child = subprocess.Popen(args, stdin=subprocess.DEVNULL)
        while child.poll() is None and os.getppid() == parent and not done.wait(0.2):
            pass
        if child.poll() is None:
            child.terminate()
            try:
                return child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
                return 1
        return child.returncode
    finally:
        token_file.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(_supervise(int(sys.argv[1]), Path(sys.argv[2]), float(sys.argv[3]), sys.argv[4:]))
