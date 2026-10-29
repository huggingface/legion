"""A private msgboard per local run. No package installs or remote services at startup.

Live SQLite files stay on node-local disk; bucket storage uploads consistent backups.
Local snapshots or bucket location metadata live in the run's private board/ directory.
The child inherits a pre-bound socket (no port selection race),
not the controller's credentials, and exits when its parent disappears.
"""

from __future__ import annotations

import contextlib
import errno
import importlib.util
import json
import logging
import os
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from flock.comms.bucket import BucketMount, BucketStorageError
from flock.models import BucketBoardStorage

if TYPE_CHECKING:
    from flock.models import CommsSpec, SwarmSpec

log = logging.getLogger("flock.board")
INVITE_ENV = "FLOCK_MANAGED_BOARD_INVITE"
SNAPSHOT_INTERVAL = 15.0
DB_NAME = "board.db"
TOKEN_FILE = "operator.token"
INSTALL_HINT = "Managed boards need msgboard: install the 'msgboard' extra (uv sync --extra msgboard)."


class ManagedBoardError(ValueError):
    """A setup/lifecycle error safe to show without exposing board credentials."""


def _signature(comms: CommsSpec) -> tuple:
    return (comms.channel, tuple(sorted(set(comms.channels))), comms.allow_create_channels,
            comms.storage.model_dump_json())


class ManagedBoard:
    def __init__(self, run_dir: Path, *, startup_timeout: float = 60.0) -> None:
        self.run_dir = Path(run_dir).resolve()
        self.archive = self.run_dir / "board"
        self.log_path = self.run_dir / "logs" / "msgboard.log"
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None
        self.local: Path | None = None
        self.bucket: BucketMount | None = None
        self.url: str | None = None
        self.invite: str | None = None
        self._operator: str | None = None
        self._signature: tuple | None = None
        self._swarm: str | None = None
        self._details: dict = {}
        self._file_lock = None
        self._lock = threading.RLock()
        self._closed = False
        self._snapshot_at: str | None = None
        self._cancel_start = threading.Event()
        self._viewer = None

    def _acquire(self) -> None:
        import fcntl

        self.archive.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.archive.chmod(0o700)
        self._file_lock = (self.archive / ".lock").open("a+b")
        os.chmod(self.archive / ".lock", 0o600)
        try:
            fcntl.flock(self._file_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file_lock.close()
            self._file_lock = None
            raise ManagedBoardError("This run's board is already running.") from None

    def start(self, spec: SwarmSpec | None = None, *, restore: bool = False) -> None:
        """Start and provision once, or reopen a snapshot without changing its identities."""
        if importlib.util.find_spec("msgboard") is None:
            raise ManagedBoardError(INSTALL_HINT)
        retained = {}
        if not restore and ((self.archive / "metadata.json").exists()
                            or (self.archive / DB_NAME).exists()):
            raise ManagedBoardError("Refusing to overwrite an existing board; use flock board.")
        if restore:
            meta = self.archive / "metadata.json"
            if meta.is_file():
                retained = json.loads(meta.read_text())
            if (retained.get("storage", {}).get("kind") != "hf-bucket"
                    and not (self.archive / DB_NAME).is_file()):
                raise ManagedBoardError(f"No retained board in {self.archive}")
        with self._lock:
            self._acquire()
            try:
                storage = (BucketBoardStorage.model_validate({
                    k: v for k, v in retained["storage"].items()
                    if k in BucketBoardStorage.model_fields
                }) if retained.get("storage", {}).get("kind") == "hf-bucket"
                    else spec.comms.storage if spec and spec.comms else None)
                if isinstance(storage, BucketBoardStorage):
                    self._check_local(Path(storage.mount_root))  # cache/staging must be local
                    self.bucket = BucketMount(
                        storage, spec.swarm if spec else "retained",
                        self.run_dir / "logs" / "bucket-storage.log",
                        prefix=retained.get("storage", {}).get("prefix"))
                    self._save_metadata()  # retain the exact remote prefix even if setup fails
                    self.local = self.bucket.start(self._cancel_start, self.startup_timeout,
                                                   restore=restore)
                    self._check_local(self.local)
                else:
                    root = Path(os.environ.get("FLOCK_BOARD_LOCAL_ROOT", "/tmp"))
                    self._check_local(root)
                    root.mkdir(parents=True, exist_ok=True)
                    self.local = Path(tempfile.mkdtemp(prefix="flock-board-", dir=root))
                    if restore:
                        for p in self.archive.iterdir():
                            if p.name.startswith(".") or p.name == "metadata.json":
                                continue
                            dest = self.local / p.name
                            if p.is_dir():
                                shutil.copytree(p, dest)
                            else:
                                shutil.copy2(p, dest)
                if restore:
                    self._details = retained.get("configuration", {})
                elif (self.archive / DB_NAME).exists():
                    raise ManagedBoardError("Refusing to overwrite an existing board; use flock board.")
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("ab") as out:
                    init_command = ([sys.executable, "-m", "flock.comms.managed",
                                     "--init-bucket", str(self.local)] if self.bucket else
                                    [sys.executable, "-m", "msgboard", "init", "--data-dir",
                                     str(self.local)])
                    result = subprocess.run(
                        init_command,
                        stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                        env=self._env(), timeout=self.startup_timeout, check=False,
                    )
                    if result.returncode:
                        raise ManagedBoardError(f"msgboard init failed; see {self.log_path}")
                    self._operator = (self.local / TOKEN_FILE).read_text().strip()
                    with socket.socket() as sock:
                        sock.bind(("127.0.0.1", 0))
                        sock.listen(128)
                        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
                        self.proc = subprocess.Popen(
                            [sys.executable, "-m", "flock.comms.managed", str(self.local),
                             str(sock.fileno()), str(os.getpid())],
                            pass_fds=(sock.fileno(),), start_new_session=True,
                            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                            env=self._env(),
                        )
                self._wait_ready()
                if not restore:
                    assert spec is not None and spec.comms is not None
                    self._provision(spec)
                self.snapshot()
                if self.bucket:
                    self.bucket.start_snapshots()
                    try:
                        from flock.board_viewer.export import Publisher
                        self._viewer = Publisher(self.bucket, self.local)
                        self._viewer.start()
                    except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                        self._viewer = None
                        log.warning("Board viewer publisher could not start")
            except BaseException as exc:
                try:
                    self._terminate()
                    if self.bucket:
                        try:
                            self.bucket.stop(verify=False)
                        except BucketStorageError:
                            log.exception("Bucket startup cleanup failed; cache retained")
                        self._save_metadata()
                finally:
                    self._release(remove_local=True)
                if isinstance(exc, BucketStorageError):
                    raise ManagedBoardError(str(exc)) from None
                raise

    @staticmethod
    def _check_local(root: Path) -> None:
        """Refuse known shared filesystems; SQLite WAL must stay on a local filesystem."""
        mounts = Path("/proc/mounts")
        if not mounts.exists():
            return
        target = str(root.resolve())
        candidates = []
        for line in mounts.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3:
                mount = parts[1].replace("\\040", " ")
                if target == mount or target.startswith(mount.rstrip("/") + "/"):
                    candidates.append((len(mount), parts[2]))
        if candidates:
            fs = max(candidates)[1]
            if fs.startswith("fuse") or fs in {"nfs", "nfs4", "wekafs", "lustre", "cifs", "smb3", "9p"}:
                raise ManagedBoardError("FLOCK_BOARD_LOCAL_ROOT must be on node-local disk.")

    @staticmethod
    def _env() -> dict[str, str]:
        return {k: v for k, v in os.environ.items()
                if k in {"PATH", "HOME", "LANG", "LANGUAGE", "TZ", "SYSTEMROOT"}
                or k.startswith("LC_")}

    def cancel_start(self) -> None:
        """Signals may interrupt readiness without stopping a board used by live workers."""
        self._cancel_start.set()

    @property
    def periodic_snapshots(self) -> bool:
        return self.bucket is None

    @property
    def operator_token_path(self) -> Path:
        return (self.local if self.bucket and self.local else self.archive) / TOKEN_FILE

    def check(self) -> None:
        if self.bucket:
            try:
                self.bucket.check()
            except BucketStorageError as exc:
                raise ManagedBoardError(str(exc)) from None
        if self.proc is None:
            raise ManagedBoardError(f"Managed msgboard is not started; see {self.log_path}")
        if (code := self.proc.poll()) is not None:
            cause = f"exit code {code}"
            if code < 0:
                with contextlib.suppress(ValueError):
                    cause = signal.Signals(-code).name
            raise ManagedBoardError(f"Managed msgboard exited ({cause}); see {self.log_path}")

    def _wait_ready(self) -> None:
        import httpx

        deadline = time.monotonic() + self.startup_timeout
        with httpx.Client(trust_env=False, timeout=0.5) as client:
            while time.monotonic() < deadline:
                if self._cancel_start.is_set():
                    raise ManagedBoardError("Managed board startup interrupted.")
                self.check()
                try:
                    r = client.get(f"{self.url}/v1/me",
                                   headers={"Authorization": f"Bearer {self._operator}"})
                    if r.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                time.sleep(0.05)
        raise ManagedBoardError(f"Managed msgboard readiness timed out; see {self.log_path}")

    def _provision(self, spec: SwarmSpec) -> None:
        import httpx

        c = spec.comms
        assert c is not None
        self._signature = _signature(c)
        self._swarm = spec.swarm
        names = list(dict.fromkeys([c.channel, *c.channels]))
        self._details = {"channel": c.channel, "channels": names,
                         "allow_create_channels": c.allow_create_channels}
        with httpx.Client(base_url=self.url, trust_env=False, timeout=10,
                          headers={"Authorization": f"Bearer {self._operator}"}) as client:
            for name in names:
                r = client.post("/v1/channels", json={"name": name})
                if r.status_code >= 400:
                    raise ManagedBoardError("Could not create the managed board's channels.")
            r = client.post("/v1/invites", json={
                "kind": "agent", "role": "member" if c.allow_create_channels else "guest",
                "can_create_channels": c.allow_create_channels, "channels": names,
                "join_open_channels": False, "max_uses": None, "expires_in_seconds": None,
                "label": f"flock {self.run_dir.name}"[:80],
            })
            if r.status_code >= 400:
                raise ManagedBoardError("Could not create the managed board's agent invite.")
            self.invite = r.json()["code"]

    def resolve(self, spec: SwarmSpec) -> None:
        """Bind an apply/update to this run's board, never create a replacement board."""
        c = spec.comms
        if (c is None or not c.managed or spec.swarm != self._swarm
                or _signature(c) != self._signature):
            raise ManagedBoardError("Managed board mode, channel, channels and creation permissions "
                                    "and storage cannot change during a run; start a new run instead.")
        self.check()
        c.url, c.invite_env = self.url, INVITE_ENV

    def metadata(self) -> dict:
        return {"managed": True, "url": self.url,
                "pid": self.proc.pid if self.proc is not None else None,
                "exit_code": self.proc.poll() if self.proc is not None else None,
                "running": self.proc is not None and self.proc.poll() is None,
                "configuration": self._details, "snapshot_at": self._snapshot_at,
                "operator_token_file": str(self.operator_token_path),
                "data_dir": (f"hf://buckets/{self.bucket.config.bucket}/{self.bucket.prefix}"
                             if self.bucket else str(self.archive)),
                "storage": self.bucket.metadata() if self.bucket else {"kind": "local"},
                "log": "logs/msgboard.log"}

    def _save_metadata(self) -> None:
        meta = self.archive / ".metadata.tmp"
        meta.write_text(json.dumps(self.metadata(), indent=2) + "\n")
        meta.chmod(0o600)
        os.replace(meta, self.archive / "metadata.json")

    def snapshot(self) -> None:
        """Atomic online SQLite backup; copy attachments and credentials before publishing it."""
        with self._lock:
            if self.bucket:
                self.bucket.snapshot()
                self._snapshot_at = self.bucket.snapshot_at
                self._save_metadata()
                return
            if self.local is None or not (self.local / DB_NAME).is_file():
                return
            tmp = self.archive / ".board.snapshot"
            tmp.touch(mode=0o600, exist_ok=True)
            try:
                with contextlib.closing(sqlite3.connect(
                    (self.local / DB_NAME).as_uri() + "?mode=ro", uri=True, timeout=5
                )) as src, contextlib.closing(sqlite3.connect(tmp)) as dst:
                    src.backup(dst)
                for p in self.local.iterdir():
                    if p.name.startswith(DB_NAME) or "lock" in p.name:
                        continue
                    dest = self.archive / p.name
                    if p.is_dir():
                        shutil.copytree(p, dest, dirs_exist_ok=True)
                    elif p.is_file():
                        copied = self.archive / (".copy-" + p.name)
                        shutil.copy2(p, copied)
                        copied.chmod(0o600)
                        os.replace(copied, dest)
                os.replace(tmp, self.archive / DB_NAME)
                self._snapshot_at = datetime.now(UTC).isoformat()
                self._save_metadata()
            finally:
                tmp.unlink(missing_ok=True)

    def _terminate(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait(timeout=5)

    def _release(self, *, remove_local: bool) -> None:
        if self.bucket is None and self.local is not None and remove_local:
            shutil.rmtree(self.local, ignore_errors=True)
            self.local = None
        if self._file_lock is not None:
            self._file_lock.close()
            self._file_lock = None

    def stop(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._viewer and not self._viewer.stop():
                self._viewer = None
            self._terminate()
            saved = False
            final_export = None
            try:
                if self.bucket:
                    if self._viewer:
                        try:
                            from flock.board_viewer.export import capture
                            final_export = tempfile.TemporaryDirectory(prefix="flock-viewer-final-")
                            capture(self.local, Path(final_export.name))
                        except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                            if final_export:
                                final_export.cleanup()
                            final_export = None
                            log.warning("Final viewer snapshot failed; retaining previous export")
                    self.bucket.stop(verify=True)
                    if final_export:
                        try:
                            from flock.board_viewer.export import publish
                            publish(self._viewer.api, self._viewer.bucket, self._viewer.prefix,
                                    Path(final_export.name), completed=True)
                        except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                            log.warning("Final viewer upload failed; retaining previous export")
                else:
                    self.snapshot()
                saved = True
            except BucketStorageError as exc:
                raise ManagedBoardError(str(exc)) from None
            finally:
                if final_export:
                    final_export.cleanup()
                if not saved:
                    log.error("Board persistence failed; live data retained at %s", self.local)
                try:
                    if self.bucket:
                        self._save_metadata()
                finally:
                    self._release(remove_local=saved)


def reopen(run_dir: Path) -> int:
    """Serve retained history until interrupted. No agents or new invites are created."""
    board = ManagedBoard(run_dir)
    done = threading.Event()
    handlers = {}
    def on_signal(*_):
        done.set()
        board.cancel_start()

    for sig in (signal.SIGINT, signal.SIGTERM):
        handlers[sig] = signal.signal(sig, on_signal)
    try:
        board.start(restore=True)
        print(f"board: {board.url}; operator token: {board.operator_token_path}", flush=True)
        next_snapshot = time.monotonic() + SNAPSHOT_INTERVAL
        while not done.wait(1):
            board.check()
            if board.periodic_snapshots and time.monotonic() >= next_snapshot:
                board.snapshot()
                next_snapshot = time.monotonic() + SNAPSHOT_INTERVAL
        return 0
    finally:
        try:
            board.stop()
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig, handler)


def _init_bucket(data_dir: str) -> int:
    """Adapt only token publication: bucket FUSE cannot create msgboard's hard link."""
    from types import SimpleNamespace

    from msgboard import cli, db, store

    root = Path(data_dir)
    target, temporary = root / TOKEN_FILE, root / (TOKEN_FILE + ".tmp")
    try:
        result = cli.cmd_init(SimpleNamespace(data_dir=data_dir))
    except OSError as exc:
        if (exc.errno != errno.ENOTSUP or exc.filename != str(temporary)
                or exc.filename2 != str(target)):
            raise
        result = 0
    if result or target.exists():
        return result
    # cmd_init may also leave this state when reopening after interrupted publication.
    secret = temporary.read_text().strip()
    with contextlib.closing(db.connect(root / DB_NAME)) as conn:
        principal = store.authenticate(conn, secret, time.time())
        if (principal.issuer, principal.external_id, principal.role) != ("local", "operator", "admin"):
            raise ManagedBoardError("Temporary board token is not the initialized operator.")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        stream.write(secret + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.unlink()
    return 0


def _serve(data_dir: str, fd: int, parent: int) -> None:
    """Child process entry point: the socket belongs to us; the operator token stays on disk."""
    import faulthandler

    import uvicorn
    from msgboard.server import Board, create_app

    faulthandler.enable()  # Keep native SQLite crash traces in msgboard.log.
    board = Board(Path(data_dir))

    class Server(uvicorn.Server):
        def handle_exit(self, sig, frame):
            board.begin_shutdown()
            super().handle_exit(sig, frame)

    def on_exit(*_):
        raise SystemExit(0)

    def watch_parent() -> None:
        while os.getppid() == parent:
            time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)

    signal.signal(signal.SIGTERM, on_exit)
    signal.signal(signal.SIGINT, on_exit)
    threading.Thread(target=watch_parent, daemon=True).start()
    try:
        with socket.socket(fileno=fd) as sock:
            Server(uvicorn.Config(create_app(board), log_level="warning", access_log=False,
                                  timeout_graceful_shutdown=5)).run(sockets=[sock])
    finally:
        board.close()


if __name__ == "__main__":
    if sys.argv[1] == "--init-bucket":
        sys.exit(_init_bucket(sys.argv[2]))
    _serve(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
