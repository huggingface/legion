"""Bucket board lifecycle tests: fake mounts/Hub, real local SQLite and msgboard."""
from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from flock.comms import bucket, managed
from flock.comms.bucket import BucketMount, BucketStorageError
from flock.comms.managed import ManagedBoard, ManagedBoardError
from flock.models import BucketBoardStorage, CommsSpec

from .test_managed_board import join, spec


class Process:
    pid = 12345

    def __init__(self):
        self.returncode = None
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode


class Hub:
    def __init__(self, token=None):
        self.private = True
        self.remote = {}
        self.downloads = 0

    def bucket_info(self, bucket_id):
        return SimpleNamespace(private=self.private)

    def get_bucket_paths_info(self, bucket_id, paths):
        return [SimpleNamespace(path=p, size=len(self.remote[p]))
                for p in paths if p in self.remote]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files):
        self.downloads += 1
        for remote, local in files:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(self.remote[remote])


@pytest.fixture
def mount(tmp_path):
    return BucketMount(BucketBoardStorage(bucket="team/private", mount_root=str(tmp_path)),
                       "test-swarm", tmp_path / "logs" / "mount.log")


@pytest.fixture
def setup_mount(mount, monkeypatch):
    import huggingface_hub

    hub = Hub()
    process = Process()
    captured = {}
    monkeypatch.setenv("HF_TOKEN", "fake-secret-never-in-argv")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: hub)
    monkeypatch.setattr(shutil, "which", lambda _: "/tools/hf-mount-nfs")
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
        returncode=0, stdout="hf-mount 0.13.2\n"))

    def popen(args, **kw):
        captured.update(args=args, **kw)
        return process

    monkeypatch.setattr(subprocess, "Popen", popen)
    monkeypatch.setattr(bucket, "_mount_entry", lambda p: ("101", "nfs", "127.0.0.1:/")
                        if process.poll() is None and mount.mount.exists() else None)
    monkeypatch.setattr(mount, "_check_writable", lambda *a: None)
    return mount, hub, process, captured


def test_storage_defaults_and_roundtrip():
    assert CommsSpec(managed=True).storage.kind == "local"
    c = CommsSpec(managed=True, storage={"kind": "hf-bucket", "bucket": "team/private"})
    assert CommsSpec.model_validate_json(c.model_dump_json()) == c
    assert c.storage.flush_debounce == "2s"
    with pytest.raises(ValidationError, match="managed"):
        CommsSpec(url="http://board", invite_env="INVITE", storage=c.storage)


@pytest.mark.parametrize("fields", [
    {"bucket": "missing-namespace"}, {"bucket": "team/bucket/../../other"},
    {"bucket": "team/."}, {"bucket": "team/private", "unknown": True},
    {"mount_root": "relative"}, {"mount_root": "/scratch/../tmp"},
    {"token_env": "bad-name"}, {"flush_debounce": "0s"},
    {"flush_shutdown_timeout": "2h"}, {"flush_max_batch_window": "1s"},
])
def test_invalid_storage(fields):
    with pytest.raises(ValidationError):
        BucketBoardStorage(**{"bucket": "team/private", **fields})


def test_unique_prefix_and_mount_per_instance(mount):
    other = BucketMount(mount.config, "test-swarm", mount.log_path)
    assert mount.prefix != other.prefix and mount.mount != other.mount
    reopened = BucketMount(mount.config, "unused", mount.log_path, prefix=mount.prefix)
    assert reopened.prefix == mount.prefix and reopened.cache != mount.cache


@pytest.mark.parametrize("prefix", ["../escape", "/flock/runs/a/abc/board",
                                    "flock/runs/../" + "a" * 32 + "/board",
                                    "flock/runs/a/" + "a" * 32 + "/board/extra"])
def test_retained_prefix_cannot_escape(mount, prefix):
    with pytest.raises(BucketStorageError, match="prefix"):
        BucketMount(mount.config, "test", mount.log_path, prefix=prefix)


def test_start_passes_flush_controls_and_private_token_file(setup_mount):
    m, _, _, captured = setup_mount
    assert m.start(threading.Event(), 1) == m.mount
    args = captured["args"]
    assert args[args.index("--flush-debounce-ms") + 1] == "2000"
    assert args[args.index("--flush-max-batch-window-ms") + 1] == "15000"
    assert args[args.index("--flush-shutdown-timeout-ms") + 1] == "45000"
    assert args[-2] == f"team/private/{m.prefix}"
    assert m.token_file.parent != m.mount and m.token_file.stat().st_mode & 0o777 == 0o600
    assert m.root.stat().st_mode & 0o777 == 0o700
    assert "fake-secret-never-in-argv" not in json.dumps(args)
    assert "HF_TOKEN" not in captured["env"]
    assert "fake-secret-never-in-argv" not in json.dumps(m.metadata())


@pytest.mark.parametrize("failure", ["missing", "version", "token", "public", "exited",
                                     "unexpected", "cancelled"])
def test_start_fails_without_writing_a_database(setup_mount, monkeypatch, failure):
    m, hub, process, _ = setup_mount
    cancel = threading.Event()
    if failure == "missing":
        monkeypatch.setattr(shutil, "which", lambda _: None)
    elif failure == "version":
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: SimpleNamespace(
            returncode=0, stdout="hf-mount 0.13.2-dev"))
    elif failure == "token":
        monkeypatch.delenv("HF_TOKEN")
    elif failure == "public":
        hub.private = False
    elif failure == "exited":
        process.returncode = 1
    elif failure == "unexpected":
        monkeypatch.setattr(bucket, "_mount_entry", lambda _: ("101", "ext4", "/dev/sda"))
    elif failure == "cancelled":
        cancel.set()
    with pytest.raises(BucketStorageError):
        m.start(cancel, 0.1)
    assert not (m.mount / "board.db").exists()
    try:
        m.stop(verify=False)
    except BucketStorageError:
        pass
    assert not m.token_file.exists() and m.verified_at is None


def test_mount_disappearance_and_replacement_fail_health_check(setup_mount, monkeypatch):
    m, _, _, _ = setup_mount
    m.start(threading.Event(), 1)
    monkeypatch.setattr(bucket, "_mount_entry", lambda _: ("999", "nfs", "127.0.0.1:/"))
    with pytest.raises(BucketStorageError, match="disappeared"):
        m.check()
    monkeypatch.setattr(bucket, "_mount_entry", lambda _: None)
    with pytest.raises(BucketStorageError):
        m.check()


def test_preflight_verifies_upload_through_hub(setup_mount, monkeypatch):
    m, hub, _, _ = setup_mount
    m.start(threading.Event(), 1)

    def find(bucket_id, paths):
        hub.remote[paths[0]] = (m.mount / paths[0].split("/")[-1]).read_bytes()
        return [SimpleNamespace(size=32)]

    monkeypatch.setattr(hub, "get_bucket_paths_info", find)
    BucketMount._check_writable(m, threading.Event(), time.monotonic() + 1)
    assert hub.downloads == 1
    assert not list(m.mount.glob(".flock-write-check-*"))


def test_preflight_rejects_unuploaded_writes(setup_mount):
    m, _, _, _ = setup_mount
    m.start(threading.Event(), 1)
    with pytest.raises(BucketStorageError, match="write-access"):
        BucketMount._check_writable(m, threading.Event(), time.monotonic())


@pytest.fixture
def stopped_board(mount, monkeypatch):
    m = mount
    m.mount.mkdir(parents=True, mode=0o700)
    m.cache.mkdir()
    m.token_file.write_text("secret")
    with sqlite3.connect(m.mount / "board.db") as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE messages(body TEXT)")
        db.execute("INSERT INTO messages VALUES ('retained')")
    (m.mount / "operator.token").write_text("operator-secret")
    (m.mount / "server.lock").touch()
    m.proc = Process()
    m._api = Hub()
    monkeypatch.setattr(m, "check", lambda: None)
    monkeypatch.setattr(bucket, "_mount_entry", lambda _: None)
    hub = m._api

    def drain():
        hub.remote = {f"{m.prefix}/{p.name}": p.read_bytes() for p in m.mount.iterdir()
                      if p.is_file()}

    monkeypatch.setattr(m, "_drain", drain)
    return m, hub


def test_shutdown_verifies_remote_bytes_then_removes_cache(stopped_board):
    m, hub = stopped_board
    m.stop(verify=True)
    assert m.verified_at and not m.root.exists() and hub.downloads == 1
    assert f"{m.prefix}/operator.token" in hub.remote
    assert m._api is None and m._token is None


@pytest.mark.parametrize("failure", ["stale", "wal", "missing", "offline", "checkpoint"])
def test_failed_persistence_retains_cache_and_never_marks_verified(stopped_board,
                                                                 monkeypatch, failure):
    m, hub = stopped_board
    original = m._drain

    def drain():
        original()
        if failure == "stale":
            hub.remote[f"{m.prefix}/board.db"] = b"stale DB"
        elif failure == "wal":
            hub.remote[f"{m.prefix}/board.db-wal"] = b"uncheckpointed"
        elif failure == "missing":
            del hub.remote[f"{m.prefix}/operator.token"]
        elif failure == "offline":
            def offline(*a, **kw):
                raise OSError("network down with secret=do-not-display")
            hub.download_bucket_files = offline

    monkeypatch.setattr(m, "_drain", drain)
    if failure == "checkpoint":
        (m.mount / "board.db").write_bytes(b"corrupt")
    with pytest.raises((BucketStorageError, sqlite3.DatabaseError)) as exc:
        m.stop(verify=True)
    assert "do-not-display" not in str(exc.value)
    assert m.root.exists() and m.cache.exists() and not m.token_file.exists()
    assert m.verified_at is None and m._api is None


def test_drain_timeout_kills_sidecar_retains_cache(stopped_board, monkeypatch):
    m, _ = stopped_board
    calls = []

    def wait(timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise subprocess.TimeoutExpired("mount", timeout)
        return -9

    monkeypatch.setattr(m.proc, "wait", wait)
    monkeypatch.setattr(m, "_drain", lambda: BucketMount._drain(m))
    with pytest.raises(BucketStorageError, match="timed out"):
        m.stop(verify=True)
    assert m.proc.killed and m.cache.exists() and not m.token_file.exists()


@pytest.fixture
def fake_mounts(tmp_path, monkeypatch):
    """Real msgboard on local disk, standing in for the unprivileged NFS boundary."""
    pytest.importorskip("msgboard")
    remote = {}
    instances = []
    hub = Hub()
    monkeypatch.setattr(bucket, "_mount_entry", lambda _: None)

    class Mount(BucketMount):
        def start(self, cancel, timeout, *, restore=False):
            self.mount.mkdir(parents=True)
            self.cache.mkdir()
            self._api = hub
            if restore:
                for name, data in remote[self.prefix].items():
                    (self.mount / name).write_bytes(data)
            instances.append(self)
            return self.mount

        def check(self):
            pass

        def _drain(self):
            data = {p.name: p.read_bytes() for p in self.mount.iterdir() if p.is_file()}
            remote[self.prefix] = data
            hub.remote.update({f"{self.prefix}/{k}": v for k, v in data.items()})

    monkeypatch.setattr(managed, "BucketMount", Mount)
    return instances


def test_managed_bucket_skips_sqlite_backups_and_reopens_prefix(tmp_path, fake_mounts, monkeypatch):
    cfg = {"kind": "hf-bucket", "bucket": "team/private", "mount_root": str(tmp_path / "scratch")}
    b = ManagedBoard(tmp_path / "run")
    b.start(spec(storage=cfg))
    try:
        assert not b.periodic_snapshots
        with join(b) as client:
            channel = client.get("/v1/channels").json()["channels"][0]["id"]
            response = client.post(f"/v1/channels/{channel}/messages", json={"body": "retained"},
                                   headers={"Idempotency-Key": "one"})
            response.raise_for_status()
            credentials = dict(client.headers)
        b.snapshot()
        assert not (b.archive / "board.db").exists()
        with pytest.raises(ManagedBoardError, match="storage"):
            b.resolve(spec(storage={**cfg, "flush_debounce": "3s"}))
        b.resolve(spec(storage=cfg))
        original_terminate = b._terminate

        def terminate():
            original_terminate()
            assert b.proc.poll() is not None  # close/checkpoint DB before mount drain
        monkeypatch.setattr(b, "_terminate", terminate)
    finally:
        b.stop()
    meta = json.loads((b.archive / "metadata.json").read_text())
    assert meta["storage"]["verified_at"] and meta["snapshot_at"] is None
    reopened = ManagedBoard(b.run_dir)
    try:
        reopened.start(restore=True)
        assert fake_mounts[0].prefix == fake_mounts[1].prefix
        assert fake_mounts[0].cache != fake_mounts[1].cache
        with httpx.Client(base_url=reopened.url, headers=credentials) as client:
            messages = client.get(f"/v1/channels/{channel}/messages").json()["messages"]
            assert [m["body"] for m in messages] == ["retained"]
    finally:
        reopened.stop()


def test_managed_mount_failure_never_initializes_fallback_db(tmp_path, monkeypatch):
    pytest.importorskip("msgboard")
    def fail(*a, **kw):
        raise BucketStorageError("mount denied")
    monkeypatch.setattr(BucketMount, "start", fail)
    b = ManagedBoard(tmp_path / "run")
    with pytest.raises(ManagedBoardError, match="mount denied"):
        b.start(spec(storage={"kind": "hf-bucket", "bucket": "team/private",
                              "mount_root": str(tmp_path / "scratch")}))
    b.stop()
    assert b.local is None and b.proc is None
    assert not (b.archive / "board.db").exists()
    assert b.metadata()["storage"]["verified_at"] is None


@pytest.mark.parametrize("failed", [False, True])
async def test_local_run_persistence_status_and_worker_credentials(tmp_path, fake_mounts,
                                                                 monkeypatch, failed):
    import io

    from flock.controller import localrun

    monkeypatch.setenv("HF_TOKEN", "mount-only-token")
    monkeypatch.setattr(localrun, "SNAPSHOT_INTERVAL", 0.01)
    jobs = []
    snapshots = []
    factory = localrun.LocalJobsClient

    def capture_jobs(**kwargs):
        value = factory(**kwargs)
        jobs.append(value)
        return value

    class Board(ManagedBoard):
        def snapshot(self):
            snapshots.append(self)
            super().snapshot()

    if failed:
        def fail(*a):
            raise BucketStorageError("upload incomplete")
        monkeypatch.setattr(BucketMount, "_verify", fail)
    monkeypatch.setattr(localrun, "LocalJobsClient", capture_jobs)
    monkeypatch.setattr(localrun, "ManagedBoard", Board)
    result = await localrun.run_local(localrun.RunOptions(
        spec=spec(storage={"kind": "hf-bucket", "bucket": "team/private",
                           "mount_root": str(tmp_path / "scratch")}),
        data_dir=tmp_path / "runs", until=0.2, refresh=0.05, grace=0, out=io.StringIO()))
    doc = json.loads((result.run_dir / "run.json").read_text())
    assert result.exit_code == (1 if failed else 0)
    assert (doc["board"]["storage"]["verified_at"] is None) == failed
    assert result.reason == ("board_persistence_failed" if failed else "until")
    assert len(snapshots) == 1  # initial metadata only; no periodic/final SQLite backups
    assert "HF_TOKEN" not in jobs[0].process_env({}, {})
    assert jobs[0].process_env({}, {"HF_TOKEN": "provider-key"})["HF_TOKEN"] == "provider-key"
    assert "mount-only-token" not in json.dumps(doc)


def test_fresh_start_preserves_existing_bucket_metadata(tmp_path):
    pytest.importorskip("msgboard")
    b = ManagedBoard(tmp_path / "run")
    b.archive.mkdir(parents=True)
    saved = '{"storage": {"kind": "hf-bucket", "prefix": "keep-me"}}'
    (b.archive / "metadata.json").write_text(saved)
    with pytest.raises(ManagedBoardError, match="overwrite"):
        b.start(spec(storage={"kind": "hf-bucket", "bucket": "team/private"}))
    assert (b.archive / "metadata.json").read_text() == saved


@pytest.mark.parametrize("spawn_failed", [False, True])
def test_mount_supervisor_drains_on_parent_death_and_removes_token(tmp_path, monkeypatch,
                                                                  spawn_failed):
    token = tmp_path / "hf.token"
    token.write_text("private")
    process = Process()
    monkeypatch.setattr(bucket.os, "getppid", lambda: 99)
    monkeypatch.setattr(bucket.signal, "signal", lambda *args: None)
    if spawn_failed:
        def fail(*args, **kw):
            raise OSError("cannot spawn")
        monkeypatch.setattr(subprocess, "Popen", fail)
        with pytest.raises(OSError):
            bucket._supervise(100, token, 1, ["hf-mount-nfs"])
    else:
        monkeypatch.setattr(subprocess, "Popen", lambda *args, **kw: process)
        assert bucket._supervise(100, token, 1, ["hf-mount-nfs"]) == 0
        assert process.terminated
    assert not token.exists()


def test_mountinfo_rejects_overmount_and_decodes_spaces(tmp_path, monkeypatch):
    original = Path.read_text
    lines = "101 1 0:1 / /scratch/my\\040mount rw - nfs 127.0.0.1:/ rw\n"
    def read(self, *a, **kw):
        return lines if str(self) == "/proc/self/mountinfo" else original(self, *a, **kw)
    monkeypatch.setattr(Path, "read_text", read)
    assert bucket._mount_entry(Path("/scratch/my mount")) == ("101", "nfs", "127.0.0.1:/")
    lines += "102 1 0:2 / /scratch/my\\040mount rw - ext4 /dev/disk rw\n"
    with pytest.raises(BucketStorageError, match="Multiple mounts"):
        bucket._mount_entry(Path("/scratch/my mount"))


def test_live_check_script_clean_path_with_fake_mount(tmp_path, fake_mounts, monkeypatch, capsys):
    from scripts import check_bucket_board

    monkeypatch.setattr("sys.argv", ["check_bucket_board.py", "--bucket", "team/private",
                                   "--mount-root", str(tmp_path / "scratch"),
                                   "--data-dir", str(tmp_path / "runs")])
    check_bucket_board.main()
    assert "PASS: concurrent reads/writes" in capsys.readouterr().out
    assert len(fake_mounts) == 2
