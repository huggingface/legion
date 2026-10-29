"""Real local SQLite and msgboard; fake Hub, no network or jobs."""
from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from flock.comms import managed
from flock.comms.bucket import BucketMount, BucketStorageError
from flock.comms.managed import ManagedBoard, ManagedBoardError
from flock.models import BucketBoardStorage, CommsSpec

from .test_managed_board import join, spec


class Hub:
    def __init__(self):
        self.private = True
        self.remote = {}
        self.fail = False

    def bucket_info(self, bucket_id):
        return SimpleNamespace(private=self.private)

    def batch_bucket_files(self, bucket_id, *, add=(), delete=()):
        if self.fail:
            raise OSError("private credential must never appear")
        for source, dest in add:
            self.remote[dest] = source if isinstance(source, bytes) else Path(source).read_bytes()
        for name in delete:
            self.remote.pop(name, None)

    def get_bucket_paths_info(self, bucket_id, paths):
        return iter(SimpleNamespace(path=p, size=len(self.remote[p]))
                    for p in paths if p in self.remote)

    def list_bucket_tree(self, bucket_id, prefix, recursive=True):
        return [SimpleNamespace(path=p, size=len(data)) for p, data in self.remote.items()
                if p.startswith(prefix)]

    def download_bucket_files(self, bucket_id, files, *, raise_on_missing_files):
        for remote, local in files:
            Path(local).parent.mkdir(parents=True, exist_ok=True)
            Path(local).write_bytes(self.remote[remote])


@pytest.fixture
def fake_mounts(monkeypatch):
    import huggingface_hub
    hub = Hub()
    monkeypatch.setenv("HF_TOKEN", "fake-secret")
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kw: hub)
    return hub


@pytest.fixture
def mount(tmp_path):
    return BucketMount(BucketBoardStorage(bucket="team/private", mount_root=str(tmp_path)),
                       "test-swarm", tmp_path / "logs" / "storage.log")


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


def test_bucket_init_publishes_authenticated_token_without_hard_links(tmp_path, monkeypatch):
    import errno

    pytest.importorskip("msgboard")
    import os

    def no_links(src, dst, *a, **kw):
        raise OSError(errno.ENOTSUP, "hard links unsupported", str(src), None, str(dst))

    monkeypatch.setattr(os, "link", no_links)
    root = tmp_path / "board"
    assert managed._init_bucket(str(root)) == 0
    token = root / "operator.token"
    original = token.read_bytes()
    assert token.stat().st_mode & 0o777 == 0o600
    assert not (root / "operator.token.tmp").exists()
    assert managed._init_bucket(str(root)) == 0
    assert token.read_bytes() == original


def test_bucket_init_rejects_unrelated_unsupported_operations(tmp_path, monkeypatch):
    import errno

    pytest.importorskip("msgboard")
    from msgboard import cli

    def fail(*a):
        raise OSError(errno.ENOTSUP, "unrelated operation", "other-file")

    monkeypatch.setattr(cli, "cmd_init", fail)
    with pytest.raises(OSError):
        managed._init_bucket(str(tmp_path))
    assert not (tmp_path / "operator.token").exists()


def test_bucket_viewer_final_export_follows_verified_shutdown(tmp_path, fake_mounts):
    cfg = {"kind": "hf-bucket", "bucket": "team/private", "mount_root": str(tmp_path / "scratch")}
    b = ManagedBoard(tmp_path / "viewer-run")
    b.start(spec(storage=cfg))
    api = b._viewer.api
    prefix = b.bucket.prefix.removesuffix("board") + "viewer/"
    try:
        with join(b) as client:
            channel = client.get("/v1/channels").json()["channels"][0]["id"]
            client.post(f"/v1/channels/{channel}/messages", json={"body": "final viewer message"},
                        headers={"Idempotency-Key": "viewer"}).raise_for_status()
    finally:
        b.stop()
    manifest = json.loads(api.remote[prefix + "manifest.json"])
    assert manifest["completed"] and b.bucket.verified_at
    doc = json.loads(api.remote[prefix + manifest["generation"] + "/board.json"])
    assert any(m["body"] == "final viewer message" for m in doc["messages"])


def test_viewer_upload_failure_does_not_fail_board_persistence(tmp_path, fake_mounts, monkeypatch):
    from flock.board_viewer import export

    def fail(*args, **kwargs):
        raise OSError("fake upload unavailable")

    monkeypatch.setattr(export, "publish", fail)
    cfg = {"kind": "hf-bucket", "bucket": "team/private", "mount_root": str(tmp_path / "scratch")}
    b = ManagedBoard(tmp_path / "failed-viewer-run")
    b.start(spec(storage=cfg))
    b.stop()
    assert b.bucket.verified_at
    assert not b._viewer.thread.is_alive()


def test_failed_bucket_verification_never_marks_viewer_complete(tmp_path, fake_mounts, monkeypatch):
    cfg = {"kind": "hf-bucket", "bucket": "team/private", "mount_root": str(tmp_path / "scratch")}
    b = ManagedBoard(tmp_path / "unverified-viewer-run")
    b.start(spec(storage=cfg))
    b._viewer.stop()
    b._viewer.once()
    api = b._viewer.api
    prefix = b.bucket.prefix.removesuffix("board") + "viewer/"

    def fail(*args):
        raise BucketStorageError("fake verification failure")

    monkeypatch.setattr(b.bucket, "_verify", fail)
    with pytest.raises(ManagedBoardError, match="verification"):
        b.stop()
    assert not json.loads(api.remote[prefix + "manifest.json"])["completed"]


def test_snapshot_recovery_retention_and_transient_failure(mount, fake_mounts):
    m = mount
    m.start(threading.Event(), 5)
    with contextlib.closing(sqlite3.connect(m.mount / "board.db")) as db:
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE messages(body TEXT)")
        db.execute("INSERT INTO messages VALUES ('retained')")
        db.commit()
        (m.mount / "operator.token").write_text("operator-secret")
        attachment = m.mount / "blobs/objects/file"
        attachment.parent.mkdir(parents=True)
        attachment.write_bytes(b"attachment")
        for _ in range(4):
            m.snapshot()
        manifest_path = m.prefix + "/manifest.json"
        good_manifest = fake_mounts.remote[manifest_path]
        doc = json.loads(good_manifest)
        assert len(doc["generations"]) == 3
        assert not any(p.endswith(("-wal", "-shm")) for p in fake_mounts.remote)
        fake_mounts.fail = True
        with pytest.raises(BucketStorageError):
            m.snapshot()
        assert fake_mounts.remote[manifest_path] == good_manifest
        db.execute("INSERT INTO messages VALUES ('still serving')")
        db.commit()
        fake_mounts.fail = False
    m.stop(verify=True)
    assert m.verified_at and not m.root.exists()
    restored = BucketMount(m.config, "test", m.log_path, prefix=m.prefix)
    restored.start(threading.Event(), 5, restore=True)
    with contextlib.closing(sqlite3.connect(restored.mount / "board.db")) as db:
        assert len(db.execute("SELECT * FROM messages").fetchall()) == 2
    assert (restored.mount / "blobs/objects/file").read_bytes() == b"attachment"
    restored.stop(verify=True)


def test_final_upload_failure_keeps_local_and_previous_generation(mount, fake_mounts):
    mount.start(threading.Event(), 5)
    with contextlib.closing(sqlite3.connect(mount.mount / "board.db")) as db:
        db.execute("CREATE TABLE messages(body TEXT)")
    (mount.mount / "operator.token").write_text("private")
    mount.snapshot()
    fake_mounts.fail = True
    with pytest.raises(BucketStorageError) as error:
        mount.stop(verify=True)
    assert "credential" not in str(error.value)
    assert mount.verified_at is None and mount.mount.exists()


def test_public_bucket_rejected(mount, fake_mounts):
    fake_mounts.private = False
    with pytest.raises(BucketStorageError, match="private"):
        mount.start(threading.Event(), 5)


def test_corrupt_latest_generation_falls_back(mount, fake_mounts):
    mount.start(threading.Event(), 5)
    with contextlib.closing(sqlite3.connect(mount.mount / "board.db")) as db:
        db.execute("CREATE TABLE messages(body TEXT)")
    (mount.mount / "operator.token").write_text("private")
    mount.snapshot()
    mount.stop(verify=True)
    latest = mount._generations[0]["id"]
    fake_mounts.remote[f"{mount.prefix}/generations/{latest}/board.db"] = b"corrupt"
    reopened = BucketMount(mount.config, "test", mount.log_path, prefix=mount.prefix)
    reopened.start(threading.Event(), 5, restore=True)
    reopened.stop(verify=True)


def test_legacy_flat_backup_recovers_on_local_disk(mount, fake_mounts, tmp_path):
    source = tmp_path / "legacy.db"
    with contextlib.closing(sqlite3.connect(source)) as db:
        db.execute("CREATE TABLE messages(body TEXT)")
        db.execute("INSERT INTO messages VALUES ('legacy')")
        db.commit()
    fake_mounts.remote[f"{mount.prefix}/board.db"] = source.read_bytes()
    fake_mounts.remote[f"{mount.prefix}/operator.token"] = b"private"
    fake_mounts.remote[f"{mount.prefix}/board.db-wal"] = b""
    mount.start(threading.Event(), 5, restore=True)
    with contextlib.closing(sqlite3.connect(mount.mount / "board.db")) as db:
        assert db.execute("SELECT * FROM messages").fetchall() == [("legacy",)]
    mount.stop(verify=True)
    assert f"{mount.prefix}/manifest.json" in fake_mounts.remote


def test_manifest_failure_does_not_publish_partial_generation(mount, fake_mounts, monkeypatch):
    mount.start(threading.Event(), 5)
    with contextlib.closing(sqlite3.connect(mount.mount / "board.db")) as db:
        db.execute("CREATE TABLE messages(body TEXT)")
    (mount.mount / "operator.token").write_text("private")
    mount.snapshot()
    manifest = f"{mount.prefix}/manifest.json"
    previous = fake_mounts.remote[manifest]
    upload = fake_mounts.batch_bucket_files
    def fail_manifest(bucket, *, add=(), delete=()):
        if any(dest == manifest for _, dest in add):
            raise OSError("manifest unavailable")
        return upload(bucket, add=add, delete=delete)
    monkeypatch.setattr(fake_mounts, "batch_bucket_files", fail_manifest)
    with pytest.raises(BucketStorageError):
        mount.snapshot()
    assert fake_mounts.remote[manifest] == previous
    monkeypatch.setattr(fake_mounts, "batch_bucket_files", upload)
    mount.stop(verify=True)
