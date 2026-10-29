"""Managed boards use local real msgboard processes, never remote boards or models."""
from __future__ import annotations

import copy
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flock.comms.managed import INVITE_ENV, ManagedBoard, ManagedBoardError
from flock.controller.api import create_app
from flock.live import diff_specs
from flock.models import CommsSpec, SwarmSpec

from . import fake_openai


def spec(**comms):
    return SwarmSpec.model_validate({
        "swarm": "managed", "compute": "local", "comms": {"managed": True, **comms},
        "types": {"worker": {"harness": "claude", "model": "test", "replicas": 0}},
    })


@pytest.fixture
def board(tmp_path, monkeypatch):
    pytest.importorskip("msgboard")
    monkeypatch.delenv("FLOCK_BOARD_LOCAL_ROOT", raising=False)
    b = ManagedBoard(tmp_path / "run")
    yield b
    b.stop()
    if b.local is not None:
        shutil.rmtree(b.local, ignore_errors=True)


def join(board, name="agent"):
    r = httpx.post(board.url + "/v1/join", json={"code": board.invite, "name": name})
    r.raise_for_status()
    return httpx.Client(base_url=board.url, headers={"Authorization": "Bearer " + r.json()["token"]})


def test_managed_defaults_and_external_compatibility():
    c = CommsSpec(managed=True)
    assert c.channel == "general" and c.channels == [] and not c.allow_create_channels
    assert c.url is None and c.invite_env is None
    c = CommsSpec(url="http://board/", invite_env="INVITE", channel="#team")
    assert c.url == "http://board" and c.channel == "team" and not c.managed
    assert CommsSpec(managed=True, channels=["#planning", "planning"]).channels == ["planning"]


@pytest.mark.parametrize("fields", [{}, {"url": "http://board"}, {"invite_env": "INVITE"},
                                    {"managed": True, "channels": ["bad name"]},
                                    {"url": "http://b", "invite_env": "INVITE", "channels": ["x"]},
                                    {"url": "http://b", "invite_env": "INVITE",
                                     "allow_create_channels": True}])
def test_invalid_comms(fields):
    with pytest.raises(ValidationError):
        CommsSpec(**fields)


@pytest.mark.parametrize("compute", ["hf-jobs", "slurm", "external"])
def test_managed_requires_local_compute(compute):
    raw = spec().model_dump(mode="json")
    raw.update(compute=compute, namespace="ns", bucket="hf://buckets/ns/test")
    with pytest.raises(ValidationError, match="managed.*local"):
        SwarmSpec.model_validate(raw)


def test_missing_dependency_fails_before_creating_state(tmp_path, monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda _: None)
    b = ManagedBoard(tmp_path / "run")
    with pytest.raises(ManagedBoardError, match="extra"):
        b.start(spec())
    assert not b.archive.exists() and b.proc is None


def test_init_failure_cleans_local_directory(board, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess([], 1))
    with pytest.raises(ManagedBoardError, match="init failed"):
        board.start(spec())
    assert board.local is None and board._file_lock is None


def test_readiness_failure_stops_child(board, monkeypatch):
    def fail(self):
        raise ManagedBoardError("readiness timed out")
    monkeypatch.setattr(ManagedBoard, "_wait_ready", fail)
    with pytest.raises(ManagedBoardError, match="readiness"):
        board.start(spec())
    assert board.proc.poll() is not None and board.local is None


def test_provision_channels_invites_and_private_files(board):
    before = dict(os.environ)
    s = spec(channels=["general", "planning", "testing"])
    board.start(s)
    assert dict(os.environ) == before
    assert board.local.is_relative_to(Path("/tmp"))
    with join(board) as c:
        channels = c.get("/v1/channels").json()["channels"]
        assert {x["name"] for x in channels} == {"general", "planning", "testing"}
        assert c.post("/v1/channels", json={"name": "forbidden"}).status_code == 403
    for name in ("agent-2", "agent-3", "agent-4", "agent-5", "managed-host", "agent-7"):
        with join(board, name):
            pass  # the same invite supports the host, every agent, and retries
    assert board.archive.stat().st_mode & 0o777 == 0o700
    for name in ("operator.token", "board.db", "metadata.json"):
        assert (board.archive / name).stat().st_mode & 0o777 == 0o600
    text = json.dumps(board.metadata()) + board.log_path.read_text()
    assert board.invite not in text and board._operator not in text


def test_member_invite_can_create_channels(board):
    board.start(spec(extended=True, allow_create_channels=True))
    with join(board) as c:
        assert c.post("/v1/channels", json={"name": "reviews"}).status_code == 201
        assert "reviews" in {x["name"] for x in c.get("/v1/channels").json()["channels"]}


def test_simultaneous_boards_isolate_names_and_credentials(board, tmp_path):
    board.start(spec())
    other = ManagedBoard(tmp_path / "other")
    try:
        other.start(spec())
        assert other.url != board.url and other.local != board.local
        assert other.invite != board.invite and other._operator != board._operator
        with join(board, "same-name") as c:
            assert httpx.get(other.url + "/v1/me", headers=dict(c.headers)).status_code == 401
        with join(other, "same-name"):
            pass
    finally:
        other.stop()


def test_snapshot_and_reopen_preserve_messages_and_tokens(board):
    board.start(spec())
    with join(board) as c:
        channel = c.get("/v1/channels").json()["channels"][0]["id"]
        r = c.post(f"/v1/channels/{channel}/messages", json={"body": "retained history"},
                   headers={"Idempotency-Key": "test-post"})
        assert r.status_code == 201
        credentials = dict(c.headers)
    board.snapshot()
    run_dir = board.run_dir
    board.stop()
    restored = ManagedBoard(run_dir)
    try:
        restored.start(restore=True)
        with httpx.Client(base_url=restored.url, headers=credentials) as c:
            messages = c.get(f"/v1/channels/{channel}/messages").json()["messages"]
            assert [m["body"] for m in messages] == ["retained history"]
    finally:
        restored.stop()
    assert not restored.metadata()["running"]


def test_cannot_reopen_active_board(board):
    board.start(spec())
    other = ManagedBoard(board.run_dir)
    with pytest.raises(ManagedBoardError, match="already running"):
        other.start(restore=True)
    board.check()


def test_snapshot_failure_retains_local_data_and_releases_lock(board, monkeypatch):
    board.start(spec())
    local = board.local
    def fail():
        raise OSError("disk full")
    monkeypatch.setattr(board, "snapshot", fail)
    with pytest.raises(OSError, match="disk full"):
        board.stop()
    assert board.proc.poll() is not None and local.exists()
    assert board._file_lock is None


def test_apply_update_and_scaling_reuse_board(board, store):
    s = spec(kickoff="Kickoff")
    board.start(s)
    original = copy.deepcopy(s)
    app = create_app(store, "token", secret_source={INVITE_ENV: board.invite},
                     managed_board=board, loop_interval=0)
    with TestClient(app, headers={"Authorization": "Bearer token"}) as c:
        assert c.put("/swarms/managed", json=s.model_dump(mode="json")).status_code == 200
        resolved = store.get_swarm("managed")["spec"]
        assert resolved.comms.url == board.url and resolved.comms.invite_env == INVITE_ENV
        assert diff_specs(resolved, original) == []
        for _ in range(2):
            assert c.put("/swarms/managed", json=original.model_dump(mode="json")).status_code == 200
        changed = original.model_dump(mode="json")
        changed["types"]["worker"]["replicas"] = 2
        assert c.put("/swarms/managed", json=changed).status_code == 200
        for key, value in (("channel", "other"), ("channels", ["testing"]),
                           ("allow_create_channels", True)):
            changed = original.model_dump(mode="json")
            changed["comms"][key] = value
            assert c.put("/swarms/managed", json=changed).status_code == 400
        changed = original.model_dump(mode="json")
        changed["comms"] = None
        assert c.put("/swarms/managed", json=changed).status_code == 400
    with httpx.Client(base_url=board.url,
                      headers={"Authorization": "Bearer " + board._operator}) as c:
        channels = c.get("/v1/channels").json()["channels"]
        messages = c.get(f"/v1/channels/{channels[0]['id']}/messages").json()["messages"]
        assert [m["body"] for m in messages] == ["Kickoff"]


def test_standalone_apply_refuses_managed_board(store):
    app = create_app(store, "token", loop_interval=0)
    with TestClient(app, headers={"Authorization": "Bearer token"}) as c:
        r = c.put("/swarms/managed", json=spec().model_dump(mode="json"))
        assert r.status_code == 400 and "flock run" in r.text


@pytest.mark.parametrize("stop", ["idle", "signal", "board_failure"])
def test_local_run_managed_board_end_to_end(tmp_path, stop):
    pytest.importorskip("msgboard")
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    raw = {
        "swarm": "managed", "compute": "local",
        "providers": {"fake": {"type": "openai_compat",
                               "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                               "api_key_env": "FAKE_KEY", "self_hosted": True}},
        "comms": {"managed": True, "kickoff": "Managed kickoff", "channels": ["planning"]},
        "types": {"dev": {"harness": "llm", "model": "fake:test", "replicas": 2,
                          "comms": True, "max_turns": 3}},
    }
    source = tmp_path / "spec.yaml"
    source.write_text(yaml.safe_dump(raw))
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
    env["FAKE_KEY"] = "fake-key"
    command = [sys.executable, "-m", "flock.cli", "run", str(source), "--skip-preflight",
               "--prompt", 'call: post {"text": "agent hello", "thread": "new"}',
               "--until", "idle" if stop == "idle" else "2m"]
    p = subprocess.Popen(command, cwd=tmp_path, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 60
        meta = None
        while time.monotonic() < deadline:
            files = list((tmp_path / ".flock").glob("*/board/metadata.json"))
            if files:
                meta = json.loads(files[0].read_text())
                break
            if p.poll() is not None:
                break
            time.sleep(0.1)
        assert meta is not None, p.communicate(timeout=5)
        if stop == "signal":
            # Wait for apply/kickoff rather than interrupting startup.
            time.sleep(2)
            p.send_signal(signal.SIGINT)
        elif stop == "board_failure":
            time.sleep(2)
            os.kill(meta["pid"], signal.SIGKILL)
        out, err = p.communicate(timeout=90)
        assert p.returncode == (1 if stop == "board_failure" else 0), out + err
        run = files[0].parent.parent
        doc = json.loads((run / "run.json").read_text())
        assert doc["board"]["managed"] and not doc["board"]["running"]
        assert source.read_text() == yaml.safe_dump(raw)
        with pytest.raises(httpx.ConnectError):
            httpx.get(meta["url"] + "/v1/health", timeout=1)
    finally:
        if p.poll() is None:
            p.kill()
            p.communicate(timeout=10)
        srv.shutdown()
    assert (run / "board" / "board.db").exists()
    if stop == "idle":
        assert doc["totals"]["turns_completed"] == 2
        restored = ManagedBoard(run)
        try:
            restored.start(restore=True)
            with httpx.Client(base_url=restored.url,
                              headers={"Authorization": "Bearer " + restored._operator}) as c:
                channels = c.get("/v1/channels").json()["channels"]
                ch = next(x["id"] for x in channels if x["name"] == "general")
                messages = c.get(f"/v1/channels/{ch}/messages").json()["messages"]
                assert sorted(m["body"] for m in messages) == [
                    "Managed kickoff", "agent hello", "agent hello"]
        finally:
            restored.stop()


def test_startup_cancellation_cleans_child(board):
    board.cancel_start()
    with pytest.raises(ManagedBoardError, match="interrupted"):
        board.start(spec())
    assert board.proc.poll() is not None and board.local is None


def test_readiness_timeout_is_reported(tmp_path, monkeypatch):
    b = ManagedBoard(tmp_path, startup_timeout=0.01)
    b.url = "http://127.0.0.1:1"
    monkeypatch.setattr(b, "check", lambda: None)
    with pytest.raises(ManagedBoardError, match="readiness timed out"):
        b._wait_ready()


def test_shared_filesystem_is_refused(monkeypatch):
    read_text = Path.read_text
    def mounts(self, *args, **kwargs):
        if str(self) == "/proc/mounts":
            return "/dev/root / ext4 rw 0 0\nshared /fsx wekafs rw 0 0\n"
        return read_text(self, *args, **kwargs)
    monkeypatch.setattr(Path, "read_text", mounts)
    with pytest.raises(ManagedBoardError, match="node-local"):
        ManagedBoard._check_local(Path("/fsx/board"))
    ManagedBoard._check_local(Path("/tmp/board"))


def test_reopen_missing_snapshot_does_not_create_directory(tmp_path):
    pytest.importorskip("msgboard")
    b = ManagedBoard(tmp_path / "not-a-run")
    with pytest.raises(ManagedBoardError, match="No retained board"):
        b.start(restore=True)
    assert not b.run_dir.exists()


def test_board_cli_reopens_without_agents(board, tmp_path):
    board.start(spec())
    first_pid = board.proc.pid
    board.stop()
    with (tmp_path / "reopen.log").open("w") as out:
        p = subprocess.Popen([sys.executable, "-m", "flock.cli", "board", str(board.run_dir)],
                             stdout=out, stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            meta = json.loads((board.archive / "metadata.json").read_text())
            if meta["running"] and meta["pid"] != first_pid:
                break
            assert p.poll() is None, (tmp_path / "reopen.log").read_text()
            time.sleep(0.1)
        else:
            pytest.fail("reopen did not become ready")
        with httpx.Client(headers={"Authorization": "Bearer " + board._operator}) as c:
            assert c.get(meta["url"] + "/v1/me").status_code == 200
        assert not (board.run_dir / "agents").exists()
        p.send_signal(signal.SIGINT)
        assert p.wait(timeout=15) == 0
        assert not json.loads((board.archive / "metadata.json").read_text())["running"]
    finally:
        if p.poll() is None:
            p.kill()
            p.wait(timeout=10)


def test_two_concurrent_local_runs_get_separate_boards(tmp_path):
    pytest.importorskip("msgboard")
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    raw = {
        "swarm": "same", "compute": "local", "comms": {"managed": True},
        "providers": {"fake": {"type": "openai_compat",
                               "base_url": f"http://127.0.0.1:{srv.server_port}/v1",
                               "api_key_env": "FAKE_KEY", "self_hosted": True}},
        "types": {"dev": {"harness": "llm", "model": "fake:test", "replicas": 1,
                          "comms": True}},
    }
    source = tmp_path / "spec.yaml"
    source.write_text(yaml.safe_dump(raw))
    env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
    env["FAKE_KEY"] = "fake"
    command = [sys.executable, "-m", "flock.cli", "run", str(source), "--skip-preflight",
               "--until", "idle", "--prompt", 'call: post {"text":"hello"}']
    processes = []
    try:
        for i in range(2):
            with (tmp_path / f"run-{i}.log").open("w") as out:
                processes.append(subprocess.Popen(command, cwd=tmp_path, env=env, stdout=out,
                                                  stderr=subprocess.STDOUT))
        for i, p in enumerate(processes):
            assert p.wait(timeout=90) == 0, (tmp_path / f"run-{i}.log").read_text()
        docs = [json.loads(p.read_text()) for p in (tmp_path / ".flock").glob("*/run.json")]
        assert len(docs) == 2
        assert docs[0]["run"] != docs[1]["run"]
        assert docs[0]["board"]["url"] != docs[1]["board"]["url"]
        assert all(d["totals"]["turns_completed"] == 1 for d in docs)
        tokens = [(Path(d["board"]["operator_token_file"])).read_text() for d in docs]
        assert tokens[0] != tokens[1]
    finally:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait(timeout=10)
        srv.shutdown()


async def test_final_snapshot_failure_fails_run_and_retains_data(tmp_path, monkeypatch):
    import io

    from flock.controller.localrun import RunOptions, run_local

    pytest.importorskip("msgboard")
    snapshot = ManagedBoard.snapshot
    boards = []
    def fail_final(self):
        if self._closed:
            boards.append(self)
            raise OSError("snapshot disk full")
        snapshot(self)
    monkeypatch.setattr(ManagedBoard, "snapshot", fail_final)
    result = await run_local(RunOptions(spec=spec(), data_dir=tmp_path / "runs", until=0.1,
                                       refresh=0.05, out=io.StringIO()))
    try:
        assert result.exit_code == 1 and result.reason == "board_snapshot_failed"
        doc = json.loads((result.run_dir / "run.json").read_text())
        assert doc["stop_reason"] == "board_snapshot_failed"
        assert not doc["board"]["running"]
        assert boards[0].local.exists()
        assert not (tmp_path / "runs" / "current").exists()
        async with httpx.AsyncClient(timeout=1) as client:
            with pytest.raises(httpx.ConnectError):
                await client.get(doc["controller_url"] + "/healthz")
    finally:
        shutil.rmtree(boards[0].local, ignore_errors=True)


def test_long_run_name_still_provisions_invite(tmp_path, monkeypatch):
    monkeypatch.delenv("FLOCK_BOARD_LOCAL_ROOT", raising=False)
    name = "escape-pi-qwen27b-d10-s00-n32-r01-20261029t120000z-a1b2c3d4-20261029-120000"
    board = ManagedBoard(tmp_path / name)
    try:
        board.start(spec())
        with join(board) as client:
            assert client.get("/v1/me").status_code == 200
        assert board.run_dir.name == name
    finally:
        board.stop()
        if board.local is not None:
            shutil.rmtree(board.local, ignore_errors=True)
