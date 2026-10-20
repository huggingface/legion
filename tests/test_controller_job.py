"""Job mode (docs/deploy-job.md): deployment config, workers launched into the controller's
network group, X-Flock-Token auth next to the jobs proxy's HF bearer, state sync to a bucket,
the public status port, and the launch arguments of scripts/deploy-job.sh."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import typer

from flock import cli
from flock.controller import jobdeploy
from flock.controller import spawn as spawning
from flock.controller.jobs import HubJobsClient
from flock.controller.main import build, deployment, serve, status_only
from flock.controller.reconcile import reconcile
from flock.controller.state import StateSync
from flock.controller.store import Store
from flock.models import Event
from flock.worker.main import WorkerConfig, expand_env

from .conftest import TOKEN, make_spec

GROUP_URL = "http://${HF_NETWORK_GROUP_PREFIX}controller:7860"
JOB_ENV = {
    "FLOCK_TOKEN": TOKEN,
    "FLOCK_MODE": "job",
    "FLOCK_NETWORK_GROUP": "flock-smoke",
    "FLOCK_NAMESPACE": "ns",
    "FLOCK_FAKE_JOBS": "1",
    "JOB_ID": "6ad7aaaa0000000000000001",
    "ANTHROPIC_API_KEY": "sk-test",
}


# --------------------------------------------------------------------------- deployment


def test_job_mode_urls_come_from_job_id_and_alias():
    dep = deployment(JOB_ENV)
    assert dep.mode == "job" and dep.network_group == "flock-smoke" and dep.namespace == "ns"
    assert dep.worker_url == GROUP_URL
    assert dep.public_url == "https://6ad7aaaa0000000000000001--7860.hf.jobs"
    assert dep.status_port == 7861
    assert dep.status_url == "https://6ad7aaaa0000000000000001--7861.hf.jobs"


def test_job_mode_needs_a_group_and_space_mode_is_unchanged():
    with pytest.raises(SystemExit, match="FLOCK_NETWORK_GROUP"):
        deployment({"FLOCK_MODE": "job"})
    dep = deployment({"SPACE_HOST": "me-flock.hf.space"})
    assert dep.mode == "space" and dep.network_group is None and dep.status_port is None
    assert dep.worker_url == dep.public_url == "https://me-flock.hf.space"
    with pytest.raises(SystemExit, match="FLOCK_MODE"):
        deployment({"FLOCK_MODE": "spaces"})


def _built(tmp_path, **extra):
    return build({**JOB_ENV, "FLOCK_DATA": str(tmp_path / "data"), **extra})


def test_job_mode_launches_workers_in_the_group_with_the_alias_url(tmp_path):
    b = _built(tmp_path)
    b.settings.secret_source = {"ANTHROPIC_API_KEY": "sk-test"}
    b.store.put_swarm(make_spec(), 0)
    reconcile(b.store, b.jobs, datetime(2026, 10, 20, tzinfo=UTC), b.settings)
    launches = b.jobs.launches
    assert len(launches) == 2  # 3 replicas, 2 per worker
    for x in launches:
        assert x["network_group"] == "flock-smoke"
        assert x["env"]["FLOCK_URL"] == GROUP_URL
        assert x["namespace"] == "ns"
    b.store.close()


def test_space_mode_launches_workers_without_a_group(tmp_path):
    b = build({"FLOCK_TOKEN": TOKEN, "FLOCK_FAKE_JOBS": "1", "FLOCK_DATA": str(tmp_path / "d"),
               "FLOCK_URL": "https://me-flock.hf.space"})
    b.store.put_swarm(make_spec(), 0)
    reconcile(b.store, b.jobs, datetime(2026, 10, 20, tzinfo=UTC), b.settings)
    assert {x["network_group"] for x in b.jobs.launches} == {None}
    assert {x["env"]["FLOCK_URL"] for x in b.jobs.launches} == {"https://me-flock.hf.space"}
    b.store.close()


async def test_job_mode_refuses_specs_in_another_namespace(tmp_path):
    b = _built(tmp_path)
    transport = httpx.ASGITransport(app=b.app)
    headers = {"X-Flock-Token": TOKEN}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        spec = make_spec().model_copy(update={"namespace": "other"})
        r = await c.put("/swarms/demo", json=spec.model_dump(mode="json"))
        assert r.status_code == 400 and "network group" in r.text
        r = await c.put("/swarms/demo", json=make_spec().model_dump(mode="json"))
        assert r.status_code == 200, r.text
    b.store.close()


def test_hub_client_sends_network_group_only_when_set(monkeypatch):
    monkeypatch.setenv("FLOCK_WHEEL_URL", "hf://buckets/ns/flock/dist/x.whl")
    seen = []

    class Api:
        def run_job(self, **kw):
            seen.append(kw)
            return type("J", (), {"id": f"job-{len(seen)}"})()

    for group in ("flock-smoke", None):
        c = HubJobsClient(token="t", image=None, network_group=group)
        c.api = Api()
        c.launch("w0", make_spec().types["researcher"], {}, {}, {}, 60, namespace="ns")
    assert seen[0]["network_group"] == "flock-smoke"
    assert "network_group" not in seen[1]


# --------------------------------------------------------------------------- worker side


def test_worker_expands_the_group_prefix_in_flock_url():
    env = {"FLOCK_URL": GROUP_URL, "FLOCK_TOKEN": "t", "WORKER_ID": "w", "SWARM": "s",
           "TYPE": "t", "DEADLINE": "2026-10-21T00:00:00+00:00",
           "HF_NETWORK_GROUP_PREFIX": "nwa-abc123-"}
    assert WorkerConfig.from_env(env).url == "http://nwa-abc123-controller:7860"
    assert expand_env("https://x.hf.space/", {}) == "https://x.hf.space/"
    del env["HF_NETWORK_GROUP_PREFIX"]
    with pytest.raises(ValueError, match="network group"):
        WorkerConfig.from_env(env)


# --------------------------------------------------------------------------- auth


async def test_flock_token_in_header_coexists_with_proxy_bearer(app, store):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as c:
        proxy = {"Authorization": "Bearer hf_proxy_token_not_ours"}
        assert (await c.get("/swarms", headers={**proxy, "X-Flock-Token": TOKEN})).status_code == 200
        assert (await c.get("/swarms", headers={"X-Flock-Token": TOKEN})).status_code == 200
        # Space mode: the bearer alone still works
        assert (await c.get("/swarms", headers={"Authorization": f"Bearer {TOKEN}"})).status_code == 200
        assert (await c.get("/swarms", headers=proxy)).status_code == 401
        assert (await c.get("/swarms", headers={"X-Flock-Token": "nope"})).status_code == 401
        assert (await c.get("/swarms")).status_code == 401


async def test_agent_token_in_header_keeps_its_scope(app, store, client, tick):
    from .test_spawn_api import LEADER, apply, spawn_spec

    await apply(client, spawn_spec())
    tick()
    tok = spawning.agent_token(store, LEADER, "demo", 0)
    transport = httpx.ASGITransport(app=app)
    headers = {"X-Flock-Token": tok, "Authorization": "Bearer hf_proxy"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        assert (await c.get(f"/agents/{LEADER}")).status_code == 200
        assert (await c.get("/swarms")).status_code == 403  # scoped: not the controller token


def test_cli_headers_space_mode():
    h = cli.auth_headers("https://me-flock.hf.space", {"FLOCK_TOKEN": "ft"})
    assert h == {"X-Flock-Token": "ft", "Authorization": "Bearer ft"}
    h = cli.auth_headers("http://nwa-x-controller:7860", {"FLOCK_TOKEN": "ft",
                                                          "FLOCK_AGENT_TOKEN": "at"})
    assert h == {"X-Flock-Token": "at", "Authorization": "Bearer at"}


def test_cli_headers_behind_the_jobs_proxy(monkeypatch):
    url = "https://6ad7aaaa0000000000000001--7860.hf.jobs"
    h = cli.auth_headers(url, {"FLOCK_TOKEN": "ft", "HF_TOKEN": "hf_x"})
    assert h == {"X-Flock-Token": "ft", "Authorization": "Bearer hf_x"}
    h = cli.auth_headers("https://proxy.example", {"FLOCK_TOKEN": "ft", "HF_TOKEN": "hf_x",
                                                   "FLOCK_PROXY_AUTH": "hf"})
    assert h["Authorization"] == "Bearer hf_x"
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_saved")
    assert cli.auth_headers(url, {"FLOCK_TOKEN": "ft"})["Authorization"] == "Bearer hf_saved"
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    with pytest.raises(typer.Exit):
        cli.auth_headers(url, {"FLOCK_TOKEN": "ft"})


async def test_cli_headers_pass_the_api_auth(app):
    """The headers the CLI sends behind the proxy are accepted by the controller."""
    headers = cli.auth_headers("https://j--7860.hf.jobs", {"FLOCK_TOKEN": TOKEN, "HF_TOKEN": "hf"})
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        assert (await c.get("/swarms")).status_code == 200


# --------------------------------------------------------------------------- status port


async def test_status_port_serves_only_the_page_and_healthz(app):
    transport = httpx.ASGITransport(app=status_only(app))
    headers = {"X-Flock-Token": TOKEN, "Authorization": f"Bearer {TOKEN}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        assert (await c.get("/")).status_code == 200
        assert (await c.get("/healthz")).json()["ok"] is True
        assert (await c.get("/swarms")).status_code == 404  # even with the token
        assert (await c.post("/", json={})).status_code == 404
        assert (await c.get("/workers/x/commands")).status_code == 404
    # the API port keeps everything, `/` included
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t", headers=headers) as c:
        assert (await c.get("/")).status_code == 200
        assert (await c.get("/swarms")).status_code == 200


# --------------------------------------------------------------------------- state sync


class FakeBucket:
    """A bucket in a local directory, with the BucketStore operations StateSync uses."""

    def __init__(self, root: Path):
        self.root = root
        self.ops: list[str] = []

    def upload(self, files):
        self.ops.append("upload")
        for local, rel in files:
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(local, self.root / rel)

    def list(self, prefix):
        return [str(p.relative_to(self.root)) for p in self.root.rglob("*")
                if p.is_file() and str(p.relative_to(self.root)).startswith(prefix)]

    def download(self, files):
        for rel, local in files:
            local.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.root / rel, local)

    def sync_up(self, local, prefix):
        self.ops.append("sync_up")
        shutil.rmtree(self.root / prefix, ignore_errors=True)
        shutil.copytree(local, self.root / prefix)

    def sync_down(self, prefix, local):
        if (self.root / prefix).is_dir():
            shutil.copytree(self.root / prefix, local, dirs_exist_ok=True)

    def delete(self, paths):
        self.ops.append("delete")
        for rel in paths:
            (self.root / rel).unlink()


def _event(agent: str, n: int) -> Event:
    return Event(type="note", ts=datetime(2026, 10, 20, tzinfo=UTC), agent=agent, data={"n": n})


def test_state_round_trip_through_the_bucket(tmp_path, caplog):
    caplog.set_level(logging.INFO, "flock.state")
    bucket = FakeBucket(tmp_path / "bucket")
    a = tmp_path / "a"
    store = Store(a)
    sync = StateSync(bucket, a)
    store.put_swarm(make_spec(), 1.0)
    store.insert_agent("researcher-0000", "demo", "researcher", 0, 1.0)
    store.append_events("demo", "researcher-0000", [_event("researcher-0000", i) for i in range(3)])
    assert sync.push(store) == 2  # state.db + one event file
    assert sync.push(store) == 0  # nothing changed: no upload
    assert bucket.ops == ["upload", "sync_up"]
    store.close()

    b = tmp_path / "b"
    assert StateSync(bucket, b).restore() is True
    assert "restored 1 swarms (demo), 1 agents, 0 workers" in caplog.text
    restored = Store(b)
    assert restored.get_swarm("demo")["spec"] == make_spec()
    assert restored.get_agent("researcher-0000")["type"] == "researcher"
    assert [e["data"]["n"] for e in restored.read_events("researcher-0000", 0)] == [0, 1, 2]
    restored.append_events("demo", "researcher-0000", [_event("researcher-0000", 3)])
    assert restored.event_count("researcher-0000") == 4
    restored.close()


def test_restore_with_no_snapshot_starts_empty(tmp_path):
    (tmp_path / "bucket").mkdir()
    assert StateSync(FakeBucket(tmp_path / "bucket"), tmp_path / "d").restore() is False


def test_restore_replaces_a_stale_local_db_and_wal(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket")
    src = Store(tmp_path / "src")
    src.put_swarm(make_spec("fromsnap"), 1.0)
    StateSync(bucket, tmp_path / "src").push(src)
    src.close()
    stale = Store(tmp_path / "d")
    stale.put_swarm(make_spec("stale"), 1.0)
    stale.close()
    StateSync(bucket, tmp_path / "d").restore()
    s = Store(tmp_path / "d")
    assert [sw["spec"].swarm for sw in s.list_swarms()] == ["fromsnap"]
    s.close()


def test_repair_events_aligns_files_with_the_restored_index(tmp_path):
    s = Store(tmp_path)
    s.append_events("demo", "a-0000", [_event("a-0000", i) for i in range(2)])
    s.append_events("demo", "b-0000", [_event("b-0000", i) for i in range(3)])
    # events appended after the DB snapshot (file longer than the index) and a torn line
    with (tmp_path / "events/demo/a-0000.jsonl").open("a") as f:
        f.write(_event("a-0000", 9).model_dump_json() + "\n" + '{"torn')
    (tmp_path / "events/demo/b-0000.jsonl").unlink()  # event files never synced
    assert s.repair_events() == 2
    assert [e["data"]["n"] for e in s.read_events("a-0000", 0)] == [0, 1]
    assert s.event_count("b-0000") == 0 and s.read_events("b-0000", 0) == []
    s.append_events("demo", "a-0000", [_event("a-0000", 2)])
    assert [e["data"]["n"] for e in s.read_events("a-0000", 0)] == [0, 1, 2]
    s.close()


def test_build_restores_state_before_opening_the_store(tmp_path, monkeypatch):
    bucket = FakeBucket(tmp_path / "bucket")
    src = Store(tmp_path / "src")
    src.put_swarm(make_spec(), 1.0)
    StateSync(bucket, tmp_path / "src").push(src)
    src.close()
    import flock.controller.main as main_mod

    monkeypatch.setattr(main_mod, "state_sync_from_env",
                        lambda env, d: StateSync(bucket, d))
    b = _built(tmp_path, FLOCK_STATE_URL="hf://buckets/ns/flock/controller/flock-smoke")
    assert b.store.get_swarm("demo") is not None
    b.store.close()
    b = _built(tmp_path / "x", FLOCK_STATE_RESTORE="0")
    assert b.store.get_swarm("demo") is None
    b.store.close()


def test_a_snapshot_without_swarms_is_not_restored(tmp_path, caplog):
    """After `flock remove` of the last swarm, the next push writes an empty snapshot: a new
    controller in the same group must start empty, not resurrect the removed swarm."""
    caplog.set_level(logging.INFO, "flock.state")
    bucket = FakeBucket(tmp_path / "bucket")
    s = Store(tmp_path / "a")
    sync = StateSync(bucket, tmp_path / "a")
    s.put_swarm(make_spec(), 1.0)
    sync.push(s)
    s.delete_swarm("demo")
    assert sync.push(s) >= 1
    s.close()
    assert StateSync(bucket, tmp_path / "b").restore() is False
    assert "no swarms" in caplog.text
    fresh = Store(tmp_path / "b")
    assert fresh.list_swarms() == []
    fresh.close()


def test_clear_deletes_the_whole_state_prefix(tmp_path):
    bucket = FakeBucket(tmp_path / "bucket")
    s = Store(tmp_path / "a")
    s.put_swarm(make_spec(), 1.0)
    s.append_events("demo", "x-0000", [_event("x-0000", 0)])
    sync = StateSync(bucket, tmp_path / "a")
    sync.push(s)
    s.close()
    assert sync.clear() == 2 and bucket.list("") == []
    assert StateSync(bucket, tmp_path / "b").restore() is False


class SlowOrBrokenBucket(FakeBucket):
    def __init__(self, root, *, fail=False, block: threading.Event | None = None):
        super().__init__(root)
        self.fail, self.block = fail, block

    def upload(self, files):
        if self.block is not None:
            self.block.wait(10)
        if self.fail:
            raise RuntimeError("bucket down")
        super().upload(files)


async def _serve_then_sigterm(b, *, long_poll: bool = False) -> float:
    """Start serve(), optionally park a 30 s long-poll, SIGTERM ourselves; seconds to exit."""
    loop = asyncio.get_running_loop()
    task = asyncio.create_task(serve(b, host="127.0.0.1"))
    client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{b.deployment.port}",
                               headers={"X-Flock-Token": TOKEN}, timeout=60)
    for _ in range(100):  # until the server answers
        try:
            if (await client.get("/healthz")).status_code == 200:
                break
        except httpx.TransportError:
            await asyncio.sleep(0.05)
    poll = None
    if long_poll:
        b.store.insert_agent("researcher-0000", "demo", "researcher", 0, 1.0)
        poll = asyncio.create_task(client.get("/agents/researcher-0000/events?wait=30"))
        await asyncio.sleep(0.3)
    t0 = loop.time()
    os.kill(os.getpid(), signal.SIGTERM)
    await asyncio.wait_for(task, 20)
    took = loop.time() - t0
    if poll is not None:
        poll.cancel()
    await client.aclose()
    return took


async def test_sigterm_stops_loops_then_pushes_and_logs_ok(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO)
    bucket = FakeBucket(tmp_path / "bucket")
    import flock.controller.main as main_mod

    monkeypatch.setattr(main_mod, "state_sync_from_env", lambda env, d: StateSync(bucket, d))
    port = _free_port()
    b = _built(tmp_path, PORT=str(port), FLOCK_STATUS_PORT="0",
               FLOCK_STATE_URL="hf://x/y/z", FLOCK_STATE_SYNC_INTERVAL="0.1")
    took = await _serve_then_sigterm(b, long_poll=True)
    # the parked 30 s long-poll does not hold the exit past the graceful window
    assert took < main_mod.GRACEFUL_SHUTDOWN + 3
    lines = [r.getMessage() for r in caplog.records]
    order = [next(i for i, m in enumerate(lines) if m.startswith(p))
             for p in ("SIGTERM: shutting down", "final push: ok (", "flock controller: exit")]
    assert order == sorted(order)
    assert "final push: ok (1 files)" in lines  # state.db; no event files yet
    pushes = b.state.pushes
    await asyncio.sleep(0.3)  # the periodic sync loop was cancelled: no push after exit
    assert b.state.pushes == pushes and "state.db" in bucket.list("state.db")
    b.store.close()


async def test_final_push_failure_and_timeout_are_logged_and_exit_anyway(tmp_path, monkeypatch,
                                                                         caplog):
    caplog.set_level(logging.INFO)
    import flock.controller.main as main_mod

    broken = SlowOrBrokenBucket(tmp_path / "b1", fail=True)
    monkeypatch.setattr(main_mod, "state_sync_from_env", lambda env, d: StateSync(broken, d))
    b = _built(tmp_path / "one", PORT=str(_free_port()), FLOCK_STATUS_PORT="0",
               FLOCK_STATE_URL="hf://x/y/z")
    await _serve_then_sigterm(b)
    assert "final push: failed (RuntimeError: bucket down)" in caplog.text
    b.store.close()

    release = threading.Event()
    slow = SlowOrBrokenBucket(tmp_path / "b2", block=release)
    monkeypatch.setattr(main_mod, "state_sync_from_env", lambda env, d: StateSync(slow, d))
    b = _built(tmp_path / "two", PORT=str(_free_port()), FLOCK_STATUS_PORT="0",
               FLOCK_STATE_URL="hf://x/y/z", FLOCK_FINAL_PUSH_TIMEOUT="0.3")
    took = await _serve_then_sigterm(b)
    release.set()
    assert "final push: failed (timeout after 0.3 s)" in caplog.text
    assert took < main_mod.GRACEFUL_SHUTDOWN + 3


def _free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------------------- deploy-job.sh


def test_controller_job_launch_arguments():
    kw = jobdeploy.run_job_args(
        wheel_url="hf://buckets/me/flock/dist/abc/flock-0.0.1-py3-none-any.whl",
        group="flock-smoke", namespace="me", flavor="cpu-upgrade", timeout="48h",
        state_url="hf://buckets/me/flock/controller/flock-smoke",
        job_labels="experiment=flock-m1", secrets={"FLOCK_TOKEN": "f", "HF_TOKEN": "h"},
    )
    assert kw["network_group"] == "flock-smoke" and kw["network_aliases"] == ["controller"]
    assert kw["expose"] == [7860] and kw["expose_public"] == [7861]
    assert kw["env"]["FLOCK_MODE"] == "job" and kw["env"]["FLOCK_NAMESPACE"] == "me"
    assert kw["env"]["FLOCK_STATE_URL"].endswith("/controller/flock-smoke")
    assert "swarm" not in kw["labels"]  # never adopted as a worker by rebuild_from_jobs
    assert kw["namespace"] == "me" and kw["timeout"] == "48h"
    script = kw["command"][2]
    assert subprocess.run(["bash", "-n", "-c", script], check=False).returncode == 0
    assert script.rstrip().endswith("exec flock-controller")
    assert jobdeploy.default_state_url(kw["env"]["FLOCK_WHEEL_URL"], "g") == (
        "hf://buckets/me/flock/controller/g")


def test_deploy_job_dry_run_never_prints_secret_values(capsys):
    env = {"FLOCK_TOKEN": "flock-secret-value", "HF_TOKEN": "hf_secret_value",
           "ANTHROPIC_KEY": "sk-secret", "OPENAI_API_KEY": "sk-openai-secret"}
    rc = jobdeploy.main(["hf://buckets/me/flock/dist/a/f.whl", "--namespace", "me",
                         "--secret", "OPENAI_API_KEY", "--dry-run"], env)
    out = capsys.readouterr().out
    assert rc == 0 and "network_group: flock" in out
    assert "OPENAI_API_KEY" in out and "ANTHROPIC_API_KEY" in out
    assert not any(v in out for v in env.values())
    with pytest.raises(SystemExit, match="FLOCK_TOKEN"):
        jobdeploy.main(["hf://buckets/me/flock/dist/a/f.whl", "--namespace", "me", "--dry-run"],
                       {"HF_TOKEN": "h"})


class FakeHub:
    """list_jobs / cancel_job for --down: the controller stays RUNNING for `polls` lists."""

    def __init__(self, polls: int = 1):
        self.polls, self.cancelled = polls, []

    def list_jobs(self, labels, namespace):
        assert labels == {"flock-role": "controller", "flock-group": "g"} and namespace == "me"
        if self.polls <= 0:
            return []
        self.polls -= 1
        status = type("S", (), {"stage": "RUNNING"})()
        return [type("J", (), {"id": "ctl-1", "status": status})()]

    def cancel_job(self, job_id, namespace):
        self.cancelled.append(job_id)


def _down_bucket(tmp_path, monkeypatch) -> FakeBucket:
    bucket = FakeBucket(tmp_path / "bucket")
    (tmp_path / "bucket/events/demo").mkdir(parents=True)
    (tmp_path / "bucket/state.db").write_text("x")
    (tmp_path / "bucket/events/demo/a.jsonl").write_text("{}\n")
    monkeypatch.setattr("flock.worker.sync.HubBucketStore", lambda url, token=None: bucket)
    monkeypatch.setattr(jobdeploy.time, "sleep", lambda s: None)
    return bucket


def test_down_cancels_the_controller_then_deletes_its_state(tmp_path, monkeypatch, capsys):
    bucket = _down_bucket(tmp_path, monkeypatch)
    hub = FakeHub(polls=2)  # running at the first check, then gone after the cancel
    rc = jobdeploy.down(hub, "me", "g", "hf://buckets/me/flock/controller/g", dry_run=False,
                        env={})
    assert rc == 0 and hub.cancelled == ["ctl-1"] and bucket.list("") == []
    assert "deleted 2 state files" in capsys.readouterr().out


def test_down_dry_run_and_stuck_controller_leave_the_state(tmp_path, monkeypatch):
    bucket = _down_bucket(tmp_path, monkeypatch)
    hub = FakeHub(polls=1)
    assert jobdeploy.down(hub, "me", "g", "hf://b/n/p", dry_run=True, env={}) == 0
    assert hub.cancelled == [] and len(bucket.list("")) == 2
    clock = iter(range(0, 10_000, 100))
    monkeypatch.setattr(jobdeploy.time, "monotonic", lambda: next(clock))
    hub = FakeHub(polls=10**6)  # never ends
    assert jobdeploy.down(hub, "me", "g", "hf://b/n/p", dry_run=False, env={}) == 1
    assert len(bucket.list("")) == 2  # never delete under a live controller


def test_down_needs_a_wheel_or_state_url():
    with pytest.raises(SystemExit):
        jobdeploy.main(["--down", "--group", "g", "--namespace", "me"], {})
