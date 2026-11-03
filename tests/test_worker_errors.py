"""`flock workers` shows why a worker last failed: the job's status and its log tail on a crash,
the exception on a failed launch (newcomer run, F3: a wheel download 401 was only in the logs)."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli

from .conftest import TOKEN, make_spec

W0 = "demo-researcher-w0000"


def test_crash_records_status_and_log_tail(store, jobs, clock, tick, app, monkeypatch):
    store.put_swarm(make_spec(researcher={"replicas": 1}), clock().timestamp())
    tick()
    job = jobs.job_for(W0)
    line = "Error: 401 Unauthorized for hf://buckets/me/flock/dist/x.whl"
    jobs.logs[job] = ["bootstrap: installing", "", line]
    jobs.set_stage(job, "ERROR", "Job failed with exit code: 1")
    clock.advance(61)
    tick()
    err = store.get_worker(W0)["last_error"]
    assert err == ("ERROR: Job failed with exit code: 1 | bootstrap: installing / "
                   "Error: 401 Unauthorized for hf://buckets/me/flock/dist/x.whl")

    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    r = CliRunner().invoke(cli.app, ["workers", "demo"])
    assert r.exit_code == 0 and "401 Unauthorized" in r.output and "last_error" in r.output
    assert json.loads(TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
                      .get(f"/workers/{W0}").text)["last_error"] == err


def test_failed_launch_and_log_errors(store, jobs, clock, tick):
    jobs.fail_launches = True
    store.put_swarm(make_spec(researcher={"replicas": 1}), clock().timestamp())
    tick()
    assert store.get_worker(W0)["last_error"] == "launch failed: RuntimeError: launch failed (fake)"
    jobs.fail_launches = False

    def broken(*a, **k):
        raise OSError("hub down")

    jobs.logs_tail = broken  # logs are a nicety: the crash is still recorded
    clock.advance(31)
    tick()
    jobs.set_stage(jobs.job_for(W0), "ERROR", "boom")
    clock.advance(92)
    tick()
    assert store.get_worker(W0)["last_error"] == "ERROR: boom"
