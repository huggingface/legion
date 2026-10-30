"""Offline accounting and experiment matrix regression tests."""
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

HERE = Path(__file__).resolve().parents[1] / "experiments/agent-escape"


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


proxy = load("usage_proxy")


@pytest.fixture
def sweep(monkeypatch):
    monkeypatch.syspath_prepend(str(HERE))
    monkeypatch.setitem(sys.modules, "slurm", load("slurm"))
    return load("sweep")


def test_matrix_balanced_and_reproducible(sweep):
    rows = sweep.matrix()
    assert rows == sweep.matrix()
    assert rows != sweep.matrix(1)
    assert len(rows) == 75
    assert len({(r["difficulty"], r["agents"], r["trial"]) for r in rows}) == 75


def test_accounting_cache_once_and_missing_usage_invalid(tmp_path):
    path = tmp_path / "model-requests.jsonl"
    rows = [{"event": "start", "id": "1", "session_id": "a"},
            {"event": "end", "id": "1", "status": 200, "usage": {
                "prompt_tokens": 100, "completion_tokens": 20,
                "prompt_tokens_details": {"cached_tokens": 80}}}]
    path.write_text("\n".join(map(json.dumps, rows)))
    result = proxy.summarize(path)
    assert result["complete"] and result["total_tokens"] == 120
    assert result["cached_input_tokens"] == 80
    rows.append({"event": "start", "id": "2", "session_id": "b"})
    path.write_text("\n".join(map(json.dumps, rows)))
    assert not proxy.summarize(path)["complete"]
    rows.append({"event": "end", "id": "2", "status": 200, "usage": {"total_tokens": 5}})
    path.write_text("\n".join(map(json.dumps, rows)))
    assert not proxy.summarize(path)["complete"]


@pytest.mark.parametrize("stream", [False, True])
def test_proxy_records_usage_forwards_session_and_closes_admission(tmp_path, stream):
    seen = []
    usage = {"prompt_tokens": 100, "completion_tokens": 20}

    def upstream(request):
        seen.append(request)
        if stream:
            return httpx.Response(200, text='data: ' + json.dumps({"usage": usage}) +
                                  '\n\ndata: [DONE]\n\n')
        return httpx.Response(200, json={"usage": usage})

    app = proxy.create_app("http://upstream/v1", "test-key", tmp_path)
    with TestClient(app) as client:
        original = app.state.client
        app.state.client = httpx.AsyncClient(transport=httpx.MockTransport(upstream))
        headers = {"Authorization": "Bearer test-key", "X-Session-ID": "conversation-1"}
        body = {"model": "fake", "stream": stream}
        response = client.post("/v1/chat/completions", headers=headers, json=body)
        assert response.status_code == 200
        assert seen[0].headers["X-Session-ID"] == "conversation-1"
        assert proxy.summarize(tmp_path / "model-requests.jsonl")["total_tokens"] == 120
        client.post("/close")
        assert client.get("/health").json()["active"] == 0
        assert client.get("/health").json()["closed"]
        assert client.post("/v1/chat/completions", headers=headers, json=body).status_code == 200
        assert len(seen) == 1
        assert client.post("/v1/chat/completions", json=body).status_code == 401
        app.state.client = original


def test_collection_keeps_failed_costs_and_missing_as_null(tmp_path, sweep):
    tasks = []
    for i in range(3):
        folder = tmp_path / str(i)
        folder.mkdir()
        tasks.append({"difficulty": 5, "agents": 2, "trial": i + 1,
                      "config": str(folder / "config.json")})
        if i < 2:
            (folder / "summary.json").write_text(json.dumps([{"exit_code": i, "result": {
                "success": i == 0, "accounting": {"complete": True, "total_tokens": 100}}}]))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"tasks": tasks}))
    rows = sweep.collect(manifest)
    assert [r["success"] for r in rows] == [True, False, False]
    assert [r["normalized_tokens"] for r in rows] == [50, 50, None]


async def test_first_model_requests_wait_for_all_agents(tmp_path):
    import asyncio

    seen = []

    def upstream(request):
        seen.append(request.headers["x-session-id"])
        return httpx.Response(200, json={"usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    app = proxy.create_app("http://upstream/v1", "key", tmp_path, agents=2)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    ) as backend:
        app.state.client = backend
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            async def request(sid):
                return await client.post("/v1/chat/completions", json={"model": "fake"},
                    headers={"Authorization": "Bearer key", "X-Session-ID": sid})
            first = asyncio.create_task(request("a"))
            await asyncio.sleep(.05)
            assert not seen and not first.done()
            second = asyncio.create_task(request("b"))
            responses = await asyncio.wait_for(asyncio.gather(first, second), 2)
            assert all(r.status_code == 200 for r in responses)
            assert set(seen) == {"a", "b"}



def test_review_config_and_generation_gate(tmp_path, sweep):
    import yaml

    from flock.models import SwarmSpec
    from flock.worker.adapters.pi import pi_model_config

    raw = yaml.safe_load((HERE / "spec-router-pi.yaml").read_text())
    spec = SwarmSpec.model_validate(raw)
    assert spec.comms.poll_cycle_s == 5
    worker = spec.types["worker"]
    assert worker.env["FLOCK_COMMS_RATE"] == worker.env["FLOCK_COMMS_AGENT_RATE"] == "0"
    model = pi_model_config(worker)
    assert model.base_url == "http://ip-10-53-199-210:8000/v1"
    assert model.extra["contextWindow"] == 65536
    assert model.extra["maxTokens"] == 16384
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(raw))
    result = {"success": True, "flock_exit_code": 0,
              "accounting": {"complete": True, "session_ids": ["agent"]},
              "board": {"verified_at": "now"}}
    row = {"event": "start", "tools": ["get_observation", "investigate", "tool_call",
           "use_item", "submit_answer", "read", "post"],
           "settings": {"reasoning_effort": "low", "max_completion_tokens": 16384}}
    journal = tmp_path / "model-requests.jsonl"
    journal.write_text(json.dumps(row) + "\n")
    assert sweep.gate_errors(result, 1, tmp_path) == []
    row["settings"]["max_completion_tokens"] = 4096
    journal.write_text(json.dumps(row) + "\n")
    assert "incorrect reasoning or generation cap" in sweep.gate_errors(result, 1, tmp_path)


async def test_cancelled_client_still_records_final_usage(tmp_path):
    import asyncio

    sent = asyncio.Event()
    release = asyncio.Event()

    class Chunks(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"choices": []}\n\n'
            sent.set()
            await release.wait()
            yield b'data: {"usage": {"prompt_tokens": 10, "completion_tokens": 7}}\n\n'

    app = proxy.create_app("http://upstream/v1", "key", tmp_path)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Chunks()))
    ) as backend:
        app.state.client = backend
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            task = asyncio.create_task(client.post("/v1/chat/completions",
                json={"model": "fake", "stream": True},
                headers={"Authorization": "Bearer key", "X-Session-ID": "a"}))
            await asyncio.wait_for(sent.wait(), 2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert (await client.get("/health")).json()["active"] == 1
            release.set()
            for _ in range(100):
                if (await client.get("/health")).json()["active"] == 0:
                    break
                await asyncio.sleep(.01)
            result = proxy.summarize(tmp_path / "model-requests.jsonl")
            assert result["complete"] and result["total_tokens"] == 17


async def test_absolute_request_deadline_and_startup_timeout(tmp_path):
    import asyncio

    class Endless(httpx.AsyncByteStream):
        async def __aiter__(self):
            while True:
                yield b'data: {"choices": []}\n\n'
                await asyncio.sleep(.005)

    app = proxy.create_app("http://upstream/v1", "key", tmp_path, request_timeout=.05)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=Endless()))
    ) as backend:
        app.state.client = backend
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            await asyncio.wait_for(client.post("/v1/chat/completions",
                json={"model": "fake", "stream": True},
                headers={"Authorization": "Bearer key", "X-Session-ID": "a"}), 2)
            assert (await client.get("/health")).json()["active"] == 0
    rows = [json.loads(line) for line in (tmp_path / "model-requests.jsonl").read_text().splitlines()]
    assert rows[-1]["error"] == "request_deadline"
    assert not proxy.summarize(tmp_path / "model-requests.jsonl")["complete"]
    app = proxy.create_app("http://upstream/v1", "key", tmp_path, agents=2, startup_timeout=.02)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chat/completions", json={"model": "fake"},
            headers={"Authorization": "Bearer key", "X-Session-ID": "a"})
        assert response.status_code == 503
        assert (await client.get("/health")).json()["missing_count"] == 1


def test_scheduler_classification_and_retry_selection(tmp_path, sweep):
    tasks = []
    reasons = ["startup_failure", "cancelled", "ok", "failed", "unknown"]
    for i, reason in enumerate(reasons):
        folder = tmp_path / str(i)
        folder.mkdir()
        task = {"config": str(folder / "config.json"), "difficulty": 5, "agents": 2, "trial": i}
        tasks.append(task)
        sweep.slurm.write_json(folder / "summary.json", [{"exit_code": 1, "result": {
            "infrastructure_outcome": reason, "reason": "timeout" if i == 2 else reason,
            "accounting": {"complete": i == 2, "total_tokens": 100}}}])
    states = {"0": "FAILED", "1": "CANCELLED", "2": "FAILED", "3": "FAILED", "4": "NODE_FAIL"}
    sweep.slurm.write_json(tmp_path / "scheduler-states.json", states)
    manifest = tmp_path / "manifest.json"
    sweep.slurm.write_json(manifest, {"tasks": tasks})
    rows = sweep.collect(manifest)
    assert [r["infrastructure_outcome"] for r in rows] == [
        "startup_failure", "cancelled", "ok", "failed", "node_failure"]
    assert rows[2]["measurement_valid"] and rows[2]["normalized_tokens"] == 50
    assert not rows[2]["success"]
    assert rows[4]["total_tokens"] is None


def test_finalizer_retries_once_and_preserves_attempts(tmp_path, sweep, monkeypatch):
    folder = tmp_path / "primary"
    folder.mkdir()
    task = {"config": str(folder / "config.json"), "difficulty": 5, "agents": 2, "trial": 1}
    manifest = tmp_path / "manifest.json"
    secret = tmp_path / "secrets.json"
    sweep.slurm.write_json(secret, {"HF_TOKEN": "fake"})
    sweep.slurm.write_json(manifest, {"tasks": [task], "server_run": "server", "secrets_file": str(secret)})
    sweep.slurm.write_json(tmp_path / "array-submission.json", {"job_id": "100"})
    monkeypatch.setattr(sweep, "scheduler_states", lambda job: {
        "100_0": "NODE_FAIL", "200_0": "FAILED"})
    monkeypatch.setattr(sweep, "prepare_task", lambda root, server, row, label: root / label / "config.json")
    calls = []
    def batch(root, mode, argument, **kwargs):
        calls.append(mode)
        return {"job_id": "200" if mode == "retry" else "201"}
    monkeypatch.setattr(sweep, "batch", batch)
    monkeypatch.setattr(sweep, "analyze", lambda path: 0)
    assert sweep.finalize(manifest) == 0
    assert calls == ["retry", "finalize"]
    assert sweep.finalize(manifest) == 0
    assert calls == ["retry", "finalize"]
    rows = sweep.collect(manifest)
    assert len(rows) == 1 and rows[0]["attempt_count"] == 2
    assert len(rows[0]["attempt_history"]) == 2
    assert not secret.exists()


def test_scheduler_uses_array_task_ids_not_allocation_ids(sweep, monkeypatch):
    def sacct(command, **kwargs):
        assert "--format=JobID%100,State" in command
        return "123_0|NODE_FAIL\n123_1|CANCELLED by 1000\n123_1.batch|CANCELLED\n"
    monkeypatch.setattr(sweep.subprocess, "check_output", sacct)
    assert sweep.scheduler_states("123")["123_0"] == "NODE_FAIL"
    assert sweep.scheduler_states("123")["123_1"] == "CANCELLED"


def test_truncated_journal_retains_observed_costs(tmp_path):
    path = tmp_path / "model-requests.jsonl"
    rows = [{"event": "start", "id": "1", "session_id": "a"},
            {"event": "end", "id": "1", "status": 200,
             "usage": {"prompt_tokens": 10, "completion_tokens": 2}}]
    path.write_text("\n".join(map(json.dumps, rows)) + '\n{"event": "sta')
    report = proxy.summarize(path)
    assert not report["complete"] and report["total_tokens"] == 12
    assert report["malformed_records"] == 1


@pytest.mark.parametrize("stream", [False, True])
async def test_open_barrier_survives_startup_deadline_and_close(tmp_path, monkeypatch, stream):
    import time
    from types import SimpleNamespace

    now = 0.0
    monkeypatch.setattr(proxy, "time", SimpleNamespace(time=time.time, monotonic=lambda: now))
    seen = []
    usage = {"prompt_tokens": 10, "completion_tokens": 2}
    def upstream(request):
        seen.append(request)
        if stream:
            return httpx.Response(200, text='data: ' + json.dumps({"usage": usage}) + '\n\n')
        return httpx.Response(200, json={"usage": usage})
    app = proxy.create_app("http://upstream/v1", "key", tmp_path)
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.MockTransport(upstream)
    ) as backend:
        app.state.client = backend
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://test") as client:
            async def request():
                return await client.post("/v1/chat/completions",
                    json={"model": "fake", "stream": stream},
                    headers={"Authorization": "Bearer key", "X-Session-ID": "a"})
            assert (await request()).status_code == 200
            opened = (await client.get("/health")).json()["barrier_opened"]
            for now in (300, 301, 3600):
                assert (await request()).status_code == 200
            assert len(seen) == 4
            assert (await client.get("/health")).json()["barrier_opened"] == opened
            await client.post("/close")
            assert (await request()).status_code == 200
            assert len(seen) == 4
    assert proxy.summarize(tmp_path / "model-requests.jsonl")["complete"]


async def test_late_first_arrival_does_not_open_expired_barrier(tmp_path, monkeypatch):
    import time
    from types import SimpleNamespace

    now = 0.0
    monkeypatch.setattr(proxy, "time", SimpleNamespace(time=time.time, monotonic=lambda: now))
    app = proxy.create_app("http://upstream/v1", "key", tmp_path)
    now = 301
    async with app.router.lifespan_context(app), httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        body = {"model": "fake"}
        headers = {"Authorization": "Bearer key", "X-Session-ID": "a"}
        assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 503
        assert (await client.get("/health")).json()["barrier_opened"] is None
        await client.post("/close")
        assert (await client.post("/v1/chat/completions", json=body, headers=headers)).status_code == 200


def test_plot_ranges_include_all_trials_and_error_bars(sweep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [{"difficulty": d, "agents": n, "normalized_tokens": v * 10 ** i,
             "success": True, "measurement_valid": True}
            for i, d in enumerate(sweep.DIFFICULTIES)
            for n in sweep.COUNTS for v in [1, 2, 30]]
    fig = sweep.plot_results(rows)
    try:
        for i, ax in enumerate(fig.axes):
            low, high = ax.get_ylim()
            assert low <= 0
            assert high > 30 * 10 ** i
            assert len(ax.get_xticks()) == 5
        assert fig.axes[-1].get_ylim()[1] > fig.axes[0].get_ylim()[1] * 1000
    finally:
        plt.close(fig)
