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
        assert client.get("/health").json() == {"active": 0, "closed": True}
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
