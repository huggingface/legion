"""Real load-generator behavior with synthetic fixtures and fake HTTP; no jobs or models."""
import argparse
import asyncio
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import httpx
import pytest

DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/agent-escape/b300"


@pytest.fixture
def load(monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark", DIRECTORY / "benchmark.py")
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    monkeypatch.setitem(sys.modules, "benchmark", benchmark)
    spec = importlib.util.spec_from_file_location("router_load", DIRECTORY / "router_load.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def row(name="real-example"):
    payload = {"messages": [{"role": "user", "content": name}]}
    return {"id": name, "prompt_tokens": 3, "payload": payload,
            "sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()}


def dataset(tmp_path, rows):
    data = tmp_path / "data"
    data.mkdir()
    raw = "".join(json.dumps(r) + "\n" for r in rows).encode()
    (data / "ultrachat.jsonl").write_bytes(raw)
    manifest = {"model": "Qwen/Qwen3.8-27B", "dataset": "test-fixture", "dataset_revision": "pin",
                "files": {"ultrachat.jsonl": hashlib.sha256(raw).hexdigest()}}
    (data / "manifest.json").write_text(json.dumps(manifest))
    return data


def test_5000_request_ramp_has_two_full_waves_at_every_level(load):
    levels = [16, 32, 64, 128, 256, 512, 1024]
    plan = load.ramp_plan(5000, levels)
    assert sum(count for _, count in plan) == 5000
    assert [concurrency for concurrency, _ in plan] == levels
    assert all(count >= 2 * concurrency for concurrency, count in plan)
    assert plan[-1] == (1024, 2824)


@pytest.mark.parametrize("requests,levels", [(4, [1, 2]), (5000, [32, 16]), (5, [1, 1]),
                                           (5000, [0]), (5000, [])])
def test_invalid_ramps_rejected(load, requests, levels):
    with pytest.raises(ValueError):
        load.ramp_plan(requests, levels)


def test_verbatim_deterministic_dataset_cycles(load, tmp_path):
    rows = [row(str(i)) for i in range(4)]
    data = dataset(tmp_path, rows)
    manifest, loaded = load.load_rows(data, "ultrachat")
    assert loaded == rows and manifest["dataset_revision"] == "pin"
    selected = load.select_rows(rows, 11, 42)
    assert selected == load.select_rows(rows, 11, 42)
    for offset in (0, 4):
        assert {r["id"] for r in selected[offset:offset + 4]} == {r["id"] for r in rows}
    assert len(selected) == 11 and all(r in rows for r in selected)
    (data / "ultrachat.jsonl").write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        load.load_rows(data, "ultrachat")


def test_payload_hash_mismatch_rejected(load, tmp_path):
    bad = row()
    bad["sha256"] = "wrong"
    data = dataset(tmp_path, [bad])
    with pytest.raises(ValueError, match="Prompt checksum"):
        load.load_rows(data, "ultrachat")


def test_metrics_match_exact_names_without_double_counting(load):
    metrics = load.parse_metrics('''# HELP ignored
vllm:num_requests_running{model_name="Qwen",engine="0"} 12
vllm:num_requests_running{model_name="Qwen",engine="1"} 2
vllm:num_requests_waiting{engine="0"} 3
vllm:num_requests_waiting_by_reason{reason="capacity"} 3
vllm:kv_cache_usage_perc{engine="0"} 0.25
vllm:prefix_cache_hits_total{engine="0"} 123
''')
    assert metrics == {"vllm:num_requests_running": 14, "vllm:num_requests_waiting": 3,
                       "vllm:kv_cache_usage_perc": 0.25, "vllm:prefix_cache_hits_total": 123}


async def fake_server(request):
    if request.url.path == "/metrics":
        return httpx.Response(200, text='vllm:num_requests_running{engine="0"} 2\n')
    if request.url.path == "/v1/models":
        return httpx.Response(200, json={"data": [{"id": "Qwen/Qwen3.8-27B"}]})
    payload = json.loads(request.content)
    assert payload["max_tokens"] == 997
    assert "ignore_eos" not in payload and "min_tokens" not in payload
    await asyncio.sleep(0.005)
    if payload["messages"][0]["content"] == "rate-limit":
        return httpx.Response(429, text="overloaded")
    events = [{"choices": [{"delta": {"content": "response"}, "finish_reason": "stop"}]},
              {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 4}}]
    return httpx.Response(200, text="".join("data: " + json.dumps(e) + "\n\n" for e in events)
                          + "data: [DONE]\n\n")


async def test_level_concurrency_and_error_accounting(load, tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(fake_server)) as client:
        result = await load.run_level(
            client, "http://router", [row(), row("rate-limit"), row(), row()], 2,
            tmp_path / "level", context=1000, timeout=5, instances=[{"url": "http://backend"}],
        )
    assert result["client_inflight_peak"] == 2
    assert result["requests"] == 4 and result["completed"] == 3
    assert result["error_rate"] == 0.25
    assert result["statuses"] == {"completed": 3, "http_error": 1}
    records = [json.loads(line) for line in (tmp_path / "level/requests.jsonl").read_text().splitlines()]
    assert len(records) == 4 and all("response" not in r for r in records)
    assert {r["request_index"] for r in records} == {0, 1, 2, 3}
    assert (tmp_path / "level/metrics.jsonl").read_text()


async def test_complete_ramp_no_secret_in_artifacts(load, tmp_path, monkeypatch):
    data = dataset(tmp_path, [row()])
    endpoint = {"model": load.MODEL, "base_url": "http://router/v1",
                "instances": [{"url": "http://backend"}]}

    async def wait(*args):
        return endpoint, "private-secret-key", 1000

    monkeypatch.setattr(load, "wait_endpoint", wait)
    monkeypatch.setattr(load, "ensure_file_limit", lambda _: None)
    client_type = httpx.AsyncClient
    monkeypatch.setattr(load.httpx, "AsyncClient", lambda **kw: client_type(
        transport=httpx.MockTransport(fake_server), **kw))
    output = tmp_path / "output"
    args = argparse.Namespace(concurrency="1,2", requests=6, timeout=5, wait_ready=5,
                              data=data, workload="ultrachat", server_run=tmp_path / "server",
                              output=output, seed=42, save_responses=False)
    await load.run(args)
    completion = json.loads((output / "completion.json").read_text())
    assert completion["requests"] == completion["completed"] == 6
    assert completion["highest_zero_error_concurrency"] == 2
    assert completion["hard_limit_established"] is False
    assert all("private-secret-key" not in path.read_text() for path in output.rglob("*.json*"))


async def test_readiness_timeout_does_not_start_requests(load, tmp_path):
    with pytest.raises(TimeoutError, match="did not become ready"):
        await load.wait_endpoint(tmp_path, 0)


def test_low_client_descriptor_limit_rejected(load, monkeypatch):
    monkeypatch.setattr(load.resource, "getrlimit", lambda _: (128, 128))
    with pytest.raises(ValueError, match="file-descriptor hard limit"):
        load.ensure_file_limit(1024)
