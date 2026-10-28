"""Controller health (docs/health.md): probes, snapshots, usage, rollups, endpoints, CLI."""

from __future__ import annotations

import httpx

from flock.controller.health import Health, parse_vllm
from flock.controller.reconcile import swarm_stream
from flock.models import SwarmSpec

from .conftest import make_spec

T = 1_789_999_200.0  # 2026-09-21 14:00:00 UTC, an hour boundary

VLLM = """\
# HELP vllm:num_requests_waiting Number of requests waiting.
vllm:num_requests_waiting{model_name="m"} 3.0
vllm:num_requests_running{model_name="m"} 5.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.42
vllm:num_requests_waiting_by_reason{reason="x"} 9.0
"""


def world_spec(**world) -> SwarmSpec:
    spec = make_spec()
    data = spec.model_dump(mode="json")
    data["world"] = {"url": "http://world", "endpoint": "http://world", "token_env": "WT",
                     **world}
    data["providers"] = {"v": {"type": "openai_compat", "base_urls": ["http://g1/v1", "http://g2/v1"],
                               "api_key_env": "K", "self_hosted": True, "metrics": True}}
    return SwarmSpec.model_validate(data)


class FakeWorld:
    def __init__(self):
        self.up = True
        self.seen_tokens: list[str | None] = []

    def __call__(self, req: httpx.Request) -> httpx.Response:
        host, path = req.url.host, req.url.path
        if host == "world":
            self.seen_tokens.append(req.headers.get("x-world-token"))
            if not self.up:
                raise httpx.ConnectError("refused")
            if path == "/manifest":
                return httpx.Response(200, json={"world_api": "v1", "name": "w", "version": "1",
                                                 "capabilities": ["metrics-v1"]})
            if path == "/metrics":
                return httpx.Response(200, json={"loop_lag_s": 0.01, "world": {"open_prs": 4}})
        if host in ("g1", "g2") and path == "/metrics":
            return httpx.Response(200, text=VLLM)
        return httpx.Response(404)


class FakeSlurm:
    def __init__(self, state):
        self.state = state

    def state_by_name(self, name):
        return self.state


def test_parse_vllm_sums_queues_and_takes_max_kv():
    assert parse_vllm(VLLM) == {"waiting": 3.0, "running": 5.0, "kv": 0.42}
    assert parse_vllm("vllm:kv_cache_usage_perc 0.5\ngarbage\n") == {"kv": 0.5}


async def test_probe_world_metrics_slurm_and_vllm(store):
    world = FakeWorld()
    store.put_swarm(world_spec(slurm_job_name="world-job"), T)
    h = Health(store, secrets={"WT": "s3cret"}, jobs=FakeSlurm("RUNNING"),
               transport=httpx.MockTransport(world))
    await h.probe(T)
    w = h.world["demo"]
    assert w["up"] and w["manifest_ms"] is not None and w["slurm_state"] == "RUNNING"
    assert w["metrics"] == {"loop_lag_s": 0.01, "world": {"open_prs": 4}}
    assert set(world.seen_tokens) == {"s3cret"}
    assert h.providers["demo"] == {"v": {"waiting": 6.0, "running": 10.0, "kv": 0.42}}


async def test_world_down_after_two_failed_probes_then_up(store):
    world = FakeWorld()
    store.put_swarm(world_spec(slurm_job_name="world-job"), T)
    slurm = FakeSlurm("PENDING")
    h = Health(store, secrets={"WT": "s"}, jobs=slurm, transport=httpx.MockTransport(world))
    world.up = False

    def kinds():
        return [e["type"] for e in store.read_events(swarm_stream("demo"), 0)]

    await h.probe(T)
    assert kinds() == [] and h.world["demo"]["up"] is False
    await h.probe(T + 30)
    assert kinds() == ["world_down"]
    ev = store.read_events(swarm_stream("demo"), 0)[0]
    assert ev["data"]["slurm_state"] == "PENDING"  # pending, no node: not a crash
    await h.probe(T + 60)
    assert kinds() == ["world_down"]  # one event per transition
    world.up, slurm.state = True, None
    await h.probe(T + 90)
    assert kinds() == ["world_down", "world_up"] and h.world["demo"]["slurm_state"] == "absent"


def test_slurm_state_by_name_prefers_running():
    import subprocess

    from flock.controller.slurm import SlurmJobsClient

    out = {"w": "PENDING\nRUNNING\n", "p": "PENDING\n", "none": ""}
    seen = []

    def run(argv):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, out[argv[3]], "")

    jobs = SlurmJobsClient(env={}, run=run)
    assert [jobs.state_by_name(n) for n in ("w", "p", "none")] == ["RUNNING", "PENDING", None]
    assert seen[0] == ["squeue", "-h", "-n", "w", "-o", "%T"]
