"""Controller health (docs/health.md): probes, snapshots, usage, rollups, endpoints, CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import httpx

from flock.controller.health import Health, parse_vllm
from flock.controller.reconcile import swarm_stream
from flock.models import Event, SwarmSpec

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


# --------------------------------------------------------------------------- snapshots

def _ev(agent: str, type_: str, data: dict | None = None, ts: float = T) -> Event:
    when = datetime.fromtimestamp(ts, UTC)
    if type_ == "message":
        return Event(type="message", ts=when, agent=agent, data={"type": "message", **(data or {})})
    return Event(type=type_, ts=when, agent=agent,
                 data={"type": "custom", "customType": type_, "data": data or {}})


def _fleet(store, now: float, tools: dict | None = None):
    """demo: one worker w0 running agents a0 (working) and a1 (idle), with telemetry."""
    store.put_swarm(make_spec(), now)
    store.insert_worker("demo-researcher-w0000", "demo", "researcher", 0, now)
    tel = {"mem": {"current_mb": 900, "limit_mb": 1000, "nonreclaim_mb": 800, "slope_mb_h": 120.0,
                   "eta_limit_s": 1800.0},
           "restore": {"agents": 2, "files_done": 10, "files_total": 40,
                       "started": "2026-09-21T13:59:00+00:00"},
           "tools": tools or {}}
    store.update_worker("demo-researcher-w0000", job_id="j1", launched_at=now - 100,
                        last_heartbeat=now, telemetry=json.dumps(tel))
    for i, st in enumerate(("working", "idle")):
        name = f"researcher-000{i}"
        store.insert_agent(name, "demo", "researcher", i, now)
        store.update_agent(name, worker="demo-researcher-w0000", hb_state=st, last_heartbeat=now,
                           rev="r1")
    return [store.get_agent(f"researcher-000{i}") for i in range(2)]


def test_tokens_turn_total_wins_over_message_usage_and_cut_turns_count(store):
    a0, _ = _fleet(store, T)
    h = Health(store)
    msg = {"message": {"role": "assistant", "model": "m", "usage": {"input": 10, "output": 2,
                                                                     "cacheRead": 5}}}
    tok = {"tokens": {"m": {"input": 12, "output": 3, "cache_read": 5, "cache_write": 0,
                            "reasoning": 0}}}
    # a finished turn: its turn_ended total counts, its message usage does not (no double count)
    h.on_events(a0, [_ev(a0["name"], "turn_started"), _ev(a0["name"], "message", msg),
                     _ev(a0["name"], "turn_ended", tok)], T)
    # a turn cut by a worker death: message usage only, counted at the next session event
    h.on_events(a0, [_ev(a0["name"], "turn_started"), _ev(a0["name"], "message", msg)], T)
    h.on_events(a0, [_ev(a0["name"], "session_resumed")], T + 5)
    h.tick(T + 10)
    assert store.usage("demo") == [{"hour": "2026-09-21T14", "type": "researcher", "input": 22,
                                    "output": 5, "cache_read": 10, "cache_write": 0,
                                    "reasoning": 0}]
    body = store.health_samples("demo")[-1]["body"]
    assert body["tokens"]["by_type"]["researcher"]["input"] == 22


def test_snapshot_workers_types_stuck_tool_rates_and_alerts(store):
    tools0 = {"open_pr": {"n": 10, "err": 0, "refused": 0, "max_s": 2.0}}
    a0, a1 = _fleet(store, T, tools0)
    h = Health(store)
    store.append_events("demo", swarm_stream("demo"), [  # as reconcile writes them
        Event(type=k, ts=datetime.fromtimestamp(T, UTC), agent="",
              data={"target": "demo-researcher-w0000", "detail": ""})
        for k in ("crash", "loop_tick", "planned_restart")])
    h.tick(T)  # baseline: first sight of the worker's tool counters
    first = store.health_samples("demo")[-1]["body"]
    assert first["tools"]["open_pr"]["rate_per_min"] is None
    t1 = T + 31 * 60
    store.update_worker("demo-researcher-w0000", last_heartbeat=t1, telemetry=json.dumps({
        "mem": {"slope_mb_h": 120.0, "eta_limit_s": 1800.0},
        "tools": {"open_pr": {"n": 40, "err": 3, "refused": 9, "max_s": 7.5}}}))
    for a in (a0, a1):
        store.update_agent(a["name"], last_heartbeat=t1)
    h.on_events(a1, [_ev(a1["name"], "turn_ended")], t1)  # a1 spoke; a0 is silent 31 min
    h.tick(t1)
    body = store.health_samples("demo")[-1]["body"]
    w = body["workers"]["demo-researcher-w0000"]
    assert w["slope_mb_h"] == 120.0 and w["eta_limit_s"] == 1800.0 and w["relaunches"] == 2
    assert body["tools"]["open_pr"] == {"rate_per_min": 0.97, "err_pct": 10.0,
                                        "refused_pct": 30.0, "max_s": 7.5}
    t = body["types"]["researcher"]
    assert t["by_state"] == {"working": 1, "idle": 1} and t["gens"] == {"r1": 2}
    assert t["stuck"] == [{"agent": "researcher-0000", "state": "working", "silent_s": 1860}]
    kinds = [a["kind"] for a in body["alerts"]]
    assert kinds == ["memory", "stuck", "tool"]
    assert [e["type"] for e in h.recent_events("demo")] == ["crash", "planned_restart"]
    # restore ETA: 10 of 40 files in 60 s -> 180 s left
    assert first["workers"]["demo-researcher-w0000"]["restore"]["eta_s"] == 180
    # files: latest.json (newest sample) and the day's 5-min jsonl
    latest = json.loads((store.data_dir / "metrics/demo/latest.json").read_text())
    assert latest["ts"] == t1
    day = (store.data_dir / "metrics/demo/2026-09-21.jsonl").read_text().splitlines()
    assert len(day) == 2  # T and T+31 min are 5-min samples, nothing in between


def test_health_samples_are_pruned_on_insert(store):
    for i in range(3):
        store.add_health_sample("demo", T + i * 3600, "30s", {"i": i}, keep_s=3600)
    store.add_health_sample("demo", T, "5m", {"i": "old"}, keep_s=86400)
    assert [r["body"]["i"] for r in store.health_samples("demo", "30s")] == [1, 2]
    assert len(store.health_samples("demo", "5m")) == 1  # other resolutions untouched
    assert [r["body"]["i"] for r in store.health_samples("demo", "30s", since=T + 7200)] == [2]


def test_health_files_mirror_through_state_sync(tmp_path, store):
    from flock.controller.state import StateSync

    from .test_controller_job import FakeBucket

    _fleet(store, T)
    h = Health(store)
    h.tick(T)
    bucket = FakeBucket(tmp_path / "bucket")
    sync = StateSync(bucket, store.data_dir)
    sync.push(store)
    assert (tmp_path / "bucket/metrics/demo/latest.json").exists()
    assert (tmp_path / "bucket/metrics/demo/2026-09-21.jsonl").exists()
    h.tick(T + 30)
    before = len(bucket.ops)
    assert sync.push(store) == 2  # state.db and latest.json; the day file did not change
    assert len(bucket.ops) == before + 2
    restored = tmp_path / "restored"
    assert StateSync(bucket, restored).restore()
    assert (restored / "metrics/demo/2026-09-21.jsonl").exists()
