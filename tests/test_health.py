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
    assert h.providers["demo"] == {"v": {"waiting": 6.0, "running": 10.0, "kv": 0.42}}  # summed
    up = {"up": True, "fails": 0, "dead": False, "waiting": 3.0, "running": 5.0, "kv": 0.42}
    assert h.servers["demo"] == {"v": {"g1": up, "g2": up}}  # and per server
    vllm = store.get_control("demo", "vllm")  # per server, for the ramp's hold_if
    assert {k: [x[1:] for x in v] for k, v in vllm.items()} == {
        "http://g1": [[3.0, 5.0, 0.0]], "http://g2": [[3.0, 5.0, 0.0]]}
    await h.probe(T)
    assert [len(v) for v in store.get_control("demo", "vllm").values()] == [2, 2]


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
    body = h.latest("demo")["body"]
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
    first = h.latest("demo")["body"]
    assert first["tools"]["open_pr"]["rate_per_min"] is None
    t1 = T + 31 * 60
    store.update_worker("demo-researcher-w0000", last_heartbeat=t1, telemetry=json.dumps({
        "mem": {"slope_mb_h": 120.0, "eta_limit_s": 1800.0},
        "tools": {"open_pr": {"n": 40, "err": 3, "refused": 9, "max_s": 7.5}}}))
    for a in (a0, a1):
        store.update_agent(a["name"], last_heartbeat=t1)
    h.on_events(a1, [_ev(a1["name"], "turn_ended")], t1)  # a1 spoke; a0 is silent 31 min
    h.tick(t1)
    body = h.latest("demo")["body"]
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
    day = (store.data_dir / "metrics/demo/2026-09-21T12.jsonl").read_text().splitlines()
    assert len(day) == 2  # T and T+31 min are 5-min samples, nothing in between
    stored = json.loads(day[-1])["body"]  # slim: worker aggregates, no per-worker map
    assert stored["workers"] == {"count": 1, "max_nonreclaim_frac": None,
                                 "min_eta_limit_s": 1800.0, "restoring": 0, "relaunches": 2}
    assert stored["types"]["researcher"]["stuck_n"] == 1
    assert [r["ts"] for r in store.health_samples("demo", "5m")] == [T, t1]
    assert [x["ts"] for x in h.history("demo", T + 1)] == [t1]


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
    assert (tmp_path / "bucket/metrics/demo/2026-09-21T12.jsonl").exists()
    assert (tmp_path / "bucket/states/demo/latest.json").exists()
    h.tick(T + 30)
    before = len(bucket.ops)
    # the two latest.json only: the DB did not change (30 s samples are in memory), nor did
    # the 6-hour file
    assert sync.push(store) == 2
    assert len(bucket.ops) == before + 1
    h.close(T + 60)  # the open hour's 5-min state buckets
    sync.push(store)
    assert (tmp_path / "bucket/states/demo/2026-09-21T14.jsonl").exists()
    restored = tmp_path / "restored"
    assert StateSync(bucket, restored).restore()
    assert (restored / "metrics/demo/2026-09-21T12.jsonl").exists()
    assert (restored / "states/demo/2026-09-21T14.jsonl").exists()


# --------------------------------------------------------------------------- rollups, run.json

def test_hourly_rollup_lines_written_when_the_hour_closes(store):
    a0, a1 = _fleet(store, T)
    h = Health(store, run="run-x")
    h.tick(T)
    assert json.loads((store.data_dir / "run.json").read_text()) == {
        "schema": 1, "run": "run-x", "swarm": "demo", "swarms": ["demo"],
        "started_at": "2026-09-21T14:00:00+00:00", "ended_at": None, "status": "running",
        "states": True, "generation": 1,
        "types": {"researcher": {"harness": "claude", "model": "claude-opus-5-5", "replicas": 3}}}
    tok = {"m": {"input": 7, "output": 1}}
    h.on_events(a0, [
        _ev(a0["name"], "turn_started"),
        _ev(a0["name"], "message", {"message": {"role": "toolResult", "toolName": "bash",
                                                "isError": True}}),
        _ev(a0["name"], "turn_ended", {"status": "completed", "tool_calls": 1, "tokens": tok}),
        _ev(a0["name"], "turn_ended", {"status": "completed", "reason": "max_turns"}),
        _ev(a0["name"], "error", {"error": "x"})], T + 10)
    store.add_prompt(a1["name"], "{}", T)
    for a in (a0, a1):
        store.update_agent(a["name"], last_heartbeat=T + 30)
    store.update_worker("demo-researcher-w0000", last_heartbeat=T + 30)
    h.tick(T + 30)  # 30 s sampled: a0 working, a1 idle with a prompt queued (pending)
    rollups = store.data_dir / "rollups" / "demo"
    assert not rollups.exists()  # the hour is still open
    h.tick(T + 3600)
    (line_a0, line_a1) = [json.loads(x) for x in (rollups / "2026-09-21T14.jsonl").read_text()
                          .splitlines()]
    assert line_a0 == {
        "h": "2026-09-21T14", "agent": "researcher-0000", "type": "researcher", "parent": None,
        "ephemeral": False, "turns": 2, "out": {"done": 1, "max_turns": 1, "max_budget": 0,
                                                "error": 0},
        "working_s": 30.0, "pending_s": 0.0, "idle_s": 0.0, "tool_calls": 1,
        "tools": {"bash": {"n": 1, "err": 1}},
        "tokens": {"m": {"input": 7, "output": 1, "cache_read": 0, "cache_write": 0,
                         "reasoning": 0}},
        "events": {"turn_started": 1, "message": 1, "turn_ended": 2, "error": 1},
        "wakes": 0, "errors": 1, "first_ts": "2026-09-21T14:00:00+00:00",
        "last_ts": "2026-09-21T14:00:00+00:00"}
    assert line_a1["pending_s"] == 30.0 and line_a1["turns"] == 0
    h.close(T + 3700)
    run = json.loads((store.data_dir / "run.json").read_text())
    assert run["status"] == "ended" and run["started_at"] == "2026-09-21T14:00:00+00:00"
    assert run["ended_at"] == "2026-09-21T15:01:40+00:00"


def test_run_json_rewritten_on_a_new_generation(store):
    _fleet(store, T)
    h = Health(store)
    h.tick(T)
    spec = make_spec(researcher={"replicas": 5, "agents_per_worker": 2})
    store.put_swarm(spec, T + 5)
    store.record_spec(spec, T + 5, {}, [], new=False)
    h.tick(T + 30)
    run = json.loads((store.data_dir / "run.json").read_text())
    assert run["generation"] == store.generation("demo")
    assert run["types"]["researcher"]["replicas"] == 5


# --------------------------------------------------------------------------- API, page, CLI

def _seed_app(app, store):
    a0, _ = _fleet(store, T, {"open_pr": {"n": 1, "err": 0, "refused": 0, "max_s": 1.0}})
    h = app.state.health
    h.on_events(a0, [_ev(a0["name"], "turn_ended", {"tokens": {"m": {"input": 5, "output": 1}}})],
                T)
    store.append_events("demo", swarm_stream("demo"), [Event(
        type="crash", ts=datetime.fromtimestamp(T, UTC), agent="",
        data={"target": "demo-researcher-w0000", "detail": "exit 1"})])
    h.tick(T)
    h.tick(T + 30)


async def test_health_endpoints(app, store, client):
    assert (await client.get("/swarms/demo/health")).status_code == 404
    _seed_app(app, store)
    r = (await client.get("/swarms/demo/health")).json()
    assert r["ts"] == T + 30 and r["generation"] == 1
    assert set(r["body"]) == {"world", "workers", "types", "tools", "providers", "provider_servers",
                             "ramps", "tokens", "alerts"}
    assert [e["type"] for e in r["recent_events"]] == ["crash"]
    hist = (await client.get("/swarms/demo/health/history", params={"since": T + 1})).json()
    assert [x["ts"] for x in hist] == [T + 30]
    iso = datetime.fromtimestamp(T, UTC).isoformat()
    assert len((await client.get("/swarms/demo/health/history",
                                 params={"since": iso, "res": "5m"})).json()) == 1
    assert (await client.get("/swarms/demo/health/history", params={"since": "x"})).status_code == 422
    assert store.usage("demo") == [{"hour": "2026-09-21T14", "type": "researcher", "input": 5,
                                    "output": 1, "cache_read": 0, "cache_write": 0,
                                    "reasoning": 0}]
    page = (await client.get("/")).text
    assert "health:" in page and "1 workers within 1 h of their memory limit" in page
    assert "researcher-0000" not in page.split("health:")[1].split("</p>")[0]  # no agent names


def test_cli_status_json_and_watch(app, store, monkeypatch):
    from typer.testing import CliRunner

    from flock import cli

    from .conftest import wire

    _seed_app(app, store)
    wire(monkeypatch, app)
    runner = CliRunner()
    out = runner.invoke(cli.app, ["status", "demo", "--json"])
    assert out.exit_code == 0, out.output
    doc = json.loads(out.output)
    assert doc["status"]["swarm"] == "demo" and doc["health"]["body"]["tools"]["open_pr"]

    def stop(_):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli.time, "sleep", stop)
    out = runner.invoke(cli.app, ["status", "demo", "--watch"])
    assert out.exit_code == 0, out.output
    for want in ("gen 1", "no world", "workers (by time to memory limit)", "demo-researcher-w0000",
                 "world tools:", "open_pr", "restores:", "control-plane events", "crash",
                 "reaches its memory limit in 30 min"):
        assert want in out.output, (want, out.output)


def test_watch_shows_world_metrics_and_provider_queues(capsys):
    from flock import cli

    m = {"loop_lag_s": 0.012, "overloaded_429_total": 7,
         "pools": {"fast": {"queue_wait_oldest_s": 0.4}, "slow": {"queue_wait_oldest_s": 12.0}},
         "world": {"open_prs": 41, "lane_comment_outbox_depth": 3,
                   "lane_comment_outbox_oldest_age_s": 95.0, "lane_push_wait_s_total": 12.5,
                   "lane_merge_tripped": 1, "lane_merge_paused": 0,
                   "lane_merge_effective_slots": 1, "lane_merge_in_flight": 1,
                   "lane_push_tripped": 0, "lane_push_paused": 1,
                   "lane_push_effective_slots": 2, "lane_push_in_flight": 0, "note": "text"}}
    body = {"world": {"up": False, "manifest_ms": None, "slurm_state": "PENDING", "metrics": m},
            "workers": {}, "types": {}, "tools": {}, "tokens": {"by_type": {}},
            "providers": {"v": {"waiting": 3.0, "running": 5.0, "kv": 0.42}},
            "provider_servers": {"v": {
                "g1:8000": {"up": True, "waiting": 3.0, "running": 5.0, "kv": 0.42, "agents": 4},
                "g2:8000": {"up": False, "fails": 2, "dead": True, "error": "ConnectError: refused",
                            "agents": 2, "dead_agents": ["a-0001", "a-0002"]}}},
            "ramps": {"w": {"step": 2, "steps": 5, "replicas": 30, "max": 100, "next_add": 20,
                            "next_at": T + 600, "holds": 0}},
            "alerts": [{"kind": "world_down", "msg": "world does not answer (Slurm job PENDING: "
                        "waiting for a node, not a crash)"}]}
    workers = [{"id": f"w{i}", "state": "launching", "job_id": str(i), "crashes": 0,
                "pending_reason": "PENDING (Resources)"} for i in range(2)]
    cli._print_health({"ts": T, "generation": 3, "body": body}, workers)
    out = capsys.readouterr().out
    for want in ("world DOWN - ms slurm PENDING", "waiting for a node", "loop lag 0.012 s",
                 "429s 7", "slow queue 12.0 s", "open_prs 41",
                 "comment outbox 3 (oldest 95 s)", "lane merge: TRIPPED slots 1 in flight 1",
                 "lane push: paused slots 2 in flight 0", "lane_push_wait_s_total 12.5",
                 "vllm v g1:8000: waiting 3 running 5 kv 42%, 4 agents",
                 "vllm v g2:8000: DOWN x2 (ConnectError: refused), 2 agents",
                 "workers: launching:2; 2 PENDING (Resources)",
                 "ramp w: step 2/5 (30 of max 100), next +20 at 14:"):
        assert want in out, (want, out)


# --------------------------------------------------------------------------- review fixes

async def test_a_failing_probe_does_not_end_the_loop(store, monkeypatch):
    import asyncio

    h = Health(store)
    ticks = []

    async def boom(now):
        raise RuntimeError("probe broke")

    monkeypatch.setattr(h, "probe", boom)
    monkeypatch.setattr(h, "tick", ticks.append)
    task = asyncio.create_task(h.run(0.01))
    for _ in range(200):
        if len(ticks) >= 2:
            break
        await asyncio.sleep(0.01)
    alive = not task.done()
    task.cancel()
    assert len(ticks) >= 2 and alive


async def test_squeue_failure_is_unknown_and_down_still_counts(store):
    class NoSqueue:
        def state_by_name(self, name):
            raise FileNotFoundError("squeue")

    world = FakeWorld()
    world.up = False
    store.put_swarm(world_spec(slurm_job_name="world-job"), T)
    h = Health(store, secrets={"WT": "s"}, jobs=NoSqueue(), transport=httpx.MockTransport(world))
    await h.probe(T)
    await h.probe(T + 30)
    assert h.world["demo"]["slurm_state"] == "unknown"
    evs = store.read_events(swarm_stream("demo"), 0)
    assert [e["type"] for e in evs] == ["world_down"] and evs[0]["data"]["slurm_state"] == "unknown"


def test_turn_without_usage_counts_its_message_usage(store):
    a0, _ = _fleet(store, T)
    h = Health(store)
    msg = {"message": {"role": "assistant", "model": "m", "usage": {"input": 9, "output": 4}}}
    h.on_events(a0, [_ev(a0["name"], "turn_started"), _ev(a0["name"], "message", msg),
                     _ev(a0["name"], "turn_ended", {"usage_missing": True, "tokens": {
                         "m": {"input": 0, "output": 0}}})], T)
    h.tick(T + 1)
    (row,) = store.usage("demo")
    assert (row["input"], row["output"]) == (9, 4)


def test_interrupted_turn_counts_its_total_once_or_its_message_usage(store):
    a0, _ = _fleet(store, T)
    h = Health(store)
    msg = {"message": {"role": "assistant", "model": "m", "usage": {"input": 9, "output": 4}}}
    tok = {"reason": "terminated", "tokens": {"m": {"input": 12, "output": 5}}}
    h.on_events(a0, [_ev(a0["name"], "turn_started"), _ev(a0["name"], "message", msg),
                     _ev(a0["name"], "turn_interrupted", tok)], T)  # the total, not both
    h.on_events(a0, [_ev(a0["name"], "turn_started"), _ev(a0["name"], "message", msg),
                     _ev(a0["name"], "turn_interrupted", {"reason": "timeout"})], T)  # no total
    h.on_events(a0, [_ev(a0["name"], "session_resumed")], T + 1)  # nothing left held
    h.tick(T + 2)
    (row,) = store.usage("demo")
    assert (row["input"], row["output"]) == (21, 9)
    assert h.latest("demo")["body"]["tokens"]["by_type"]["researcher"]["input"] == 21


def test_restart_within_an_hour_keeps_one_line_per_agent_with_summed_totals(store):
    a0, _ = _fleet(store, T)
    tok = {"tokens": {"m": {"input": 10, "output": 1}}}
    first = Health(store)
    first.tick(T)
    first.on_events(a0, [_ev(a0["name"], "turn_ended", tok)], T + 10)
    first.tick(T + 300)  # the open hour's file, rewritten every 5 min
    path = store.data_dir / "rollups/demo/2026-09-21T14.jsonl"
    assert json.loads(path.read_text())["turns"] == 1
    # SIGKILL here; a new controller (the state sync restored the file) goes on from it
    second = Health(store)
    second.on_events(a0, [_ev(a0["name"], "turn_ended", tok)], T + 400)
    second.close(T + 500)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    assert [x["agent"] for x in lines] == ["researcher-0000"]  # 0001 idle all hour: not active
    a = lines[0]
    assert a["turns"] == 2 and a["tokens"]["m"]["input"] == 20 and a["events"]["turn_ended"] == 2
    second.close(T + 600)  # closing again rewrites, never appends
    assert len(path.read_text().splitlines()) == 1


def test_state_buckets_per_agent_and_5_minutes_rewritten_whole(store):
    a0, a1 = _fleet(store, T)
    w = "demo-researcher-w0000"

    def beat(t, **hb):
        for name in (a0["name"], a1["name"]):
            store.update_agent(name, last_heartbeat=t, **({"hb_state": hb[name]} if name in hb else {}))
        store.update_worker(w, last_heartbeat=t)

    states = store.data_dir / "states" / "demo"
    h = Health(store)
    h.tick(T)  # the first sample has no interval: nothing counted
    beat(T + 30)
    h.tick(T + 30)  # 14:00: a0 working 30, a1 idle 30
    store.add_prompt(a1["name"], "{}", T)
    beat(T + 60)
    h.tick(T + 60)  # a1 pending from now
    assert json.loads((states / "latest.json").read_text()) == {
        "ts": T + 60, "states": {"researcher-0000": "working", "researcher-0001": "pending"}}
    beat(T + 300, **{a0["name"]: "stuck"})
    h.tick(T + 300)  # 14:00 (interval capped at 60 s), after the file was written
    assert [json.loads(x) for x in (states / "2026-09-21T14.jsonl").read_text().splitlines()] == [
        {"b": "2026-09-21T14:00", "agent": "researcher-0000", "type": "researcher", "w": 60.0},
        {"b": "2026-09-21T14:00", "agent": "researcher-0001", "type": "researcher", "i": 30.0,
         "p": 30.0}]
    beat(T + 3600)
    h.tick(T + 3600)  # [T+3540, T+3600] goes to 14:55
    h.close(T + 3630)  # the hour closed: 14 rewritten once more with 14:55, then never again
    assert h._buckets.keys() == {"2026-09-21T15"}
    lines = [json.loads(x) for x in (states / "2026-09-21T14.jsonl").read_text().splitlines()]
    assert lines == [
        {"b": "2026-09-21T14:00", "agent": "researcher-0000", "type": "researcher", "w": 60.0,
         "s": 60.0},
        {"b": "2026-09-21T14:00", "agent": "researcher-0001", "type": "researcher", "i": 30.0,
         "p": 90.0},
        {"b": "2026-09-21T14:55", "agent": "researcher-0000", "type": "researcher", "s": 60.0},
        {"b": "2026-09-21T14:55", "agent": "researcher-0001", "type": "researcher", "p": 60.0}]
    # a restart within the hour reloads the hour's file and keeps adding: one line per bucket
    h2 = Health(store)
    beat(T + 3640)
    h2.tick(T + 3640)
    beat(T + 3670)
    h2.tick(T + 3670)
    h2.close(T + 3700)
    out = [json.loads(x) for x in (states / "2026-09-21T15.jsonl").read_text().splitlines()]
    assert [(r["b"], r["agent"], r.get("s"), r.get("p")) for r in out] == [
        ("2026-09-21T15:00", "researcher-0000", 30.0 + 30 + 30, None),
        ("2026-09-21T15:00", "researcher-0001", None, 30.0 + 30 + 30)]


def test_state_buckets_split_the_interval_at_bucket_boundaries(store):
    a0, a1 = _fleet(store, T)
    h = Health(store)
    for k in range(21):  # a sample every 31 s: never on a boundary
        t = T + 31 * k
        for a in (a0, a1):
            store.update_agent(a["name"], last_heartbeat=t)
        store.update_worker("demo-researcher-w0000", last_heartbeat=t)
        h.tick(t)
    h.close(T + 620)
    out = [json.loads(x) for x in (store.data_dir / "states/demo/2026-09-21T14.jsonl")
           .read_text().splitlines()]
    assert [(r["b"][-5:], r["agent"], r.get("w") or r.get("i")) for r in out] == [
        ("14:00", "researcher-0000", 300.0), ("14:00", "researcher-0001", 300.0),
        ("14:05", "researcher-0000", 300.0), ("14:05", "researcher-0001", 300.0),
        ("14:10", "researcher-0000", 20.0), ("14:10", "researcher-0001", 20.0)]
