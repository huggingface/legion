"""`flock status` and `flock.status`: verdict, model servers per server and dead ones, Slurm
pending reasons, filters and one `idle`, normalised events, the unreachable line."""

from __future__ import annotations

import itertools
import json
import os
import subprocess
import time
from datetime import UTC, datetime, timedelta

import httpx
from typer.testing import CliRunner

from flock import cli
from flock import status as fs
from flock.client import Client
from flock.controller.health import Health
from flock.controller.reconcile import record_heartbeat, swarm_stream
from flock.controller.slurm import SlurmJobsClient
from flock.models import AgentHeartbeat, SwarmSpec, WorkerHeartbeat

from .conftest import make_spec, wire
from .test_api import W0, apply, hb_body
from .test_reconcile import _nudge_spec, _stuck_tick

runner = CliRunner()


def _status(**types) -> dict:
    return {"swarm": "s", "types": {n: {"name": n, "health": "ok", "launch_error": None, **t}
                                    for n, t in types.items()}}


def test_verdict_levels():
    now = 1000.0
    h = {"ts": now, "body": {"alerts": []}}
    assert fs.verdict(_status(a={}), h, now) == {"level": "ok", "reasons": []}
    assert fs.verdict(_status(a={}), None, now)["level"] == "degraded"
    assert fs.verdict(_status(a={}), {**h, "ts": now - 600}, now)["reasons"] == [
        "health snapshot 600 s old"]
    stuck = {"ts": now, "body": {"alerts": [{"kind": "stuck", "msg": "2 agents stuck or silent"}]}}
    assert fs.verdict(_status(a={}), stuck, now) == {
        "level": "degraded", "reasons": ["2 agents stuck or silent"]}
    down = {"ts": now, "body": {"alerts": [{"kind": "world_down", "msg": "world does not answer"}]}}
    v = fs.verdict(_status(a={"launch_error": "503"}), down, now)
    assert v["level"] == "failing" and v["reasons"][0] == "world does not answer"
    v = fs.verdict(_status(a={"health": "launch_refused", "launch_error": "bad flavor"}), h, now)
    assert v == {"level": "failing", "reasons": ["type a: launches refused: bad flavor"]}


def _vllm_spec() -> SwarmSpec:
    data = make_spec(researcher={"replicas": 2}).model_dump(mode="json")
    data["providers"] = {"v": {"type": "openai_compat", "base_urls": ["http://g1/v1", "http://g2/v1"],
                               "api_key_env": "K", "self_hosted": True, "metrics": True}}
    return SwarmSpec.model_validate(data)


async def test_dead_server_flags_its_agents(store, clock, tick):
    store.put_swarm(_vllm_spec(), 0)
    tick()
    hb = WorkerHeartbeat(worker=W0, deadline=clock() + timedelta(hours=1), agents=[
        AgentHeartbeat(name="researcher-0000", state="idle", endpoint="g1"),
        AgentHeartbeat(name="researcher-0001", state="idle", endpoint="g2")])
    assert record_heartbeat(store, W0, hb, clock())
    dead = {"g2"}

    def vllm(req: httpx.Request) -> httpx.Response:
        if req.url.host in dead:
            raise httpx.ConnectError("refused")
        return httpx.Response(200, text="vllm:num_requests_waiting 1\nvllm:num_requests_running 2\n")

    h = Health(store, transport=httpx.MockTransport(vllm))
    t = clock().timestamp()
    await h.probe(t)
    h.tick(t)
    body = h.latest("demo")["body"]
    assert body["alerts"] == [] and fs.dead_servers({"body": body}) == set()  # one miss: no alert
    assert body["providers"]["v"] == {"waiting": 1.0, "running": 2.0}  # summed, as before
    await h.probe(t)
    h.tick(t)
    body = h.latest("demo")["body"]
    g1, g2 = body["provider_servers"]["v"]["g1"], body["provider_servers"]["v"]["g2"]
    assert g1 == {"up": True, "fails": 0, "dead": False, "waiting": 1.0, "running": 2.0,
                  "agents": 1}
    assert g2["dead"] and g2["agents"] == 1 and g2["dead_agents"] == ["researcher-0001"]
    assert [a["kind"] for a in body["alerts"]] == ["dead_server"]
    assert "researcher-0001" in body["alerts"][0]["msg"]
    assert fs.verdict(_status(), {"ts": t, "body": body}, t)["level"] == "degraded"
    assert fs.dead_servers({"body": body}) == {"g2"}
    dead.add("g1")
    for _ in range(2):
        await h.probe(t + 30)
        h.tick(t + 30)
    body = h.latest("demo")["body"]
    assert [a["kind"] for a in body["alerts"]] == ["provider_down"]
    assert fs.verdict(_status(), {"ts": t, "body": body}, t)["level"] == "failing"


async def test_status_filters_flag_agents_on_a_dead_server(app, store, clock, tick, monkeypatch):
    store.put_swarm(_vllm_spec(), 0)
    tick()
    hb = WorkerHeartbeat(worker=W0, deadline=clock() + timedelta(hours=1), agents=[
        AgentHeartbeat(name="researcher-0000", state="idle", endpoint="g1"),
        AgentHeartbeat(name="researcher-0001", state="working", turn_id="t", endpoint="g2")])
    assert record_heartbeat(store, W0, hb, clock())
    wire(monkeypatch, app)
    monkeypatch.setattr(Client, "health", lambda *_: {"ts": clock().timestamp(), "body": {
        "alerts": [], "types": {}, "workers": {}, "tools": {},
        "providers": {}, "provider_servers": {"v": {"g1": {"up": True, "dead": False},
                                                    "g2": {"up": False, "dead": True}}}}})
    v = fs.view(Client(), "demo", state="working")
    assert [(a["name"], a["endpoint"], a["on_dead_server"]) for a in v["agents"]] == [
        ("researcher-0001", "g2", True)]
    r = runner.invoke(cli.app, ["status", "demo", "--state", "idle"])
    assert r.exit_code == 0 and "researcher-0000" in r.output and "researcher-0001" not in r.output
    r = runner.invoke(cli.app, ["status", "demo", "--type", "researcher"])
    assert "g2 DEAD" in r.output and "g1 DEAD" not in r.output


async def test_one_idle_for_status_and_prompt(client, store, clock, tick):
    await apply(client)
    tick()
    agents = [AgentHeartbeat(name="researcher-0000", state="idle"),
              AgentHeartbeat(name="researcher-0001", state="idle")]
    assert (await client.post(f"/workers/{W0}/heartbeat",
                              json=hb_body(clock, W0, agents))).status_code == 200

    async def idle() -> list[str]:
        r = await client.get("/swarms/demo/agents", params={"state": "idle"})
        return [a["name"] for a in r.json()["agents"]]

    assert await idle() == ["researcher-0000", "researcher-0001"]
    await client.post("/agents/researcher-0000/prompt", json={"text": "go"})
    await client.get(f"/workers/{W0}/commands", params={"wait": 0})  # handed out: turn starting
    # the heartbeat still says idle; both routes now agree it is not
    assert await idle() == ["researcher-0001"]
    assert (await client.get("/agents/researcher-0000")).json()["state"] == "working"
    ack = (await client.post("/swarms/demo/prompts", json={
        "type": "researcher", "state": "idle", "text": "x"})).json()
    assert ack["skipped_state"] == 2 and ack["queued"] == 1  # 0002 never reported


def test_slurm_pending_reason():
    out = {"squeue": "PENDING|Resources\n"}
    jobs = SlurmJobsClient(env={}, run=lambda argv: subprocess.CompletedProcess(
        argv, 0, out[argv[0]], ""))
    st = jobs.inspect("42")
    assert (st.stage, st.message) == ("SCHEDULING", "PENDING (Resources)")
    out["squeue"] = "PENDING|None\n"
    assert jobs.inspect("42").message == "PENDING"
    out["squeue"] = "7|w|swarm=s|PENDING|Priority\n"
    (job,) = jobs.list({"swarm": "s"})
    assert job.message == "PENDING (Priority)"


async def test_pending_reason_reaches_status(client, store, jobs, clock, tick):
    await apply(client)
    tick()
    jobs.set_stage(jobs.job_for(W0), "SCHEDULING", "PENDING (Resources)")
    clock.advance(60)
    tick()
    rows = {w["id"]: w for w in (await client.get("/workers", params={"swarm": "demo"})).json()}
    assert rows[W0]["pending_reason"] == "PENDING (Resources)"
    jobs.set_stage(jobs.job_for(W0), "RUNNING")
    clock.advance(60)
    tick()
    assert store.get_worker(W0)["pending_reason"] is None


def test_normalise_swarm_events():
    def ev(kind, **data):
        return {"type": kind, "ts": "2026-11-03T10:00:00+00:00", "agent": "", "data": data}

    got = [fs.normalise(e, "s") for e in (
        ev("crash", target="s-w-w0001", detail="exit 1"),
        ev("stuck", target="w-0001", detail="turn timed out"),
        ev("ramp_held", type="w", holds=3),
        ev("paused", types=["w"]), ev("spec_applied", generation=2),
        ev("loop_tick", type="w"), ev("sync_failed", worker="s-w-w0002", agents={}))]
    assert [(g["name"], g.get("agent"), g.get("worker"), g.get("type")) for g in got] == [
        ("worker.crashed", None, "s-w-w0001", None), ("agent.stuck", "w-0001", None, None),
        ("ramp.held", None, None, "w"), ("swarm.stopped", None, None, None),
        ("swarm.spec_applied", None, None, None), ("loop.tick", None, None, "w"),
        ("worker.sync_failed", None, "s-w-w0002", None)]
    assert got[0] == {"name": "worker.crashed", "ts": "2026-11-03T10:00:00+00:00", "swarm": "s",
                      "worker": "s-w-w0001", "data": {"target": "s-w-w0001", "detail": "exit 1"}}
    assert fs.event_name("comms_mention", "agent") == "comms.mention"
    assert fs.event_name("turn_interrupted", "agent") == "agent.turn_interrupted"


def test_events_follow_pages_and_prefix(monkeypatch):
    seen = []

    def call(_, method, path, **kw):
        seen.append(kw["params"])
        n = kw["params"]["after"]
        return {"events": [{"type": "loop_tick" if n % 2 else "crash", "ts": "t", "agent": "",
                            "data": {"target": "w"}}], "next": n + 1}

    monkeypatch.setattr(Client, "_call", call)
    pages = list(itertools.islice(fs.events(Client(), "demo", after=4, follow=True,
                                            prefix="worker."), 3))
    assert [p["next"] for p in pages] == [5, 6, 7]
    assert [[e["name"] for e in p["events"]] for p in pages] == [
        ["worker.crashed"], [], ["worker.crashed"]]
    assert [s["wait"] for s in seen] == [25, 25, 25]


def test_stuck_agent_writes_one_agent_stuck_event(store, jobs, clock, tick):
    store.put_swarm(make_spec(researcher={"replicas": 1}), 0)  # no resume_prompt: still an event
    tick()
    for _ in range(3):
        _stuck_tick(store, clock, tick, 30)
    evs = [fs.normalise(e, "demo") for e in store.read_events(swarm_stream("demo"), 0, 1000)]
    assert [e["agent"] for e in evs if e["name"] == "agent.stuck"] == ["researcher-0000"]
    store.put_swarm(_nudge_spec(), 0)
    _stuck_tick(store, clock, tick, 1, "idle")
    _stuck_tick(store, clock, tick, 1)
    evs = store.read_events(swarm_stream("demo"), 0, 1000)
    assert sum(e["type"] == "stuck" for e in evs) == 2


def test_unreachable_names_the_host_file(tmp_path, monkeypatch):
    monkeypatch.setattr("flock.client._sleep", lambda _: None)
    run = tmp_path / "run"
    (run / "hosts").mkdir(parents=True)
    host = run / "hosts" / "controller.host"
    host.write_text("node9:7999\n")
    os.utime(host, (1_790_000_000, 1_790_000_000))
    monkeypatch.setattr("flock.client.connect", lambda *_: httpx.Client(
        base_url="http://node9:7999", transport=httpx.MockTransport(_refuse)))
    r = runner.invoke(cli.app, ["--run", str(run), "status", "demo"])
    assert r.exit_code == 1 and r.output.strip() == (
        "controller unreachable at http://node9:7999; hosts/controller.host says node9:7999, "
        "written at 2026-09-21 14:13:20Z")
    assert "Traceback" not in r.output
    monkeypatch.delenv("FLOCK_RUN_DIR", raising=False)
    r = runner.invoke(cli.app, ["status", "demo"], env={"FLOCK_URL": "http://node9:7999"})
    assert r.output.strip() == "controller unreachable at http://node9:7999"


def _refuse(req):
    raise httpx.ConnectError("refused")


def test_status_json_carries_the_verdict(monkeypatch):
    st = {**_status(a={"health": "launch_refused", "launch_error": "x"}), "agents_total": 0,
          "workers_total": 0, "pending_prompts": 0, "cost_usd": 0}
    monkeypatch.setattr(Client, "_call", lambda _, m, path, **k: [] if path == "/workers"
                        else {"status": st})
    monkeypatch.setattr(Client, "health", lambda *_: None)
    r = runner.invoke(cli.app, ["status", "s", "--json"])
    assert r.exit_code == 1 and json.loads(r.output)["verdict"]["level"] == "failing"
    r = runner.invoke(cli.app, ["status", "s"])
    assert r.output.startswith("FAILING: type a: launches refused: x; no health snapshot yet")


def test_world_alerts_wait_for_two_misses_and_pending_is_degraded():
    from flock.controller.health import alerts

    def kinds(**w):
        body = {"world": {"up": False, **w}, "workers": {}, "types": {}, "tools": {}}
        return [a["kind"] for a in alerts(body)]

    assert kinds(fails=1) == []
    assert kinds(fails=2) == ["world_down"]
    assert kinds(fails=2, slurm_state="PENDING") == ["world_pending"]
    h = {"ts": 0, "body": {"alerts": [{"kind": "world_pending", "msg": "queued"}]}}
    assert fs.verdict(_status(), h, 0)["level"] == "degraded"


def test_status_reads_an_old_format_sample(app, store, clock, monkeypatch):
    """After a restart `latest()` may be a stored slim sample from an older controller:
    summed `providers`, no `provider_servers`, `workers` as aggregates."""
    store.put_swarm(make_spec(), clock().timestamp())
    old = {"world": None, "tools": {}, "tokens": {"by_type": {}}, "alerts": [],
           "types": {"researcher": {"by_state": {"dead": 2}, "gens": {}, "stuck": []}},
           "providers": {"v": {"waiting": 3.0, "running": 5.0, "kv": 0.4}},
           "workers": {"count": 2, "max_nonreclaim_frac": None, "min_eta_limit_s": None,
                       "restoring": 0, "relaunches": 0}}
    store.add_health_sample("demo", clock().timestamp(), "5m", old, 3600)
    wire(monkeypatch, app)
    r = runner.invoke(cli.app, ["status", "demo"])
    assert r.exit_code == 0 and "Traceback" not in r.output, r.output
    # its per-type counts are from before the restart (agents `dead` then): not shown
    assert "dead:2" not in r.output and "predates the controller's restart" in r.output
    for bad in ({"v": None}, [], {"v": {"g": 1}}):
        assert fs.servers({"body": {"provider_servers": bad}}) in ({}, {"v": {}})
        assert fs.dead_servers({"body": {"provider_servers": bad}}) == set()


def test_events_of_a_removed_agent(monkeypatch):
    evs = [{"type": "turn_ended", "ts": "t", "agent": "w-0001", "data": {"data": {"status": "ok"}}}]

    def call(_, method, path, **kw):
        if path == "/agents/w-0001":
            from flock.client import ClientError
            raise ClientError("error 404", 404, "{}")
        return {"events": evs, "next": 1}

    monkeypatch.setattr(Client, "_call", call)
    (page,) = fs.events(Client(), "w-0001")
    assert page["events"][0]["name"] == "agent.turn_ended" and page["events"][0]["agent"] == "w-0001"


def test_hidden_old_events_and_agents(monkeypatch):
    """Prompts in resumed runs still call them: raw JSONL and the old agents table."""
    evs = [{"type": "spawned", "ts": "t", "agent": "l-0000", "data": {"child": "l-0000-a-0001"}}]
    rows = [{"name": "l-0000-a-0001", "state": "working", "provider": None, "worker": "w0",
             "turns": 2, "pending": 0, "cost_usd": 0.01, "last_heartbeat": "t"}]
    seen = []

    def call(_, method, path, **kw):
        seen.append(path)
        return {"agents": rows, "cursor": None} if path.endswith("/agents") else \
            {"events": evs, "next": 1}

    monkeypatch.setattr(Client, "_call", call)
    r = runner.invoke(cli.app, ["events", "l-0000", "--wait", "--timeout", "0"])
    assert r.exit_code == 0 and json.loads(r.stdout.splitlines()[0]) == evs[0]
    assert "  events " not in runner.invoke(cli.app, ["--help"]).output  # hidden
    r = runner.invoke(cli.app, ["agents", "s", "--parent", "l-0000"])
    lines = r.output.splitlines()
    assert lines[0].split() == ["name", "state", "provider", "worker", "turns", "pending", "cost",
                                "last_heartbeat"]
    assert lines[1].split()[:3] == ["l-0000-a-0001", "working", "-"]  # awk '$2=="working"'


def test_status_prints_the_age_of_heartbeat_and_snapshot_numbers(monkeypatch):
    now = time.time()
    hb = datetime.fromtimestamp(now - 12, UTC).isoformat()
    rows = [{"name": "r-0001", "state": "idle", "type": "r", "worker": "w0", "turns": 0,
             "pending": 0, "cost_usd": 0.0, "last_heartbeat": hb}]
    snap = {"ts": now - 34, "body": {"alerts": [], "workers": {}, "tools": {},
                                     "types": {"r": {"by_state": {"idle": 1}, "gens": {}, "stuck": []}}}}
    st = {"swarm": "s", "agents_total": 1, "workers_total": 1, "pending_prompts": 0,
          "cost_usd": 0.0, "types": {}}
    monkeypatch.setattr(fs, "view", lambda *a, **k: {
        "verdict": {"level": "ok", "reasons": []}, "status": st, "health": snap,
        "workers": [], "agents": rows})
    out = runner.invoke(cli.app, ["status", "s", "--type", "r"]).output
    assert "type r: idle:1  revs -  (health 34 s ago)" in out
    assert "$0.0000  12 s ago" in out  # turns and cost as of that heartbeat
