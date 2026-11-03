"""`compute: local` (docs/local.md): the spec field, the LocalJobsClient process backend, the
controller's compute check, `.flock/current` discovery and the `flock run` pieces. The end-to-end
runs are in test_local_run_e2e.py."""

from __future__ import annotations

import asyncio
import io
import json
import os
import random
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import yaml
from typer.testing import CliRunner

from flock import cli
from flock.controller import localrun
from flock.controller.api import create_app
from flock.controller.local import LocalJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.controller.store import Store
from flock.current import clear_current, read_current, write_current
from flock.models import SwarmSpec
from flock.worker.main import Worker, WorkerConfig

from .conftest import TOKEN, make_spec

runner = CliRunner()
LLM_TYPE = {"harness": "llm", "model": "fake:m", "replicas": 2}
FAKE = {"fake": {"type": "openai_compat", "base_url": "http://127.0.0.1:9/v1",
                 "api_key_env": "FAKE_KEY", "self_hosted": True}}


def local_spec(**kw) -> SwarmSpec:
    return SwarmSpec.model_validate({"swarm": "loc", "compute": "local", "providers": FAKE,
                                     "types": {"w": {**LLM_TYPE, **kw}}})


# --------------------------------------------------------------------------- spec


def test_compute_defaults_to_hf_jobs_and_keeps_job_fields_required():
    spec = make_spec()
    assert spec.compute.kind == "hf-jobs"
    types = {"a": {"harness": "claude", "model": "m"}}
    with pytest.raises(ValueError, match="namespace and bucket required for compute hf-jobs"):
        SwarmSpec.model_validate({"swarm": "s", "types": types})
    with pytest.raises(ValueError, match="bucket required"):
        SwarmSpec.model_validate({"swarm": "s", "namespace": "n", "types": types})


def test_example_switches_compute_with_one_line():
    from flock.cli import load_spec

    path = Path(__file__).parent.parent / "examples" / "local.yaml"
    assert load_spec(str(path)).compute.kind == "local"
    text = path.read_text()
    assert "\ncompute: local\n" in text
    jobs = SwarmSpec.model_validate(yaml.safe_load(text.replace("\ncompute: local\n",
                                                                "\ncompute: hf-jobs\n")))
    assert jobs.compute.kind == "hf-jobs" and jobs.namespace and jobs.bucket


def test_compute_local_short_and_long_form():
    short = local_spec()
    assert short.compute.kind == "local" and short.namespace == "" and short.bucket == ""
    long = SwarmSpec.model_validate({"swarm": "s", "compute": {"kind": "local"},
                                     "types": {"a": {"harness": "claude", "model": "m"}}})
    assert long.compute.kind == "local"
    again = SwarmSpec.model_validate(short.model_dump(mode="json"))  # stored form round-trips
    assert again.compute.kind == "local"
    with pytest.raises(ValueError):
        SwarmSpec.model_validate({"swarm": "s", "compute": "gpu", "types": {}})
    with pytest.raises(ValueError):
        SwarmSpec.model_validate({"swarm": "s", "compute": {"kind": "local", "x": 1},
                                  "types": {}})


# --------------------------------------------------------------------------- controller check


async def _put(app, spec: SwarmSpec) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        return await c.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))


async def test_controller_accepts_only_its_compute(tmp_path):
    local_app = create_app(Store(tmp_path / "a"), TOKEN, compute="local")
    r = await _put(local_app, make_spec())
    assert r.status_code == 400 and "compute 'local'" in r.text and "hf-jobs" in r.text
    assert (await _put(local_app, local_spec())).status_code == 200

    jobs_app = create_app(Store(tmp_path / "b"), TOKEN, compute="hf-jobs")
    r = await _put(jobs_app, local_spec())
    assert r.status_code == 400 and "flock run" in r.text
    assert (await _put(jobs_app, make_spec())).status_code == 200

    any_app = create_app(Store(tmp_path / "c"), TOKEN)  # tests and FLOCK_FAKE_JOBS: unchecked
    assert (await _put(any_app, local_spec())).status_code == 200


def test_build_with_flock_compute_local(tmp_path):
    from flock.controller.main import build

    b = build({"FLOCK_TOKEN": "t", "FLOCK_COMPUTE": "local", "FLOCK_DATA": str(tmp_path)})
    assert isinstance(b.jobs, LocalJobsClient) and b.jobs.root == tmp_path.resolve()
    with pytest.raises(SystemExit, match="FLOCK_MODE=job"):
        build({"FLOCK_TOKEN": "t", "FLOCK_COMPUTE": "local", "FLOCK_MODE": "job",
               "FLOCK_NETWORK_GROUP": "g", "FLOCK_DATA": str(tmp_path)})


# --------------------------------------------------------------------------- LocalJobsClient


def _script(code: str) -> list[str]:
    return [sys.executable, "-c", code]


DUMP_ENV = ("import json, os, sys, time; json.dump(dict(os.environ), "
            "open(os.environ['WORKER_ID'] + '.env.json', 'w')); print('hello from', "
            "os.environ['WORKER_ID'], flush=True); time.sleep(float(os.environ.get('NAP', '0'))); "
            "sys.exit(int(os.environ.get('CODE', '0')))")


def _wait(jobs: LocalJobsClient, job_id: str, stage: str, timeout: float = 10) -> None:
    end = time.monotonic() + timeout
    while jobs.inspect(job_id, namespace="") .stage != stage:
        assert time.monotonic() < end, jobs.inspect(job_id, namespace="")
        time.sleep(0.05)


def test_local_jobs_env_contract_and_stages(tmp_path):
    base = {"PATH": os.environ["PATH"], "FLOCK_URL": "http://elsewhere", "HOME": "/h"}
    jobs = LocalJobsClient(root=tmp_path, command=_script(DUMP_ENV), base_env=base)
    env = {"FLOCK_URL": "http://127.0.0.1:1", "WORKER_ID": "s-w-w0000", "HEARTBEAT_INTERVAL": "1",
           "BUCKET": "hf://buckets/me/b"}
    jid = jobs.launch("s-w-w0000", local_spec().types["w"], env, {"FLOCK_TOKEN": "tok",
                      "FAKE_KEY": "k"}, {"swarm": "s"}, 60, namespace="")
    _wait(jobs, jid, "COMPLETED")
    seen = json.loads((tmp_path / "s-w-w0000.env.json").read_text())
    assert seen["FLOCK_URL"] == "http://127.0.0.1:1"  # the controller's, not the caller's
    assert seen["FLOCK_TOKEN"] == "tok" and seen["FAKE_KEY"] == "k"  # secrets arrive as env
    assert seen["WORK_ROOT"] == str(tmp_path.resolve() / "agents")
    assert seen["FLOCK_LLM_HOME"] == str(tmp_path.resolve() / "state" / "llm")
    assert seen["FLOCK_LOCAL_PARENT_PID"] == str(os.getpid())
    assert seen["HEARTBEAT_INTERVAL"] == "1"  # the type's env wins over the local default
    assert seen["HOME"] == "/h"
    assert seen["BUCKET"] == ""  # no bucket sync locally, whatever the spec says
    assert "hello from s-w-w0000" in "\n".join(jobs.logs_tail(jid, namespace=""))
    assert [j.job_id for j in jobs.list({"swarm": "s"}, namespace="")] == [jid]
    assert jobs.list({"swarm": "other"}, namespace="") == []
    assert jobs.inspect("local-9999", namespace="").stage == "DELETED"


def test_local_jobs_error_cancel_and_timeout(tmp_path):
    jobs = LocalJobsClient(root=tmp_path, command=_script(DUMP_ENV),
                           base_env={"PATH": os.environ["PATH"]})
    ts = local_spec().types["w"]
    bad = jobs.launch("a", ts, {"WORKER_ID": "a", "CODE": "3"}, {}, {}, 60, namespace="")
    _wait(jobs, bad, "ERROR")
    assert jobs.inspect(bad, namespace="").message == "exit 3"

    slow = jobs.launch("b", ts, {"WORKER_ID": "b", "NAP": "30"}, {}, {}, 60, namespace="")
    assert jobs.inspect(slow, namespace="").stage == "RUNNING"
    jobs.cancel(slow, namespace="")
    _wait(jobs, slow, "CANCELED")

    late = jobs.launch("c", ts, {"WORKER_ID": "c", "NAP": "30"}, {}, {}, 0.2, namespace="")
    time.sleep(0.3)
    st = jobs.inspect(late, namespace="")
    assert (st.stage, st.message) == ("ERROR", "Job timeout")  # a planned restart, as on HF

    left = jobs.launch("d", ts, {"WORKER_ID": "d", "NAP": "30"}, {}, {}, 60, namespace="")
    assert jobs.alive() == [left]
    jobs.kill_all(grace=1)
    assert jobs.alive() == [] and jobs.wait_all(0)


def test_reconcile_launches_local_processes_with_the_job_env(tmp_path):
    store = Store(tmp_path / "data")
    jobs = LocalJobsClient(root=tmp_path / "run", command=_script(DUMP_ENV),
                           base_env={"PATH": os.environ["PATH"]})
    settings = Settings(public_url="http://127.0.0.1:5", token=TOKEN, rng=random.Random(0),
                        secret_source={"FAKE_KEY": "k"})
    spec = local_spec(agents_per_worker=1)
    store.put_swarm(spec, time.time())
    acts = reconcile(store, jobs, datetime.now(UTC), settings)
    launched = sorted(a.target for a in acts if a.kind == "launch")
    assert launched == ["loc-w-w0000", "loc-w-w0001"]
    for wid in launched:
        end = time.monotonic() + 10
        while not (tmp_path / "run" / f"{wid}.env.json").exists():
            assert time.monotonic() < end
            time.sleep(0.05)
        time.sleep(0.1)
        env = json.loads((tmp_path / "run" / f"{wid}.env.json").read_text())
        assert env["WORKER_ID"] == wid and env["SWARM"] == "loc" and env["TYPE"] == "w"
        assert env["FLOCK_URL"] == "http://127.0.0.1:5" and env["FLOCK_TOKEN"] == TOKEN
        assert env["FAKE_KEY"] == "k" and env["BUCKET"] == "" and env["WORKER_INCARNATION"]
    jobs.kill_all(grace=1)
    store.close()


# --------------------------------------------------------------------------- worker side


def test_worker_reads_parent_pid_and_exits_when_it_is_gone(tmp_path):
    env = {"FLOCK_URL": "http://x", "FLOCK_TOKEN": "t", "WORKER_ID": "w", "SWARM": "s",
           "TYPE": "a", "DEADLINE": "2030-01-01T00:00:00+00:00"}
    assert WorkerConfig.from_env(env).parent_pid is None
    cfg = WorkerConfig.from_env({**env, "FLOCK_LOCAL_PARENT_PID": "1234567"})
    assert cfg.parent_pid == 1234567

    w = Worker(cfg, client=None)  # never called: abandon only stops agents
    asyncio.run(asyncio.wait_for(w.parent_loop(every=0.01), 5))  # our ppid is not 1234567
    assert w.stale and w._drained.is_set()


# --------------------------------------------------------------------------- current file


def test_current_file_roundtrip_and_dead_pid(tmp_path):
    path = write_current(tmp_path, {"url": "http://127.0.0.1:9", "token": "s3", "pid": os.getpid()})
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert read_current(path)["token"] == "s3"
    clear_current(tmp_path, os.getpid() + 1)  # someone else's run: kept
    assert path.exists()
    clear_current(tmp_path, os.getpid())
    assert not path.exists()
    write_current(tmp_path, {"url": "u", "token": "t", "pid": 2**22 + 12345})  # no such pid
    assert read_current(tmp_path / "current") is None
    assert read_current(tmp_path / "missing") is None


def test_cli_finds_a_live_run_without_flock_url(tmp_path, monkeypatch):
    write_current(tmp_path / ".flock", {"url": "http://127.0.0.1:4321", "token": "run-tok",
                                         "pid": os.getpid()})
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLOCK_URL", raising=False)
    monkeypatch.delenv("FLOCK_TOKEN", raising=False)
    monkeypatch.delenv("FLOCK_AGENT_TOKEN", raising=False)
    with cli.client() as c:
        assert str(c.base_url).rstrip("/") == "http://127.0.0.1:4321"
        assert c.headers["X-Flock-Token"] == "run-tok"
    monkeypatch.setenv("FLOCK_URL", "http://other:1")  # explicit config wins
    monkeypatch.setenv("FLOCK_TOKEN", "explicit")
    with cli.client() as c:
        assert str(c.base_url).rstrip("/") == "http://other:1"
        assert c.headers["X-Flock-Token"] == "explicit"
    monkeypatch.delenv("FLOCK_URL")
    elsewhere = tmp_path / "elsewhere"
    write_current(elsewhere, {"url": "http://127.0.0.1:99", "token": "t2", "pid": os.getpid()})
    monkeypatch.setenv("FLOCK_CURRENT", str(elsewhere / "current"))
    with cli.client() as c:
        assert str(c.base_url).rstrip("/") == "http://127.0.0.1:99"


# --------------------------------------------------------------------------- flock run pieces


def test_flock_run_refuses_a_jobs_spec_and_a_bad_until(tmp_path):
    jobs_spec = tmp_path / "jobs.yaml"
    jobs_spec.write_text(yaml.safe_dump(make_spec().model_dump(mode="json")))
    r = runner.invoke(cli.app, ["run", str(jobs_spec)])
    assert r.exit_code == 1 and "compute: local" in r.output and "flock launch" in r.output
    loc = tmp_path / "local.yaml"
    loc.write_text(yaml.safe_dump(local_spec().model_dump(mode="json")))
    r = runner.invoke(cli.app, ["run", str(loc), "--until", "soon"])
    assert r.exit_code == 2 and "--until" in r.output


def test_redacted_spec_hides_secret_looking_env():
    spec = local_spec(env={"OPENAI_API_KEY": "sk-1", "MY_TOKEN": "t", "MODE": "fast"})
    env = localrun.redacted_spec(spec)["types"]["w"]["env"]
    assert env == {"OPENAI_API_KEY": "<redacted>", "MY_TOKEN": "<redacted>", "MODE": "fast"}


def _events(store: Store, agent: str, *kinds: str) -> None:
    from flock.models import Event

    store.append_events("loc", agent, [
        Event(type=k, ts=datetime.now(UTC), agent=agent, turn_id="t-1",
              data={"data": {"status": "completed"}} if k == "turn_ended" else {})
        for k in kinds])


def test_tracker_idle_and_render(tmp_path):
    store = Store(tmp_path / "data")
    spec = local_spec()
    store.put_swarm(spec, time.time())
    reconcile(store, LocalJobsClient(root=tmp_path, command=_script("pass")),
              datetime.now(UTC), Settings(token=TOKEN, secret_source={}))
    tr = localrun.Tracker(store, "loc")
    tr.prompted = {"w-0000", "w-0001"}
    now = time.time()
    for name in ("w-0000", "w-0001"):
        store.record_agent_heartbeat(name, {"hb_state": "idle", "last_heartbeat": now}, {},
                                     lambda t: 0.0)
    for w in store.list_workers("loc"):
        store.update_worker(w["id"], last_heartbeat=now, last_seen=now, launched_at=now - 1)
    _events(store, "w-0000", "turn_started", "turn_ended")
    _events(store, "w-0001", "turn_started")
    snap = tr.snapshot(spec, now)
    assert snap["completed"] == 1 and snap["running"] == 1  # open turn counts as working
    assert not tr.idle(snap)
    _events(store, "w-0001", "turn_ended")
    snap = tr.snapshot(spec, now)
    assert snap["completed"] == 2 and tr.idle(snap)
    frame = localrun.render(spec, snap, elapsed=65, url="http://127.0.0.1:1",
                            current=tmp_path / ".flock" / "current", recent=list(tr.recent))
    assert "flock run · loc · compute local · 00:01:05" in frame
    assert "prompts  pending 0 · running 0 · completed 2" in frame
    assert "w-0001" in frame and "turn_ended" in frame
    assert "FLOCK_CURRENT=" in frame  # not ./.flock: the hint names the file
    assert "completed 2" in localrun.status_line(snap, 3)
    # an agent removed with a turn still open (an ephemeral child whose turn_ended was never
    # read): its open turn goes with its row and does not block idle (chatbot-v1 run 2)
    store.insert_agent("w-gone", "loc", "w", 9, now)
    _events(store, "w-gone", "turn_started")
    assert not tr.idle(tr.snapshot(spec, now)) and "w-gone" in tr.open_turns
    store.delete_agent_rows("w-gone")
    assert tr.idle(tr.snapshot(spec, now)) and "w-gone" not in tr.open_turns
    store.close()


def test_write_bundle(tmp_path):
    store = Store(tmp_path / "data")
    spec = local_spec(env={"X_TOKEN": "hunter2"})
    store.put_swarm(spec, time.time())
    store.insert_agent("w-0000", "loc", "w", 0, time.time())
    store.insert_agent("w-0001", "loc", "w", 1, time.time())
    _events(store, "w-0001", "turn_started", "turn_ended")
    _events(store, "w-0000", "turn_started", "turn_ended")
    tr = localrun.Tracker(store, "loc")
    t0 = datetime.now(UTC)
    totals = localrun.write_bundle(tmp_path / "run", store, spec, started=t0, ended=t0,
                                   reason="idle", url="u", tracker=tr)
    assert totals["agents"] == 2 and totals["turns_completed"] == 2 and totals["events"] == 4
    doc = json.loads((tmp_path / "run" / "run.json").read_text())
    assert doc["stop_reason"] == "idle" and doc["compute"] == "local"
    assert doc["spec"]["types"]["w"]["env"]["X_TOKEN"] == "<redacted>"
    assert "hunter2" not in (tmp_path / "run" / "run.json").read_text()
    merged = [json.loads(x) for x in (tmp_path / "run" / "events.jsonl").read_text().splitlines()]
    assert [e["agent"] for e in merged] == ["w-0001", "w-0001", "w-0000", "w-0000"]  # by time
    for a in ("w-0000", "w-0001"):
        lines = (tmp_path / "run" / "events" / f"{a}.jsonl").read_text().splitlines()
        assert [json.loads(x)["type"] for x in lines] == ["turn_started", "turn_ended"]
    store.close()


def test_bundle_counts_a_removed_agents_interrupted_turn(tmp_path):
    from flock.models import Event

    store = Store(tmp_path / "data")
    spec = local_spec()
    store.put_swarm(spec, time.time())
    store.insert_agent("p-w-0009", "loc", "w", 9, time.time())  # a spawned child, then removed
    store.append_events("loc", "p-w-0009", [Event(
        type="turn_interrupted", ts=datetime.now(UTC), agent="p-w-0009", turn_id="t-1",
        data={"data": {"reason": "terminated", "tokens": {"fake:m": {"input": 10, "output": 2}}}})])
    store.delete_agent_rows("p-w-0009")
    localrun.write_bundle(tmp_path / "run", store, spec, started=datetime.now(UTC),
                          ended=datetime.now(UTC), reason="interrupted", url="u",
                          tracker=localrun.Tracker(store, "loc"))
    (gone,) = json.loads((tmp_path / "run" / "run.json").read_text())["removed_agents"]
    assert gone["type"] == "w" and gone["cost_usd"] is not None
    assert (gone["tokens"]["fake:m"]["input"], gone["tokens"]["fake:m"]["output"]) == (10, 2)
    store.close()


def test_run_local_refuses_a_jobs_spec(tmp_path):
    opts = localrun.RunOptions(spec=make_spec(), data_dir=tmp_path, out=io.StringIO())
    with pytest.raises(ValueError, match="compute: local"):
        asyncio.run(localrun.run_local(opts))


def test_cli_hint(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert "from this directory" in localrun.cli_hint(Path(".flock"))
    assert localrun.cli_hint(tmp_path / "x").startswith("FLOCK_CURRENT=")


# --------------------------------------------------------------------------- review of #13


def test_local_env_keeps_harness_overrides_and_isolates_pi(tmp_path):
    from flock.worker.adapters.pi import PiModelConfig, write_models_json

    home = tmp_path / "home"
    user_pi = home / ".pi" / "agent"
    user_pi.mkdir(parents=True)
    (user_pi / "models.json").write_text('{"providers": {"mine": {}}}')
    base = {"HOME": str(home), "FLOCK_CLAUDE_CLI": "/c/claude", "FLOCK_CODEX_CLI": "/c/codex",
            "FLOCK_PI_CLI": "/c/pi", "FLOCK_URL": "http://mine", "FLOCK_TOKEN": "my-token",
            "FLOCK_AGENT_TOKEN": "agent", "FLOCK_CURRENT": "/x"}
    jobs = LocalJobsClient(root=tmp_path / "run", base_env=base)
    a = jobs.process_env({"WORKER_ID": "s-a-w0000", "TYPE": "a"}, {})
    b = jobs.process_env({"WORKER_ID": "s-b-w0000", "TYPE": "b"}, {})
    for k in ("FLOCK_CLAUDE_CLI", "FLOCK_CODEX_CLI", "FLOCK_PI_CLI"):
        assert a[k] == base[k]  # documented executable overrides survive
    for k in ("FLOCK_URL", "FLOCK_TOKEN", "FLOCK_AGENT_TOKEN", "FLOCK_CURRENT"):
        assert k not in a  # the caller's connection and auth never reach a worker
    pa, pb = Path(a["PI_CODING_AGENT_DIR"]), Path(b["PI_CODING_AGENT_DIR"])
    assert pa != pb and tmp_path / "run" in pa.parents and tmp_path / "run" in pb.parents
    write_models_json(pa, PiModelConfig("http://a/v1", "KA", "model-a", "a"))
    write_models_json(pb, PiModelConfig("http://b/v1", "KB", "model-b", "b"))
    assert list(json.loads((pa / "models.json").read_text())["providers"]) == ["a"]
    assert list(json.loads((pb / "models.json").read_text())["providers"]) == ["b"]
    assert json.loads((user_pi / "models.json").read_text()) == {"providers": {"mine": {}}}


def _pid_dead(pid: int) -> bool:
    from flock.controller.local import _alive

    end = time.monotonic() + 30  # a loaded host reaps slowly; the tool sleeps 600 s
    while _alive(pid) and time.monotonic() < end:
        time.sleep(0.05)
    return not _alive(pid)


def test_stop_kills_detached_descendants_of_a_stubborn_worker(tmp_path):
    # a "worker" that ignores SIGTERM and has a tool in its own session (as the llm harness
    # starts bash): the group SIGKILL misses the tool, the descendant sweep does not
    code = ("import signal, subprocess, sys, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "p = subprocess.Popen(['sleep', '600'], start_new_session=True); "
            "open('tool.pid', 'w').write(str(p.pid)); time.sleep(600)")
    jobs = LocalJobsClient(root=tmp_path, command=_script(code),
                           base_env={"PATH": os.environ["PATH"]})
    jobs.launch("w", local_spec().types["w"], {"WORKER_ID": "w"}, {}, {}, 60, namespace="")
    end = time.monotonic() + 60
    while not (tmp_path / "tool.pid").exists() or not (tmp_path / "tool.pid").read_text():
        assert time.monotonic() < end
        time.sleep(0.05)
    tool = int((tmp_path / "tool.pid").read_text())
    jobs.kill_all(grace=0.5)
    assert jobs.alive() == []
    assert _pid_dead(tool)


async def test_live_budget_put_stops_the_run_and_lands_in_the_bundle(tmp_path):
    spec = SwarmSpec.model_validate({
        "swarm": "budget", "compute": "local", "limits": {"budget_usd": 10},
        "types": {"w": {"harness": "claude", "model": "m", "replicas": 0}}})
    out = io.StringIO()
    task = asyncio.create_task(localrun.run_local(localrun.RunOptions(
        spec=spec, data_dir=tmp_path, until=30, refresh=0.1, grace=0, out=out)))
    current = tmp_path / "current"
    while not current.exists():
        assert not task.done(), out.getvalue()
        await asyncio.sleep(0.02)
    info = json.loads(current.read_text())
    async with httpx.AsyncClient(base_url=info["url"],
                                 headers={"X-Flock-Token": info["token"]}) as c:
        while (await c.get("/swarms/budget")).status_code == 404:
            await asyncio.sleep(0.02)
        live = spec.model_copy(deep=True)
        live.limits.budget_usd = 0
        live.types["w"].labels = {"applied": "live"}
        r = await c.put("/swarms/budget", json=live.model_dump(mode="json"))
        assert r.status_code == 200, r.text
    result = await asyncio.wait_for(task, 20)
    assert result.reason == "budget"
    doc = json.loads((result.run_dir / "run.json").read_text())
    assert doc["spec"]["limits"]["budget_usd"] == 0
    assert doc["spec"]["types"]["w"]["labels"] == {"applied": "live"}
