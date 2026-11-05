"""`compute: slurm` (controller/slurm.py), against shell stubs of
sbatch/squeue/sacct/scancel (tests/fixtures/slurm_stubs) that keep job states in a temp dir."""

from __future__ import annotations

import os
import random
from datetime import timedelta
from pathlib import Path

import pytest

from flock.controller.reconcile import Settings, reconcile, record_heartbeat
from flock.controller.slurm import (
    PREEMPT_EXIT,
    SlurmJobsClient,
    env_file_text,
    read_env_file,
    resolve,
)
from flock.models import AgentHeartbeat, SwarmSpec, WorkerHeartbeat

from .conftest import TOKEN

W0 = "demo-researcher-w0000"
SECRET_VALUES = {"FLOCK_TOKEN": TOKEN, "ANTHROPIC_API_KEY": "sk-secret-anthropic",
                 "HF_TOKEN": "hf_secret_value"}


def slurm_spec(tmp_path: Path, **compute) -> SwarmSpec:
    block = {
        "kind": "slurm",
        "partition": "cpu",
        "account": "acct",
        "worker_script": "/fsx/run3/slurm/worker.sbatch",
        "log_dir": str(tmp_path / "logs"),
        "launch_dir": str(tmp_path / "launch"),
        "host_file": "/fsx/run3/controller.host",
        "venv": "/fsx/run3/venv",
        "secrets_file": "/fsx/run3/secrets.env",
        **compute,
    }
    return SwarmSpec.model_validate({
        "swarm": "demo", "bucket": "hf://buckets/ns/sh", "compute": block,
        "types": {"researcher": {"harness": "pi", "model": "claude-opus-5-5", "replicas": 2,
                                 "agents_per_worker": 2, "secrets": ["ANTHROPIC_API_KEY"],
                                 "resume_prompt": "Your turn was cut; continue.",
                                 "env": {"MSGBOARD_URL": "https://example.invalid"}}},
    })


def settings() -> Settings:
    return Settings(public_url="http://ctl-node-1:7860", token=TOKEN,
                    secret_source=SECRET_VALUES, rng=random.Random(0))


def heartbeat(store, clock, wid, preempted=False):
    hb = WorkerHeartbeat(worker=wid, deadline=clock() + timedelta(hours=24),
                         agents=[AgentHeartbeat(name="researcher-0000", state="idle")])
    assert record_heartbeat(store, wid, hb, clock())


# --------------------------------------------------------------------------- spec


def test_compute_spec_keys_and_resolution(tmp_path):
    spec = slurm_spec(tmp_path, types={"researcher": {"cpus": 2, "mem": "3500M"}})
    assert spec.compute.kind == "slurm" and spec.namespace == ""
    env = {"FLOCK_SLURM_PARTITION": "other", "FLOCK_SLURM_CPUS": "8", "FLOCK_SLURM_TIME": "1:00:00",
           "FLOCK_SLURM_MAX_CPUS": "64"}
    cfg = resolve(spec.compute, "researcher", env)
    # defaults < FLOCK_SLURM_* env < spec < per-type entry: the spec wins
    assert cfg["partition"] == "cpu" and cfg["cpus"] == "2" and cfg["mem"] == "3500M"
    assert cfg["time"] == "1:00:00" and cfg["account"] == "acct"  # unset in the spec: env
    assert cfg["max_cpus"] == "64"  # the spec's default is not a value it sets
    assert resolve(spec.compute, "other", env)["cpus"] == "8"
    assert "partition" not in resolve(None, "x", {})  # no built-in cluster: sbatch's default
    with pytest.raises(ValueError, match="bucket is required"):
        SwarmSpec.model_validate({"swarm": "d", "compute": "slurm", "types": {}})
    with pytest.raises(ValueError, match="unknown types"):
        slurm_spec(tmp_path, types={"nope": {"cpus": 1}})


# --------------------------------------------------------------------------- client


def test_no_partition_leaves_it_to_the_cluster(tmp_path, slurm):
    # `partition: ${SLURM_PARTITION:-}` with no profile renders "": no -p, sbatch's default
    spec = slurm_spec(tmp_path, partition="")
    job = SlurmJobsClient(env={}).launch(W0, spec.types["researcher"], {"TYPE": "researcher"},
                                         {}, {"worker": W0}, 3600, namespace="",
                                         compute=spec.compute)
    assert slurm.argv(job)[:3] == ["--parsable", "-A", "acct"]


def test_state_mapping_lifecycle(tmp_path, slurm):
    jobs = SlurmJobsClient(env={})
    spec = slurm_spec(tmp_path)
    ts = spec.types["researcher"]
    job = jobs.launch(W0, ts, {"TYPE": "researcher", "FLOCK_URL": "http://h:7860",
                               "WORKER_ID": W0}, {}, {"worker": W0}, 3600, namespace="",
                      compute=spec.compute)
    assert jobs.inspect(job).stage == "SCHEDULING"  # PENDING (autoscaling): never a failure
    slurm.set(job, "RUNNING")
    assert jobs.inspect(job).stage == "RUNNING"
    slurm.set(job, "PREEMPTED")
    st = jobs.inspect(job)
    assert (st.stage, st.preempted) == ("ERROR", True) and "PREEMPTED" in st.message
    slurm.set(job, "TIMEOUT")
    assert (jobs.inspect(job).message, jobs.inspect(job).preempted) == ("Job timeout", False)
    slurm.set(job, "FAILED", "1:0")
    st = jobs.inspect(job)
    assert (st.stage, st.message, st.preempted) == ("ERROR", "FAILED exit 1", False)
    slurm.set(job, "FAILED", f"{PREEMPT_EXIT}:0")  # the worker's own preemption exit code
    assert jobs.inspect(job).preempted
    slurm.set(job, "CANCELLED by 0")  # a Spot reclaim: nobody at flock asked for it
    st = jobs.inspect(job)
    assert st.preempted and "not requested" in st.message
    slurm.set(job, "COMPLETED")
    slurm.flag(job, "purged")  # gone from squeue: sacct answers
    assert jobs.inspect(job).stage == "COMPLETED"
    slurm.flag(job, "lag")  # in neither yet, right after sbatch: still scheduling
    assert jobs.inspect(job).stage == "SCHEDULING"
    assert SlurmJobsClient(env={}).inspect(job).stage == "DELETED"  # unknown and not ours


def test_cancel_requested_is_canceled_also_after_restart(tmp_path, slurm):
    spec = slurm_spec(tmp_path)
    jobs = SlurmJobsClient(env={})
    job = jobs.launch(W0, spec.types["researcher"], {"TYPE": "researcher", "FLOCK_URL": "x"},
                      {}, {"worker": W0}, 60, namespace="", compute=spec.compute)
    slurm.set(job, "RUNNING")
    jobs.cancel(job)
    assert f"scancel {job}" in slurm.calls()
    st = jobs.inspect(job)
    assert (st.stage, st.preempted) == ("CANCELED", False)
    # a restarted controller knows its cancels from the markers in the launch dir
    again = SlurmJobsClient(env={"FLOCK_SLURM_LAUNCH_DIR": str(tmp_path / "launch")})
    assert again.inspect(job).stage == "CANCELED"


def test_list_and_logs_tail(tmp_path, slurm):
    spec = slurm_spec(tmp_path)
    jobs = SlurmJobsClient(env={})
    job = jobs.launch(W0, spec.types["researcher"],
                      {"TYPE": "researcher", "FLOCK_URL": "x", "DEADLINE": "2026-10-15T09:00:00+00:00"},
                      {}, {"swarm": "demo", "type": "researcher", "worker": W0}, 60,
                      namespace="", compute=spec.compute)
    slurm.set(job, "RUNNING")
    (listing,) = jobs.list({"swarm": "demo"})
    assert listing.job_id == job and listing.stage == "RUNNING"
    assert listing.labels["worker"] == W0 and listing.env["DEADLINE"].startswith("2026-10-15")
    assert jobs.list({"swarm": "other"}) == []
    (tmp_path / "logs" / f"{W0}-{job}.out").write_text("a\nb\nTraceback: boom\n")
    assert jobs.logs_tail(job, n=2) == ["b", "Traceback: boom"]


def test_launch_needs_script_and_dirs(tmp_path, slurm, store, clock):
    spec = slurm_spec(tmp_path, worker_script=None)
    store.put_swarm(spec, 0)
    actions = reconcile(store, SlurmJobsClient(env={}), clock(), settings())
    refused = [a for a in actions if a.kind == "launch_refused"]
    assert refused and "needs worker_script" in refused[0].detail
    assert slurm.ids() == []


def test_env_file_quoting_round_trips(tmp_path):
    import subprocess

    env = {"A": "plain", "B": "http://${CTL_HOST}:7860", "C": "it's a b", "D": ""}
    path = tmp_path / "x.env"
    path.write_text(env_file_text(env, ["FLOCK_TOKEN"]))
    assert read_env_file(path) == env
    script = f'set -a; . {path}; set +a; printf "%s|" "$A" "$B" "$C" "$D"'
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                         env={"PATH": os.environ["PATH"]})
    assert out.stdout == "plain|http://${CTL_HOST}:7860|it's a b||"


def test_space_mode_honours_worker_url():
    from flock.controller.main import deployment

    d = deployment({"FLOCK_URL": "http://ip-10-0-0-1:7860",
                    "FLOCK_WORKER_URL": "http://${CTL_HOST}:7860"})
    assert d.mode == "space" and d.public_url == "http://ip-10-0-0-1:7860"
    assert d.worker_url == "http://${CTL_HOST}:7860"
    assert deployment({"FLOCK_URL": "http://h:1"}).worker_url == "http://h:1"


# --------------------------------------------------------------------------- planned restarts


async def test_preempt_flag_cancelled_relaunch_same_name_registers_fresh(tmp_path, slurm, store,
                                                                        clock):
    """launch -> SIGUSR1 drain (flag) -> CANCELLED -> relaunch same name -> 200; the old
    incarnation gets 409. No backoff, no crash count, the cut turn resumes."""
    import httpx

    from flock.controller.api import create_app
    from flock.models import AssignCommand

    spec = slurm_spec(tmp_path)
    store.put_swarm(spec, 0)
    jobs, s = SlurmJobsClient(env={}), settings()
    reconcile(store, jobs, clock(), s)
    (old_job,) = slurm.ids()
    old_inc = store.get_worker(W0)["incarnation"]
    slurm.set(old_job, "RUNNING")
    app = create_app(store, TOKEN, clock=clock)

    def hb(**kw):
        return WorkerHeartbeat(worker=W0, deadline=clock() + timedelta(hours=24), agents=[
            AgentHeartbeat(name="researcher-0000", state="working", turn_id="t1"),
            AgentHeartbeat(name="researcher-0001", state="idle")], **kw).model_dump(mode="json")

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        url = f"/workers/{W0}/heartbeat"
        r = await c.post(url, json=hb(), params={"incarnation": old_inc})
        assert r.status_code == 200
        clock.advance(10)
        # the worker's last heartbeat after its drain: synced, exiting
        last = hb(preempted=True, draining=True)
        last["agents"][0].update(state="idle", turn_id=None)
        last["interrupted"] = ["researcher-0000"]
        r = await c.post(url, json=last, params={"incarnation": old_inc})
        assert r.status_code == 200
        actions = reconcile(store, jobs, clock(), s)  # the very next pass, no 90 s wait
        kinds = [a.kind for a in actions]
        assert "planned_restart" in kinds and "launch" in kinds and "crash" not in kinds
        assert f"scancel {old_job}" in slurm.calls()
        new_job = next(i for i in slurm.ids() if i != old_job)
        argv = slurm.argv(new_job)
        assert argv[argv.index("-J") + 1] == W0
        w = store.get_worker(W0)
        assert w["job_id"] == new_job and w["incarnation"] != old_inc
        assert (w["crashes"], w["consecutive_crashes"], w["preempted"]) == (0, 0, 0)
        # the old job ends CANCELLED (ours, now): it does not disturb the new slot
        assert jobs.inspect(old_job).stage == "CANCELED"
        # a late request from the old process: 409, it exits without syncing
        r = await c.post(url, json=hb(), params={"incarnation": old_inc})
        assert r.status_code == 409
        # the relaunched worker registers fresh
        slurm.set(new_job, "RUNNING")
        r = await c.post(url, json=hb(), params={"incarnation": w["incarnation"]})
        assert r.status_code == 200
    assigns = [AssignCommand.model_validate_json(c["body"]) for c in store.commands_for(W0)
               if '"assign"' in c["body"]]
    by = {a.agent: a for a in assigns}
    assert by["researcher-0000"].restore and by["researcher-0000"].resume_prompt
    assert by["researcher-0001"].resume_prompt is None


async def test_reused_worker_id_while_old_job_runs_stays_409(tmp_path, slurm, store, clock):
    import httpx

    from flock.controller.api import create_app

    store.put_swarm(slurm_spec(tmp_path), 0)
    jobs = SlurmJobsClient(env={})
    reconcile(store, jobs, clock(), settings())
    (job,) = slurm.ids()
    slurm.set(job, "RUNNING")
    w = store.get_worker(W0)
    app = create_app(store, TOKEN, clock=clock)
    body = WorkerHeartbeat(worker=W0, deadline=clock() + timedelta(hours=1),
                           agents=[]).model_dump(mode="json")
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = await c.post(f"/workers/{W0}/heartbeat", json=body,
                         params={"incarnation": "f" * 32})
        assert r.status_code == 409
    clock.advance(30)
    reconcile(store, jobs, clock(), settings())
    assert store.get_worker(W0)["job_id"] == job and store.get_worker(W0)["incarnation"] == \
        w["incarnation"]


def test_spot_churn_never_crashloops_and_timeout_is_planned(tmp_path, slurm, store, clock):
    store.put_swarm(slurm_spec(tmp_path), 0)
    jobs, s = SlurmJobsClient(env={}), settings()
    reconcile(store, jobs, clock(), s)
    for i in range(7):  # more than CRASHLOOP_THRESHOLD reclaims in a row
        job = store.get_worker(W0)["job_id"]
        slurm.set(job, "RUNNING")
        clock.advance(30)
        heartbeat(store, clock, W0)
        slurm.set(job, "TIMEOUT" if i == 3 else "CANCELLED by 0")
        clock.advance(100)
        kinds = [a.kind for a in reconcile(store, jobs, clock(), s)]
        assert "planned_restart" in kinds and "launch" in kinds, kinds
    w = store.get_worker(W0)
    assert w["crashes"] == 0 and store.type_health("demo", "researcher") == "ok"
    # a real failure is still a crash, with backoff
    job = w["job_id"]
    slurm.set(job, "FAILED", "1:0")
    clock.advance(100)
    kinds = [a.kind for a in reconcile(store, jobs, clock(), s)]
    assert "crash" in kinds and "launch" not in kinds


# --------------------------------------------------------------------------- hardening


def test_size_guard_at_apply_and_at_launch(tmp_path, slurm, store, clock):
    from flock.models import mem_mb

    assert mem_mb("54G") == 54 * 1024 and mem_mb("3500M") == 3500 and mem_mb("1T") == 1024**2
    assert mem_mb("512") == 512 and mem_mb("2048K") == 2
    node = {"max_cpus": 28, "max_mem": "54G"}
    with pytest.raises(ValueError, match="cpus 32 is over max_cpus 28"):
        slurm_spec(tmp_path, cpus=32, **node)
    with pytest.raises(ValueError, match="mem 60G is over max_mem 54G"):
        slurm_spec(tmp_path, types={"researcher": {"mem": "60G"}}, **node)
    with pytest.raises(ValueError, match="expected a number"):
        slurm_spec(tmp_path, mem="lots")
    assert slurm_spec(tmp_path, cpus=32, max_cpus=64).compute.cpus == 32
    assert slurm_spec(tmp_path, cpus=99).compute.cpus == 99  # no node limits: nothing checked
    # an env override that does not fit the profile's node limits: refused at launch
    store.put_swarm(slurm_spec(tmp_path), 0)
    actions = reconcile(store, SlurmJobsClient(env={"FLOCK_SLURM_MEM": "100G",
                                                    "FLOCK_SLURM_MAX_MEM": "54G"}), clock(),
                        settings())
    refused = [a for a in actions if a.kind == "launch_refused"]
    assert refused and "max_mem" in refused[0].detail and slurm.ids() == []


def test_crash_log_tail_survives_a_controller_restart(tmp_path, slurm, store, clock):
    store.put_swarm(slurm_spec(tmp_path), 0)
    s = settings()
    reconcile(store, SlurmJobsClient(env={}), clock(), s)
    (job,) = slurm.ids()
    w = store.get_worker(W0)
    assert w["log_path"] == str(tmp_path / "logs" / f"{W0}-{job}.out")
    env = read_env_file(next((tmp_path / "launch").glob(f"{W0}-*.env")))  # no container_root:
    assert (env["WORK_ROOT"], env["FLOCK_LLM_HOME"]) == (  # node-local, fixed per slot
        f"/tmp/flock/{W0}/work", f"/tmp/flock/{W0}/llm") and "HOME" not in env
    (tmp_path / "logs" / f"{W0}-{job}.out").write_text("start\nImportError: no module x\n")
    slurm.set(job, "FAILED", "1:0")
    clock.advance(100)
    fresh = SlurmJobsClient(env={})  # a requeued controller: nothing in memory
    assert fresh.logs_tail(job) == []
    kinds = [a.kind for a in reconcile(store, fresh, clock(), s)]
    assert "crash" in kinds
    assert "ImportError: no module x" in store.get_worker(W0)["last_error"]
    # with FLOCK_SLURM_LOG_DIR the client finds it on its own too
    again = SlurmJobsClient(env={"FLOCK_SLURM_LOG_DIR": str(tmp_path / "logs")})
    assert again.logs_tail(job)[-1] == "ImportError: no module x"


def test_unknown_compute_kind_is_a_hard_error(tmp_path, caplog):
    from flock.controller.main import build

    env = {"FLOCK_TOKEN": TOKEN, "FLOCK_DATA": str(tmp_path / "d"), "FLOCK_COMPUTE": "slrum"}
    with pytest.raises(SystemExit, match="FLOCK_COMPUTE='slrum' is not one of"):
        build(env)
    assert not (tmp_path / "d").exists()  # refused before anything was opened
    caplog.set_level("INFO", logger="flock.controller")
    b = build({**env, "FLOCK_COMPUTE": "slurm"})
    assert "compute slurm: SlurmJobsClient" in caplog.text
    b.store.close()


def test_container_root_puts_every_path_under_it(tmp_path, slurm, store, clock):
    """Workers in a container (RAM /tmp): the env file moves every agent/harness path under
    container_root; per type too."""
    spec = slurm_spec(tmp_path, container_root="/work")
    store.put_swarm(spec, 0)
    reconcile(store, SlurmJobsClient(env={}), clock(), settings())
    (path,) = (tmp_path / "launch").glob(f"{W0}-*.env")
    env = read_env_file(path)
    base = f"/work/flock/{W0}"
    assert {k: env[k] for k in ("WORK_ROOT", "FLOCK_LLM_HOME", "HOME", "HF_HOME", "TMPDIR")} == {
        "WORK_ROOT": f"{base}/work", "FLOCK_LLM_HOME": f"{base}/llm", "HOME": f"{base}/home",
        "HF_HOME": f"{base}/hf", "TMPDIR": f"{base}/tmp"}
    assert not any(v.startswith("/tmp") for v in env.values())
    cfg = resolve(slurm_spec(tmp_path, types={"researcher": {"container_root": "/c"}}).compute,
                  "researcher", {})
    assert cfg["container_root"] == "/c"
