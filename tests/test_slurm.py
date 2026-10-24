"""`compute: slurm` and `compute: external` (controller/slurm.py), against shell stubs of
sbatch/squeue/sacct/scancel (tests/fixtures/slurm_stubs) that keep job states in a temp dir."""

from __future__ import annotations

import os
import random
import stat
from datetime import timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from flock import cli
from flock.controller.reconcile import Settings, reconcile, record_heartbeat
from flock.controller.slurm import (
    PREEMPT_EXIT,
    ExternalJobsClient,
    SlurmJobsClient,
    env_file_text,
    read_env_file,
    resolve,
)
from flock.models import AgentHeartbeat, ComputeSpec, SwarmSpec, WorkerHeartbeat

from .conftest import TOKEN

STUBS = Path(__file__).parent / "fixtures" / "slurm_stubs"
W0 = "demo-researcher-w0000"
SECRET_VALUES = {"FLOCK_TOKEN": TOKEN, "ANTHROPIC_API_KEY": "sk-secret-anthropic",
                 "HF_TOKEN": "hf_secret_value"}


class Slurm:
    """The stub cluster: job files under `dir/jobs/<id>.*`."""

    def __init__(self, root: Path) -> None:
        self.dir = root

    def ids(self) -> list[str]:
        return sorted(p.stem for p in (self.dir / "jobs").glob("*.state"))

    def argv(self, job_id: str) -> list[str]:
        return (self.dir / "jobs" / f"{job_id}.argv").read_text().splitlines()

    def set(self, job_id: str, state: str, exit_code: str = "0:0") -> None:
        (self.dir / "jobs" / f"{job_id}.state").write_text(state + "\n")
        (self.dir / "jobs" / f"{job_id}.exit").write_text(exit_code + "\n")

    def flag(self, job_id: str, name: str) -> None:
        (self.dir / "jobs" / f"{job_id}.{name}").touch()

    def calls(self) -> list[str]:
        p = self.dir / "calls"
        return p.read_text().splitlines() if p.exists() else []


@pytest.fixture
def slurm(tmp_path, monkeypatch) -> Slurm:
    d = tmp_path / "slurm"
    d.mkdir()
    for f in STUBS.iterdir():  # executable even if the checkout lost the mode bits
        f.chmod(f.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{STUBS}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("SLURM_STUB_DIR", str(d))
    return Slurm(d)


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
        "types": {"researcher": {"harness": "claude", "model": "claude-opus-5-5", "replicas": 2,
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
    cfg = resolve(spec.compute, "researcher", {"FLOCK_SLURM_PARTITION": "other",
                                               "FLOCK_SLURM_CPUS": "8"})
    # defaults < spec < FLOCK_SLURM_* env < per-type entry
    assert cfg["partition"] == "other" and cfg["cpus"] == "2" and cfg["mem"] == "3500M"
    assert cfg["time"] == "29:00:00" and cfg["account"] == "acct"
    assert resolve(None, "x", {})["partition"] == "hopper-cpu"
    with pytest.raises(ValueError, match="bucket is required"):
        SwarmSpec.model_validate({"swarm": "d", "compute": "slurm", "types": {}})
    with pytest.raises(ValueError, match="unknown types"):
        slurm_spec(tmp_path, types={"nope": {"cpus": 1}})
    ext = SwarmSpec.model_validate({"swarm": "d", "bucket": "hf://buckets/n/b",
                                    "compute": "external", "types": {}})
    assert ext.compute == ComputeSpec(kind="external")


# --------------------------------------------------------------------------- client


def test_launch_writes_env_file_and_sbatch_line_without_secrets(tmp_path, slurm, store, clock):
    spec = slurm_spec(tmp_path)
    store.put_swarm(spec, 0)
    jobs = SlurmJobsClient(env={})
    actions = reconcile(store, jobs, clock(), settings())
    assert [a.kind for a in actions if a.kind == "launch"] == ["launch"]
    (job_id,) = slurm.ids()
    argv = slurm.argv(job_id)
    inc = store.get_worker(W0)["incarnation"]
    env_path = tmp_path / "launch" / f"{W0}-{inc}.env"
    assert argv == [
        "--parsable", "-p", "cpu", "-A", "acct", "-c", "4", "--mem=7G", "--time=29:00:00",
        "--no-requeue", "-J", W0, f"--comment=swarm=demo,type=researcher,worker={W0}",
        "-o", f"{tmp_path / 'logs'}/{W0}-%j.out",
        "/fsx/run3/slurm/worker.sbatch", str(env_path),
    ]
    assert not any(a.startswith(("--export", "--signal")) for a in argv)
    # every line is a comment or KEY=value (the cluster's worker.sbatch sources it with set -a)
    assert all(ln.startswith("#") or ln.split("=", 1)[0].isidentifier()
               for ln in env_path.read_text().splitlines())
    text = env_path.read_text()
    assert stat.S_IMODE(env_path.stat().st_mode) == 0o600
    for value in SECRET_VALUES.values():  # no secret value in the file or on the command line
        assert value not in text and not any(value in a for a in argv)
    assert "Secrets this worker expects" in text
    for name in ("FLOCK_TOKEN", "HF_TOKEN", "ANTHROPIC_API_KEY"):
        assert name in text.splitlines()[1]
    env = read_env_file(env_path)
    assert env["FLOCK_URL"] == "http://${CTL_HOST}:7860"  # expanded by the worker
    assert env["FLOCK_CTL_HOST_FILE"] == "/fsx/run3/controller.host"
    assert env["WORKER_ID"] == W0 and env["SWARM"] == "demo" and env["TYPE"] == "researcher"
    assert env["BUCKET"] == "hf://buckets/ns/sh" and len(env["WORKER_INCARNATION"]) == 32
    assert env["WORK_ROOT"] == f"/tmp/flock/{W0}/work"
    assert env["FLOCK_LLM_HOME"] == f"/tmp/flock/{W0}/llm"
    assert env["RUN"] == tmp_path.name  # the launch dir's parent: <runs>/<RUN>/launch
    assert env["FLOCK_VENV"] == "/fsx/run3/venv"
    assert env["FLOCK_SECRETS_FILE"] == "/fsx/run3/secrets.env"
    assert env["MSGBOARD_URL"] == "https://example.invalid" and "FLOCK_TOKEN" not in env
    assert "DEADLINE" in env and env["DRAIN_WINDOW"] == "10m"


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


def test_preempted_job_relaunches_same_name_without_backoff(tmp_path, slurm, store, clock):
    spec = slurm_spec(tmp_path, types={"researcher": {"cpus": 2}})
    store.put_swarm(spec, 0)
    jobs, s = SlurmJobsClient(env={}), settings()
    reconcile(store, jobs, clock(), s)
    (first,) = slurm.ids()
    slurm.set(first, "RUNNING")
    clock.advance(30)
    heartbeat(store, clock, W0)
    slurm.set(first, "CANCELLED by 0")  # the node watcher, after SIGUSR1 and 30 s
    clock.advance(100)  # the slot goes quiet: the next pass checks the job
    actions = reconcile(store, jobs, clock(), s)
    kinds = [a.kind for a in actions]
    assert "planned_restart" in kinds and "crash" not in kinds
    w = store.get_worker(W0)
    assert w["consecutive_crashes"] == 0 and w["crashes"] == 0
    assert "launch" in kinds  # same pass: next_launch_at = now
    second = [i for i in slurm.ids() if i != first]
    assert len(second) == 1
    argv = slurm.argv(second[0])
    assert argv[argv.index("-J") + 1] == W0 and "-c" in argv and argv[argv.index("-c") + 1] == "2"


# --------------------------------------------------------------------------- external


def test_external_waits_without_backoff(tmp_path, store, clock, caplog):
    spec = SwarmSpec.model_validate({
        "swarm": "demo", "bucket": "hf://buckets/ns/sh",
        "compute": {"kind": "external", "launch_dir": str(tmp_path / "launch"),
                    "host_file": "/fsx/ctl.host"},
        "types": {"researcher": {"harness": "claude", "model": "claude-opus-5-5",
                                 "replicas": 2, "agents_per_worker": 2}},
    })
    store.put_swarm(spec, 0)
    jobs, s = ExternalJobsClient(env={}), settings()
    caplog.set_level("INFO", logger="flock.slurm")
    reconcile(store, jobs, clock(), s)
    w = store.get_worker(W0)
    assert w["job_id"].startswith("external-")
    assert "waiting for an external start of " + W0 in caplog.text
    (path,) = (tmp_path / "launch").glob(f"{W0}-*.env")
    env = read_env_file(path)
    assert env["WORKER_ID"] == W0 and env["FLOCK_URL"] == "http://${CTL_HOST}:7860"
    assert path.name == f"{W0}-{env['WORKER_INCARNATION']}.env"
    assert TOKEN not in path.read_text()
    # hours without a worker: still waiting, never a crash
    for _ in range(5):
        clock.advance(3600)
        kinds = [a.kind for a in reconcile(store, jobs, clock(), s)]
        assert "crash" not in kinds and "cancel" not in kinds
    assert store.get_worker(W0)["job_id"] == w["job_id"]
    # a worker shows up, then goes away: the slot waits again, no backoff or crash count
    heartbeat(store, clock, W0)
    clock.advance(200)
    kinds = [a.kind for a in reconcile(store, jobs, clock(), s)]
    assert "external_wait" in kinds and "planned_restart" in kinds and "crash" not in kinds
    w2 = store.get_worker(W0)
    assert w2["consecutive_crashes"] == 0 and w2["job_id"] != w["job_id"]  # new env file


def test_flock_workers_shows_external(monkeypatch, store, clock, tmp_path):
    from fastapi.testclient import TestClient

    from flock.controller.api import create_app

    spec = SwarmSpec.model_validate({
        "swarm": "demo", "bucket": "hf://buckets/ns/sh", "compute": "external",
        "types": {"researcher": {"harness": "claude", "model": "claude-opus-5-5", "replicas": 1}},
    })
    store.put_swarm(spec, 0)
    reconcile(store, ExternalJobsClient(env={}), clock(), settings())
    app = create_app(store, TOKEN, clock=clock)
    monkeypatch.setattr(cli, "client",
                        lambda: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    r = CliRunner().invoke(cli.app, ["workers", "demo"])
    assert r.exit_code == 0, r.output
    assert "external" in r.output.split(W0)[1].splitlines()[0]


def test_controller_build_selects_backend(tmp_path):
    from flock.controller.main import build

    env = {"FLOCK_TOKEN": TOKEN, "FLOCK_DATA": str(tmp_path / "d"), "FLOCK_COMPUTE": "slurm"}
    b = build(env)
    assert isinstance(b.jobs, SlurmJobsClient)
    b.store.close()
    b = build({**env, "FLOCK_COMPUTE": "external", "FLOCK_DATA": str(tmp_path / "e")})
    assert isinstance(b.jobs, ExternalJobsClient)
    b.store.close()


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


async def test_slurm_controller_accepts_only_slurm_specs(store, clock, tmp_path):
    import httpx

    from flock.controller.api import create_app

    app = create_app(store, TOKEN, clock=clock, compute="slurm")
    spec = slurm_spec(tmp_path)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        r = await c.put("/swarms/demo", json=spec.model_dump(mode="json"))
        assert r.status_code == 200, r.text
        other = spec.model_copy(update={"compute": ComputeSpec(kind="local")})
        r = await c.put("/swarms/demo", json=other.model_dump(mode="json"))
        assert r.status_code == 400 and "compute 'slurm'" in r.text


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
