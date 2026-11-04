"""`flock launch` starts what serves the run when no controller answers (`ops.up`), and `flock
stop --release` ends it after a verified checkpoint (`ops.down`): Slurm with fake sbatch /
squeue / scancel on PATH, HF Jobs with a fake HfApi, the controller in process."""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli, ops
from flock.client import Client
from flock.controller.api import create_app
from flock.ops import SpecError, read_spec
from flock.rundir import SLURM_SCRIPTS

from .conftest import TOKEN, make_spec, wire, write_spec

runner = CliRunner()
MANIFEST = {"world_api": "v1", "name": "w", "version": "1", "tools": [{"name": "act"}]}


def _commands(tmp_path: Path) -> tuple[Path, Path]:
    """Fake sbatch/squeue/scancel (the jobs start when the test says so) and their call log."""
    bin_, calls, queued = tmp_path / "bin", tmp_path / "calls", tmp_path / "queued"
    bin_.mkdir()
    queued.write_text("")
    scripts = {
        "sbatch": f"""echo "sbatch $*" >> {calls}
n=$(( $(wc -l < {queued}) + 101 ))
echo $n >> {queued}
echo "$n;cluster\"""",
        "squeue": f"""echo "squeue $*" >> {calls}
grep -qx "$3" {queued} && echo RUNNING && exit 0
echo "slurm_load_jobs error: Invalid job id specified" >&2; exit 1""",
        # a cancelled job, or a world signalled for its clean stop, ends at once
        "scancel": f"""echo "scancel $*" >> {calls}
for a in "$@"; do sed -i "s/^$a\\$/ended-$a/" {queued}; done""",
    }
    for name, body in scripts.items():
        (bin_ / name).write_text(f"#!/bin/bash\n{body}\n")
        (bin_ / name).chmod(0o755)
    return bin_, calls


def _log(calls: Path) -> list[str]:
    return [x for x in calls.read_text().splitlines() if not x.startswith("squeue")]


@pytest.fixture
def slurm_run(tmp_path, monkeypatch, store):
    """A run directory (as `flock init` leaves it) whose spec has a world command, the fake
    Slurm commands on PATH and the controller in process (its world faked)."""
    bin_, calls = _commands(tmp_path)
    monkeypatch.setenv("PATH", f"{bin_}:{__import__('os').environ['PATH']}")
    run = tmp_path / "run"
    for sub in ("hosts", "logs", "experiment"):
        (run / sub).mkdir(parents=True)
    shutil.copytree(SLURM_SCRIPTS, run / "slurm")
    (run / "run.env").write_text(f"FLOCK_TOKEN={TOKEN}\nWORLD_CPUS=8\n")
    spec = make_spec().model_dump(mode="json", exclude_none=True)
    spec["compute"] = {"kind": "slurm"}
    spec["world"] = {"command": ["{python}", "-m", "flock.world", "serve", "w:W"],
                     "url": "http://world.invalid:8000", "token_env": "WT"}
    f = run / "experiment" / "spec.yaml"
    f.write_text(yaml.safe_dump(spec))

    def checkpoint() -> int:  # the controller's StateSync.push, faked: logged in order
        with calls.open("a") as out:
            out.write("checkpoint\n")
        if (tmp_path / "bucket-down").exists():
            raise OSError("bucket 503")
        return 3

    world = httpx.MockTransport(lambda req: httpx.Response(200, json=MANIFEST))
    app = create_app(store, TOKEN, checkpoint=checkpoint, secret_source={"WT": "s"},
                     world_transport=world, loop_interval=0)
    wire(monkeypatch, app)
    # a controller answers once its job wrote hosts/controller.host
    monkeypatch.setattr(Client, "answers",
                        lambda self: (run / "hosts" / "controller.host").is_file())
    real_workers = Client.workers

    def workers(self, swarm):  # the release's drain: logged in order
        with calls.open("a") as out:
            out.write("drain\n")
        return real_workers(self, swarm)

    monkeypatch.setattr(Client, "workers", workers)
    return SimpleNamespace(run=run, spec_file=str(f), calls=calls, tmp=tmp_path)


def _start_jobs_on_first_poll(monkeypatch, run: Path) -> list[str]:
    """The submitted jobs come up while `up` waits: their host files appear at its first poll."""
    polls: list[str] = []

    def poll(_s):
        polls.append("poll")
        (run / "hosts" / "world.host").write_text("node-w:8000\n")
        (run / "hosts" / "controller.host").write_text("node-c:7999\n")

    monkeypatch.setattr(ops, "_sleep", poll)
    return polls


def test_launch_starts_controller_and_world_waits_then_creates_and_release_ends_both(
        slurm_run, monkeypatch, store):
    r = slurm_run
    api = Client.for_run(r.run, need_host=False)
    polls = _start_jobs_on_first_poll(monkeypatch, r.run)
    said: list[str] = []
    done = ops.launch(api, read_spec(r.spec_file, api.env_files), r.spec_file,
                      env_files=api.env_files, preflight=False, say=said.append)
    assert done["up"] == {"jobs": {"world": "101", "controller": "102"},
                          "url": "http://node-c:7999"}
    assert polls == ["poll"] and any("waiting for the world and controller" in s for s in said)
    assert done["status"]["swarm"] == "demo" and store.get_swarm("demo") is not None  # created
    world, ctl = _log(r.calls)
    assert world.startswith(f"sbatch --parsable -J flock-world-run -o {r.run}/logs/%x-%j.out "
                            f"-c 8 {r.run}/slurm/world.sbatch {r.run}")
    assert ctl == (f"sbatch --parsable -J flock-ctl-run -o {r.run}/logs/%x-%j.out "
                   f"{r.run}/slurm/controller.sbatch {r.run}")
    assert (r.run / "hosts" / "world.job").read_text() == "101\n"
    assert (r.run / "hosts" / "controller.job").read_text() == "102\n"

    # stop --release: drain, checkpoint, the world's clean stop, then cancel what still runs
    r.calls.write_text("")
    res = runner.invoke(cli.app, ["--run", str(r.run), "stop", "demo", "--release"])
    assert res.exit_code == 0, res.output
    assert _log(r.calls) == ["drain", "checkpoint", "scancel --signal=TERM --batch 101",
                             "scancel 102"]
    assert "ended world job 101" in res.output and "ended controller job 102" in res.output
    assert not any((r.run / "hosts").iterdir())  # launch starts new jobs, never waits on old ones

    # launch on the released run: the infrastructure comes back, the swarm resumes
    r.calls.write_text("")
    api = Client.for_run(r.run, need_host=False)
    _start_jobs_on_first_poll(monkeypatch, r.run)
    done = ops.launch(api, read_spec(r.spec_file, api.env_files), r.spec_file,
                      env_files=api.env_files)
    assert done["up"]["jobs"] == {"world": "103", "controller": "104"}
    assert "resumed" in done and "status" not in done
    assert "released" not in store.swarm_flags("demo")


def test_a_failed_final_checkpoint_is_reported_and_nothing_is_cancelled(slurm_run, monkeypatch):
    r = slurm_run
    api = Client.for_run(r.run, need_host=False)
    _start_jobs_on_first_poll(monkeypatch, r.run)
    ops.launch(api, read_spec(r.spec_file, api.env_files), r.spec_file, preflight=False)
    (r.tmp / "bucket-down").write_text("")
    r.calls.write_text("")
    res = runner.invoke(cli.app, ["--run", str(r.run), "stop", "demo", "--release"])
    assert res.exit_code == 1
    assert "final checkpoint failed: error 500" in res.output and "bucket 503" in res.output
    assert "nothing was ended" in res.output
    assert _log(r.calls) == ["drain", "checkpoint"]  # no scancel: the controller keeps its state
    assert (r.run / "hosts" / "controller.job").read_text() == "102\n"


def test_launch_waits_for_a_queued_job_instead_of_submitting_another(slurm_run, monkeypatch):
    r = slurm_run
    (r.calls.parent / "queued").write_text("101\n102\n")  # both still pending from a launch
    (r.run / "hosts" / "world.job").write_text("101\n")
    (r.run / "hosts" / "controller.job").write_text("102\n")
    api = Client.for_run(r.run, need_host=False)
    _start_jobs_on_first_poll(monkeypatch, r.run)
    done = ops.launch(api, read_spec(r.spec_file, api.env_files), r.spec_file, preflight=False)
    assert done["up"]["jobs"] == {"world": "101", "controller": "102"}
    assert _log(r.calls) == []  # no sbatch


def test_a_job_that_ends_before_it_comes_up_fails_the_launch(slurm_run, monkeypatch):
    r = slurm_run
    api = Client.for_run(r.run, need_host=False)
    monkeypatch.setattr(ops, "_sleep", lambda s: pytest.fail("no wait for a dead job"))
    (r.calls.parent / "bin" / "sbatch").write_text(  # submitted, then gone (no squeue entry)
        "#!/bin/bash\necho '7;cluster'\n")
    with pytest.raises(SpecError, match=r"world job 7 ended before it came up"):
        ops.launch(api, read_spec(r.spec_file, api.env_files), r.spec_file, preflight=False)


def test_checkpoint_route(store):
    def push(fail: str) -> int:
        if fail:
            raise OSError(fail)
        return 2

    for checkpoint, code, body in ((None, 409, "no state sync"), (lambda: push(""), 200, "2"),
                                   (lambda: push("bucket 503"), 500, "bucket 503")):
        with TestClient(create_app(store, TOKEN, checkpoint=checkpoint),
                        headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            r = c.post("/checkpoint")
            assert r.status_code == code and body in r.text


# ----------------------------------------------------------------- HF Jobs


class FakeHf:
    """HfApi's job calls; a job runs until cancelled. `log` is shared with the checkpoint."""

    def __init__(self, log: list[str]):
        self.log, self.jobs, self.kwargs = log, {}, None

    def whoami(self):
        return {"name": "ns"}

    def list_jobs(self, labels, namespace):
        return [SimpleNamespace(id=j, status=SimpleNamespace(stage=s))
                for j, s in self.jobs.items() if s == "RUNNING"]

    def run_job(self, **kw):
        self.kwargs = kw
        self.log.append("run_job")
        self.jobs["job1"] = "RUNNING"
        return SimpleNamespace(id="job1")

    def cancel_job(self, job_id, namespace):
        self.log.append(f"cancel_job {job_id} {namespace}")
        self.jobs[job_id] = "CANCELED"

    def inspect_job(self, job_id, namespace):
        return SimpleNamespace(status=SimpleNamespace(stage=self.jobs[job_id]))


def test_hf_jobs_launch_starts_the_controller_job_and_release_cancels_it(
        tmp_path, monkeypatch, store):
    log: list[str] = []
    hf = FakeHf(log)
    monkeypatch.setattr(ops, "_hf_api", lambda env: hf)
    monkeypatch.setattr(ops, "_sleep", lambda s: None)
    app = create_app(store, TOKEN, checkpoint=lambda: log.append("checkpoint") or 1)
    wire(monkeypatch, app)
    monkeypatch.setattr(Client, "answers", lambda self: "job1" in hf.jobs)
    env = {"FLOCK_TOKEN": TOKEN, "HF_TOKEN": "hf", "ANTHROPIC_API_KEY": "sk",
           "FLOCK_WHEEL_URL": "hf://buckets/ns/flock/dist/abc/flock.whl"}
    api = Client(env)
    f = write_spec(tmp_path / "spec.yaml", make_spec())
    done = ops.launch(api, read_spec(f), f, preflight=False)
    assert done["up"] == {"jobs": {"controller": "job1"}, "url": "https://job1--7860.hf.jobs"}
    assert api.env["FLOCK_URL"] == "https://job1--7860.hf.jobs" and "status" in done
    kw = hf.kwargs
    assert kw["network_group"] == "demo" and kw["namespace"] == "ns"
    assert sorted(kw["secrets"]) == ["ANTHROPIC_API_KEY", "FLOCK_TOKEN", "HF_TOKEN"]
    assert kw["env"]["FLOCK_STATE_URL"] == "hf://buckets/ns/flock/controller/demo"

    log.clear()
    out = ops.stop(api, "demo", release=True)  # the job id from FLOCK_URL
    assert log == ["checkpoint", "cancel_job job1 ns"]
    assert out["down"] == {"checkpoint": 1, "ended": ["controller job job1"]}


def test_a_controller_with_no_record_or_other_swarms_is_kept(api, tmp_path, store, clock):
    f = write_spec(tmp_path / "spec.yaml", make_spec())
    ops.launch(api, read_spec(f), f, preflight=False)
    assert ops.stop(api, "demo", release=True)["down"]["kept"].startswith("no record")
    api.env = {**api.env, "FLOCK_URL": "https://job9--7860.hf.jobs"}
    store.put_swarm(make_spec("other"), clock().timestamp())
    assert ops.down(api, "demo") == {"kept": "it also serves other"}
