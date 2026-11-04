"""flock.rundir: `flock init RUN_DIR --from FOLDER` (files, run.env, checks with fakes, no
network) and `flock --run RUN_DIR` / FLOCK_RUN_DIR."""

from __future__ import annotations

import os
import stat

import pytest
import yaml
from typer.testing import CliRunner

from flock import cli, rundir
from flock.client import Client
from flock.models import PreflightReport, PreflightResult
from flock.ops import read_spec
from flock.specrender import read_env_file

runner = CliRunner()
SPEC = {
    "swarm": "${SWARM:-demo}",
    "namespace": "ns",
    "bucket": "hf://buckets/ns/sh",
    "world": {"url": "http://${WORLD_HOST}:7860", "token_env": "CHATBOT_WORLD_TOKEN"},
    "types": {"researcher": {
        "harness": "claude", "model": "claude-opus-5-5", "replicas": 2, "agents_per_worker": 2,
        "secrets": ["ANTHROPIC_API_KEY"], "system_prompt_file": "prompts/system.md",
        "env": {"MSGBOARD_URL": "${BOARD_URL:-https://board.invalid}"}}},
}


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No Hub token from this machine; the network checks report what they would."""
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)


def _folder(tmp_path, spec=SPEC):
    f = tmp_path / "exp"
    (f / "prompts").mkdir(parents=True)
    (f / "spec.yaml").write_text(yaml.safe_dump(spec))
    (f / "prompts" / "system.md").write_text("You research.\n")
    return f


def _by_name(rep):
    return {c.name: c for c in rep.checks}


def test_init_creates_everything_then_changes_nothing(tmp_path):
    src, run = _folder(tmp_path), tmp_path / "runs" / "r1"
    rep = rundir.init(run, src, env={})
    assert (run / "logs").is_dir() and (run / "hosts").is_dir()
    assert not (run / "slurm").exists()  # job scripts for `compute: slurm` only
    assert (run / "experiment" / "prompts" / "system.md").read_text() == "You research.\n"
    env_file = run / "run.env"
    assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
    values = read_env_file(env_file)
    assert len(values["FLOCK_TOKEN"]) > 30 and values["FLOCK_WORLD_TOKEN"] != values["FLOCK_TOKEN"]
    assert values["CHATBOT_WORLD_TOKEN"] == values["FLOCK_WORLD_TOKEN"]  # world.token_env
    text = env_file.read_text()
    assert "# swarm (default: demo)\n#SWARM=demo\n" in text  # a default stays the spec's
    assert "#ANTHROPIC_API_KEY=" in text and "#HF_TOKEN=" in text  # secrets named, unset
    checks = _by_name(rep)
    assert checks["tokens"].ok and checks["spec"].ok and checks["worker slots"].ok
    assert checks["bucket"].ok is None and "no HF token" in checks["bucket"].detail
    assert checks["providers"].ok is None and checks["board"].ok is None
    assert checks["world"].ok is None and checks["venv"].ok and not rep.failed

    before = env_file.read_text()
    (run / "experiment" / "prompts" / "system.md").write_text("Edited.\n")
    env_file.write_text(before.replace("#SWARM=demo", "SWARM=mine"))
    again = rundir.init(run, src, env={})  # the doctor
    assert again.created == [] and env_file.read_text() == before.replace("#SWARM=demo", "SWARM=mine")
    assert (run / "experiment" / "prompts" / "system.md").read_text() == "Edited.\n"
    assert any("(kept): edited prompts/system.md" in n for n in again.notes)
    assert _by_name(again)["spec"].detail.startswith("mine:")
    assert rundir.init(run, env={}).created == []  # --from not needed once copied


def test_empty_experiment_folder_is_filled_and_missing_source_refused(tmp_path):
    src, run = _folder(tmp_path), tmp_path / "r"
    (run / "experiment").mkdir(parents=True)
    with pytest.raises(ValueError, match="--from"):
        rundir.init(run, env={})
    rundir.init(run, src, env={})
    assert (run / "experiment" / "spec.yaml").is_file()


def test_secrets_fill_without_echo_and_never_overwrite(tmp_path, monkeypatch):
    src, run = _folder(tmp_path), tmp_path / "r"
    sec = tmp_path / "secrets.env"
    sec.write_text("ANTHROPIC_API_KEY=sk-very-secret\nFLOCK_TOKEN=shared-token\nOTHER=x y\n")
    sec.chmod(0o600)
    monkeypatch.setattr(rundir, "check_venv", lambda env: rundir.Check("venv", True))
    r = runner.invoke(cli.app, ["init", str(run), "--from", str(src), "--secrets", str(sec)])
    values = read_env_file(run / "run.env")
    assert values["ANTHROPIC_API_KEY"] == "sk-very-secret" and values["FLOCK_TOKEN"] != "shared-token"
    for secret in ("sk-very-secret", "shared-token", values["FLOCK_TOKEN"],
                   values["FLOCK_WORLD_TOKEN"]):
        assert secret not in r.output
    assert "ANTHROPIC_API_KEY (from --secrets)" in r.output and "not written: OTHER" in r.output
    assert r.exit_code == 0, r.output

    sec.write_text("ANTHROPIC_API_KEY=sk-other\n")
    r = runner.invoke(cli.app, ["init", str(run), "--secrets", str(sec)])
    assert read_env_file(run / "run.env")["ANTHROPIC_API_KEY"] == "sk-very-secret"
    assert "ANTHROPIC_API_KEY: kept, differs" in r.output and "sk-other" not in r.output

    sec.chmod(0o644)
    r = runner.invoke(cli.app, ["init", str(run), "--secrets", str(sec)])
    assert r.exit_code == 1 and "FAIL secrets file" in r.output


def test_spec_check_fails_on_an_empty_required_var(tmp_path):
    t = SPEC["types"]["researcher"]
    spec = {**SPEC, "types": {"researcher": {**t, "env": {"NOTE": "${NS}"}}}}
    run = tmp_path / "r"
    rep = rundir.init(run, _folder(tmp_path, spec), env={})
    assert not _by_name(rep)["spec"].ok and "NS" in " ".join(_by_name(rep)["spec"].lines)
    assert rep.failed and "#NS=\n" in (run / "run.env").read_text()
    (run / "run.env").write_text((run / "run.env").read_text().replace("#NS=", "NS="))
    assert _by_name(rundir.init(run, env={}))["spec"].detail == "empty, no default: NS"
    (run / "run.env").write_text((run / "run.env").read_text().replace("NS=", "NS=ns"))
    assert _by_name(rundir.init(run, env={}))["spec"].ok


def _spec(tmp_path, spec=SPEC):
    run = tmp_path / "r"
    rundir.init(run, _folder(tmp_path, spec), env={})
    return read_spec(str(run / "experiment" / "spec.yaml"), [str(run / "run.env")])


def test_slots_count_ramp_max_and_spawned_children(tmp_path):
    t = SPEC["types"]["researcher"]
    spec = {**SPEC, "types": {
        "lead": {**t, "replicas": 0, "workers": {"initial": 1},
                 "replicas_ramp": {"start": 1, "add": 1, "every": "1m", "max": 3}},
        "helper": {**t, "replicas": 0, "workers": {"initial": 0, "max_extra": 2}}},
        "spawn": {"lead": {"types": {"helper": 2}}}}
    c = rundir.check_slots(_spec(tmp_path, spec))
    assert c.ok is False and sorted(c.lines) == [
        "OVER: type helper: 6 agents planned, 2 workers x 2 = 4 slots",
        "OVER: type lead: 3 agents planned, 1 workers x 2 = 2 slots"]


def test_tokens_must_match(tmp_path):
    data = {"world": {"token_env": "CHATBOT_WORLD_TOKEN"}}
    ok = {"FLOCK_TOKEN": "a", "FLOCK_WORLD_TOKEN": "w", "CHATBOT_WORLD_TOKEN": "w"}
    assert rundir.check_tokens(data, ok).ok
    assert not rundir.check_tokens(data, {**ok, "CHATBOT_WORLD_TOKEN": "x"}).ok
    assert "FLOCK_TOKEN" in rundir.check_tokens(data, {**ok, "FLOCK_TOKEN": ""}).detail


class FakeBucket:
    def __init__(self, found=None, fail=False):
        self.found, self.fail, self.ops = found, fail, []

    def upload(self, files):
        if self.fail:
            raise PermissionError("403 read-only token")
        self.ops += [("up", r) for _, r in files]

    def delete(self, paths):
        self.ops += [("del", p) for p in paths]

    def first(self, prefix):
        return self.found and prefix + self.found


def test_bucket_writable_reused_and_read_only(tmp_path, monkeypatch):
    spec = _spec(tmp_path)
    b = FakeBucket()
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: b)
    rep = rundir.InitReport()
    assert rundir.check_bucket(spec, {"HF_TOKEN": "t"}, rep).ok and rep.notes == []
    assert b.ops == [("up", rundir.PROBE), ("del", rundir.PROBE)]
    b.found = "a-0001/base.tar"
    assert rundir.check_bucket(spec, {"HF_TOKEN": "t"}, rep).ok
    assert "already holds data for swarm demo (demo/a-0001/base.tar)" in rep.notes[0]
    b.fail = True
    c = rundir.check_bucket(spec, {"HF_TOKEN": "t"}, rep)
    assert c.ok is False and "403" in c.detail
    assert rundir.check_bucket(spec, {}, rep).ok is None


def _result(ok, unverified=False):
    return PreflightResult(model="p:m", provider="p", types=["researcher"], api="responses",
                           ok=ok, unverified=unverified, latency_ms=5,
                           error=None if ok else "HTTP 500")


@pytest.mark.parametrize("ok", [True, False])
def test_providers_reuse_the_preflight_probe(tmp_path, monkeypatch, ok):
    spec = _spec(tmp_path)
    monkeypatch.setattr("flock.ops.provider_backed", lambda s: True)

    async def fake(s, secrets, client):
        assert secrets["K"] == "v"
        return PreflightReport(swarm=s.swarm, ok=ok, results=[_result(ok)])

    monkeypatch.setattr("flock.controller.preflight.run_preflight", fake)
    c = rundir.check_providers(spec, {"K": "v"})
    assert c.ok is ok and c.lines == (["p:m: ok 5 ms"] if ok else ["p:m: FAIL 5 ms (HTTP 500)"])


def test_board_checks_the_invite_and_health_without_joining(tmp_path, monkeypatch):
    import httpx

    spec = _spec(tmp_path, {**SPEC, "comms": {"kind": "msgboard", "url": "https://b.invalid/x/",
                                              "invite_env": "INV", "channel": "general"}})
    seen = []
    monkeypatch.setattr(httpx, "get", lambda url, timeout: seen.append(url) or httpx.Response(200))
    monkeypatch.setattr("flock.controller.comms.resolve", lambda *a, **k: pytest.fail("joined"))
    c = rundir.check_board(spec, {"INV": "code"})
    assert c.ok and "join not tested" in c.detail and "code" not in c.detail
    assert seen == ["https://b.invalid/x/v1/health"]
    assert rundir.check_board(spec, {}).ok is False and seen == ["https://b.invalid/x/v1/health"]
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(503))
    assert rundir.check_board(spec, {"INV": "code"}).ok is False


def test_world_dir_made_absolute_for_the_copy(tmp_path):
    src = tmp_path / "repo" / "experiments" / "exp"
    (src / "prompts").mkdir(parents=True)
    (tmp_path / "repo" / "worlds" / "w").mkdir(parents=True)
    spec = {**SPEC, "world": {**SPEC["world"], "dir": "${WORLD_DIR:-../../worlds/w}",
                              "command": ["{python}", "w.py"]}}
    (src / "spec.yaml").write_text(yaml.safe_dump(spec))
    (src / "prompts" / "system.md").write_text("x\n")
    run = tmp_path / "runs" / "r"
    rep = rundir.init(run, src, env={})
    assert read_env_file(run / "run.env")["WORLD_DIR"] == str(tmp_path / "repo" / "worlds" / "w")
    assert _by_name(rep)["spec"].ok, _by_name(rep)["spec"].lines
    assert rundir.init(run, src, env={}).created == []


def test_a_spec_failure_still_reports_the_other_checks(tmp_path):
    rep = rundir.init(tmp_path / "r", _folder(tmp_path, {**SPEC, "namespace": "${NS}"}), env={})
    checks = _by_name(rep)
    assert checks["spec"].ok is False and checks["tokens"].ok and checks["venv"].ok
    assert checks["bucket"].detail == "skipped: needs the spec" and checks["world"].ok is None


def test_existing_open_run_env_is_made_private(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "run.env").write_text("")
    (run / "run.env").chmod(0o644)
    rundir.init(run, _folder(tmp_path), env={})
    assert stat.S_IMODE((run / "run.env").stat().st_mode) == 0o600
    assert read_env_file(run / "run.env")["FLOCK_TOKEN"]


def test_venv_must_import_the_worker(tmp_path):
    assert rundir.check_venv({}).ok
    c = rundir.check_venv({"WORKER_VENV": str(tmp_path / "nope"), "VENV": "/other"})
    assert c.ok is False and "nope/bin/python" in c.detail
    assert "other/bin/python" in rundir.check_venv({"VENV": "/other"}).detail


def test_world_conformance_when_up(tmp_path, monkeypatch):
    import httpx

    spec = _spec(tmp_path)
    run = tmp_path / "r"
    assert "no " in rundir.check_world(spec, {}, run).detail  # no hosts/world.host yet
    (run / "hosts" / "world.host").write_text("node7\n")
    calls = []

    def down(url, timeout):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(httpx, "get", down)
    assert rundir.check_world(spec, {}, run).ok is None
    monkeypatch.setattr(httpx, "get", lambda url, timeout: calls.append(url))
    for failed in (False, True):
        monkeypatch.setattr("flock.world.doctor.doctor",
                            lambda url, token, f=failed: (calls.append((url, token)) or ["manifest ok"], f))
        c = rundir.check_world(spec, {"CHATBOT_WORLD_TOKEN": "w"}, run)
        assert c.ok is (not failed) and c.lines == ["manifest ok"]
    assert calls[:2] == ["http://node7:7860/healthz", ("http://node7:7860", "w")]


def test_run_dir_resolution(tmp_path, monkeypatch):
    run = tmp_path / "r"
    (run / "hosts").mkdir(parents=True)
    (run / "run.env").write_text("FLOCK_TOKEN=run-tok\n")
    assert rundir.run_context(run) == ({"FLOCK_RUN_DIR": str(run), "FLOCK_TOKEN": "run-tok"},
                                       [str(run / "run.env")])
    (run / ".flock").mkdir()
    (run / ".flock" / "current").write_text("{}")
    assert rundir.run_context(run)[0]["FLOCK_CURRENT"] == str(run / ".flock" / "current")
    (run / "hosts" / "controller.host").write_text("node3\n")
    assert rundir.run_context(run)[0]["FLOCK_URL"] == "http://node3:7860"
    (run / "hosts" / "controller.host").write_text("https://abc--7860.hf.jobs\n")  # an HF Job's
    assert rundir.run_context(run)[0]["FLOCK_URL"] == "https://abc--7860.hf.jobs"
    (run / "hosts" / "controller.host").write_text("node3:7999\n")
    assert rundir.run_context(run)[0]["FLOCK_URL"] == "http://node3:7999"

    seen = []
    monkeypatch.setattr(Client, "swarms", lambda self: seen.append(
        (self.env.get("FLOCK_URL"), self.env.get("FLOCK_TOKEN"))) or [])
    monkeypatch.setenv("FLOCK_URL", "http://shell:1")  # restored by monkeypatch at teardown
    monkeypatch.setenv("FLOCK_TOKEN", "shell-tok")
    monkeypatch.setenv("FLOCK_RUN_DIR", str(run))
    assert runner.invoke(cli.app, ["status"]).exit_code == 0  # FLOCK_RUN_DIR loses to FLOCK_URL
    assert runner.invoke(cli.app, ["--run", str(run), "status"]).exit_code == 0  # --run wins
    monkeypatch.delenv("FLOCK_URL")
    assert runner.invoke(cli.app, ["status"]).exit_code == 0
    assert seen == [("http://shell:1", "shell-tok"), ("http://node3:7999", "run-tok"),
                    ("http://node3:7999", "run-tok")]
    (run / "hosts" / "controller.host").unlink()
    (run / ".flock" / "current").unlink()
    monkeypatch.setenv("FLOCK_URL", "http://shell:1")
    r = runner.invoke(cli.app, ["--run", str(run), "status"])
    assert r.exit_code == 1 and "no controller host file" in r.output
    assert os.environ["FLOCK_URL"] == "http://shell:1" and len(seen) == 3  # not mutated


def test_run_env_is_the_first_env_file(tmp_path, monkeypatch):
    monkeypatch.setenv("FLOCK_TOKEN", "t")  # --run sets both: restored at teardown
    monkeypatch.setenv("FLOCK_URL", "http://shell:1")
    run = tmp_path / "r"
    rundir.init(run, _folder(tmp_path), env={})
    (run / "hosts" / "controller.host").write_text("node3\n")
    with (run / "run.env").open("a") as f:
        f.write("SWARM=from-run\nBOARD_URL=https://run.invalid\n")
    over = tmp_path / "over.env"
    over.write_text("SWARM=from-flag\n")
    got = {}

    def fake_update(api, spec, spec_file, **kw):
        got["spec"] = spec
        return {"warnings": [], "probed": [], "kept": [], "diff": [], "deletes": [],
                "world_tools": None, "applied": False}

    monkeypatch.setattr("flock.ops.update", fake_update)
    r = runner.invoke(cli.app, ["--run", str(run), "update", str(run / "experiment"),
                                "--env", str(over)])
    assert r.exit_code == 0, r.output
    assert got["spec"].swarm == "from-flag"  # an explicit --env still wins
    assert got["spec"].types["researcher"].env["MSGBOARD_URL"] == "https://run.invalid"


SLURM_SPEC = {**SPEC, "compute": {"kind": "slurm", "partition": "${SLURM_PARTITION:-}",
                                  "cpus": "${WORKER_CPUS:-4}", "mem": "7G"}}


def test_profile_is_copied_once_and_read_before_run_env(tmp_path, monkeypatch):
    src, run = _folder(tmp_path, SLURM_SPEC), tmp_path / "r"
    prof = tmp_path / "site.env"
    prof.write_text("SLURM_PARTITION=cpu-spot\nWORKER_CPUS=6\nSWARM=from-profile\n")
    monkeypatch.setattr(rundir, "_sinfo", lambda argv: pytest.fail("sinfo in a unit test"))
    monkeypatch.setattr(rundir, "check_capacity", lambda spec, env: rundir.Check("slurm capacity", True))
    rep = rundir.init(run, src, profile=prof, env={})
    assert (run / "cluster.env").read_text() == prof.read_text()
    assert any("cluster.env" in c for c in rep.created)
    assert _by_name(rep)["spec"].detail.startswith("from-profile:")  # the profile fills ${VAR}s
    scripts = sorted(p.name for p in rundir.SLURM_SCRIPTS.iterdir())
    assert sorted(p.name for p in (run / "slurm").iterdir()) == scripts  # the job scripts' copy
    (run / "slurm" / "world.sbatch").write_text("# edited\n")
    with (run / "run.env").open("a") as f:
        f.write("WORKER_CPUS=8\n")  # run.env wins over the profile
    (run / "cluster.env").write_text(prof.read_text() + "# edited\n")
    again = rundir.init(run, profile=prof, env={})
    assert (run / "cluster.env").read_text().endswith("# edited\n")  # never overwritten
    assert any("cluster.env differs" in n for n in again.notes)
    assert (run / "slurm" / "world.sbatch").read_text() == "# edited\n"  # kept, and said
    assert any("slurm differs" in n and "edited world.sbatch" in n for n in again.notes)
    _, files = rundir.run_context(run)
    assert files == [str(run / "cluster.env"), str(run / "run.env")]
    spec = read_spec(str(run / "experiment" / "spec.yaml"), files)
    assert spec.compute.partition == "cpu-spot" and spec.compute.cpus == 8
    with pytest.raises(ValueError, match="no such file"):
        rundir.init(run, profile=tmp_path / "missing.env", env={})


def test_cluster_env_is_0644_whatever_the_umask_and_the_doctor_checks_it_like_the_jobs(
        tmp_path, monkeypatch):
    src, run = _folder(tmp_path, SLURM_SPEC), tmp_path / "r"
    prof = tmp_path / "site.env"
    prof.write_text("SLURM_PARTITION=cpu-spot\n")
    monkeypatch.setattr(rundir, "check_capacity", lambda spec, env: rundir.Check("slurm capacity", True))
    old = os.umask(0o002)  # a shared login node's
    try:
        rep = rundir.init(run, src, profile=prof, env={})
    finally:
        os.umask(old)
    f = run / "cluster.env"
    assert stat.S_IMODE(f.stat().st_mode) == 0o644 and _by_name(rep)["cluster.env"].ok
    f.chmod(0o664)
    c = _by_name(rundir.init(run, env={}))["cluster.env"]
    assert c.ok is False and f"chmod go-w {f}" in c.detail
    f.chmod(0o644)
    monkeypatch.setattr(os, "getuid", lambda: f.stat().st_uid + 1)
    c = rundir.check_cluster_env(f)
    assert c.ok is False and "not you" in c.detail


def test_slurm_capacity_from_a_fake_sinfo(tmp_path):
    spec = read_spec(str(_folder(tmp_path, SLURM_SPEC) / "spec.yaml"), [])
    calls = []

    def sinfo(idle):
        def run(argv):  # one row per partition; a node in both is counted in each
            calls.append(argv)
            return rundir.subprocess.CompletedProcess(
                argv, 0, f"cpu* 10/{idle}/0/40\ngpu 0/500/0/500\n", "")
        return run

    # 2 researchers, 2 per worker: 1 worker x 4 CPUs
    c = rundir.check_capacity(spec, {"FLOCK_SLURM_PARTITION": "cpu"}, run=sinfo(30))
    assert c.ok and not c.warn and c.lines == ["partition cpu: 4 CPUs planned, 30 idle now"]
    assert calls[-1] == ["sinfo", "-h", "-p", "cpu", "-o", "%P %C"]
    c = rundir.check_capacity(spec, {}, run=sinfo(2))  # the default (`*`) row only, not gpu's
    assert c.ok and c.warn and c.lines == ["SHORT: partition (default): 4 CPUs planned, 2 idle now"]
    assert calls[-1] == ["sinfo", "-h", "-o", "%P %C"]
    failing = lambda argv: rundir.subprocess.CompletedProcess(argv, 1, "", "no partition")
    assert rundir.check_capacity(spec, {}, run=failing).ok is None
    # the profile's node limits (FLOCK_SLURM_MAX_*) are seen by init too
    c = rundir.check_capacity(spec, {"FLOCK_SLURM_MAX_CPUS": "2"}, run=sinfo(30))
    assert c.ok is False and "over max_cpus 2" in c.detail


def test_container_mounts_follow_every_symlink_hop(tmp_path):
    uv, venv = tmp_path / "uv" / "python3.12", tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    uv.parent.mkdir()
    uv.write_text("")
    (venv / "bin" / "python").symlink_to(uv)
    assert rundir.check_mounts({"VENV": str(venv)}).ok is None  # no container in the profile
    assert rundir.check_mounts({"VENV": str(venv), "CONTAINER_IMAGE": "x.sqsh"}).ok is False
    c = rundir.check_mounts({"VENV": str(venv), "CONTAINER_MOUNTS": "/repo"})
    assert c.ok is False and str(uv) in c.detail  # the field failure: rc=127 in the container
    c = rundir.check_mounts({"WORKER_VENV": str(venv),
                             "CONTAINER_MOUNTS": f"{venv}:{venv}:ro,{uv.parent}/"})
    assert c.ok, c.detail
    c = rundir.check_mounts({"VENV": str(venv), "CONTAINER_MOUNTS": "uv,/repo"})
    assert c.ok is False and "must be absolute" in c.detail and "uv" in c.detail
    # a mounted symlinked dir: /shared -> real; the container sees /shared/python3.12
    real = tmp_path / "real"
    real.mkdir()
    (real / "python3.12").write_text("")
    (tmp_path / "shared").symlink_to(real)
    (venv / "bin" / "python").unlink()
    (venv / "bin" / "python").symlink_to(tmp_path / "shared" / "python3.12")
    assert rundir.check_mounts({"VENV": str(venv), "CONTAINER_MOUNTS": str(tmp_path / "shared")}).ok
    # only the target mounted: the link path /shared/... does not exist in the container
    c = rundir.check_mounts({"VENV": str(venv), "CONTAINER_MOUNTS": str(real)})
    assert c.ok is False and str(tmp_path / "shared" / "python3.12") in c.detail
    # a chain: every hop must be mounted, the last one alone is not enough
    (uv.parent / "python").symlink_to("python3.12")
    (venv / "bin" / "python").unlink()
    (venv / "bin" / "python").symlink_to(uv.parent / "python")
    assert rundir.check_mounts({"VENV": str(venv), "CONTAINER_MOUNTS": str(uv.parent)}).ok


