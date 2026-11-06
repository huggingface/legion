"""scripts/slurm (the generic copies `flock init` makes per run): they parse (and pass
shellcheck when it is installed), and worker.sbatch without a container hands flock-worker its
secrets in a private file only."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

import pytest

GENERIC = Path(__file__).parent.parent / "scripts" / "slurm"


@pytest.mark.parametrize("name", sorted(p.name for p in GENERIC.iterdir()))
def test_generic_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(GENERIC / name)], check=False).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_generic_scripts_pass_shellcheck():
    r = subprocess.run(["shellcheck", "-x", "-S", "warning", *sorted(GENERIC.iterdir())],
                       cwd=GENERIC, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout


def make_run(tmp_path, flock_worker, env_extra="", cluster_extra=""):
    """A run directory with the generic scripts, a venv of stubs (flock-worker: the given script
    body) and one worker env file. Returns (worker.sbatch, the env file)."""
    run = tmp_path / "run"
    shutil.copytree(GENERIC, run / "slurm")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    for name, body in [("python", "exit 0"), ("flock", "exit 0"), ("flock-controller", "exit 0"),
                       ("flock-worker", flock_worker)]:
        (venv / "bin" / name).write_text(f"#!/bin/bash\n{body}\n")
        (venv / "bin" / name).chmod(0o755)
    (run / "cluster.env").write_text(f"VENV={venv}\n{cluster_extra}")
    (run / "run.env").write_text("FLOCK_TOKEN=tok\nHF_TOKEN=hf_tok\nOPENAI_API_KEY=key\n"
                                 "FLOCK_WORLD_ADMIN_TOKEN=admin\n")
    (run / "hosts").mkdir()
    (run / "hosts" / "controller.host").write_text("node1:7999\n")
    (run / "launch").mkdir()
    env_file = run / "launch" / "s-t-w0001-abc.env"
    env_file.write_text(
        "# flock worker env (written by the controller; no secret values).\n"
        "# Secrets this worker expects from the cluster's secrets file: FLOCK_TOKEN HF_TOKEN "
        "OPENAI_API_KEY\n"
        "DEADLINE=2026-11-05T00:00:00+00:00\nDRAIN_WINDOW=10m\nFLOCK_URL='http://${CTL_HOST}'\n"
        f"SWARM=s\nTYPE=t\n"
        f"WORK_ROOT={tmp_path / 'w' / 'work'}\nWORKER_ID=s-t-w0001\nWORKER_INCARNATION=abc\n"
        + env_extra)
    for f in (run / "run.env", env_file):
        f.chmod(0o600)
    return run / "slurm" / "worker.sbatch", env_file


def test_generic_worker_gets_its_secrets_in_a_private_file_only(tmp_path):
    log = tmp_path / "worker.log"
    script, env_file = make_run(tmp_path, f"""\
{{ echo "url=$FLOCK_URL ctl=$CTL_HOST env_token=${{FLOCK_TOKEN:-none}} admin=${{FLOCK_WORLD_ADMIN_TOKEN:-none}}"
  stat -c %a "$FLOCK_WORKER_SECRETS_FILE"; cat "$FLOCK_WORKER_SECRETS_FILE"; }} > {log}
exit 3""")
    r = subprocess.run(["bash", str(script), str(env_file)],
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 3, r.stdout + r.stderr  # exec: flock-worker's own exit code
    assert log.read_text().splitlines() == [
        "url=http://${CTL_HOST} ctl=node1:7999 env_token=none admin=none", "600",
        "FLOCK_TOKEN=tok", "HF_TOKEN=hf_tok", "OPENAI_API_KEY=key"]
    assert "tok" not in r.stdout + r.stderr


PODMAN = """#!/bin/bash
# a podman stub: logs its arguments; `info` prints --root, `unshare CMD` runs CMD
echo "$*" >> {log}
root=
while [ $# -gt 0 ]; do
  case $1 in
    --root) root=$2; shift 2;;
    --runroot|--storage-driver|--storage-opt|--cgroup-manager|--events-backend) shift 2;;
    *) break;;
  esac
done
case ${{1:-}} in info) echo "$root";; unshare) shift; "$@";; esac
"""


def test_harbor_worker_runs_on_the_node_with_the_jobs_own_podman(tmp_path):
    """FLOCK_HARNESS=harbor: no container even with an image set; flock-worker runs here with
    shepherd's podman setup (this job's storage, podman-compose, health checks), gets its
    secrets in the file only (and the health checks none), git gets the HF token for
    huggingface.co from a 0600 file (never an environment), the worker's preempt exit (75)
    reaches Slurm, and at exit the job's containers are stopped with the job's own wrapper."""
    stubs, log, podman_log = tmp_path / "stubs", tmp_path / "worker.log", tmp_path / "podman.log"
    stubs.mkdir()
    (stubs / "podman").write_text(PODMAN.format(log=podman_log))
    (stubs / "srun").write_text("#!/bin/sh\necho srun >> " + str(podman_log) + "\nexit 9\n")
    for f in stubs.iterdir():
        f.chmod(0o755)
    image = tmp_path / "worker.sqsh"
    image.write_text("")
    health = tmp_path / "health.log"
    script, env_file = make_run(tmp_path, f"""\
for _ in $(seq 100); do [ -s {health} ] && break; sleep 0.05; done  # the health checks started
{{ echo "compose=$PODMAN_COMPOSE_PROVIDER podman=$(command -v podman) env_token=${{HF_TOKEN:-none}}"
  echo "xdg=$XDG_RUNTIME_DIR git_prompt=$GIT_TERMINAL_PROMPT"
  podman info --format '{{{{.Store.GraphRoot}}}}'; cat "$FLOCK_WORKER_SECRETS_FILE"
  printf 'protocol=https\\nhost=huggingface.co\\n\\n' | git credential fill | grep password=
  env | grep -c hf_tok; }} > {log}
exit 75""", env_extra="FLOCK_HARNESS=harbor\n", cluster_extra=f"CONTAINER_IMAGE={image}\n")
    venv = script.parents[2] / "venv"
    (venv / "bin" / "podman-compose").write_text("#!/bin/sh\nexit 0\n")
    (venv / "bin" / "podman-compose").chmod(0o755)
    (venv / "bin" / "python").write_text(
        f'#!/bin/bash\necho "$* hf=${{HF_TOKEN:-none}} flock=${{FLOCK_TOKEN:-none}}" > {health}\n')
    r = subprocess.run(["bash", str(script), str(env_file)],
                       env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 75, r.stdout + r.stderr  # flock-worker's exit code, through the wait
    compose, xdg, graphroot, *secrets, password, in_env = log.read_text().splitlines()
    local = Path(graphroot).parent
    try:
        assert (local / "hf-token").stat().st_mode & 0o777 == 0o600
    finally:
        shutil.rmtree(local)  # the job's /tmp takes it on the cluster
    assert (password, in_env) == ("password=hf_tok", "0")
    assert compose == (f"compose={venv}/bin/podman-compose podman={local}/bin/podman "
                       "env_token=none")
    assert xdg == f"xdg={local}/runtime git_prompt=0" and local.parent == Path("/tmp")
    assert secrets == ["FLOCK_TOKEN=tok", "HF_TOKEN=hf_tok", "OPENAI_API_KEY=key"]
    assert health.read_text() == "-m flock.harbor.podman_healthchecks hf=none flock=none\n"
    calls = podman_log.read_text().splitlines()
    assert "srun" not in calls and all(c.startswith(f"--root {local}/graphroot ") for c in calls)
    assert [c.split("file ", 1)[1].split()[0] for c in calls[-2:]] == ["stop", "rm"]
    assert "tok" not in r.stdout + r.stderr


def test_harbor_worker_gets_the_jobs_sigterm(tmp_path):
    """`--signal=B:TERM@60` reaches only this shell: it forwards TERM to flock-worker, its child
    here (not through slurmd, as a srun step), waits for it and cleans up."""
    stubs, log, ready = tmp_path / "stubs", tmp_path / "worker.log", tmp_path / "ready"
    stubs.mkdir()
    (stubs / "podman").write_text(PODMAN.format(log=tmp_path / "podman.log"))
    (stubs / "podman").chmod(0o755)
    script, env_file = make_run(tmp_path, f"""\
trap 'echo term > {log}; exit 0' TERM
touch {ready}
while :; do sleep 0.05; done""", env_extra="FLOCK_HARNESS=harbor\n")
    (script.parents[2] / "venv" / "bin" / "podman-compose").symlink_to("python")
    p = subprocess.Popen(["bash", str(script), str(env_file)], stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)})
    try:
        for _ in range(200):
            if ready.exists():
                break
            time.sleep(0.05)
        time.sleep(0.2)  # the wait loop's traps are set right after the start
        p.terminate()
        out, _ = p.communicate(timeout=20)
    finally:
        p.kill()
    shutil.rmtree(Path((tmp_path / "podman.log").read_text().split()[1]).parent)  # the job's /tmp
    assert p.returncode == 0 and log.read_text() == "term\n", out
    assert "forwarding to flock-worker" in out
