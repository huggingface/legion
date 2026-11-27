"""scripts/slurm (the generic copies `legion init` makes per run): they parse (and pass
shellcheck when it is installed), and worker.sbatch without a container hands legion-worker its
secrets in a private file only."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import pytest

GENERIC = Path(__file__).parent.parent / "scripts" / "slurm"
SHELL = sorted(p for p in GENERIC.iterdir() if p.suffix in (".sh", ".sbatch"))  # not registries.conf


@pytest.mark.parametrize("name", [p.name for p in SHELL])
def test_generic_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(GENERIC / name)], check=False).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_generic_scripts_pass_shellcheck():
    r = subprocess.run(["shellcheck", "-x", "-S", "warning", *SHELL],
                       cwd=GENERIC, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout


def make_run(tmp_path, legion_worker, env_extra="", cluster_extra=""):
    """A run directory with the generic scripts, a venv of stubs (legion-worker: the given script
    body) and one worker env file. Returns (worker.sbatch, the env file)."""
    run = tmp_path / "run"
    shutil.copytree(GENERIC, run / "slurm")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    for name, body in [("python", "exit 0"), ("legion", "exit 0"), ("legion-controller", "exit 0"),
                       ("legion-worker", legion_worker)]:
        (venv / "bin" / name).write_text(f"#!/bin/bash\n{body}\n")
        (venv / "bin" / name).chmod(0o755)
    (run / "cluster.env").write_text(f"VENV={venv}\n{cluster_extra}")
    (run / "run.env").write_text("LEGION_TOKEN=tok\nHF_TOKEN=hf_tok\nOPENAI_API_KEY=key\n"
                                 "LEGION_WORLD_ADMIN_TOKEN=admin\n")
    (run / "hosts").mkdir()
    (run / "hosts" / "controller.host").write_text("node1:7999\n")
    (run / "launch").mkdir()
    env_file = run / "launch" / "s-t-w0001-abc.env"
    env_file.write_text(
        "# legion worker env (written by the controller; no secret values).\n"
        "# Secrets this worker expects from the cluster's secrets file: LEGION_TOKEN HF_TOKEN "
        "OPENAI_API_KEY\n"
        "DEADLINE=2026-11-05T00:00:00+00:00\nDRAIN_WINDOW=10m\nLEGION_URL='http://${CTL_HOST}'\n"
        f"SWARM=s\nTYPE=t\n"
        f"WORK_ROOT={tmp_path / 'w' / 'work'}\nWORKER_ID=s-t-w0001\nWORKER_INCARNATION=abc\n"
        + env_extra)
    for f in (run / "cluster.env", run / "run.env", env_file):  # whatever the umask
        f.chmod(0o600)
    return run / "slurm" / "worker.sbatch", env_file


def test_generic_worker_gets_its_secrets_in_a_private_file_only(tmp_path):
    log = tmp_path / "worker.log"
    script, env_file = make_run(tmp_path, f"""\
{{ echo "url=$LEGION_URL ctl=$CTL_HOST env_token=${{LEGION_TOKEN:-none}} admin=${{LEGION_WORLD_ADMIN_TOKEN:-none}}"
  stat -c %a "$LEGION_WORKER_SECRETS_FILE"; cat "$LEGION_WORKER_SECRETS_FILE"; }} > {log}
exit 3""")
    r = subprocess.run(["bash", str(script), str(env_file)],
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 3, r.stdout + r.stderr  # exec: legion-worker's own exit code
    assert log.read_text().splitlines() == [
        "url=http://${CTL_HOST} ctl=node1:7999 env_token=none admin=none", "600",
        "LEGION_TOKEN=tok", "HF_TOKEN=hf_tok", "OPENAI_API_KEY=key"]
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
    """LEGION_HARNESS=harbor: no container even with an image set; legion-worker runs here with
    the upstream podman setup (this job's storage, podman-compose, health checks), gets its
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
  podman info --format '{{{{.Store.GraphRoot}}}}'; cat "$LEGION_WORKER_SECRETS_FILE"
  printf 'protocol=https\\nhost=huggingface.co\\n\\n' | git credential fill | grep password=
  env | grep -c hf_tok; }} > {log}
exit 75""", env_extra="LEGION_HARNESS=harbor\n", cluster_extra=f"CONTAINER_IMAGE={image}\n")
    venv = script.parents[2] / "venv"
    (venv / "bin" / "podman-compose").write_text("#!/bin/sh\nexit 0\n")
    (venv / "bin" / "podman-compose").chmod(0o755)
    (venv / "bin" / "python").write_text(
        f'#!/bin/bash\necho "$* hf=${{HF_TOKEN:-none}} legion=${{LEGION_TOKEN:-none}}" > {health}\n')
    r = subprocess.run(["bash", str(script), str(env_file)],
                       env={"PATH": f"{stubs}:{os.environ['PATH']}", "HOME": str(tmp_path)},
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 75, r.stdout + r.stderr  # legion-worker's exit code, through the wait
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
    assert secrets == ["LEGION_TOKEN=tok", "HF_TOKEN=hf_tok", "OPENAI_API_KEY=key"]
    assert health.read_text() == "-m legion.harbor.podman_healthchecks hf=none legion=none\n"
    calls = podman_log.read_text().splitlines()
    assert "srun" not in calls and all(c.startswith(f"--root {local}/graphroot ") for c in calls)
    assert [c.split("file ", 1)[1].split()[0] for c in calls[-2:]] == ["stop", "rm"]
    assert "tok" not in r.stdout + r.stderr


def test_harbor_worker_gets_the_jobs_sigterm(tmp_path):
    """`--signal=B:TERM@60` reaches only this shell: it forwards TERM to legion-worker, its child
    here (not through slurmd, as a srun step), waits for it and cleans up."""
    stubs, log, ready = tmp_path / "stubs", tmp_path / "worker.log", tmp_path / "ready"
    stubs.mkdir()
    (stubs / "podman").write_text(PODMAN.format(log=tmp_path / "podman.log"))
    (stubs / "podman").chmod(0o755)
    script, env_file = make_run(tmp_path, f"""\
trap 'echo term > {log}; exit 0' TERM
touch {ready}
while :; do sleep 0.05; done""", env_extra="LEGION_HARNESS=harbor\n")
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
    assert "forwarding to legion-worker" in out


def test_vllm_rejects_occupied_port(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        r = subprocess.run(["bash", str(GENERIC / "serve-vllm.sbatch"), "requested"],
                           env={**os.environ, "PORT": str(listener.getsockname()[1])},
                           capture_output=True, text=True, timeout=10, check=False)
    assert r.returncode != 0
    assert "Address already in use" in r.stderr
    assert "serving " not in r.stdout


@pytest.mark.parametrize("response,server,expected", [
    ('{"data": [{"id": "alias"}]}', "/bin/sleep 1", True),
    ('{"data": [{"id": "other"}]}', "/bin/sleep 1", False),
    ('{"data": [{"id": "alias"}]}', "exit 1", False),
])
def test_vllm_readiness_checks_model_and_process(tmp_path, response, server, expected):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    stubs = tmp_path / "bin"
    stubs.mkdir()
    for name, body in {
        "srun": server,
        "curl": f"sleep 0.1\nprintf '%s' '{response}'",
        "sleep": "/bin/sleep 0.1",
        "mkdir": "exit 0",
    }.items():
        path = stubs / name
        path.write_text(f"#!/bin/sh\n{body}\n")
        path.chmod(0o755)
    r = subprocess.run(["bash", str(GENERIC / "serve-vllm.sbatch"), "requested",
                        "--served-model-name", "alias"],
                       env={**os.environ, "PATH": f"{stubs}:{os.environ['PATH']}", "PORT": str(port), "USER": "test"},
                       capture_output=True, text=True, timeout=10, check=False)
    assert ("serving requested" in r.stdout) is expected, r.stdout + r.stderr
    assert (r.returncode == 0) is expected, r.stdout + r.stderr
    if not expected:
        assert "exited during startup" in r.stderr
