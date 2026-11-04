"""experiments/run3/slurm: the scripts parse, and worker.sbatch (bare mode) runs flock-worker
from the env file and the secrets file, forwards TERM/USR1 once and returns its exit code.
scripts/slurm (the generic copies `flock init` makes per run): they parse (and pass shellcheck
when it is installed), and worker.sbatch without a container hands flock-worker its secrets
in a private file only."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

SLURM = Path(__file__).parent.parent / "experiments" / "run3" / "slurm"
GENERIC = Path(__file__).parent.parent / "scripts" / "slurm"


@pytest.mark.parametrize("name", sorted(p.name for p in SLURM.iterdir()
                                        if p.suffix in (".sbatch", ".sh")))
def test_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(SLURM / name)], check=False).returncode == 0


@pytest.mark.parametrize("signals", [("USR1", "USR1"), ("USR1", "TERM"), ("TERM",)])
def test_worker_sbatch_forwards_one_signal_and_returns_the_exit_code(tmp_path, signals):
    venv = tmp_path / "venv" / "bin"
    venv.mkdir(parents=True)
    log = tmp_path / "worker.log"
    fake = venv / "flock-worker"  # a stand-in that drains on USR1 and exits 75 like the real one
    fake.write_text(f"""#!/bin/bash
echo "start $WORKER_ID token=${{#FLOCK_TOKEN}} hf=${{#HF_TOKEN}} root=$WORK_ROOT" >> {log}
trap 'echo got-usr1 >> {log}; sleep 0.3; exit 75' USR1
trap 'echo got-term >> {log}; exit 1' TERM
while true; do sleep 0.1; done
""")
    fake.chmod(0o755)
    secrets = tmp_path / "secrets.env"
    secrets.write_text("FLOCK_TOKEN=tok-value\nHF_TOKEN=hf-value\n")
    secrets.chmod(0o600)
    wid = "run3-agent-w0001"
    env_file = tmp_path / f"{wid}-abc.env"
    env_file.write_text(f"# no secrets\nWORKER_ID={wid}\nFLOCK_URL='http://${{CTL_HOST}}:7860'\n"
                        f"FLOCK_CTL_HOST_FILE={tmp_path / 'controller.host'}\n"
                        f"WORK_ROOT={tmp_path / 'w' / wid / 'work'}\n"
                        f"FLOCK_LLM_HOME={tmp_path / 'w' / wid / 'llm'}\n"
                        f"FLOCK_VENV={venv.parent}\nFLOCK_SECRETS_FILE={secrets}\n")
    (tmp_path / "controller.host").write_text("ctl-node\n")
    proc = subprocess.Popen(["bash", str(SLURM / "worker.sbatch"), str(env_file)],
                            env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    deadline = time.monotonic() + 10
    while not (log.exists() and "start" in log.read_text()):
        assert time.monotonic() < deadline and proc.poll() is None, proc.stdout.read()
        time.sleep(0.05)
    for sig in signals:
        proc.send_signal(getattr(signal, f"SIG{sig}"))
        time.sleep(0.05)
    out, _ = proc.communicate(timeout=10)
    assert proc.returncode == 75, out
    lines = log.read_text().splitlines()
    assert lines[0] == f"start {wid} token=9 hf=8 root={tmp_path / 'w' / wid / 'work'}"
    assert lines[1:] == ["got-usr1"]  # forwarded once, as USR1, whatever arrived
    assert "controller ctl-node" in out and "exited 75" in out


@pytest.mark.parametrize("name", sorted(p.name for p in GENERIC.iterdir()))
def test_generic_scripts_parse(name):
    assert subprocess.run(["bash", "-n", str(GENERIC / name)], check=False).returncode == 0


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_generic_scripts_pass_shellcheck():
    r = subprocess.run(["shellcheck", "-x", "-S", "warning", *sorted(GENERIC.iterdir())],
                       cwd=GENERIC, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout


def test_generic_worker_gets_its_secrets_in_a_private_file_only(tmp_path):
    run = tmp_path / "run"
    shutil.copytree(GENERIC, run / "slurm")
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "worker.log"
    for name in ("python", "flock", "flock-controller"):
        (venv / "bin" / name).write_text("#!/bin/sh\nexit 0\n")
        (venv / "bin" / name).chmod(0o755)
    (venv / "bin" / "flock-worker").write_text(f"""#!/bin/bash
{{ echo "url=$FLOCK_URL ctl=$CTL_HOST env_token=${{FLOCK_TOKEN:-none}} admin=${{FLOCK_WORLD_ADMIN_TOKEN:-none}}"
  stat -c %a "$FLOCK_WORKER_SECRETS_FILE"; cat "$FLOCK_WORKER_SECRETS_FILE"; }} > {log}
exit 3
""")
    (venv / "bin" / "flock-worker").chmod(0o755)
    (run / "cluster.env").write_text(f"VENV={venv}\n")
    (run / "run.env").write_text("FLOCK_TOKEN=tok\nHF_TOKEN=hf\nOPENAI_API_KEY=key\n"
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
        f"FLOCK_LLM_HOME={tmp_path / 'w' / 'llm'}\nSWARM=s\nTYPE=t\n"
        f"WORK_ROOT={tmp_path / 'w' / 'work'}\nWORKER_ID=s-t-w0001\nWORKER_INCARNATION=abc\n")
    for f in (run / "run.env", env_file):
        f.chmod(0o600)
    r = subprocess.run(["bash", str(run / "slurm" / "worker.sbatch"), str(env_file)],
                       env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)},
                       capture_output=True, text=True, timeout=30, check=False)
    assert r.returncode == 3, r.stdout + r.stderr  # exec: flock-worker's own exit code
    assert log.read_text().splitlines() == [
        "url=http://${CTL_HOST} ctl=node1:7999 env_token=none admin=none", "600",
        "FLOCK_TOKEN=tok", "HF_TOKEN=hf", "OPENAI_API_KEY=key"]
    assert "tok" not in r.stdout + r.stderr
