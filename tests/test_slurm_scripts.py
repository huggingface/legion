"""scripts/slurm (the generic copies `flock init` makes per run): they parse (and pass
shellcheck when it is installed), and worker.sbatch without a container hands flock-worker its
secrets in a private file only."""

from __future__ import annotations

import os
import shutil
import subprocess
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
