"""experiments/run3/slurm: the scripts parse, and worker.sbatch (bare mode) runs flock-worker
from the env file and the secrets file, forwards TERM/USR1 once and returns its exit code."""

import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

SLURM = Path(__file__).parent.parent / "experiments" / "run3" / "slurm"


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
