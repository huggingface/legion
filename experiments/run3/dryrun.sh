#!/bin/bash
# run3 dry run: one command, no model and no real board. Proves a node (or a container) can run
# flock: the venv, the paths, the signal handling, and the whole single-job path.
#   1. the Slurm-side tests: the sbatch/squeue/sacct/scancel contract (stubs), the worker's
#      preemption drain (SIGUSR1/SIGTERM, exit 75, parallel sync), USR1 = TERM for the
#      controller and `flock world serve`, the home-less /work defaults
#   2. the smoke rehearsal: the fake model server and the fake board from tests/, the
#      bootstrap's `flock run` (compute local) with the smoke waves (minutes as seconds), a live
#      server switch, the stop marker, and the bundle checked in a local "bucket"
# Exits non-zero on any failure. About 2-3 minutes. Run it inside one Slurm job:
#   srun -p <partition> -c 4 --mem=8G --time=00:20:00 experiments/run3/dryrun.sh
# Python: DRYRUN_PYTHON, else $VENV/bin/python if it has pytest, else a fresh env from uv.lock
# in $TMPDIR (needs uv and PyPI). Nothing is written outside $TMPDIR and the repo checkout.
set -euo pipefail
REPO=$(cd "$(dirname "$(readlink -f "$0")")/../.." && pwd)
PY=${DRYRUN_PYTHON:-}
if [ -z "$PY" ] && [ -n "${VENV:-}" ] && "$VENV/bin/python" -c "import pytest, pytest_asyncio, respx" 2>/dev/null; then
  PY=$VENV/bin/python
fi
if [ -z "$PY" ]; then
  ENVDIR=${DRYRUN_ENV:-${TMPDIR:-/tmp}/flock-dryrun-${USER:-$(id -u)}}
  echo "dryrun: building a test env in $ENVDIR (uv sync --frozen)"
  UV_PROJECT_ENVIRONMENT=$ENVDIR uv sync --project "$REPO" --frozen --all-extras -q
  PY=$ENVDIR/bin/python
fi
export PATH="$(dirname "$PY"):$PATH"   # `flock` = this env's, never util-linux /usr/bin/flock
cd "$REPO"
echo "dryrun: $("$PY" --version) at $PY on $(hostname)"
"$PY" -m pytest -q -p no:cacheprovider tests/test_slurm.py tests/test_worker_preempt.py \
  tests/test_stop_signals.py tests/test_world_move.py
# the smoke rehearsal; one retry: its live server switch is timing-sensitive on a loaded node
for attempt in 1 2; do
  if "$PY" -m pytest -q -p no:cacheprovider tests/test_run3_experiment.py -k "end_to_end and smoke"; then
    echo "dryrun: ok"; exit 0
  fi
  echo "dryrun: smoke rehearsal failed (attempt $attempt)"
done
exit 1
