# Sourced by the run3 sbatch scripts: the run's settings, its secrets, the venv first on PATH,
# a container-safe HOME/caches, and `once`, a guard that runs a stop handler a single time
# (the Spot watcher signals every process of the job; the script may also get TERM).
# $1: the run's env file (<RUN_DIR>/run3.env).
set -euo pipefail
RUN_ENV=${1:?usage: sbatch <script> <RUN_DIR>/run3.env}
set -a
# shellcheck disable=SC1090
. "$RUN_ENV"
# shellcheck disable=SC1090
. "${SECRETS_FILE:?SECRETS_FILE is not set in $RUN_ENV}"
set +a
# the venv's python and console scripts first: `flock` must never be util-linux /usr/bin/flock
export PATH="$VENV/bin:$PATH" PYTHONUNBUFFERED=1
if [ -z "${HOME:-}" ] || [ ! -w "${HOME:-/nonexistent}" ]; then  # a container with no home
  export HOME=/tmp/$USER-home; mkdir -p "$HOME"
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-/tmp/$USER-uv} HF_HOME=${HF_HOME:-/tmp/$USER-hf}
PY="$VENV/bin/python"
log() { echo "$(date -u +%FT%TZ) ${SLURM_JOB_NAME:-run3}[${SLURM_JOB_ID:-?}]: $*"; }
STOPPING=0
once() { [ "$STOPPING" = 0 ] || return 0; STOPPING=1; "$@"; }
# wait for a child; a trapped signal interrupts `wait`, so wait again until it is gone
wait_for() { local pid=$1 rc=0; while kill -0 "$pid" 2>/dev/null; do wait "$pid" || rc=$?; done; return "$rc"; }
