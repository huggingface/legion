#!/bin/bash
# One flock worker slot, submitted BY THE CONTROLLER (compute: slurm, controller/slurm.py):
#   sbatch --parsable -p P -c C --mem=M --time=T --no-requeue -J <worker id> --comment=<labels>
#          -o <log_dir>/<worker id>-%j.out worker.sbatch <launch_dir>/<worker id>-<incarnation>.env
# Reference script: the cluster may ship its own with the same contract ($1 = the env file).
# The env file has no secret values; the secrets come from FLOCK_SECRETS_FILE (mode 600).
# sbatch runs with a scrubbed environment (PATH, HOME, USER, LANG, TZ, SLURM_*), so everything
# else comes from the two files.
#
# Cancel: flock runs a plain `scancel <job>` (TERM reaches the worker and the drain runs); it
# never signals srun itself (that would SIGKILL the task).
# Stop paths, all ending in the worker's own preemption drain (worker/main.py, <= 20 s:
# interrupt turns, sync every agent to the bucket in parallel, preempted marker, exit 75):
#   - a Spot notice: the node watcher sends SIGUSR1 to every process of the job (the worker
#     gets it directly and from this trap; it drains once), then cancels ~30 s later
#   - scancel / the time limit: SIGTERM, forwarded as SIGUSR1
# Bare node (default): flock-worker from FLOCK_VENV, paths under /tmp/flock/<worker id>.
# Container (FLOCK_CONTAINER_IMAGE in the env file, from compute container_image, with
# container_root /work): `srun --container-image` runs /opt/flock/bin/flock-worker inside the
# image (Containerfile); WORK_ROOT, FLOCK_LLM_HOME, HOME, HF_HOME, TMPDIR are under
# /work/flock/<worker id>/ in the
# container's root filesystem (never its /tmp: RAM counted against the job), agents never touch
# /fsx. The secrets are env vars of this script; srun passes the environment into the
# container (pyxis default; verify once with `srun --container-image=... env | cut -d= -f1`).
set -euo pipefail
ENV_FILE=${1:?usage: worker.sbatch <env file>}
set -a
# shellcheck disable=SC1090
. "$ENV_FILE"
# shellcheck disable=SC1090
. "${FLOCK_SECRETS_FILE:?FLOCK_SECRETS_FILE is not in $ENV_FILE}"
# shellcheck disable=SC1090
. "$ENV_FILE"   # a key in the env file wins over the shared secrets file
set +a
: "${WORKER_ID:?}" "${FLOCK_URL:?}" "${FLOCK_TOKEN:?FLOCK_TOKEN missing from the secrets file}"
: "${HF_TOKEN:?HF_TOKEN missing from the secrets file (bucket sync)}"
export FLOCK_WORLD_TOKEN=${FLOCK_WORLD_TOKEN:-${CHATBOT_WORLD_TOKEN:-}}
export PYTHONUNBUFFERED=1
[ -n "${FLOCK_CONTAINER_IMAGE:-}" ] || export PATH="${FLOCK_VENV:?}/bin:$PATH"
# the controller's host: read at start (and re-read by the worker when it stops answering)
if [ -n "${FLOCK_CTL_HOST_FILE:-}" ]; then export CTL_HOST=$(head -1 "$FLOCK_CTL_HOST_FILE"); fi
echo "worker $WORKER_ID job ${SLURM_JOB_ID:-?} on $(hostname); controller ${CTL_HOST:-$FLOCK_URL}"
BASE=""
if [ -n "${FLOCK_CONTAINER_IMAGE:-}" ]; then
  # the paths are inside the container's own (writable, ephemeral) root filesystem
  MOUNTS=()  # the controller host file's dir, read-only: the worker re-reads it on reconnect
  if [ -n "${FLOCK_CTL_HOST_FILE:-}" ]; then
    d=$(dirname "$FLOCK_CTL_HOST_FILE"); MOUNTS=(--container-mounts="$d:$d:ro")
  fi
  srun --ntasks=1 --container-image="$FLOCK_CONTAINER_IMAGE" --no-container-mount-home \
       --container-writable "${MOUNTS[@]}" \
       bash -c 'mkdir -p "$WORK_ROOT" "$FLOCK_LLM_HOME" "$HOME" "$HF_HOME" "$TMPDIR" \
                 && exec /opt/flock/bin/flock-worker' &
else
  # node-local and fixed per slot (restores go to the same absolute paths); wiped at start
  BASE=/tmp/flock/$WORKER_ID
  rm -rf "$BASE"; mkdir -p "$WORK_ROOT" "$FLOCK_LLM_HOME"
  if [ -z "${HOME:-}" ] || [ ! -w "${HOME:-/nonexistent}" ]; then export HOME=$BASE/home; mkdir -p "$HOME"; fi
  "$FLOCK_VENV/bin/flock-worker" &
fi
PID=$!   # srun forwards USR1/TERM to the step
SENT=0
forward() { [ "$SENT" = 0 ] || return 0; SENT=1; kill -USR1 "$PID" 2>/dev/null || true; }
trap forward USR1 TERM
rc=0
while kill -0 "$PID" 2>/dev/null; do wait "$PID" || rc=$?; done
echo "worker $WORKER_ID exited $rc"   # 75 = preempted (synced), 0 = drained
if [ -n "$BASE" ]; then rm -rf "$BASE"; fi
exit "$rc"
