#!/bin/bash
#SBATCH --job-name=run3-controller
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton
#SBATCH --open-mode=append
#SBATCH --output=%x-%j.out
# Pin to a non-Spot node if the cluster has one (a controller move costs ~30 s of polls, no
# relaunch: workers re-read controller.host). For example:
##SBATCH --constraint=on-demand
#
# The run's controller: FLOCK_COMPUTE=slurm, it sbatches every worker (worker.sbatch).
#   sbatch --output=$RUN_DIR/logs/%x-%j.out controller.sbatch $RUN_DIR/run3.env
# Exactly one per run (--dependency=singleton by job name and user). Its state lives in the
# bucket (FLOCK_STATE_URL), restored at start; FLOCK_DATA is node-local.
# Stop: SIGTERM (scancel, time limit) or SIGUSR1 (Spot notice): one final state push (<= 30 s,
# FLOCK_FINAL_PUSH_TIMEOUT=15 here to fit the ~25 s Spot budget), then exit.
# shellcheck disable=SC1091
# (Slurm runs a spooled copy of this script: common.sh is found through RUN3_SLURM_DIR, else
# the submit directory; submit from experiments/run3/slurm or export RUN3_SLURM_DIR)
. "${RUN3_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(dirname "$0")}}/common.sh" "$@"
: "${FLOCK_TOKEN:?}" "${HF_TOKEN:?}" "${FLOCK_STATE_URL:?}" "${RUN_DIR:?}"
mkdir -p "$RUN_DIR/logs" "$FLOCK_SLURM_LAUNCH_DIR"
HOST=$(hostname)
write_host "$FLOCK_SLURM_HOST_FILE" "$HOST"
export FLOCK_COMPUTE=slurm FLOCK_MODE=space PORT=7860
export FLOCK_URL="http://$HOST:7860"                # the API, for the CLI and waves
export FLOCK_WORKER_URL='http://${CTL_HOST}:7860'   # what workers get: expanded from the host file
export FLOCK_DATA=/tmp/run3-controller-${SLURM_JOB_ID:-0}/data FLOCK_FINAL_PUSH_TIMEOUT=15
mkdir -p "$FLOCK_DATA"
log "controller on $HOST (host file $FLOCK_SLURM_HOST_FILE); state $FLOCK_STATE_URL"
"$PY" -m flock.controller.main &
PID=$!
stop() { log "stop: TERM to the controller (final state push)"; kill -TERM "$PID" 2>/dev/null || true; }
trap 'once stop' TERM USR1 INT
rc=0; wait_for "$PID" || rc=$?
rm -rf "/tmp/run3-controller-${SLURM_JOB_ID:-0}"
log "controller exited $rc"
exit "$rc"
