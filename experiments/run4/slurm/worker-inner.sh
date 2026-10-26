#!/bin/bash
# worker-inner.sh: the worker's start-up INSIDE its container (started by worker.sbatch through
# `srun --container-image ... bash worker-inner.sh`). Never run it on the host.
#
# Creates the /work dirs (container root: node disk, deleted with the container; NOT /tmp, a
# tmpfs counted against the job memory), then exec's flock-worker with SIGUSR1 ignored: SIG_IGN
# survives exec, so the watcher's direct SIGUSR1 (slurmd signals every process of the step) does
# not kill it by default action; flock-worker's own Python handler (drain + sync, being added
# upstream, 20 s) replaces SIG_IGN once installed. SIGTERM keeps its default / flock's handler.
# The batch script's environment (secrets, the env file's settings) passes into the container.
set -uo pipefail
log() { printf '%s worker-inner[%s]: %s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-nojob}" "$*" >&2; }
for v in VENV WORK_ROOT FLOCK_LLM_HOME; do
  [ -n "${!v:-}" ] || { log "FATAL: $v is not set (worker.sbatch exports it)"; exit 2; }
done
case $WORK_ROOT$FLOCK_LLM_HOME in /work/*/work/*) ;; *) log "FATAL: WORK_ROOT/FLOCK_LLM_HOME not under /work"; exit 2;; esac
trap '' USR1
export HOME=/work/home HF_HOME=/work/hf TMPDIR=/work/tmp
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
mkdir -p "$WORK_ROOT" "$FLOCK_LLM_HOME" "$HOME" "$HF_HOME" "$TMPDIR" || { log "FATAL: cannot create /work dirs"; exit 2; }
cd /work || exit 2
# Pi harness (an image with the Pi CLI, e.g. pi-worker-node22.sqsh for agent-pi): its binary and one
# agent dir per worker (models.json, sessions) on /work. Images without pi are untouched.
if [ -x /usr/local/bin/pi ]; then
  export FLOCK_PI_CLI=/usr/local/bin/pi PI_CODING_AGENT_DIR=/work/pi/${WORKER_ID:-worker}
  mkdir -p "$PI_CODING_AGENT_DIR" || { log "FATAL: cannot create $PI_CODING_AGENT_DIR"; exit 2; }
  log "pi harness: FLOCK_PI_CLI=$FLOCK_PI_CLI PI_CODING_AGENT_DIR=$PI_CODING_AGENT_DIR"
fi
log "flock-worker in the container: WORK_ROOT=$WORK_ROOT FLOCK_LLM_HOME=$FLOCK_LLM_HOME HOME=$HOME"
exec "$VENV/bin/flock-worker"
