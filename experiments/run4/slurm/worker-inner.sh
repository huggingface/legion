#!/bin/bash
# worker-inner.sh: the worker's start-up INSIDE its container (started by worker.sbatch through
# `srun --container-image ... bash worker-inner.sh`). Never run it on the host.
#
# Creates the /work dirs (container root: node disk, deleted with the container; NOT /tmp, a
# tmpfs counted against the job memory), then exec's flock-worker with SIGUSR1 ignored: SIG_IGN
# survives exec, so the watcher's direct SIGUSR1 (slurmd signals every process of the step) does
# not kill it by default action; flock-worker's own Python handler (drain + sync, being added
# upstream, 20 s) replaces SIG_IGN once installed. SIGTERM keeps its default / flock's handler.
# The batch script's environment (secrets, the env file's settings) passes into the container; the
# secrets are then moved out of it: flock-worker reads them from a private file (below), so its
# /proc/<pid>/environ and its children never carry them.
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
# Pi harness (an image with the Pi CLI, e.g. pi-worker-node22.sqsh for agent-pi): its binary and the
# agent dir (models.json, sessions) on /work. Like WORK_ROOT the path carries no worker id, so a
# slot relaunched under a new id finds its sessions at the same place (the container root is
# per job: no collision between workers on one node). Images without pi are untouched.
if [ -x /usr/local/bin/pi ]; then
  export FLOCK_PI_CLI=/usr/local/bin/pi PI_CODING_AGENT_DIR=/work/pi/agents
  mkdir -p "$PI_CODING_AGENT_DIR" || { log "FATAL: cannot create $PI_CODING_AGENT_DIR"; exit 2; }
  log "pi harness: FLOCK_PI_CLI=$FLOCK_PI_CLI PI_CODING_AGENT_DIR=$PI_CODING_AGENT_DIR"
fi
# Secrets: flock-worker reads them from a 0600 file named by FLOCK_WORKER_SECRETS_FILE (docs/slurm.md,
# "Worker secrets file") and removes those names from its os.environ; here they are also unset from
# its exec environment, so /proc/<pid>/environ never shows them. The file lives next to WORK_ROOT on
# the container root (node disk, deleted with the container; never /tmp or the shared disk) and flock-worker
# deletes it once read (UNLINK=1; a new step runs this script again and writes a new one). Only the
# names that are set are written: an empty NAME= line would shadow a real value. CHATBOT_WORLD_TOKEN
# (copied to FLOCK_WORLD_TOKEN by worker.sbatch) and FLOCK_WORLD_ADMIN_TOKEN (the world's /admin
# secret, never a worker's) are not read by the worker and are only unset.
umask 077
SECRET_NAMES=(FLOCK_TOKEN HF_TOKEN OPENAI_API_KEY FLOCK_WORLD_TOKEN CHATBOT_BOARD_INVITE)
SECRETS_DIR=$(dirname "$WORK_ROOT")
SECRETS_FILE=$(mktemp "$SECRETS_DIR/flock-worker-secrets.XXXXXX") || { log "FATAL: cannot create the secrets file in $SECRETS_DIR"; exit 2; }
written=()
for v in "${SECRET_NAMES[@]}"; do
  if [ -n "${!v:-}" ]; then
    printf '%s=%s\n' "$v" "${!v}" >> "$SECRETS_FILE" || { log "FATAL: cannot write $SECRETS_FILE"; exit 2; }
    written+=("$v")
  fi
done
unset_args=()
for v in "${SECRET_NAMES[@]}" CHATBOT_WORLD_TOKEN FLOCK_WORLD_ADMIN_TOKEN; do unset_args+=(-u "$v"); done
log "secrets file $SECRETS_FILE: ${written[*]:-<none>} (unset from the exec environment: ${SECRET_NAMES[*]} CHATBOT_WORLD_TOKEN FLOCK_WORLD_ADMIN_TOKEN)"
log "flock-worker in the container: WORK_ROOT=$WORK_ROOT FLOCK_LLM_HOME=$FLOCK_LLM_HOME HOME=$HOME"
# Isolated Pi never sees this experiment's world/board/repo credentials (pi.isolated_env).
exec env "${unset_args[@]}" FLOCK_WORKER_SECRETS_FILE="$SECRETS_FILE" FLOCK_WORKER_SECRETS_FILE_UNLINK=1 \
  FLOCK_HARNESS_ENV_DROP=CHATBOT_WORLD_TOKEN,CHATBOT_BOARD_INVITE,CHATBOT_REPO_TOKEN \
  "$VENV/bin/flock-worker"
