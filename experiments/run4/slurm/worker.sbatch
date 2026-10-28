#!/bin/bash
# worker.sbatch <envfile>: one flock worker slot (12 agents) as a CPU Slurm job.
#
# Launched BY THE CONTROLLER's Slurm jobs client (FLOCK_COMPUTE=slurm), roughly as
#   sbatch --parsable -p $FLOCK_SLURM_PARTITION -c $FLOCK_SLURM_CPUS --mem=$FLOCK_SLURM_MEM \
#          --time=$FLOCK_SLURM_TIME --no-requeue -J <worker id> --comment '<labels>' \
#          -o $FLOCK_SLURM_LOG_DIR/<worker id>-%j.out \
#          $FLOCK_SLURM_WORKER_SCRIPT $FLOCK_SLURM_LAUNCH_DIR/<worker id>-<incarnation>.env
# The #SBATCH lines below are fallback defaults for a launch by hand. The env file format is
# worker.env.example (one KEY=value per line, mode 600, written by the client).
#
# Fact sheet §2 (worker env), §3 (one process per slot), §5 (restore from the bucket):
# - secrets.env is sourced first (common.sh), then the env file. Since flock fc22785 the env file
#   holds NO secret values (FLOCK_TOKEN, HF_TOKEN, ... come from secrets.env); it carries WORKER_ID,
#   WORKER_INCARNATION, SWARM, TYPE, DEADLINE, DRAIN_WINDOW, BUCKET, RUN, WORK_ROOT, FLOCK_LLM_HOME,
#   FLOCK_URL, FLOCK_CTL_HOST_FILE, FLOCK_VENV, FLOCK_SECRETS_FILE, PYTHONUNBUFFERED (worker.env.example).
# - CTL_HOST is read from <run>/hosts/controller.host and exported BEFORE the env file is read, so
#   FLOCK_URL='http://${CTL_HOST}:7860' expands inside flock-worker (worker/main.py expand_env).
#   <run>/hosts is mounted read-only in the container, so FLOCK_CTL_HOST_FILE is readable there.
# - CONTAINER: flock-worker runs in an Enroot/Pyxis container (worker-inner.sh): home not mounted,
#   /fsx read-only and limited to the venv, uv Python, repo, slurm dir and <run>/hosts. Agent files
#   go to /work/flock/agents/{work,llm} on the container root (node disk, deleted with the
#   container; not /tmp, a tmpfs counted against the job memory). WORK_ROOT and FLOCK_LLM_HOME of
#   the env file (/tmp/flock/<id>/...) are overridden here; the paths are the same on every node
#   AND carry no worker id, so sessions/workdirs restore to the same absolute paths when the
#   controller relaunches a slot under a new id (w0000 -> w0001 after a Spot reclaim): a Pi session
#   only resumes from the cwd it was written from (adapters/pi.py find_session), the agent's
#   workdir WORK_ROOT/<agent>. No collision: agent names are unique and each job has its own
#   container root, so two workers on one node never share /work.
# - Spot reclaim: the watcher sends SIGUSR1 to this shell, to srun AND to every process of the
#   container step. worker-inner.sh exec's flock-worker with USR1 ignored (SIG_IGN survives exec),
#   so the direct signal cannot kill it by default action; flock-worker's own Python handler (being
#   added upstream: drain + sync, 20 s) replaces SIG_IGN once installed. The handler below forwards
#   USR1 to the srun step (srun forwards it to the task), then SIGTERM 3 s later if the step is
#   still alive (through `scancel --signal=TERM <job>.<step>`: srun itself turns a SIGTERM into a
#   SIGKILL of the task, common.sh step_signal), and we wait. NO requeue: the controller sees the
#   job end and relaunches the slot.
#   flock-worker may see USR1 two or three times (direct, srun, this script): its handler must be
#   idempotent.
# - A plain `scancel <id>` (flock's SlurmJobsClient.cancel) is fine: slurmd sends SIGTERM to
#   flock-worker in the container directly and srun waits for the step (up to KillWait); tested,
#   job 255727. Never `kill -TERM $SRUN_PID`: srun answers that with a SIGKILL of the task.
#
#SBATCH --job-name=flock-worker
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G              # hopper-cpu MaxMemPerCPU=2000: 7G / 4 = 1792 MB per CPU (more would silently raise the CPU count)
#SBATCH --time=54:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@60    # 60 s before the time limit (DEADLINE + drain should end the worker long before)
#SBATCH -o /fsx/carlos_patino/prj-eliza/runs/_slurm-logs/%x-%j.out

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal
ENVFILE=${1:?usage: worker.sbatch <envfile>  (<runs>/<RUN>/launch/<worker id>-<incarnation>.env)}
[ -f "$ENVFILE" ] || { echo "worker.sbatch: $ENVFILE: no such file" >&2; exit 2; }

# RUN: from a RUN= line in the env file, else from its path <runs>/<RUN>/launch/<file>.
RUN=$(sed -n 's/^RUN=//p' "$ENVFILE" | tail -1 | tr -d "\"'[:space:]")
[ -n "$RUN" ] || RUN=$(basename "$(dirname "$(dirname "$ENVFILE")")")
# shellcheck disable=SC2034  # LOG_TAG is read by log() in common.sh
LOG_TAG=worker
. /fsx/carlos_patino/prj-eliza/slurm/common.sh

# ---------------------------------------------------------------- controller address, then the worker env
CTL_HOST_FILE=$HOSTS_DIR/controller.host
[ -s "$CTL_HOST_FILE" ] || die "$CTL_HOST_FILE is missing: is controller.sbatch running?"
CTL_HOST=$(tr -d '[:space:]' < "$CTL_HOST_FILE")
export CTL_HOST
source_env_600 "$ENVFILE"   # the controller's settings for this slot (no secret values since fc22785)
# The world token: secrets.env has CHATBOT_WORLD_TOKEN (the controller-side name); the worker reads
# FLOCK_WORLD_TOKEN (WORLD_TOKEN_ENV).
export FLOCK_WORLD_TOKEN="${FLOCK_WORLD_TOKEN:-${CHATBOT_WORLD_TOKEN:-}}"
require_var FLOCK_URL FLOCK_TOKEN WORKER_ID WORKER_INCARNATION SWARM TYPE DEADLINE DRAIN_WINDOW BUCKET \
  HF_TOKEN FLOCK_WORLD_TOKEN OPENAI_API_KEY CHATBOT_BOARD_INVITE
case $WORKER_ID in *[!A-Za-z0-9._-]*|"") die "WORKER_ID=$WORKER_ID: unexpected characters";; esac

# ---------------------------------------------------------------- agent files: the container root, never /fsx
# No WORKER_ID in these paths (see the header): an agent's workdir must be the same absolute
# path on every worker that ever hosts it, or its Pi session cannot be resumed.
export WORK_ROOT=/work/flock/agents/work        # agents' workdirs WORK_ROOT/<agent> (restored from <BUCKET>/<swarm>/<type>/<agent>/)
export FLOCK_LLM_HOME=/work/flock/agents/llm    # llm harness sessions
export SYNC_INTERVAL=${SYNC_INTERVAL:-180} HEARTBEAT_INTERVAL=${HEARTBEAT_INTERVAL:-30} LOG_LEVEL=${LOG_LEVEL:-INFO}
select_worker_image "$TYPE"        # per-type image (common.sh): run.env CONTAINER_IMAGE_<type>, else the default
container_srun_args "$HOSTS_DIR"   # -> CTR_SRUN; <run>/hosts ro for FLOCK_CTL_HOST_FILE

# ---------------------------------------------------------------- signals: forward to the srun step, wait, never requeue
SRUN_PID= ; STOPPING=0
fwd_term() {  # SIGTERM to flock-worker through slurmd, never `kill -TERM $SRUN_PID` (srun would SIGKILL it: common.sh step_signal)
  step_signal TERM
}
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  # Worst case in this handler: USR1 forwarded at once, SIGTERM at +3 s, then the wait loop
  # (flock-worker's own handler must finish its drain + sync within ~25 s; the kill comes at 30 s).
  # With an empty SRUN_PID (signal before the step started) the start-up guard below exits.
  [ "$STOPPING" = usr1 ] && return 0
  STOPPING=usr1
  log "SIGUSR1 (Spot reclaim): forwarding to the flock-worker step (srun pid ${SRUN_PID:-<none>}), SIGTERM in 3 s; no requeue"
  [ -n "$SRUN_PID" ] || return 0
  kill -USR1 "$SRUN_PID" 2>/dev/null
  sleep 3
  if kill -0 "$SRUN_PID" 2>/dev/null; then
    log "flock-worker step still alive 3 s after USR1: SIGTERM"
    fwd_term
  fi
}
on_term() {
  [ "$STOPPING" = 0 ] || return 0
  STOPPING=term
  log "SIGTERM (scancel by the controller, preemption or time limit): forwarding to flock-worker"
  [ -n "$SRUN_PID" ] && fwd_term
}
trap on_usr1 USR1
trap on_term TERM INT

# ---------------------------------------------------------------- run
log "worker=$WORKER_ID incarnation=${WORKER_INCARNATION:0:8}... swarm=$SWARM type=$TYPE deadline=$DEADLINE ctl=$CTL_HOST"
log "container=$CONTAINER_IMAGE WORK_ROOT=$WORK_ROOT FLOCK_LLM_HOME=$FLOCK_LLM_HOME BUCKET=${BUCKET:-<none>}"
[ "$STOPPING" = 0 ] || { log "signal before flock-worker started: exit"; exit 0; }
# srun starts under `trap '' USR1` like the other children (srun installs its own forwarding
# handler; the container task's USR1 disposition is set by worker-inner.sh).
trap '' USR1
srun "${CTR_SRUN[@]}" bash "$SLURM_DIR/worker-inner.sh" &
SRUN_PID=$!
trap on_usr1 USR1
while kill -0 "$SRUN_PID" 2>/dev/null; do
  wait "$SRUN_PID"
done
wait "$SRUN_PID"; rc=$?   # the real status even when the last wait was cut by a trap
log "flock-worker step exited with $rc"
exit "$rc"
