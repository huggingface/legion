#!/bin/bash
# controller.sbatch <run dir>: the run's flock controller as one CPU Slurm job.
#
# `flock launch` submits it (from <run dir>/slurm/, `flock init` copies it there) when no
# controller answers for the run, waits until it answers, then creates or resumes the swarm;
# `flock stop --release` cancels it after a verified checkpoint (docs/slurm.md "Deploying").
# By hand: sbatch -J flock-ctl-<run> -o <run dir>/logs/%x-%j.out <run dir>/slurm/controller.sbatch <run dir>
#
# - `flock-controller` on a free port; <run>/hosts/controller.host = host:port. SQLite in FLOCK_DATA under /tmp; the state is pushed to
#   FLOCK_STATE_URL (default <the spec's bucket>/controller/<run>) every 60 s and once more on
#   SIGTERM, and restored at start, so a requeue or a later job continues the run.
# - Workers are sbatch'ed BY the controller (FLOCK_COMPUTE=slurm) with <run>/slurm/worker.sbatch.
# - <run>/controller.hold (`flock launch --hold` writes it): this start restores every swarm held
#   and released (FLOCK_HOLD_ON_START=1); the file is removed once the controller answers, so a
#   later requeue holds nothing.
# - Spot reclaim: SIGUSR1 -> SIGTERM to flock-controller (bounded final push), wait, then
#   `scontrol requeue`. The watcher kills the job 30 s after SIGUSR1, so the final push is bounded
#   well under that (FLOCK_FINAL_PUSH_TIMEOUT below). The watcher sends SIGUSR1 to EVERY process
#   of the job, not only to this shell: the controller is started while the shell ignores USR1
#   (SIG_IGN survives fork and exec, a trap handler does not), so the direct signal is a no-op
#   for it and only this script's handler acts.
# - The controller stays on the HOST (no container): its Slurm jobs client runs sbatch, squeue,
#   sacct and scancel, which do not work inside a container (no Slurm client, no munge socket).
# No `--export`: everything comes from common.sh (cluster.env, run.env) inside the job.
#
#SBATCH --job-name=flock-ctl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M           # 1900 MB per CPU: fits a MaxMemPerCPU of 2000 (more silently raises the CPU count)
#SBATCH --time=60:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton  # one controller per run (the job name flock-ctl-<run> is per run)
#SBATCH --open-mode=append      # a requeue keeps the job id: append to the .out instead of truncating it
#SBATCH --signal=B:TERM@120   # 120 s before the time limit: SIGTERM to the batch shell -> clean stop (final push)

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal
RUN_DIR=${1:?usage: sbatch controller.sbatch <run dir>}
# shellcheck disable=SC2034  # LOG_TAG is read by common.sh
LOG_TAG=controller
# shellcheck source=common.sh
. "${RUN_DIR%/}/slurm/common.sh" || exit 2

# ---------------------------------------------------------------- signals (first thing: a reclaim during start-up must requeue too)
CTL_PID= ; REQUEUE=0 ; STOPPING=0
stop_controller() {
  STOPPING=1
  [ -n "$CTL_PID" ] && kill -TERM "$CTL_PID" 2>/dev/null
}
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  # Worst case: SIGTERM -> final push <= FLOCK_FINAL_PUSH_TIMEOUT (10 s) + 5 s graceful drain
  # = 15 s, capped by the 18 s loop below, + `scontrol requeue` (~1 s) = 19 s of the 30 s budget.
  # With an empty PID (signal during start-up) nothing is waited for: it requeues at once.
  log "SIGUSR1 (Spot reclaim): SIGTERM to flock-controller for its final push, then requeue"
  REQUEUE=1
  stop_controller
  for _ in $(seq 1 18); do
    { [ -n "$CTL_PID" ] && kill -0 "$CTL_PID" 2>/dev/null; } || break
    sleep 1
  done
  log "requeue: scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || log "scontrol requeue failed"
}
on_term() {  # scancel / preemption / time limit: a clean stop, no requeue
  log "SIGTERM: stopping the controller (final push; no requeue)"
  stop_controller
}
trap on_usr1 USR1
trap on_term TERM INT

# ---------------------------------------------------------------- inputs
require_var HF_TOKEN FLOCK_TOKEN
command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v sacct >/dev/null \
  || die "sbatch/squeue/sacct are not on PATH on $(hostname): the Slurm jobs client cannot launch workers"
# FLOCK_COMPUTE=slurm needs the backend in the venv; fail fast instead of at the first launch.
"$VENV/bin/python" -c 'import flock.controller.slurm' 2>/dev/null \
  || die "venv has no slurm backend (import flock.controller.slurm failed); update the venv"
if [ -z "${FLOCK_STATE_URL:-}" ]; then   # run.env may set it; else <the spec's bucket>/controller/<run>
  BUCKET_ID=$(cat "$RUN_DIR/hosts/bucket_id") \
    || die "cannot read the resolved bucket_id (run \`flock launch $SPEC\`)"
  [ -n "$BUCKET_ID" ] || die "$SPEC has no bucket_id: the controller's state has nowhere to go"
  FLOCK_STATE_URL=hf://buckets/$BUCKET_ID/controller/$RUN
fi
export FLOCK_STATE_URL

CTL_PORT=$(free_port) || die "no free port on $(hostname)"   # the world may share this node
CTL_HOST=$(hostname)
HEALTH_WAIT_S=${HEALTH_WAIT_S:-300}

# ---------------------------------------------------------------- controller environment
export FLOCK_MODE=space                         # space mode: workers get FLOCK_URL as-is (main.py)
use_port() {  # CTL_PORT, and the URLs built from it
  CTL_PORT=$1
  export PORT=$CTL_PORT
  export FLOCK_URL="http://$CTL_HOST:$CTL_PORT"   # this node; also what workers are handed in space mode
  # What workers are told: they fill ${CTL_HOST} from <run>/hosts/controller.host (host:port, so the
  # port of a requeued controller is followed too) without a fleet-wide relaunch.
  export FLOCK_WORKER_URL='http://${CTL_HOST}:'"$CTL_PORT"
}
use_port "$CTL_PORT"
export FLOCK_COMPUTE=slurm                      # the Slurm backend (controller/slurm.py), checked above
# /tmp is per job on most clusters (wiped at job end); the job id in the path keeps the `rm -rf`
# below on this job's files even where /tmp is shared.
CTL_TMP=/tmp/flock-controller-$SLURM_JOB_ID
export FLOCK_DATA=$CTL_TMP/data                 # restored from the bucket at start
export HF_HOME=$CTL_TMP/hf                      # never ~/.cache/huggingface (token, xet cache)
export FLOCK_STATE_RESTORE=1
export FLOCK_STATE_SYNC_INTERVAL=${FLOCK_STATE_SYNC_INTERVAL:-60}
# Spot budget: the node watcher kills the job 30 s after SIGUSR1. The controller's shutdown is
# push (bounded here) + 5 s drain = 15 s, and on_usr1 still needs time to `scontrol requeue`.
export FLOCK_FINAL_PUSH_TIMEOUT=${FLOCK_FINAL_PUSH_TIMEOUT:-10}
export FLOCK_RECONCILE_INTERVAL=${FLOCK_RECONCILE_INTERVAL:-5}
export FLOCK_JOB_LABELS="run=$RUN"              # -> worker labels, so `list` finds this run's workers
# The Slurm jobs client (docs/slurm.md "The launch contract"). Worker size, partition and time are
# the spec's `compute:`; the spec wins over FLOCK_SLURM_*. The cluster profile's limits:
# SLURM_MAX_MEM_PER_CPU_MB (the partition's MaxMemPerCPU: more silently raises the CPU count) is
# checked here; FLOCK_SLURM_MAX_CPUS/_MEM by the Slurm client and `flock init`. Unset: not checked.
export FLOCK_SLURM_PARTITION=${FLOCK_SLURM_PARTITION:-${SLURM_PARTITION:-}}
if [ -n "${SLURM_MAX_MEM_PER_CPU_MB:-}" ]; then
  _cpus=${WORKER_CPUS:-4} _mem=${WORKER_MEM:-7G}
  case $_cpus in ''|0|*[!0-9]*) die "WORKER_CPUS=$_cpus: must be a positive integer";; esac
  _mem_n=${_mem%[GgMm]}                           # sbatch --mem: an integer, bare (MB) or with M or G
  case $_mem_n in ''|*[!0-9]*) die "WORKER_MEM=$_mem: use an integer with M or G (e.g. 7G, 11G, 3800M)";; esac
  case $_mem in *[Gg]) _mem_mb=$(( _mem_n * 1024 ));; *) _mem_mb=$_mem_n;; esac
  _mem_per_cpu=$(( (_mem_mb + _cpus - 1) / _cpus ))  # ceiling, as Slurm rounds
  [ "$_mem_per_cpu" -le "$SLURM_MAX_MEM_PER_CPU_MB" ] \
    || die "WORKER_MEM=$_mem / WORKER_CPUS=$_cpus = $_mem_per_cpu MB per CPU, over SLURM_MAX_MEM_PER_CPU_MB=$SLURM_MAX_MEM_PER_CPU_MB: raise WORKER_CPUS or lower WORKER_MEM"
  unset _cpus _mem _mem_n _mem_mb _mem_per_cpu
fi
export FLOCK_SLURM_WORKER_SCRIPT=$SLURM_DIR/worker.sbatch   # the run's copy
export FLOCK_SLURM_LAUNCH_DIR=$RUN_DIR/launch   # one mode-600 env file per worker launch
export FLOCK_SLURM_LOG_DIR=$RUN_DIR/logs        # sbatch -o <dir>/<worker id>-%j.out, so logs_tail can read them
# The controller's host:port for the workers (slurm.py host_file -> FLOCK_CTL_HOST_FILE in the env
# file; FLOCK_URL becomes http://${CTL_HOST}:<port>, the file's host:port wins).
export FLOCK_SLURM_HOST_FILE=$HOSTS_DIR/controller.host
export FLOCK_SLURM_RUN=$RUN FLOCK_SLURM_VENV=$VENV FLOCK_SLURM_SECRETS_FILE=${SECRETS_FILE:-}
export FLOCK_WORLD_HOST_FILE=$HOSTS_DIR/world.host   # world.sbatch writes it once the world answers

rm -rf "$CTL_TMP" && mkdir -p "$FLOCK_DATA" "$HF_HOME"
HOLD_FILE=$RUN_DIR/controller.hold
[ -f "$HOLD_FILE" ] && export FLOCK_HOLD_ON_START=1

# ---------------------------------------------------------------- the controller
[ "$STOPPING" = 0 ] || exit 0  # a signal arrived during the start-up (already requeued on USR1)
start_controller() {
  publish_host controller "$CTL_PORT"   # every start: a requeue may land on another node
  log "controller.host=$CTL_HOST:$CTL_PORT; FLOCK_URL=$FLOCK_URL; state=$FLOCK_STATE_URL; restarts=${SLURM_RESTART_COUNT:-0}"
  # Started while USR1 is ignored: SIG_IGN is inherited across fork and exec (a trap handler is
  # not), so the watcher's direct SIGUSR1 does not kill flock-controller before on_usr1 has sent
  # it the SIGTERM for its final push. The real handler is restored right after.
  trap '' USR1
  "$VENV/bin/flock-controller" >> "$RUN_DIR/logs/controller-$RUN.log" 2>&1 &
  CTL_PID=$!
  trap on_usr1 USR1
  log "flock-controller pid $CTL_PID; log $RUN_DIR/logs/controller-$RUN.log"
  t0=$(date +%s)
}
start_controller
RETRIED=0
until [ "$(http_status "http://127.0.0.1:$CTL_PORT/healthz")" = 200 ]; do  # /healthz: no auth (api.py)
  if ! kill -0 "$CTL_PID" 2>/dev/null; then
    wait "$CTL_PID"
    # Once: the free port may have been taken before the bind (exit 1 is not requeued by Slurm)
    [ "$RETRIED" = 0 ] && [ "$STOPPING" = 0 ] || { log "flock-controller exited before becoming healthy"; exit 1; }
    RETRIED=1
    use_port "$(free_port)" || die "no free port on $(hostname)"
    log "flock-controller exited before becoming healthy: once more on port $CTL_PORT"
    start_controller
    continue
  fi
  [ $(( $(date +%s) - t0 )) -lt "$HEALTH_WAIT_S" ] || { log "no /healthz after ${HEALTH_WAIT_S}s"; stop_controller; wait "$CTL_PID"; exit 1; }
  [ "$STOPPING" = 1 ] && break
  sleep 2
done
log "controller healthy at $FLOCK_URL/healthz: \`flock launch\` creates or resumes the swarm"
if [ "${FLOCK_HOLD_ON_START:-}" = 1 ] && [ "$STOPPING" = 0 ]; then  # held and pushed before it answered
  rm -f "$HOLD_FILE"
  log "every swarm restored held (flock launch --hold): \`flock update\`, then \`flock launch\`"
fi

# ---------------------------------------------------------------- wait (returns early on a signal: loop)
rc=0
while kill -0 "$CTL_PID" 2>/dev/null; do
  wait "$CTL_PID"; rc=$?
done
log "flock-controller exited with $rc (requeue=$REQUEUE)"
if [ "$REQUEUE" = 1 ]; then
  exit 0  # the job is already requeued; the restart writes a new controller.host
fi
exit "$rc"
