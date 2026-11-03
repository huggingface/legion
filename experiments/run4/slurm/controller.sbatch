#!/bin/bash
# controller.sbatch: the flock controller as one CPU Slurm job; it applies the spec on the first start.
#
#   mkdir -p /fsx/carlos_patino/prj-eliza/runs/<RUN>/logs
#   sbatch -J flock-ctl-<RUN> -o /fsx/carlos_patino/prj-eliza/runs/<RUN>/logs/%x-%j.out \
#          ${CTL_PARTITION:+-p $CTL_PARTITION} ${CTL_QOS:+--qos=$CTL_QOS} \
#          /fsx/carlos_patino/prj-eliza/slurm/controller.sbatch <RUN>
#
# Fact sheet (notes/run3-slurm-facts.md) §3 controller, §5 restart, §7 proposed layout.
# - Step 1: `flock-controller` (FastAPI on 7860, SQLite in FLOCK_DATA under /tmp, state pushed to
#   FLOCK_STATE_URL every 60 s and once more on SIGTERM; restored at start).
# - Step 2, once GET /healthz answers: `flock apply` the experiment's spec.yaml only when the
#   swarm is not on the controller (HTTP 404: the first start); on 200 (a Spot requeue restored
#   it) nothing, so a requeue never undoes a stop or resets live counts. The controller runs the
#   ramp (replicas_ramp). The step runs in a background subshell: the apply preflight can be
#   slow and bash runs a trap only once the foreground command returns.
# - Spot reclaim: SIGUSR1 -> SIGTERM to flock-controller (bounded final push), wait, then
#   `scontrol requeue`. The watcher kills the job 30 s after SIGUSR1 (cpu-partition.md), so the
#   final push is bounded well under that (FLOCK_FINAL_PUSH_TIMEOUT below). The watcher sends
#   SIGUSR1 to EVERY process of the job, not only to this shell: the children are started while
#   the shell ignores USR1 (SIG_IGN survives fork and exec, a trap handler does not), so the
#   direct signal is a no-op for them and only this script's handler acts.
# - Workers are sbatch'ed BY the controller through its Slurm jobs client (FLOCK_COMPUTE=slurm,
#   being written in flock): it reads the FLOCK_SLURM_* knobs below and runs worker.sbatch.
# - The controller stays on the HOST (no container): its Slurm jobs client runs sbatch, squeue,
#   sacct and scancel, which do not work inside the container (no Slurm client, no munge socket).
# No `--export`: everything comes from common.sh (secrets.env, run.env) inside the job.
#
#SBATCH --job-name=flock-ctl
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M           # hopper-cpu MaxMemPerCPU=2000: more than 2000 MB per CPU silently raises the CPU count (4G / 2 = 2048)
#SBATCH --time=60:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton  # one controller per run (the job name flock-ctl-<RUN> is per run)
#SBATCH --open-mode=append      # a requeue keeps the job id: append to the .out instead of truncating it
#SBATCH --signal=B:TERM@120   # 120 s before the time limit: SIGTERM to the batch shell -> clean stop (final push / sync)
#SBATCH -o /fsx/carlos_patino/prj-eliza/runs/_slurm-logs/%x-%j.out

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal
RUN="${1:-${RUN:-}}"
# shellcheck disable=SC2034  # LOG_TAG is read by log() in common.sh
LOG_TAG=controller
. /fsx/carlos_patino/prj-eliza/slurm/common.sh

# ---------------------------------------------------------------- signals (first thing: a reclaim during start-up must requeue too)
CTL_PID= ; APPLY_PID= ; REQUEUE=0 ; STOPPING=0
stop_children() {  # SIGTERM to the apply step first (it may be mid `flock apply`), then the controller
  STOPPING=1
  [ -n "$APPLY_PID" ] && kill -TERM "$APPLY_PID" 2>/dev/null
  [ -n "$CTL_PID" ] && kill -TERM "$CTL_PID" 2>/dev/null
}
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  # Worst case: SIGTERM -> final push <= FLOCK_FINAL_PUSH_TIMEOUT (10 s) + 5 s graceful drain
  # = 15 s, capped by the 18 s loop below, + `scontrol requeue` (~1 s) = 19 s of the 30 s budget.
  # With empty PIDs (signal during start-up) nothing is waited for: it requeues at once.
  log "SIGUSR1 (Spot reclaim): SIGTERM to flock-controller for its final push, then requeue"
  REQUEUE=1
  stop_children
  for _ in $(seq 1 18); do
    { [ -n "$CTL_PID" ] && kill -0 "$CTL_PID" 2>/dev/null; } ||
      { [ -n "$APPLY_PID" ] && kill -0 "$APPLY_PID" 2>/dev/null; } || break
    sleep 1
  done
  log "requeue: scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || log "scontrol requeue failed"
}
on_term() {  # scancel / preemption / time limit: a clean stop, no requeue
  log "SIGTERM: stopping controller and apply step (no requeue)"
  stop_children
}
trap on_usr1 USR1
trap on_term TERM INT

# ---------------------------------------------------------------- inputs
require_var HF_TOKEN FLOCK_TOKEN CHATBOT_WORLD_TOKEN CHATBOT_BOARD_INVITE OPENAI_API_KEY
require_var RUN3_BUCKET RUN3_NAMESPACE CHATBOT_BOARD_URL CHATBOT_CHANNEL MODEL_WORKER
[ -n "${OPENAI_BASE_URLS:-}" ] || die "OPENAI_BASE_URLS (comma-separated, one URL is fine) is unset in $RUN_ENV"
command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v sacct >/dev/null \
  || die "sbatch/squeue/sacct are not on PATH on $(hostname): the Slurm jobs client cannot launch workers"
# FLOCK_COMPUTE=slurm needs the backend in the venv: controller/main.py only special-cases `local`,
# so a venv without it would silently fall back to the HF Jobs backend. Fail fast instead.
"$VENV/bin/python" -c 'import flock.controller.slurm' 2>/dev/null \
  || die "venv has no slurm backend (import flock.controller.slurm failed); update the repo and venv"
[ "${COMPUTE_KIND:-slurm}" = slurm ] \
  || die "COMPUTE_KIND=$COMPUTE_KIND: this layout runs workers as Slurm jobs (spec.yaml compute.kind): unset it"

RUN3_BUCKET=${RUN3_BUCKET%/}
CTL_PORT=7860
CTL_HOST=$(hostname)
EXPERIMENT=${RUN3_EXPERIMENT:-run4}              # experiments/<name>: run4, hierarchy, voting or teams
EXP=$RUN_DIR/experiment                         # a copy of experiments/$EXPERIMENT, taken at the first start
SPEC=$EXP/spec.yaml                             # the one spec: compute, ramp and ${VAR}s inline
HEALTH_WAIT_S=${HEALTH_WAIT_S:-300}
WORLD_WAIT_S=${WORLD_WAIT_S:-1800}               # first launch only: how long `flock apply` waits for the world

# ---------------------------------------------------------------- controller environment (fact sheet §2)
export FLOCK_MODE=space                         # space mode: workers get FLOCK_URL as-is (main.py)
export PORT=$CTL_PORT
export FLOCK_URL="http://$CTL_HOST:$CTL_PORT"   # this node; also what workers are handed in space mode
# PROVISIONAL: today main.py reads FLOCK_WORKER_URL only in job mode. If the Slurm client (or main.py)
# learns to hand workers this template instead of FLOCK_URL, worker.sbatch already exports CTL_HOST
# from <run>/controller.host, so a requeued controller would not force a fleet-wide relaunch.
export FLOCK_WORKER_URL='http://${CTL_HOST}:7860'
export FLOCK_COMPUTE=slurm                      # the Slurm backend (controller/slurm.py), checked above
# /tmp is per job on this cluster (wiped at job end), so the `rm -rf` below can only hit this job's
# files; the job id in the path keeps it safe even on a node where /tmp were shared.
CTL_TMP=/tmp/flock-controller-$SLURM_JOB_ID
export FLOCK_DATA=$CTL_TMP/data                 # restored from the bucket at start
export HF_HOME=$CTL_TMP/hf                      # never ~/.cache/huggingface (token, xet cache)
export FLOCK_STATE_URL="$RUN3_BUCKET/controller/$RUN"
export FLOCK_STATE_RESTORE=1
export FLOCK_STATE_SYNC_INTERVAL=${FLOCK_STATE_SYNC_INTERVAL:-60}
# Spot budget: the node watcher kills the job 30 s after SIGUSR1. The controller's shutdown is
# push (bounded here) + 5 s drain = 15 s, and on_usr1 still needs time to `scontrol requeue`.
export FLOCK_FINAL_PUSH_TIMEOUT=${FLOCK_FINAL_PUSH_TIMEOUT:-10}
export FLOCK_RECONCILE_INTERVAL=${FLOCK_RECONCILE_INTERVAL:-5}
export FLOCK_JOB_LABELS="run=$RUN"              # -> worker labels, so `list` finds this run's workers
# PROVISIONAL names, read by the Slurm jobs client that is being added to flock (fact sheet §7,
# "SlurmJobsClient"). They are what `launch` passes to sbatch and where it writes/reads files.
# The client must NOT pass `--export` to sbatch: worker.sbatch reads secrets.env and its env file
# itself; with the default export the worker inherits this environment (same user: tolerable).
# Worker size, partition and time are the spec's `compute:` (${WORKER_CPUS:-4}, ${WORKER_MEM:-7G}
# from run.env; the spec wins over FLOCK_SLURM_*). hopper-cpu MaxMemPerCPU=2000: more than 2000 MB
# per CPU silently raises the CPU count, so check the run.env values here (7G / 4 = 1792); the
# Slurm client checks the 28 CPU / 54G node rule (slurm.py size_problem).
_cpus=${WORKER_CPUS:-4} _mem=${WORKER_MEM:-7G}
case $_cpus in ''|0|*[!0-9]*) die "WORKER_CPUS=$_cpus: must be a positive integer";; esac
_mem_n=${_mem%[GgMm]}                           # sbatch --mem: an integer, bare (MB) or with M or G
case $_mem_n in ''|*[!0-9]*) die "WORKER_MEM=$_mem: use an integer with M or G (e.g. 7G, 11G, 3800M)";; esac
case $_mem in *[Gg]) _mem_mb=$(( _mem_n * 1024 ));; *) _mem_mb=$_mem_n;; esac
_mem_per_cpu=$(( (_mem_mb + _cpus - 1) / _cpus ))  # ceiling, as Slurm rounds
[ "$_mem_per_cpu" -le 2000 ] \
  || die "WORKER_MEM=$_mem / WORKER_CPUS=$_cpus = $_mem_per_cpu MB per CPU, over hopper-cpu MaxMemPerCPU=2000: raise WORKER_CPUS or lower WORKER_MEM"
unset _cpus _mem _mem_n _mem_mb _mem_per_cpu
export FLOCK_SLURM_WORKER_SCRIPT=$SLURM_DIR/worker.sbatch
export FLOCK_SLURM_LAUNCH_DIR=$RUN_DIR/launch   # one mode-600 env file per worker launch (worker.env.example)
export FLOCK_SLURM_LOG_DIR=$RUN_DIR/logs        # sbatch -o <dir>/<worker id>-%j.out, so logs_tail can read them
# The controller's hostname for the workers (slurm.py host_file -> FLOCK_CTL_HOST_FILE in the env
# file; FLOCK_URL becomes http://${CTL_HOST}:7860). Under <run>/hosts: mounted ro in the worker's
# container, and a directory mount sees the rename publish_host does.
export FLOCK_SLURM_HOST_FILE=$HOSTS_DIR/controller.host
export FLOCK_SLURM_RUN=$RUN FLOCK_SLURM_VENV=$VENV FLOCK_SLURM_SECRETS_FILE=$SECRETS_FILE   # -> RUN, FLOCK_VENV, FLOCK_SECRETS_FILE in the env file
# PROVISIONAL name (flock is adding a re-read of the world's host file): where the world's
# hostname lives, rewritten by world.sbatch at every start (a requeue may move it).
export FLOCK_WORLD_HOST_FILE=$HOSTS_DIR/world.host

rm -rf "$CTL_TMP" && mkdir -p "$FLOCK_DATA" "$HF_HOME"

# ---------------------------------------------------------------- step 1: the controller
[ "$STOPPING" = 0 ] || exit 0  # a signal arrived during the start-up (already requeued on USR1)
publish_host controller   # <run>/hosts/controller.host (+ the <run>/controller.host symlink), every start
log "controller.host=$CTL_HOST; FLOCK_URL=$FLOCK_URL; state=$FLOCK_STATE_URL; restarts=${SLURM_RESTART_COUNT:-0}"
# Started while USR1 is ignored: SIG_IGN is inherited across fork and exec (a trap handler is
# not), so the watcher's direct SIGUSR1 does not kill flock-controller before on_usr1 has sent
# it the SIGTERM for its final push. The real handler is restored right after.
trap '' USR1
"$VENV/bin/flock-controller" >> "$RUN_DIR/logs/controller-$RUN.log" 2>&1 &
CTL_PID=$!
trap on_usr1 USR1
log "flock-controller pid $CTL_PID; log $RUN_DIR/logs/controller-$RUN.log"

t0=$(date +%s)
until [ "$(http_status "http://127.0.0.1:$CTL_PORT/healthz")" = 200 ]; do  # /healthz: no auth (api.py)
  kill -0 "$CTL_PID" 2>/dev/null || { log "flock-controller exited before becoming healthy"; wait "$CTL_PID"; exit 1; }
  [ $(( $(date +%s) - t0 )) -lt "$HEALTH_WAIT_S" ] || { log "no /healthz after ${HEALTH_WAIT_S}s"; stop_children; wait "$CTL_PID"; exit 1; }
  [ "$STOPPING" = 1 ] && break
  sleep 2
done
log "controller healthy at $FLOCK_URL/healthz"

# ---------------------------------------------------------------- step 2: apply on the first start only
apply_step() {  # runs in a background subshell (below)
  # The experiment folder, copied once per run (the spec's prompts/, tools/, protocols/ are read
  # relative to the spec file). One spec.yaml: run.env (sourced by common.sh) fills its ${VAR}s.
  if [ ! -d "$EXP" ]; then
    mkdir -p "$EXP" && cp -aL "$REPO/experiments/$EXPERIMENT/." "$EXP/" || { log "copy of experiments/$EXPERIMENT failed"; return 1; }
    log "experiment folder: $EXP (from $REPO/experiments/$EXPERIMENT)"
  fi
  local swarm code
  swarm=$("$VENV/bin/python" -c 'import sys; from flock.specrender import render; print(render(sys.argv[1])["swarm"])' "$SPEC") \
    || { log "cannot read the swarm name from $SPEC"; return 1; }
  code=$(http_status "http://127.0.0.1:$CTL_PORT/swarms/$swarm" X-Flock-Token FLOCK_TOKEN)
  if [ "$code" = 200 ]; then  # a requeue: the controller restored the swarm; never re-apply (S2)
    log "swarm $swarm is live on the controller: nothing to apply (its ramp continues from the restored state)"
    return 0
  fi
  [ "$code" = 404 ] || { log "GET /swarms/$swarm -> $code"; return 1; }
  # First start only: apply checks the world, so wait for it (a requeue restores its state and
  # follows the world's host by itself; it never waits).
  local t0 world_host
  t0=$(date +%s)
  until world_host=$(tr -d '[:space:]' < "$HOSTS_DIR/world.host" 2>/dev/null) && [ -n "$world_host" ] &&
        [ "$(http_status "http://$world_host:7860/manifest" X-World-Token CHATBOT_WORLD_TOKEN)" = 200 ]; do
    [ $(( $(date +%s) - t0 )) -lt "$WORLD_WAIT_S" ] \
      || { log "world unavailable after ${WORLD_WAIT_S}s: start world.sbatch, then flock apply by hand (README)"; return 1; }
    sleep 10
  done
  log "world: http://$world_host:7860 answers /manifest"
  log "swarm $swarm not on the controller: flock apply $SPEC ${RUN3_APPLY_ARGS:-}"
  # shellcheck disable=SC2086
  (cd "$EXP" && "$VENV/bin/flock" apply "$SPEC" ${RUN3_APPLY_ARGS:-}) >> "$RUN_DIR/logs/apply-$RUN.log" 2>&1 \
    || { log "flock apply failed: see $RUN_DIR/logs/apply-$RUN.log"; return 1; }
  log "applied: the controller ramps the replicas (replicas_ramp in the spec)"
}
if [ "$STOPPING" = 0 ]; then
  # In the background: `flock apply` (preflight included) can take a while and bash would hold the
  # USR1 trap until it returned. Started under `trap '' USR1` like the controller (see step 1).
  trap '' USR1
  ( apply_step || log 'apply step failed: the controller keeps running; fix and `flock apply` by hand (README)' ) &
  APPLY_PID=$!
  trap on_usr1 USR1
  log "apply step pid $APPLY_PID"
fi

# ---------------------------------------------------------------- wait (returns early on a signal: loop)
rc=0
while kill -0 "$CTL_PID" 2>/dev/null; do
  wait "$CTL_PID"; rc=$?
done
if [ -n "$APPLY_PID" ]; then  # the apply step ends with the controller (same loop: wait returns on a signal)
  kill -TERM "$APPLY_PID" 2>/dev/null
  while kill -0 "$APPLY_PID" 2>/dev/null; do
    wait "$APPLY_PID" 2>/dev/null
  done
fi
log "flock-controller exited with $rc (requeue=$REQUEUE)"
if [ "$REQUEUE" = 1 ]; then
  exit 0  # the job is already requeued; the restart writes a new controller.host
fi
exit "$rc"
