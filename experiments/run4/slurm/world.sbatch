#!/bin/bash
# world.sbatch: the run-3 world (worlds/chatbot4, `flock world serve`) as one CPU Slurm job.
#
#   mkdir -p /fsx/carlos_patino/prj-eliza/runs/<RUN>/logs
#   sbatch -J flock-world-<RUN> -o /fsx/carlos_patino/prj-eliza/runs/<RUN>/logs/%x-%j.out \
#          /fsx/carlos_patino/prj-eliza/slurm/world.sbatch <RUN>
#
# This batch script stays on the HOST (signals, <run>/hosts/world.host, `scontrol requeue`). The
# world itself runs in an Enroot/Pyxis container through `srun ... bash world-inner.sh`, which
# does restore -> serve -> periodic sync -> final sync INSIDE the container: the workspace lives on
# the container root (/work/world/ws, node disk, deleted with the container), home is not mounted
# and /fsx is visible only read-only (venv, uv Python, repo, slurm dir, <run>/world).
# Mirrors experiments/run4/world-space/start.sh (fact sheet §3 world, §5 restart):
# - CHATBOT_WS is a fresh EMPTY dir in the container, so `ws/` is restored from <CHATBOT_BUCKET>/ws
#   at every start; the whole workspace is pushed every WS_SYNC_S (300 s); best/ is mirrored by
#   the world itself at each promotion.
# - `runs.parallel` is read ONCE at start and 0 means os.cpu_count() = the NODE's CPUs, not the
#   job's: a run-specific <run>/world/world.yaml (repo default + runs.parallel: 14) is written here
#   and handed over with CHATBOT_CONFIG (the only mechanism: `flock world serve` has no config
#   flag). <run>/world is mounted ro, so an edit there still reaches the world (re-read on mtime).
#   16 CPUs = 14 bot slots + 2 for the server and the sync.
# - <run>/hosts/world.host (and the <run>/world.host symlink) is rewritten at EVERY start, a
#   requeue onto a new node included.
# - Spot reclaim: the watcher sends SIGUSR1 to this shell, to srun and to every process in the
#   container. srun is started while this shell ignores USR1, then on_usr1 forwards USR1 to the
#   step (`kill -USR1 $SRUN_PID`), waits for it (<= USR1_WAIT_S, 22 s; world-inner.sh: stop 5 s,
#   SIGKILL, final sync <= 12 s), then `scontrol requeue`. Worst case 22 + ~1 s of the 30 s.
# - Clean stop: `scancel --signal=TERM --batch <jobid>` (this shell only, no kill): on_term writes
#   the final-sync bound (FINAL_SYNC_LONG_S, 600 s, capped by the time left) to
#   <run>/world/final-sync-bound and forwards SIGUSR2 to the step (world-inner.sh's clean stop),
#   and the job ends by itself. A plain `scancel` (TERM to every process, SIGKILL after KillWait=30
#   s) gets a best-effort 25 s sync. on_term tells the two apart by the job state.
# - --dependency=singleton: one world per run (the job name is per run); --open-mode=append: a
#   requeue (same job id) appends to the same .out instead of truncating it.
# No `--export`: everything comes from common.sh (secrets.env, run.env) inside the job.
# WORLD_VENV / WORLD_REPO in run.env switch this job (not the controller/workers) to another venv
# and checkout; common.sh applies them at run time, so a requeue picks them up too.
#
#SBATCH --job-name=flock-world
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=30G             # hopper-cpu MaxMemPerCPU=2000: 30G / 16 = 1920 MB per CPU (more would silently raise the CPU count)
#SBATCH --time=54:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120   # 120 s before the time limit: SIGTERM to the batch shell -> clean stop (final push / sync)
#SBATCH -o /fsx/carlos_patino/prj-eliza/runs/_slurm-logs/%x-%j.out

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal
RUN="${1:-${RUN:-}}"
# shellcheck disable=SC2034  # LOG_TAG is read by log() in common.sh
LOG_TAG=world
. /fsx/carlos_patino/prj-eliza/slurm/common.sh

# ---------------------------------------------------------------- signals (first thing: a reclaim during start-up must requeue too)
SRUN_PID= ; REQUEUE=0 ; STOPPING=0
TERM_MODE=                  # set by on_term: signal-only (job still RUNNING) or cancel (plain scancel / requeue)
FINAL_SYNC_S=${FINAL_SYNC_S:-12}              # Spot path bound for the last sync (inside the 30 s budget)
FINAL_SYNC_LONG_S=${FINAL_SYNC_LONG_S:-600}   # clean stop (signal-only TERM) or crash: not under a kill
CANCEL_SYNC_S=25                              # plain scancel: SIGKILL comes KillWait=30 s after the TERM
USR1_WAIT_S=22                                # how long on_usr1 waits for the step before the requeue
WORLD_CTL=$RUN_DIR/world                      # world.yaml + final-sync-bound; mounted ro in the container
mkdir -p "$WORLD_CTL" || die "cannot create $WORLD_CTL"
rm -f "$WORLD_CTL/final-sync-bound"
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  # Worst case: the step (stop 5 s + sync 12 s + 2 s kill grace + container teardown) is waited
  # for at most USR1_WAIT_S = 22 s, then `scontrol requeue` (~1 s). With no step yet (signal
  # during the world.yaml step) it requeues at once.
  [ "$REQUEUE" = 1 ] && return 0
  REQUEUE=1; STOPPING=1
  log "SIGUSR1 (Spot reclaim): USR1 to the world step (srun pid ${SRUN_PID:-<none>}): stop, final sync; then requeue"
  if [ -n "$SRUN_PID" ]; then
    kill -USR1 "$SRUN_PID" 2>/dev/null
    for _ in $(seq 1 "$USR1_WAIT_S"); do
      kill -0 "$SRUN_PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$SRUN_PID" 2>/dev/null && log "world step still running after ${USR1_WAIT_S}s: requeue anyway"
  fi
  log "requeue: scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || log "scontrol requeue failed"
}
on_term() {  # clean stop; world-inner.sh runs the final sync after the server exits, bounded by the file below
  if [ "$REQUEUE" = 1 ] || [ "$STOPPING" = 1 ]; then  # our own requeue's TERM, or a repeat
    return 0
  fi
  STOPPING=1
  # Signal-only TERM (`scancel --signal=TERM --batch`, or --signal=B:TERM@120 near the time limit):
  # the job stays RUNNING. A plain scancel (or a preemption) marks it CANCELLED/COMPLETING at once.
  local state bound left
  state=$(timeout 5 squeue -h -j "$SLURM_JOB_ID" -o %T 2>/dev/null)
  if [ "$state" = RUNNING ]; then
    TERM_MODE=signal-only
    bound=$FINAL_SYNC_LONG_S
    left=$(time_left_s)
    [ -n "$left" ] && [ $(( left - 10 )) -lt "$bound" ] && bound=$(( left > 20 ? left - 10 : 10 ))
  else
    TERM_MODE=cancel
    bound=$CANCEL_SYNC_S
  fi
  printf '%s\n' "$bound" > "$WORLD_CTL/final-sync-bound.tmp" && mv -f "$WORLD_CTL/final-sync-bound.tmp" "$WORLD_CTL/final-sync-bound"
  log "SIGTERM ($TERM_MODE, job state ${state:-?}): USR2 to the world step (stop, then final sync <= ${bound}s; no requeue)"
  # USR2, not TERM: srun forwards USR1/USR2 to the task as is (tests/ctr-signal-test.slurm).
  [ -n "$SRUN_PID" ] && kill -USR2 "$SRUN_PID" 2>/dev/null
}
time_left_s() {  # seconds before the time limit (squeue %L: [D-]HH:MM:SS or MM:SS); empty if unknown
  local l d=0 h=0 m=0 sec=0
  l=$(timeout 5 squeue -h -j "$SLURM_JOB_ID" -o %L 2>/dev/null) || return 0
  case $l in *-*) d=${l%%-*}; l=${l#*-};; esac
  case $l in
    *:*:*) IFS=: read -r h m sec <<< "$l";;
    *:*)   IFS=: read -r m sec <<< "$l";;
    *)     return 0;;                       # UNLIMITED, NOT_SET, ...
  esac
  echo $(( 10#$d * 86400 + 10#$h * 3600 + 10#$m * 60 + 10#$sec ))
}
trap on_usr1 USR1
trap on_term TERM INT

# ---------------------------------------------------------------- inputs (exported: world-inner.sh reads them in the container)
require_var HF_TOKEN CHATBOT_REPO_TOKEN CHATBOT_WORLD_TOKEN RUN3_BUCKET
RUN3_BUCKET=${RUN3_BUCKET%/}
export CHATBOT_BUCKET="${CHATBOT_BUCKET:-$RUN3_BUCKET/$RUN}"   # hf://buckets/<ns>/<bucket>/<run> (README runbook B step 2)
export B=${CHATBOT_BUCKET%/}
export WS_SYNC_S=${WS_SYNC_S:-300}
export WORLD_PORT=7860
WORLD_HOST=$(hostname)
export WORLD_SRC=$REPO/worlds/chatbot4            # `flock world serve module:Class` imports from the cwd
export CHATBOT_WS=/work/world/ws                  # container root: restored empty at each start, deleted with the container
export FLOCK_WORLD_TOKEN=$CHATBOT_WORLD_TOKEN     # start.sh: the token the controller/workers send as X-World-Token
export FINAL_SYNC_S FINAL_SYNC_LONG_S CANCEL_SYNC_S WORLD_CTL

# Run 4's world has no probe/promote configuration.
export CHATBOT_RUN_SLOTS=${CHATBOT_RUN_SLOTS:-4}

# ---------------------------------------------------------------- serve (in the container)
[ "$STOPPING" = 0 ] || exit 0  # a signal during the start-up (already requeued on USR1; nothing to sync)
publish_host world             # every start: a requeue may land on another node
container_srun_args "$WORLD_CTL"   # -> CTR_SRUN; <run>/world ro (world.yaml, final-sync-bound)
log "world.host=$WORLD_HOST; url http://$WORLD_HOST:$WORLD_PORT; ws=$CHATBOT_WS (container); bucket=$B; restarts=${SLURM_RESTART_COUNT:-0}"
log "container=$CONTAINER_IMAGE; world log $RUN_DIR/logs/world-$RUN.log"
# srun is started while USR1 is ignored, like every child here; inside, world-inner.sh installs its
# own traps and starts the server and the syncs with USR1 ignored. stdout (the server) -> world log,
# stderr (world-inner.sh's lines, Pyxis) -> this job's .out.
trap '' USR1
srun "${CTR_SRUN[@]}" bash "$SLURM_DIR/world-inner.sh" >> "$RUN_DIR/logs/world-$RUN.log" &
SRUN_PID=$!
trap on_usr1 USR1
log "world step: srun pid $SRUN_PID"

t0=$(date +%s)   # the restore runs first, inside the container: allow for it
until [ "$(http_status "http://127.0.0.1:$WORLD_PORT/manifest" X-World-Token FLOCK_WORLD_TOKEN)" = 200 ]; do
  kill -0 "$SRUN_PID" 2>/dev/null || { log "world step exited before /manifest answered"; break; }
  [ "$STOPPING" = 1 ] && break
  [ $(( $(date +%s) - t0 )) -lt 900 ] || { log "no /manifest after 900s (restore + start); still waiting on the step"; break; }
  sleep 2
done
kill -0 "$SRUN_PID" 2>/dev/null && [ "$STOPPING" = 0 ] && log "world answers /manifest at http://$WORLD_HOST:$WORLD_PORT"

# ---------------------------------------------------------------- wait (returns early on a signal: loop)
while kill -0 "$SRUN_PID" 2>/dev/null; do
  wait "$SRUN_PID"
done
wait "$SRUN_PID"; rc=$?   # the real status even when the last wait was cut by a trap
log "world step exited with $rc (requeue=$REQUEUE, term=${TERM_MODE:-none}); the final sync ran inside it"
[ "$REQUEUE" = 1 ] && exit 0  # already requeued
exit "$rc"
