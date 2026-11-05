#!/bin/bash
# world.sbatch <run dir>: the run's world (the spec's `world.command`, run in `world.dir`) as one
# CPU Slurm job. `flock launch` submits it with the controller when the spec has a world command;
# `flock stop --release` stops it before it cancels the controller (docs/slurm.md "Deploying").
# By hand: sbatch -J flock-world-<run> -o <run dir>/logs/%x-%j.out <run dir>/slurm/world.sbatch <run dir>
# Size: sbatch -c / --mem (`flock launch` passes WORLD_CPUS / WORLD_MEM from cluster.env or run.env).
# The #SBATCH defaults below fit a small world only: set both for yours (docs/slurm.md "Sizing the world job").
#
# - The command runs on this node, as `compute: local` starts it (docs/world.md): {port} is a free
#   port, {python} the venv's python; FLOCK_WORLD_PORT, FLOCK_WORLD_HOST=0.0.0.0, FLOCK_SWARM and
#   FLOCK_WORLD_TOKEN (run.env) are set, every other FLOCK_* variable is dropped. Its output goes
#   to <run>/logs/world-<run>.log.
# - hosts/world.host = host:port once the world answers GET /manifest (every start: a requeue may land on another node). The controller follows the
#   file (FLOCK_WORLD_HOST_FILE; the spec's world.url uses ${WORLD_HOST}).
# - The world keeps its own state (flock never reads it). A clean stop (`flock stop --release`
#   sends `scancel --signal=TERM --batch`: this shell only) sends it SIGTERM and waits for it to
#   exit, so a world that saves on SIGTERM gets the time it needs; the job ends with it. A plain
#   `scancel` gives it KillWait (30 s) before the SIGKILL.
# - Spot reclaim: SIGUSR1 -> SIGTERM to the world, wait up to 22 s, `scontrol requeue`.
# - A crash requeues the job. Only FAST crashes (within WORLD_STABLE_S of the start) count toward
#   a cap of 5; past it the job sleeps with backoff (WORLD_RETRY_MIN_S doubling to
#   WORLD_RETRY_MAX_S) and starts again, so a transient problem heals without a person.
# No `--export`: everything comes from common.sh (cluster.env, run.env) inside the job.
#
#SBATCH --job-name=flock-world
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=60:00:00
#SBATCH --requeue
#SBATCH --dependency=singleton  # one world per run (the job name flock-world-<run> is per run)
#SBATCH --open-mode=append
#SBATCH --signal=B:TERM@120   # 120 s before the time limit: SIGTERM to the batch shell -> clean stop

set -uo pipefail  # no -e: the traps and the wait loop must keep running after a signal
RUN_DIR=${1:?usage: sbatch world.sbatch <run dir>}
# shellcheck disable=SC2034  # LOG_TAG is read by common.sh
LOG_TAG=world
# shellcheck source=common.sh
. "${RUN_DIR%/}/slurm/common.sh" || exit 2

# ---------------------------------------------------------------- signals (first thing: a reclaim during start-up must requeue too)
WORLD_PID= ; REQUEUE=0 ; STOPPING=0
WORLD_CTL=$RUN_DIR/world                      # the crash count and backoff across requeues
mkdir -p "$WORLD_CTL" || die "cannot create $WORLD_CTL"
stop_world() {
  STOPPING=1
  [ -n "$WORLD_PID" ] && kill -TERM "$WORLD_PID" 2>/dev/null
}
on_usr1() {  # Spot reclaim: the watcher cancels the job 30 s after this signal
  [ "$REQUEUE" = 1 ] && return 0
  REQUEUE=1
  log "SIGUSR1 (Spot reclaim): SIGTERM to the world (pid ${WORLD_PID:-<none>}), then requeue"
  stop_world
  for _ in $(seq 1 22); do
    { [ -n "$WORLD_PID" ] && kill -0 "$WORLD_PID" 2>/dev/null; } || break
    sleep 1
  done
  log "requeue: scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || log "scontrol requeue failed"
}
on_term() {  # clean stop: the world saves and exits; the job ends with it (no requeue)
  [ "$STOPPING" = 1 ] && return 0
  log "SIGTERM (clean stop): SIGTERM to the world; the job ends when it exits"
  stop_world
}
trap on_usr1 USR1
trap on_term TERM INT

# ---------------------------------------------------------------- serve
WORLD_PORT=$(free_port) || die "no free port on $(hostname)"   # the controller may share this node
WORLD_HOST=$(hostname)
[ "$STOPPING" = 0 ] || exit 0  # a signal during the start-up (already requeued on USR1)
log "world: $SPEC world.command on port $WORLD_PORT; log $RUN_DIR/logs/world-$RUN.log; restarts=${SLURM_RESTART_COUNT:-0}"
# Started while USR1 is ignored (see controller.sbatch); a flock world installs its own USR1 handler.
trap '' USR1
"$VENV/bin/python" -I - "$SPEC" "$WORLD_PORT" >> "$RUN_DIR/logs/world-$RUN.log" 2>&1 <<'PY' &
import os, sys
from flock.ops import read_spec
spec = read_spec(sys.argv[1])  # ${VAR}s from the job env; world.dir made absolute
w = spec.world
if w is None or not w.command:
    sys.exit(f"{sys.argv[1]}: no world.command to run")
argv = [x.replace("{port}", sys.argv[2]).replace("{python}", sys.executable) for x in w.command]
env = {k: v for k, v in os.environ.items()
       if not k.startswith("FLOCK_") or k in ("FLOCK_WORLD_TOKEN", "FLOCK_WORLD_ADMIN_TOKEN")}
env.update(FLOCK_WORLD_PORT=sys.argv[2], FLOCK_WORLD_HOST="0.0.0.0", FLOCK_SWARM=spec.swarm)
print(f"world: {' '.join(argv)} in {w.dir}", flush=True)
os.chdir(w.dir)
os.execvpe(argv[0], argv, env)
PY
WORLD_PID=$!
STARTED_AT=$(date +%s)
trap on_usr1 USR1

until [ "$(http_status "http://127.0.0.1:$WORLD_PORT/manifest" X-World-Token FLOCK_WORLD_TOKEN)" = 200 ]; do
  kill -0 "$WORLD_PID" 2>/dev/null || break
  [ "$STOPPING" = 1 ] && break
  sleep 2
done
if kill -0 "$WORLD_PID" 2>/dev/null && [ "$STOPPING" = 0 ]; then
  publish_host world "$WORLD_PORT"
  log "world answers /manifest at http://$WORLD_HOST:$WORLD_PORT (hosts/world.host)"
fi

# ---------------------------------------------------------------- wait (returns early on a signal: loop)
while kill -0 "$WORLD_PID" 2>/dev/null; do
  wait "$WORLD_PID"
done
wait "$WORLD_PID"; rc=$?   # the real status even when the last wait was cut by a trap
log "world exited with $rc (requeue=$REQUEUE, stopping=$STOPPING)"
[ "$REQUEUE" = 1 ] && exit 0  # already requeued by the Spot handler
if [ "$rc" = 0 ] || [ "$STOPPING" = 1 ]; then
  exit "$rc"
fi
# Crash requeues only: SLURM_RESTART_COUNT also counts Spot requeues, so keep our own count on the shared disk.
CRASHES_FILE=$WORLD_CTL/crash-requeues
BACKOFF_FILE=$WORLD_CTL/retry-backoff-s
WORLD_STABLE_S=${WORLD_STABLE_S:-600}
WORLD_RETRY_MIN_S=${WORLD_RETRY_MIN_S:-300}
WORLD_RETRY_MAX_S=${WORLD_RETRY_MAX_S:-1800}
crashes=$(cat "$CRASHES_FILE" 2>/dev/null); case $crashes in ""|*[!0-9]*) crashes=0;; esac
if [ $(( $(date +%s) - STARTED_AT )) -ge "$WORLD_STABLE_S" ]; then
  log "world ran >= ${WORLD_STABLE_S}s before failing (rc=$rc): crash count and backoff reset"
  rm -f "$BACKOFF_FILE"; echo 0 > "$CRASHES_FILE"
elif [ "$crashes" -lt 5 ]; then
  log "world failed fast (rc=$rc): automatic recovery requeue $(( crashes + 1 ))/5"
  echo $(( crashes + 1 )) > "$CRASHES_FILE"
  scontrol requeue "$SLURM_JOB_ID" || exit "$rc"
  exit 0
else
  delay=$(cat "$BACKOFF_FILE" 2>/dev/null); case $delay in ""|*[!0-9]*) delay=$WORLD_RETRY_MIN_S;; esac
  [ "$delay" -le "$WORLD_RETRY_MAX_S" ] || delay=$WORLD_RETRY_MAX_S
  log "world keeps failing fast (rc=$rc, $crashes crash requeues): retry in ${delay}s, never giving up"
  echo $(( delay * 2 > WORLD_RETRY_MAX_S ? WORLD_RETRY_MAX_S : delay * 2 )) > "$BACKOFF_FILE"
  sleep "$delay" & wait $!   # `wait` returns on a trap: a Spot reclaim or a clean stop cuts the backoff
  [ "$REQUEUE" = 1 ] && exit 0
  [ "$STOPPING" = 1 ] && exit "$rc"
  log "retry after backoff: scontrol requeue $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID" || exit "$rc"
  exit 0
fi
log "world failed (rc=$rc) after a stable run: automatic recovery requeue"
scontrol requeue "$SLURM_JOB_ID" || exit "$rc"
exit 0
