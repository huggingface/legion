#!/bin/bash
# world-inner.sh: the world's driver INSIDE its container (started by world.sbatch through
# `srun --container-image ... bash world-inner.sh`). Never run it on the host.
#
# restore world.tar.gz from the bucket into an EMPTY $CHATBOT_WS -> `flock world serve` -> push ws/ every
# WS_SYNC_S -> final sync when the server ends. The workspace lives on the container root (/work,
# node disk, deleted with the container), so every archive upload runs here, not on the host.
#
# Input: the batch script's environment (secrets included: srun passes it into the container):
# VENV B CHATBOT_WS WORLD_SRC WORLD_PORT WORLD_CTL WS_SYNC_S FINAL_SYNC_S FINAL_SYNC_LONG_S
# CANCEL_SYNC_S FLOCK_WORLD_TOKEN HF_TOKEN CHATBOT_REPO_TOKEN.
# Output: the server's stdout/stderr on this script's stdout (world.sbatch appends it to
# <run>/logs/world-<RUN>.log); this script's own lines on stderr (the job's .out).
#
# Signals (every process of the step gets them directly from slurmd, and srun forwards what it
# receives, so each may arrive 2-3 times: both handlers are idempotent):
# - USR1 (Spot reclaim; the watcher cancels the job 30 s later): stop the server (5 s, then
#   SIGKILL), one final sync bounded by FINAL_SYNC_S (12 s), exit. world.sbatch then requeues.
#   Worst case 5 + 12 + 2 (timeout -k) = 19 s.
# - TERM / INT (plain scancel, requeue, preemption) and USR2 (world.sbatch's clean-stop forward:
#   `scancel --signal=TERM --batch`, or the time limit): stop the server (no SIGKILL: the main
#   loop waits for it), then a final sync bounded by $WORLD_CTL/final-sync-bound, which
#   world.sbatch writes before it forwards (CANCEL_SYNC_S when the file is missing).
# - The server, the restore and the sync loop start with USR1 ignored (SIG_IGN survives exec), so
#   the direct USR1 does not kill them; this script stops them itself.

set -uo pipefail
log() { printf '%s world-inner[%s]: %s\n' "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-nojob}" "$*" >&2; }
for v in VENV B CHATBOT_WS WORLD_SRC WORLD_PORT WORLD_CTL WS_SYNC_S FINAL_SYNC_S FINAL_SYNC_LONG_S CANCEL_SYNC_S; do
  [ -n "${!v:-}" ] || { log "FATAL: $v is not set (world.sbatch exports it)"; exit 2; }
done
case $CHATBOT_WS in /work/*) ;; *) log "FATAL: CHATBOT_WS=$CHATBOT_WS is not under /work"; exit 2;; esac

# No home, no /fsx writes: every cache and temp dir on the container root.
export HOME=/work/home HF_HOME=/work/hf TMPDIR=/work/tmp
export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"

WORLD_PID= ; SYNC_PID= ; RESTORE_PID= ; MODE=     # MODE: usr1 | stop (TERM/INT/USR2)
final_sync() {  # final_sync BOUND_S: one bounded push of ws/ (skipped when the server never started)
  local bound=$1
  if [ -z "$WORLD_PID" ]; then
    log "no final ws sync: the server never started (ws/ is empty or a partial restore)"
    return 0
  fi
  log "final ws sync -> $B/world.tar.gz (bounded ${bound}s)"
  if timeout -k 2 "$bound" "$VENV/bin/python" "$WORLD_SRC/snapshot.py" push "$CHATBOT_WS" "$B" >/dev/null 2>&1; then
    log "final ws sync ok"
  else
    log "final ws sync FAILED or timed out (up to ${WS_SYNC_S}s of PR state may be lost; merged code remains on the Hub)"
  fi
}
stop_world() {  # stop_world [kill]: TERM the restore, the sync loop and the server; 5 s; SIGKILL if asked
  [ -n "$RESTORE_PID" ] && kill -TERM "$RESTORE_PID" 2>/dev/null
  [ -n "$SYNC_PID" ] && { kill -TERM "$SYNC_PID" 2>/dev/null; pkill -TERM -P "$SYNC_PID" 2>/dev/null; }
  [ -n "$WORLD_PID" ] && kill -TERM "$WORLD_PID" 2>/dev/null
  for _ in 1 2 3 4 5; do
    { [ -n "$WORLD_PID" ] && kill -0 "$WORLD_PID" 2>/dev/null; } || break
    sleep 1
  done
  # uvicorn's graceful shutdown can wait indefinitely on an in-flight promotion
  if [ "${1:-}" = kill ] && [ -n "$WORLD_PID" ] && kill -0 "$WORLD_PID" 2>/dev/null; then
    log "flock world serve still alive after 5 s: SIGKILL"
    kill -KILL "$WORLD_PID" 2>/dev/null
  fi
}
on_usr1() {
  [ "$MODE" = usr1 ] && return 0          # the 2nd/3rd copy of the same reclaim
  MODE=usr1
  log "SIGUSR1 (Spot reclaim): stop the server, final sync (${FINAL_SYNC_S}s), exit"
  stop_world kill
  final_sync "$FINAL_SYNC_S"
  log "exit after SIGUSR1"
  exit 0
}
on_stop() {  # $1: the signal name
  [ -n "$MODE" ] && return 0
  MODE=stop
  log "SIG$1 (clean stop): stopping the world; the final sync runs after it exits"
  stop_world
}
trap on_usr1 USR1
trap 'on_stop TERM' TERM
trap 'on_stop INT' INT
trap 'on_stop USR2' USR2

wait_pid() {  # wait_pid PID: wait until PID exits (wait returns early on a trap: loop); sets WRC
  while kill -0 "$1" 2>/dev/null; do wait "$1" 2>/dev/null; done
  wait "$1" 2>/dev/null; WRC=$?   # bash keeps the status of a reaped child
}

# ---------------------------------------------------------------- restore into an EMPTY ws (start.sh)
rm -rf /work/world && mkdir -p "$CHATBOT_WS" "$HOME" "$HF_HOME" "$TMPDIR" || { log "FATAL: cannot create /work dirs"; exit 2; }
trap '' USR1
"$VENV/bin/python" "$WORLD_SRC/snapshot.py" restore "$CHATBOT_WS" "$B" >/dev/null 2>&1 &
RESTORE_PID=$!
trap on_usr1 USR1
wait_pid "$RESTORE_PID"; rrc=$WRC; RESTORE_PID=
[ -z "$MODE" ] || { log "signal during the restore: exit, nothing to sync"; exit 0; }
if [ "$rrc" = 0 ]; then
  log "restored $CHATBOT_WS from $B/world.tar.gz ($(find "$CHATBOT_WS" -type f | wc -l) files)"
elif [ "$rrc" = 3 ]; then
  log "no archive yet at $B/world.tar.gz (first start)"
else
  log "FATAL: world state restore failed; refusing to start with empty PR/review state"
  exit 2
fi

# ---------------------------------------------------------------- serve + periodic push
cd "$WORLD_SRC" || { log "FATAL: cannot cd $WORLD_SRC"; exit 2; }   # `module:Class` imports from the cwd
trap '' USR1
"$VENV/bin/flock" world serve chatbot4:Chatbot4World --host 0.0.0.0 --port "$WORLD_PORT" \
  --token-env FLOCK_WORLD_TOKEN 2>&1 &
WORLD_PID=$!
(
  trap 'exit 0' TERM
  while sleep "$WS_SYNC_S" & wait $!; do
    "$VENV/bin/python" "$WORLD_SRC/snapshot.py" push "$CHATBOT_WS" "$B" >/dev/null 2>&1 || log "ws sync failed"
  done
) &
SYNC_PID=$!
trap on_usr1 USR1
log "flock world serve pid $WORLD_PID, ws sync loop pid $SYNC_PID (every ${WS_SYNC_S}s), ws=$CHATBOT_WS"
[ -z "$MODE" ] || stop_world   # a stop signal between the restore and here

wait_pid "$WORLD_PID"; rc=$WRC
[ -n "$SYNC_PID" ] && { kill -TERM "$SYNC_PID" 2>/dev/null; pkill -TERM -P "$SYNC_PID" 2>/dev/null; wait "$SYNC_PID" 2>/dev/null; }
log "flock world serve exited with $rc (mode=${MODE:-none})"

# ---------------------------------------------------------------- final sync (USR1 never gets here: on_usr1 exits)
if [ "$MODE" = stop ]; then          # bound from world.sbatch (signal-only: long; plain scancel: 25 s)
  bound=$(cat "$WORLD_CTL/final-sync-bound" 2>/dev/null)
  case $bound in ''|*[!0-9]*) bound=$CANCEL_SYNC_S;; esac
else                                 # crash: not under a kill; cap by the job's end time when known
  bound=$FINAL_SYNC_LONG_S
  if [ -n "${SLURM_JOB_END_TIME:-}" ]; then
    left=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    [ $(( left - 10 )) -lt "$bound" ] && bound=$(( left > 20 ? left - 10 : 10 ))
  fi
fi
final_sync "$bound"
exit "$rc"
