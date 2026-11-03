#!/bin/bash
# The world process itself, run by world.sbatch on the node (bare venv) or inside the
# container (srun --container-image; then /work is the container's root filesystem, never /tmp).
# Restores the workspace from <CHATBOT_BUCKET>/ws into a fresh CHATBOT_WS, serves
# worlds/chatbot3, pushes the workspace every WS_SYNC_S, and on TERM/USR1 (once) stops the
# server and makes ONE push bounded by WS_STOP_S (default 15 s: the Spot budget is ~25 s).
# Env: CHATBOT_WS, CHATBOT_BUCKET, CHATBOT_CONFIG, FLOCK_WORLD_TOKEN, HF_TOKEN, REPO, PY.
set -euo pipefail
: "${CHATBOT_WS:?}" "${CHATBOT_BUCKET:?}" "${CHATBOT_CONFIG:?}" "${REPO:?}" "${PY:?}"
B=${CHATBOT_BUCKET%/}
log() { echo "$(date -u +%FT%TZ) world: $*"; }
HF=$(dirname "$PY")/hf
rm -rf "$CHATBOT_WS"; mkdir -p "$CHATBOT_WS" "${TMPDIR:-/tmp}"
if "$HF" buckets sync "$B/ws" "$CHATBOT_WS" -q >/dev/null 2>&1; then log "restored ws from $B/ws"
else log "no ws in $B/ws yet (a new run)"; fi
cd "$REPO/worlds/chatbot3"
"$PY" -m flock.world serve chatbot3:Chatbot3World --host 0.0.0.0 --port 7860 \
  --token-env FLOCK_WORLD_TOKEN &
PID=$!
( while sleep "${WS_SYNC_S:-300}"; do
    "$HF" buckets sync "$CHATBOT_WS" "$B/ws" -q >/dev/null 2>&1 || log "ws sync failed"
  done ) &
SYNC=$!
STOPPING=0
stop() {
  [ "$STOPPING" = 0 ] || return 0; STOPPING=1
  log "stop: server down, last ws push (<= ${WS_STOP_S:-15} s)"
  kill "$SYNC" 2>/dev/null || true
  kill -TERM "$PID" 2>/dev/null || true
}
trap stop TERM USR1 INT
rc=0
while kill -0 "$PID" 2>/dev/null; do wait "$PID" || rc=$?; done
kill "$SYNC" 2>/dev/null || true
if [ "$STOPPING" = 1 ]; then
  timeout "${WS_STOP_S:-15}" "$HF" buckets sync "$CHATBOT_WS" "$B/ws" -q >/dev/null 2>&1 \
    && log "ws pushed" || log "ws push cut at ${WS_STOP_S:-15} s"
fi
log "exited $rc"
exit "$rc"
