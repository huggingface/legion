# chatbot-v1 in ONE HF Job (README.md "Run it as an HF Job"): `flock run --until idle` in local
# mode inside the container (controller, workers, agents and the world are child processes; no
# proxy, no network group). Passed inline: hf jobs run ... python:3.12-slim bash -c "$(cat job/bootstrap.sh)".
#
# env:     CHATBOT_BUCKET   hf://buckets/<ns>/<bucket>/<run> (payload/ in, everything else out)
#          CHATBOT_BOARD_URL, CHATBOT_CHANNEL (+ CHATBOT_CHANNEL_B for the two-channel variant)
#          CHATBOT_VARIANT  one (default) | two-channels
#          CHATBOT_SYNC_S   seconds between bucket syncs (default 300)
#          CHATBOT_CHECK_S  seconds between looks for the control marker (default 30)
#          CHATBOT_FLOCK_ARGS  extra `flock run` flags (e.g. --skip-preflight)
#          CHATBOT_SWITCH_AT  optional: seconds after start for the phase switch (else: the
#                           control marker <bucket>/control/switch, uploaded by the operator)
# secrets: HF_TOKEN (router + bucket), CHATBOT_BOARD_INVITE, CHATBOT_WORLD_TOKEN
# `hf jobs cancel` (SIGTERM): stop flock (drain, bundle), then the last sync. Exit: flock's code.
# A local rehearsal: CHATBOT_PREINSTALLED=1 (flock on PATH) and a local directory as the bucket.
set -euo pipefail
: "${CHATBOT_BUCKET:?}" "${CHATBOT_BOARD_URL:?}" "${CHATBOT_CHANNEL:?}" "${CHATBOT_WORLD_TOKEN:?}"
: "${CHATBOT_BOARD_INVITE:?}"
B=${CHATBOT_BUCKET%/}
H=${CHATBOT_HOME:-/opt/chatbot}
VARIANT=${CHATBOT_VARIANT:-one}
SYNC_S=${CHATBOT_SYNC_S:-300}
export PYTHON=${PYTHON:-python3} CHATBOT_VARIANT=$VARIANT CHATBOT_WS=$H/ws CHATBOT_BUCKET=$B
t0=$(date +%s)
mkdir -p "$H/logs"
log() { echo "bootstrap $(($(date +%s) - t0))s: $*"; }

remote() { [[ $1 == hf://* || $2 == hf://* ]]; }
pull() { if remote "$1" "$2"; then hf buckets sync "$1" "$2" -q >/dev/null; else mkdir -p "$2"; cp -a "$1/." "$2/"; fi; }
push() {  # a directory, without flock's `current` file (it holds the controller token)
  if remote "$1" "$2"; then hf buckets sync "$1" "$2" --exclude current -q >/dev/null
  else mkdir -p "$2"; cp -a "$1/." "$2/"; find "$2" -name current -delete; fi
}
put() { if remote "$1" "$2"; then hf buckets cp "$1" "$2" -q >/dev/null; else mkdir -p "$(dirname "$2")"; cp "$1" "$2"; fi; }
exists() { if [[ $1 == hf://* ]]; then hf buckets cp "$1" "$H/.probe" -q >/dev/null 2>&1; else [ -e "$1" ]; fi; }

if [ -z "${CHATBOT_PREINSTALLED:-}" ]; then
  pip() { python3 -m pip install -q --no-cache-dir --root-user-action=ignore "$@"; }
  pip "huggingface_hub==2.2.0"
fi
pull "$B/payload" "$H/payload"
if [ -z "${CHATBOT_PREINSTALLED:-}" ]; then
  W=$(ls "$H"/payload/flock-*.whl)
  pip "$W[controller]" "huggingface_hub==2.2.0"
fi
mkdir -p "$H/repo" && tar -xzf "$H/payload/repo.tar.gz" -C "$H/repo"
EXP=$H/repo/experiments/chatbot-v1
for f in spec.yaml phase2/spec.yaml variant-two-channels/spec-a.yaml variant-two-channels/spec-b.yaml \
         variant-two-channels/phase2/spec-a.yaml variant-two-channels/phase2/spec-b.yaml; do
  sed -e "s|https://msgboard.example.invalid|$CHATBOT_BOARD_URL|" \
      -e "s|your-team-b-channel|${CHATBOT_CHANNEL_B:-your-team-b-channel}|" \
      -e "s|your-chatbot-channel|$CHATBOT_CHANNEL|" "$EXP/$f" > "$EXP/${f%.yaml}.local.yaml"
done
log "installed $(flock --help >/dev/null && echo flock) $($PYTHON --version); variant $VARIANT"

PIDS=()
RUNS=()
if [ "$VARIANT" = two-channels ]; then
  (cd "$H/repo/worlds/chatbot" && FLOCK_WORLD_TOKEN=$CHATBOT_WORLD_TOKEN FLOCK_WORLD_PORT=8790 \
     exec "$PYTHON" chatbot.py serve) > "$H/logs/world.log" 2>&1 &
  WORLD=$!
  for i in $(seq 30); do [ -s "$CHATBOT_WS/world.json" ] && "$PYTHON" "$H/repo/worlds/chatbot/chatbot.py" status >/dev/null 2>&1 && break; sleep 1; done
  cd "$EXP/variant-two-channels"
  for t in a b; do
    flock run "spec-$t.local.yaml" --until idle --data-dir ".flock-$t" ${CHATBOT_FLOCK_ARGS:-} > "$H/logs/run-$t.out" 2>&1 &
    PIDS+=($!); RUNS+=("$EXP/variant-two-channels/.flock-$t")
  done
else
  cd "$EXP"
  flock run spec.local.yaml --until idle ${CHATBOT_FLOCK_ARGS:-} > "$H/logs/run.out" 2>&1 &
  PIDS+=($!); RUNS+=("$EXP/.flock")
fi
log "flock run started (pids ${PIDS[*]})"

sync_out() {
  [ -d "$CHATBOT_WS" ] && push "$CHATBOT_WS" "$B/ws" || log "sync ws failed"
  for d in "${RUNS[@]}"; do [ -d "$d" ] && { push "$d" "$B/flock/$(basename "$d")" || log "sync $d failed"; }; done
  push "$H/logs" "$B/logs" || log "sync logs failed"
}
STOP=0
on_term() { log "stop requested: stopping flock (drain, bundle)"; STOP=1; kill -TERM "${PIDS[@]}" 2>/dev/null || true; }
trap on_term TERM INT
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
failed() {  # the two-channel variant: one swarm that failed (e.g. its apply) stops the other
  for p in "${PIDS[@]}"; do if ! kill -0 "$p" 2>/dev/null; then wait "$p" || return 0; fi; done; return 1; }
last_sync=$(date +%s); last_check=0; switched=0
while alive; do
  sleep "${CHATBOT_TICK_S:-5}" & wait $! || true
  now=$(date +%s)
  if [ "$STOP" = 0 ] && [ "${#PIDS[@]}" -gt 1 ] && failed; then log "a flock run failed"; on_term; fi
  if [ "$STOP" = 1 ] && [ -z "${stop_synced:-}" ]; then  # the workspace first, while flock drains
    stop_synced=1; push "$CHATBOT_WS" "$B/ws" || log "sync ws failed"
  fi
  if [ "$switched" = 0 ] && [ "$STOP" = 0 ] && [ $((now - last_check)) -ge "${CHATBOT_CHECK_S:-30}" ]; then
    last_check=$now
    if { [ -n "${CHATBOT_SWITCH_AT:-}" ] && [ $((now - t0)) -ge "$CHATBOT_SWITCH_AT" ]; } || exists "$B/control/switch"; then
      log "phase switch"
      if bash "$EXP/job/switch.sh" > "$H/logs/switch.log" 2>&1; then log "phase switch ok"; else log "phase switch FAILED (logs/switch.log)"; fi
      switched=1
      put "$H/logs/switch.log" "$B/control/switch.log" || true
    fi
  fi
  if [ "$STOP" = 0 ] && [ $((now - last_sync)) -ge "$SYNC_S" ]; then sync_out; last_sync=$now; fi
done
code=0
for p in "${PIDS[@]}"; do wait "$p" || code=$?; done
[ -n "${WORLD:-}" ] && kill -TERM "$WORLD" 2>/dev/null || true
log "flock exited with $code; final sync"
sync_out
log "done"
exit "$code"
