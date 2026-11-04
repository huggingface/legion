# chatbot-v2 in ONE HF Job (README.md "Run it as an HF Job"): `flock run --until idle` in local
# mode inside the container (controller, workers, agents and the world are child processes; no
# proxy, no network group). Passed inline: hf jobs run ... python:3.12-slim bash -c "$(cat job/bootstrap.sh)".
#
# env:     CHATBOT_BUCKET   hf://buckets/<ns>/<bucket>/<run> (payload/ in, everything else out)
#          CHATBOT_BOARD_URL, CHATBOT_CHANNEL
#          CHATBOT_SYNC_S   seconds between bucket syncs (default 300)
#          CHATBOT_CHECK_S  seconds between looks for the control marker (default 30)
#          CHATBOT_FLOCK_ARGS  extra `flock run` flags (e.g. --skip-preflight)
#          CHATBOT_SMOKE=1  a short run: job/smoke.json's loop and budget on every rendered spec
#          CHATBOT_RUN      e.g. run3: the swarm is chatbot-v2-run3 (a fresh kickoff on a reused
#                           channel) and the kickoff says "chatbot-v2 run3"
# control: <bucket>/control/switch (phase 2), control/update (a live `flock update` with the
#          folder uploaded to <bucket>/experiment/ first; the marker is deleted after; the world
#          re-reads world.yaml from it too), control/stop
#          CHATBOT_SWITCH_AT  optional: seconds after start for the phase switch (else: the
#                           control marker <bucket>/control/switch, uploaded by the operator)
# secrets: HF_TOKEN (router + bucket), CHATBOT_BOARD_INVITE, CHATBOT_WORLD_TOKEN
# Stop: upload <bucket>/control/stop: flock drains and writes its bundle, then the last sync.
# `hf jobs cancel` (SIGTERM) tries the same, but the platform killed the smoke job ~3 s after
# the cancel: only what the periodic sync already pushed survives. Exit: flock's code.
# A local rehearsal: CHATBOT_PREINSTALLED=1 (flock on PATH) and a local directory as the bucket.
set -euo pipefail
: "${CHATBOT_BUCKET:?}" "${CHATBOT_BOARD_URL:?}" "${CHATBOT_CHANNEL:?}" "${CHATBOT_WORLD_TOKEN:?}"
: "${CHATBOT_BOARD_INVITE:?}"
B=${CHATBOT_BUCKET%/}
H=${CHATBOT_HOME:-/opt/chatbot}
SYNC_S=${CHATBOT_SYNC_S:-300}
export PYTHON=${PYTHON:-python3} CHATBOT_WS=$H/ws CHATBOT_BUCKET=$B
t0=$(date +%s)
mkdir -p "$H/logs"
log() { echo "bootstrap $(date -u +%H:%M:%S) +$(($(date +%s) - t0))s: $*" | tee -a "$H/logs/bootstrap.log"; }

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
EXP=$H/repo/experiments/chatbot-v2; if [ -n "${CHATBOT_VARIANT:-}" ]; then "$PYTHON" "$EXP/$CHATBOT_VARIANT/render.py" --into "$EXP" > "$H/logs/variant.log" 2>&1 || { cat "$H/logs/variant.log"; exit 1; }; fi  # CHATBOT_VARIANT=selfhosted: selfhosted/README.md
export CHATBOT_CONFIG=$EXP/world.yaml  # the world's knobs (worlds/chatbot2)
render() {  # the *.local.yaml specs: board, channels, run name (and job/smoke.json in a smoke)
  local f
  for f in spec.yaml phase2/spec.yaml; do
    sed -e "s|https://msgboard.example.invalid|$CHATBOT_BOARD_URL|" \
        -e "s|your-chatbot-channel|$CHATBOT_CHANNEL|" \
        -e "s|^swarm: chatbot-v2|swarm: chatbot-v2${CHATBOT_RUN:+-$CHATBOT_RUN}|" \
        "$EXP/$f" > "$EXP/${f%.yaml}.local.yaml"
  done
  if [ -n "${CHATBOT_RUN:-}" ]; then
    sed -i "s|^Kickoff (chatbot-v2)|Kickoff (chatbot-v2 $CHATBOT_RUN)|" "$EXP/prompts/kickoff.md"
  fi
  # job/render.py: the smoke settings (CHATBOT_SMOKE=1) and the kickoff's real duration
  SMOKE_SETTINGS=$("$PYTHON" "$EXP/job/render.py" "$EXP" "$@")
}
render --first
[ -n "${CHATBOT_SMOKE:-}" ] && log "smoke settings: $SMOKE_SETTINGS"
log "installed $(flock --help >/dev/null && echo flock) $($PYTHON --version)"

PIDS=()
RUNS=()
cd "$EXP"
flock run spec.local.yaml --until idle ${CHATBOT_FLOCK_ARGS:-} > "$H/logs/run.out" 2>&1 &
PIDS+=($!); RUNS+=("$EXP/.flock")
log "flock run started (pids ${PIDS[*]})"

sync_out() {
  if [ -d "$CHATBOT_WS" ]; then push "$CHATBOT_WS" "$B/ws" || log "sync ws failed"; fi
  for d in "${RUNS[@]}"; do [ -d "$d" ] && { push "$d" "$B/flock/$(basename "$d")" || log "sync $d failed"; }; done
  push "$H/logs" "$B/logs" || log "sync logs failed"
}
report() {  # flock's start lines and its latest status line, into the job log
  for o in "$H"/logs/run*.out; do
    [ -f "$o" ] || continue
    local n; n=$(wc -l < "$o"); local k="seen_${o//[^a-z]/_}"; local s=${!k:-0}
    [ "$n" -gt "$s" ] && tail -n +"$((s + 1))" "$o" | head -n "$((n - s))" | grep -aE \
      '^(preflight|tools:|apply|prompt to|flock run|stopping|run |bundle:|update:)' | sed "s|^|$(basename "$o"): |" || true
    printf -v "$k" '%s' "$n"
    tail -n 30 "$o" | grep -a '^\[' | tail -1 | sed "s|^|$(basename "$o"): |" || true
  done
}
del() { if [[ $1 == hf://* ]]; then hf buckets remove "$1" -y -q >/dev/null; else rm -f "$1"; fi; }
resync() {  # <bucket>/experiment/ (the operator's edited folder), if any, over the experiment
  rm -rf "$H/overlay" && mkdir -p "$H/overlay"
  pull "$B/experiment" "$H/overlay" 2>/dev/null || true
  [ -n "$(ls -A "$H/overlay")" ] || return 0
  cp -a "$H/overlay/." "$EXP/" && render && log "re-synced $(find "$H/overlay" -type f | wc -l) files from experiment/"
}
UPDATES=0
live_update() {  # control/update: re-sync, then `flock update` on the specs of the current phase
  local p="" out t rc=0; [ "$switched" = 1 ] && p="phase2/"
  UPDATES=$((UPDATES + 1)); out="$H/logs/update-$UPDATES.log"
  resync || log "re-sync failed"
  (cd "$EXP" && flock update "${p}spec.local.yaml") > "$out" 2>&1 || rc=$?
  put "$out" "$B/control/update-$UPDATES.log" || true  # both before the result line below
  del "$B/control/update" || log "could not delete control/update"
  if [ "$rc" = 0 ]; then log "update $UPDATES ok:"; else log "update $UPDATES FAILED ($rc):"; fi
  head -40 "$out" | sed 's|^|  |' | tee -a "$H/logs/bootstrap.log"
}
STOP=0
on_term() { log "stop requested: stopping flock (drain, bundle)"; STOP=1; kill -TERM "${PIDS[@]}" 2>/dev/null || true; }
trap on_term TERM INT
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
last_sync=$(date +%s); last_check=0; switched=0
while alive; do
  sleep "${CHATBOT_TICK_S:-5}" & wait $! || true
  now=$(date +%s)
  if [ "$STOP" = 1 ] && [ -z "${stop_synced:-}" ]; then  # the workspace first, while flock drains
    stop_synced=1; if [ -d "$CHATBOT_WS" ]; then push "$CHATBOT_WS" "$B/ws" || log "sync ws failed"; fi
  fi
  if [ "$STOP" = 0 ] && [ $((now - last_check)) -ge "${CHATBOT_CHECK_S:-30}" ]; then
    last_check=$now
    if exists "$B/control/stop"; then  # a graceful stop: `hf jobs cancel` leaves only ~3 s
      log "stop marker found"; on_term
    elif [ "$switched" = 0 ] && { { [ -n "${CHATBOT_SWITCH_AT:-}" ] &&
         [ $((now - t0)) -ge "$CHATBOT_SWITCH_AT" ]; } || exists "$B/control/switch"; }; then
      log "phase switch"
      resync || log "re-sync failed"  # phase-2 prompts edited in <bucket>/experiment/ count
      src=0; bash "$EXP/job/switch.sh" > "$H/logs/switch.log" 2>&1 || src=$?
      switched=1
      put "$H/logs/switch.log" "$B/control/switch.log" || true  # before the log line
      if [ "$src" = 0 ]; then log "phase switch ok"; else log "phase switch FAILED (logs/switch.log)"; fi
    elif exists "$B/control/update"; then
      live_update
    fi
  fi
  if [ "$STOP" = 0 ] && [ $((now - last_sync)) -ge "$SYNC_S" ]; then sync_out; last_sync=$now; fi
  if [ $((now - ${last_report:-0})) -ge "${CHATBOT_REPORT_S:-60}" ]; then report; last_report=$now; fi
done
code=0
for p in "${PIDS[@]}"; do wait "$p" || code=$?; done
report
log "flock exited with $code; final sync"
sync_out
log "done"
exit "$code"
