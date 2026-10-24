# run3 in ONE HF Job (README.md "Single job"): `flock run --until idle` in local mode inside the
# container (controller, workers, agents and the world are child processes), plus the wave
# runner. For smokes and up to ~50 agents; the 48 h run at scale is jobs mode (README.md).
# Passed inline: hf jobs run ... python:3.12-slim bash -c "$(cat job/bootstrap.sh)".
#
# env:     CHATBOT_BUCKET   hf://buckets/<ns>/<bucket>/<run> (payload/ in, everything else out)
#          CHATBOT_BOARD_URL, CHATBOT_CHANNEL, CHATBOT_RUN (swarm run3-<run>, optional)
#          OPENAI_BASE_URL  the vLLM server (/v1); MODEL_WORKER, MODEL_INTERVIEWER (optional)
#          RUN3_VARIANT     selfhosted (default) | router-smoke (the HF router, $ caps)
#          RUN3_WAVES       the wave schedule file in the experiment (default waves.yaml)
#          CHATBOT_SYNC_S   seconds between bucket syncs (default 300)
#          CHATBOT_CHECK_S  seconds between looks for control markers and due waves (default 30)
#          CHATBOT_FLOCK_ARGS  extra `flock run` flags (e.g. --skip-preflight)
# control: <bucket>/control/update   re-sync <bucket>/experiment/ over the folder, re-render,
#                                    keep the live replica count, `flock update` (marker deleted)
#          <bucket>/control/add      its content N: N more agents now (after the schedule: the
#                                    operator's call); deleted after, log in control/add-<k>.log
#          <bucket>/control/waves-off  the schedule stops (`add` and `update` still work)
#          <bucket>/control/stop     flock drains and writes its bundle, then the last sync
# secrets: HF_TOKEN (bucket; the router in router-smoke), CHATBOT_BOARD_INVITE,
#          CHATBOT_WORLD_TOKEN, OPENAI_API_KEY (optional: the vLLM server's key)
# A local rehearsal: CHATBOT_PREINSTALLED=1 (flock on PATH) and a local directory as the bucket.
set -euo pipefail
: "${CHATBOT_BUCKET:?}" "${CHATBOT_BOARD_URL:?}" "${CHATBOT_CHANNEL:?}" "${CHATBOT_WORLD_TOKEN:?}"
: "${CHATBOT_BOARD_INVITE:?}"
B=${CHATBOT_BUCKET%/}
H=${CHATBOT_HOME:-/opt/run3}
SYNC_S=${CHATBOT_SYNC_S:-300}
export PYTHON=${PYTHON:-python3} CHATBOT_WS=$H/ws CHATBOT_BUCKET=$B
export OPENAI_API_KEY=${OPENAI_API_KEY:-no-key}  # vLLM without --api-key takes any value
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
fetch() { if [[ $1 == hf://* ]]; then hf buckets cp "$1" "$2" -q >/dev/null; else cp "$1" "$2"; fi; }
del() { if [[ $1 == hf://* ]]; then hf buckets remove "$1" -y -q >/dev/null; else rm -f "$1"; fi; }

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
EXP=$H/repo/experiments/run3
export CHATBOT_CONFIG=$EXP/world.yaml  # the world's knobs (worlds/chatbot3)
WAVES=$EXP/${RUN3_WAVES:-waves.yaml}
render() {  # spec.local.yaml from spec.yaml and the env (endpoint, models, board, run name)
  "$PYTHON" "$EXP/render.py" --variant "${RUN3_VARIANT:-selfhosted}" --out "$EXP/spec.local.yaml" > /dev/null
  if [ -n "${CHATBOT_RUN:-}" ]; then
    sed -i "s|^Kickoff (run3)|Kickoff (run3 $CHATBOT_RUN)|" "$EXP/prompts/kickoff.md"
  fi
}
waves() { (cd "$EXP" && "$PYTHON" waves.py "$@" --spec spec.local.yaml --waves "$WAVES" --log "$H/logs/waves.jsonl"); }
render
log "rendered ${RUN3_VARIANT:-selfhosted}; waves $(basename "$WAVES"); installed $(flock --help >/dev/null && echo flock) $($PYTHON --version)"

PIDS=()
cd "$EXP"
flock run spec.local.yaml --until idle ${CHATBOT_FLOCK_ARGS:-} > "$H/logs/run.out" 2>&1 &
PIDS+=($!)
log "flock run started (pids ${PIDS[*]})"

sync_out() {
  if [ -d "$CHATBOT_WS" ]; then push "$CHATBOT_WS" "$B/ws" || log "sync ws failed"; fi
  [ -d "$EXP/.flock" ] && { push "$EXP/.flock" "$B/flock/.flock" || log "sync .flock failed"; }
  push "$H/logs" "$B/logs" || log "sync logs failed"
}
report() {  # flock's start lines and its latest status line, into the job log
  local o=$H/logs/run.out n s
  [ -f "$o" ] || return 0
  n=$(wc -l < "$o"); s=${seen:-0}
  [ "$n" -gt "$s" ] && tail -n +"$((s + 1))" "$o" | head -n "$((n - s))" | grep -aE \
    '^(preflight|tools:|apply|prompt to|flock run|stopping|run |bundle:|update:)' | sed "s|^|run.out: |" || true
  seen=$n
  tail -n 30 "$o" | grep -a '^\[' | tail -1 | sed "s|^|run.out: |" || true
}
resync() {  # <bucket>/experiment/ (the operator's edited folder), if any, over the experiment
  rm -rf "$H/overlay" && mkdir -p "$H/overlay"
  pull "$B/experiment" "$H/overlay" 2>/dev/null || true
  [ -n "$(ls -A "$H/overlay")" ] || return 0
  cp -a "$H/overlay/." "$EXP/" && render && log "re-synced $(find "$H/overlay" -type f | wc -l) files from experiment/"
}
UPDATES=0
live_update() {  # control/update: re-sync, keep the live replicas, then `flock update`
  local out rc=0
  UPDATES=$((UPDATES + 1)); out="$H/logs/update-$UPDATES.log"
  resync || log "re-sync failed"
  { waves pin && (cd "$EXP" && flock update spec.local.yaml); } > "$out" 2>&1 || rc=$?
  if [ "$rc" = 0 ]; then log "update $UPDATES ok:"; else log "update $UPDATES FAILED ($rc):"; fi
  head -40 "$out" | sed 's|^|  |' | tee -a "$H/logs/bootstrap.log"
  put "$out" "$B/control/update-$UPDATES.log" || true
  del "$B/control/update" || log "could not delete control/update"
}
ADDS=0
add_now() {  # control/add: N more agents now
  local n out rc=0
  ADDS=$((ADDS + 1)); out="$H/logs/add-$ADDS.log"
  fetch "$B/control/add" "$H/add.n" && n=$(tr -dc '0-9' < "$H/add.n")
  if [ -n "${n:-}" ]; then waves add "$n" > "$out" 2>&1 || rc=$?; else echo "control/add: no number" > "$out"; rc=1; fi
  log "add $ADDS ($n): $(tail -1 "$out")"
  put "$out" "$B/control/add-$ADDS.log" || true
  del "$B/control/add" || log "could not delete control/add"
}
STOP=0
on_term() { log "stop requested: stopping flock (drain, bundle)"; STOP=1; kill -TERM "${PIDS[@]}" 2>/dev/null || true; }
trap on_term TERM INT
alive() { for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
last_sync=$(date +%s); last_check=0; waves_on=1
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
    elif exists "$B/control/update"; then
      live_update
    elif exists "$B/control/add"; then
      add_now
    else
      if [ "$waves_on" = 1 ] && exists "$B/control/waves-off"; then waves_on=0; log "waves off (control/waves-off)"; fi
      if [ "$waves_on" = 1 ]; then  # a due wave raises replicas (nothing when none is due)
        if out=$(waves tick --start "$t0" 2>&1); then [ -n "$out" ] && log "$out"
        else log "wave tick failed: $(echo "$out" | tail -1)"; fi
      fi
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
