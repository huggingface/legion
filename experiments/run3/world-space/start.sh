# The world Space's entry point. Secrets: CHATBOT_WORLD_TOKEN (the world token, the same value
# the controller has), HF_TOKEN (bucket write, for the mirror and the workspace sync).
# Variables: CHATBOT_BUCKET (hf://buckets/<ns>/<bucket>/<run>; unset: no mirror, no sync),
# CHATBOT_WS (default /data/ws with persistent storage, else /tmp/ws), WS_SYNC_S (300).
set -euo pipefail
: "${CHATBOT_WORLD_TOKEN:?set the Space secret CHATBOT_WORLD_TOKEN}"
WS=${CHATBOT_WS:-$([ -w /data ] && echo /data/ws || echo /tmp/ws)}
export CHATBOT_WS=$WS FLOCK_WORLD_TOKEN=$CHATBOT_WORLD_TOKEN
mkdir -p "$WS"
B=${CHATBOT_BUCKET:-}; B=${B%/}
if [ -n "$B" ] && [ -z "$(ls -A "$WS")" ]; then  # a restarted Space: the workspace comes back
  hf buckets sync "$B/ws" "$WS" -q >/dev/null 2>&1 && echo "restored $WS from $B/ws" || true
fi
if [ -n "$B" ]; then  # the whole workspace every WS_SYNC_S (best/ is mirrored at each promotion)
  ( while sleep "${WS_SYNC_S:-300}"; do hf buckets sync "$WS" "$B/ws" -q >/dev/null 2>&1 || echo "ws sync failed"; done ) &
fi
cd /app/worlds/chatbot3
exec python -m flock.world serve chatbot3:Chatbot3World --host 0.0.0.0 --port 7860 --token-env FLOCK_WORLD_TOKEN
