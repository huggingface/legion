#!/bin/sh
# Start the canvas next to vLLM inside one HF Job (plain processes, no Space).
#   sh experiments/pixel/run_in_job.sh vllm serve Qwen/Qwen3-4B --port 8000
# Canvas runs in the background on $CANVAS_PORT (default 8765); the args run in the foreground.
# Secrets (CANVAS_TOKEN / CANVAS_ADMIN_TOKEN) come from job secrets, never from this file.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
python3 -m pip install -q "fastapi>=0.115" "uvicorn[standard]>=0.30" "numpy>=1.26" "pillow>=10"
export CANVAS_DATA="${CANVAS_DATA:-/tmp/canvas-data}" CANVAS_PORT="${CANVAS_PORT:-8765}"
(cd "$HERE" && exec python3 -m canvas) > "${CANVAS_DATA%/}.log" 2>&1 &
i=0
until python3 -c "import urllib.request,os;urllib.request.urlopen('http://127.0.0.1:'+os.environ['CANVAS_PORT']+'/healthz')" 2>/dev/null; do
  i=$((i + 1)); [ "$i" -gt 60 ] && { echo "canvas did not start; see ${CANVAS_DATA%/}.log"; exit 1; }
  sleep 1
done
echo "canvas up on :$CANVAS_PORT (replay at /replay)"
if [ "$#" -gt 0 ]; then exec "$@"; else wait; fi
