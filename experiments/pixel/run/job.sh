#!/usr/bin/env bash
# The ONE GPU job of a pixel run: vLLM serving LFM2.5-2.6B on :8000 and the canvas on :8765,
# same container. Run 1 uses the jobs proxy, not a network group (flock dropped provider
# network_group on purpose, 0c1b4b2 / design decision 27; flock-board/probe-reachability-lfm.md:
# `Authorization: Bearer $HF_TOKEN` through the proxy -> 200 for chat completions, same latency).
#   vLLM:   --expose 8000         token-protected by the proxy, so vLLM runs WITHOUT --api-key
#                                 (a keyed vLLM would need its key in Authorization too)
#   canvas: --expose-public 8765  open reads (replay, progress Space); writes need CANVAS_TOKEN
# URLs (verify at launch with `hf jobs inspect <id>`): https://<JOB_ID>--8000.hf.jobs/v1 and
# https://<JOB_ID>--8765.hf.jobs
#
#   RUN_ID=run1-20261019 experiments/pixel/run/job.sh      # prints the job id + exposed URLs
# Use the same RUN_ID for the driver's --run-id (progress Space) so the bucket dir
# ($BUCKET/pixel/runs/$RUN_ID) and the Space run match.
#
# Local env: HF_TOKEN, CANVAS_TOKEN (canvas shared secret). `--secrets NAME` forwards the local
# value; never printed.
# Optional: FLAVOR (rtx-pro-6000; the smoke used l4x1), TIMEOUT (90m), BUCKET (hf://buckets/cmpatino/flock), VLLM_IMAGE.
# Image: vllm/vllm-openai:v0.31.0 (the probe's image; it has the CUDA toolkit). The first smoke
# used the uv image + `pip install vllm` and died at startup: FlashInfer's sampler JIT needs nvcc
# ("Could not find nvcc"). VLLM_USE_FLASHINFER_SAMPLER=0 is set as well, belt and braces.
# The command below replaces the image's `vllm serve` entrypoint (as in the probe job).
# Code + target ship through the bucket (experiments/pixel synced to pixel/code/$RUN_ID).
# CANVAS_DATA lives on local disk and is copied to $BUCKET/pixel/runs/$RUN_ID every 60 s and on
# exit; a restarted job with the same RUN_ID resumes from that copy (the log is the truth).
set -euo pipefail
RUN_ID=${RUN_ID:?set RUN_ID, e.g. pixel-100-a}
BUCKET=${BUCKET:-hf://buckets/cmpatino/flock}
PIXEL=$(cd "$(dirname "$0")/.." && pwd)
: "${HF_TOKEN:?}" "${CANVAS_TOKEN:?}"

hf buckets sync "$PIXEL" "$BUCKET/pixel/code/$RUN_ID" -q \
  --exclude "tests/*" --exclude "*/__pycache__/*" --exclude "uv.lock" >/dev/null

read -r -d '' INNER <<'SH' || true
set -uo pipefail
# canvas deps on top of the vLLM image (pip skips what is already satisfied)
python3 -m pip install -q "fastapi>=0.115" "uvicorn[standard]>=0.30" "numpy>=1.26" "pillow>=10"
cp -r "/bucket/pixel/code/$RUN_ID" /app
DEST="/bucket/pixel/runs/$RUN_ID"; DATA=/data/canvas
mkdir -p "$DEST" "$DATA"
cp "$DEST"/paint.log.jsonl "$DEST"/claims.npz "$DEST"/agents.json "$DATA"/ 2>/dev/null || true
sync_out() {
  for f in paint.log.jsonl claims.npz agents.json canvas.png; do
    [ -f "$DATA/$f" ] && cp "$DATA/$f" "$DEST/$f.tmp" && mv "$DEST/$f.tmp" "$DEST/$f"
  done
  cp /data/*.log "$DEST/" 2>/dev/null || true
}
VLLM_PIDS=""
trap 'kill $CANVAS $VLLM_PIDS 2>/dev/null; sleep 3; sync_out; echo "SYNCED $DEST"' EXIT TERM INT
(cd /app && CANVAS_DATA=$DATA CANVAS_PORT=8765 exec python3 -m canvas) > /data/canvas.log 2>&1 &
CANVAS=$!
# ===================== BEGIN vLLM LAUNCH RECIPE (paste the probe's recipe here) ==============
# Contract: the OpenAI API answers on 127.0.0.1:8000 (`--expose 8000` proxies it; /health is
# polled below), logs go to /data/vllm*.log (synced to the bucket), and every server process
# that must stay up is in VLLM_PIDS (space-separated). Default = the smoke's single engine
# (baseline), so the script works unchanged if the probe says "baseline".
vllm serve LiquidAI/LFM2.5-2.6B --host 0.0.0.0 --port 8000 \
  --reasoning-parser qwen3 --tool-call-parser lfm2 --enable-auto-tool-choice \
  --max-model-len 8192 > /data/vllm.log 2>&1 &
VLLM_PIDS="$VLLM_PIDS $!"
# ===================== END vLLM LAUNCH RECIPE ================================================
alive() { for p in $VLLM_PIDS; do kill -0 "$p" 2>/dev/null || return 1; done; }
until python3 -c "import urllib.request as u;u.urlopen('http://127.0.0.1:8000/health')" 2>/dev/null; do
  alive || { echo "VLLM DIED"; tail -n 50 /data/vllm*.log; exit 1; }
  sleep 5
done
echo "VLLM READY"; tail -3 /data/canvas.log
while alive && kill -0 $CANVAS 2>/dev/null; do sleep 60; sync_out; done
echo "a process exited"; tail -n 20 /data/vllm*.log /data/canvas.log
SH

hf jobs run --detach --namespace cmpatino --flavor "${FLAVOR:-rtx-pro-6000}" --timeout "${TIMEOUT:-90m}" \
  --name "$RUN_ID" --label experiment=pixel \
  --expose 8000 --expose-public 8765 \
  --secrets HF_TOKEN --secrets CANVAS_TOKEN \
  -v "$BUCKET:/bucket" -e RUN_ID="$RUN_ID" -e VLLM_USE_FLASHINFER_SAMPLER=0 \
  "${VLLM_IMAGE:-vllm/vllm-openai:v0.31.0}" bash -c "$INNER"
