#!/usr/bin/env bash
# The 1k run's GPUs: three rtx-pro-6000 jobs, same recipe as run 1 (job.sh: vLLM image v0.31.0,
# one engine with --api-server-count 4 --max-num-seqs 512). gpu1 also runs the canvas
# (--expose-public 8765) with run-1 settings (MAX_PIXELS_PER_CALL=10, scattered claims, white
# blank). All three share RUN_ID and stop after --timeout 75m at the latest.
#
#   RUN_ID=run1k-20261020 experiments/pixel/run/job-1k.sh
#
# Prints the three job ids and the URLs to paste into spec-1k.yaml (gpu1..gpu3 base_url) and
# CANVAS_URL. Local env as for job.sh: HF_TOKEN, CANVAS_TOKEN (never printed).
set -euo pipefail
RUN_ID=${RUN_ID:?set RUN_ID, e.g. run1k-20261020}
HERE=$(cd "$(dirname "$0")" && pwd)
export RUN_ID TIMEOUT=${TIMEOUT:-75m} FLAVOR=${FLAVOR:-rtx-pro-6000}
IDS=()
for n in 1 2 3; do
  out=$(JOB_NAME="$RUN_ID-gpu$n" GPU_TAG="gpu$n" WITH_CANVAS=$([ "$n" = 1 ] && echo 1 || echo 0) \
        SKIP_SYNC=$([ "$n" = 1 ] && echo 0 || echo 1) "$HERE/job.sh" 2>&1)
  echo "$out" | sed "s/^/[gpu$n] /"
  id=$(echo "$out" | grep -oE '\b[0-9a-f]{24}\b' | head -n1 || true)
  [ -n "$id" ] || { echo "gpu$n: no job id in the output above; stop and check" >&2; exit 1; }
  IDS+=("$id")
done
echo
echo "RUN_ID=$RUN_ID"
for n in 1 2 3; do
  echo "gpu$n job ${IDS[$((n-1))]}  base_url https://${IDS[$((n-1))]}--8000.hf.jobs/v1"
done
echo "CANVAS_URL=https://${IDS[0]}--8765.hf.jobs"
echo "verify: hf jobs inspect --json cmpatino/<id>  ->  status.expose_urls"
