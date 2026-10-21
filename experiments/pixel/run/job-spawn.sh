#!/usr/bin/env bash
# Experiment 2 GPUs: exactly job-1k.sh (three rtx-pro-6000 jobs, canvas on gpu1, --timeout 75m)
# with region leases enforced on the canvas: REGION_ENFORCE=1, so painters only claim and paint
# inside the bands their leader granted them.
#   RUN_ID=spawn-YYYYMMDD experiments/pixel/run/job-spawn.sh
set -euo pipefail
export REGION_ENFORCE=1 CANVAS_REGION_TTL_MAX=${CANVAS_REGION_TTL_MAX:-7200}
exec "$(cd "$(dirname "$0")" && pwd)/job-1k.sh" "$@"
