#!/usr/bin/env bash
# Packing job body (runs inside the cpu-basic job after common/job_bootstrap.sh).
set -euo pipefail
export IS_SANDBOX=1 CLAUDE_CONFIG_DIR=${CLAUDE_CONFIG_DIR:-/state/claude}
OUT=${OUT:-/out}; DEST=${DEST:-hf://buckets/cmpatino/flock/packing/job}
python3 /scripts/packing/packing.py --harness claude --levels "${LEVELS:-5,10,20,40}" --idle "${IDLE:-60}" \
  --base /work/pack --out "$OUT" --after-level "hf buckets sync $OUT $DEST -q >/dev/null 2>&1 || true"
hf buckets sync "$OUT" "$DEST" -q
