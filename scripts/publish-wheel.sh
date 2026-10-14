#!/usr/bin/env bash
# Build the flock wheel and upload it to the bucket for bootstrap-mode workers.
#   scripts/publish-wheel.sh [bucket-dir]     (default hf://buckets/cmpatino/flock/dist)
# Uploads to <bucket-dir>/<git short sha>[-dirty]/<wheel> and prints that hf:// URL, which is
# what the controller's FLOCK_WHEEL_URL should hold. One directory per revision, so a
# relaunched worker never picks up a newer wheel than the controller was configured with.
set -euo pipefail
DEST=${1:-hf://buckets/cmpatino/flock/dist}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REV=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" diff --quiet HEAD -- src pyproject.toml || REV="$REV-dirty-$(date +%s)"
OUT=$(mktemp -d)
uv build --wheel --project "$ROOT" --out-dir "$OUT" -q
WHEEL=$(ls "$OUT"/*.whl)
URL="$DEST/$REV/$(basename "$WHEEL")"
hf buckets cp "$WHEEL" "$URL" >/dev/null
rm -rf "$OUT"
echo "$URL"
