#!/usr/bin/env bash
# Build the flock wheel and upload it to the bucket for bootstrap-mode workers.
#   scripts/publish-wheel.sh <bucket-dir>     e.g. hf://buckets/<you>/flock/dist
# The bucket must exist: hf buckets create <you>/flock --private
# Uploads to <bucket-dir>/<git short sha>[-dirty]/<wheel> and prints that hf:// URL, which is
# what the controller's FLOCK_WHEEL_URL should hold. One directory per revision, so a
# relaunched worker never picks up a newer wheel than the controller was configured with.
set -euo pipefail
USAGE="usage: scripts/publish-wheel.sh hf://buckets/<namespace>/<bucket>/<dir>"
DEST=${1:?$USAGE}
case "$DEST" in hf://buckets/*/*) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
BUCKET=$(echo "${DEST#hf://buckets/}" | cut -d/ -f1-2)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REV=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" diff --quiet HEAD -- src pyproject.toml || REV="$REV-dirty-$(date +%s)"
OUT=$(mktemp -d)
uv build --wheel --project "$ROOT" --out-dir "$OUT" -q
WHEEL=$(ls "$OUT"/*.whl)
URL="$DEST/$REV/$(basename "$WHEEL")"
if ! ERR=$(hf buckets cp "$WHEEL" "$URL" 2>&1 >/dev/null); then
  rm -rf "$OUT"
  if grep -q "404" <<<"$ERR" && grep -q "xet-write-token" <<<"$ERR"; then
    echo "bucket $BUCKET does not exist: run \`hf buckets create $BUCKET --private\`" >&2
  else
    echo "$ERR" >&2
  fi
  exit 1
fi
rm -rf "$OUT"
echo "$URL"
