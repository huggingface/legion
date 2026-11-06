#!/usr/bin/env bash
# Build the flock wheel and upload it to the bucket for bootstrap-mode workers.
#   scripts/publish-wheel.sh <bucket-dir>     e.g. hf://buckets/<you>/flock/dist
# The bucket must exist: hf buckets create <you>/flock --private
# Uploads to <bucket-dir>/<git short sha>[-dirty]/<wheel> and prints that hf:// URL, which is
# what the controller's FLOCK_WHEEL_URL should hold. One directory per revision, so a
# relaunched worker never picks up a newer wheel than the controller was configured with.
# Next to it harbor-deps.tar (the controller job and every worker need it, controller/jobdeploy.py
# and controller/jobs.py; the Space build too, scripts/deploy-space.sh): the msgboard wheel (a
# private GitHub dependency a job cannot clone; built here with your git credentials) and
# constraints.txt, uv.lock's versions, so the controller and every worker of a run install the
# same ones.
# --harbor is accepted and ignored (it used to make harbor-deps.tar optional).
set -euo pipefail
USAGE="usage: scripts/publish-wheel.sh hf://buckets/<namespace>/<bucket>/<dir>"
if [ "${1:-}" = --harbor ]; then shift; fi
DEST=${1:?$USAGE}
case "$DEST" in hf://buckets/*/*) ;; *) echo "$USAGE" >&2; exit 2 ;; esac
BUCKET=$(echo "${DEST#hf://buckets/}" | cut -d/ -f1-2)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
REV=$(git -C "$ROOT" rev-parse --short HEAD)
git -C "$ROOT" diff --quiet HEAD -- src pyproject.toml || REV="$REV-dirty-$(date +%s)"
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
uv build --wheel --project "$ROOT" --out-dir "$OUT" -q
WHEEL=$(ls "$OUT"/*.whl)
URL="$DEST/$REV/$(basename "$WHEEL")"
mkdir "$OUT/harbor"
uv export -q --project "$ROOT" --frozen --extra controller --no-hashes --no-header \
  --no-emit-project --no-annotate > "$OUT/lock.txt"
uvx -q --from pip pip wheel -q --no-deps -w "$OUT/harbor" \
  "$(sed -n 's/^\(msgboard @ git+[^ ;]*\).*/\1/p' "$OUT/lock.txt")"
grep -v '^msgboard @' "$OUT/lock.txt" > "$OUT/harbor/constraints.txt"
tar -cf "$OUT/harbor-deps.tar" -C "$OUT/harbor" .
if ! ERR=$(hf buckets cp "$WHEEL" "$URL" 2>&1 >/dev/null); then
  if grep -q "404" <<<"$ERR" && grep -q "xet-write-token" <<<"$ERR"; then
    echo "bucket $BUCKET does not exist: run \`hf buckets create $BUCKET --private\`" >&2
  else
    echo "$ERR" >&2
  fi
  exit 1
fi
hf buckets cp "$OUT/harbor-deps.tar" "$DEST/$REV/harbor-deps.tar" >/dev/null
echo "$URL"
