#!/usr/bin/env bash
# Stage the world Space's files (it deploys nothing):
#   experiments/run3/world-space/prepare.sh <out dir>
# Then, by hand: hf repo create <ns>/<space> --repo-type space --space-sdk docker (public: the
# workers send no HF token, the world token guards every route), set the secrets
# CHATBOT_WORLD_TOKEN and HF_TOKEN and the variable CHATBOT_BUCKET, pick CPU hardware
# (README.md: probe runs are the world's load), then `hf upload <ns>/<space> <out dir> --repo-type space`.
set -euo pipefail
OUT=${1:?usage: prepare.sh <out dir>}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
HERE=$ROOT/experiments/run3/world-space
mkdir -p "$OUT/dist" "$OUT/worlds"
uv build --wheel --project "$ROOT" --out-dir "$OUT/dist" -q
for w in chatbot3 chatbot2 chatbot eliza; do
  rsync -a --exclude ws --exclude __pycache__ "$ROOT/worlds/$w" "$OUT/worlds/"
done
cp "$HERE/Dockerfile" "$HERE/start.sh" "$HERE/README.md" "$OUT/"
cp "$ROOT/experiments/run3/world.yaml" "$OUT/world.yaml"
echo "staged $OUT: $(cd "$OUT" && find . -type f | wc -l) files"
