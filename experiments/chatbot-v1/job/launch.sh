#!/usr/bin/env bash
# Prepare the payload for the chatbot-v1 job and print the `hf jobs run` command (it launches
# nothing). Run from anywhere in the repo, after committing:
#   experiments/chatbot-v1/job/launch.sh hf://buckets/<ns>/<bucket>/<run>          # upload
#   experiments/chatbot-v1/job/launch.sh <local dir>                                # rehearsal
# Uploads <dest>/payload/: the flock wheel of this checkout (uv build) and repo.tar.gz (git
# archive of HEAD: experiments/chatbot-v1, worlds/chatbot, worlds/eliza).
set -euo pipefail
DEST=${1:?usage: launch.sh hf://buckets/<ns>/<bucket>/<run> | <local dir>}
DEST=${DEST%/}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
git -C "$ROOT" diff --quiet HEAD -- src experiments/chatbot-v1 worlds || \
  echo "warning: uncommitted changes are NOT in the payload (git archive of HEAD)" >&2
OUT=$(mktemp -d)
uv build --wheel --project "$ROOT" --out-dir "$OUT" -q
git -C "$ROOT" archive --format=tar.gz -o "$OUT/repo.tar.gz" HEAD \
  experiments/chatbot-v1 worlds/chatbot worlds/eliza
if [[ $DEST == hf://* ]]; then
  hf buckets sync "$OUT" "$DEST/payload" -q >/dev/null
else
  mkdir -p "$DEST/payload" && cp "$OUT"/* "$DEST/payload/"
fi
echo "payload: $DEST/payload/ ($(cd "$OUT" && ls | tr '\n' ' '))" >&2
rm -rf "$OUT"
cat <<CMD
hf jobs run --detach --flavor cpu-upgrade --timeout 270m --name chatbot-v1 \\
  --label experiment=chatbot-v1 \\
  --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN \\
  -e CHATBOT_BUCKET=$DEST -e CHATBOT_BOARD_URL=<board url> -e CHATBOT_CHANNEL=<channel> \\
  python:3.12-slim bash -c "\$(cat $ROOT/experiments/chatbot-v1/job/bootstrap.sh)"
CMD
