#!/usr/bin/env bash
# Prepare the payload for the run3 single job and print the `hf jobs run` command (it launches
# nothing). Run from anywhere in the repo, after committing:
#   experiments/run3/job/launch.sh hf://buckets/<ns>/<bucket>/<run>          # upload
#   experiments/run3/job/launch.sh <local dir>                                # rehearsal
# Uploads <dest>/payload/: the flock wheel of this checkout (uv build) and repo.tar.gz (git
# archive of HEAD: experiments/run3 and the worlds it stacks on).
set -euo pipefail
DEST=${1:?usage: launch.sh hf://buckets/<ns>/<bucket>/<run> | <local dir>}
DEST=${DEST%/}
ROOT=$(cd "$(dirname "$0")/../../.." && pwd)
git -C "$ROOT" diff --quiet HEAD -- src experiments/run3 worlds || \
  echo "warning: uncommitted changes are NOT in the payload (git archive of HEAD)" >&2
OUT=$(mktemp -d)
uv build --wheel --project "$ROOT" --out-dir "$OUT" -q
git -C "$ROOT" archive --format=tar.gz -o "$OUT/repo.tar.gz" HEAD \
  experiments/run3 worlds/chatbot3 worlds/chatbot2 worlds/chatbot worlds/eliza
if [[ $DEST == hf://* ]]; then
  hf buckets sync "$OUT" "$DEST/payload" -q >/dev/null
else
  mkdir -p "$DEST/payload" && cp "$OUT"/* "$DEST/payload/"
fi
echo "payload: $DEST/payload/ ($(cd "$OUT" && ls | tr '\n' ' '))" >&2
rm -rf "$OUT"
cat <<CMD
hf jobs run --detach --flavor cpu-xl --timeout 48h --name run3 --label experiment=run3 \\
  --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN --secrets OPENAI_API_KEY \\
  -e CHATBOT_BUCKET=$DEST -e CHATBOT_BOARD_URL=<board url> -e CHATBOT_CHANNEL=<channel> \\
  -e OPENAI_BASE_URL=<vllm url>/v1 -e MODEL_WORKER=<served model id> \\
  python:3.12-slim bash -c "\$(cat $ROOT/experiments/run3/job/bootstrap.sh)"
CMD
