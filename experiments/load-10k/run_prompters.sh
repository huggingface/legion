#!/usr/bin/env bash
# run_prompters.sh <swarm> <run-name> [N=5] [rate=10] [extra prompt_job.py args...]
# Launches N cpu-basic prompter jobs (experiments/load-10k/prompt_job.py), slice k/N each, writing
# hf://buckets/cmpatino/swarm-harness/load-10k-runs/<run-name>/slice-k.jsonl. Prints one job id
# per line. Needs SWARM_HARNESS_URL, SWARM_HARNESS_TOKEN and HF_TOKEN in the environment (the
# tokens go in as job secrets and are never printed).
set -euo pipefail
SWARM=${1:?swarm}; RUN=${2:?run name}; N=${3:-5}; RATE=${4:-10}; shift 4 || shift $#
HERE=$(cd "$(dirname "$0")" && pwd)
for ((k = 0; k < N; k++)); do
  hf jobs uv run --detach --namespace cmpatino --flavor cpu-basic --timeout "${TIMEOUT:-40m}" \
    --name "prompter-$RUN-$k" --label experiment=swarm-harness-10k --label "prompter=$RUN" \
    -e "SWARM_HARNESS_URL=$SWARM_HARNESS_URL" --secrets SWARM_HARNESS_TOKEN --secrets HF_TOKEN \
    "$HERE/prompt_job.py" -- "$SWARM" --slice "$k/$N" --rate "$RATE" \
    --out "hf://buckets/cmpatino/swarm-harness/load-10k-runs/$RUN/slice-$k.jsonl" "$@"
done
