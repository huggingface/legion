#!/usr/bin/env bash
# Upload experiments/ to the bucket and launch one job (FLAVOR, default cpu-basic) running experiments/<JOB_SCRIPT>.
# LABEL overrides the experiment label (default experiment=swarm-harness-m1).
#   launch.sh <name> <job_script relative to experiments/> [-e KEY=VAL ...]
# Prints the job id. Needs HF_TOKEN and ANTHROPIC_KEY in the environment (values never printed).
set -euo pipefail
NAME=${1:?name}; SCRIPT=${2:?script}; shift 2
HERE=$(cd "$(dirname "$0")/.." && pwd)
REV=${SCRIPTS_REV:-$(git -C "$HERE" rev-parse --short HEAD)-$(date +%s)}
hf buckets sync "$HERE" "hf://buckets/cmpatino/swarm-harness/scripts/$REV" --exclude "*/out/*" -q >/dev/null
ANTHROPIC_API_KEY="$ANTHROPIC_KEY" hf jobs run --detach --namespace cmpatino --flavor "${FLAVOR:-cpu-basic}" \
  --timeout "${TIMEOUT:-30m}" --name "$NAME" --label "${LABEL:-experiment=swarm-harness-m1}" \
  --secrets HF_TOKEN --secrets ANTHROPIC_API_KEY \
  -e SCRIPTS_REV="$REV" -e JOB_SCRIPT="$SCRIPT" "$@" \
  node:22-bookworm bash -c "$(cat "$HERE/common/job_bootstrap.sh")"
