#!/usr/bin/env bash
# Launch the controller as an HF Job in its workers' network group (job mode, docs/deploy-job.md).
#   scripts/deploy-job.sh <wheel-url> [--flavor cpu-upgrade] [--group flock] [--timeout 48h]
#                         [--namespace NS] [--state-url URL] [--secret NAME]... [--replace] [--dry-run]
# Needs HF_TOKEN and FLOCK_TOKEN (and ANTHROPIC_API_KEY or ANTHROPIC_KEY, plus every --secret)
# in the environment; values are never printed. JOB_LABELS sets the workers' deployment-wide
# job labels (default experiment=flock-m1). Prints the job id and its two proxy URLs.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" exec python3 -m flock.controller.jobdeploy "$@"
