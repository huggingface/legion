#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@1200
#SBATCH --no-requeue
set -euo pipefail
umask 077
repo=${SLURM_SUBMIT_DIR:?Submit from the Flock repository}
cd "$repo"
exec "$repo/.venv/bin/python" "$repo/experiments/agent-escape/ablation.py" "$@"
