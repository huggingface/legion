#!/bin/bash
# Submit through slurm.py submit; the only argument is a non-secret config-file path.
#SBATCH --job-name=agent-escape
#SBATCH --partition=hopper-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@1200
#SBATCH --no-requeue
set -euo pipefail
umask 077
repo=${SLURM_SUBMIT_DIR:?Submit from the Flock repository}
cd "$repo"
exec "$repo/.venv/bin/python" "$repo/experiments/agent-escape/slurm.py" run "${1:?config path required}"
