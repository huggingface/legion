#!/bin/bash
#SBATCH --job-name=flock-openenv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=B:TERM@1200
#SBATCH --no-requeue
set -euo pipefail
umask 077
cd "${SLURM_SUBMIT_DIR:?Submit from Flock}"
exec "${1:?Python path}" -m experiments.benchmarks array-run "${2:?campaign}" "${3:?secrets file}"
