#!/bin/bash
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=7G
#SBATCH --time=00:20:00
#SBATCH --requeue
# Prepared for the operator; never submitted by the implementation or tests.
set -euo pipefail
RUN=${1:-run4}
LOG_TAG=worker
. /fsx/carlos_patino/prj-eliza/slurm/common.sh
CONTAINER_IMAGE=${CONTAINER_IMAGE_agent:-$PRJ/images/pi-worker-node22.sqsh}
container_srun_args
trap '' USR1
srun "${CTR_SRUN[@]}" bash -c '
  set -euo pipefail
  export PATH="$VENV/bin:/usr/local/bin:/usr/bin:/bin"
  export HOME=/work/home HF_HOME=/work/hf TMPDIR=/work/tmp RUFF_CACHE_DIR=/work/ruff-cache
  export FLOCK_PI_CLI=/usr/local/bin/pi
  export FLOCK_REQUIRE_ISOLATION=1
  mkdir -p "$HOME" "$HF_HOME" "$TMPDIR"
  bash "$REPO/experiments/run4/dryrun.sh"
' &
TEST_PID=$!
trap 'scontrol requeue "$SLURM_JOB_ID"; exit 0' USR1
wait "$TEST_PID"
