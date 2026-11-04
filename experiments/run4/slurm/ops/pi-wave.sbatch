#!/bin/bash
#SBATCH --partition=hopper-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=3800M
#SBATCH --time=00:10:00
# Optional manual wave; waves.py performs the same all-node capacity check.
set -euo pipefail
RUN=${1:-run4}
LOG_TAG=controller
. /fsx/carlos_patino/prj-eliza/slurm/common.sh
export FLOCK_URL="http://$(cat "$HOSTS_DIR/controller.host"):7860"
cd "$RUN_DIR/experiment"
"$VENV/bin/python" waves.py add "${2:-20}" --type agent --spec spec.slurm.local.yaml --waves waves.yaml
