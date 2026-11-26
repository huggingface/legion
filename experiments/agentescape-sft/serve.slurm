#!/bin/bash
# From the repository root, after creating RUN_DIR:
# sbatch --nodes=2 -o RUN_DIR/slurm.log \
#   experiments/agentescape-sft/serve.slurm MODEL_SNAPSHOT RUN_DIR
# One independent Qwen3.5-27B instance per allocated GPU; router on :10400.
# Set QWEN_IMAGE to the shared vLLM SquashFS image.
# MAX_MODEL_LEN, MAX_NUM_SEQS and MAX_BATCHED_TOKENS are optional overrides.
# QWEN_LAUNCHER may name qwen_serve.py when submitting from another directory.
#SBATCH --job-name=sft35-router
#SBATCH --partition=hopper-b300
#SBATCH --qos=high
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=496000M
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --no-kill
set -euo pipefail
umask 077
launcher=${QWEN_LAUNCHER:-${SLURM_SUBMIT_DIR:-$PWD}/experiments/agentescape-sft/qwen_serve.py}
[[ -f $launcher ]] || { echo "Set QWEN_LAUNCHER to qwen_serve.py's absolute path" >&2; exit 2; }
exec python3 "$launcher" launch "${1:?local model snapshot required}" "${2:?run directory required}"
