#!/bin/bash
# From the repository root, after creating RUN_DIR:
# sbatch --nodes=4 -o RUN_DIR/slurm.log \
#   experiments/agent-escape/b300/serve-qwen38-27b.slurm MODEL_SNAPSHOT RUN_DIR
# One independent Qwen3.8-27B instance per allocated GPU; router on :8000.
# QWEN_IMAGE, MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS are optional overrides.
# QWEN_LAUNCHER may name qwen_serve.py when submitting from another directory.
#SBATCH --job-name=qwen27b-serve
#SBATCH --partition=hopper-b300
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-gpu=11
#SBATCH --mem-per-gpu=496000M
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
#SBATCH --no-kill
set -euo pipefail
umask 077
launcher=${QWEN_LAUNCHER:-${SLURM_SUBMIT_DIR:-$PWD}/experiments/agent-escape/b300/qwen_serve.py}
[[ -f $launcher ]] || { echo "Set QWEN_LAUNCHER to qwen_serve.py's absolute path" >&2; exit 2; }
exec python3 "$launcher" launch "${1:?local model snapshot required}" "${2:?run directory required}"
