#!/bin/bash
# The sweep's model server: one vLLM on one GPU (docs/open-models.md "On the HF cluster"), in
# Pyxis on hopper-b200. The agents (podman on hopper-cpu) reach it by this node's address.
#   sbatch -p hopper-b200 -o <dir>/vllm-%j.log experiments/s3-aggregation/sweep/serve-vllm.sbatch
# Its log ends with: serving <model> at http://<node address>:8000/v1
#SBATCH --job-name=s3-sweep-vllm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=3:00:00
set -uo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 &
until curl -sf -o /dev/null "localhost:$PORT/v1/models"; do
  sleep 10
  kill -0 $! 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait
