#!/bin/bash
# The thinking-off re-runs' model server: as the sweep's serve-vllm.sbatch (the same image and
# parsers), plus `--default-chat-template-kwargs '{"enable_thinking": false}'`: Qwen3.5-27B's
# chat template thinks by default. 11 CPUs, so eight servers fit a B200 node (92 CPUs).
#   sbatch -p hopper-b200 -J s3-maze-vllm-off-9001 --export=ALL,MODEL=Qwen/Qwen3.5-27B,PORT=9001 -o <dir>/vllm-%j.log serve-vllm.sbatch
# Its log ends with: serving <model> at http://<node address>:<port>/v1
#SBATCH --job-name=s3-maze-vllm-off
#SBATCH --gpus=1
#SBATCH --cpus-per-task=11
#SBATCH --mem=96G
#SBATCH --time=12:00:00
set -uo pipefail
MODEL=${MODEL:-Qwen/Qwen3.5-27B}
PORT=${PORT:-9000}
IMAGE=${IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --default-chat-template-kwargs '{"enable_thinking": false}' &
until curl -sf -o /dev/null "localhost:$PORT/v1/models"; do
  sleep 10
  kill -0 $! 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait
