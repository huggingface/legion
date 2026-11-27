#!/bin/bash
# One vLLM server on one GPU, in Pyxis (docs/open-models.md "On the HF cluster"). Agents on other
# nodes (podman on a CPU node, or a `compute: slurm` run's workers) reach it by this node's address.
#   GPU_PARTITION=hopper-b300  # B300 example on the HF cluster
#   sbatch -p "$GPU_PARTITION" scripts/slurm/serve-vllm.sbatch [MODEL [VLLM_ARG...]]
#   e.g. thinking off for every request (a request can still turn it on):
#   sbatch -p "$GPU_PARTITION" scripts/slurm/serve-vllm.sbatch Qwen/Qwen3.5-27B \
#     --default-chat-template-kwargs '{"enable_thinking": false}'
# MODEL may also come from the environment (sbatch --export=ALL,MODEL=...,PORT=8011); PORT
# (default 8000) and IMAGE (a vLLM .sqsh) likewise. The weights go to the node's own disk.
# VLLM_PARSER_ARGS: the tool-call and reasoning flags, by default Qwen's (below); set your
# model's, or empty (VLLM_PARSER_ARGS= sbatch ...) for none. sbatch passes the environment.
# The log is vllm-<job id>.log (%x-%j.log: -J renames it) in the directory sbatch ran from,
# Slurm's default working directory (-o elsewhere/%x-%j.log moves it). It ends with:
#   serving <model> at http://<node address>:<port>/v1
#SBATCH --job-name=vllm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
set -uo pipefail
MODEL=${1:-${MODEL:-Qwen/Qwen3.5-0.8B}}
[ $# -eq 0 ] || shift  # the rest: extra `vllm serve` arguments
PORT=${PORT:-8000}
IMAGE=${IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
QWEN_PARSERS="--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
read -ra PARSER_ARGS <<< "${VLLM_PARSER_ARGS-$QWEN_PARSERS}"  # set but empty: none
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" ${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} "$@" &
SRUN_PID=$!
until curl -sf -o /dev/null "localhost:$PORT/v1/models"; do
  sleep 10
  kill -0 "$SRUN_PID" 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait "$SRUN_PID"  # the job fails when vLLM does
