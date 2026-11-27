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
# If --config sets served-model-name, set READINESS_MODEL_ID to the effective API model ID
# (including any command-line override); the config file is read inside the container.
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
# vLLM accepts underscores in option names and the last occurrence wins.
SERVED_MODEL=$MODEL
args=("${PARSER_ARGS[@]}" "$@")
for ((i=0; i<${#args[@]}; i++)); do
  option=${args[i]%%=*}
  option=${option//_/-}
  if [[ $option == --served-model-name ]]; then
    if [[ ${args[i]} == *=* ]]; then
      SERVED_MODEL=${args[i]#*=}
    else
      SERVED_MODEL=${args[i+1]:-$MODEL}
      ((i+=1))
    fi
  fi
done
SERVED_MODEL=${READINESS_MODEL_ID:-$SERVED_MODEL}
# Do not mistake a pre-existing server for this job's server, even when it serves the same ID.
python3 - "$PORT" <<'PYPORT' || exit 1
import socket, sys
try:
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", int(sys.argv[1])))
except OSError as exc:
    print(f"Cannot start vLLM on port {sys.argv[1]}: {exc}", file=sys.stderr)
    sys.exit(1)
PYPORT
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" ${PARSER_ARGS[@]+"${PARSER_ARGS[@]}"} "$@" &
SRUN_PID=$!
while :; do
  kill -0 "$SRUN_PID" 2>/dev/null || { echo "vLLM exited during startup (srun PID $SRUN_PID)" >&2; exit 1; }
  if curl -sf --max-time 5 "localhost:$PORT/v1/models" | python3 -c '
import json, sys
try:
    sys.exit(0 if any(m.get("id") == sys.argv[1] for m in json.load(sys.stdin)["data"]) else 1)
except (ValueError, KeyError, TypeError, AttributeError):
    sys.exit(1)
' "$SERVED_MODEL"; then
    kill -0 "$SRUN_PID" 2>/dev/null || { echo "vLLM exited during startup (srun PID $SRUN_PID)" >&2; exit 1; }
    break
  fi
  # Check for a failed launch each second while keeping probes ten seconds apart.
  for _ in {1..10}; do
    sleep 1
    kill -0 "$SRUN_PID" 2>/dev/null || { echo "vLLM exited during startup (srun PID $SRUN_PID)" >&2; exit 1; }
  done
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait "$SRUN_PID"  # the job fails when vLLM does
