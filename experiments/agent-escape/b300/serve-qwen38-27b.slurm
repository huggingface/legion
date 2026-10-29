#!/bin/bash
# Review draft for Qwen/Qwen3.8-27B; not submitted.
# Prepare an immutable, complete local HF snapshot of Qwen/Qwen3.8-27B and an empty
# run directory first. Uses the same image and parsers as this directory's benchmark.
# Submit after review (RUN_DIR must exist before Slurm opens its output file):
#   sbatch -o RUN_DIR/slurm.log serve-qwen38-27b.slurm MODEL_SNAPSHOT RUN_DIR
# Defaults: eight BF16 TP1 replicas, native vLLM DP frontend, text-only, 32K context.
# Override TP_SIZE=2 DP_SIZE=4 for four TP2 replicas; TP_SIZE * DP_SIZE must be eight.
# MAX_MODEL_LEN, MAX_NUM_SEQS, MAX_BATCHED_TOKENS and QWEN_IMAGE are configurable.
# No speculative decoding or DeepSeek-specific expert/attention kernels.
# Outputs: endpoint.json (non-secret) and api-key (0600); Flock uses OPENAI_API_KEY.
#SBATCH --job-name=qwen27b-serve
#SBATCH --partition=hopper-b300
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --no-requeue
set -euo pipefail
umask 077

MODEL_ID=Qwen/Qwen3.8-27B
MODEL_SOURCE=$(realpath "${1:?complete local Qwen3.8-27B snapshot required}")
RUN_DIR=$(realpath "${2:?existing empty run directory required}")
IMAGE=${QWEN_IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-8}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_BATCHED_TOKENS=${MAX_BATCHED_TOKENS:-8192}
[[ -d $RUN_DIR && -f $MODEL_SOURCE/config.json && -f $IMAGE ]] || {
    echo 'Missing run directory, model config, or container image' >&2; exit 2;
}
[[ -z $(find "$RUN_DIR" -mindepth 1 -maxdepth 1 ! -name slurm.log -print -quit) ]] || {
    echo 'Use a fresh run directory (only slurm.log may already exist)' >&2; exit 2;
}
for number in "$TP_SIZE" "$DP_SIZE" "$MAX_MODEL_LEN" "$MAX_NUM_SEQS" "$MAX_BATCHED_TOKENS"; do
    [[ $number =~ ^[1-9][0-9]*$ ]] || { echo 'Serving limits must be positive integers' >&2; exit 2; }
done
(( TP_SIZE * DP_SIZE == 8 )) || { echo 'TP_SIZE * DP_SIZE must equal 8' >&2; exit 2; }

# Cache by canonical snapshot path; snapshots must not be modified in place.
CACHE_ID=$(printf '%s' "$MODEL_SOURCE" | sha256sum | cut -d ' ' -f 1)
LOCAL_ROOT=/scratch/$USER/qwen-serving
MODEL_LOCAL=$LOCAL_ROOT/models/$CACHE_ID
JOB_LOCAL=$LOCAL_ROOT/jobs/$SLURM_JOB_ID
mkdir -p "$LOCAL_ROOT/models" "$JOB_LOCAL"
(
    /usr/bin/flock -x 9
    if [[ ! -f $MODEL_LOCAL/.complete ]]; then
        mkdir -p "$MODEL_LOCAL"
        rsync -aL -- "$MODEL_SOURCE/" "$MODEL_LOCAL/"
        touch "$MODEL_LOCAL/.complete"
    fi
) 9>"$LOCAL_ROOT/model-stage.lock"

# Generate a private credential; never put it in process arguments or logs.
export VLLM_API_KEY
VLLM_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
printf '%s\n' "$VLLM_API_KEY" > "$RUN_DIR/api-key"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export TMPDIR=/job/tmp
export VLLM_CACHE_ROOT=/job/cache/vllm TRITON_CACHE_DIR=/job/cache/triton
mkdir -p "$JOB_LOCAL/tmp"
STEP_PID=
cleanup() {
    rm -f "$RUN_DIR/endpoint.json"
    if [[ -n $STEP_PID ]]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

srun --container-image="$IMAGE" --no-container-mount-home \
    --container-mounts="$MODEL_LOCAL:/model:ro,$JOB_LOCAL:/job" \
    --container-env=VLLM_API_KEY,HF_HUB_OFFLINE,TOKENIZERS_PARALLELISM,TMPDIR,VLLM_CACHE_ROOT,TRITON_CACHE_DIR \
    vllm serve /model --served-model-name "$MODEL_ID" \
    --host 0.0.0.0 --port 8000 --dtype bfloat16 \
    --tensor-parallel-size "$TP_SIZE" --data-parallel-size "$DP_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" --gpu-memory-utilization 0.9 \
    --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
    --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --default-chat-template-kwargs '{"reasoning_effort":"low"}' \
    --language-model-only --enable-prefix-caching &
STEP_PID=$!

deadline=$((SECONDS + 3600))
until printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" | \
    curl --max-time 10 -fs -o /dev/null -H @- http://localhost:8000/v1/models; do
    kill -0 "$STEP_PID" 2>/dev/null || { echo 'vLLM exited during startup' >&2; exit 1; }
    (( SECONDS < deadline )) || { echo 'Startup exceeded one hour' >&2; exit 1; }
    sleep 10
done
noauth=$(curl --max-time 10 -s -o /dev/null -w '%{http_code}' http://localhost:8000/v1/models)
[[ $noauth == 401 ]] || { echo "API key not enforced (HTTP $noauth); stopping" >&2; exit 1; }
python3 - "$RUN_DIR/endpoint.json" "$MODEL_ID" "$(hostname)" "$SLURM_JOB_ID" <<'PY'
import json
import pathlib
import sys

path, model, host, job = sys.argv[1:]
target = pathlib.Path(path)
temporary = target.with_suffix('.tmp')
temporary.write_text(json.dumps({
    'model': model, 'base_url': f'http://{host}:8000/v1',
    'metrics_url': f'http://{host}:8000/metrics', 'job_id': job,
}, indent=2) + '\n')
temporary.replace(target)
PY
echo "Serving $MODEL_ID at http://$(hostname):8000/v1; endpoint: $RUN_DIR/endpoint.json"
wait "$STEP_PID"
STEP_PID=
