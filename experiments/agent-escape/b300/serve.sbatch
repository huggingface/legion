#!/bin/bash
#SBATCH --job-name=qwen27b-bench
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

# sbatch -o /absolute/run/slurm.log serve.sbatch REPO DATA RUN [benchmark|serve] [tp|replicas] [GPUS]
REPO=$(realpath "${1:?repository path required}")
DATA=$(realpath "${2:?prepared dataset directory required}")
RUN=$(realpath "${3:?existing empty run directory required}")
MODE=${4:-benchmark}
LAYOUT=${5:-tp}
GPUS=${6:-1}
IMAGE=${QWEN_IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
[[ $MODE == benchmark || $MODE == serve ]] || exit 2
[[ $LAYOUT == tp || $LAYOUT == replicas ]] || exit 2
[[ $GPUS =~ ^(1|2|4|8)$ ]] || exit 2
[[ -f $IMAGE && -f $DATA/manifest.json ]] || exit 2

MODEL_SOURCE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["model_path"])' "$DATA/manifest.json")
REVISION=$(basename "$MODEL_SOURCE")
LOCAL_ROOT=/scratch/$USER/qwen-b300
MODEL_LOCAL=$LOCAL_ROOT/models/$REVISION
JOB_LOCAL=$LOCAL_ROOT/jobs/$SLURM_JOB_ID
mkdir -p "$JOB_LOCAL" "$LOCAL_ROOT/models"
# One writer per node; retain model weights and JIT cache between sequential configurations.
(
    flock -x 9
    if [[ ! -f $MODEL_LOCAL/.complete ]]; then
        mkdir -p "$MODEL_LOCAL"
        rsync -aL -- "$MODEL_SOURCE/" "$MODEL_LOCAL/"
        touch "$MODEL_LOCAL/.complete"
    fi
) 9>"$LOCAL_ROOT/model-stage.lock"

sha256sum "$IMAGE" > "$RUN/image.sha256"
nvidia-smi -q > "$RUN/gpu-info.txt"
nvidia-smi topo -m > "$RUN/gpu-topology.txt"
scontrol show job "$SLURM_JOB_ID" > "$RUN/slurm-job.txt"
test -f "$RUN/source/benchmark.py"
export VLLM_API_KEY
VLLM_API_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')
printf '%s\n' "$VLLM_API_KEY" > "$RUN/api-key"
export VLLM_CACHE_ROOT=/job/cache/vllm TRITON_CACHE_DIR=/job/cache/triton
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
STEP_PID=
cleanup() {
    if [[ -n $STEP_PID ]]; then
        kill -TERM "$STEP_PID" 2>/dev/null || true
        wait "$STEP_PID" || true
    fi
}
trap cleanup EXIT
trap 'exit 143' TERM INT

srun --container-image="$IMAGE" --no-container-mount-home \
    --container-mounts="$RUN/source:/code:ro,$DATA:/data:ro,$RUN:/results,$MODEL_LOCAL:/model:ro,$JOB_LOCAL:/job" \
    --container-env=VLLM_API_KEY,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,HF_HUB_OFFLINE,TOKENIZERS_PARALLELISM,PYTHONUNBUFFERED \
    bash -c 'set -euo pipefail
        python3 -m pip install --disable-pip-version-check --target /job/plotting "matplotlib==3.10.7" > /results/plot-install.log 2>&1
        export BENCH_PLOT_PATH=/job/plotting
        exec python3 /code/benchmark.py \
            --data /data --output /results/measurements --model-path /model \
            --mode "$1" --layout "$2" --gpus "$3"
    ' _ "$MODE" "$LAYOUT" "$GPUS" &
STEP_PID=$!
wait "$STEP_PID"
STEP_PID=
