#!/bin/bash -l
# Run 4: TWO DeepSeek-V4.1-Flash-1017 servers, one per B300 node (array tasks 0 and 1), each vLLM DP8 +
# EP + dspark behind vllm-router on :8000. Copy of /fsx/anton/shared/serve-deepseek-4-1-flash-1017.slurm
# (run 3's server, job dsv41-flash) with only these changes:
#   - --array=0-1 (two independent nodes; --exclusive keeps them apart), --time=60:00:00 (48 h run +
#     up to 2 h final rounds + startup and ramp; vLLM must outlive the controller's last round)
#   - the router/vLLM API key = OPENAI_API_KEY from secrets.env, passed as VLLM_API_KEY in the container
#     env (never on a command line, so not in ps/sacct); a check after startup says whether it is enforced
#   - writes runs/run4/hosts/vllm-<task>.host and prints the run.env lines once serving
#   - logs under runs/run4/logs
#
#   sbatch /fsx/carlos_patino/prj-eliza/slurm/ops/serve-dsv41-flash-run4.slurm [--python-frontend] [vllm args]
#   sbatch --array=1 .../serve-dsv41-flash-run4.slurm      # restart only server 1 (it gets a NEW node name)
#
# NOT SUBMITTED. Startup: ~13 min warm, ~30 min cold, +8 min staging on a fresh node.
#SBATCH --job-name=run4-vllm
#SBATCH --partition=hopper-b300
#SBATCH --qos=normal
#SBATCH --array=0-1
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=88
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=60:00:00
#SBATCH --output=/fsx/carlos_patino/prj-eliza/runs/run4/logs/%x-%A_%a.log
set -euo pipefail
umask 0000  # the shared caches stay writable for every user (Anton's cache dir)

PRJ=/fsx/carlos_patino/prj-eliza
RUN=${RUN:-run4}
HOSTS_DIR=$PRJ/runs/$RUN/hosts
SECRETS_FILE=$PRJ/secrets/secrets.env
SHARED=/fsx/anton/shared
TAG=v0.32.0
ROUTER=vllm-router==0.1.16
MODEL=deepseek-ai/DeepSeek-V4.1-Flash-1017
CACHE=$SHARED/vllm_cache                 # hf_hub (weights) + $TAG (JIT kernels), shipped as tar parts
JOBTMP=/scratch/$USER/job-$SLURM_JOB_ID  # per-job scratch: TMPDIR, router install, logging config
PATH=$SHARED/bin:$PATH                   # dir-pload / dir-pship
IDX=${SLURM_ARRAY_TASK_ID:-0}
FRONTEND=rust args=()
for a in "$@"; do [[ $a == --python-frontend ]] && FRONTEND=python || args+=("$a"); done
set -- ${args[@]+"${args[@]}"}
t0=$SECONDS
log() { echo "[$(date +%T) +$((SECONDS - t0))s] $*"; }
log "job $SLURM_JOB_ID (array ${SLURM_ARRAY_JOB_ID:-?}_$IDX) on $(hostname): $MODEL, vLLM $TAG, $FRONTEND API frontend${*:+, extra args: $*}"

# The key: only OPENAI_API_KEY is taken from secrets.env (a subshell, so HF_TOKEN etc. stay out of this job).
VLLM_API_KEY=$(set +u; . "$SECRETS_FILE" >/dev/null 2>&1; printf '%s' "${OPENAI_API_KEY:-}")
[ -n "$VLLM_API_KEY" ] || { log "ERROR: OPENAI_API_KEY is empty in $SECRETS_FILE"; exit 1; }
export VLLM_API_KEY

# B300 nodes sit in the Atlanta AZ, ~15 ms from /fsx: unpack the weights and JIT kernel cache
# to node-local NVMe once per node instead of reading /fsx file by file.
log "staging weights (510 GB) + JIT cache to local NVMe: ~8 min on a fresh node (one-time), seconds if already there"
HUB=$(dir-pload "$CACHE/hf_hub")
if JIT=$(dir-pload "$CACHE/$TAG" 2>/dev/null); then log "JIT cache version $(cat "$CACHE/$TAG.pshipped/current"): warm start, ~13 min"
else JIT=/scratch/$USER/jit-$TAG; mkdir -p "$JIT"; log "no shipped JIT cache yet: cold start, ~30 min"; fi
mkdir -p "$JOBTMP"; touch "$JOBTMP/.start"
trap 'rm -f "$HOSTS_DIR/vllm-$IDX.host"; kill $(jobs -p) 2>/dev/null; rm -rf "$JOBTMP"' EXIT INT TERM
log "staged: weights in $HUB, JIT cache in $JIT"

# vLLM's default logging config, minus the DP coordinator's benign "out-of-order step" warnings.
cat >"$JOBTMP/vllm-logging.json" <<'EOF'
{"version": 1, "disable_existing_loggers": false,
 "formatters": {"vllm": {"()": "vllm.logging_utils.NewLineFormatter", "datefmt": "%m-%d %H:%M:%S", "format": "(%(vllm_process_name)s pid=%(process)d) %(levelname)s %(asctime)s [%(fileinfo)s:%(lineno)d] %(message)s"}},
 "handlers": {"vllm": {"class": "logging.StreamHandler", "formatter": "vllm", "level": "INFO", "stream": "ext://sys.stdout"}},
 "loggers": {"vllm": {"handlers": ["vllm"], "level": "INFO", "propagate": false}, "vllm.v1.engine.coordinator": {"level": "ERROR"}}}
EOF

export HF_HUB_CACHE=/hf-hub TMPDIR=/scratch HOME=/jit-cache/home VLLM_CACHE_ROOT=/jit-cache/vllm \
       TRITON_CACHE_DIR=/jit-cache/triton TILELANG_CACHE_DIR=/jit-cache/tilelang \
       DG_JIT_CACHE_DIR=/jit-cache/deep_gemm FLASHINFER_WORKSPACE_BASE=/jit-cache/flashinfer \
       VLLM_LOGGING_CONFIG_PATH=/scratch/vllm-logging.json \
       VLLM_ENGINE_READY_TIMEOUT_S=3600 VLLM_USE_RUST_FRONTEND=$([[ $FRONTEND == rust ]] && echo 1 || echo 0)
container=(
  --container-image="$SHARED/images/vllm-openai-$TAG.sqsh" --no-container-mount-home
  --container-mounts="$HUB:/hf-hub,$JIT:/jit-cache,$JOBTMP:/scratch"
  --container-env=HF_HUB_CACHE,TMPDIR,HOME,VLLM_CACHE_ROOT,TRITON_CACHE_DIR,TILELANG_CACHE_DIR,DG_JIT_CACHE_DIR,FLASHINFER_WORKSPACE_BASE,VLLM_API_KEY
)

log "starting vLLM on :8100 (weight load ~3.5 min, then compile + CUDA graphs) and vllm-router on :8000"
# Tuned for throughput with long-context headroom (unchanged from Anton's script).
srun --overlap "${container[@]}" vllm serve "$MODEL" --port 8100 \
  --served-model-name "$MODEL" deepseek-v4.1-flash \
  --data-parallel-size 8 --enable-expert-parallel \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' \
  --attention-config '{"backend":"FLASHMLA_MEGA_ATTN_DSV41","indexer_kv_dtype":"mxfp4","indexer_sparse_logits":true}' \
  --kv-cache-dtype fp8 --engram-config '{"embedding_across_dp":true}' \
  --max-model-len 1048576 --max-num-seqs 384 --max-num-batched-tokens 16384 \
  --long-prefill-token-threshold 8192 --long-prefill-token-threshold-adaptive \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"probabilistic","rejection_sample_method":"block","enable_adaptive_verification":true}' \
  --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41 \
  --mm-encoder-tp-mode data \
  "$@" &

# The router isn't in the image; RUST_LOG=warn: it ignores --log-level.
srun --overlap "${container[@]}" bash -c "pip install -q --no-deps --target /scratch/router $ROUTER orjson &&
  RUST_LOG=warn PYTHONPATH=/scratch/router exec python3 -m vllm_router.launch_router \"\$@\"" _ \
  --host 0.0.0.0 --port 8000 --policy consistent_hash --intra-node-data-parallel-size 8 \
  --worker-urls http://localhost:8100 --worker-startup-timeout-secs 3600 \
  --request-timeout-secs 14400 --disable-retries &

# Readiness with the key (curl reads the header from stdin: the key is never on a command line).
until printf 'Authorization: Bearer %s\n' "$VLLM_API_KEY" | curl -sf -o /dev/null -H @- localhost:8000/v1/models; do
  sleep 10; [[ $(jobs -rp | wc -l) == 2 ]] || { log "ERROR: vLLM or the router exited during startup, see above"; exit 1; }
done
noauth=$(curl -s -o /dev/null -w '%{http_code}' localhost:8000/v1/models || true)
if [[ $noauth == 401 ]]; then log "API key enforced (GET /v1/models without key: 401)"
else log "WARNING: GET /v1/models without key returned $noauth: the key is NOT enforced on this frontend (run 3 ran like this; agents still send it)"; fi
mkdir -p "$HOSTS_DIR"
printf '%s\n' "$(hostname)" > "$HOSTS_DIR/vllm-$IDX.host.tmp" && mv -f "$HOSTS_DIR/vllm-$IDX.host.tmp" "$HOSTS_DIR/vllm-$IDX.host"
log "serving $MODEL at http://$(hostname):8000/v1 (server $IDX; metrics http://$(hostname):8000/metrics)"
others=$(cat "$HOSTS_DIR"/vllm-*.host 2>/dev/null | sort -u | sed 's#.*#http://&:8000/v1#' | paste -sd, -)
log "run.env once both are up: OPENAI_BASE_URLS=$others   (OPENAI_METRICS_URLS may be left out: load_guard derives <base>/metrics)"

# Warmup is done: if this run compiled new kernels, ship the JIT cache back for everyone's next start.
if [[ -n $(find "$JIT" -newer "$JOBTMP/.start" -type f -print -quit) ]]; then
  log "shipping newly compiled JIT kernels to $CACHE/$TAG.pshipped"
  if dir-pship --from "$JIT" "$CACHE/$TAG" >"$JIT.sha"; then log "JIT cache shipped: version $(cat "$JIT.sha")"
  else log "warning: could not ship the JIT cache (serving continues)"; fi
fi
wait -n || true
log "vLLM or the router exited, shutting down"
