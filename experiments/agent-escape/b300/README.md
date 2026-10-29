# Qwen3.8-27B on B300

This directory contains a multi-node serving launcher and a single-node benchmark
for the original BF16 `Qwen/Qwen3.8-27B` checkpoint. Both use vLLM 0.32.0. The
benchmark measures TP1/2/4/8 and 2/4/8 replicas using the native DP frontend; the
multi-node launcher uses independent instances behind an external router.

## Multi-node serving behind one router

`serve-qwen38-27b.slurm` runs eight **independent TP1/DP1 instances per node**, one
per allocated GPU, behind a single cache-aware router. This is separate from the
benchmark launcher below, which still supports its original TP/DP layouts.

Prepare a complete, immutable local snapshot of `Qwen/Qwen3.8-27B` on the shared
filesystem. From the repository root, create a fresh run directory, then submit
**only after reviewing the resource request**:

```bash
mkdir -p /absolute/path/qwen-run
sbatch --nodes=4 --output=/absolute/path/qwen-run/slurm.log \
  experiments/agent-escape/b300/serve-qwen38-27b.slurm \
  /absolute/path/to/model/snapshot /absolute/path/qwen-run
```

This example reserves **four exclusive eight-B300 nodes (32 GPUs)**. Defaults are
one node, 12 hours, 88 CPUs per node, and all host RAM (`--mem=0`). Slurm command-line
options override those defaults. All model replicas use BF16, 32,768-token context,
128 sequences, 8,192 batched tokens, prefix caching, and low reasoning. Override
`MAX_MODEL_LEN`, `MAX_NUM_SEQS`, `MAX_BATCHED_TOKENS`, or `QWEN_IMAGE` through the
submission environment. TP/DP are fixed at one. `QWEN_LAUNCHER` can point to
`qwen_serve.py` when submitting from outside the repository root.

The script freezes its Python helper and non-secret configuration in the run
directory. Each node stages weights once onto `/scratch/$USER/qwen-serving` and
runs instances on ports 8100–8107. vLLM uses the pinned 0.32.0 image; the first node
also installs `vllm-router==0.1.16` into job-local storage (requires package-index
access). Its router on port 8000 registers every instance URL, uses `cache_aware`,
and disables request retries. Clients use the router, not individual instances.

After **all** requested instances and the router pass readiness within one hour
(including staging), `endpoint.json` contains the router `base_url`, model ID,
job ID and each backend's host, GPU assignment, port and URL. Read the credential
from `api-key` (0600) into the client's `OPENAI_API_KEY`; use an `openai_compat`
provider with that base URL and model `Qwen/Qwen3.8-27B`. Inference authentication
is enforced by the backends using the Authorization header forwarded by the
router. Health and operational endpoints are internal-cluster endpoints, not a
public authenticated control plane.

`logs/<host>-gpu-<index>.log` records each instance; supervisor and router logs
are alongside it. `status.json` reports the latest healthy/total instance count.
A failed or unresponsive instance restarts independently with 10–120-second
exponential backoff. Five consecutive unsuccessful starts exhaust its retry
budget; ten continuous healthy minutes reset the counter. Other instances
continue serving. If none recover within ten minutes, the job fails. An in-flight
request on a failed instance may fail; it is not replayed automatically.

Cancel the allocation with `scancel <job_id>`. Cleanup stops every child step and
removes `endpoint.json`; logs, credential and final status remain for inspection.
Node loss reduces capacity and cannot be replaced within this fixed allocation;
`--no-kill` requests survival of the other nodes. Loss of the first node, its
supervisor, or the router ends the service. There is no router failover.

### Validation and later GPU smoke

CPU-only checks (no jobs submitted):

```bash
bash -n experiments/agent-escape/b300/serve-qwen38-27b.slurm
uv run pytest -q tests/test_qwen_slurm_serving.py tests/test_b300_benchmark.py tests/test_slurm_scripts.py
uv run ruff check .
```

The later, separately authorized two-node GPU smoke should verify 16 distinct
GPU assignments/ports, authenticated streaming and required-tool requests through
the router, rejection of missing/wrong keys, and traffic in multiple backend
logs. Terminate one instance and confirm the other 15 keep serving while it
restarts and rejoins. Finally cancel the allocation and verify all steps exit
and `endpoint.json` disappears. The CPU tests do not establish GPU memory fit,
container compatibility or measured throughput. A separate CPU-only check with
`vllm-router==0.1.16` and two fake HTTP backends passed startup, authentication
forwarding, unhealthy-backend exclusion, and recovered-backend routing.

## Load the router with real prompts

`router_load.py` sends a bounded **5,000-request** concurrency ramp through an
existing router. It does not submit jobs. Its default workload is the prepared,
revision-pinned UltraChat dataset below; `--workload agent_escape` replays the
prepared real agent contexts instead. File and prompt hashes are checked before
requests begin.

```bash
uv run python experiments/agent-escape/b300/router_load.py \
  --server-run /absolute/path/qwen-run \
  --data .flock/b300-qwen/data \
  --requests 5000 --concurrency 16,32,64,128,256,512,1024 \
  --output /absolute/path/load-5000
```

The server directory supplies `endpoint.json`, `config.json`, and the private
`api-key`; credentials are never command-line arguments or result artifacts.
The client waits up to one hour for readiness. The output directory must be new.
Default request allocation is 128/128/128/256/512/1024/2824 across those seven
levels, totaling exactly 5,000 with at least two full waves per level. To test a
higher concurrency, increase the request budget to at least twice the sum of the
requested levels. For a small check, use `--requests 16 --concurrency 8`.

Prompts are shuffled deterministically, reused verbatim after exhausting the
selected dataset, and retain their real content/tool schemas. Prefix caching
stays enabled: this is a repeated-workload test, not a cold-cache test. Responses
stop naturally at EOS; the only output cap is remaining context. Timeouts (900 s
by default), HTTP failures, malformed streams/tools, and context exhaustion are
recorded separately, never counted as successful natural completions.

Each level writes `requests.jsonl`, sampled backend `metrics.jsonl`, and
`summary.json`. Aggregate summaries include valid output tokens/s, requests/s,
p50/p95 latency, TTFT/TPOT, errors, and observed client concurrency. Per-instance
metrics include running/waiting requests, generated/prompt token totals, and
KV/prefix-cache counters. Counter deltas show throughput during active serving,
separately from the end-to-end summary that includes draining long responses. Content
and chunk gaps are omitted by default to keep the load driver light; use
`--save-responses` to retain them. The client raises its own soft file-descriptor
limit when possible and refuses a ramp that exceeds its hard limit.

`completion.json` appears only after all levels finish. It reports the best
measured throughput, context-limit counts, transport/protocol failures, and the
largest tested transport-clean concurrency, **not a proven hard limit**. A response
ending at the context limit is not by itself a concurrency failure. Compare
throughput plateaus, latency and queue growth to identify
saturation. Inference interruptions remain visible as errors and are not retried.
The script never cancels the serving allocation.

## Prepare and launch

From the flock repository, prepare the dataset once (requires model/data network access):

```bash
uv run --with datasets==5.1.0 --with transformers==5.19.0 \
  python experiments/agent-escape/b300/prepare.py \
  --output .flock/b300-qwen/data \
  --cache /fsx/edward/.cache/huggingface/hub \
  --escape-summary .flock/agent-escape-attempts/communication-sweep-20261028T205651Z/summary.json

uv run --no-sync python experiments/agent-escape/b300/submit.py --data .flock/b300-qwen/data
```

The submitter freezes code and dataset copies inside the printed run directory before
calling `sbatch`. It submits one 12-hour allocation on `hopper-b300`; no configurations
run concurrently. Logs and results are under `.flock/b300-qwen/runs/<timestamp>/`.
The pinned image defaults to `/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh`.
Override `QWEN_IMAGE` only with a compatible image; its checksum is recorded.
Model files are staged to node-local `/scratch/$USER/qwen-b300` and reused.

For a serving endpoint instead of the sweep:

```bash
uv run --no-sync python experiments/agent-escape/b300/submit.py --data .flock/b300-qwen/data \
  --mode serve --layout tp --gpus 1
```

The endpoint appears in `measurements/endpoint.json` after smoke checks; the private
credential is in `api-key` (mode 600). Set `LOCAL_KEY` from that file and replace
`providers.local.base_url` in `../spec-local-qwen27b.yaml` with the advertised URL.
The serving job also reserves the full node and stops at its 12-hour time limit.
Cancel with `scancel <job_id>`; cleanup terminates the server process group.
The existing hosted-provider YAML is unchanged.

## Real prompts and EOS

The public dataset is [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k),
`test_sft`: deterministic seed-42 shuffled row indices, first user message verbatim,
512 measured prompts plus eight disjoint warmup prompts. Prompts above 8,192 tokens
are excluded, not truncated. The manifest pins the dataset/model/tokenizer revisions,
row IDs, hashes and preparation versions.

AgentEscape uses 100 reconstructed request prefixes (20 per difficulty) from real
saved runs, with original system instructions and tool schemas. These are reconstructed
from public session events, **not exact HTTP captures**: eligible prefixes precede
recorded compaction and fit 16,384 tokens. Generated tools are validated against their
schemas but never executed. Replay does not measure puzzle success.

Both workloads use temperature 0, seed 42 and low reasoning. Preparation verifies
that the checkpoint's template distinguishes low from xhigh; requests explicitly
pass low to both the API and template. Text and required-tool streaming/nonstreaming
smokes gate each configuration. See the [official vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B).

**EOS is honored.** There is no `ignore_eos`, minimum output length, or fixed completion
length. `max_tokens` is the unused portion of the 32,768-token context, measured by the
server tokenizer including tools. A 15-minute request timeout and the allocation's
wall clock are safety limits. Context exhaustion, timeout, interrupted streams and
missing usage are explicitly incomplete, never counted as natural completions.

## Measurements

- UltraChat: concurrency 1/4/16/32/64/128, `max(32, 4 * concurrency)` prompts,
  three repetitions per point; prefix caching disabled.
- AgentEscape: all 100 requests at concurrency 4/16, three repetitions; prefix caching
  enabled and reset before each repetition. Cache metrics are captured before/after.
- Warmup requests and server startup are excluded from throughput timing. All layouts
  use the same prompts/order, sampling, precision, 128 sequences per replica, and
  8,192 batched-token limit. No speculative decoding or quantization is enabled.

`measurements/summary.csv` and `summary.jsonl` are updated after every case, along with
`throughput.png`. Per-request JSONL contains response content, actual token usage,
finish reason, latency and tool validation. Reported throughput counts valid natural
completions over the entire case duration, including time spent on failed requests.
Raw observed token totals also retain reported usage from incomplete requests.
Reasoning tokens are already part of output tokens and are not added twice.

TTFT includes the first reasoning/content/tool delta. TPOT is average decode time per
output token; SSE chunks may contain multiple tokens, so saved chunk gaps are **not**
claimed as exact per-token interarrival times. Report p50/p95 TTFT, TPOT, latency and
output lengths alongside output/total tokens/s, requests/s, tokens/s/GPU and errors.
`gpu.csv` samples utilization, memory and power; runtime, command, image, topology and
Slurm manifests document reproducibility. Active GPU-hours include startup/cleanup;
full allocated GPU-hours are recorded separately. Slurm `sacct` is authoritative for
allocation billing, including model staging.

There are 168 measured cases in a complete sweep. Configurations that cannot start or
pass smoke checks get `failure.json`; later configurations are still attempted.
`completion.json` records coverage when the sweep exits normally. Walltime interruption
can leave partial request logs; missing cases must be reported, never extrapolated.
Full completion is not promised within 12 hours when natural responses are long.

## Validation

```bash
uv run pytest -q tests/test_b300_benchmark.py
uv run ruff check .
```

Tests fake HTTP streaming and subprocesses; they launch no GPU workloads. They cover
EOS safety caps, reasoning token accounting, streamed tool fragments, malformed tools,
HTTP errors, missing usage, timeouts, GPU topology selection and process cleanup.

To compare completed measurements (including a partially completed sweep):

```bash
uv run --no-sync python experiments/agent-escape/b300/analyze.py \
  .flock/b300-qwen/runs/<timestamp>/measurements
```

The report ranks measured throughput and four-agent latency only where three
repetitions completed without errors, and explicitly states missing coverage.

## Deadline-bounded screening

`quick.py` reuses the same request client and EOS behavior, but first measures all
seven GPU layouts on 32 fixed UltraChat prompts at concurrency 4 and 32, once each.
Only after the full GPU comparison does it replay 20 agent contexts (four per
 difficulty) at concurrency 4. Each server gets one disjoint warmup prompt once.
There are 21 cases; reports explicitly permit one repetition and are screening
results, not peak-throughput or variance estimates. No output-length cap is added.

Run inside the same prepared GPU container as `benchmark.py`:

```bash
python3 /code/quick.py --data /data --output /results/measurements \
  --model-path /model --deadline-epoch <absolute-Unix-deadline> \
  --prior /prior/measurements
```

`--prior` is optional. Reuse requires matching model/data revisions, sampling and
context settings, plus matching request IDs/hashes and successful completions.
Each server configuration has a ten-minute total budget, and the absolute deadline
reserves cleanup time. Timed-out configurations retain partial request logs but are
excluded from throughput summaries. The final report states measured coverage.
For a live handoff, retain the current allocation, stop the old driver and its
inference processes, and run the quick sweep as a replacement step. Preserve the
old source snapshot and result directory; use a new output/source snapshot.

On vLLM 0.32, replay benchmark servers set `VLLM_SERVER_DEV_MODE=1` to expose the
cache-reset management endpoint. Normal serving and UltraChat stages leave that
mode disabled. Omitting it produces a 404 before replay measurement starts.
