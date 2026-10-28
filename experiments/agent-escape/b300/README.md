# Qwen3.8-27B on B300

Serve the original BF16 `Qwen/Qwen3.8-27B` checkpoint, or measure TP1/2/4/8 and
2/4/8 independent TP1 replicas sequentially on one exclusive eight-B300 node.
Replica routing uses vLLM's native data-parallel frontend; it is pinned with vLLM
0.32.0 rather than installing a separate proxy. TP1 is the one-replica baseline.

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
