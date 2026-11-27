# SmallMath

600 exact-integer tasks, 100 at each difficulty level 1–6. Each agent writes `/app/answer.txt`; the Harbor verifier grades it offline. [Dataset](https://huggingface.co/datasets/HuggingFaceSwarms/SmallMath), pinned at [`57dbf3f7e83aa8c5c980387a2216d7bb87e4e7cc`](https://huggingface.co/datasets/HuggingFaceSwarms/SmallMath/tree/57dbf3f7e83aa8c5c980387a2216d7bb87e4e7cc).

Adapted from [MathArena/smallmath](https://huggingface.co/datasets/MathArena/smallmath), source revision `13ef9727659718de8bcba221cfe8f8a65bf88a99`. The dataset is CC-BY-NC-SA-4.0; retain MathArena attribution and observe its noncommercial and share-alike terms.

## Prerequisites and cost

Install Legion with `uv sync --all-extras`. Set `HF_TOKEN` to a token able to read the dataset, run Jobs, and create and manage a board Space; set `HF_INFERENCE_TOKEN` to a separate token that can access the model endpoint and read the Job's namespace. Never use the infrastructure token as the agents' inference key. See [mini-swarm's token setup](../mini-swarm/README.md#tokens) and [billing](../../docs/billing.md).

Each team run uses four CPU HF Sandboxes and a public board Space; each solo uses one sandbox and no board. Standalone specs also start an `h200` model Job. Both the GPU startup and the run are billed; the loop can run 30 minutes, and server startup can take several minutes. The mini-swarm example estimates $5/hour for `h200`; check current prices and sandbox/endpoint charges before launching. A shared server remains billed until stopped. Use a fresh board and results directory for each run; the board Space persists after the run. No live validation of these specs has been performed.

## Launch

The [Qwen3.8 chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/chat_template.jinja#L47-L50) accepts `xhigh`, `medium`, or `low` for thinking; it rejects `high`. The thinking examples use `xhigh`.

Choose a task, unique run and public board name. Set an absolute results directory accessible to local workers; this preserves the verifier outputs used by the scorer. For Slurm, Legion writes `<run dir>/results` on the shared disk instead. See [environment choices](../README.md#where-agents-run) and [Slurm](../../docs/slurm.md). These commands launch paid resources; run only after budget approval.

```sh
export TASK_ID=smallmath-0015
export RUN_NAME=smallmath-example
export BOARD_SPACE="$RUN_NAME"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
# One standalone run, no thinking:
uv run legion launch experiments/smallmath/hf-jobs-nothink.yaml
# Or, with thinking and reasoning_effort=xhigh (use a fresh RUN_NAME/BOARD_SPACE/RESULTS_DIR):
uv run legion launch experiments/smallmath/hf-jobs-thinking.yaml
uv run legion status "$RUN_NAME"
```

The `hf_jobs` specs start and stop their own model server. To share one server across multiple tasks, start one of those server specs with `legion serve`; use the printed `/v1` URL with `spec.yaml` for each independent run. For thinking, set `REASONING_EFFORT=xhigh`; for the no-thinking server leave it `off`. Select the corresponding server configuration, and give each run a fresh `RUN_NAME`, `BOARD_SPACE`, `RESULTS_DIR`, and `TASK_ID`.

```sh
# BOARD_SPACE and TASK_ID must be set. Choose one server mode:
export SERVER_RUN_NAME="$RUN_NAME"
uv run legion serve experiments/smallmath/hf-jobs-nothink.yaml --timeout 4h
# Or use the thinking server instead:
uv run legion serve experiments/smallmath/hf-jobs-thinking.yaml --timeout 4h
export SERVER_URL='https://<job-id>--8000.hf.jobs/v1'
# Set xhigh for the thinking server, or off for the no-thinking server.
export REASONING_EFFORT=xhigh
uv run legion launch experiments/smallmath/spec.yaml
# After all runs finish, stop the server using the same spec you started:
RUN_NAME="$SERVER_RUN_NAME" uv run legion serve experiments/smallmath/hf-jobs-thinking.yaml --stop
```

`spec.yaml` also accepts a local or Slurm vLLM endpoint via `SERVER_URL`, `MODEL` and `ENV_BACKEND` (default `hf-sandbox`); see [open models](../../docs/open-models.md). The built-in `message_board.*` tools take text as tool arguments, including quotes and backslashes; no external tools directory is required.

## Solo versus team evaluation

The thinking-off comparison uses the same task and model for four independently launched solo runs (`solo.yaml`, N=1, no board) and a separate team run (`spec.yaml`, N=4). Use a shared no-thinking server from `hf-jobs-nothink.yaml` via `legion serve`, or a compatible Slurm/local vLLM server with thinking disabled. The standalone `hf-jobs-solo-nothink.yaml` starts a separate model Job for one solo run; it is more expensive when repeated. Never reuse a results directory: each independent run must have its own `RUN_NAME` and `RESULTS_DIR`. `BOARD_SPACE` is required only for the team. The commands below are examples that incur charges; wait for budget approval.

```sh
# Start the shared no-thinking server as shown above and set SERVER_URL to its /v1 URL.
export REASONING_EFFORT=off
# Run this block four separate times, substituting i=1, 2, 3, 4.
i=1
export RUN_NAME="smallmath-solo-$i"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
uv run legion launch experiments/smallmath/solo.yaml
uv run legion status "$RUN_NAME"
# Wait until this run and its verifier have finished before launching the next one.
# Then launch an independent team run on the same task:
export RUN_NAME="smallmath-team"
export BOARD_SPACE="$RUN_NAME"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
uv run legion launch experiments/smallmath/spec.yaml
```

Pass the four solo result directories and team result directory together to `score.py`, for example using `runs/smallmath-solo-{1,2,3,4}/results runs/smallmath-team/results` in a shell. Every result root is treated as an independent run. For binary answers, `pass@1` is the mean of solo rewards and `pass@k` is the unbiased estimator `1 - C(n-c,k)/C(n,k)`; both `pass@k` and independent majority require at least `k` solos. Independent majority uses the first `k` solos sorted by result-root path and swarm (the chosen roots are included in output); ties score zero. Team `pass@k` is the oracle maximum of the k teammates' rewards and team `maj@k` is their answer vote. `--k` defaults to 4 and must be at least 2. The comparison includes per-task and macro averages; SmallMath also reports levels 1–6. For ProofBench, `pass@1` is mean percent of 7, `pass@k` uses the threshold ≥6/7, and the maximum observed proof grade is also reported. Its team comparison reports mean, max and identical-proof consensus percent of 7. JSONL contains `comparison` and `comparison_summary` records.

Output-token means use the saved Harbor trial's `result.json` field `agent_result.n_output_tokens`; counts indicate how many trials supplied usage. Team totals require all teammates' counts. A missing value is excluded and an empty mean is null; counts cover the selected latest trial, not earlier attempts.

## Scoring

Pass one or more result roots, each containing `<swarm>/<agent>/trials/<trial>/result.json` and its `verifier/` directory. The trial with the latest completion timestamp per agent is used (attempt number breaks timestamp ties); multiple trials with a missing completion timestamp are rejected as ambiguous; each root/swarm/task is a separate team. A missing verifier directory or required grading metadata is an error. The examples above write to `$RESULTS_DIR`.

```sh
uv run python experiments/smallmath/score.py "$RESULTS_DIR" --manifest /path/to/tasks/manifest.json > scores.json
```

Omit `--manifest` to download the pinned dataset manifest. The output contains solo and unique-plurality `maj@N` accuracy overall, by team size, and by difficulty level. Plain signed integer spellings such as `+001` and `1` vote together regardless of correctness, provided the stripped answer is at most 1,000 characters (the verifier's cleaned-answer limit). Longer or non-integer submissions retain their spelling. Missing answers abstain and ties score zero. Empty strata have a null mean.
