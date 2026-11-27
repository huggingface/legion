# IMO-AnswerBench

400 short-answer tasks. Each agent writes `/app/answer.txt`; the Harbor verifier grades it offline. [Dataset](https://huggingface.co/datasets/HuggingFaceSwarms/IMO-AnswerBench), pinned at [`40d9d407c8b5e4b62c6b2047e21c807214f2a068`](https://huggingface.co/datasets/HuggingFaceSwarms/IMO-AnswerBench/tree/40d9d407c8b5e4b62c6b2047e21c807214f2a068).

Copyright 2025 Google LLC and the IMO-Bench authors. Upstream software is Apache-2.0; problem statements and answers are CC-BY-4.0. See the [upstream license](https://github.com/google-deepmind/superhuman/blob/main/imobench/README.md#license-and-disclaimer).

## Prerequisites and cost

Install Legion with `uv sync --all-extras`. Set `HF_TOKEN` to a token able to read the dataset, run Jobs, and create and manage a board Space; set `HF_INFERENCE_TOKEN` to a separate token that can access the model endpoint and read the Job's namespace. Never use the infrastructure token as the agents' inference key. See [mini-swarm's token setup](../mini-swarm/README.md#tokens) and [billing](../../docs/billing.md).

Each run uses four CPU HF Sandboxes, a public board Space and, in the standalone specs, an `h200` model Job. Both the GPU startup and the run are billed; the loop can run 30 minutes, and server startup can take several minutes. The mini-swarm example estimates $5/hour for `h200`; check current prices and sandbox/endpoint charges before launching. A shared server remains billed until stopped. Use a fresh board and results directory for each run; the board Space persists after the run. No live validation of these specs has been performed.

## Launch

Choose a task, unique run and public board name. Set an absolute results directory accessible to local workers; this preserves the verifier outputs used by the scorer. For Slurm, Legion writes `<run dir>/results` on the shared disk instead. See [environment choices](../README.md#where-agents-run) and [Slurm](../../docs/slurm.md). These commands launch paid resources; run only after budget approval.

```sh
export TASK_ID=imo-bench-algebra-001
export RUN_NAME=imo-answerbench-example
export BOARD_SPACE="$RUN_NAME"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
# One standalone run, no thinking:
uv run legion launch experiments/imo-answerbench/hf-jobs-nothink.yaml
# Or, with thinking and reasoning_effort=high (use a fresh RUN_NAME/BOARD_SPACE/RESULTS_DIR):
uv run legion launch experiments/imo-answerbench/hf-jobs-thinking.yaml
uv run legion status "$RUN_NAME"
```

The `hf_jobs` specs start and stop their own model server. To share one server across multiple tasks, start one of those server specs with `legion serve`; use the printed `/v1` URL with `spec.yaml` for each independent run. For thinking, set `REASONING_EFFORT=high`; for the no-thinking server leave it `off`. Select the corresponding server configuration, and give each run a fresh `RUN_NAME`, `BOARD_SPACE`, `RESULTS_DIR`, and `TASK_ID`.

```sh
# BOARD_SPACE and TASK_ID must be set. Choose one server mode:
export SERVER_RUN_NAME="$RUN_NAME"
uv run legion serve experiments/imo-answerbench/hf-jobs-nothink.yaml --timeout 4h
# Or use the thinking server instead:
uv run legion serve experiments/imo-answerbench/hf-jobs-thinking.yaml --timeout 4h
export SERVER_URL='https://<job-id>--8000.hf.jobs/v1'
# Set high for the thinking server, or off for the no-thinking server.
export REASONING_EFFORT=high
uv run legion launch experiments/imo-answerbench/spec.yaml
# After all runs finish, stop the server using the same spec you started:
RUN_NAME="$SERVER_RUN_NAME" uv run legion serve experiments/imo-answerbench/hf-jobs-thinking.yaml --stop
```

`spec.yaml` also accepts a local or Slurm vLLM endpoint via `SERVER_URL`, `MODEL` and `ENV_BACKEND` (default `hf-sandbox`); see [open models](../../docs/open-models.md). The built-in `message_board.*` tools take text as tool arguments, including quotes and backslashes; no external tools directory is required.

## Scoring

Pass one or more result roots, each containing `<swarm>/<agent>/trials/<trial>/result.json` and its `verifier/` directory. The latest numbered trial per agent is used; each root/swarm/task is a separate team. A missing verifier directory or required grading metadata is an error. The examples above write to `$RESULTS_DIR`.

```sh
uv run python experiments/imo-answerbench/score.py "$RESULTS_DIR" > scores.json
# Optional Stage 2: dual-order LLM judge for rejected structured/literal answers.
uv run python experiments/imo-answerbench/score.py "$RESULTS_DIR" --judge \
  --judge-model YOUR_JUDGE_MODEL --judge-base-url https://router.huggingface.co/v1 \
  --judge-api-key-env HF_INFERENCE_TOKEN > scores-judged.json
```

The script downloads reference files at the pinned revision when Stage 2 is enabled. For offline references, supply `--reference-dir` pointing to a dataset checkout or a directory of `<task-id>.json` files. Both orderings must be judged equivalent; model errors abort rather than count as correct. Number answers stay with the offline verifier. The output reports individual accuracy, unique-plurality `maj@N` and oracle `best-of-N`, including breakdowns by team size. Voting compares NFC-normalized whitespace only; it does not infer mathematical equivalence between differently written submissions. Missing answers abstain and a tie scores zero; best-of-N is an oracle, not a deployable selection policy.
