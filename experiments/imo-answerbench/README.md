# IMO-AnswerBench

400 short-answer tasks. Each agent writes `/app/answer.txt`; the Harbor verifier grades it offline. [Dataset](https://huggingface.co/datasets/HuggingFaceSwarms/IMO-AnswerBench), pinned at [`40d9d407c8b5e4b62c6b2047e21c807214f2a068`](https://huggingface.co/datasets/HuggingFaceSwarms/IMO-AnswerBench/tree/40d9d407c8b5e4b62c6b2047e21c807214f2a068).

Copyright 2025 Google LLC and the IMO-Bench authors. Upstream software is Apache-2.0; problem statements and answers are CC-BY-4.0. See the [upstream license](https://github.com/google-deepmind/superhuman/blob/main/imobench/README.md#license-and-disclaimer).

## Network backend

The pinned Harbor tasks require `network_mode = "allowlist"` for both the agent and verifier. Until HF Sandbox enforces allowlists ([huggingface/legion#140](https://github.com/huggingface/legion/issues/140)), use the URL specs with their default `podman` backend on Slurm or local compute, or set `ENV_BACKEND=docker` for local compute. The standalone `hf-jobs-*.yaml` examples currently select `hf-sandbox` and cannot enforce this task requirement; use the URL specs with a compatible backend and a reachable model server instead.

## Prerequisites and cost

Install Legion with `uv sync --all-extras`. Set `HF_TOKEN` to a token able to read the dataset, run Jobs, and create and manage a board Space; set `HF_INFERENCE_TOKEN` to a separate token that can access the model endpoint and read the Job's namespace. Never use the infrastructure token as the agents' inference key. See [mini-swarm's token setup](../mini-swarm/README.md#tokens) and [billing](../../docs/billing.md).

Each team run uses four task environments and a board; each solo uses one environment and no board. The HF Sandbox variants would use CPU sandboxes and a public board Space, but do not enforce the task allowlists described above. Standalone specs also start an `h200` model Job. Both the GPU startup and the run are billed; the team loop can run 30 minutes, and server startup can take several minutes. The mini-swarm example estimates $5/hour for `h200`; check current prices and sandbox/endpoint charges before launching. A shared server remains billed until stopped. Use a fresh board and results directory for each run; the board Space persists after the run. These revised specs have not yet been validated in a live run.

## Judge and cluster notes

For AnswerBench and ProofBench scoring, the default open-weight judge is `moonshotai/Kimi-K3.1-Pro` on the HF Router (`https://router.huggingface.co/v1`, `HF_INFERENCE_TOKEN`); SmallMath itself uses its offline verifier. For a Gemini reference judge, use `--provider gemini` (default model `gemini-3.1-pro-preview`) with `GEMINI_API_KEY` or, if it is unset or empty, `GOOGLE_API_KEY`. Phase 1 cluster validation reported agreement with Gemini on 60/60 ProofBench and 86/86 AnswerBench cases, and the best open-weight MAE on human-graded IMO-GradingBench.

The Phase 1 runs also found that Pi's default `max_tokens=16384` can truncate long thinking turns; override the limit at the model proxy or server where supported. Solo and standalone thinking variants allow 30 minutes per turn; thinking-off team variants retain six minutes. For a shared thinking server, set `TURN_TIMEOUT=30m` in the URL team spec. When invoking `localrun` directly under heavy concurrency, provide a generous verifier drain grace, for example `--grace 600` (the option is in seconds; `--grace 300` means 300 seconds). The current default is 330 seconds, so do not lower it when a larger margin is needed.

## Launch

The [Qwen3.8 chat template](https://huggingface.co/Qwen/Qwen3.8-27B/blob/37561816e66b5b9153b1b64c10968c18c1e6f4a0/chat_template.jinja#L47-L50) accepts `xhigh`, `medium`, or `low` for thinking; it rejects `high`. The thinking examples use `xhigh`.

Choose a task and a unique run name. The standalone team variants also require a unique public board name; the URL team spec uses an in-worker board. Set an absolute results directory accessible to local workers; this preserves the verifier outputs used by the scorer. For Slurm, Legion writes `<run dir>/results` on the shared disk instead. See [environment choices](../README.md#where-agents-run) and [Slurm](../../docs/slurm.md). These commands launch paid resources; run only after budget approval.

```sh
export TASK_ID=imo-bench-algebra-001
export RUN_NAME=imo-answerbench-example
export BOARD_SPACE="$RUN_NAME"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
# One standalone run, no thinking:
uv run legion launch experiments/imo-answerbench/hf-jobs-nothink.yaml
# Or, with thinking and reasoning_effort=xhigh (use a fresh RUN_NAME/BOARD_SPACE/RESULTS_DIR):
uv run legion launch experiments/imo-answerbench/hf-jobs-thinking.yaml
uv run legion status "$RUN_NAME"
```

The `hf_jobs` specs start and stop their own model server. The standalone thinking spec declares a 90-minute run timeout; compilation gives the server a 135-minute timeout including startup and margin, though Legion cancels it when the run ends. Budget for the longer possible lifetime; `legion serve --timeout` sets a separate shared-server lifetime. To share one server across multiple tasks, start one of those server specs with `legion serve`; use the printed `/v1` URL with `spec.yaml` for each independent run. For thinking, set `REASONING_EFFORT=xhigh` and `TURN_TIMEOUT=30m`; for the no-thinking server use `REASONING_EFFORT=off` and the default `TURN_TIMEOUT=6m`. Select the corresponding server configuration, and give each run a fresh `RUN_NAME`, `RESULTS_DIR`, and `TASK_ID`. The URL team spec uses a local board, and the solo spec uses none; neither requires `BOARD_SPACE`. The server spec still requires it to load, although `legion serve` does not deploy the board.

```sh
# BOARD_SPACE and TASK_ID must be set. Choose one server mode:
export SERVER_RUN_NAME="$RUN_NAME"
uv run legion serve experiments/imo-answerbench/hf-jobs-nothink.yaml --timeout 4h
# Or use the thinking server instead:
uv run legion serve experiments/imo-answerbench/hf-jobs-thinking.yaml --timeout 4h
export SERVER_URL='https://<job-id>--8000.hf.jobs/v1'
# For thinking, set both; for no-thinking use off and unset TURN_TIMEOUT (6m).
export REASONING_EFFORT=xhigh
export TURN_TIMEOUT=30m
uv run legion launch experiments/imo-answerbench/spec.yaml
# After all runs finish, stop the server using the same spec you started:
RUN_NAME="$SERVER_RUN_NAME" uv run legion serve experiments/imo-answerbench/hf-jobs-thinking.yaml --stop
```

`spec.yaml` and `solo.yaml` also accept a local or Slurm vLLM endpoint via `SERVER_URL`, `MODEL` and `ENV_BACKEND` (default `podman`); see [open models](../../docs/open-models.md). The team URL spec runs one local board on its worker and applies the AoPS message rules. The built-in `message_board.*` tools take text as tool arguments, including quotes and backslashes; no external tools directory is required.

## Solo versus team evaluation

Solo specs have no loop: they wait for one explicit kickoff prompt and are not automatically re-prompted. Wait for that turn to finish, then drain the worker so Harbor runs the verifier. A local no-loop run does not stop itself; ending it also stops a standalone model Job. Team specs start a loop, wake on board messages and send a final submission reminder at the end. Their per-agent output-token cap is 32,768, including reasoning tokens.

The thinking-off comparison uses the same task and model for four independently launched solo runs (`solo.yaml`, N=1, no board) and a separate team run (`spec.yaml`, N=4). Use a shared no-thinking server from `hf-jobs-nothink.yaml` via `legion serve`, or a compatible Slurm/local vLLM server with thinking disabled. The standalone `hf-jobs-solo-nothink.yaml` starts a separate model Job for one solo run; it is more expensive when repeated. Never reuse a results directory: each independent run must have its own `RUN_NAME` and `RESULTS_DIR`. `BOARD_SPACE` is not needed for the URL team or solo specs; it is required for the standalone HF Jobs team specs. The commands below are examples that incur charges; wait for budget approval.

```sh
# Start the shared no-thinking server as shown above and set SERVER_URL to its /v1 URL.
export REASONING_EFFORT=off
unset TURN_TIMEOUT  # restore the team URL spec's 6m no-thinking default
# Run this block four separate times, substituting i=1, 2, 3, 4.
i=1
export RUN_NAME="imo-answerbench-solo-$i"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
uv run legion launch experiments/imo-answerbench/solo.yaml
# Wait until agent-001 appears in status, then send exactly one kickoff.
uv run legion status "$RUN_NAME"
uv run legion prompt --swarm "$RUN_NAME" --type agent "Solve the task and write your final submission to /app/answer.txt."
# Wait until the turn has finished and the agent is idle; do not send another prompt.
uv run legion status "$RUN_NAME"
# For local compute, drain the worker and run the verifier before the next run:
uv run legion stop "$RUN_NAME" --delete --yes
# On Slurm, use --release to drain and preserve the run as described in docs/slurm.md.
# Then launch an independent team run on the same task:
export RUN_NAME="imo-answerbench-team"
export RESULTS_DIR="$PWD/runs/$RUN_NAME/results"
uv run legion launch experiments/imo-answerbench/spec.yaml
```

Pass the four solo result directories and team result directory together to `score.py`, for example using `runs/imo-answerbench-solo-{1,2,3,4}/results runs/imo-answerbench-team/results` in a shell. Every result root is treated as an independent run. For binary answers, `pass@1` is the mean of solo rewards and `pass@k` is the unbiased estimator `1 - C(n-c,k)/C(n,k)`; both `pass@k` and independent majority require at least `k` solos. Independent majority uses the first `k` solos sorted by result-root path and swarm (the chosen roots are included in output); ties score zero. Team `pass@k` is the oracle maximum of the k teammates' rewards and team `maj@k` is their answer vote. `--k` defaults to 4 and must be at least 2. The comparison includes per-task and macro averages; SmallMath also reports levels 1–6. For ProofBench, `pass@1` is mean percent of 7, `pass@k` uses the threshold ≥6/7, and the maximum observed proof grade is also reported. Its team comparison reports mean, max and identical-proof consensus percent of 7. JSONL contains `comparison` and `comparison_summary` records.

Output-token means use the saved Harbor trial's `result.json` field `agent_result.n_output_tokens`; counts indicate how many trials supplied usage. Team totals require all teammates' counts. A missing value is excluded and an empty mean is null; counts cover the selected latest trial, not earlier attempts.

## Scoring

Pass one or more result roots, each containing `<swarm>/<agent>/trials/<trial>/result.json` and its `verifier/` directory. The trial with the latest completion timestamp per agent is used (attempt number breaks timestamp ties); multiple trials with a missing completion timestamp are rejected as ambiguous; each root/swarm/task is a separate team. A missing verifier directory or required grading metadata is an error. The examples above write to `$RESULTS_DIR`.

```sh
uv run python experiments/imo-answerbench/score.py "$RESULTS_DIR" > scores.json
# Optional Stage 2: single-pass QED-Nano judge for rejected structured/literal answers.
uv run python experiments/imo-answerbench/score.py "$RESULTS_DIR" --judge > scores-judged.json
# To compare with Gemini, set GEMINI_API_KEY or GOOGLE_API_KEY and use:
uv run python experiments/imo-answerbench/score.py "$RESULTS_DIR" --judge \
  --provider gemini --judge-model gemini-3.1-pro-preview > scores-gemini.json
```

Stage 2 uses the verbatim [QED-Nano AnswerBench prompt](https://github.com/CMU-AIRe/QED-Nano/blob/02a4699ed380f8e980a8d44c755b8bfd4e310718/eval/configs/prompts/answerbench.txt) in a single pass, with the candidate as the student answer and the reference as the gold answer. It parses the final brace-matched `\boxed{Correct}` or `\boxed{Incorrect}` verdict; missing or malformed verdicts abort. The script downloads the reference and problem statement at the pinned dataset revision. For offline inputs, supply `--reference-dir` pointing to a dataset checkout or a directory containing both `<task-id>.json` and `<task-id>.md` files. Use `--provider gemini --judge-model gemini-3.1-pro-preview` with `GEMINI_API_KEY` or `GOOGLE_API_KEY` for the native Gemini API; OpenAI-compatible endpoints use `--judge-base-url` and `--judge-api-key-env`, including `HF_TOKEN` or `HF_INFERENCE_TOKEN` for the HF Router. Number answers stay with the offline verifier. The output reports individual accuracy, unique-plurality `maj@N` and oracle `best-of-N`, including breakdowns by team size. Voting keeps the submitted text, including leading whitespace and evaluation sections, while ignoring terminal line breaks; whitespace-only answers abstain. The QED judge receives a separate copy with self-evaluation removed once. This preserves distinctions relevant to both the offline verifier and QED extraction. It does not infer mathematical equivalence between differently written submissions. Missing answers abstain and a tie scores zero; best-of-N is an oracle, not a deployable selection policy.
