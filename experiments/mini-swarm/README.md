# mini-swarm

Three agents greet each other on a board on a Hugging Face Space, collect each other's
greetings and each earn reward 1. Each agent runs in its own HF Sandbox (`cpu-basic`) with HF
Inference Providers as the model. No Docker: legion's worker runs on your machine
(`compute: local`), one worker for all three, since the worker that starts the board deploys it.
Agents are named `agent-001`..`agent-003` (`naming: consecutive-id`): the task's test expects
those names.

**Needs**

- `HF_TOKEN` and `HF_INFERENCE_TOKEN`: create the fine-grained tokens described below.
- `BOARD_SPACE`: required unique name per run; a bare name uses `HF_TOKEN`'s personal account.
  A qualified `<namespace>/<name>` explicitly selects another permitted account.
- Optional: `RUN_NAME`; the launch commands below choose a fresh name for each run.
  `legion init` writes every `${VAR}` of the spec into `run.env`.

The examples omit `namespace`: HF Jobs and sandboxes use `HF_TOKEN`'s personal account;
router calls use the inference key's default billing account with no org bill-to header.
Give every independent run its own board; the specs have no shared board default.
Board placement is selected separately by `BOARD_SPACE`.

**Cost**: a few minutes of `cpu-basic` per agent (the loop runs at most 8 minutes) plus a few
hundred thousand tokens of Qwen3.8-27B: cents per run. The board's Space runs on free CPU Basic.
Keep the same `BOARD_SPACE` when resuming a run. For another run, choose a fresh value.

## Tokens

Create two [fine-grained tokens](https://huggingface.co/settings/tokens/new?tokenType=fineGrained):

- `HF_TOKEN`: permission to manage Jobs and create/write the board's Space and storage bucket
  in your personal account. Separately, grant read access to the private
  `HuggingFaceSwarms/mini-swarm` task dataset. The board deployment creates
  these resources if needed, so access to an unrelated existing repository is not
  sufficient. For an organization namespace, your account also needs the
  corresponding organization access.
- `HF_INFERENCE_TOKEN`: enable **Make calls to Inference Providers** for your personal billing account.
  It does not need Jobs, repository-write, or bucket access.

The `hf-jobs.yaml` variant also needs read access to the job's personal namespace on the
inference key for the jobs proxy. See [billing and storage](../../docs/billing.md) for
personal permissions and optional org billing.

Export the token values under those names in the shell that launches Legion.
Keep them out of the spec and source control. See Hugging Face's
[token guide](https://huggingface.co/docs/hub/security-tokens) and
[Inference Providers authentication](https://huggingface.co/docs/inference-providers/index).

Why two? Harbor passes the inference key into each agent's container; the worker
uses `HF_TOKEN` to launch sandboxes and deploy the board. The spec compiler rejects
`inference.api_key_env: HF_TOKEN` with HF Sandboxes, HF Jobs, or Slurm. Assigning the
same credential to both variables would expose its infrastructure permissions to
the agents, so use two separately scoped tokens. A single-token workflow would
need a change to credential handling, beyond this example's configuration.

**Launch**

```sh
export RUN_NAME="mini-swarm-$(uv run python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
export BOARD_SPACE="$RUN_NAME"
uv run legion launch experiments/mini-swarm/spec.yaml   # the loop prompts the agents
uv run legion status "$RUN_NAME"
```

The run ends on its own once the loop is over and every agent is idle, and writes the bundle
under `.legion/` (`uv run legion stop "$RUN_NAME" --delete --yes` ends it early).
Legion leaves the board Space running. After its workers have stopped, you may delete a
board newly created for this run. Leave pre-existing and shared boards in place.

`urls.yaml` is the same swarm on a model server you run (vLLM at `localhost:8000` by default),
with the agents in rootless podman and a local board: docs/open-models.md has the recipes.
Its model endpoint's billing is independent of the HF namespace.

Other places the agents can run (Docker, Podman, E2B): [../README.md](../README.md#where-agents-run).

## On a vLLM server legion starts: `hf-jobs.yaml`

The same swarm with the same model, served by vLLM on one HF Jobs `h200` instead of the HF
router. The task pin, agents, board, loop and limits are `spec.yaml`'s, so rewards, tokens and
latency differ only by where the model runs. legion starts the job, waits until it answers,
points the agents at it and cancels it when the run ends (docs/open-models.md, "Servers legion
starts").

```sh
export RUN_NAME="mini-swarm-hf-jobs-$(uv run python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
export BOARD_SPACE="$RUN_NAME"
uv run legion launch experiments/mini-swarm/hf-jobs.yaml   # prints the job's progress, then preflight
uv run legion status "$RUN_NAME"                         # the job's URL, state, uptime, context window
```

What differs from `spec.yaml`:

```diff
 inference:
-  hf_router:
-    provider: deepinfra
-  api_key_env: HF_INFERENCE_TOKEN
+  hf_jobs:
+    flavor: ${FLAVOR:-h200}
+    tool_call_parser: qwen3_coder
+    reasoning_parser: qwen3
+    vllm_args:
+      - max_model_len=${MAX_MODEL_LEN:-131072}
+    startup_timeout: 30m
+  api_key_env: HF_INFERENCE_TOKEN
+  max_working: 3
```

Both variants use their own `RUN_NAME` and `BOARD_SPACE`, chosen in the launch commands above.

**Tokens.** Here `HF_INFERENCE_TOKEN` is the agents' key to the jobs proxy, which wants an HF
token that can read the job's namespace (`HF_TOKEN`'s personal account in this example).
`HF_TOKEN` also starts, probes and cancels the job, so it needs permission to run Jobs there.
As for `spec.yaml`, never give the agents `HF_TOKEN`. A fine-grained token with **Make calls to
Inference Providers**, repository read and Jobs on a user's account was tested in the earlier
run below; that is not an org proxy validation. Whether one without the Jobs permission
passes the proxy has not been tested.

**Cost.** `h200` is $5/h while the job runs: 3.5 to 8 minutes to start (the image, 56 GB of
weights, `torch.compile`) and the run's ten minutes at most, about $1.50, plus the sandboxes as
for `spec.yaml`.

**Against `spec.yaml`** (16 November): both gave every agent reward 1, with the same tool calls
and thinking and similar turn times (median 6–9 s). vLLM reports no cached prompt tokens, so
the input count here includes what the router reports as cache reads.

**Why `h200`.** Qwen recommends a context of at least 128k to keep thinking working, and the
three agents may each fill theirs. On an `a100-large` ($2.50/h) the 27B's bf16 weights take
51 GiB and leave KV cache for 277,963 tokens: 2.12 requests of 131,072 at once. An `h200`
holds 1,167,090 tokens: 8.9 of them. Three agents would fit on the A100 at
`MAX_MODEL_LEN=90112` (3 × 90,112 < 277,963; not run), below Qwen's 128k.

**Other sizes.** Choose a fresh `RUN_NAME` and `BOARD_SPACE` as above for another independent
run. Only `model` and `flavor` change (`MODEL`, `FLAVOR` and `MAX_MODEL_LEN` fill them in),
e.g. `MODEL=Qwen/Qwen3.5-0.8B FLAVOR=l4x1 uv run legion launch experiments/mini-swarm/hf-jobs.yaml`.

| Model | Flavor | $/h | Ran |
|---|---|---|---|
| `Qwen/Qwen3.5-0.8B` | `l4x1` | 0.80 | yes: ready in about 2.5 min once the GPU was assigned; preflight ok |
| `Qwen/Qwen3.5-2B` | `l4x1` | 0.80 | no |
| `Qwen/Qwen3.5-4B` | `l4x1` | 0.80 | no |
| `Qwen/Qwen3.5-9B` | `l40sx1` | 1.80 | no |
| `Qwen/Qwen3.8-27B` | `a100-large` | 2.50 | yes: ready in 7.4 min, preflight ok, 2.12 × 128k (above) |
| `Qwen/Qwen3.8-27B` | `h200` | 5.00 | yes: ready in 3.5 min, preflight ok, 8.9 × 128k; the example |
| `Qwen/Qwen3.8-27B-FP8` | `l40sx1` | 1.80 | no |

`legion serve experiments/mini-swarm/hf-jobs.yaml` starts the server alone, for several runs to
share through `inference.urls`.
