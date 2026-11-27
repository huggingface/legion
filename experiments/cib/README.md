# Collaborative Intelligence Benchmark (CIB)

One Pi agent is evaluated as a teammate in four dimensions of
[Stanford's Collaborative Intelligence Benchmark](https://github.com/zou-group/collaborative-intelligence).
Each Harbor task of the
[dataset](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark)
runs the other teammates, the judges and a model gateway (`model-compat`) in its own containers.
`spec.yaml` runs one task (a case) with `legion launch`.

## Setup

- Linux, rootless Podman 4+ (the spec's default backend), Python 3.12+, uv, Git and Git LFS.
  With dockerd, set `BACKEND=docker`.
- Read access to the private dataset through Git: `hf auth login --add-to-git-credential`.
- On an HF cluster CPU node, use Legion’s [Podman setup](../../docs/slurm.md).
- Inference is paid, including the partner and judge calls of team episodes.

### Prerequisites

The dataset's runtime tarballs use Git LFS; install Git LFS and fetch the objects before
running. Set the evaluated model's key and readout key in separate exports, so the latter
receives the actual key value:

```sh
export GEMINI_API_KEY=...   # plus the partner and judge keys below
export CI_PRIVATE_READOUT_API_KEY="$GEMINI_API_KEY"
```

Set `CI_PARTNER_QWEN14_API_KEY` too: the task validates it even for Sharing. Credentials stay
in the shell that runs `legion launch`; local workers inherit them and compose passes them to
the task's services. The spec only names them.

| Variable | Purpose |
|---|---|
| the evaluated model's key, e.g. `GEMINI_API_KEY` | Pi's calls through the gateway (`MODEL_KEY_ENV` names it) |
| `CI_PRIVATE_READOUT_API_KEY` | the same key, for the team's private final-answer call |
| `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` | partner models and judges |
| `HARNESS_VLLM_ENDPOINTS` + `HARNESS_VLLM_API_KEY`, or `HARNESS_OPENAI_COMPAT` + `CI_PARTNER_QWEN14_API_KEY` | the Qwen3-14B partner ([dataset README](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark#setup)) |
| `ANTHROPIC_WORKSPACE_ID` | only if your Anthropic key requires it |

## Run one case

```sh
export GEMINI_API_KEY=...   # plus the team's keys
export CI_PRIVATE_READOUT_API_KEY="$GEMINI_API_KEY"
uv run legion launch experiments/cib/spec.yaml
uv run legion status cib
```

`CASE` picks the task (default: one Sharing solo). The run sends the task's instruction to the
agent once, ends after that turn and grades it: the verifier's result is
`.legion/<run>/agents/_harbor/agent-001/trials/*/verifier/result.json`. The model's endpoint,
`http://model-compat:8003`, exists only inside the Trial, so the spec sets
`inference.preflight: false` and the controller reports its probe as unverified. Use the case
driver below to score a single run against its Legion host and Harbor Trial evidence.

## Other models

Set the variables for one command, as the spec reads them:

| Model | Variables |
|---|---|
| Gemini 3.8 Flash (default) | `GEMINI_API_KEY` |
| GPT-6.1 Sol, Responses API | `MODEL=gpt-6.1-sol MODEL_KEY_ENV=OPENAI_API_KEY API_FORMAT=openai_responses CI_MODEL_UPSTREAM=https://api.openai.com/v1 CI_PRIVATE_READOUT_API=openai-responses` |
| Claude Opus 5.5, Messages API | `MODEL=claude-opus-5-5 MODEL_KEY_ENV=ANTHROPIC_API_KEY API_FORMAT=anthropic_messages GATEWAY_URL=http://model-compat:8003 CI_MODEL_UPSTREAM=https://api.anthropic.com/v1 CI_PRIVATE_READOUT_API=anthropic-messages` (add `CI_MODEL_HEADERS='{"anthropic-workspace-id": "..."}'` if your key needs it) |
| DeepSeek-V4.1-Flash on the HF router, thinking off | `MODEL=deepseek-ai/DeepSeek-V4.1-Flash:together MODEL_KEY_ENV=HF_INFERENCE_TOKEN EFFORT=off CI_MODEL_UPSTREAM=https://router.huggingface.co/v1 CI_MODEL_BODY='{"reasoning": {"enabled": false}}' CI_PRIVATE_READOUT_API=openai-compat-chat` (optional `CI_MODEL_HEADERS='{"X-HF-Bill-To": "<org>"}'`) |
| Another OpenAI-compatible endpoint | `MODEL=<id> MODEL_KEY_ENV=<key var> CI_MODEL_UPSTREAM=https://<endpoint>/v1 CI_PRIVATE_READOUT_API=openai-compat-chat` |

`CI_PRIVATE_READOUT_API_KEY` is always the evaluated model's key. `CI_MODEL_HEADERS` and
`CI_MODEL_BODY` reach only the evaluated model's calls, never the partners' or judges'.

## A whole dimension

The benchmark runs each scenario's solo first; a passing solo unlocks three team seeds, and the
dataset's scorer aggregates them. Until `legion sweep` supports gated schedules, `parity.py`
runs one dimension that way, one `legion launch` per case in its own folder, and writes
`scores.json` with the pinned dataset's scorer. It needs a clone of the dataset at the spec's
revision and supports Sharing (`manifest.json`); the
other dimensions use aliases and reader grading and are not supported by this driver.

```sh
git clone https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence-benchmark ../cib-harbor
git -C ../cib-harbor checkout 070e8bf4264f0d88d88d0286e2772fb497f147ba
git -C ../cib-harbor lfs pull
uv run --with ../cib-harbor/adapter python experiments/cib/parity.py --dataset ../cib-harbor \
  --output results/cib/sharing --concurrency 2
```

To run and score just the default solo case with incomplete coverage allowed (`parity.py`
sets `allow_incomplete` for `--case`):

```sh
uv run --with ../cib-harbor/adapter python experiments/cib/parity.py --dataset ../cib-harbor \
  --output results/cib/single --case ci-sharing-h2-r-choosing-the-safe-field-station-solo-s0
```

Running the same configuration again reuses admitted cases and waits for live local controllers.
Unavailable attempts are retained when the driver starts a retry. Use a fresh output directory
for a new model, gateway configuration or manifest. If a sparse dataset checkout leaves Git LFS
pointer files in place of task archives, include the root `.gitattributes` and fetch the archives
before launching.

For a small Sharing validation on a cluster CPU node with Podman configured, create a
one-scenario manifest (the solo and its three team seeds), then run the same driver. It runs
all solo cells first and starts team seeds only for a passing solo:

```sh
python - <<'PY'
import json
from pathlib import Path

root = Path('../cib-harbor')
manifest = json.loads((root / 'manifest.json').read_text())
scenario = 'h2_r_choosing_the_safe_field_station'
manifest['tasks'] = [t for t in manifest['tasks'] if t['identity']['scenario_id'] == scenario]
(root / 'manifest-sharing-small.json').write_text(json.dumps(manifest, indent=2) + '\n')
PY
uv run --with ../cib-harbor/adapter python experiments/cib/parity.py --dataset ../cib-harbor \
  --manifest manifest-sharing-small.json --output results/cib/sharing-small --concurrency 2
```

This uses local controllers on the node where the command runs; keep that allocation alive until
the driver exits. Results in `scores.json` are partial benchmark coverage, not a full Sharing score.
If invoking the dataset scorer directly against the full manifest for a single-scenario run,
pass `--allow-incomplete`; the filtered manifest above instead describes only that scenario.
