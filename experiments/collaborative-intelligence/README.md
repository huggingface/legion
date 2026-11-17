# Collaborative Intelligence

One Pi agent is evaluated as a teammate in
[Stanford's Collaborative Intelligence Benchmark](https://github.com/zou-group/collaborative-intelligence).
The task's team service runs the other teammates, private final-answer calls and judges separately
from the evaluated agent.

**Needs:** this Flock checkout, Linux with Docker and Compose (or rootless Podman), and Hugging Face
Git access to the [dataset](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence).
Set `GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and the original Qwen3-14B partner
endpoint/credential as described in the [dataset's setup](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence#setup).
Inference is paid, including partner and judge calls.
For rootless Podman, follow the [dataset's runtime setup](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence/blob/main/adapter/README.md#podman).

**Launch one Gemini solo episode:**

```sh
uv run flock launch experiments/collaborative-intelligence/spec.yaml
uv run flock status ci-fullpi-gemini
```

The run sends the task instruction once, grades the episode and ends automatically.
Raw Trial and verifier evidence is retained under `results/cib/`.
Keep one evaluated agent: the task supplies its teammates. Select another episode with
`task.selector`. The task revision, 256 Pi turns and one-hour guard are pinned in the spec.

This uses Flock's controller and worker directly; no benchmark adapter installation is needed.
The dataset's [native model specs](https://huggingface.co/datasets/HuggingFaceSwarms/collaborative-intelligence/tree/feat%2Fnative-flock-launch/flock)
provide the same setup for other models.
A single episode's reward is not the complete benchmark score: the latter needs its solo eligibility
schedule, three team seeds and dimension aggregation.
