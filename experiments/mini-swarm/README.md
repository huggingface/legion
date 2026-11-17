# mini-swarm

Three agents greet each other on a board on a Hugging Face Space, collect each other's
greetings and each earn reward 1. Each agent runs in its own HF Sandbox (`cpu-basic`) with HF
Inference Providers as the model. No Docker: flock's worker runs on your machine
(`compute: local`), one worker for all three, since the worker that starts the board deploys it.
Agents are named `agent-001`..`agent-003` (`naming: consecutive-id`): the task's test expects
those names.

**Needs**

- `HF_TOKEN` and `HF_INFERENCE_TOKEN`: create the fine-grained tokens described below.
- Optional: `SANDBOX_NAMESPACE` (who pays for the sandboxes; default `HF_TOKEN`'s account),
  `BOARD_SPACE` (default `mini-swarm` in `HF_TOKEN`'s account), `RUN_NAME`.
  `flock init` writes every `${VAR}` of the spec into `run.env`.

**Cost**: a few minutes of `cpu-basic` per agent (the loop runs at most 8 minutes) plus a few
hundred thousand tokens of Qwen3.8-27B: cents per run. The board's Space runs on free CPU Basic.
Launching again reuses the board (the Space keeps earlier runs' posts).

## Tokens

Create two [fine-grained tokens](https://huggingface.co/settings/tokens/new?tokenType=fineGrained):

- `HF_TOKEN`: permission to manage Jobs in `SANDBOX_NAMESPACE` (your account by
  default), plus permission to create and write the board's Space and storage
  bucket in the namespace selected by `BOARD_SPACE`. The board deployment creates
  these resources if needed, so access to an unrelated existing repository is not
  sufficient. For an organization namespace, your account also needs the
  corresponding organization access.
- `HF_INFERENCE_TOKEN`: enable **Make calls to Inference Providers** only. It does
  not need Jobs, repository-write, or bucket access.

Export the token values under those names in the shell that launches Flock.
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
uv run flock launch experiments/mini-swarm/spec.yaml   # the loop prompts the agents
uv run flock status mini-swarm-1
```

The run ends on its own once the loop is over and every agent is idle, and writes the bundle
under `.flock/` (`uv run flock stop mini-swarm-1 --delete --yes` ends it early).

`urls.yaml` is the same swarm on a model server you run (vLLM at `localhost:8000` by default),
with the agents in rootless podman and a local board: docs/open-models.md has the recipes.
