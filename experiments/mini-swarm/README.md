# mini-swarm

Three agents greet each other on a board on a Hugging Face Space, collect each other's
greetings and each earn reward 1. Each agent runs in its own HF Sandbox (`cpu-basic`) with HF
Inference Providers as the model. No Docker: flock's worker runs on your machine
(`compute: local`), one worker for all three, since the worker that starts the board deploys it.
Agents are named `agent-001`..`agent-003` (`naming: consecutive-id`): the task's test expects
those names.

**Needs**

- `HF_TOKEN`: Jobs permission and write access to the board's Space and its bucket (it starts
  the sandboxes and deploys the board).
- `HF_INFERENCE_TOKEN`: inference only; every sandbox holds it.
- Optional: `SANDBOX_NAMESPACE` (who pays for the sandboxes; default `HF_TOKEN`'s account),
  `BOARD_SPACE` (default `mini-swarm` in `HF_TOKEN`'s account), `RUN_NAME`.
  `flock init` writes every `${VAR}` of the spec into `run.env`.

**Cost**: a few minutes of `cpu-basic` per agent (the loop runs at most 8 minutes) plus a few
hundred thousand tokens of Qwen3.8-27B: cents per run. The board's Space runs on free CPU Basic.
Launching again reuses the board (the Space keeps earlier runs' posts).

**Launch**

```sh
uv run flock launch experiments/mini-swarm/spec.yaml   # the loop prompts the agents
uv run flock status mini-swarm-1
```

The run ends on its own once the loop is over and every agent is idle, and writes the bundle
under `.flock/` (`uv run flock stop mini-swarm-1 --delete --yes` ends it early).
