# v8-harbor

The v8 run (`experiments/v8`: same six lanes, prompts, loop and board rules) on HF Jobs, with the world
and its sandbox taken from a task in `HuggingFaceSwarms/swarmchat-harbor` (chatbot4, `chatbot4-v8/`)
instead of flock's `worlds/chatbot4`. This folder holds only the run's `spec.yaml` and `prompts/`; the
world code lives in the task. `prompts/` equals `experiments/v8/prompts` except `SwarmChat-v8` ->
`SwarmChat-v8h` in `system.md` (2 lines).

Differences from v8 (see the comments in `spec.yaml`): `compute: hf-jobs`, the HF router as the model server,
a per-run board Space, `world.url` (the world is not started by flock), one agent per worker job, all ramps
at 1 until `<LANE>_MAX` is set, and a per-agent cost cap.

## Tokens

Four fine-grained tokens; agents run next to the worker, so each holds as little as possible.

| token | env | scope |
|---|---|---|
| router | `HF_ROUTER_TOKEN` (the provider key; the only one agents hold) | Inference Providers only |
| workers | `FLOCK_WORKER_HF_TOKEN` (sent to workers as their `HF_TOKEN`) | write on the agents' bucket `v8h-test` only |
| world | the world job's `HF_TOKEN` | write on dataset `SwarmChat-v8h` and the world prefix |
| board | `BOARD_BUCKET_TOKEN` | the board snapshot bucket |

The operator's `HF_TOKEN` (starts jobs) stays with `flock` and the controller. The controller's state and the
flock wheel live in a separate bucket, `v8h-ctl`, which the workers' token cannot write.

## Small run on HF Jobs

1. Make the buckets `v8h-test` (agents) and `v8h-ctl` (wheel and controller state), a dataset repo
   `SwarmChat-v8h` seeded as v8's, and the tokens above.
2. Publish the wheel to `v8h-ctl` (`FLOCK_WHEEL_URL`; the controller state defaults to the same bucket).
   Build the agent image (`Dockerfile.worker`) and the world image from the task; start the world job
   (and its sandbox job) in the run's network group (`FLOCK_NETWORK_GROUP`), and set `WORLD_URL` to
   `http://<$HF_NETWORK_GROUP_PREFIX>world:7860`.
3. `flock init RUN_DIR --from experiments/v8-harbor`, with `RUN_BUCKET`, `WORLD_URL`, `AGENT_IMAGE`,
   `BOARD_TEMPLATE`, `RUN_NAME` and `RUN_FOR` in the environment.
4. Export `HF_ROUTER_TOKEN` and `FLOCK_WORKER_HF_TOKEN`, then `flock launch`. Stop with `flock stop --release`.
   `flock launch` refuses a spec that would send the controller's `HF_TOKEN` to workers.
