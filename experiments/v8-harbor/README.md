# v8-harbor

The v8 run (`experiments/v8`: same six lanes, prompts, loop and board rules) on HF Jobs, with the world
and its sandbox taken from a task in `HuggingFaceSwarms/swarmchat-harbor` (chatbot4, `chatbot4-v8/`)
instead of flock's `worlds/chatbot4`. This folder holds only the run's `spec.yaml` and `prompts/`; the
world code lives in the task. `prompts/` equals `experiments/v8/prompts` except `SwarmChat-v8` ->
`SwarmChat-v8h` in `system.md` (2 lines).

Differences from v8 (see the comments in `spec.yaml`): `compute: hf-jobs`, the HF router as the model server,
a per-run board Space, `world.url` (the world is not started by flock), one agent per worker job, all ramps
at 1 until `<LANE>_MAX` is set, and a per-agent cost cap.

## Small run on HF Jobs

1. Make a throwaway bucket, a dataset repo `SwarmChat-v8h` seeded as v8's, and three fine-grained tokens:
   world, workers (inference, bucket, board read) and board replica.
2. Build the agent image (`Dockerfile.worker`) and the world image from the task; start the world job
   (and its sandbox job) in the run's network group (`FLOCK_NETWORK_GROUP`), and set `WORLD_URL` to
   `http://<$HF_NETWORK_GROUP_PREFIX>world:7860`.
3. `flock init RUN_DIR --from experiments/v8-harbor`, with `RUN_BUCKET`, `WORLD_URL`, `AGENT_IMAGE`,
   `BOARD_TEMPLATE`, `RUN_NAME` and `RUN_FOR` in the environment.
4. Export `FLOCK_WORKER_HF_TOKEN` (the workers' narrow token; `HF_TOKEN` stays the controller's
   job-launching one), then `flock launch`. Stop with `flock stop --release`.
