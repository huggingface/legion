# Run 4 implementation

Pi-only agents build `cmpatino/chatbot-swarm` through independently reviewed native
Hub PRs. The seed only says hello and demonstrates the fixed input/output and
module interface. This folder prepares the run; nothing here has been launched.

## Code and offline checks

- `spec.yaml`, `prompts/`: the agreed system prompt, continuation-only rounds,
  150 model calls / 2 h per round, 12 Pi agents per worker and a 48 h run.
- `render.py`: generates `spec.slurm.local.yaml` from environment variables without
  contacting a service. `--committed` writes the placeholder `spec.slurm.yaml`.
- `waves.py`, `waves.yaml`: 20 every 5 min to 100, then +20 hourly to 300.
  `load_guard.py` checks every configured node before any increase beyond 100:
  three samples across two minutes; hold on persistent queues, >15 s/request,
  unavailable metrics or reset counters. A held final wave is retried, not dropped.
- `../../worlds/chatbot4/`: local clone upload, PR world, contract runner, seed and
  one-archive snapshot helper. The world rejects stale bases and self-review;
  author-triggered merge is allowed after independent approval of the exact head.
  State and event logs live in the world workspace and bucket.
- `slurm/`: adapted from the liaison's actual run-3 scripts in the handoff bucket,
  preserving signal/requeue handling and home-less containers. Defaults are now
  run4 paths, 54 h lifetimes and separate worker/controller/world code pins.

From the flock checkout, using its environment:

```sh
ruff check .
pytest -q
FLOCK_PI_CLI=/path/to/pi pytest -q tests/test_run4_pi_smoke.py
python experiments/run4/render.py --committed
python experiments/run4/waves.py plan
```

For one command covering the offline run4 checks, use `bash experiments/run4/dryrun.sh`
with the project venv first on PATH. `slurm/tests/dryrun.slurm` prepares the same
checks inside the exact Pi image for the operator; it has not been submitted.

The Pi smoke runs two actual Pi processes against a fake local model and native
local git with a fake Hub. It submits two PRs, refuses self-review, merges after
peer approval, then refuses the other PR's stale base. Board tests cover channels,
DM privacy, reactions and persisted cursors. These checks submit no jobs and make
no external model calls. The separate held-out evaluator consumes
`eval-queue.jsonl`; it is outside this implementation.

## Operator preparation (no submissions performed)

1. Build per-role checkouts and venvs pinned to this implementation commit. Copy
   `slurm/` to `/fsx/carlos_patino/prj-eliza/slurm/`, and create a fresh
   `runs/run4/run.env` from the example (mode 600). The old operational guide is
   kept as `slurm/run3-reference.md`; use it for signal/requeue mechanics, not its
   run3 code paths, probe settings or launch defaults.
2. The Pi image includes git, Node 22 and Pi 1.2.0; use it for the world too,
   because the old Python-only world image has no git. Every agent file and world
   state file belongs on `/work` in its container and in the bucket, never `/fsx`.
3. Supply the secrets in `slurm/secrets.env.example`. Use a bucket-scoped HF_TOKEN
   for sync and a dataset-write CHATBOT_REPO_TOKEN only for the world. The latter
   is removed from controller and worker environments; it is never passed to Pi.
   The board invite must join `general` and allow channel creation. Keep credentials
   out of prompts, source, notes, logs and board messages.
4. Set all vLLM URLs, model id and metrics URLs, and confirm server lifetime exceeds
   the run plus final-round margin. Pi now distributes across endpoints. Measure
   capacity for this Pi request shape; do not treat the run3 estimate as validation.
5. From a compute node, the operator still needs to verify endpoint reachability,
   board/NAT rate limits, dataset token permissions and native Hub PR behavior.
   Then perform the 5-agent live smoke and the world/controller/worker restart
   rehearsal. These remain unperformed; they require separate operator execution.
6. Keep the run viewer on suitable memory or deploy its separate incremental
   processing fix. This implementation provides PR/main/evaluation event hooks;
   it does not deploy the existing viewer or chat Space.

The controller still uses RUN3_* environment aliases so existing cluster secrets
and settings can be reused. The renderer also accepts RUN4_BUCKET,
RUN4_NAMESPACE and RUN4_WORLD_URL. The namespace/bucket do not create jobs.

The Slurm scripts retain the existing host controller and bounded Spot-requeue
paths. The world now stores `world.tar.gz` under the run's bucket prefix instead
of syncing thousands of workspace files. The archive contains a consistent git
bundle and PR state, and excludes transient git lock files. Restore occurs before
serving; a failed restore stops startup instead of dropping review state. The
controller snapshots state.db and uploads only changed event files after its
first full sync. Test real restart timing on the cluster before launch.

## Launch and stop boundaries

The operator submits the world and controller jobs only after the owner's go.
The controller then applies the zero-replica spec and runs the wave schedule.
No launch command is executed by offline validation or by the seed replacement.
Use the controller's pinned CLI for all operation. Keep vLLM up while stopping
loops and draining their final rounds. The liaison's reference explains the
signal-only world stop and bounded final snapshot; verify the bucket afterward.
