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
  Three consecutive holds print `ACTION REQUIRED` and persist the streak across
  restarts. `RUN4_HOLD_WARN_AFTER` and `RUN4_MAX_REQUEST_SECONDS` are configurable;
  calibrate the latter on the live Pi smoke before growing past 100.
- `../../worlds/chatbot4/`: local clone upload, PR world, contract runner, seed and
  one-archive snapshot helper. The world rejects stale bases and self-review;
  author-triggered merge is allowed after independent approval of the exact head.
  State and event logs live in the world workspace and bucket.
- `flock.isolation`: Pi and bot subprocesses use user/PID/mount namespaces,
  private `/proc`, explicit file mounts, dropped capabilities and scrubbed
  environments. Bots also have a separate network namespace and read-only root.
  Worker/world processes disable dumpability; Pi cannot see sibling tool pins.
  Setup fails closed. Language audit hooks are an additional restriction.
- `slurm/`: adapted from the liaison's actual run-3 scripts in the handoff bucket,
  preserving signal/requeue handling and home-less containers. Defaults are now
  run4 paths, 60 h world/controller lifetimes (54 h workers) and separate
  worker/controller/world code pins. The world reserves 28 CPUs / 54G, with
  `CHATBOT_RUN_SLOTS=26` and two CPUs left for the service and snapshots.

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
It sets `FLOCK_REQUIRE_ISOLATION=1`: kernel isolation tests must pass, including
private process visibility, sibling-pin denial, live pin updates, read-only bot
files, network denial and an isolated real Pi against a local fake model. On a
host that blocks `unshare`, ordinary pytest skips those tests; that is not cluster
validation. Run4 Slurm scripts always enable process isolation.

The Pi smoke runs two actual Pi processes against a fake local model and native
local git with a fake Hub. It submits two PRs, refuses self-review, merges after
peer approval, then refuses the other PR's stale base. Board tests cover channels,
DM privacy, reactions and persisted cursors. These checks submit no jobs and make
no external model calls. The separate held-out evaluator consumes
`eval-queue.jsonl`; it is outside this implementation.

## Operator preparation (no submissions performed)

1. Prepare per-role checkouts and venvs pinned to a full implementation commit:

   ```sh
   bash experiments/run4/prepare_roles.sh FULL_COMMIT_SHA
   bash experiments/run4/prepare_roles.sh FULL_COMMIT_SHA --apply
   ```

   The first command prints the plan; the second prepares code and environments,
   including pytest and ruff. Neither submits jobs. Existing role checkouts must
   be clean and at the requested commit; the helper refuses to overwrite them.
   Copy `slurm/` to `/fsx/carlos_patino/prj-eliza/slurm/`, and create a fresh
   `runs/run4/run.env` from the example (mode 600). The old operational guide is
   kept as `slurm/run3-reference.md`; use it for signal/requeue mechanics, not its
   run3 code paths, probe settings or launch defaults.
2. The Pi image needs git, Node 22, Pi 1.2.0 and util-linux `unshare`; use it for the world too,
   because the old Python-only world image has no git. Every agent file and world
   state file belongs on `/work` in its container and in the bucket, never `/fsx`.
3. Supply the secrets in `slurm/secrets.env.example`. Use **one `HF_TOKEN`** for
   read/write access to the experiment bucket and `cmpatino/chatbot-swarm` only.
   There is no separate dataset token. Check its fine-grained grants in the Hub;
   metadata checks cannot prove write permission or exclude grants on other repos.
   The controller and workers also hold this token and its dataset-write access.
   Pi's allow-listed environment omits it. Private process/filesystem views and
   disabled dumpability protect the trusted worker/world processes; validate that
   boundary inside the actual image with `slurm/tests/dryrun.slurm`. Narrow token
   grants still limit damage if a credential is exposed.
   The board invite must join `general` and allow channel creation. Keep credentials
   out of prompts, source, notes, logs and board messages.
4. Set all vLLM URLs, model id and metrics URLs. Reserve **at least two actual model
   nodes** for 300 Pi agents; two URL aliases do not establish two nodes. Confirm
   remaining server lifetime covers the 48 h run, up to 2 h of final rounds and
   startup waits; use a 60 h reservation. No vLLM job is supplied or launched by
   this folder. Pi distributes across endpoints. Measure
   capacity for this Pi request shape; do not treat the run3 estimate as validation.
5. With the worker environment first on PATH, run the read-only preparation checks:

   ```sh
   set -a
   . /fsx/carlos_patino/prj-eliza/secrets/secrets.env
   . /fsx/carlos_patino/prj-eliza/runs/run4/run.env
   set +a
   python experiments/run4/preflight.py
   python experiments/run4/preflight.py --online
   ```

   Offline checks inspect paths, executables and secret presence without printing
   values. `--online` reads model metadata, metrics, board reachability and Hub
   metadata; it makes no inference calls or writes. Run it from a compute node.
   The operator still needs to verify board/NAT rate limits, token write grants
   and the native Hub PR create/update/review/merge lifecycle.
   Then perform the 5-agent live smoke and the world/controller/worker restart
   rehearsal. These remain unperformed; they require separate operator execution.
   During the smoke count HTTP 429s across the shared NAT, including agent tool
   traffic, before going past 100 agents. Extended polling rotates 12 identities
   in at least 60 s, even on a busy feed: at most 5 initial polls/s across
   25 workers, plus retries and explicit tool requests.
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
serving; transient failures retry five times with 30 s backoff. A missing archive
still permits a first start. Exhausted restore failures stop startup without
dropping review state. Abnormal world exits requeue up to five Slurm restarts
unless stopping. The controller stops waiting for a world after 30 min instead
of waiting forever. `WORLD_RESTORE_ATTEMPTS`, `WORLD_RESTORE_BACKOFF_S` and
`WORLD_WAIT_S` configure these bounds. The controller snapshots state.db and uploads only changed event files after its
first full sync. Test real restart timing on the cluster before launch.

During the smoke and first waves, watch stale-base orphan PRs, PRs left open
after an uncertain Hub merge, PR-lock latency, full-tree upload memory, poller
health and final-push duration. These remain observation items from the review.

## Launch and stop boundaries

The operator submits the world and controller jobs only after the owner's go.
The controller then applies the zero-replica spec and runs the wave schedule.
No launch command is executed by offline validation or by the seed replacement.
Use the controller's pinned CLI for all operation. Keep vLLM up while stopping
loops and draining their final rounds. The liaison's reference explains the
signal-only world stop and bounded final snapshot; verify the bucket afterward.
