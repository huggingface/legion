# antbench

Ten agents each control one ant of an [AntBench](https://huggingface.co/spaces/HuggingFaceSwarms/AntBench)
world (private Space): together the ants must carry a load through two walls. The world runs in
real time and does not wait for the agents; they talk only through AntBench's short text signals
(`ANT_COMMS=text`), seen by nearby ants. One episode per world job: about 7 minutes of world
time once every ant is ready.

Each agent is a Trial of the Harbor task `antbench-ant` (`task/tasks/antbench-ant`) in its own
E2B sandbox (`python:3.12-slim`), with one worker and the controller on HF Jobs
(`compute: hf-jobs`). Agents never hold an ant's credential or a key: their sandboxes call the
worker's gateway, which adds each agent's own ant credential (docs/world.md, "Env worlds").
One trusted CPU job runs AntBench on port 8000 in the run's network group, with no exposed
port, so only the workers reach it. `agent-001` plays ant 0, `agent-002` ant 1, and so on.
Each credential is derived from `ANTBENCH_ENV_SECRET` and the agent's name, by the gateway and
by the world job's launcher (`python -m legion.harbor.env`) alike. The transcript stays in the
world job; at the episode's end it logs AntBench's result and copies the transcript to
`$BUCKET_ID` under `antbench/<RUN_NAME>/`. The job script is `world-job.sh`; the job ends when
AntBench exits.

The world's clock starts once all ten ants called `ready`. The world job readies the ants still
missing 300 s after the first one did (once): sandboxes start together once the worker has
built the template (once, behind a lock), so the spread is Pi's install and the first model
turns, a minute or three. The clock is stopped while waiting, so a longer wait costs only idle
tokens; a seat filled in stands still until its agent's first command, which may still come.

**Needs**

- `HF_TOKEN`: Jobs permission and bucket create/write access in your personal account, read
  access to the task dataset and the AntBench Space (it starts the jobs; the world job mounts
  the Space with it).
- `HF_INFERENCE_TOKEN`: inference only; every sandbox holds it.
- `E2B_API_KEY`: your E2B key; the worker starts the sandboxes with it.
- `ANTBENCH_ENV_SECRET`: the run's secret, new for each run (`openssl rand -hex 24`); the
  worker and the world job get it as a job secret.
- `ANTBENCH_TASK_REPO`, `ANTBENCH_TASK_REVISION`: the dataset `task/` is uploaded to
  (`HuggingFaceSwarms/antbench`), and that commit. The dataset is private: if `legion launch`
  cannot clone it, give git your Hub login (`hf auth login --add-to-git-credential`).
- `BUCKET_ID` (`<your-username>/<name>`: the world job mounts its `antbench/` folder);
  optional `RUN_NAME`. `legion init` writes every `${VAR}` of the spec
  into `run.env`.

The spec omits `namespace`: jobs use `HF_TOKEN`'s personal account and router calls the
inference key's default billing account ([billing and storage](../../docs/billing.md)). The
world job must run in the same namespace as the swarm (`hf jobs run --namespace <ns>` if the
spec sets one): a network group belongs to one namespace. AntBench gets one ant per agent
(`agents.count`); the world job counts the credentials the launcher hands it.

**Cost** (estimates, for one episode and about an hour of wall time):

- Tokens: about $15-40. Each agent calls a tool every few seconds for ~7 minutes of play plus
  the wait, ~1k tokens per observation, with a context that grows each call.
  `limits.budget_usd` (50) stops the run. After
  `done`, each extra loop round is a short turn per agent: `loop.while.max_rounds` (8) bounds
  them; tune it after the first run.
- E2B: 10 sandboxes of 1 vCPU and 1 GB for about an hour, under $1 at E2B's list price.
- HF Jobs: the world job and the worker on `cpu-basic`, the controller on `cpu-upgrade`, plus a
  small hourly rate per exposed port (the worker's gateway): well under $1.

**Launch**

Use an `hf` CLI whose `hf jobs run` has `--network-group` and `--network-alias`
(huggingface_hub 2.5 has both).

```sh
# 1. the task, once per change: note the commit SHA it prints
hf upload HuggingFaceSwarms/antbench experiments/antbench/task . --repo-type dataset
# 2. legion's wheel from this checkout, and the run folder; fill in run.env
W=$(uv run scripts/publish-wheel.sh "hf://buckets/<your-username>/legion/dist")
uv run legion init $PWD/runs/antbench-1 --from experiments/antbench
# 3. the world job, in the run's network group as `antbench` (the spec's upstream); nothing public
set -a; . runs/antbench-1/run.env; set +a
export LEGION_NETWORK_GROUP=antbench-1
hf jobs run --detach --name antbench-world --flavor cpu-basic --timeout 3h \
  --network-group "$LEGION_NETWORK_GROUP" --network-alias antbench \
  --secrets ANTBENCH_ENV_SECRET -e RUN_NAME="${RUN_NAME:-antbench-1}" \
  -v hf://spaces/HuggingFaceSwarms/AntBench@5c3ce1ee2827cbbb4a8eede633c375020a62645c:/antbench \
  -v "${W%/*}:/wheel:ro" -v runs/antbench-1/experiment:/spec \
  -v "hf://buckets/$BUCKET_ID/antbench:/out:rw" \
  python:3.12-slim bash /spec/world-job.sh
# 4. the swarm, in the same group: a controller job starts too (docs/deploy-job.md)
export LEGION_RUN_DIR=$PWD/runs/antbench-1 LEGION_WHEEL_URL=$W
uv run legion launch runs/antbench-1/experiment
```

The `/spec` folder is synced to your `jobs-artifacts` bucket first (`hf jobs run -v`). Calls made
before AntBench is up answer `upstream_unreachable` and agents retry; start the world job first
so it is installed by the time the sandboxes are.

**Watch**

- `uv run legion status antbench-1`: agents, turns, cost.
- `hf jobs logs <job id>`: a `deadline: ready filled in ...` line names the ants whose agents
  missed the start (ant `i` is agent `i+1`); `antbench result: {...}` is the episode's outcome (`success`, `ticks`, `timing`).
  The transcript lands in `$BUCKET_ID` under `antbench/<RUN_NAME>/episode.jsonl`; replay it
  with `python ant_service.py replay episode.jsonl` in the AntBench Space.

**Stop**

```sh
uv run legion stop antbench-1 --release   # ends the Trials (and sandboxes), then the controller
hf jobs cancel <job id>                   # the world job
```

Another episode needs a new world job (a new `run_id` and `ANTBENCH_ENV_SECRET`) and a new run.
