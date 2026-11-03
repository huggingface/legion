# run3 on a CPU Slurm cluster

Runbook for the cluster side. Mechanism and spec keys: `docs/slurm.md`. The model (vLLM) is
reached inside the cluster; the board, the Hub and the buckets over the internet. Agent state,
the controller's state, the world's workspace and the bundle go to the HF bucket, never to
`/fsx`. `/fsx` holds the code checkout, the venv or image, `secrets.env` (mode 600), and the
run directory (host files, worker env files, logs).

| file | what |
|---|---|
| `common.sh` | sourced by the three run jobs: run settings, secrets, venv first on PATH, home-less defaults, once-only stop guard |
| `controller.sbatch` | the controller, on the host (bare venv: it calls `sbatch`), `FLOCK_COMPUTE=slurm` |
| `world.sbatch` + `world-run.sh` | the world (worlds/chatbot3), bare or in the container (`SQSH`) |
| `waves.sbatch` | render + apply (once the controller and the world answer) + `waves.py run` |
| `worker.sbatch` | one worker slot, sbatch'ed by the controller (contract in `docs/slurm.md`) |
| `Containerfile` | the worker/world image (`/opt/flock`, `/work`) |
| `Containerfile.pi` | that image + Node 22.23.3 + Pi 1.2.0, for the `agent-pi` workers |
| `run3.env.example`, `secrets.env.example` | the run's settings and the secret names |
| `../dryrun.sh` | one-command rehearsal with the fake model and board |

## Once per cluster

1. Checkout and venv on `/fsx` (Python 3.12, uv):
   ```
   git clone -b run3 https://github.com/huggingface/flock /fsx/run3/flock
   cd /fsx/run3/flock && uv build --wheel
   uv venv --managed-python --python 3.12 /fsx/run3/venv
   uv pip install --python /fsx/run3/venv/bin/python "dist/flock-*.whl[controller]" huggingface_hub==2.2.0
   ```
   The same venv serves the controller, the world, the waves runner and bare workers. Always
   call `$VENV/bin/python -m flock ...` or put `$VENV/bin` first on PATH (the scripts do):
   `/usr/bin/flock` is util-linux.
2. `secrets.env` from `secrets.env.example`, `chmod 600`. The scripts source it; the Slurm
   client never reads or writes a secret (env files and the sbatch line hold names only, and
   sbatch runs with a scrubbed environment).
3. Container image (optional; workers and the world, never the controller):
   ```
   mkdir -p /fsx/run3/image/dist && cp dist/flock-*.whl /fsx/run3/image/dist/
   container-build -f experiments/run3/slurm/Containerfile -t run3-flock:$(git rev-parse --short HEAD) /fsx/run3/image
   enroot import -o /fsx/run3/images/run3-flock-$(git rev-parse --short HEAD).sqsh dockerd://run3-flock:$(git rev-parse --short HEAD)
   ```
   (`container-build` is the cluster's builder; any OCI builder works, then `enroot import`.)
   Then in `run3.env`: `SQSH=<the .sqsh>` (the world), `FLOCK_SLURM_CONTAINER_IMAGE=<the .sqsh>`
   and `FLOCK_SLURM_CONTAINER_ROOT=/work` (the workers' env files).
   In the container everything is under `/work`: a worker's `WORK_ROOT`, `FLOCK_LLM_HOME`,
   `HOME`, `HF_HOME`, `TMPDIR` are `/work/flock/<WORKER_ID>/{work,llm,home,hf,tmp}`, the world's
   workspace `/work/world/ws`; the container's `/tmp` is RAM counted against the job's memory.
   The controller stays on the host (bare venv: it calls `sbatch`). Mounts: `<RUN_DIR>/hosts`
   read-only (workers), the repo and the run dir read-only (world). Secrets arrive as env, sourced before `srun`: **verify once** that they pass into
   the container (`srun --container-image=$SQSH env | cut -d= -f1 | sort`).
   The `agent-pi` type (harness pi) needs Node + Pi: build `Containerfile.pi` FROM that image
   (`--build-arg BASE=run3-flock:<sha>`, tag `run3-flock-pi:<sha>`, `enroot import` it the same
   way) and set `CONTAINER_IMAGE_agent_pi=<the pi .sqsh>` in the secrets file: worker.sbatch
   uses `CONTAINER_IMAGE_<TYPE, - as _>` when set, else `FLOCK_CONTAINER_IMAGE`. A worker's type
   is `TYPE` in its env file (`TYPE=agent-pi`; worker id and job name `<swarm>-agent-pi-wNNNN`).
   Pi refuses to prompt without its key variable: `OPENAI_API_KEY` must be non-empty.
4. Rehearsal inside one job (no model, no board; ~3 min; non-zero on failure):
   `srun -p <partition> -c 4 --mem=8G --time=00:20:00 /fsx/run3/flock/experiments/run3/dryrun.sh`

## A run

```
RUN_DIR=/fsx/run3/runs/<run>; mkdir -p $RUN_DIR/logs
cp run3.env.example $RUN_DIR/run3.env   # edit: RUN, paths, bucket, endpoint, board, variant
cd /fsx/run3/flock/experiments/run3/slurm   # submit from here (or export RUN3_SLURM_DIR)
sbatch -o $RUN_DIR/logs/%x-%j.out controller.sbatch $RUN_DIR/run3.env
sbatch -o $RUN_DIR/logs/%x-%j.out world.sbatch      $RUN_DIR/run3.env
sbatch -o $RUN_DIR/logs/%x-%j.out waves.sbatch      $RUN_DIR/run3.env
```

Order does not matter: each job writes its host file (`<RUN_DIR>/hosts/controller.host`,
`hosts/world.host`, rewritten at every start; `<RUN_DIR>/*.host` are symlinks) once it answers, and `waves.sbatch` polls both (a line a minute, never gives up) before it renders
(`render.py --variant $RUN3_VARIANT --compute slurm`), applies, and starts `waves.py run`.
The waves clock starts at the controller's first answer and is kept next to the spec, so a
requeued runner resumes the same schedule. The controller sbatches every worker
(`worker.sbatch`, 4 CPU / 7G agents, 2 CPU / 3800M interviewers, `--no-requeue`).

The smoke: `RUN3_VARIANT=smoke RUN3_WAVES=waves-smoke.yaml` (30-min loop, `max_rounds` 120,
cap 40). The 48 h run: `RUN3_VARIANT=selfhosted RUN3_WAVES=waves.yaml` (cap from waves.yaml).

**Exactly once per run:** the controller, the world and the waves runner. The three jobs use
`--dependency=singleton` (by job name and user): keep their names. A second controller would
double-launch workers.

Pin the controller and the world to non-Spot nodes if the cluster has them (the commented
`--constraint` line). If not, a requeue moves them: workers re-read `controller.host` after a
connection error (no relaunch), and the controller re-reads `world.host` every 15 s and
republishes the world URL to workers at their next heartbeat (same credential, no drain).

### Watching

```
export FLOCK_URL=http://$(cat $RUN_DIR/controller.host):7860 FLOCK_TOKEN=...   # from secrets.env
$VENV/bin/python -m flock status run3-<CHATBOT_RUN>
$VENV/bin/python -m flock workers run3-<CHATBOT_RUN>     # last_error carries a crash's log tail
squeue --me; sacct -j <id> -o JobID,State,ExitCode,MaxRSS,TotalCPU
tail $RUN_DIR/logs/<worker id>-<job id>.out
```

Worker job states: PENDING for up to ~15 min is normal (autoscaling from 0). A Spot reclaim,
`PREEMPTED`, `NODE_FAIL`, exit 75 and `TIMEOUT` are planned restarts (relaunch at once, no
backoff); a `FAILED` job is a crash (backoff 30 s doubling to 15 min, `crashlooping` after 5).

### Spot budget

After SIGUSR1 the watcher cancels the job ~30 s later, so every stop path fits in ~20 s:
the worker drains in `FLOCK_DRAIN_BUDGET_S` (20 s: turns interrupted, agents synced in
parallel, exit 75); the controller makes one final state push (`FLOCK_FINAL_PUSH_TIMEOUT=15`);
the world stops its server and pushes its workspace once, cut at `WS_STOP_S` (15 s; `best/`,
probes, log and verdicts are mirrored at each event anyway). Signals may arrive twice (the
watcher signals every process, the scripts forward): each handler runs once.

## Stopping

The `control/*` bucket markers are for the single-job path only. On Slurm:

1. `scancel --name run3-waves` (no new waves).
2. `$VENV/bin/python -m flock remove run3-<CHATBOT_RUN>` (or `flock stop --release` to keep it resumable):
   the controller drains the agents (final sync per agent) and cancels the workers itself.
   Wait until `flock workers` is empty.
3. `scancel --name run3-world` (last workspace push), then `scancel --name run3-controller`
   (final state push). Use `scancel --signal=TERM` if the cluster's default is KILL.

`scancel` on workers directly is safe too (each drains on TERM in ≤ 20 s), but the controller
then counts them as planned restarts and launches them again unless the swarm was removed.
