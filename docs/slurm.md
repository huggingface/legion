# compute: slurm (and external)

Workers as Slurm jobs, sbatch'ed by the controller. The reconcile loop, placement, heartbeats,
events and bucket sync are the same as for `hf-jobs`; only the backend that starts a worker
differs (`controller/slurm.py`). Agent state, the controller's state, the world's workspace
and the run bundle stay in HF buckets; the shared disk holds only code, the venv or image,
the secrets file, host files, env files and logs.

## The launch contract

For each worker slot the controller writes `<launch_dir>/<worker id>-<incarnation>.env`
(mode 600, `KEY=value` lines, **no secret values**) and runs, with a scrubbed environment
(`PATH HOME USER LOGNAME LANG LANGUAGE TZ TMPDIR TERM SHELL LC_* SLURM_*` only; never
`--export`):

```
sbatch --parsable -p <partition> [-A <account>] -c <cpus> --mem=<mem> --time=<time>
       --no-requeue -J <worker id> --comment=swarm=..,type=..,worker=..
       -o <log_dir>/<worker id>-%j.out <worker_script> <env file>
```

The env file holds `WORKER_ID WORKER_INCARNATION SWARM TYPE DEADLINE DRAIN_WINDOW BUCKET RUN
FLOCK_URL WORK_ROOT FLOCK_LLM_HOME PYTHONUNBUFFERED`, the type's `env`, and when configured
`FLOCK_CTL_HOST_FILE FLOCK_VENV FLOCK_SECRETS_FILE FLOCK_CONTAINER_IMAGE` (and, with
`container_root`, `HOME HF_HOME TMPDIR`). A comment names the secrets the worker
expects from the cluster's secrets file: `FLOCK_TOKEN HF_TOKEN` plus the launch's (provider
key, board invite, `FLOCK_WORLD_TOKEN`). The worker script (the cluster's, or
`experiments/run3/slurm/worker.sbatch`) sources the secrets file and the env file and runs
`flock-worker`.

### Worker secrets file

Sourcing the secrets file puts them in the worker's exec environment, which any same-user
process can read from `/proc/<pid>/environ` for the worker's whole life (unsetting them later
does not change it). To keep them out, the worker script writes the worker's secrets to a
private file and execs `flock-worker` with only its path:

- `FLOCK_WORKER_SECRETS_FILE=<path>`: read once at start. A regular file (not a symlink)
  owned by the worker's user, mode `0600` (any group/other bit is refused and the worker
  stops). `FLOCK_SECRETS_FILE` above is the cluster's shell-sourced file; flock never reads it.
- Format: one `NAME=value` per line; blank lines and `#` comment lines are ignored; an
  optional `export ` prefix; a value in matching `'...'` or `"..."` is unquoted. No `$VAR`
  expansion, escapes or multi-line values.
- Names: the worker's credentials, `FLOCK_TOKEN` and `HF_TOKEN`, plus the launch's provider
  key (e.g. `OPENAI_API_KEY`, the type's `api_key_env`), board invite (e.g.
  `CHATBOT_BOARD_INVITE`), `FLOCK_WORLD_TOKEN`, and any name in a type's `secrets`.
- Precedence: a name in the file wins over the same name in the environment, and every name
  in the file is removed from the worker's `os.environ`, so processes it starts never inherit
  it. Names not in the file are still read from the environment (the old way keeps working).
- `FLOCK_WORKER_SECRETS_FILE_UNLINK=1` deletes the file once it is read (a requeued step then
  needs a new one). The worker logs the names it read, never the values.

```
# worker-inner.sh
umask 077
secrets=$(mktemp "${TMPDIR:-/tmp}/flock-worker-secrets.XXXXXX")
printf 'FLOCK_TOKEN=%s\nHF_TOKEN=%s\nOPENAI_API_KEY=%s\nFLOCK_WORLD_TOKEN=%s\nCHATBOT_BOARD_INVITE=%s\n' \
  "$FLOCK_TOKEN" "$HF_TOKEN" "$OPENAI_API_KEY" "$FLOCK_WORLD_TOKEN" "$CHATBOT_BOARD_INVITE" > "$secrets"
exec env -u FLOCK_TOKEN -u HF_TOKEN -u OPENAI_API_KEY -u FLOCK_WORLD_TOKEN -u CHATBOT_BOARD_INVITE \
  FLOCK_WORKER_SECRETS_FILE="$secrets" FLOCK_WORKER_SECRETS_FILE_UNLINK=1 flock-worker
```

`FLOCK_URL` is `http://${CTL_HOST}:<port>` when a host file is configured: the worker reads
the controller's hostname from `FLOCK_CTL_HOST_FILE` at start and again after a connection
error, so a controller requeued on another node is followed without relaunching the fleet.
The controller itself gets the same template through `FLOCK_WORKER_URL` (honoured in space
mode too).

## Spec keys and env overrides

```yaml
compute:
  kind: slurm
  partition: hopper-cpu      # FLOCK_SLURM_PARTITION
  account: null              # FLOCK_SLURM_ACCOUNT (-A only when set)
  cpus: 4                    # FLOCK_SLURM_CPUS
  mem: 7G                    # FLOCK_SLURM_MEM
  time: "29:00:00"           # FLOCK_SLURM_TIME
  worker_script: /fsx/...    # FLOCK_SLURM_WORKER_SCRIPT (required)
  log_dir: /fsx/...          # FLOCK_SLURM_LOG_DIR (required)
  launch_dir: /fsx/...       # FLOCK_SLURM_LAUNCH_DIR (required)
  host_file: /fsx/...        # FLOCK_SLURM_HOST_FILE: the controller's hostname
  run: null                  # FLOCK_SLURM_RUN; default: launch_dir's parent dir name
  venv: null                 # FLOCK_SLURM_VENV -> FLOCK_VENV
  secrets_file: null         # FLOCK_SLURM_SECRETS_FILE -> FLOCK_SECRETS_FILE (path only)
  container_root: null       # FLOCK_SLURM_CONTAINER_ROOT, e.g. /work (see Containers)
  container_image: null      # FLOCK_SLURM_CONTAINER_IMAGE -> FLOCK_CONTAINER_IMAGE
  max_cpus: 28               # FLOCK_SLURM_MAX_CPUS
  max_mem: 54G               # FLOCK_SLURM_MAX_MEM
  types:                     # per type: partition, cpus, mem, time, container_root
    interviewer: {cpus: 2, mem: 3800M}
```

Precedence: built-in defaults < spec keys < `FLOCK_SLURM_*` on the controller < `types.<type>`.
A size over `max_cpus`/`max_mem` is refused at apply and at launch (`launch_refused`): Slurm
would accept it and leave it pending forever. A missing `worker_script`/`log_dir`/`launch_dir`
is refused too. `bucket` is required; `world.url` is required (no `command` worlds).

The controller: `FLOCK_COMPUTE=slurm` (or `external`). An unknown `FLOCK_COMPUTE` stops it at
start (it never falls back to HF Jobs); the first log line names the kind and the client.

## Job states

`squeue -h -j <id> -o %T` first (finished jobs stay a few minutes), then
`sacct -j <id> -X -n -P -o State,ExitCode` (it may lag; a job unknown to both within 120 s of
its sbatch is still SCHEDULING).

| Slurm | flock |
|---|---|
| PENDING, CONFIGURING, REQUEUE_HOLD, SUSPENDED, ... | SCHEDULING (autoscaling from 0 can take ~15 min: normal) |
| RUNNING, COMPLETING, SIGNALING, STAGE_OUT | RUNNING |
| COMPLETED | COMPLETED |
| TIMEOUT | ERROR "Job timeout": planned restart |
| CANCELLED that flock requested | CANCELED |
| CANCELLED that flock did not request (a Spot reclaim), PREEMPTED, REQUEUED, NODE_FAIL, exit 75 | ERROR "preempted: ...": planned restart |
| FAILED, OUT_OF_MEMORY, BOOT_FAIL, DEADLINE | ERROR "<STATE> exit <code>": a crash (backoff, crashlooping after 5) |

flock's own cancels are remembered in `<launch_dir>/cancelled/<job id>`, so a restarted
controller still tells them from a reclaim. The job's log path is kept on the slot row: a
crash's last log lines reach `last_error` (`flock workers`) also after a controller restart.

## Preemption (Spot)

The cluster's watcher sends SIGUSR1 to every process of a reclaimed job and cancels it about
30 s later. The worker (also on SIGTERM, outside compute local) drains once, bounded by
`FLOCK_DRAIN_BUDGET_S` (default 20 s); a second USR1 or a TERM during the drain changes nothing:

1. no new turns; every running turn is interrupted at once (`turn_interrupted`);
2. every agent's session, harness files and workdir go to the bucket **in parallel** (one
   thread per agent: 25 agents with 0.25 s bucket calls take ~1.7 s in the test, 12.5 s one by
   one);
3. `<swarm>/_workers/<worker>/preempted.json` in the bucket;
4. events flushed, a last heartbeat with `preempted: true` and the interrupted agents;
5. exit 75.

The controller ends the slot on that heartbeat (it cancels the job if it is still listed) and
relaunches it in the same pass: a planned restart (no backoff, crash streak 0, never
crashlooping). The new job registers with a fresh incarnation (the old process gets 409 and
exits without syncing), restores every agent from the bucket, and the interrupted agents get
the type's `resume_prompt`. Without the flag (a hard kill), the unrequested `CANCELLED` (or
`PREEMPTED`, `NODE_FAIL`, exit 75) gives the same planned restart once the job is checked
(after 90 s of silence); `TIMEOUT` is planned as before.

The controller and `flock world serve` treat SIGUSR1 like SIGTERM, once (one final state push
for the controller).

## The world on a moving host

`world.url` may contain `${WORLD_HOST}` (e.g. `http://${WORLD_HOST}:7860`) with
`FLOCK_WORLD_HOST_FILE` on the controller. The host is read at apply and every 15 s after;
when the file names another host and the world answers there, the stored spec gets the new
endpoint and workers get it with their next heartbeat. The world credential is bound to the
template, so workers keep their secret and do not drain; only a URL of that form (one host
name, optional port) receives it.

## Containers (Enroot/Pyxis)

Workers and the world may run in a container (`srun --container-image`), the controller stays
on the host (it needs `sbatch`). In the container `/tmp` is a RAM tmpfs counted against the
job's memory, so nothing goes there: with `container_root: /work` the env file sets
`WORK_ROOT FLOCK_LLM_HOME HOME HF_HOME TMPDIR` to `/work/flock/<worker id>/{work,llm,home,hf,tmp}`
(the world: `/work/world/ws`). The run-4 `worker.sbatch` overrides `WORK_ROOT FLOCK_LLM_HOME`
with worker-id-free paths (`/work/flock/agents/{work,llm}`): a Pi session resumes only from the
cwd it was written from, so an agent's workdir must not change when the controller relaunches
its slot under a new worker id. A worker with no usable HOME derives HOME, HF_HOME,
XDG_CACHE_HOME, UV_CACHE_DIR and TMPDIR from the parent of `WORK_ROOT`; other processes use
`FLOCK_HOMELESS_ROOT`. A world's bot runs get their own run dir as TMPDIR.
`experiments/run3/slurm/Containerfile` builds the image.

## compute: external

The controller launches nothing. Each "launch" writes the slot's env file (when `launch_dir` is
set) and logs `waiting for an external start of <worker> (...)`; the slot shows `external` in
`flock workers` until a worker started by someone else (same env file, `FLOCK_TOKEN` from the
secrets) heartbeats. A silent external worker sends the slot back to waiting with a fresh env
file: never a crash, never backoff. Start workers from the newest
`<launch_dir>/<worker>-*.env`: an older incarnation gets 409.

## Singletons on Spot (run 4 scripts)

The world and the controller are single jobs that Spot reclaim requeues. `world.sbatch` counts
only crashes within `WORLD_STABLE_S` of the start toward its cap of 5 (a longer run resets it)
and, past the cap, retries with 5 min to 30 min backoff instead of ending. A requeued controller
restores its state and follows the world's host without waiting for it. For non-preemptible
placement set `WORLD_PARTITION`/`WORLD_QOS` and `CTL_PARTITION`/`CTL_QOS` in run.env and pass them
to `sbatch` (`experiments/run4/slurm/run.env.example`).
