# compute: slurm

Workers as Slurm jobs, sbatch'ed by the controller; the controller and the world are Slurm jobs
too, which `flock launch` submits and `flock stop --release` cancels. The reconcile loop,
placement, heartbeats, events and bucket sync are the same as for `hf-jobs`; only the backend
that starts a worker differs (`controller/slurm.py`). Agent state, the controller's state and the
run bundle stay in HF buckets. The shared disk holds code, the venv or image, the run directory
(env files, host files, logs) and the world's own state when the world keeps it there: chatbot4
uses `$CHATBOT_WS`, default `<world.dir>/ws`, so set a per-run `CHATBOT_WS` in run.env (runs from
one checkout would share it).

## Deploying

One run directory per run. `flock init` prepares it; `flock launch` starts its jobs; every later
command reads it. The steps and their order are the same on every backend (docs/live.md
"Lifecycle"):

```sh
flock init $PWD/runs/hierarchy --from my-experiment --profile my-cluster.env --secrets secrets.env
#   logs/ hosts/        the jobs' logs and host files (controller.host, world.host)
#   experiment/         a copy of the folder (links followed): what is launched
#   slurm/              a copy of scripts/slurm: the job scripts `flock launch` submits
#   cluster.env         a copy of the cluster profile (below)
#   run.env (mode 600)  every ${VAR} the spec reads (commented: the spec's default),
#                       the secrets it names, per-run FLOCK_TOKEN and FLOCK_WORLD_TOKEN
$EDITOR runs/hierarchy/run.env              # fill what the checks report; rerun `flock init runs/hierarchy`
export FLOCK_RUN_DIR=$PWD/runs/hierarchy    # or `flock --run runs/hierarchy ...` on each command
flock launch runs/hierarchy/experiment      # sbatch the world and the controller, wait, create
flock status hierarchy
flock stop hierarchy --release              # drain, checkpoint, stop the world, cancel both jobs
flock launch runs/hierarchy/experiment      # the jobs again: the controller restores, the swarm resumes
```

The jobs are named `flock-world-<run>` and `flock-ctl-<run>` (one of each per run:
`--dependency=singleton`); `flock launch` submits one only when no job of that name is pending
or running, with `-o logs/%x-%j.out`, `-p` from `WORLD_PARTITION` / `CTL_PARTITION`, else
`SLURM_PARTITION`, and `--qos`, `-c`, `--mem` from `WORLD_QOS` / `CTL_QOS`, `WORLD_CPUS` /
`CTL_CPUS`, `WORLD_MEM` / `CTL_MEM` when set (cluster.env or run.env). It waits up to 30 min (a
node may have to scale up). The controller restores its checkpoint (`<bucket>/controller/<run>`,
or `FLOCK_STATE_URL` in run.env) at every start, and sbatch'es the workers with
`slurm/worker.sbatch`. `flock stop --release` sends a running world `scancel --signal=TERM
--batch` and waits up to 11 min for it to end, then `scancel -n` both names.

The world job runs the spec's `world.command` in `world.dir` on its node (no container), the way
a local run starts it (docs/world.md): `{port}` and `FLOCK_WORLD_PORT` are a free port,
`{python}` the venv's python, `FLOCK_WORLD_TOKEN` comes from run.env. Once it answers
`/manifest` it writes `hosts/world.host`, which the spec's `world.url` reaches through
`http://${WORLD_HOST}:7860` (the controller fills host and port from the file).

A job script takes the run directory and sources `<run dir>/slurm/common.sh`, so each run keeps
the scripts it was set up with. `init` never overwrites an edited copy: for newer scripts,
delete `RUN_DIR/slurm/` and run `flock init RUN_DIR` again (a running job keeps the copy it
started with). The partition is the sbatch line's `-p`, never an `#SBATCH` line.

`init` creates only what is missing. It never changes a set `run.env` key or an edited copy;
it reports how they differ. `--secrets FILE` (mode 600) fills missing keys and prints names
only. Running it again is the doctor: it exits 1 when a check fails. None needs a controller:
the spec loads and validates with `run.env`; planned agents (ramp `max`, spawn caps) fit the
worker slots; the bucket is writable (skipped without an HF token; a note when it already holds
the swarm's data, which `flock launch` refuses); the tokens are set and the spec's
`world.token_env` equals `FLOCK_WORLD_TOKEN`; each provider-backed model answers a tool call
(the launch preflight); the board invite is set and `<board>/v1/health` answers (the join is not
tested: it could use up an invite); `WORKER_VENV` or `VENV` from `run.env` (else this Python)
imports `flock.worker`; the world passes `python -m flock.world doctor` when it is up
(`hosts/world.host`). With `compute.kind: slurm` also: the planned workers' CPUs against the
idle CPUs `sinfo -h -p <partition> -o %C` reports (read-only; short is a warning, not a failure:
Spot capacity moves and pending workers start as it frees up), and, when the profile names a
container, the venv's `bin/python` with symlinks resolved sits in the venv or under a
`CONTAINER_MOUNTS` entry (a uv-managed Python outside every mount fails with rc=127 in the
container). A spec whose `world.dir` default is relative (`${WORLD_DIR:-../../worlds/x}`)
cannot reach it from the copy: `init` writes `WORLD_DIR` as an absolute path from `--from`.

`--run RUN_DIR` (or `FLOCK_RUN_DIR`) points any command at a run: `cluster.env` then `run.env`
are the first `--env` files (later wins; an explicit `--env` wins over both), the controller URL comes from `hosts/controller.host`
(`host` means port 7860; `host:port` as written), else a local run's `RUN_DIR/.flock/current`,
and the token from `run.env`. `--run` beats an exported `FLOCK_URL`; `FLOCK_RUN_DIR` loses to it.
With neither host file every command but `launch` refuses ("launch first").

## Cluster profile

One cluster's values live in an env file, never in flock's code or the experiment's spec.
`flock init --profile FILE` copies it to `RUN_DIR/cluster.env` (an existing, maybe edited, copy
is kept and a difference reported). It is read before `run.env` everywhere: by every command
(`--run`), by `init`'s checks and by the job scripts (`common.sh` sources it first). Later wins,
so `run.env` overrides the profile; the spec still wins over both (it only fills `${VAR}`s).
It is env, not YAML: no structure, no second spec.

```sh
# my-cluster.env: placeholders; one cluster's real values live with that cluster's kit
SLURM_PARTITION=cpu                      # sbatch -p and the specs' ${SLURM_PARTITION:-}; unset: the cluster default
WORKER_CPUS=4                            # the specs' compute.cpus / compute.mem
WORKER_MEM=7G
SLURM_MAX_MEM_PER_CPU_MB=2000            # the partition's MaxMemPerCPU: more silently raises the CPU count; unset: not checked
FLOCK_SLURM_MAX_CPUS=32                  # the largest job a node runs (spec max_cpus/max_mem); unset: not checked
FLOCK_SLURM_MAX_MEM=60G
VENV=/shared/flock/venv                  # the job scripts: bin/flock, flock-controller, flock-worker, python
REPO=/shared/flock/repo                  # optional: a flock checkout, mounted in the workers' container
SECRETS_FILE=/shared/flock/secrets.env   # optional, mode 600, sourced by every job before run.env
CONTAINER_IMAGE=/shared/flock/images/worker.sqsh   # optional: Enroot/Pyxis image for the workers
CONTAINER_MOUNTS=/shared/uv/python       # comma-separated, read-only at the same path, besides VENV, REPO and the run's slurm/
```

Who reads what: the specs read `SLURM_PARTITION`, `WORKER_CPUS`, `WORKER_MEM`. The controller
job checks `WORKER_MEM / WORKER_CPUS` against `SLURM_MAX_MEM_PER_CPU_MB` and hands
`SLURM_PARTITION` to the Slurm client as `FLOCK_SLURM_PARTITION`; the Slurm client and
`flock init` read `FLOCK_SLURM_MAX_CPUS/_MEM` as they are (a size over them is refused).
`cluster.env` is sourced as shell by jobs that hold secrets: it must be owned by the job's user
and not group- or world-writable (the jobs refuse it otherwise). The job scripts read the paths, the
image and the mounts. Without a partition anywhere, sbatch gets no `-p`. `flock init` layers
`cluster.env`, then `SECRETS_FILE` (owner-only, mode 600 or 400, else a failed check), then `run.env`, as the jobs do, for its board, provider and bucket checks.

## The launch contract

For each worker slot the controller writes `<launch_dir>/<worker id>-<incarnation>.env`
(mode 600, `KEY=value` lines, **no secret values**) and runs, with a scrubbed environment
(`PATH HOME USER LOGNAME LANG LANGUAGE TZ TMPDIR TERM SHELL LC_* SLURM_*` only; never
`--export`):

```
sbatch --parsable [-p <partition>] [-A <account>] -c <cpus> --mem=<mem> --time=<time>
       --no-requeue -J <worker id> --comment=swarm=..,type=..,worker=..
       -o <log_dir>/<worker id>-%j.out <worker_script> <env file>
```

The env file holds `WORKER_ID WORKER_INCARNATION SWARM TYPE DEADLINE DRAIN_WINDOW BUCKET RUN
FLOCK_URL WORK_ROOT PYTHONUNBUFFERED`, the type's `env`, and when configured
`FLOCK_CTL_HOST_FILE FLOCK_VENV FLOCK_SECRETS_FILE FLOCK_CONTAINER_IMAGE` (and, with
`container_root`, `HOME HF_HOME TMPDIR`). A comment names the secrets the worker
expects from the cluster's secrets file: `FLOCK_TOKEN HF_TOKEN` plus the launch's (provider
key, the type's `secrets`). The worker script (the run's `slurm/worker.sbatch`;
the controller job sets `FLOCK_SLURM_WORKER_SCRIPT` to it) sources the secrets file, run.env and
the env file and runs `flock-worker`, handing it those secrets in a private file (below).

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
the controller's `host:port` from `FLOCK_CTL_HOST_FILE` at start and again after a connection
error, so a controller requeued on another node (and port) is followed without relaunching the
fleet. Ports: the controller and the world jobs each bind a free port (Slurm may place both on
one node) and write `host:port` to `hosts/controller.host` / `hosts/world.host`; every reader
takes the port from the file, and an old host-only file means the URL's own port (7860).
The controller itself gets the same template through `FLOCK_WORKER_URL` (honoured in space
mode too).

## Spec keys and env overrides

```yaml
compute:
  kind: slurm
  partition: ${SLURM_PARTITION:-}   # FLOCK_SLURM_PARTITION; empty: no -p
  account: null              # FLOCK_SLURM_ACCOUNT (-A only when set)
  cpus: 4                    # FLOCK_SLURM_CPUS
  mem: 7G                    # FLOCK_SLURM_MEM
  time: "29:00:00"           # FLOCK_SLURM_TIME
  worker_script: /shared/... # FLOCK_SLURM_WORKER_SCRIPT (required)
  log_dir: /shared/...       # FLOCK_SLURM_LOG_DIR (required)
  launch_dir: /shared/...    # FLOCK_SLURM_LAUNCH_DIR (required)
  host_file: /shared/...     # FLOCK_SLURM_HOST_FILE: the controller's host:port
  run: null                  # FLOCK_SLURM_RUN; default: launch_dir's parent dir name
  venv: null                 # FLOCK_SLURM_VENV -> FLOCK_VENV
  secrets_file: null         # FLOCK_SLURM_SECRETS_FILE -> FLOCK_SECRETS_FILE (path only)
  container_root: null       # FLOCK_SLURM_CONTAINER_ROOT, e.g. /work (see Containers)
  container_image: null      # FLOCK_SLURM_CONTAINER_IMAGE -> FLOCK_CONTAINER_IMAGE
  max_cpus: null             # FLOCK_SLURM_MAX_CPUS (a profile key)
  max_mem: null              # FLOCK_SLURM_MAX_MEM (a profile key)
  worker_dead_after: 180s    # no env; also worker_unknown_after, crash_backoff (docs/live.md)
  types:                     # per type: partition, cpus, mem, time, container_root
    interviewer: {cpus: 2, mem: 3800M}
```

Precedence: built-in defaults (`cpus` 4, `mem` 7G, `time` 29:00:00; no partition, no node
limits) < `FLOCK_SLURM_*` on the controller < spec keys < `types.<type>`: the spec wins; the env
only fills keys the spec leaves unset.
A size over `max_cpus`/`max_mem` (when set) is refused at apply and at launch (`launch_refused`): Slurm
would accept it and leave it pending forever. A missing `worker_script`/`log_dir`/`launch_dir`
is refused too. `bucket` is required; `world.url` is required (no `command` worlds).

### Worker knobs

A type's `worker:` block sets its workers' own settings. Workers take it at assign and through
heartbeats, so a `flock update` changes running workers without a relaunch. Each key set wins
over the worker's env variable (from the job env, e.g. `run.env`), which wins over the default;
removing a key returns to the env value.

```yaml
types:
  coder:
    worker:
      drain_budget_s: 20        # FLOCK_DRAIN_BUDGET_S: the whole preemption handler
      sync_interval_s: 180      # SYNC_INTERVAL: periodic bucket push
      sync_exclude: [data/raw/] # FLOCK_SYNC_EXCLUDE: added to the default excludes
      restore_batch: 1000       # FLOCK_RESTORE_BATCH: files per download batch
      restore_threads: 4        # FLOCK_RESTORE_THREADS: one agent's batches in parallel
      memory: {rotate: null, drain: null, preempt: null}
                                # FLOCK_MEM_ROTATE, _DRAIN, _PREEMPT
```

Each heartbeat carries `knobs_rev`,
a hash of the settings in force, and `flock status --json` shows it (`knobs_rev` per worker), so workers running
an older configuration stand out.

The controller: `FLOCK_COMPUTE=slurm`. An unknown `FLOCK_COMPUTE` stops it at
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
crash's last log lines reach `last_error` (`flock status`) also after a controller restart.

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
(after `compute.worker_unknown_after` of silence, 90 s by default); `TIMEOUT` is planned as
before.

The controller and `python -m flock.world serve` treat SIGUSR1 like SIGTERM, once (one final state push
for the controller).

## Memory

`OUT_OF_MEMORY` is a crash with backoff, so the worker keeps itself under its limit. Every
15 s it reads its cgroup (the smallest `memory.max` up the tree; when none, Slurm's in-job
`SLURM_MEM_PER_NODE`, then `FLOCK_MEMORY_LIMIT_MB`, both MB) and compares nonreclaimable memory (anon + shmem + slab_unreclaimable + kernel_stack +
pagetables, not page cache) with the limit. Without a limit nothing below is armed. Each rung
is logged once until memory falls below the first one again. `rotate`, `drain` and `preempt`
can also be set per type under `worker.memory` (spec, then the env variable, then the default):

| Nonreclaimable / limit | Worker |
|---|---|
| ≥ `FLOCK_MEM_ROTATE` (0.75) | `wants_rotation: true` in the heartbeat (`mem.eta_limit_s` next to it), for the controller's rotation (below) |
| ≥ `FLOCK_MEM_DRAIN` (0.85) | drain: turns still running after `FLOCK_MEM_DRAIN_DEADLINE_S` (300) are interrupted (`drain_deadline`, resumed with `resume_prompt` after the restart), sync, exit 0: a planned restart |
| ≥ `FLOCK_MEM_PREEMPT` (0.93), or a new `oom_kill` in `memory.events` | `preempt("memory")`, as on SIGUSR1: exit 75, relaunched at once |

The ladder acts on the whole worker. For Harbor agents that means: no single Trial is picked
and restarted for memory (the worker cannot tell which one grows: a Trial's agent runs in its
own container, so its memory is not in the worker's process tree). Relief is a drain or a
preemption of the whole worker; every agent's Trial stops, its Pi sessions go to the bucket and
the agents resume on the relaunched worker. On docker and podman a Trial's container memory
counts against the worker's job only when the engine puts its containers inside the job's
cgroup; on hf-sandbox it does not count at all. `worker_lifetime` stays the
backstop. (`worker.memory.spike_mb` and `recycle_gap_s`, the removed per-agent guard, are
dropped from stored specs with a warning.)

The controller rotates workers that ask for it: each pass, among workers whose heartbeat says
`wants_rotation`, it sends a drain to the one with the smallest `eta_limit_s` while fewer than
`FLOCK_MAX_ROTATING` (default 10% of the swarm's worker slots, at least 1) are launching,
restoring or draining, and writes `memory_rotation {worker, nonreclaim_mb, eta_limit_s}` to
the swarm stream. A worker's own drain or preemption at 85% / 93% does not wait for this budget.

`memory.oom.group` (logged at start): when it is 1 on the job's cgroup, one OOM kill ends every
process in it, so `oom_score_adj` does not help; only this ladder does.

## The world on a moving host

`world.url` may contain `${WORLD_HOST}` (e.g. `http://${WORLD_HOST}:7860`) with
`FLOCK_WORLD_HOST_FILE` on the controller; a `host:port` in the file replaces the URL's port
(7860 stays the default for an old host-only file). The host is read at apply and every 15 s after;
when the file names another host and the world answers there, the stored spec gets the new
endpoint and workers get it with their next heartbeat. The world credential is bound to the
template, so workers keep their secret and do not drain; only a URL of that form (one host
name, optional port) receives it.

## Containers (Enroot/Pyxis)

`harness: harbor` is the only harness, so `container_root` and `container_image` (and
`CONTAINER_IMAGE`) are ignored for workers (below). A worker with no usable HOME derives HOME, HF_HOME, XDG_CACHE_HOME, UV_CACHE_DIR and TMPDIR from the
parent of `WORK_ROOT`; other processes use `FLOCK_HOMELESS_ROOT`.

`harness: harbor` workers never run in a container: each agent is a Harbor Trial in rootless
podman, which cannot run inside Enroot. Their env file carries `FLOCK_HARNESS=harbor` and node
paths, and `worker.sbatch` runs `flock-worker` here as its child with the upstream podman setup:
the job's own podman storage in `/tmp/flock-podman.*` (left to the job's own `/tmp`, a per-job
tmpfs on hopper-cpu), `podman-compose` from the venv (a flock dependency, Python 3.12+), podman's
health checks run by `python -m flock.harbor.podman_healthchecks` (compute nodes have no user
systemd), and the upstream git credential helper for huggingface.co (private task repos), which
reads `HF_TOKEN` from a 0600 file there. At exit the job's containers are stopped, within
KillWait. With `--cgroup-manager cgroupfs` and no user systemd, conmon and the containers should
stay in the job's cgroup (Slurm's `Contain` kills them, the memory ladder sees them): unverified
on compute nodes so far.
Every user of a node can read a process's command line (`ps`): a Trial's `compose exec` passes
the agent's env (its inference key, resource tokens) and sensitive-looking names (`*_TOKEN`,
`*_API_KEY`, `*SECRET*`, `*PASSWORD*`) by name only. A task's `[verifier.env]` value taken from
a secret must have such a name, or it stays on the command line.
The pinned task is trusted with the host's container engine: its compose file and its
`[environment.env]` reach podman (or docker) as written, and flock only keeps secret values out
of `ps`. Run a task you do not trust on `hf-sandbox`.

A private task repo is pinned on the operator's machine too (`flock init` and `flock launch` run
Harbor's plain `git`: `ls-remote`, then a shallow fetch to list the tasks), where no helper is set up. Export one in that shell first; it
reads `HF_TOKEN` from the environment, so no token is on any command line (the empty value first
drops other helpers for huggingface.co). `HF_TOKEN` must be *exported* in this shell: a value
only present in the run's `run.env` is not seen by git:

```sh
export GIT_CONFIG_COUNT=2 \
  GIT_CONFIG_KEY_0=credential.https://huggingface.co.helper GIT_CONFIG_VALUE_0= \
  GIT_CONFIG_KEY_1=credential.https://huggingface.co.helper \
  GIT_CONFIG_VALUE_1='!f() { test "$1" = get && printf "username=hf_user\npassword=%s\n" "$HF_TOKEN"; }; f'
GIT_TERMINAL_PROMPT=0 git ls-remote <task repo URL> HEAD   # check: a sha, no prompt
```

## Singletons on Spot

The world and the controller are single jobs that Spot reclaim requeues (`--dependency=singleton`:
one of each per run). `world.sbatch` counts only crashes within `WORLD_STABLE_S` of the start
toward its cap of 5 (a longer run resets it) and, past the cap, retries with 5 min to 30 min
backoff instead of ending. A requeued controller restores its state and follows the world's host
without waiting for it. For non-preemptible placement set `WORLD_PARTITION`/`WORLD_QOS` and
`CTL_PARTITION`/`CTL_QOS` in run.env: `flock launch` passes them to `sbatch`.
