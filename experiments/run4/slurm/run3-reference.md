# flock run 3 on CPU Slurm (hopper-cpu): runbook

Scripts for running the run-3 agent swarm (`/fsx/carlos_patino/prj-eliza/flock`, branch `run3`)
as CPU Slurm jobs instead of HF Jobs. Spec: `notes/run3-slurm-facts.md`. Nothing here is
submitted automatically; this file is the runbook.

| file | job | sbatch shape | what it does |
|---|---|---|---|
| `common.sh` | (sourced) | - | RUN + run dir, `umask 077`, sources `secrets/secrets.env` (600) and `<run>/run.env` (600), `log()`, `require_var`, venv check, `HF_HOME` (job-private), `publish_host`, container helpers (`CONTAINER_IMAGE`, `container_srun_args`), `step_signal` |
| `world.sbatch <RUN>` | `flock-world-<RUN>` | `-c 16 --mem=30G --time=50:00:00 --requeue --dependency=singleton --open-mode=append` | host side: `<run>/world/world.yaml` (`runs.parallel: 14`), `<run>/hosts/world.host` at every start, signals, requeue; runs `world-inner.sh` in a container; USR1 -> USR1 to the step (stop, final sync 12 s), requeue; signal-only TERM -> USR2 to the step (stop, final sync up to 600 s) |
| `world-inner.sh` | (in the container) | - | restore `ws/` from the bucket into an empty `/work/world/ws`, `flock world serve chatbot3:Chatbot3World` on 7860, push every 300 s, bounded final sync |
| `controller.sbatch <RUN>` | `flock-ctl-<RUN>` | `-c 2 --mem=3800M --time=50:00:00 --requeue --dependency=singleton --open-mode=append` | ON THE HOST (needs sbatch/squeue/sacct/scancel); checks the venv has `flock.controller.slurm`; waits for `hosts/world.host`; `flock-controller` (space mode, `FLOCK_COMPUTE=slurm`, state in the bucket); writes `hosts/controller.host`; once `/healthz` is 200: render spec (+ `compute.types.interviewer` 2 CPU / 3800M), `flock apply` (first start) or `waves.py pin` (restart), then `waves.py run` (in the background); USR1 -> TERM to the controller (final push), requeue |
| `worker.sbatch <envfile>` | `<worker id>` | `-c 4 --mem=7G --time=29:00:00 --no-requeue` | launched by the controller's Slurm client; sources secrets then the env file; `CTL_HOST` from `hosts/controller.host`; runs `worker-inner.sh` in a container; USR1 forwarded (TERM 3 s later through `scancel --signal=TERM <job>.<step>`), no requeue |
| `worker-inner.sh` | (in the container) | - | creates `/work/flock/<id>/{work,llm}`, `HOME`/`HF_HOME`/`TMPDIR` under `/work`, exec's `flock-worker` with USR1 ignored |
| `worker.env.example` | - | - | the env file the Slurm client writes per launch (fc22785: no secret values) |
| `run.env.example` | - | - | the run's non-secret settings (bucket, board, endpoint, model, container image, interviewer size) |
| `tests/ctr-signal-test.slurm` | `ctr-signal-test` | `-c 2 --mem=3800M` | host + container signal plumbing with a dummy process (jobs 255717, 255727) |

Paths: venv `/fsx/carlos_patino/prj-eliza/venv` (used as `venv/bin/<cmd>`, no activate), runs under
`/fsx/carlos_patino/prj-eliza/runs/<RUN>/` with `logs/`, `launch/`, `experiment/` (a writable copy of
`experiments/run3`, rendered spec `spec.slurm.local.yaml`), `hosts/{controller,world}.host` (plus the
symlinks `controller.host`, `world.host` in the run dir), `world/world.yaml` (an older
`<run>/world.yaml` is moved there once), `run.env`. Agent files never go on `/fsx` (worker
`/work/flock/<id>` and world `/work/world/ws` on the container root, controller
`/tmp/flock-controller-<jobid>/data` on the host); everything durable lives in the HF bucket, as today.
`sbatch --export` is never used: the jobs read `secrets.env` and `run.env` themselves.

Memory: `hopper-cpu` has `MaxMemPerCPU=2000`. More than 2000 MB per CPU silently raises the CPU
count, so every request stays at or under it: controller 3800M / 2 = 1900, world 30G / 16 = 1920,
worker 7G / 4 = 1792 (`FLOCK_SLURM_MEM` / `FLOCK_SLURM_CPUS`).

Signals: the Spot watcher sends SIGUSR1 to the batch shell AND to every process of the job (srun
and the processes in the container included). Each script starts its children while the shell
ignores USR1 (`trap '' USR1`; SIG_IGN is inherited across exec, a trap handler is not), then
installs its handler. A container task does not inherit the host's dispositions, so the inner
scripts do the same inside: `world-inner.sh` starts the server, the restore and the sync loop with
USR1 ignored; `worker-inner.sh` exec's `flock-worker` with USR1 ignored (its Python handler, being
added, replaces SIG_IGN). The controller and world install their traps right after `common.sh`, so a
reclaim during the start-up still requeues. Measured (tests/ctr-signal-test.slurm):

- `scancel --signal=USR1 --full`: the host trap ran 38 ms after the scancel, the container process
  got USR1 twice (direct from slurmd, then forwarded by srun from `kill -USR1 $SRUN_PID`): handlers
  must be idempotent. Dummy drain 3 s; step gone 4.0 s after the trap; the job kept running.
- `kill -TERM $SRUN_PID` from the batch shell: srun prints `forcing job termination` and SIGKILLs the
  task at once (88 ms), even when srun was started with TERM ignored. So the scripts NEVER send TERM
  to srun: the world forwards a clean stop as USR2 (srun forwards USR1/USR2 as is, verified), the
  worker sends TERM through `scancel --signal=TERM <job>.<step>` (`step_signal`, verified: drain ran).
- A plain `scancel <job>` and `scancel --signal=TERM <job>` deliver TERM straight to the container
  process; srun waits (`Waiting up to 32 seconds`): the drain ran in both.

**The controller's Slurm client must not pass `--export` to sbatch.** `worker.sbatch` reads
`secrets.env` and its env file itself. With the default export (`ALL`) the worker inherits the
controller's environment (`FLOCK_*`, `RUN3_*`, secrets): same user and the env file wins, so this
is tolerable; `--export=NONE` or a list would drop `PATH` and the Slurm variables the scripts need.

## Containers (world and worker)

`flock world serve` and `flock-worker` run in an Enroot/Pyxis container; the batch scripts, the
controller and the waves runner stay on the host. srun line (`common.sh container_srun_args`):

```sh
srun --ntasks=1 --cpus-per-task=$SLURM_CPUS_PER_TASK --kill-on-bad-exit=1 \
     --container-image=$CONTAINER_IMAGE --no-container-mount-home \
     --container-mounts=$VENV:$VENV:ro,/fsx/carlos_patino/uv/python:...:ro,$REPO:$REPO:ro,$SLURM_DIR:$SLURM_DIR:ro,<extra>:ro \
     --container-workdir=/tmp  bash $SLURM_DIR/{world,worker}-inner.sh
```

Image: `CONTAINER_IMAGE`, default `/fsx/carlos_patino/prj-eliza/images/python-3.12-slim.sqsh`
(Debian 13, no curl; override in `run.env`). Per worker type (common.sh `select_worker_image`,
called by worker.sbatch): `CONTAINER_IMAGE_<TYPE>` in `run.env` (`-`/`.` in the type become `_`), else
`FLOCK_CONTAINER_IMAGE` from the env file (flock's client: one image for all types), else `CONTAINER_IMAGE`.
run3: `CONTAINER_IMAGE_agent_pi=$PRJ/images/pi-worker-node22.sqsh` (Node 22 + Pi 1.2.0 + git/rg, built by
`tests/pi-image.slurm`; no Python, it comes from the mounts). In an image with `/usr/local/bin/pi`,
worker-inner.sh exports `FLOCK_PI_CLI` and `PI_CODING_AGENT_DIR=/work/pi/<worker id>`. Extra mounts: `<run>/world` for the world
(`world.yaml`, `final-sync-bound`), `<run>/hosts` for the worker (`FLOCK_CTL_HOST_FILE`).

What is isolated:

- **Home is not mounted** (`--no-container-mount-home`): `~/.cache/huggingface/token`, ssh keys,
  `.netrc` are not visible (probe: `token-not-visible`). `HOME=/work/home`, `HF_HOME=/work/hf`,
  `TMPDIR=/work/tmp` inside.
- **`/fsx` is read-only and limited** to the venv, the uv Python, the flock repo, this slurm dir and
  the one run subdir above. Writes fail (`Read-only file system`, tested). The secrets dir, the run's
  `launch/`, `logs/` and other runs are not mounted.
- **Agent and world files live on the container root** (`/work/...`, the node's NVMe, ~300 GB on
  CPU nodes), deleted with the container at the end of the step. Not `/tmp`: in the container it is
  a tmpfs counted against the job memory.
- **The venv works without home**: `venv/bin/python` resolves to
  `/fsx/carlos_patino/uv/python/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12`, mounted ro (the
  review's concern about `~/.local/share/uv/python` does not apply here).

What is NOT isolated:

- **Network**: the container uses the host network (same hostname, ports, internet via the NAT, HF
  Hub, vLLM, the board).
- **Secrets**: sourced on the host from `secrets.env` and passed as environment variables (the
  batch script's env goes into the container): agent code can read `HF_TOKEN`, `FLOCK_TOKEN`,
  `OPENAI_API_KEY`, ... from its environment. Use a fine-grained `HF_TOKEN` (write on the run's
  bucket only).
- Same UID as the host user; CPU and memory are the job's cgroup.
- Logs: the processes' stdout/stderr go through srun to `/fsx` (`world-<RUN>.log`, the job `.out`).

**Why the controller stays on the host:** its Slurm jobs client runs `sbatch`, `squeue`, `sacct`
and `scancel`. The container has no Slurm client, `/etc/slurm` or munge socket; bind-mounting them
failed in the probe (`enroot-mount: failed to mount /lib/libmunge.so.2`) and would be fragile.
`scontrol requeue` and the `world.host`/`controller.host` files are host-side for the same reason.

Pyxis prints a harmless `[Auto Resume] Failed to initialize the logger.` line at each step start.

## 0. One-time setup

```sh
P=/fsx/carlos_patino/prj-eliza
ls -l $P/secrets/secrets.env          # -rw------- ; defines HF_TOKEN OPENAI_API_KEY CHATBOT_BOARD_INVITE CHATBOT_WORLD_TOKEN FLOCK_TOKEN
ls $P/venv/bin/{flock,flock-controller,flock-worker,hf,python}
mkdir -p $P/runs/_slurm-logs          # fallback -o dir of the #SBATCH headers (used when -o is not passed)
for f in $P/slurm/*.sbatch $P/slurm/*.sh; do bash -n "$f"; done
ls -l $P/images/python-3.12-slim.sqsh     # the container image (CONTAINER_IMAGE)
```

Per run:

```sh
RUN=run3a                             # letters, digits, . _ - only
mkdir -p $P/runs/$RUN/logs $P/runs/$RUN/launch
cp $P/slurm/run.env.example $P/runs/$RUN/run.env && chmod 600 $P/runs/$RUN/run.env
$EDITOR $P/runs/$RUN/run.env          # RUN3_NAMESPACE, RUN3_BUCKET, CHATBOT_BOARD_URL, CHATBOT_CHANNEL, OPENAI_BASE_URL, MODEL_WORKER, ...
```

## 1. Start order: world -> controller -> waves (inside the controller) -> workers (by the controller)

```sh
sbatch -J flock-world-$RUN -o $P/runs/$RUN/logs/%x-%j.out $P/slurm/world.sbatch $RUN
# the controller job waits for $P/runs/$RUN/hosts/world.host and GET /manifest (no deadline, re-reads the file, logs every 5 min)
sbatch -J flock-ctl-$RUN   -o $P/runs/$RUN/logs/%x-%j.out $P/slurm/controller.sbatch $RUN
squeue --me -o "%.10i %.20j %.8T %.10M %.20R"
```

`-J` and `-o` are passed on the command line because `#SBATCH` lines cannot contain the run name;
without them the defaults `flock-world` / `flock-ctl` and `runs/_slurm-logs/` apply. Keep the
names per run: `--dependency=singleton` then holds a second world or controller of the same run in
PENDING (`Dependency`) until the first one ends. The first node
takes 5-8 min to boot (the partition autoscales from 0).

The controller job, once `/healthz` answers on its node: copies `experiments/run3` to
`<run>/experiment/` (first start only), renders `spec.slurm.local.yaml` with
`render.py --variant ${RUN3_VARIANT:-jobs} --compute slurm` (`RUN3_VARIANT=smoke` for the 30-min smoke;
world.url = `http://${WORLD_HOST}:7860`, followed via `FLOCK_WORLD_HOST_FILE`, so leave `RUN3_WORLD_URL` unset;
bucket, worker slots) and sets
`compute.kind` to `slurm` (the job fails fast unless the venv has `flock.controller.slurm`: today's
`main.py` would otherwise silently fall back to the HF Jobs backend), then `flock apply` when the swarm is not on the controller yet (replicas 0:
swarm, kickoff, world check), else `waves.py pin`; then `waves.py run`. The whole step runs in a
background subshell, so a signal is handled even while `flock apply` (preflight) runs. Workers
appear as Slurm jobs named by worker id once the first wave is due (the Slurm jobs client in
flock does the `sbatch`).

Talk to the controller from a login node:

```sh
export FLOCK_URL=http://$(cat $P/runs/$RUN/controller.host):7860
set -a; . $P/secrets/secrets.env; set +a         # FLOCK_TOKEN (never echo it)
$P/venv/bin/flock status                         # or: curl -s $FLOCK_URL/healthz
cd $P/runs/$RUN/experiment && $P/venv/bin/python waves.py status --spec spec.slurm.local.yaml
$P/venv/bin/python waves.py add 100 --spec spec.slurm.local.yaml      # after the schedule
```

If the "waves step" failed in the controller log (for example the apply preflight), the controller
keeps running; fix the cause and run the same commands by hand from `<run>/experiment/`
(`flock apply spec.slurm.local.yaml`, then `waves.py run --spec spec.slurm.local.yaml --waves waves.yaml --log ../logs/waves-$RUN.jsonl` in tmux).

## 2. Stop everything cleanly

Order matters: drain the agents first, then remove the swarm (the controller cancels the worker
jobs), then the controller (final state push), then the world (final `ws/` sync).

```sh
export FLOCK_URL=http://$(cat $P/runs/$RUN/controller.host):7860; set -a; . $P/secrets/secrets.env; set +a
SWARM=$($P/venv/bin/python -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["swarm"])' $P/runs/$RUN/experiment/spec.slurm.local.yaml)
pkill -f "waves.py run" 2>/dev/null                              # if you run one by hand
$P/venv/bin/flock loop $SWARM --type agent --stop                 # final round, then the loop ends
$P/venv/bin/flock loop $SWARM --type agent-think --stop           # if agent-think has replicas
$P/venv/bin/flock remove $SWARM                                   # drains and cancels the worker jobs (scancel by the client)
squeue --me                                                       # wait until no worker job is left
scancel -n flock-ctl-$RUN                                         # SIGTERM -> controller final push (bounded, 10 s), exit
WJOB=$(squeue --me -h -n flock-world-$RUN -o %i)
scancel --signal=TERM --batch $WJOB                               # signal only: world stops, final ws sync (up to 10 min), exits by itself
squeue -j $WJOB                                                   # wait until it is gone; `final ws sync ok` in its .out
```

Do not stop the world with a plain `scancel`: it sends SIGTERM and then SIGKILL after
`KillWait=30 s`, which kills a long final sync. The script detects that case (the job is no longer
RUNNING when the TERM handler runs) and bounds its sync to a best-effort 25 s.

All three scripts also carry `--signal=B:TERM@<grace>`, so a time-limit end sends SIGTERM to the batch
shell before the kill and the same clean-stop traps run.

Emergency: `scancel --me` kills everything; you lose up to 60 s of controller writes, 180 s of
agent sessions and 300 s of world edits (minus what the world's 25 s best-effort sync saves) (the world's `best/` mirror is always current).
`control/*` markers from the README (`control/stop`, `control/add`, ...) belong to the single-job
bootstrap only; in this layout the `flock` CLI and `waves.py` are the controls.

## 3. Logs

| what | where |
|---|---|
| harness lines (`log()` of each script) | `runs/<RUN>/logs/flock-ctl-<RUN>-<jobid>.out`, `flock-world-<RUN>-<jobid>.out` (or `runs/_slurm-logs/` without `-o`) |
| controller process | `runs/<RUN>/logs/controller-<RUN>.log` (appended across requeues) |
| waves runner | `runs/<RUN>/logs/waves-<RUN>.out`, one JSON line per change in `waves-<RUN>.jsonl`; first `flock apply` in `apply-<RUN>.log` |
| world process | `runs/<RUN>/logs/world-<RUN>.log` (server output; `world-inner.sh` lines are in the world's `.out`) |
| workers | `runs/<RUN>/logs/<worker id>-<jobid>.out` (`FLOCK_SLURM_LOG_DIR`; the controller's `logs_tail` reads these) |
| agent state | the bucket: `<BUCKET>/<swarm>/<type>/<agent>/`, world `<CHATBOT_BUCKET>/ws`, controller `<RUN3_BUCKET>/controller/<RUN>` |

```sh
tail -f $P/runs/$RUN/logs/flock-ctl-$RUN-*.out $P/runs/$RUN/logs/waves-$RUN.out
sacct -X --me -S today -o JobID,JobName%30,State,Elapsed,NodeList,ExitCode
```

## 4. Worker RSS and CPU (size the next run)

```sh
sacct --me -S today -o JobID,JobName%30,State,Elapsed,MaxRSS,TotalCPU,AllocCPUS -X   # finished jobs
sacct -j <jobid> -o JobID,JobName%30,MaxRSS,TotalCPU,Elapsed                        # one job (step .batch has MaxRSS)
sstat -j <jobid>.batch -o JobID,MaxRSS,AveCPU                                      # a RUNNING job
```

Fact sheet §4: if `TotalCPU / (Elapsed x AllocCPUS)` stays under 50 % for workers, go to 3 CPUs /
5.5 GB (`FLOCK_SLURM_CPUS`, `FLOCK_SLURM_MEM` in `controller.sbatch`). MaxRSS includes the page
cache. Worker RSS for 25 run-3 agents was never measured: read it from the first smoke.

## 5. Test a Spot kill by hand

The node watcher sends SIGUSR1 to the batch script AND to every process of the job's steps
(the same as `scancel --signal=USR1 --full`), waits 30 s (fixed), then cancels the job.

```sh
scancel --signal=USR1 --full <jobid>        # realistic: what the watcher does (shell + every process)
scancel --signal=USR1 --batch <jobid>       # the batch shell only: tests the shell path, NOT the children
```

`--batch` alone never shows whether the children survive the direct signal: use it only to debug a
handler. To also reproduce the 30 s kill: `scancel --signal=USR1 --full <jobid>; sleep 30; scancel <jobid>`
(the handlers are sized to end in ~20 s, so the kill should find nothing left to do).

What to observe after `--full`:

- **controller**: the `.out` shows `SIGUSR1 ... final push` then `requeue` within ~19 s;
  `controller-<RUN>.log` has `final push: ok` (flock-controller and `waves.py` did NOT die at the
  signal itself: no traceback, no "Killed" before the push);
  `scontrol show job <jobid> | grep -E 'Restarts|JobState'` -> Restarts=1, PENDING (BeginTime), then
  RUNNING on a new node; `controller.host` changes; workers lose the controller until they are
  relaunched (fact sheet §5 "Stable addresses"); `waves-<RUN>.out` continues the schedule (`pin`).
- **world**: the `.out` shows `SIGUSR1` (host), then `world-inner` lines: `SIGUSR1`, possibly
  `still alive after 5 s: SIGKILL`, `final ws sync ok`, `exit after SIGUSR1`; then `requeue`, all
  within ~23 s; `world-<RUN>.log` shows a shutdown started by the script, not a
  death at the signal; the new job logs `restored ... files` from the bucket.
- **worker**: the `.out` shows `forwarding to the flock-worker step`, and `SIGTERM` 3 s later if it
  is still alive (`scancel --signal=TERM <job>.0` in the log); with the maintainer's handler the worker log shows its drain + sync, then the exit (no
  requeue); the controller relaunches the slot; check the agent's `sessions/` in the bucket and
  `resume_prompt` in its transcript. Without that handler flock-worker ignores USR1 and exits on
  the SIGTERM (no sync: up to 180 s of agent sessions lost).

## 6. OPEN QUESTIONS (not settled from the code; decide before the first smoke)

1. **Spot budget is 30 s, not 2 min.** The watcher kills the job 30 s after SIGUSR1
   (cpu-partition.md). The fact sheet (§5) sized the handlers for 2 min and the worker handler
   for ~90 s. These scripts bound the controller push at 10 s (`FLOCK_FINAL_PUSH_TIMEOUT`; whole
   USR1 path ~19 s) and the world's last sync at 12 s (whole path ~20 s); the worker gets SIGTERM
   3 s after USR1, and its drain+sync must finish within ~25 s or accept the loss of what did not
   sync (180 s window). Ask the maintainer to size the handler for 25 s.
2. **`compute: slurm` in the spec.** `render.py --variant jobs` writes `compute: hf-jobs`;
   `FLOCK_COMPUTE=slurm` makes the controller refuse any other kind (api.py:681), and
   `ComputeSpec.kind` is still `Literal["hf-jobs","local"]`. `controller.sbatch` rewrites the line
   to `slurm` (`RUN3_COMPUTE_KIND`; `hf-jobs` is refused) and exits unless the venv has
   `flock.controller.slurm`; confirm the final kind name and whether render.py gets a
   `slurm` variant instead.
3. **The Slurm client contract is settled at flock `origin/run3` fc22785** (`slurm.py`):
   `sbatch --parsable -p -c --mem --time --no-requeue -J <worker id> --comment -o <log_dir>/<worker id>-%j.out
   worker.sbatch <launch_dir>/<worker id>-<incarnation>.env`. The env file holds NO secret values
   (worker.env.example): WORKER_ID, WORKER_INCARNATION, SWARM, TYPE, DEADLINE, DRAIN_WINDOW, BUCKET,
   RUN, WORK_ROOT, FLOCK_LLM_HOME, FLOCK_URL, FLOCK_CTL_HOST_FILE, FLOCK_VENV, FLOCK_SECRETS_FILE,
   PYTHONUNBUFFERED. `worker.sbatch` takes every secret from `secrets.env` and overrides
   WORK_ROOT/FLOCK_LLM_HOME (`/tmp/flock/<id>/...` in the file) with the container's
   `/work/flock/<id>/...`. **The venv is still at 572da85** (no `flock.controller.slurm`, no
   `compute.types`): reinstall flock from fc22785 or later before the first smoke.
4. **Controller host file.** `controller.sbatch` exports `FLOCK_SLURM_HOST_FILE=<run>/hosts/controller.host`,
   so the client writes `FLOCK_URL='http://${CTL_HOST}:7860'` and `FLOCK_CTL_HOST_FILE`. The worker's
   re-read of that file is still pending in flock; until it lands, a controller that moves host
   means a fleet relaunch after 180 s. `FLOCK_WORLD_HOST_FILE=<run>/hosts/world.host` is exported
   for the world's host (PROVISIONAL name: flock is adding the re-read). The host files sit in
   `<run>/hosts/` (not the run dir) so the worker's container can mount the directory ro and still
   see a rewrite (a file bind mount keeps the old inode).
5. **Health endpoint**: `GET /healthz` on 7860, no auth (api.py:654, on `app`, not the authed router)
   is what the script polls. Confirmed by reading; not run.
6. **Waves command**: `python waves.py run --spec <rendered spec> --waves <schedule> --log <jsonl>
   --interval 30` with `FLOCK_URL`/`FLOCK_TOKEN` in the env, from the experiment folder (the spec's
   `prompts/`, `tools/` are relative). It needs the swarm to exist (`update_swarm` raises
   "no swarm ... `flock apply`"), hence the apply step. Not run.
7. **`flock world serve` has no config flag**: the only knob path is `CHATBOT_CONFIG` (chatbot3.py:82),
   so a run-specific `world.yaml` is generated. Confirm 14 is the wanted `runs.parallel` for 16 CPUs,
   and that the sed-free YAML rewrite keeps the comments the operator expects (it does not: it is
   regenerated from the parsed YAML).
8. **Preflight on the first `flock apply`** probes the vLLM endpoint and the world from the controller's
   node. The vLLM router answered 401 to an unauthenticated `/v1/models` from a CPU node (probe
   2026-10-24); with `OPENAI_API_KEY` set it should pass. If not, `RUN3_APPLY_ARGS=--skip-preflight`.
9. **`hostname` resolution**: short names `ip-10-53-x-y` resolved across CPU nodes in the probe; the
   scripts use `$(hostname)` for `FLOCK_URL` and `world.url`. Confirm from a GPU node too if the
   vLLM side ever needs to call back (it does not today).
10. **`scontrol requeue` from inside the trap** needs the Slurm commands on the compute node's PATH
    (the probe's trap assumed so). The controller job also needs `sbatch`, `squeue`, `sacct` there for
    the Slurm client; `controller.sbatch` fails fast if they are missing.
11. **Worker SIGTERM.** In job mode the worker has no SIGTERM handler (worker/main.py:635-644): a
    `scancel` from the controller's `cancel` kills it without a sync, as `hf jobs cancel` does today.
    A plain `scancel` delivers TERM straight to flock-worker in the container (tested); the
    maintainer's handler should cover TERM and USR1 and be idempotent (USR1 arrives twice).
12. **Preemption as a planned restart** (fact sheet §7 item 3): a Spot kill today counts as a crash
    (backoff 30 s doubling, `crashlooping` after 5). Needs the reconcile change; nothing in these
    scripts can work around it.
13. **`runs/<RUN>/world/world.yaml` and `experiment/` persist across requeues on /fsx** (wanted: the waves
    start time in `spec.slurm.local.yaml.waves.json` continues the schedule). Delete `experiment/`
    to start a run from a fresh copy of the repo folder.
14. **Interviewer size.** flock has no per-type env override; per-type sizes exist only in the spec
    (`compute.types.<type>`, which wins over `FLOCK_SLURM_*`). `controller.sbatch` writes
    `compute: {kind: slurm, types: {interviewer: {cpus: 2, mem: 3800M}}}` into the rendered spec
    (`RUN3_INTERVIEWER_CPUS`/`_MEM` in run.env). Needs the fc22785 venv; agents keep 4 CPU / 7G.
15. **Self-requeue vs the watcher's cancel** (review finding 1): if the watcher cancels by job id
    30 s after USR1, it may cancel the job we just requeued. Ask how the watcher kills before
    relying on the requeue.
16. **The world's crash-path sync bound** uses `SLURM_JOB_END_TIME` (present in the container,
    tested); a time-limit extension with `hfctl` is not reflected in it.
