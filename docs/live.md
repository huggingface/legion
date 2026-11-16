# Live changes: `flock update`

Everything an experiment customises lives in its folder, next to the spec: `spec.yaml`, the
prompt files it names, `tools/`, the comms protocol files. flock reads them at runtime, and
they can change while the swarm runs. `flock update` does this:

- `flock update` applies an edited folder to the running swarm.

## `flock update`

```console
$ flock update experiments/my-experiment   # the folder, or its spec.yaml
tools: 3f9c0e2a7b61d4c8 -> 9a1b2c3d4e5f6a7b (+mul)
prompts changed: calc (system_prompt); comms (protocols_text)
types.calc.replicas: 2 -> 3
```

It reads the folder exactly as `flock launch` does: the spec, `system_prompt_file`, the loop
prompt files, the comms kickoff and protocols, and the tools folder. (`system_prompt*` and
`comms:` come only from a hand-written swarm spec, not a Harbor spec.) It checks the folder,
compares it with the spec the controller is running and applies it. The output is one line per
change: the tools version (old -> new, with the tools added and removed), the prompts changed
per type, then every other key as `path: old -> new`. When nothing changed it prints
`no changes` and applies nothing.

Replicas: `flock update` keeps the running replica count of every type that already exists
(a type with `replicas: 20` in the file and 300 running stays at 300) and prints
`agent: replicas kept (spec 20, live 300)`. `--replicas spec` applies the file's counts
instead. `flock update` does this client-side; `PUT /swarms/{swarm}` applies the replicas it
is sent unless the request asks `?replicas=keep` (`flock launch` sends the spec
as is). `--replicas spec` never lowers a type with `replicas_ramp` (the ramp raised it): lower
`replicas_ramp.max` to stop it growing. `--check` also probes every provider-backed model and
lists the agents (and their spawned children) a lowered count or a removed type would delete.

Lists print element by element (`types.agent.tools: +world:get_pr -world:foo`); only long
scalar strings are shortened. `world.endpoint`, `manifest` and the credential fields are
written by the controller and never listed; after the update the world's tool changes print
as `world tools: +a -b`.

`flock update --check` runs every check the controller runs on an apply (tools against the
bundle and the live world manifest, world, comms channel, fixed Harbor settings) and prints the
diff, storing and posting nothing (`PUT /swarms/{swarm}?dry_run=1` returns the resolved
spec). Each changed value names where it came from: `(spec.yaml:41)`, `(run.env WORKER_MEM)`,
`(env WORKER_MEM)`, `(spec.yaml:12 default of WORKER_MEM)` or `(default)`. It exits 1 with
the reason when the update would be refused. A world started for the
check is stopped again.

A refused update exits 1, prints the reason and applies nothing. The reasons are the ones
`flock launch` gives: a spec error, a tools folder error, an unknown tool name, or a 400/409
from the controller.

Preflight runs only when the update adds a model, or sends one somewhere new (other URLs, router
provider, key or headers), compared with the running spec (`--preflight` / `--skip-preflight`
to choose). Locally, `flock update` finds the run
through `.flock/current`, like the other commands do.

## Env files and `--ref`

`flock launch` and `flock update` render the spec as they read it, so one folder
(one `spec.yaml` per run) serves several clusters without a generated copy or an overlay:

```console
$ flock launch my-experiment/spec.yaml --env runs/demo/run.env
$ COMPUTE_KIND=local flock launch my-experiment/spec.yaml --env runs/demo/run.env
$ flock update my-experiment --env runs/demo/run.env --ref origin/main --check
```

- Every string value, inline prompts and commands included, may use `${VAR}` and
  `${VAR:-default}` with an UPPERCASE name (as in `sh`, an empty value takes the default);
  lowercase placeholders such as the loop's `${spend}` are left alone. Sources, later winning: the process environment, then each `--env FILE`
  in order (`NAME=value` lines, `export ` and `#` comments allowed, no shell expansion). A
  value that is one placeholder takes its YAML type (`replicas: "${REPLICAS:-20}"` is the
  integer 20); otherwise it is plain text. Keys and prompt files are not substituted. An unset
  variable without a default refuses the spec, naming every such variable. `$${` is a literal
  `${` (any other `$$` stays as written); `${WORLD_HOST}` is always left for the controller
  (docs/slurm.md), even when the environment sets it. `inference.urls` also takes a
  comma-separated string, so `urls: ${VLLM_URLS}` works.
- `--ref GIT_REF[:PATH]` (apply and update) exports the repo holding the given folder at
  `GIT_REF` (`git archive`) into a temp dir and reads the spec from there: the folder's own
  path in the repo, or `PATH` (repo-relative, a folder or a spec file). `--env` files are read
  where they are. The source sent is
  `{git_sha, ref, dirty: false, by}`. `--check` works with it.

## Lifecycle: `flock launch` and `flock stop`

| Want | Command |
|---|---|
| start a new run | `flock launch FOLDER` (refused when the bucket already holds agents of that swarm name) |
| hold turns, keep compute | `flock stop S` (running turns finish; `--interrupt` cuts them; `--type`/`--agent` narrow it) |
| free the workers, keep everything | `flock stop S --release` |
| end the loops with their final round | `flock stop S --final [--type T]` |
| continue after either | `flock launch S` (or `flock launch FOLDER`; `--type`/`--agent` lift one hold) |
| bring a released Slurm run back held, to change its spec first | `flock launch FOLDER --hold`, `flock update FOLDER`, then `flock launch FOLDER` |

`flock launch FOLDER` creates the swarm when the controller does not have it (with
`If-None-Match: *`: of two launches racing, one creates and the other gets 412). When it has
it, `launch` resumes it, and refuses if the folder's spec differs from the stored one
(`replicas` aside, which the ramp rewrites): run `flock update` first. So one call either
creates or resumes and never changes the spec.

`flock stop S --release` holds every type, then drains every worker: running turns finish,
each agent's final push to the bucket is retried once, and a push that still fails is named in
the worker's last heartbeat (`sync_failed` on the swarm stream, the worker's `last_error`,
`status.released.unsynced`; a worker that ends unplanned mid-release, without that heartbeat,
puts all its agents there). That last heartbeat (`drained`) ends the worker's slot at once:
the controller does not wait for the scheduler to show the job ended. The command waits until
no worker runs, then exits 0 when every agent synced, or 1 listing the agents that did not.
Rows, names, spawn trees, loops, spec and history stay on the controller; nothing launches until
`flock launch S`, when each agent restores from the bucket by name.

A release then ends the run's controller and world jobs (`ops.down`; Slurm:
[slurm.md](slurm.md#deploying), HF Jobs: [deploy-job.md](deploy-job.md#deploy)), in this order:
a final controller checkpoint (`POST /checkpoint`; when it fails it is reported and nothing is
ended), the world's clean stop (SIGTERM: it saves and exits), then the jobs. A local run, a
controller the run has no job for, or one serving other swarms that are not released keeps
running. `flock launch FOLDER` with no controller answering (`ops.up`) starts them again, waits
until the controller answers (it restores its checkpoint) and the world serves, then creates or
resumes.
`flock launch FOLDER --hold` (Slurm, no controller up) stops there: it writes
`RUN_DIR/controller.hold`, and the controller job started with it restores every swarm held and
released, as `flock stop --release` leaves it (`FLOCK_HOLD_ON_START=1`), pushes that checkpoint,
then removes the file (a later requeue holds nothing). No worker starts, so `flock update` can
change the spec first (a new provider URL); a plain `flock launch` then resumes.

A controller that restarts restores its swarms from its own checkpoint (`state.db` and
`events/<swarm>/`, pushed every 60 s to `FLOCK_STATE_URL`; `FLOCK_STATE_RESTORE=0` skips it).
There is no takeover by another controller: a fresh launch whose bucket already holds the
swarm's agents is refused; restart that run's controller, or pick a new swarm name or bucket.

## Generations

Every apply that changes the spec (`flock launch`, `flock update`, the first
apply of `flock launch`) gives the swarm a new **generation**: 1 for a new swarm, then +1. An
apply that changes nothing keeps it. `flock status` shows it as `gen N` on the swarm line.

The controller keeps each generation's applied spec with its UTC time, the diff lines (as
`flock update` prints them) and its source. The CLI sends the source in the `X-Flock-Source`
header: `by` from `$USER` and, when the spec's folder is in a git repo, `git_sha`, `ref` (the
branch) and `dirty` (the folder has uncommitted changes). Each generation is also a
`spec_applied {generation, source, diff_summary}` event on the swarm stream.

The history is kept in the controller's store (no route reads it). Removing the swarm drops
it.

## Control-plane events

Besides `spec_applied`, the swarm stream (`flock status <swarm> --events`, names in docs/health.md "Events") carries what the reconcile
loop does, as `{target, detail}`: `launch`, `launch_failed`, `launch_refused`, `crash`
(detail: stage, message, backoff), `crashloop`, `planned_restart` (job timeout, drain,
preemption), `end` (the same in a released swarm: no relaunch), `cancel`, `retire`, `remove_worker`. Per-agent placement
(`assign`, `unassign`) and conditions repeated every pass (`workers_full`, the launch rate
budget) are not written. The health sampler adds `world_down` and `world_up`; `flock status
--watch` shows the last ten of these events ([health.md](health.md)).

World knob changes made through the controller (`python -m flock.world set`; docs/world.md, "Admin
API") are recorded as `world_knob {name, old, new, by, reason}`. They take effect in the
world at once and survive its restarts; they are not part of the spec.

The ramp adds `ramp_step`.

## Ramp (`replicas_ramp`)

```yaml
types:
  agent:
    replicas: 0
    replicas_ramp:
      start: 20                       # step 0, on the first pass that sees the ramp
      phases:                         # or one phase as `add: 20, every: 10m`
        - every: 5m
          add: 20
          until: 100
        - every: 10m                  # until: max
          add: 20
      max: 300
      stagger: 10s                    # the start raise adds one agent every 10 s (default)
      hold_if:
        waiting: 1
        request_s: 15
        above: 100
```

The reconciler raises the type's replicas along the schedule: `start`, then each phase adds
`add` every `every` until its `until`; the next phase starts where it stopped. Replicas change
only when a new step is due, and never go down. An operator's `flock update --replicas spec` between steps (up only on a
ramp type), stays until the next step asks for more. A lowered `max` caps later raises; it never
deletes agents. `stagger` spaces the agents of the `start` raise (a phase's own `stagger`,
default 0, those of its steps). Each step writes `ramp_step {type, from, replicas, step, max}`.
The start time, the last step and a staggered raise in progress are stored (`ramp:<type>`), so
a controller restart keeps the schedule. While the type is held (`flock stop`) or the swarm
released (`flock stop --release`), the clock stops: a resume does not jump the missed steps.

`hold_if` is the load guard: a step that takes the swarm's replicas above `above` waits while
any vLLM server of a provider with `metrics: true` had `waiting` or more requests queued in
every health sample of the last 2 minutes, or needs more than `request_s` seconds per request
(running requests / completions per second), or its metrics are missing or reset. It asks
again every 2 minutes and writes `ramp_held {type, want, holds, reason}`; from the third hold
in a row the event carries a `warning`. `flock update` keeps the live replicas by default.
It needs 3 health samples in the last 150 s, so keep `FLOCK_HEALTH_INTERVAL` under 50 s (the
default 30 s is fine); slower sampling holds every step ("metrics unavailable").

## Per-type concurrency (`max_working`)

`types.T.max_working: N` admits at most N turns of the type at once, swarm-wide. Paused agents
never take a slot. It uses the
same admission as a provider's `max_working`: prompts wait in the controller, and free slots
go to the oldest pending prompt first. When both caps are set, a prompt needs a slot in both.
A type with it runs its `turn_end` loop on the controller (docs/loop.md).

Local runs: after editing the folder, run `flock update FOLDER`; there is no watch mode in the CLI yet.

## Worker liveness

How long the controller waits on a silent worker, set under `compute:` (durations such as
`20s`, `2m`); unset keys take the backend's default. A worker is silent when neither a
heartbeat, a command poll nor an event post arrives from it.

| key | hf-jobs, slurm | local | meaning |
|---|---|---|---|
| `worker_unknown_after` | 90s | 15s | silent this long: the worker and its agents are `unknown` |
| `worker_dead_after` | 180s | 30s | silent this long: `dead`; its job is cancelled and relaunched (a crash). Must be longer than `worker_unknown_after` |
| `crash_backoff` | 30s | 2s | wait before relaunching a crashed slot, doubling per crash in a row (cap 15 min); also the first wait after a failed launch (cap 5 min) |

The local defaults are 3 and 6 missed heartbeats at the local 5 s interval (the jobs ratio at
30 s). A local worker is also checked on every reconcile pass, so a worker process that exits
is a crash at once, without waiting for `worker_dead_after`. Raise the keys with a type's
`HEARTBEAT_INTERVAL`, or for workers that block for long stretches.

## What takes effect when

| change | takes effect |
|---|---|
| `loop:` settings and loop prompt files | the next loop tick (docs/loop.md) |
| `comms.kickoff_file` | never: the kickoff is posted once per channel |
| `limits.budget_usd`, `limits.alive`, `spawn:` | at once (the controller) |
| `replicas` | the next reconcile pass |
| other type fields (`max_turns`, `max_budget_usd`, `turn_timeout`, ...) | the next turn boundary |
| `compute.worker_unknown_after`, `worker_dead_after`, `crash_backoff` | at once (the controller) |
| `compute.kind`, `swarm` | refused |
| a Harbor spec's `inference` | `max_working` at once (the controller); any other change restarts the type's workers (a `restart` event on the swarm stream per worker, at the next reconcile pass): they drain, and their agents resume on new workers with the new key, endpoints and allowlist |
| a Harbor spec's `task`, `tools`, `resources`, `environment`, `model`, `team_file`, `role_file`, `agents.role_files`, `limits.agent_*` or `limits.trial_hours` (its roles have agents) | refused: the workers prepared them at launch. Counts, loops, `workers:` and the run's limits change as above. A task branch that moved since the launch is not a change: the run keeps its pinned revision |

A turn boundary is a prompt that starts a new turn. A prompt that joins a turn in flight
changes nothing until that turn ends. Changed type specs reach the workers in the
heartbeat response. The worker sends its `spec_rev`, and the controller answers with the new
`type_spec` only when they differ.

**The note.** An agent that already had a turn gets one line in front of its next prompt for
each kind of change:

```
[flock] Since your last turn: your system prompt changed.

<the prompt>
```
