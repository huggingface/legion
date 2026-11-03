# Live changes: `flock update`, `flock run --watch`

Everything an experiment customises lives in its folder, next to the spec: `spec.yaml`, the
prompt files it names, `tools/`, the comms protocol files. flock reads them at runtime, and
they can change while the swarm runs. Two commands do this:

- `flock update` applies an edited folder to the running swarm.
- `flock run --watch` does the same on every save, locally.

## `flock update`

```console
$ flock update experiments/calc          # the folder, or its spec.yaml
tools: 3f9c0e2a7b61d4c8 -> 9a1b2c3d4e5f6a7b (+mul)
prompts changed: calc (system_prompt); comms (protocols_text)
types.calc.replicas: 2 -> 3
```

It reads the folder exactly as `flock apply` does: the spec, `system_prompt_file`, the loop
prompt files, the comms kickoff and protocols, and the tools folder. It checks the folder,
compares it with the spec the controller is running and applies it. The output is one line per
change: the tools version (old -> new, with the tools added and removed), the prompts changed
per type, then every other key as `path: old -> new`. When nothing changed it prints
`no changes` and applies nothing.

Replicas: `flock update` keeps the running replica count of every type that already exists
(a type with `replicas: 20` in the file and 300 running stays at 300) and prints
`agent: replicas kept (spec 20, live 300)`. `--replicas spec` applies the file's counts
instead. `flock update` does this client-side; `PUT /swarms/{swarm}` applies the replicas it
is sent unless the request asks `?replicas=keep` (`flock apply`, `flock scale` and `flock run`
send the spec as is).

Lists print element by element (`types.agent.tools: +world:get_pr -world:foo`); only long
scalar strings are shortened. `world.endpoint`, `manifest` and the credential fields are
written by the controller and never listed; after the update the world's tool changes print
as `world tools: +a -b`.

`flock update --check` runs every check the controller runs on an apply (tools against the
bundle and the live world manifest, world, comms channel, harness changes) and prints the
diff, storing and posting nothing (`PUT /swarms/{swarm}?dry_run=1` returns the resolved
spec). It exits 1 with the reason when the update would be refused. A world started for the
check is stopped again.

A refused update exits 1, prints the reason and applies nothing. The reasons are the ones
`flock apply` gives: a spec error, a tools folder error, an unknown tool name, or a 400/409
from the controller. On a running swarm the controller also refuses to change the harness of a
type that has agents. Use a new type name for that, or remove the type first.

Preflight runs only when the update adds a provider-backed model the running spec does not
have (`--preflight` / `--skip-preflight` to choose). Locally, `flock update` finds the run
through `.flock/current`, like the other commands do.

## Env files, overlays and `--ref`

`flock apply`, `flock update` and `flock run` render the spec as they read it, so one folder
serves several clusters without a generated copy:

```console
$ flock apply experiments/run4/spec.yaml --env runs/run4/run.env --overlay experiments/run4/spec.slurm.yaml
$ flock update experiments/run4 --env runs/run4/run.env --overlay experiments/run4/spec.slurm.yaml --ref origin/run5
```

- `--overlay FILE` (repeatable, in order) is deep-merged over the spec: mappings merge key by
  key, lists replace, `null` deletes a key (`world: {dir: null, command: null, url: ...}`
  turns a local world into a remote one). Relative paths in it (prompts, `world.dir`) stay
  relative to the spec file.
- Then every string value, inline prompts and commands included, may use `${VAR}` and
  `${VAR:-default}` with an UPPERCASE name (as in `sh`, an empty value takes the default);
  lowercase placeholders such as the loop's `${spend}` are left alone. Sources, later winning: each `--env FILE` in order (`NAME=value` lines,
  `export ` and `#` comments allowed, no shell expansion), then the process environment. A
  value that is one placeholder takes its YAML type (`replicas: "${REPLICAS:-20}"` is the
  integer 20); otherwise it is plain text. Keys and prompt files are not substituted. An unset
  variable without a default refuses the spec, naming every such variable. `$${` is a literal
  `${` (any other `$$` stays as written); `${WORLD_HOST}` is always left for the controller
  (docs/slurm.md), even when the environment sets it. `providers.*.base_urls` also takes a
  comma-separated string, so `base_urls: ${OPENAI_BASE_URLS:-}` works (empty: none).
- `--ref GIT_REF[:PATH]` (apply and update) exports the repo holding the given folder at
  `GIT_REF` (`git archive`) into a temp dir and reads the spec from there: the folder's own
  path in the repo, or `PATH` (repo-relative, a folder or a spec file). Overlays inside the repo
  are read at the ref too; `--env` files are read where they are. The source sent is
  `{git_sha, ref, dirty: false, by}`. `--check` works with it.

## Generations

Every apply that changes the spec (`flock apply`, `flock update`, `flock scale`, the first
apply of `flock run`) gives the swarm a new **generation**: 1 for a new swarm, then +1. An
apply that changes nothing keeps it. `flock status` shows it as `gen N` on the swarm line.

The controller keeps each generation's applied spec with its UTC time, the diff lines (as
`flock update` prints them) and its source. The CLI sends the source in the `X-Flock-Source`
header: `by` from `$USER` and, when the spec's folder is in a git repo, `git_sha`, `ref` (the
branch) and `dirty` (the folder has uncommitted changes). Each generation is also a
`spec_applied {generation, source, diff_summary}` event on the swarm stream.

The history is kept in the controller's store (no route reads it). Removing the swarm drops
it.

## Control-plane events

Besides `spec_applied`, the swarm stream (`flock events <swarm>`) carries what the reconcile
loop does, as `{target, detail}`: `launch`, `launch_failed`, `launch_refused`, `crash`
(detail: stage, message, backoff), `crashloop`, `planned_restart` (job timeout, drain,
preemption), `cancel`, `retire`, `remove_worker`. Per-agent placement
(`assign`, `unassign`) and conditions repeated every pass (`workers_full`, the launch rate
budget) are not written. The health sampler adds `world_down` and `world_up`; `flock status
--watch` shows the last ten of these events ([health.md](health.md)).

World knob changes made through the controller (`flock world set`; docs/world.md, "Admin
API") are recorded as `world_knob {name, old, new, by, reason}`. They take effect in the
world at once and survive its restarts; they are not part of the spec.

The ramp adds `ramp_step`.

## Ramp (`replicas_ramp`)

```yaml
types:
  agent:
    replicas: 0
    replicas_ramp: {start: 20, add: 20, every: 10m, max: 300}
```

The reconciler raises the type's replicas to `start` on the first pass that sees the ramp.
It then adds `add` every `every`, up to `max`. Replicas change only when a new step is due,
and never go down. An operator's `flock scale` between steps, up or down, stays until the
next step asks for more. Each raise rewrites the stored spec and writes `ramp_step {type,
from, replicas, step, max}`. The start time and the last step are stored, so a controller
restart keeps the schedule. `flock update` keeps the live replicas by default. This replaces
`experiments/run4/waves.py run` (fixed schedule only; there is no load guard).

## Per-type concurrency (`max_working`)

`types.T.max_working: N` admits at most N turns of the type at once, swarm-wide. Paused agents
never take a slot. It uses the
same admission as a provider's `max_working`: prompts wait in the controller, and free slots
go to the oldest pending prompt first. When both caps are set, a prompt needs a slot in both.
A type with it runs its `turn_end` loop on the controller (docs/loop.md).

## `flock run --watch`

`flock run spec.yaml --watch` polls the files the spec references and the tools folder every
second. Once a change has been quiet for a second, it runs `flock update`. The terminal view
gets a `live updates` pane with the update's lines. A broken save (half-written YAML, a
syntax error in a tool) shows the refusal and `not applied: the swarm keeps running as it
was`. The next good save applies. Watching is off by default.

## What takes effect when

| change | takes effect |
|---|---|
| tools folder (a new content hash) | each agent's next turn. The worker fetches the version within a heartbeat (30 s on jobs, 5 s locally); a turn in flight keeps its version (docs/tools.md) |
| `system_prompt` / `system_prompt_file` | each agent's next turn boundary. The worker rewrites `CLAUDE.md` / `AGENTS.md` and the `llm` harness rebuilds its system message |
| `loop:` settings and loop prompt files | the next loop tick (docs/loop.md) |
| `comms.rules`, `comms.protocols` | the next heartbeat; the instructions' board section at the next turn |
| `comms.kickoff_file` | never: the kickoff is posted once per channel |
| `limits.budget_usd`, `limits.alive`, `spawn:` | at once (the controller) |
| `replicas` | the next reconcile pass |
| other type fields (`max_turns`, `max_budget_usd`, `tool_choice`, `turn_timeout`, ...) | the next turn boundary. A CLI harness takes the per-prompt limits, `model`, `env` and `secrets` at its next start, after `idle_grace` |
| a type's `harness` (with agents), `compute`, `swarm` | refused |

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

CLI harnesses (`claude`, `codex`, `pi`) read their instructions file only at start. For them
the note says which file to re-read (`your instructions in CLAUDE.md changed; re-read them`,
`your tools changed; re-read the Tools section of AGENTS.md`).

A running `pi` agent takes a provider's new endpoint list (`base_urls`), model id or
reasoning level at its next Pi start: models.json is rewritten then, and a transport error
or turn timeout restarts Pi. A change of the provider's `api_key_env` needs a worker restart.
