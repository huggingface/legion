# Loops, final rounds and the budget stop

A type's `loop:` keeps its agents prompted on a cadence until a condition fails, then runs one
final round and leaves them idle. `limits.budget_usd` is a hard stop for the whole swarm.

Code: `models.py` (`LoopSpec`, `LoopMeta`, `LoopStartCommand`, `LoopStopCommand`,
`LoopStatus`), `controller/loop.py` (the engine), `worker/agents.py` + `worker/loop.py` (the
worker side and rendering), `cli.py` (`flock stop --final`, `flock status`).

## Spec

```yaml
loop:                          # a Harbor spec's run loop; a role's own `agents.roles.<r>.loop` replaces it
  prompt: "Round {round}: claim and paint. Repeat until the result says you are done."
  # or prompt_file: prompts/round.md (read by `flock launch`, relative to the spec)
  every: turn_end            # turn_end (default) | a duration: 30s, 2m
  on: [message_board.mention]        # wake-ups from the board resource (see "Wake queue")
  wake:                      # how wake-ups spend rounds (all optional; these are the defaults)
    min_gap: 30s             # per agent, between two wake prompts; wakes in between coalesce
    max: 10                  # wake prompts per agent per run of the loop (null: no cap)
    # window: 10m            # set: `max` per this window instead of per run
    counts_as_round: false   # true: a wake prompt is a round and counts toward max_rounds
  while:                     # all set conditions must hold
    until: 2h                # wall time since the loop started (type-wide); or a clock time: 2026-10-28T20:00Z
    max_rounds: 400          # loop prompts per agent
    budget_usd: 12           # the type's total cost (type-wide)
    agent_usd: 0.05          # per agent; or a map by model alias / id: {deepseek: 3.60, glm: 2.10}
  final:                     # optional: one last prompt per live agent once the loop stops
    prompt: "Final round: submit your best result with submit_final."
    # or prompt_file: prompts/final.md
    reserve_usd: 0.60        # held back from limits.budget_usd for this round
limits:
  budget_usd: 15                 # hard stop for the swarm (below)
```

| field | meaning |
|---|---|
| `prompt` / `prompt_file` | the loop prompt, a template rendered on the worker (placeholders below); one is required |
| `every` | `turn_end`: the next loop prompt as soon as the agent's turn completes. A duration: an idle agent gets the next one that long after its previous loop prompt was sent |
| `on` | wake sources (`message_board.mention`, ...: the board resource's events): a wake queues the loop prompt early (2 s debounce). `world.*` needs a world. Write it plain: `flock launch` reads specs with YAML 1.2 booleans (only `true`/`false`), so `on:` stays a key |
| `wake.min_gap` | per agent: at least this long between two wake prompts (default `30s`; `0s` turns it off). Wakes that arrive in the gap are coalesced into one prompt when it ends |
| `wake.max`, `wake.window` | per agent: at most `max` wake prompts (default 10) per run of the loop, or per `window` when set; `null` for no cap. Past it, wakes are ignored (`wake()` returns false) |
| `wake.counts_as_round` | default `false`: a wake prompt is not a round. It carries the current `{round}`, does not advance it and does not count toward `max_rounds`; `wake.max` caps it instead. `true`: a wake prompt is the next round |
| `notices` | default `false`. `true`: the round prompts name the board notifications since the agent's last round in one line after the rendered prompt (see "Notices") |
| `while.until`, `while.budget_usd` | type-wide: when either fails the loop stops for every agent of the type |
| `while.max_rounds`, `while.agent_usd` | per agent: the agent leaves the loop; when every agent has left, the loop stops |
| `final.prompt` / `prompt_file` | sent once to every live agent when the loop stops (any reason except a stop with `final: false` (`POST /swarms/{swarm}/loop`) and the hard budget stop) |
| `final.reserve_usd` | subtracted from `limits.budget_usd` to get the point where loops stop; summed over the types that have a final round |

Every field is optional except a prompt; a type without `loop:` is never re-prompted.

### Placeholders

The worker fills these in `prompt` and `final.prompt`, at send time:

| placeholder | value |
|---|---|
| `{round}` | this prompt's round (1, 2, ...; the final prompt carries the last round) |
| `{elapsed}` | minutes since the loop started, rounded |
| `{wall}` | `while.until` in minutes, or `none` |
| `{spend}` | the agent's cost so far in USD, 3 decimals (worker-side, same price table as the controller) |
| `{cap}` | the agent's `while.agent_usd`, 2 decimals, or `none` |
| `{why}` | what sent it: `start`, `turn_end`, `every 120s`, `wake: message_board.mention`, `retry after error`, `final: max_rounds` |
| `{last_summary}` | the agent's last reply; `(none yet)` before any. A turn that stopped early (`max_turns`, `max_budget_usd`, an error) sets it to `previous turn ended (<reason>) after N tool call(s) without finishing (last calls: a, b, c)` instead, so the next prompt never repeats an older turn's summary |
| `{workspace}` | the agent's workdir at its last push: `12k files, 340 MB; largest: snaps/ 9k, src/ 2k` (sync excludes and `workdir.ephemeral` left out), or `not measured yet` |

Anything else in braces, and format specs such as `{cap:.2f}`, is left exactly as written, so
JSON and code in a prompt survive.

## How it runs

The controller runs a loop engine next to the reconcile loop: one tick per second
(`FLOCK_LOOP_INTERVAL`). A tick first checks the budget, then for each type with a loop
evaluates `while` and sends what is due. Every controller-sent prompt goes through the
bulk-prompt path (`POST /swarms/{s}/prompts` with `state: idle, skip_pending: true`): an agent
that is working or already has an undelivered prompt is skipped, so prompts never stack, and
every turn is admitted under the provider's `max_working` like any other prompt.

**Who re-prompts on `turn_end`** (`flock status` shows it as the mode):

- `worker` (provider without `max_working`): the controller sends one `loop_start` command per
  agent. The worker then sends the rendered prompt when the agent is
  idle and again each time a turn ends `completed` (or at `max_turns`), with no controller round trip (no idle gap
  between turns). It stops at `max_rounds`, at the `until` deadline, on
  a `loop_stop` command, or after 3 failed turns in a row (a failed turn is retried after 2 s).
  A turn that ends at the type's `max_turns` is not a failed turn, on either side. Each one is
  a `loop_max_turns` event.
  Each prompt it sends is a `loop_prompt` event, which is how the controller counts rounds. An
  agent idle for 30 s with no loop activity (a lost command, a relaunched worker) gets its
  `loop_start` again; the command is idempotent.
- `controller` (a duration `every`, or `turn_end` under `max_working`): the controller sends
  each round itself, on the `turn_ended` event or when `every` has passed. Under `max_working`
  this matters: a worker-side loop would keep its slot from turn to turn and starve the agents
  queued behind it, while controller-sent rounds queue FIFO with everybody else's. A `turn_end`
  round whose prompt never started a turn (deferred by a draining worker, or lost) is sent again
  after 30 s.

**Rounds.** A round is one loop prompt to one agent. The counter is per agent; `max_rounds`
counts loop prompts (not the final prompt, and wake prompts only with
`wake.counts_as_round: true`). In controller mode
an agent leaves the loop at `max_rounds` only once its last round's turn ended, so a queued
round is never dropped. When the controller does drop undelivered loop prompts (a loop stop, a
handover to the workers, the budget stop) their rounds are taken back: only delivered prompts
count.

**Stopping.** The loop stops when a type-wide condition fails (`until`, `budget_usd`, the
budget threshold), when every agent has left it (`max_rounds`, `agent_usd`,
`errors`; the reason is theirs if they agree, else `agents_done`), on `flock stop --final`, or
when `loop:` is removed from the spec (no final round then). Stopping drops the type's
undelivered loop prompts and sends `loop_stop` to worker-side loops.

`until` is a duration since the loop started (`2h`) or an ISO-8601 clock time
(`until: 2026-10-28T20:00Z`, or with an offset like `+02:00`; no zone means UTC). A clock time
ends the loop when `now >= until` whatever the start time, so the final round can be scheduled
for a wall-clock moment. `{wall}` is then the minutes from the loop start to that time.

**Undelivered prompts.** A worker that takes a loop prompt and then defers it (draining,
stopping) reports `prompt_deferred` with the prompt's `{kind, round}`: the controller undoes
what the prompt advanced and sends the same prompt again (the same round number, the final
prompt again), without counting another round. A final prompt that no worker took within 30 s is sent again too; both go through `state: idle, skip_pending`, so a
retry never joins a running turn or stacks on a queued prompt.

**Final round.** Then, if `final:` is set, every live agent gets the final prompt once: after
its worker confirmed the `loop_stop` (or 60 s passed), and only when it is idle, so the final
prompt never joins a loop turn. Agents that are dead are skipped. When every final turn has
ended the loop is `done` and the agents stay idle.

**Budget stop** (`limits.budget_usd`, swarm-wide, removed agents included; also checked at
spawn):

1. At `budget_usd - sum(final.reserve_usd)`: event `budget_stop {phase: final_reserve}`, every
   pending prompt of the swarm is dropped, every running loop stops (reason `budget`) and the
   final rounds run.
2. At `budget_usd`: event `budget_stop {phase: hard}`, loops and final rounds end, pending
   prompts are dropped, every placed agent gets `stop` (its harness is killed, it stays
   assigned), and the prompt routes answer 409 until the budget is raised.

While the hard stop holds, no pending prompt is delivered to any worker, a relaunched worker's
resume prompt is held as a pending prompt instead of riding in the assign, and any agent seen
working again (a worker that crashed before its `stop` arrived) gets `stop` again, at most every
30 s. Re-applying the spec with a budget above the cost lifts the stop (`budget_cleared`); held
prompts are then delivered. Cost is
what the controller's ledger knows, which is updated by heartbeats (every 30 s), so the overshoot
is bounded by about one heartbeat of spend of the working agents.

**Live settings.** `flock launch` with a changed `loop:` takes effect at the
next tick: controller-side settings (`every`, `while`, `final`) are read every tick,
and worker-side loops get a fresh `loop_start` with the new prompt and limits (`loop_updated`
event). The mode is recomputed on every tick from `every` and the provider's `max_working`, so a
re-apply can move the cadence: worker to controller sends `loop_stop {reason: mode_change}` and
the controller prompts an agent only after its worker confirmed (or 60 s passed); controller to
worker drops the type's queued loop prompts before the `loop_start`s go out, and since they
were never delivered their rounds are taken back, so the `loop_start` carries the rounds actually
sent. Removing `final`
while the final round is in flight ends it (`final_done`, note "final removed from the spec");
its final prompts not yet fetched by a worker are dropped, and one already fetched is not sent
again. Any end of a final round (also a stop with `final: false` during it, and the hard budget stop)
drops its undelivered final prompts.
A stopped loop stays stopped until the loop is started again (`POST /swarms/{swarm}/loop`; no CLI command). `limits.budget_usd` is live too.

**Restart.** Loop state is in the controller's SQLite (`loops`, `loop_agents`, `swarm_flags`),
so a controller restart resumes the loops; only queued wake-ups are lost.

### Pause and resume

A hold (`flock stop`) stops new work reaching some agents without ending their loop. Unlike
`flock stop --final` it resets nothing and runs no final round.

```
flock stop demo --type ds --reason "provider maintenance"      # one or more --type / --agent
flock stop demo --agent ds-0003 --interrupt --reason "look"    # also stop the turn in flight
flock stop demo                                                # no --type/--agent: every type
flock launch demo --type ds                                    # no --type/--agent: every hold
```

`flock stop demo --release` also frees the compute (docs/live.md "Lifecycle").

While an agent is held (its type or its name is held):

- the `commands` poll hands it no prompt: loop rounds, notices, hand prompts, spawn kickoffs and
  a relaunched worker's resume prompt stay queued and go out on the first poll after `resume`;
- the loop engine skips it: no round is sent or counted, loop errors ignore
  turns that end during the hold, and queued wakes stay queued (delivered after `resume`).
  Type-wide conditions keep running: `while.until` is wall-clock time and can end the loop
  during a hold (its final prompts then wait for `resume`);
- a worker-side loop gets `loop_stop {reason: held}` once; after `resume` it gets a
  `loop_start` of the same run, so its round count goes on;
- `--interrupt` sends `stop` (the hard budget stop's command) to its turns in flight, after a
  `loop_stop` to an active worker-side loop (else the worker would re-prompt the cut turn). A
  hand prompt whose turn `--interrupt` cuts is lost, as with `stop`.

Holds are stored in `swarm_flags` (a controller restart keeps them). Events on the swarm stream:
`paused {types, agents, reason, by, auto, interrupt}` and `resumed {types, agents, by, auto}`
(only the names a call actually changed). `GET /swarms/{s}` has
`status.held: {types: {T: {since, reason, by, auto}}, agents: {...}}` (null when nothing is
held) and `flock status` prints a `HELD` line per held type or agent. An agent held through its
type is resumed with its type, not with `--agent`.

API: `POST /swarms/{s}/pause {types?, agents?, interrupt?, reason?, by?}` and
`POST /swarms/{s}/resume {types?, agents?, by?}`; both answer `{swarm, held}` and 404 for an
unknown type or agent. The controller makes holds of its own when the world is down
(`auto: true`, docs/health.md); an operator hold is never replaced or lifted by those.

### Wake queue (`on:`)

`LoopEngine.wake(agent, reason)` is fed by the board resource (docs/comms.md): every `resource`
event a worker posts for it (`{resource: board, kind: mention, ...}`) becomes
`wake(agent, "message_board.mention")`. A wake queues the agent's loop prompt (`why: wake: <reasons>`)
once 2 s have passed without a new wake for that agent, or at most 10 s after the first, and
not before `wake.min_gap` has passed since the agent's previous wake prompt, through the same
`state: idle, skip_pending: true` path. Everything queued until then (a thread's worth of
replies) becomes that one prompt. By default the prompt is not a round (`wake.counts_as_round`):
`max_rounds` stays a budget for the cadence, and `wake.max` (per run, or per `wake.window`)
caps the wakes. A wake is accepted only for an agent in a running loop whose `on:` lists the
reason (or its prefix before `:`, so `message_board.mention:42` matches `message_board.mention`), under
`wake.max`, and never in a swarm stopped by the hard budget stop; wakes for an agent that is
working wait until it is idle and then give one prompt. A cadence prompt that goes out first
takes the queued wakes with it. A wake prompt a worker defers, or one dropped while still
queued, is taken back from the wake count, and from the round count when it counted one
(`LoopMeta.wake` marks it either way); it is not sent again as a wake.

Why: in a thread of five agents each reply woke the four others, and each wake was a full round,
so `max_rounds` went on board chatter in minutes. With the defaults one agent gets at most one
wake prompt per 30 s and 10 per run, and they leave the round budget alone. `flock launch` warns when an `on:` source can never
fire: a `message_board.*` source in a run without a board resource, or any other source (only
`message_board.*` is fed).

### Notices (`notices: true`)

Wakes say only why a prompt went out; in a `turn_end` loop they mostly fold into the next round.
With `notices: true` the controller keeps each agent's board events (msg id, thread, sender) with its loop state in the store (a controller restart keeps
them) and appends one line to its next loop prompt, whatever sent it (turn end, `every`, wake):

    New for you since your last round: @alice mentioned you in thread 12; 2 replies in your thread 40 (tan-jackal, ochre-spoon). Read them if useful.

Grouped by (kind, thread): mentions by people (`@name`: a sender that is not an agent of the
controller) first, then other mentions, then replies; at most 5 groups and 3 names each, the
rest counted (`+N more`). Named once, then cleared; no line when nothing came in. Not the
final prompt. Controller mode puts it in the prompt text. Worker mode (the worker
renders each round from its loop_start text) sends a loop_start with the line (same run: the
round count stays), and once a round carried it (a `loop_prompt` that came at least 5 s after
the command left the queue), another loop_start with the plain prompt, or the next line. The
key never reaches workers (`worker_type_spec`).

### Events

Swarm stream (`GET /swarms/{s}/events`, `flock status <swarm> --events`): `loop_started {type, mode,
every, restart}`, `loop_tick {type, prompted, rounds}` (only on ticks that sent
something), `loop_updated {type}`, `loop_stopped {type, reason, rounds_total, dropped}`,
`final_round {type, agent}`, `final_done {type, sent, done, skipped}`,
`budget_stop {phase, cost_usd, budget_usd, threshold_usd?}`, `budget_cleared`,
`loop_wake {type, agent, reason, events, coalesced, round, counted, wakes}` (a wake prompt was
sent: `reason` the wake sources, `events` the wakes folded into it, `coalesced = events - 1`,
`wakes` the agent's wake prompts so far; also in the agent's stream),
`loop_max_turns {type, agent, kind: loop|final, round, calls}` (a turn ended at
`max_turns`; also in the agent's stream).

Agent stream (from the worker): `loop_prompt {kind: loop|final, round, why}`
right before the turn it starts, `loop_stopped {reason, round, active}` when a worker-side loop
ends, and `turn_ended` carries `tool_calls` (tool calls in the turn). A wake
prompt outside the round count has `counted: false` in its `loop_prompt`. The controller also writes `final_round` into the agent's stream.

### Commands and API

- `PromptCommand.loop: LoopMeta` (optional, omitted from the wire when unset): `text` is a
  template; `LoopMeta {kind, type, round, started_at, wall_s, cap_usd, why, counted, wake}` is what
  the worker renders it with. `round: null` lets the worker number it.
- `LoopStartCommand {agent, text, meta, max_rounds, deadline, run}` and `LoopStopCommand {agent,
  reason}`: worker-side `turn_end`. `run` names one run of the loop: a new run (after `flock
  loop --start`) resets the worker's round counter, a repeat of the same run keeps it. Deploy the worker wheel with the controller.
- `GET /swarms/{s}/loop`: `LoopStatus` per type with per-agent detail (`rounds_total` counts rounds, `wakes_total` wake
prompts, so wakes and cadence rounds are apart; the same fields are in `run.json` `loops`); `POST /swarms/{s}/loop
  {type, action: start|stop, final: true}`; `TypeStatus.loop` and `SwarmStatus.budget_stop` in
  `GET /swarms/{s}`.

### CLI

```
flock stop demo --final --type ds                    # stop now, then the final round
# no final round, or starting a stopped loop again: POST /swarms/demo/loop (no CLI command)
flock status demo                                    # adds: loop ds: running (controller, every 120s), rounds 3-5, 0/3 agents done
flock status demo --events                           # the swarm stream
```

`--start` begins a new run of the loop: the clock (`until`, `{elapsed}`) and the round counters
start again. It is refused while the
budget stop is in effect.

## Examples

### A cadence loop

```yaml
loop:
  every: turn_end
  prompt: "Claim a task and work on it. Repeat until the result says you are done."
  while: {until: 2h, max_rounds: 400, budget_usd: 12}
```

Each agent gets the next round as soon as its turn completes, never a stacked prompt
(`skip_pending`), until the time, round or budget limit ends the loop.

### A board-driven loop with a final round

```yaml
agents:
  roles:                                    # a Harbor spec; models are the run's model ids
    ds: {count: 3, model: deepseek}
    glm: {count: 2, model: glm}
loop:
  prompt_file: prompts/round.md             # {round} {elapsed} {wall} {spend} {cap} {why} {last_summary} {workspace}
  every: 120s                               # an idle agent's next round
  on: [message_board.mention]               # a board mention wakes it early (2 s debounce)
  while: {until: 2h, max_rounds: 200, agent_usd: {deepseek: 3.60, glm: 2.10}}
  final: {prompt_file: prompts/final.md, reserve_usd: 0.60}
limits: {agent_max_turns: 14, budget_usd: 15}   # model calls per prompt; the swarm's budget
```

Final-round spend is exempt from `agent_usd` and bounded only by the reserve and the hard stop.

## Not supported yet

`on: world.*` is accepted by the model and refused by `flock launch` and `PUT /swarms` with
"needs a world (not in this version)". Not done: a `run_stopping` hook to a world, per-call
booking to tighten the budget overshoot, a `loop.start: manual` switch, and a swarm-level
`limits.final_reserve_usd` (the reserve is `final.reserve_usd` per type, summed for the swarm
threshold).


## Stop on task success

Set `stop_on_success: true` in a Harbor experiment and provide an operator-owned
`success_check: {service: puzzle, command: ...}`. The command runs in a trusted
Docker/Podman sidecar (not `main`) and returns JSON with a boolean `success` and
optional `evidence`. Commands must be read-only, bounded, and independent of agent-owned files.
A task with no check cannot use this policy.

Workers check every five seconds, with a five-second timeout, even between model turns.
Errors never count as success. The next authenticated heartbeat reports the first positive
result; the controller records `task_success`, interrupts active turns, blocks new work,
and releases workers. There are no final model turns. The decision survives restarts and
cannot be resumed; launch a new run to repeat the experiment. Default: disabled.

Status includes `success` (winner, trial, observation and acceptance times, evidence,
and verification). Environment success is distinct from native verification: trials close,
collect artifacts and run Harbor's verifier within the existing five-minute cleanup budget.
The winner passes verification only with native `reward == 1` and no trial exception.
Failures or missing verification are shown explicitly; release reports failed artifact syncs.
Actual usage includes detection delay and cleanup; interrupted requests can lack returned usage.
