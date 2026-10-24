# Loops, final rounds and the budget stop

A type's `loop:` keeps its agents prompted on a cadence until a condition fails, then runs one
final round and leaves them idle. `limits.budget_usd` is a hard stop for the whole swarm. This
replaces the drivers that lived outside flock: pixel's `experiments/pixel/run/driver.py`
(re-prompt idle painters) and the Eliza harness's `swarm.py::main_loop` (rounds with a 2 s
debounce or a 120 s wake, per-agent caps, a final round). Design: `worlds-design.md` §9 (loop
and rounds), §4.4 (live settings), §4.7 (the Eliza yardstick), §13 (budget stop).

Code: `models.py` (`LoopSpec`, `LoopMeta`, `LoopStartCommand`, `LoopStopCommand`,
`LoopStatus`), `controller/loop.py` (the engine), `worker/agents.py` + `worker/loop.py` (the
worker side and rendering), `cli.py` (`flock loop`).

## Spec

```yaml
types:
  painter:
    harness: llm
    model: lfm
    replicas: 1000
    loop:
      first: "Run `python3 canvas status` once, then call done."   # optional, sent alone
      prompt: "Round {round}: claim and paint. Repeat until the result says you are done, then call done."
      # or prompt_file: prompts/round.md (read by `flock apply`, relative to the spec)
      every: turn_end            # turn_end (default) | a duration: 30s, 2m | rounds (needs a world)
      stagger: 50ms              # agent i gets its first prompt i * stagger after the loop starts
      on: [comms.mention]        # wake-ups from the board (see "Wake queue")
      wake:                      # how wake-ups spend rounds (all optional; these are the defaults)
        min_gap: 30s             # per agent, between two wake prompts; wakes in between coalesce
        max: 10                  # wake prompts per agent per run of the loop (null: no cap)
        # window: 10m            # set: `max` per this window instead of per run
        counts_as_round: false   # true: a wake prompt is a round and counts toward max_rounds
      stop_when: {all_agents_done: true}   # optional: done(finished: true) takes an agent out
      while:                     # all set conditions must hold
        until: 2h                # wall time since the loop started (type-wide)
        max_rounds: 400          # loop prompts per agent
        budget_usd: 12           # the type's total cost (type-wide)
        agent_usd: 0.05          # per agent; or a map by model alias / id: {deepseek: 3.60, glm: 2.10}
        idle_rounds: 3           # this many loop turns in a row without a state-changing call end an agent's loop
        # score_lt: {coverage_pct: 100}   # needs a world: refused in this version
      final:                     # optional: one last prompt per live agent once the loop stops
        prompt: "Final round: submit your best result with submit_final, then call done."
        # or prompt_file: prompts/final.md
        reserve_usd: 0.60        # held back from limits.budget_usd for this round
        add_tools: [submit_final, best_vote]   # carried to the worker for the tools layer
limits:
  budget_usd: 15                 # hard stop for the swarm (below)
```

| field | meaning |
|---|---|
| `prompt` / `prompt_file` | the loop prompt, a template rendered on the worker (placeholders below); one is required |
| `first` | sent alone to each agent before its first loop prompt; the loop prompt follows only after that turn ended (run 3's "install and paint folded into one turn" cannot recur) |
| `every` | `turn_end`: the next loop prompt as soon as the agent's turn completes. A duration: an idle agent gets the next one that long after its previous loop prompt was sent. `rounds`: needs a world |
| `stagger` | spreads the first prompts (`first`, or round 1 without it) over `i * stagger`, agents in name order |
| `on` | wake sources (`comms.mention`, `comms.thread`, ...): a wake queues the loop prompt early (2 s debounce). `world.*` needs a world. Write it plain: `flock apply` reads specs with YAML 1.2 booleans (only `true`/`false`), so `on:` stays a key and `"on":` quoting is no longer needed |
| `wake.min_gap` | per agent: at least this long between two wake prompts (default `30s`; `0s` turns it off). Wakes that arrive in the gap are coalesced into one prompt when it ends |
| `wake.max`, `wake.window` | per agent: at most `max` wake prompts (default 10) per run of the loop, or per `window` when set; `null` for no cap. Past it, wakes are ignored (`wake()` returns false) |
| `wake.counts_as_round` | default `false`: a wake prompt is not a round. It carries the current `{round}`, does not advance it and does not count toward `max_rounds`; `wake.max` caps it instead. `true`: a wake prompt is the next round, as before this setting |
| `stop_when.all_agents_done` | `llm` harness only (refused otherwise, or with `done_tool: false`): the `done` tool gets a `finished` argument (and is offered even without `tool_choice: required`). A loop turn that calls `done(finished: true)` takes the agent out of the loop (reason `done`); the loop stops when every live agent is out, then the final round runs |
| `context` | per-prompt context sources (`world.state`, `comms.recent:20`, `artifacts.recent:15`): accepted by the model, refused at apply until their feeders exist |
| `while.until`, `while.budget_usd` | type-wide: when either fails the loop stops for every agent of the type |
| `while.max_rounds`, `while.agent_usd`, `while.idle_rounds` | per agent: the agent leaves the loop; when every agent has left, the loop stops |
| `while.idle_rounds` | a loop turn is idle when it made no state-changing call. Not state-changing: `done`, the comms `read` and `search` tools, and folder tools marked read-only (`@flock.tools.readonly`, `readonly: true` in the manifest; docs/tools.md). Everything else counts, `bash` included, so with a CLI harness (tools through `flock-tool` in its shell) every call counts. The `llm` harness marks each call `readonly` from the tools version pinned for the turn (the one that ran it), so a re-apply mid-turn does not change the count |
| `final.prompt` / `prompt_file` | sent once to every live agent when the loop stops (any reason except `flock loop --stop --no-final` and the hard budget stop) |
| `final.reserve_usd` | subtracted from `limits.budget_usd` to get the point where loops stop; summed over the types that have a final round |
| `final.add_tools` | tool names the final round unlocks; carried in the final prompt's `LoopMeta` and its events for the tools layer to honour (not enforced here) |

Every field is optional except a prompt; a type without `loop:` behaves exactly as before.

### Placeholders

The worker fills these in `prompt`, `first` and `final.prompt`, at send time:

| placeholder | value |
|---|---|
| `{round}` | this prompt's round (1, 2, ...; `first` is round 0; the final prompt carries the last round) |
| `{elapsed}` | minutes since the loop started, rounded |
| `{wall}` | `while.until` in minutes, or `none` |
| `{spend}` | the agent's cost so far in USD, 3 decimals (worker-side, same price table as the controller) |
| `{cap}` | the agent's `while.agent_usd`, 2 decimals, or `none` |
| `{why}` | what sent it: `first`, `start`, `turn_end`, `every 120s`, `wake: comms.mention`, `retry after error`, `final: max_rounds` |
| `{last_summary}` | the text the agent last passed to `done` (`llm` harness), else its last reply; `(none yet)` before any. A turn that stopped without `done` (`max_turns`, `max_budget_usd`, an error) sets it to `previous turn ended (<reason>) after N tool call(s) without finishing (last calls: a, b, c)` instead, so the next prompt never repeats an older turn's summary |

Anything else in braces, and format specs such as `{cap:.2f}`, is left exactly as written, so
JSON and code in a prompt survive.

## How it runs

The controller runs a loop engine next to the reconcile loop: one tick per second
(`FLOCK_LOOP_INTERVAL`). A tick first checks the budget, then for each type with a loop
evaluates `while` and sends what is due. Every controller-sent prompt goes through the
bulk-prompt path (`POST /swarms/{s}/prompts` with `state: idle, skip_pending: true`): an agent
that is working or already has an undelivered prompt is skipped, so prompts never stack, and
every turn is admitted under the provider's `max_working` like any other prompt.

**Who re-prompts on `turn_end`** (`flock loop` shows it as the mode):

- `worker` (provider without `max_working`): the controller sends one `loop_start` command per
  agent once its `first` turn ended. The worker then sends the rendered prompt when the agent is
  idle and again each time a turn ends `completed` (or at `max_turns`), with no controller round trip (no idle gap,
  run 1 lost ~17 % of agent time there). It stops at `max_rounds`, at the `until` deadline, on
  a `loop_stop` command, or after 3 failed turns in a row (a failed turn is retried after 2 s).
  A turn that ends at the type's `max_turns` is not a failed turn, on either side: with
  `tool_choice: required` it is how turns normally end. Each one is a `loop_max_turns` event.
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
counts loop prompts (not `first`, not the final prompt, and wake prompts only with
`wake.counts_as_round: true`). In controller mode
an agent leaves the loop at `max_rounds` only once its last round's turn ended, so a queued
round is never dropped. When the controller does drop undelivered loop prompts (a loop stop, a
handover to the workers, the budget stop) their rounds are taken back: only delivered prompts
count.

**Stopping.** The loop stops when a type-wide condition fails (`until`, `budget_usd`, the
budget threshold), when every agent has left it (`max_rounds`, `agent_usd`, `idle_rounds`,
`errors`, `done`; the reason is theirs if they agree, else `agents_done`), on `flock loop --stop`, or
when `loop:` is removed from the spec (no final round then). Stopping drops the type's
undelivered loop prompts and sends `loop_stop` to worker-side loops.

**Undelivered prompts.** A worker that takes a loop prompt and then defers it (draining,
stopping) reports `prompt_deferred` with the prompt's `{kind, round}`: the controller undoes
what the prompt advanced and sends the same prompt again (a `first` again, the same round
number, the final prompt again), without counting another round. A `first` or final prompt that
no worker took within 30 s is sent again too; both go through `state: idle, skip_pending`, so a
retry never joins a running turn or stacks on a queued prompt.

**Final round.** Then, if `final:` is set, every live agent gets the final prompt once: after
its worker confirmed the `loop_stop` (or 60 s passed), and only when it is idle, so the final
prompt never joins a loop turn. Agents that are dead are skipped. When every final turn has
ended the loop is `done` and the agents stay idle.

**Budget stop** (`limits.budget_usd`, swarm-wide, removed agents included; before this PR it
was checked only at spawn):

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
is bounded by about one heartbeat of spend of the working agents; Eliza's `log_overshoot` makes
the same point.

**Live settings** (worlds-design §4.4). `flock apply` with a changed `loop:` takes effect at the
next tick: controller-side settings (`every`, `while`, `final`, `stagger`) are read every tick,
and worker-side loops get a fresh `loop_start` with the new prompt and limits (`loop_updated`
event). The mode is recomputed on every tick from `every` and the provider's `max_working`, so a
re-apply can move the cadence: worker to controller sends `loop_stop {reason: mode_change}` and
the controller prompts an agent only after its worker confirmed (or 60 s passed); controller to
worker drops the type's queued loop prompts before the `loop_start`s go out, and since they
were never delivered their rounds are taken back, so the `loop_start` carries the rounds actually
sent. Removing `final`
while the final round is in flight ends it (`final_done`, note "final removed from the spec");
its final prompts not yet fetched by a worker are dropped, and one already fetched is not sent
again. Any end of a final round (also `--stop --no-final` during it, and the hard budget stop)
drops its undelivered final prompts.
A stopped loop stays stopped until `flock loop --start`. `limits.budget_usd` is live too.

**Restart.** Loop state is in the controller's SQLite (`loops`, `loop_agents`, `swarm_flags`),
so a controller restart resumes the loops; only queued wake-ups are lost.

### Wake queue (`on:`)

`LoopEngine.wake(agent, reason)` is fed by the board (docs/comms.md): every `comms_mention` and
`comms_thread_reply` event a worker posts becomes `wake(agent, "comms.mention")` or
`wake(agent, "comms.thread")`. A wake queues the agent's loop prompt (`why: wake: <reasons>`)
once 2 s have passed without a new wake for that agent, or at most 10 s after the first, and
not before `wake.min_gap` has passed since the agent's previous wake prompt, through the same
`state: idle, skip_pending: true` path. Everything queued until then (a thread's worth of
replies) becomes that one prompt. By default the prompt is not a round (`wake.counts_as_round`):
`max_rounds` stays a budget for the cadence, and `wake.max` (per run, or per `wake.window`)
caps the wakes. A wake is accepted only for an agent in a running loop whose `on:` lists the
reason (or its prefix before `:`, so `comms.mention:42` matches `comms.mention`), under
`wake.max`, and never in a swarm stopped by the hard budget stop; wakes for an agent that is
working wait until it is idle and then give one prompt. A cadence prompt that goes out first
takes the queued wakes with it. A wake prompt a worker defers, or one dropped while still
queued, is taken back from the wake count, and from the round count when it counted one
(`LoopMeta.wake` marks it either way); it is not sent again as a wake.

Why: in a thread of five agents each reply woke the four others, and each wake was a full round,
so `max_rounds` went on board chatter in minutes. With the defaults one agent gets at most one
wake prompt per 30 s and 10 per run, and they leave the round budget alone. `flock apply` and `flock run` warn when an `on:` source can never
fire: a `comms.*` source on a type that is not on the board (no `comms: true` or no swarm
`comms:` block), a kind the swarm's `comms.wake` leaves out, or a source with no feeder.

### Events

Swarm stream (`GET /swarms/{s}/events`, `flock events <swarm>`): `loop_started {type, mode,
every, restart}`, `loop_tick {type, prompted, first, rounds}` (only on ticks that sent
something), `loop_updated {type}`, `loop_stopped {type, reason, rounds_total, dropped}`,
`final_round {type, agent, add_tools}`, `final_done {type, sent, done, skipped}`,
`budget_stop {phase, cost_usd, budget_usd, threshold_usd?}`, `budget_cleared`,
`loop_wake {type, agent, reason, events, coalesced, round, counted, wakes}` (a wake prompt was
sent: `reason` the wake sources, `events` the wakes folded into it, `coalesced = events - 1`,
`wakes` the agent's wake prompts so far; also in the agent's stream),
`loop_max_turns {type, agent, kind: first|loop|final, round, calls}` (a turn ended at
`max_turns`; also in the agent's stream).

Agent stream (from the worker): `loop_prompt {kind: first|loop|final, round, why, add_tools?}`
right before the turn it starts, `loop_stopped {reason, round, active}` when a worker-side loop
ends, and `turn_ended` now carries `tool_calls` (tool calls in the turn, `done` excluded) and
`state_calls` (those that are not read-only; this is what `idle_rounds` counts, falling back to
`tool_calls` for an older worker), plus `finished: true` after `done(finished: true)`. A wake
prompt outside the round count has `counted: false` in its `loop_prompt`. The controller also writes `final_round` into the agent's stream.

### Commands and API

- `PromptCommand.loop: LoopMeta` (new, optional, omitted from the wire when unset): `text` is a
  template; `LoopMeta {kind, type, round, started_at, wall_s, cap_usd, why, add_tools, counted, wake}` is what
  the worker renders it with. `round: null` lets the worker number it.
- `LoopStartCommand {agent, text, meta, max_rounds, deadline, run}` and `LoopStopCommand {agent,
  reason}`: worker-side `turn_end`. `run` names one run of the loop: a new run (after `flock
  loop --start`) resets the worker's round counter, a repeat of the same run keeps it. A worker older than this PR rejects them, so deploy the
  worker wheel with the controller before applying a `turn_end` loop.
- `GET /swarms/{s}/loop`: `LoopStatus` per type with per-agent detail (`rounds_total` counts rounds, `wakes_total` wake
prompts, so wakes and cadence rounds are apart; the same fields are in `run.json` `loops`); `POST /swarms/{s}/loop
  {type, action: start|stop, final: true}`; `TypeStatus.loop` and `SwarmStatus.budget_stop` in
  `GET /swarms/{s}`.

### CLI

```
flock loop eliza-v1                                  # state per type and per agent
flock loop eliza-v1 --type ds --stop                 # stop now, then the final round
flock loop eliza-v1 --type ds --stop --no-final      # stop, no final round
flock loop eliza-v1 --type ds --start                # run the loop again from round 1
flock status eliza-v1                                # adds: loop ds: running (controller, every 120s), rounds 3-5, 0/3 agents done
flock events eliza-v1                                # the swarm stream
```

`--start` begins a new run of the loop: the clock (`until`, `{elapsed}`) and the round counters
start again, and agents that already had `first` do not get it again. It is refused while the
budget stop is in effect.

## Examples

### Pixel painters (worlds-design §9)

```yaml
loop:
  every: turn_end
  first: "Call status once, then done."
  prompt: "Claim and paint. Repeat until the result says you are done, then call done."
  stagger: 50ms
  while: {until: 2h, max_rounds: 400, budget_usd: 12}
  # while.score_lt: {coverage_pct: 100} is the note's stop condition; it needs a world
```

This is `driver.py` without the driver: no 2-call canvas poll and bulk prompt per tick, no
install-check sweep, no stacked prompts (`skip_pending` everywhere), and with `turn_end` no
idle gap between a painter's turns. Until the world PR, coverage is the painters' own stop
(the CLI says "You are done") plus `idle_rounds`.

### Eliza (worlds-design §4.7)

```yaml
types:
  ds: &dev
    harness: llm
    model: deepseek
    replicas: 3
    system_prompt_file: prompts/system.md
    tool_choice: required
    max_turns: 14                           # Eliza's tool calls per round
    loop:
      prompt_file: prompts/round.md         # {round} {elapsed} {wall} {spend} {cap} {why} {last_summary}
      every: 120s                           # idle_wake_s
      on: [comms.mention, comms.thread]     # "new board message", debounced 2 s
      stagger: 4s
      while: {until: 2h, max_rounds: 200, agent_usd: {deepseek: 3.60, glm: 2.10}}
      final: {prompt_file: prompts/final.md, reserve_usd: 0.60, add_tools: [submit_final, best_vote]}
      # context: [world.state, artifacts.recent:15, comms.threads, comms.recent:20]  -> later PRs
  glm: {<<: *dev, model: glm, replicas: 2}
limits: {budget_usd: 15}
```

How `config.yaml` and `swarm.py::main_loop` map:

| Eliza harness | flock |
|---|---|
| `loop.debounce_s: 2` | the wake queue's 2 s debounce (fixed) |
| `loop.idle_wake_s: 120` | `every: 120s` |
| `loop.max_steps_per_round: 14` | the type's `max_turns: 14` (model calls per prompt) |
| `loop.max_rounds_per_agent: 200` | `while.max_rounds: 200` |
| `loop.stagger_start_s: 4` | `stagger: 4s` |
| `caps.agent_usd_by_model` | `while.agent_usd: {deepseek: 3.60, glm: 2.10}` |
| `caps.global_usd: 15` | `limits.budget_usd: 15` |
| `caps.final_reserve_usd: 0.60`, `soft_cap = global - reserve` | `final.reserve_usd: 0.60`; the `final_reserve` budget stop |
| `caps.wall_clock_s: 7200` | `while.until: 2h` |
| `final_round: auto` | `final:` |
| `final_round: off` / `manual` | no `final:` / `flock loop --stop --no-final`, then a manual prompt |
| `ctx.halt(...)` when `global_spend >= soft_cap` | `budget_stop {phase: final_reserve}` |
| `ROUND` template `{rnd} {el} {wall} {spent} {cap} {why} {summary}` | `{round} {elapsed} {wall} {spend} {cap} {why} {last_summary}` |
| `done_for_now(summary)` | the `done` tool; its summary is `{last_summary}` |
| `board_context_msgs: 20`, `ws_recent_in_prompt: 15` | `context:` (deferred, below) |
| `caps.final_round_agent_usd: 0.10` | not modelled: final-round spend is exempt from `agent_usd` (as in Eliza) and bounded only by the reserve and the hard stop |

What still stays outside flock for Eliza: the board kickoff and final-round announcement (comms
PR), the per-round context block (context sources), and the final-submission bookkeeping
(tools layer, via `add_tools`).

## Deferred, and what differs from the note

Deferred to the world PR (accepted by the model, refused by `flock apply` and `PUT /swarms`
with "needs a world (not in this version)"): `every: rounds` (prompt all, wait for every
`turn_ended` or `rounds.timeout_s`, `POST /rounds/{n}/close`), `while.score_*`, `on: world.*`,
`context: world.*`.

Deferred to the comms and artifacts PRs: `context: comms.* | artifacts.*` (refused: "is not in
this version"). Deferred to the
tools PR: honouring `final.add_tools` (carried, not enforced). Not done: a `run_stopping` hook to
a world, per-call booking to tighten the budget overshoot, a `loop.start: manual` switch, and
`limits.final_reserve_usd`: the note's §4.7 put the reserve under `limits`; here it is
`final.reserve_usd` per type, summed for the swarm threshold.
