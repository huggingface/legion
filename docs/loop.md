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
      on: [comms.mention]        # wake-ups (fed by the comms PR; see "Wake queue")
      while:                     # all set conditions must hold
        until: 2h                # wall time since the loop started (type-wide)
        max_rounds: 400          # loop prompts per agent
        budget_usd: 12           # the type's total cost (type-wide)
        agent_usd: 0.05          # per agent; or a map by model alias / id: {deepseek: 3.60, glm: 2.10}
        idle_rounds: 3           # this many loop turns in a row without a tool call end an agent's loop
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
| `on` | wake sources (`comms.mention`, `comms.thread`, ...): a wake queues the loop prompt early (2 s debounce). `world.*` needs a world |
| `context` | per-prompt context sources (`world.state`, `comms.recent:20`, `artifacts.recent:15`): accepted by the model, refused at apply until their feeders exist |
| `while.until`, `while.budget_usd` | type-wide: when either fails the loop stops for every agent of the type |
| `while.max_rounds`, `while.agent_usd`, `while.idle_rounds` | per agent: the agent leaves the loop; when every agent has left, the loop stops |
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
| `{last_summary}` | the text the agent last passed to `done` (`llm` harness), else its last reply; `(none yet)` before any |

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
  idle and again each time a turn ends `completed`, with no controller round trip (no idle gap,
  run 1 lost ~17 % of agent time there). It stops at `max_rounds`, at the `until` deadline, on
  a `loop_stop` command, or after 3 failed turns in a row (a failed turn is retried after 2 s).
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
counts loop prompts (not `first`, not the final prompt, wake-ups included). In controller mode
an agent leaves the loop at `max_rounds` only once its last round's turn ended, so a queued
round is never dropped.

**Stopping.** The loop stops when a type-wide condition fails (`until`, `budget_usd`, the
budget threshold), when every agent has left it (`max_rounds`, `agent_usd`, `idle_rounds`,
`errors`; the reason is theirs if they agree, else `agents_done`), on `flock loop --stop`, or
when `loop:` is removed from the spec (no final round then). Stopping drops the type's
undelivered loop prompts and sends `loop_stop` to worker-side loops.

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

Re-applying the spec with a budget above the cost lifts the stop (`budget_cleared`). Cost is
what the controller's ledger knows, which is updated by heartbeats (every 30 s), so the overshoot
is bounded by about one heartbeat of spend of the working agents; Eliza's `log_overshoot` makes
the same point.

**Live settings** (worlds-design §4.4). `flock apply` with a changed `loop:` takes effect at the
next tick: controller-side settings (`every`, `while`, `final`, `stagger`) are read every tick,
and worker-side loops get a fresh `loop_start` with the new prompt and limits (`loop_updated`
event). A stopped loop stays stopped until `flock loop --start`. `limits.budget_usd` is live too.

**Restart.** Loop state is in the controller's SQLite (`loops`, `loop_agents`, `swarm_flags`),
so a controller restart resumes the loops; only queued wake-ups are lost.

### Wake queue (`on:`)

`LoopEngine.wake(agent, reason)` is the controller-side seam the comms PR will feed (board
mentions, thread replies): it queues the agent's loop prompt (`why: wake: <reasons>`) once 2 s
have passed without a new wake for that agent, or at most 10 s after the first, through the
same `state: idle, skip_pending: true` path. A wake is accepted only for an agent in a running
loop whose `on:` lists the reason (or its prefix before `:`, so `comms.mention:42` matches
`comms.mention`); wakes for an agent that is working wait until it is idle and then give one
prompt. Nothing calls it in this version; `flock apply` warns that `on: comms.*` has no feeder
yet, and `tests/test_loop_controller.py::test_wake_queue_debounces_and_skips_pending` drives it
directly.

### Events

Swarm stream (`GET /swarms/{s}/events`, `flock events <swarm>`): `loop_started {type, mode,
every, restart}`, `loop_tick {type, prompted, first, rounds}` (only on ticks that sent
something), `loop_updated {type}`, `loop_stopped {type, reason, rounds_total, dropped}`,
`final_round {type, agent, add_tools}`, `final_done {type, sent, done, skipped}`,
`budget_stop {phase, cost_usd, budget_usd, threshold_usd?}`, `budget_cleared`.

Agent stream (from the worker): `loop_prompt {kind: first|loop|final, round, why, add_tools?}`
right before the turn it starts, `loop_stopped {reason, round, active}` when a worker-side loop
ends, and `turn_ended` now carries `tool_calls` (tool calls in the turn, `done` excluded; this
is what `idle_rounds` counts). The controller also writes `final_round` into the agent's stream.

### Commands and API

- `PromptCommand.loop: LoopMeta` (new, optional, omitted from the wire when unset): `text` is a
  template; `LoopMeta {kind, type, round, started_at, wall_s, cap_usd, why, add_tools}` is what
  the worker renders it with. `round: null` lets the worker number it.
- `LoopStartCommand {agent, text, meta, max_rounds, deadline}` and `LoopStopCommand {agent,
  reason}`: worker-side `turn_end`. A worker older than this PR rejects them, so deploy the
  worker wheel with the controller before applying a `turn_end` loop.
- `GET /swarms/{s}/loop`: `LoopStatus` per type with per-agent detail; `POST /swarms/{s}/loop
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

Deferred to the comms and artifacts PRs: the feeder for `on: comms.*` (the engine's `wake()` is
ready), and `context: comms.* | artifacts.*` (refused: "is not in this version"). Deferred to the
tools PR: honouring `final.add_tools` (carried, not enforced). Not done: a `run_stopping` hook to
a world, per-call booking to tighten the budget overshoot, a `loop.start: manual` switch, and
`limits.final_reserve_usd`: the note's §4.7 put the reserve under `limits`; here it is
`final.reserve_usd` per type, summed for the swarm threshold.
