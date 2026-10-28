# Policies and the digest

The controller checks rules against each health snapshot ([health.md](health.md), every 30 s)
and raises alerts. It can also act on them: drain a worker, hold a type, or turn a world knob.
Rules live in the spec and are applied live, like `loop:`. Without a `policies:` block, the
built-in rules run in alert mode.

```yaml
policies:
  mode: alert            # alert (default) | enforce; a rule can override it
  sinks: [file:alerts.log, webhook:https://hooks.example/flock]   # for `notify`
  rules:
    - {name: worker-mem, metric: worker.nonreclaim_frac, op: ">", value: 0.9, for: 2m,
       clear: 0.8, do: drain_worker, cooldown: 30m, max_concurrent: 2, max_per_hour: 4}
    - {name: world-slow, metric: world.manifest_ms, op: ">", value: 10000, for: 3m,
       do: [notify, {world_knob: {pause_pushes: true}}],
       clear_do: {world_knob: {pause_pushes: false}}}
    - {name: open-refused, metric: tools.open_pr.refused_pct, op: ">=", value: 50,
       for: 5m, do: {world_pause: {lanes: [open]}}, clear_do: {world_resume: {lanes: [open]}}}

report:
  every: 1h
  sinks: [file:digest.txt, webhook:https://hooks.example/flock]
```

## Rules

| field | meaning |
|---|---|
| `name` | unique in the block; used by `flock policy mute` |
| `metric` | a fixed snapshot path (below); no expressions |
| `op`, `value` | `>`, `>=`, `<`, `<=`, `==`, `!=` against a number |
| `for` | how long the condition must hold, counted in fresh samples: `ceil(for / sample interval)` in a row (default `0s`: one sample) |
| `clear` | hysteresis: a firing rule clears when `metric op clear` stops holding (default: when `metric op value` stops holding) |
| `do`, `clear_do` | one action or a list; an action is a name or `{name: argument}` |
| `cooldown` | per rule and target, between two enforced episodes (default `30m`) |
| `max_concurrent` | targets of this rule acted on and still firing |
| `max_per_hour` | enforced episodes of this rule per hour |
| `mode` | `alert` or `enforce`; default: the block's `mode` |

Metrics:

- `worker.nonreclaim_frac` (non-reclaimable memory / limit), `worker.eta_limit_s`,
  `worker.slope_mb_h`: evaluated per worker; the target is the worker id.
- `world.up` (1 or 0), `world.manifest_ms`, `world.metrics.<key>` (the world's `/metrics`).
- `tools.<tool>.rate_per_min|err_pct|refused_pct|max_s`.
- `types.<type>.stuck_n` and `types.<type>.<state>` (agents in `working`, `idle`, ...).
- `providers.<provider>.waiting|running|kv`.

Everything except `worker.*` is evaluated once per swarm, with the target `swarm` (or `world`).

**Freshness.** A worker's sample counts only when its heartbeat moved since the last tick. A
world sample counts only when the world was probed this tick. A stale sample neither counts
toward `for` nor resets it, so a silent worker never triggers a drain. A metric with no
value (no world, a tool with no calls yet) is skipped the same way.

## Actions

| action | argument | enforced in this version |
|---|---|---|
| `notify` | none | always runs: the event, plus each `policies.sinks` entry |
| `drain_worker` | none (target worker) | yes. Uses the memory-rotation budget: `DrainCommand` and a `memory_rotation {worker, ..., rule}` event, while fewer than `FLOCK_MAX_ROTATING` (default 10 % of the swarm's worker slots) are draining or restoring. The drain ends in a planned restart |
| `pause_type` / `resume_type` | a type or a list; none: every type with world tools | yes. Automatic holds (docs/loop.md "Pause and resume"): `auto: true`, reason = the rule name. `resume_type` lifts automatic holds only, never an operator's |
| `world_knob` | `{knob: value, ...}` | yes, through the world's admin API (`world_knob` events, `by: policy`) |
| `world_pause` / `world_resume` | `{lanes: [...]}` or a list | yes, through the world's admin API |
| `prompt` | `{text}` | no: always logged as `would_do` |

In `alert` mode every action except `notify` is logged as `would_do`. A target's `clear_do` is
enforced only when that episode's `do` was enforced, and it runs even while the rule is muted.

**Holds a policy leaves behind.** Each policy hold has `by: policy` and its rule's name as the
reason, and `resume_type` lifts only that rule's holds. The world-down hold (`world_down`) and
a policy hold on the same type are lifted separately. Holds are kept in the store, but rule
state is not. So when a rule has a fresh sample under its threshold and no target is firing,
the controller lifts any hold that rule left: after a restart, for example, or when a
`clear_do` failed. When a rule is removed by an apply, its holds are lifted on the next tick.
Lane pauses (`world_pause`) are kept by the world and have no such sweep: a lane paused by a
rule that was removed, or that was firing when the controller restarted, stays paused until
`flock world resume`.

World down already has a built-in response: workers refuse world calls at once, and after
`world.down_pause_s` (default 30 min) the controller holds the types that use the world. It
lifts the hold when the world comes back. No policy rule is needed for it.

**Built-in rules** (no `policies:` block, alert mode): `worker-mem`
(`worker.nonreclaim_frac > 0.9` for 2 min, clear 0.8, `drain_worker`, cooldown 30 min).
To enforce it, copy it into a block with `mode: enforce`.

## Safety

- An enforced episode acts once. While a cap holds it back it is tried again on later ticks,
  as long as the rule still fires. The caps are `cooldown`, `max_concurrent`,
  `max_per_hour`, and for `drain_worker` a free rotation slot.
- Global budget: `FLOCK_POLICY_MAX_ACTIONS_PER_HOUR` enforced episodes per swarm (default 20).
  Past it, rules drop to alert-only (`would_do`) until the hour frees a slot, and
  `policy_budget_exhausted {rule, max_actions_per_hour}` is written at most once an hour.
- `flock policy mute <swarm> <rule> [--for 2h]`: no alert events and no new episodes. The
  rule's state keeps updating, and an enforced episode that clears still runs its
  `clear_do`. `flock policy unmute <swarm> <rule>`. Mutes are stored and survive a
  restart.
- Rule state (counts, firing, cooldowns) is in memory. After a controller restart, a
  condition that still holds fires again after `for`. Drains still respect the rotation
  budget.

**Checks.** At apply, a metric naming a type or provider the spec does not have is refused,
and so is a `pause_type` or `resume_type` naming an unknown type. A tool name is checked only
when the world's tools are known: types list `world:*` tools, or the manifest is known. While
no target of a rule has a value (for example a tool no agent has called yet), the controller
writes `policy_warning {rule, msg}` once.

## Events

On the swarm stream (`flock events <swarm>`):

- `alert {rule, target, value}`
- `alert_cleared {rule, target, value}` (`value: null, gone: true` when the worker left)
- `policy_action {rule, target, action, arg?, mode: done|would_do, clear?, error?, ...}`
- `policy_budget_exhausted`, `policy_muted`, `policy_unmuted`
- `digest {text}`

## Sinks

- `file:<path>`: a line is appended to `reports/<path>` under the controller's state dir,
  which the state sync mirrors. The path must be relative, with no `..`.
- `webhook:<url>`: a POST with JSON and a 5 s timeout. `notify` sends
  `{rule, target, metric, value, firing, ts}`; the digest sends `{swarm, ts, text}`.

A failing sink is logged and skipped.

## Digest

`report: {every, sinks}`: every `every` (the first one comes one period after the controller
starts), the controller writes a short text from the latest snapshot and emits it as a
`digest` event. The text is capped at 4 KB. It covers: the world (up, manifest latency, Slurm
state), the world metrics named `*last_hour*` or `*per_hour*` (for example merges and opens),
workers with the nearest memory limits, restores in progress, agents by state and stuck agents
per type, tokens this hour by type, the snapshot's alerts, and the policy events since the
last digest.

## API and CLI

```
GET  /swarms/{s}/policies                      # mode, rules, mutes, firing targets, budget used
POST /swarms/{s}/policies/{rule}/mute {for?}   # / unmute
flock policy list <swarm>
flock policy mute <swarm> <rule> [--for 2h]
flock policy unmute <swarm> <rule>
```
