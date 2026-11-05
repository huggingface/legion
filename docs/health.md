# Health: snapshots, usage, rollups

The controller samples each swarm every 30 s (`FLOCK_HEALTH_INTERVAL`, 0 = off). Agent event
files are never read: counters are fed by the event batches as they arrive, and the swarm
stream is tailed by byte offset.

## What is probed

- **World.** `GET <endpoint>/manifest` with a 5 s timeout: up/down and latency. When the
  manifest lists the `metrics-v1` capability, `GET /metrics` is stored as-is under
  `world.metrics`. Two failed probes in a row write `world_down` to the swarm stream; the next
  success writes `world_up`.
- **World down: refuse, then pause.** From `world_down` on, every heartbeat response carries
  `world.down_since` (the first failed probe) and the workers' world hub refuses world calls at
  once, read-only ones too, without uploading anything: `world <name> down since HH:MM UTC;
  do local work or end your turn` (`code: world_down`, `local: true`). When the
  world has been down for `world.down_pause_s` seconds (default `FLOCK_WORLD_DOWN_PAUSE_S`,
  1800; `0` turns it off) the controller holds every type that has world tools (docs/loop.md
  "Pause and resume"): `paused {reason: world_down, by: controller, auto: true}`. The first
  successful probe lifts those automatic holds (`resumed {auto: true}`), also ones left by a
  controller restart; operator holds stay. `down_pause_s` is controller-only and never sent
  to workers.
  An automatic hold cannot be lifted by hand while the world is still down (the next probe
  holds the types again); set `world.down_pause_s: 0` to turn it off.
- **World Slurm job.** With `world.slurm_job_name: <name>` and `compute: slurm`, the
  controller asks `squeue -n <name>` and shows `slurm_state` (`RUNNING`, `PENDING`, ...,
  `absent` when no such job is queued, `unknown` when squeue is missing, fails or times out),
  so a world waiting for a node is told apart from a crash.
- **vLLM.** With `providers.<p>.metrics: true`, each base URL's server root `/metrics`
  (Prometheus text): `providers` sums `waiting` and `running` over servers with the highest
  `kv` (`gpu_cache_usage_perc` or `kv_cache_usage_perc`), as before; `provider_servers` has
  each server (`host:port`): `up`, its queues, or `up: false`, the `error` and `fails` (failed
  scrapes in a row; `dead` from 2 on). Each agent's heartbeat names the server it is pinned to
  (`endpoint`, `pick_endpoint`), so a server counts its `agents`; a dead one lists them
  (`dead_agents`, first 20) and raises `dead_server` (`provider_down` when every server of the
  provider is dead). Moving agents off a dead server is manual: drop it from `base_urls` and `flock update`.

## Snapshot body

```
{world: {up, manifest_ms, slurm_state, metrics, ts} | null,
 workers: {wid: {hb, mem, slope_mb_h, eta_limit_s, restore: {..., eta_s}, relaunches}},
 types: {t: {by_state, stuck: [{agent, state, silent_s}], gens: {rev: n}}},
 tools: {name: {rate_per_min, err_pct, refused_pct, max_s}},
 providers: {p: {waiting, running, kv, completed}},
 provider_servers: {p: {"host:port": {up, fails, dead, waiting, running, kv, completed, error, agents, dead_agents}}},
 ramps: {t: {step, steps, replicas, max, next_add, next_at, holds}},
 tokens: {hour: "YYYY-MM-DDTHH", by_type: {t: {input, output, cache_read, cache_write, reasoning}}},
 alerts: [{kind, msg}]}
```

- `mem`, `restore` and the tool counters come from the worker heartbeats (docs/design.md,
  "Heartbeat"); tool rates are the deltas of those counters between two samples (a worker
  seen for the first time has no rate yet; a restarted worker's counters start over).
- `relaunches`: `crash` and `planned_restart` events of that worker on the swarm stream.
- `stuck`: agents in `stuck`, or in `working` with no event for more than 30 min
  (`FLOCK_STUCK_MIN`). After a controller restart the clock starts at its first sample.
- `gens`: agents by the type spec revision they run (`rev` in the agent heartbeat).
- `hb` is the worker's last heartbeat time and `ts` the world probe's time.
- `ramps`: per `replicas_ramp` type, its step (`ramp:<type>`), the next step's size and time
  (epoch s; null while held, released or at the end) and the consecutive capacity holds.
- Alerts: world down (2 failed probes in a row; `world_pending` instead while its Slurm job is
  PENDING), a dead model server (`dead_server`, `provider_down`), 3 or more ramp
  holds in a row (`ramp_held`), a worker within 1 h of its memory limit, stuck agents, a tool
  with 20 % or more errors or refusals over the last sample.

Retention: the 30 s samples stay in the controller's memory for 6 h (gone after a restart).
Every 10th one is stored as a 5-min sample for 48 h (table `health_samples`, pruned on
insert) in a slim form: `workers` becomes `{count, max_nonreclaim_frac, min_eta_limit_s,
restoring, relaunches}` and each type keeps at most 20 `stuck` entries (`stuck_n` has the
count).

## Token usage

Tokens per UTC hour and agent type, stored in the `usage` table. The rule: a turn's
`turn_ended` tokens count; assistant `message` usage counts instead when the `turn_ended`
total is empty (`usage_missing`), and for a turn that never got a `turn_ended` (cut by a
worker death or an interrupt), at the agent's next `turn_started`, `turn_interrupted` or
`session_*` event. A cut turn of an agent that never sends another event
is not counted. Hours are by the time the controller received the events.

## Files (mirrored by the state sync)

Under the controller's data dir (with `FLOCK_STATE_URL` also in the bucket; `flock launch`:
`<run dir>/controller/`):

```
run.json                                 {schema: 1, run, swarm, swarms, started_at, ended_at|null,
                                          status: running|ended, states: true, generation,
                                          types: {t: {harness, model, replicas}}}
metrics/<swarm>/latest.json              {ts, body}: the newest 30 s sample (full), overwritten
metrics/<swarm>/<YYYY-MM-DD>T<HH>.jsonl  {ts, body} per 5-min sample (slim); one file per 6 h,
                                         HH = 00, 06, 12, 18, so each upload stays small
rollups/<swarm>/<YYYY-MM-DDTHH>.jsonl    one line per agent active that hour
states/<swarm>/latest.json               {ts, states: {agent: working|pending|idle|stuck|unknown|dead}},
                                         every 30 s sample, overwritten
states/<swarm>/<YYYY-MM-DDTHH>.jsonl     one line per agent per 5-min bucket of that hour
```

`ts` is epoch seconds. `run` is `FLOCK_RUN`, else the last part of `FLOCK_STATE_URL`, else the
data dir name. `run.json` is written at the first sample, on each new generation and at
shutdown.

A rollup line: `{h, agent, type, parent, ephemeral, turns, out: {done, max_turns, max_budget,
error}, working_s, pending_s, idle_s, tool_calls, tools: {name: {n, err}}, tokens: {model:
{input, output, cache_read, cache_write, reasoning}}, events: {type: n}, wakes, errors, first_ts,
last_ts}`. Token keys are the full names. The open hour's file is rewritten whole with the
running totals every 5 min, once more when the hour closes and at shutdown; a closed hour
never changes. There is exactly one line per (agent, h): read it with replace semantics. A
controller restarted within the hour reloads that hour's file (restored from the bucket) and
keeps adding to it, so a SIGKILL loses at most the last 5 minutes.
`working_s`/`pending_s`/`idle_s` are sampled (each sample adds its interval to the agent's
state; pending = idle with a prompt queued); `tools` counts tool results; `wakes` counts
`loop_wake`; `errors` counts `error` events (failed turns are in `out.error`).

A state line, for the viewer's per-agent lanes: `{"b": "YYYY-MM-DDTHH:MM", "agent", "type",
"w", "p", "i", "s"}`, the seconds sampled working, pending, idle and stuck in the 5-min bucket
starting at `b` (UTC; each sampled interval is split at the bucket boundaries, so a bucket holds at most 300 s per agent). Zero fields are left out; an agent with no sample in a
bucket (dead, unknown, removed) has no line. Same sampling and rewrite rules as the rollups: the
open hour's file is rewritten whole every 5 min, once more after the hour closes and at
shutdown, then never changes; one line per (agent, b), read with replace semantics; a restart
within the hour reloads the file. About 100 bytes per line: 360 live agents give 4,320 lines
(about 430 KB) per hour file, roughly 10 MB a day. `latest.json` seeds the lanes after a viewer restart (`pending`: idle
with a prompt queued).

## Surface

- `GET /swarms/{s}/health`: `{swarm, generation, ts, body, recent_events}` (the last 10
  control-plane events of the swarm stream, `loop_tick`/`loop_wake` left out). Right after a
  controller restart, until the first sample, `body` is the newest stored slim one.
- `GET /swarms/{s}/health/history?since=<epoch s or ISO>&res=30s|5m`: `[{ts, body}]`; `30s`
  from memory (full bodies), `5m` from the store (slim).
- The `/` status page has one health line per swarm: aggregates only, no agent names.

## `flock status`

One operator view per swarm, built by `flock.status.view` (the Python API uses the same
function) from `GET /swarms/{s}`, the newest snapshot, `GET /workers` and, with a filter, the
agents. The first line is the verdict (`flock.status.verdict`):

- `FAILING`: the world is down, every server of a provider is down under its agents, or a
  type's launches are refused. Exit 1.
- `DEGRADED`: any other alert (a world whose Slurm job is PENDING, stuck agents, a dead
  server, memory, tool errors, ramp holds),
  a launch error being retried, agents whose final sync failed, a budget stop, or no snapshot
  in the last 2 min (the controller's prober is not running).
- `OK`.

Then: counts per type, holds, loops, providers; workers by state with the scheduler's pending
reason (`PENDING (Resources)`, from squeue `%r`) and one row per worker with crashes or a
`last_error`; the snapshot's world (up/down, Slurm state, `/metrics`: event-loop lag, 429s,
each pool's oldest queue wait, `lane_<lane>_*` per lane, the comment outbox, any other number
under `world`), each model server, each ramp (`step k/n, next +m at HH:MMZ`), alerts, agents
per type and state, workers by time to memory limit, restores, tool rates, stuck agents and
the last control-plane events.

- `--type T`, `--state S`, `--parent P` list the matching agents (with their server, `DEAD`
  when it is down); `--type` adds the type's per-agent loop rows. `--state idle` means what
  `flock prompt --state idle` means (the controller's `turn_state`: a prompt just handed out
  is working, a turn just ended is idle, without waiting for a heartbeat).
- `--agent A` (or an agent's name as the argument): that agent's status as JSON.
- `--watch` reprints every 10 s; `--json` prints the whole view (`verdict`, `status`,
  `health`, every worker row, `agents`, `loop`).
- A controller that does not answer: `controller unreachable at URL; hosts/controller.host
  says HOST, written at T` (T is the file's time, i.e. the controller's last start; the host
  file part needs `--run` or FLOCK_RUN_DIR). Exit 1.
- Inside an agent: `flock status` is your own status; `flock status [SWARM] --parent P` lists
  your (or a descendant's) children.

## Events

`flock status SWARM --events` prints the swarm stream, `flock status AGENT --events` (or
`--agent A --events`) an agent's; `--only PREFIX` keeps the names starting with it (`--only
agent.`), `--after N` starts after a cursor, `--follow` keeps long-polling, `--wait` returns
once something new arrived (the next cursor on stderr), `--json` prints JSONL.
`flock.status.events` yields the same pages (a removed agent's stored events too). Hidden,
for prompts in resumed runs: `flock events NAME [--after --follow --wait --timeout --pretty]`
(raw JSONL) and `flock agents SWARM [--type --state --parent]` (the old table). Stored events are never rewritten (resume reads
them); `flock.status.normalise` maps each to one schema:

`{name, ts, swarm, agent?, type?, worker?, data}`; `data` is the event's payload, with
`turn_id` when it has one.

| name | stored as | stream | when |
|---|---|---|---|
| `agent.message` | `message` | agent | a harness message (user, assistant, tool result) |
| `agent.turn_started`, `agent.turn_ended`, `agent.turn_interrupted` | same | agent | a turn starts, ends, is cut (`reason: timeout` makes the agent stuck) |
| `agent.session_started`, `agent.session_resumed`, `agent.session_restarted` | same | agent | the harness session |
| `agent.sync`, `agent.error`, `agent.prompt_deferred`, `agent.harness_recycled`, ... | same | agent | any other worker event: `agent.<stored type>` |
| `agent.stuck` | `stuck` | swarm | the controller first sees an agent stuck (its turn timed out) |
| `agent.nudged` | `nudge` | swarm | a stuck agent got its type's `resume_prompt` |
| `agent.spawned`, `agent.child_removed` | `spawned`, `child_removed` | parent agent | a child spawned or removed |
| `worker.launched`, `worker.launch_failed` | `launch`, `launch_failed` | swarm | a worker job submitted or refused |
| `worker.crashed`, `worker.crashloop`, `worker.restarted` | `crash`, `crashloop`, `planned_restart` | swarm | a worker job ended |
| `worker.ended` | `end` | swarm | a released swarm's worker job ended as planned: nothing relaunches it |
| `worker.cancelled`, `worker.retired`, `worker.removed`, `worker.released` | `cancel`, `retire`, `remove_worker`, `release` | swarm | a worker taken down |
| `worker.rotated`, `worker.sync_failed` | `memory_rotation`, `sync_failed` | swarm | memory rotation; a final sync failed |
| `type.launch_refused` | `launch_refused` | swarm | a type's launches stop until the next apply |
| `loop.started`, `loop.updated`, `loop.tick`, `loop.wake`, `loop.stopped`, `loop.max_turns`, `loop.prompt` | `loop_*` | swarm, agent | the loop engine (docs/loop.md) |
| `loop.final_round`, `loop.final_done` | `final_round`, `final_done` | swarm, agent | the final round |
| `ramp.step`, `ramp.held` | `ramp_step`, `ramp_held` | swarm | a ramp step, a capacity hold |
| `world.down`, `world.up`, `world.knob` | `world_down`, `world_up`, `world_knob` | swarm | the world probe; a knob changed |
| `budget.stop`, `budget.cleared` | `budget_stop`, `budget_cleared` | swarm | `limits.budget_usd` |
| `comms.dm`, `comms.mention`, `comms.joined`, ... | `comms_*` | agent | the board (docs/comms.md) |
| `swarm.stopped`, `swarm.resumed`, `swarm.released`, `swarm.unreleased` | `paused`, `resumed`, `released`, `unreleased` | swarm | `flock stop` / `flock launch` |
| `swarm.spec_applied`, `swarm.removed` | `spec_applied`, `remove_swarm` | swarm | the stored spec; `--delete` done |
