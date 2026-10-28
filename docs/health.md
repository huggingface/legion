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
  (Prometheus text): `waiting` and `running` are summed over servers, `kv` is the highest
  `gpu_cache_usage_perc` (or `kv_cache_usage_perc`).

## Snapshot body

```
{world: {up, manifest_ms, slurm_state, metrics, ts} | null,
 workers: {wid: {hb, mem, slope_mb_h, eta_limit_s, restore: {..., eta_s}, relaunches}},
 types: {t: {by_state, stuck: [{agent, state, silent_s}], gens: {rev: n}}},
 tools: {name: {rate_per_min, err_pct, refused_pct, max_s}},
 providers: {p: {waiting, running, kv}},
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
- `hb` is the worker's last heartbeat time and `ts` the world probe's time. Policies use them
  to skip stale samples ([policies.md](policies.md)).
- Alerts: world down, a worker within 1 h of its memory limit, stuck agents, a tool with
  20 % or more errors or refusals over the last sample.

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

Under the controller's data dir (with `FLOCK_STATE_URL` also in the bucket; `flock run`:
`<run dir>/controller/`):

```
run.json                                 {schema: 1, run, swarm, swarms, started_at, ended_at|null,
                                          status: running|ended, generation, types: {t: {harness, model, replicas}}}
metrics/<swarm>/latest.json              {ts, body}: the newest 30 s sample (full), overwritten
metrics/<swarm>/<YYYY-MM-DD>T<HH>.jsonl  {ts, body} per 5-min sample (slim); one file per 6 h,
                                         HH = 00, 06, 12, 18, so each upload stays small
rollups/<swarm>/<YYYY-MM-DDTHH>.jsonl    one line per agent active that hour
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

## Surface

- `GET /swarms/{s}/health`: `{swarm, generation, ts, body, recent_events}` (the last 10
  control-plane events of the swarm stream, `loop_tick`/`loop_wake` left out). Right after a
  controller restart, until the first sample, `body` is the newest stored slim one.
- `GET /swarms/{s}/health/history?since=<epoch s or ISO>&res=30s|5m`: `[{ts, body}]`; `30s`
  from memory (full bodies), `5m` from the store (slim).
- `flock status --watch` also shows the world's `/metrics`: event-loop lag, 429s, each pool's
  oldest queue wait, `lane_<lane>_tripped/paused/effective_slots/in_flight` per lane, the
  comment outbox depth and age, and any other number under `world`.
- `GET /swarms/{s}/usage?by=type|hour`: token rows.
- `flock status [swarm] --watch` (every 10 s), `flock status [swarm] --json`,
  `flock usage <swarm> [--by type|hour]`.
- The `/` status page has one health line per swarm: aggregates only, no agent names.
