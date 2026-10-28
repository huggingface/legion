# Health: snapshots, usage, rollups

The controller samples each swarm every 30 s (`FLOCK_HEALTH_INTERVAL`, 0 = off) and keeps the
result in its store and next to its event files. Nothing here re-reads agent event files:
counters are fed by the event batches as they arrive.

## What is probed

- **World.** `GET <endpoint>/manifest` with a 5 s timeout: up/down and latency. When the
  manifest lists the `metrics-v1` capability, `GET /metrics` is stored as-is under
  `world.metrics`. Two failed probes in a row write `world_down` to the swarm stream; the next
  success writes `world_up`.
- **World Slurm job.** With `world.slurm_job_name: <name>` and `compute: slurm`, the
  controller asks `squeue -n <name>` and shows `slurm_state` (`RUNNING`, `PENDING`, ...,
  `absent`), so a world waiting for a node is told apart from a crash.
- **vLLM.** With `providers.<p>.metrics: true`, each base URL's server root `/metrics`
  (Prometheus text): `waiting` and `running` are summed over servers, `kv` is the highest
  `gpu_cache_usage_perc` (or `kv_cache_usage_perc`).

## Snapshot body

```
{world: {up, manifest_ms, slurm_state, metrics} | null,
 workers: {wid: {mem, slope_mb_h, eta_limit_s, restore: {..., eta_s}, relaunches}},
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
- Alerts: world down, a worker within 1 h of its memory limit, stuck agents, a tool with
  20 % or more errors or refusals over the last sample.

Retention: 30 s samples for 6 h and every 10th one as a 5-min sample for 7 days (table
`health_samples`, pruned on insert).

## Token usage

Tokens per UTC hour and agent type, stored in the `usage` table. The rule: a turn's
`turn_ended` tokens count; assistant `message` usage counts only for a turn that never got a
`turn_ended` (cut by a worker death or an interrupt), at the agent's next `turn_started`,
`turn_interrupted` or `session_*` event. A cut turn of an agent that never sends another event
is not counted. Hours are by the time the controller received the events.

## Files (mirrored by the state sync)

Under the controller's data dir (with `FLOCK_STATE_URL` also in the bucket; `flock run`:
`<run dir>/controller/`):

```
run.json                                 {schema: 1, run, swarm, swarms, started_at, ended_at|null,
                                          status: running|ended, generation, types: {t: {harness, model, replicas}}}
metrics/<swarm>/latest.json              {ts, body}: the newest 30 s sample, overwritten
metrics/<swarm>/<YYYY-MM-DD>.jsonl       {ts, body} per 5-min sample
rollups/<swarm>/<YYYY-MM-DDTHH>.jsonl    one line per agent active that hour
```

`ts` is epoch seconds. `run` is `FLOCK_RUN`, else the last part of `FLOCK_STATE_URL`, else the
data dir name. `run.json` is written at the first sample, on each new generation and at
shutdown.

A rollup line: `{h, agent, type, parent, ephemeral, turns, out: {done, max_turns, max_budget,
error}, working_s, pending_s, idle_s, tool_calls, tools: {name: {n, err}}, tokens: {model:
{input, output, cache_read, cache_write, reasoning}}, events: {type: n}, wakes, errors, first_ts,
last_ts}`. An hour's file is appended when the hour closes and at shutdown, so after a
controller restart within an hour an agent can have two lines for that hour: sum them.
`working_s`/`pending_s`/`idle_s` are sampled (each sample adds its interval to the agent's
state; pending = idle with a prompt queued); `tools` counts tool results; `wakes` counts
`loop_wake`; `errors` counts `error` events (failed turns are in `out.error`).

## Surface

- `GET /swarms/{s}/health`: `{swarm, generation, ts, body, recent_events}` (the last 10
  control-plane events of the swarm stream, `loop_tick`/`loop_wake` left out).
- `GET /swarms/{s}/health/history?since=<epoch s or ISO>&res=30s|5m`: `[{ts, body}]`.
- `GET /swarms/{s}/usage?by=type|hour`: token rows.
- `flock status [swarm] --watch` (every 10 s), `flock status [swarm] --json`,
  `flock usage <swarm> [--by type|hour]`.
- The `/` status page has one health line per swarm: aggregates only, no agent names.
