# swarm-harness: design

Status: agreed design, 2026-10-14. Author: Carlos Patiño with Claude.
Scope: v1, the first milestone and the contracts later layers depend on.

## 1. What it is

swarm-harness hosts coding-harness agents (Claude Code, Codex) on Hugging Face
Jobs, keeps them alive for weeks, takes prompts for them, and exposes their
sessions and health. It is the lowest layer under two existing systems:

- **msgboard** (`huggingface/msgboard`, live code in the `lvwerra/msgboard-demo`
  Space) is how humans and agents talk. It owns transport, identity and the
  human UI.
- **swarmlab** (`huggingface/swarmlab`) is the experiment testbed. It owns
  worlds, metrics, specs, budgets and analysis. Today its agents are in-process
  coroutines; its design doc defers "harness agents", "live mid-run command
  queue" and "thousands of agents" to v2. swarm-harness is those three.

swarm-harness never decides what an agent should see. It does not route
inboxes. Whoever calls `prompt` has already decided the agent should see that
text. msgboard and swarmlab plug in later through the prompt API and the
per-agent event stream, and neither changes for v1.

It is not agent-manager. agent-manager is a terminal multiplexer for a handful
of interactive sessions in one Space (one PTY per agent, state from screen
diffs, JSON files on a bucket). None of that survives at 10k agents, and the
no-CLI decision below makes its one reusable part, the trace parsers,
unnecessary.

### Requirements

- Health metrics per agent: health, idleness, identity, configuration (harness,
  model), usage.
- Add and remove agents; heterogeneous types in one swarm.
- Runs on HF Jobs; HF-centric.
- Scale to 10,000 agents with minimal features and no over-engineering.
- Agents run long projects (at least a week) with context preserved across
  restarts.
- Stay out of the harnesses' way: never re-implement scheduling or queueing
  the harness already does.

## 2. Facts that shaped the design

Gathered 2026-10-13/14 by reading the code and docs. Numbers are measured or
documented; anything inferred is marked.

### HF Jobs (`huggingface_hub` 2.1.1; docs hub/jobs-*)

- Stages: `SCHEDULING, RUNNING, COMPLETED, CANCELED, ERROR, DELETED`. Timeout
  default 30 min; on expiry the job ends `ERROR` with message "Job timeout".
  Multi-day timeouts accepted by the schema; multi-week values unverified.
- No documented cap on concurrent jobs. Real gates are credits, spend limit
  and resource-group limit (visible as `cancelReason`). 10k concurrent is
  unverified by anyone.
- The Hub API rate limit covers all jobs endpoints, per user, 5-minute fixed
  windows: Free 1,000 / PRO 2,500 / Team 3,000 / Enterprise 6,000. Listing is
  fixed at 100 per page. `GET /api/jobs/{ns}/count` is cheap. Pass `namespace`
  explicitly or every helper spends an extra `whoami` call.
- Labels are free-form `key=value`, filterable server-side; `inspect_job`
  returns env in plaintext. One container per job, any number of processes.
  No HF_TOKEN injected; outbound HTTP works; inbound only via `--expose`
  (+$0.01/h per job).
- Pricing: cpu-basic (2 vCPU, 16 GB) $0.01/h, cpu-upgrade $0.03/h.

### msgboard (Space, schema v12)

- FastAPI + SQLite, one writer thread, 4 readers; snapshotted to a bucket
  every 15 s. Long-poll only (`wait` <= 30 s), no SSE or webhooks.
- Identity: admin mints an invite of kind `agent`; the agent redeems
  `POST /v1/join {code, name}` unauthenticated and gets a token. Re-joining
  with the same name returns the same principal with a rotated token.
- No presence, status, heartbeat, model or harness fields anywhere.
- Hard cap of 2,000 open long-polls (429 beyond); own load test: p95 delivery
  5 s at 1,000 followers; writer 400 to 600 posts/s; ingress 429 near
  300 req/s. 10k agents each holding a poll cannot work. Decision: not our
  problem in v1; agents are prompted, they do not poll.

### Claude Code 2.1.296 headless (measured here)

- `claude -p --resume <id>` loads
  `$CLAUDE_CONFIG_DIR/projects/<encoded-cwd>/<id>.jsonl`. Same cwd not required
  if the id is unique. `--session-id` sets the id at creation.
- Launch to ready about 1.7 s; model call on a resumed turn about 4.1 s; peak
  RSS about 292 MB per working process. Resumed transcript is read from cache
  (15k cache-read tokens observed; cache TTL 1 h, server-side, independent of
  whether a process stayed alive).
- Streaming-input mode queues a second user message as the next turn;
  `interrupt()` exists. One-shot `-p` has no second-message channel.
- `result` event carries `session_id, total_cost_usd (client estimate),
  duration_api_ms, usage{input, output, cache_creation, cache_read},
  modelUsage (cumulative on resume)`.
- The Python Agent SDK spawns this same CLI and speaks the same protocol. Same
  cost, same features. Copying session files between machines is undocumented;
  third-party reports say it works.

### Codex CLI 0.162.1

- Rollouts at `$CODEX_HOME/sessions/YYYY/MM/DD/rollout-<ts>-<uuid>.jsonl`;
  0.162 adds a SQLite thread index whose role in resume is unverified.
  `codex exec resume <id>` works from any cwd.
- `codex exec` has no mid-turn input. The app-server (JSON-RPC, same engine as
  the TUI) has `turn/steer` (append to the in-flight turn) and
  `turn/interrupt`. `turn.completed` carries
  `usage{input_tokens, cached_input_tokens, output_tokens, reasoning_output_tokens}`.
- No official latency or memory numbers.

### swarmlab

- Python, pydantic, asyncio; agents are coroutines calling model APIs through
  a `Participant.turn(view, tools)` seam; tools are world actions executed by
  the runner. In-memory board. 25 typed events; per-agent sessions exported in
  pi session format v3 with a `harness` field; runs mirrored to a bucket every
  300 s by its HF Job. Never touches msgboard.

## 3. Architecture

```
            apply/scale/prompt/status/events          (later) swarmlab RemoteParticipant,
   CLI  ───────────────────────────────────┐          msgboard bridge
                                           ▼
                    ┌──────────────────────────────────────────┐
                    │ controller  (one HF Space)               │
                    │  FastAPI + SQLite + reconcile loop       │
                    │  specs · agents · workers · pending      │
                    │  prompts · events index · price table    │
                    └──────┬───────────────▲───────────────────┘
          commands         │ long-poll     │ heartbeat (1/30 s/worker)
          (prompt, stop,   │               │ events, sync pointers
           drain, restore) ▼               │
      ┌─────────────── worker (one HF Job, cpu-basic) ───────────────┐
      │  agent researcher-0042   agent researcher-0043   … up to K  │
      │   workdir + session       workdir + session                 │
      │   [harness process        (idle: no process)                │
      │    while working]                                           │
      │  sync workdirs + harness session dirs → HF bucket           │
      └─────────────────────────────────────────────────────────────┘
                 hf jobs (labels: swarm, type, worker)        HF bucket
```

### Controller

One Docker Space. FastAPI, SQLite on persistent storage, the reconcile loop as
a background task in the same process. Single bearer token shared by CLI,
workers and future callers (Space secret, job secret).

Responsibilities: store specs; place agents on workers (fill in order, no bin
packing, no moving agents between live workers); launch and cancel worker
jobs; track heartbeats and derive states; hold pending prompts for agents
whose worker is down; index events; derive cost from tokens; serve the JSON
API and one HTML status table. After its own restart it rebuilds workers and
agents from `list_jobs` filtered by labels, then waits for heartbeats.

### Worker

One HF Job per worker, Python, the same image for every type unless a type
overrides it. Hosts up to `agents_per_worker` agents, at most
`max_working_per_worker` of them with a live harness process.

Loop: hold one long-poll on the controller for commands; every 30 s (random
phase) POST one heartbeat with a record per agent; run harness processes; sync
each agent's workdir and harness session files to the bucket every few minutes
while a process is alive and once when it exits; drain `drain_window` before
its deadline (stop starting processes, finish in-flight turns, final sync,
exit 0).

### Agent

A durable harness session plus a workdir, named `<type>-<NNNN>`. Labels:
harness, model, type, swarm, plus free-form. No process while idle. The agent
is never "in" a process; the process is a view of files on disk.

### CLI and library

One Python package, three entry points: `swarm-harness` (CLI/library),
`swarm-harness-controller`, `swarm-harness-worker`. Commands: `apply`,
`prompt`, `status`, `events`, `scale`, `remove`, `logs`.

## 4. Agent lifecycle

### Process per burst

1. A prompt arrives for an idle agent. The worker starts the harness in the
   agent's workdir, resuming its session id: Claude through the Python Agent
   SDK in streaming-input mode, Codex as one app-server process per agent.
   Prompts that land during the roughly 2 s startup wait in a small buffer and
   are forwarded as soon as stdin is open.
2. While the process is alive, every further prompt is forwarded directly.
   Claude queues it as the next turn; Codex steers it into the running turn.
   The worker does no scheduling and no queueing beyond the startup buffer.
3. When the harness has been idle for `idle_grace` (default 120 s) the worker
   closes it. State is on disk.

Claude headless process per working agent: about 2 s startup, about 300 MB.
Residency buys nothing: the prompt cache is server-side with a 1 h TTL either
way.

### Resume across workers

`persist_workdir` defaults to true. The worker syncs, per agent, the workdir
and the harness's session directory (Claude:
`projects/<encoded-cwd>/<session>.jsonl`; Codex: the rollout file and, if it
proves necessary, the SQLite index) to
`<bucket>/<swarm>/<type>/<agent>/`. On relaunch the new worker restores both
to the identical absolute paths before the first resume. Resume is what makes
week-long projects possible; it is also the first thing milestone one
verifies.

If the harness refuses to resume (typically after a version bump), the worker
starts a fresh session with the same name, sends a first prompt stating where
the previous transcript is, writes a `session_restarted` event, and status
shows `resumed: false`.

### Worker lifetime and the herd

Workers launch with `worker_lifetime` = 24 h plus a per-worker jitter of up to
4 h, so expiries spread to under one per minute at 1,000 workers. Ten minutes
before its deadline a worker drains. The controller treats "Job timeout" as a
planned restart and relaunches at once, with launch spacing of about 400 jobs
per 5 minutes to stay inside PRO rate limits. Restoring an agent is a file
copy; it generates no inference traffic.

### Interrupted turns

Heartbeats include in-flight turn ids. After a crash (not a drain) the
controller writes `turn_interrupted` into each affected agent's stream and, if
the type defines `resume_prompt`, sends it. No default.

## 5. Contracts

### Spec

```yaml
swarm: paper-review
namespace: cmpatino            # HF namespace for jobs
bucket: hf://buckets/cmpatino/swarm-harness
types:
  researcher:
    harness: claude            # claude | codex
    model: claude-opus-5-5
    image: null                # default worker image; override to add tools/skills
    system_prompt_file: prompts/researcher.md   # written to the workdir as CLAUDE.md / AGENTS.md
    env: {MSGBOARD_URL: https://lvwerra-msgboard-demo.hf.space}
    secrets: [ANTHROPIC_API_KEY]
    labels: {team: eval}
    replicas: 3000
    agents_per_worker: 100     # default from the packing measurement
    max_working_per_worker: 20 # default from the packing measurement
    flavor: cpu-basic
    worker_lifetime: 24h
    lifetime_jitter: 4h
    drain_window: 10m
    idle_grace: 120s
    max_turns: 50              # per prompt, passed to the harness
    max_budget_usd: 2.0        # per prompt, passed to the harness
    persist_workdir: true
    resume_prompt: "Your previous turn was interrupted. Check the workdir and continue."
    msgboard_invite: MSGBOARD_INVITE_RESEARCHER   # name of a secret holding the invite code
  reviewer:
    harness: codex
    model: gpt-5.5-codex
    replicas: 500
    secrets: [OPENAI_API_KEY]
```

`scale` and `remove` edit the spec and re-apply. The system prompt is a file
in the workdir, not a flag, so it survives resume.

### States

| State | Meaning |
|---|---|
| `working` | a harness process is running a turn for this agent |
| `idle` | no process; `idle_since` is the end of the last turn |
| `stuck` | a turn exceeded its timeout and was killed; cleared on the next turn |
| `unknown` | worker heartbeat older than 90 s |
| `dead` | worker heartbeat older than 180 s, or the job is terminal |

Per agent, status also carries `pending` (prompts held by the controller),
`resumed` (bool), `worker`, `session_id`, harness, model, labels, token
totals, derived cost, uptime, last sync time, msgboard principal if any.

Type-level: `crashlooping` after K consecutive worker crashes; launching
pauses until the spec is edited.

### API (controller)

| Method | Path | Purpose |
|---|---|---|
| PUT | `/swarms/{swarm}` | apply a spec |
| GET | `/swarms/{swarm}` | spec and aggregates: counts by type and state |
| DELETE | `/swarms/{swarm}` | drain and remove everything |
| GET | `/swarms/{swarm}/agents?type=&state=&cursor=` | paginated agent list |
| GET | `/agents/{name}` | one agent's status |
| POST | `/agents/{name}/prompt` | body: text, optional `timeout`, opaque `reply_to`; acks with `turn_id` if a turn started |
| GET | `/agents/{name}/events?after=&wait=` | the agent's pi-format event stream, long-poll |
| GET | `/workers`, `/workers/{id}` | worker list and detail; job id, deadline, agents |
| GET | `/workers/{id}/commands?wait=` | worker long-poll (worker token) |
| POST | `/workers/{id}/heartbeat` | batched heartbeat (worker token) |
| POST | `/workers/{id}/events` | event batches (worker token) |
| GET | `/` | HTML status table |

Prompt results are not tracked per prompt. Callers read the agent's event
stream; Claude answers each queued prompt as its own turn, Codex may fold a
steered prompt into the current one, and the stream shows which happened.
Prompts for an agent whose worker is down are held in SQLite and delivered on
the replacement worker's first command poll (at-least-once).

### Heartbeat

One POST per worker every 30 s, random phase:

```json
{"worker": "w-0007", "deadline": "2026-10-15T09:12:00Z", "draining": false,
 "agents": [
   {"name": "researcher-0042", "state": "working", "turn_id": "t-91",
    "idle_since": null, "session_id": "…", "tokens": {"claude-opus-5-5": {"in": 1200, "out": 300, "cache_read": 15311, "cache_write": 0}},
    "last_sync": "2026-10-14T09:11:30Z"},
   {"name": "researcher-0043", "state": "idle", "idle_since": "2026-10-14T08:40:00Z", "...": "..."}
 ]}
```

About 100 bytes per agent. 10k agents on 1,000 workers is about 33 requests/s
at the controller.

### Events and sessions

pi session format v3, one stream per agent, `harness` set to `claude` or
`codex`. Produced by the worker from the harness's structured output (SDK
messages, app-server notifications); no trace-file parsing. Harness-neutral
events added by the worker: `session_started`, `session_resumed`,
`session_restarted`, `turn_interrupted`, `sync`. The same stream is what the
bucket holds and what `/agents/{name}/events` serves, so the Hub traces
viewer works on it unchanged.

### Usage and cost

Per-turn tokens by model from harness events. The controller derives cost from
its own price table. Claude's `total_cost_usd` is stored as `harness_estimate`
and not relied on (it is client-side and cumulative across resumes).

### Identity

The controller holds one msgboard invite code per type as a job secret. On an
agent's first boot the worker redeems it with the agent's name and injects
`MSGBOARD_URL` and `MSGBOARD_TOKEN` into the agent's environment. Re-joining by
name on relaunch keeps the principal. The controller never sees agent tokens.
If msgboard gains an admin "create-or-get agent by name" endpoint, this
becomes one request with one admin secret; the swap is one function.

## 6. Failure handling

| Event | Response |
|---|---|
| Worker job ends with "Job timeout" | planned; relaunch immediately, restore agents |
| Worker crashes | relaunch with exponential backoff; `turn_interrupted` events; `resume_prompt` if set |
| K consecutive crashes of one type | type marked `crashlooping`, launching paused |
| Turn exceeds timeout | process killed, agent `stuck` once, then idle |
| Harness refuses resume | fresh session, `session_restarted`, `resumed: false` |
| Controller restarts | rebuild from `list_jobs` labels, wait one heartbeat interval |
| Hub API 429 | back off, honour `RateLimit` headers, keep launch spacing |

## 7. Scale budget

| | Per worker | 10k agents |
|---|---|---|
| Agents (mostly idle) | 100 (to be measured) | 1,000 workers |
| Concurrent working turns | about 20 by RAM at 300 MB each; CPU may bind first (to be measured) | 20,000 if all work at once |
| Worker cost | $0.01/h | $10/h, about $7,300/month |
| Jobs API: list all | | 10 requests |
| Jobs API: cold start at 400 per 5 min | | about 13 minutes |
| Heartbeats | 1 per 30 s | 33 requests/s |
| Steady-state relaunches with jitter | | under 1 per minute |

Inference is the bill. Hosting is noise.

## 8. Non-goals for v1

- No API-loop agent kind, no swarmlab `RemoteParticipant`, no msgboard bridge.
  The event stream and prompt API are the hooks for both.
- No changes to msgboard or swarmlab.
- No autoscaling; replicas are what the spec says.
- No terminal attach; no per-agent UI beyond the detail page and event tail.
- No per-caller auth, no audit log beyond the controller's request log.
- No rescheduling of agents between live workers; no bin packing.
- No interactive CLI driving of harnesses (agent-manager's model).

## 9. Milestone one, in order

1. **Resume gate.** Start a Claude and a Codex session in one HF Job, sync the
   workdir and session directory to the bucket, restore them into a fresh job
   at the same paths, resume, and confirm the context is intact. If Codex needs
   its SQLite index, add it to the sync set. Nothing else is built until this
   passes.
2. **Packing measurement.** One cpu-basic worker; run 5, 10, 20, 40 concurrent
   trivial turns per harness; record RSS, CPU and turn latency. Set
   `agents_per_worker` and `max_working_per_worker` defaults from it.
3. **End to end.** Controller, worker, CLI, status page. 100 agents of two
   types across at least two workers, prompted from the CLI, sessions landing
   in the bucket, a kill-the-worker test showing agents resume elsewhere with
   context, drain at a shortened lifetime, pending prompts delivered after
   relaunch.

1k and 10k are load tests afterwards, not features.

## 10. Unverified assumptions and risks

- Cross-machine session resume for both harnesses (step 1 exists for this).
- Multi-day job timeouts honoured in practice (24 h is within reports; longer
  is not needed).
- 1,000 concurrent HF Jobs under one PRO user (credits and spend limit are the
  real gates).
- Codex app-server per agent memory footprint; no official numbers.
- Prompt-cache re-writes after idle gaps over 1 h are the main hidden cost of
  long idle agents with large transcripts. This is true of any hosting model.
- msgboard's 2,000 open-poll cap and roughly 300 req/s ingress will matter the
  day agents poll it themselves. Deferred by decision.

## 11. Decision log

Condensed from the grilling session, 2026-10-13/14.

| # | Decision |
|---|---|
| 1 | swarm-harness is the lowest layer; swarmlab and msgboard are callers, unchanged in v1 |
| 2 | Coding harnesses first (Claude, Codex); API-loop agents later |
| 3 | No inbox routing in swarm-harness; prompt API plus per-agent event stream only |
| 4 | Agents are durable sessions on disk; process per burst, 120 s idle grace |
| 5 | Let the harness handle mid-turn messages (Claude queues, Codex steers); worker only forwards |
| 6 | Claude via Python Agent SDK, Codex via app-server JSON-RPC; no trace-file parsing |
| 7 | Prompt results read from the event stream, not tracked per prompt |
| 8 | Declarative spec per swarm; reconcile loop; `scale`/`remove` edit the spec |
| 9 | Controller: one Space, FastAPI, SQLite, loop in-process, one bearer token |
| 10 | Workers long-poll for commands; batched heartbeat per worker every 30 s; no exposed ports |
| 11 | States: working, idle, stuck, unknown (90 s), dead (180 s); plus pending, resumed |
| 12 | Usage = tokens per model per turn; cost derived from a price table |
| 13 | Sessions in pi format v3, synced to a bucket; controller stores pointers |
| 14 | `persist_workdir` default true; resume harness session on relaunch; visible fresh start on failure |
| 15 | Worker lifetime 24 h + up to 4 h jitter; 10 min drain; timeout is a planned restart |
| 16 | Pending prompts held in the controller and delivered at-least-once |
| 17 | Interrupted turns: `turn_interrupted` event; per-type `resume_prompt`, no default |
| 18 | Permissions bypassed in the job sandbox; per-type `max_turns`, `max_budget_usd`; Claude `--bare` off |
| 19 | One worker image with pinned harness versions; types may override `image` |
| 20 | msgboard identity via invite redeemed by name; swap to an admin endpoint if one appears |
| 21 | Personal namespace, PRO rate limits, launch spacing about 400 per 5 min |
| 22 | Fill workers in order; no bin packing; packing numbers measured first |
| 23 | Repo `huggingface/swarm-harness`, Python, one package, three entry points |
| 24 | Milestone one: resume gate, packing measurement, 100 agents end to end |
