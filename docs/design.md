# flock: design

Status: agreed design, 2026-10-14. Author: Carlos Patiño with Claude.
Scope: v1, the first milestone and the contracts later layers depend on.

## 1. What it is

flock hosts coding-harness agents (Claude Code, Codex, Pi) on Hugging Face
Jobs, keeps them alive for weeks, takes prompts for them, and exposes their
sessions and health. It is the lowest layer under two existing systems:

- **msgboard** (`huggingface/msgboard`, live code in the `lvwerra/msgboard-demo`
  Space) is how humans and agents talk. It owns transport, identity and the
  human UI.
- **swarmlab** (`huggingface/swarmlab`) is the experiment testbed. It owns
  worlds, metrics, specs, budgets and analysis. Today its agents are in-process
  coroutines; its design doc defers "harness agents", "live mid-run command
  queue" and "thousands of agents" to v2. flock is those three.

flock never decides what an agent should see. It does not route
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
- Added 2026-10-20 (`huggingface_hub` 2.2.0, Hub OpenAPI, docs hub/jobs-configuration):
  - `timeoutSeconds` is bounded only by 2^53 in the schema, and the docs name no maximum.
  - Every container gets `JOB_ID`. The proxy URL of a port is
    `https://<JOB_ID>--<port>.hf.jobs`, also listed in `status.expose_urls`.
  - `--expose` requires a read token for the namespace, and `--expose-public` requires none.
    Both can be changed on a running job (`update_job_expose`). Env and secrets cannot.
  - Network groups: `network_group` puts jobs of one namespace and resource group together on
    every port. Members get `HF_NETWORK_GROUP_HOSTNAME` (all members) and
    `HF_NETWORK_GROUP_PREFIX`, and `${HF_NETWORK_GROUP_PREFIX}<alias>` resolves to every member
    claiming the alias. Several jobs may claim one alias. Members resolve before they are
    ready, so callers retry.
  - Env names must match `^[a-zA-Z][_a-zA-Z0-9]+$`, which is why one-letter names are rejected.

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
- Model providers (checked 2026-10-16 against the installed 0.162.1 with `--strict-config`
  and a local fake Responses server): `[model_providers.<id>]` takes `name`, `base_url`,
  `env_key`, `wire_api`, `request_max_retries`, `stream_max_retries`,
  `stream_idle_timeout_ms`; `model_provider`, `model` and `model_reasoning_effort` select
  them. `wire_api = "chat"` is refused ("no longer supported", openai/codex discussion
  7782): an OpenAI-compatible server must serve the **Responses API** (`POST /v1/responses`,
  streamed). `openai`, `ollama` and `lmstudio` are reserved ids. The same keys work as
  `codex app-server -c key=value` overrides.

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
a background task in the same process. Since 2026-10-20 the same process can instead run as
an HF Job in its workers' network group, with its SQLite synced to a bucket (job mode,
decision 32, [deploy-job.md](deploy-job.md)). Single bearer token shared by CLI,
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

One Python package, three entry points: `flock` (CLI/library),
`flock-controller`, `flock-worker`. Commands: `launch`,
`stop`, `update`, `prompt`, `status`, `events`, `remove`.

## 4. Agent lifecycle

### Process per burst

1. A prompt arrives for an idle agent. The worker starts the harness in the
   agent's workdir, resuming its session id: Claude through the Python Agent
   SDK in streaming-input mode, Codex as one app-server process per agent, Pi as
   one `pi --mode rpc` process per agent.
   Prompts that land during the roughly 2 s startup wait in a small buffer and
   are forwarded as soon as stdin is open.
2. While the process is alive, every further prompt is forwarded directly.
   Claude queues it as the next turn; Codex and Pi steer it into the running turn.
   The worker does no scheduling and no queueing beyond the startup buffer.
3. When the harness has been idle for `idle_grace` (default 120 s) the worker
   closes it. State is on disk.

Measured on cpu-basic (experiments/packing/RESULTS.md, 2026-10-14): a cold Claude
turn costs about 1.5 CPU-seconds, mostly CLI startup, and 85 to 110 MB of real
container memory (RSS double-counts shared pages); an idle resident session
costs about 116 MB and no CPU. The two cores saturate at 5 concurrent cold
turns, throughput about 1.35 turns/s, and memory never bound up to N=60. So
CPU, not memory, limits a worker, and residency does save CPU: a prompt to a
live process costs about 0.1 CPU-s against about 1 CPU-s cold. The prompt
cache is server-side with a 1 h TTL either way. Hence the 120 s idle grace and
the measured defaults of 50 agents and 20 concurrent turns per worker.

Pi (`@earendil-works/pi-coding-agent` 1.2.0, `harness: pi`, docs/pi.md) follows the same
model: one `pi --mode rpc` process per burst, cwd = the workdir, JSONL commands on stdin and
events on stdout. Every agent on a worker shares `PI_CODING_AGENT_DIR=/state/pi` (same type,
same `models.json`, which the worker writes with the API key as an env-var reference) and has
its own `--session-dir /state/pi/sessions/<agent>`; the worker passes a session id it
generates. Prompts go out as `prompt` with `streamingBehavior: "steer"`, which starts a run
when Pi is idle and queues into the running one when busy; a flock turn runs from the start
of a Pi run to `agent_settled`. Usage is per model call, not cumulative. The system prompt is
the workdir's `AGENTS.md`. Pi's harness environment is an allow-list (no `FLOCK_TOKEN`).
Pi has no turn or budget options, so the worker enforces `max_turns` and `max_budget_usd`: it
counts model calls and prices each one with the flock price table, and sends Pi `abort` before
a call past either limit (turn ends `error`, `reason` max_turns|max_budget_usd).
Packing numbers for Pi are not measured yet; idle VmRSS of one RPC process was 117 MB locally.

The `llm` harness (`harness: llm`, docs/llm-harness.md) has no process: a "burst" is an asyncio
task in the worker that runs a Chat Completions loop with one `bash` tool against an
`openai_compat` provider (required). The session is a JSONL file of chat messages under
`$FLOCK_LLM_HOME/sessions/<agent>/` (`/state/llm`), appended per step, reloaded on resume and
compacted (system + a note + the newest messages and steps) when the prompt projected from the server's last `prompt_tokens` passes the type's `context_tokens`. A flock
turn ends on a reply without tool calls; prompts sent mid-turn become the next user message;
`max_turns` / `max_budget_usd` are checked before each model call, as for Pi and Codex. Usage
is per call. `extra` is sent as body fields; `reasoning` is ignored. Tool calls written as
text (`{"tool": "bash", ...}` or `[bash(command=...)]`) are accepted for small models. A type's
`tool_choice` (`required` or `bash`) forces a call on every request; with the `done` tool
(`done_tool`, default on for `required`) a turn runs until `done` or `max_turns` (50 if unset),
without it each prompt is one action. Thinking (`reasoning_content` or inline `<think>`) is shown, never resent; the
transcript keeps exactly one system message, first (LFM2's template requires it).

### Resume across workers

`persist_workdir` defaults to true. The worker syncs, per agent, the workdir
and the harness's session directory (Claude:
`projects/<encoded-cwd>/<session>.jsonl`; Codex: the rollout file and, if it
proves necessary, the SQLite index; Pi: `sessions/<agent>/<ts>_<session>.jsonl`) to
`<bucket>/<swarm>/<type>/<agent>/`. On relaunch the new worker restores both
to the identical absolute paths before the first resume. Resume is what makes
week-long projects possible; it is also the first thing milestone one
verifies.

Buckets keep bytes, not file modes, so each push also writes `<workdir>/.flock-modes.json`
(the octal mode of every file with an executable bit) and restore re-applies it; without it an
executable script came back 0644 after a re-apply (pixel run 1, 2026-10-19).

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
compute: hf-jobs               # default; `local`: workers as subprocesses, `flock run` (docs/local.md)
namespace: cmpatino            # HF namespace for jobs (required for hf-jobs, ignored locally)
bucket: hf://buckets/cmpatino/flock
providers:                     # optional: where open models are served (see "Providers")
  hf:   {type: openai_compat, base_url: https://router.huggingface.co/v1, api_key_env: HF_TOKEN,
         pricing: {"Qwen/Qwen3.8-27B:deepinfra": [0.2, 2.5, 0.2]}}  # router provider pinned
  gpu1: {type: openai_compat, base_url: "https://qwen.example.org/v1",  # your vLLM, over HTTPS
         api_key_env: VLLM_API_KEY,
         self_hosted: true, max_working: 256}
models:                        # optional aliases
  qwen-27b: {model: "gpu1:Qwen/Qwen3.8-27B-FP8", reasoning: medium}
types:
  researcher:
    harness: claude            # claude | codex | pi | llm
    model: claude-opus-5-5
    image: null                # default worker image; override to add tools/skills
    system_prompt_file: prompts/researcher.md   # written to the workdir as CLAUDE.md / AGENTS.md
    env: {MSGBOARD_URL: https://lvwerra-msgboard-demo.hf.space}
    secrets: [ANTHROPIC_API_KEY]
    labels: {team: eval}
    replicas: 3000
    agents_per_worker: 50      # measured default, experiments/packing
    max_working_per_worker: 20 # measured default, experiments/packing
    flavor: cpu-basic
    worker_lifetime: 24h
    lifetime_jitter: 4h
    drain_window: 10m
    idle_grace: 120s
    max_turns: 50              # model calls per prompt; enforced for every harness
    max_budget_usd: 2.0        # spend per prompt (flock price table); enforced likewise
    context_tokens: null       # llm harness only: compaction threshold, default 32000
    tool_choice: null          # llm harness only: auto | required | none | bash
    done_tool: null            # llm harness only: offer `done` (default: on for tool_choice required)
    persist_workdir: true
    workdir: {soft_files: 50000, soft_mb: 2048, ephemeral: [scratch/]}  # docs/pi.md "Workdir"
    worker: {sync_interval_s: 180, restore_threads: 4}   # live worker knobs, docs/slurm.md
    resume_prompt: "Your previous turn was interrupted. Check the workdir and continue."
    msgboard_invite: MSGBOARD_INVITE_RESEARCHER   # name of a secret holding the invite code
    tools: [ws_write, run_bot] # functions from the tools folder, or `all` (docs/tools.md)
  reviewer:
    harness: codex
    model: gpt-5.5-codex
    replicas: 500
    secrets: [OPENAI_API_KEY]
  analyst:
    harness: codex
    model: qwen-27b            # alias; or "hf:Qwen/Qwen3.8-27B:deepinfra" directly
    reasoning: low             # optional; overrides the alias
    replicas: 200
```

`scale` and `remove` edit the spec and re-apply. The system prompt is a file
in the workdir, not a flag, so it survives resume. `flock launch` reads
`system_prompt_file` (relative to the spec file) and sends its text as
`system_prompt`, since neither the controller nor the workers can read the
caller's files (added in the e2e run). `system_prompt_append_file` (optional) is appended
after a blank line, so several types can share one prompt plus a section of their own
(experiments/run4's spawner and child types).

Tools folder (docs/tools.md): `tools_dir` (top level, default `tools/` next to the spec file)
holds Python functions; `types.<t>.tools` selects them. `flock launch` packs the folder,
uploads it and writes its content hash into the spec as `tools_version`; assigns
(`AssignCommand.tools_version`) and heartbeat responses (`{"tools": {"version", "select"}}`)
carry it to the workers, which switch at the next turn.

Comms (docs/comms.md): a swarm-level `comms:` block (`kind: msgboard`, `url`, `invite_env`,
`channel`, `kickoff_file`, `rules`, `protocols`, `wake`, `search`) and per-type `comms: true` or
`comms:*` entries in `tools`. The controller resolves the channel id and posts the kickoff once
at apply (`comms.channel_id`, `comms.kickoff_id`); assigns (`AssignCommand.comms`) and heartbeat
responses (`{"comms": {...block, "tools"}}`) carry the block, so rules and protocols are live.

### Providers

Open models (vLLM in a GPU job, the HF router, any OpenAI-compatible or Anthropic-compatible
server) are reached through a `providers:` block. Field names are swarmlab's
(`swarmlab/providers/`), so a block moves between the two projects. How-to:
`docs/open-models.md`.

| Field | Meaning |
|---|---|
| `providers.<name>.type` | `openai_compat` (Codex: Responses API; Pi and `llm`: Chat Completions) or `anthropic` (Claude Code; Messages API) |
| `providers.<name>.base_url` | required for `openai_compat` (ends in `/v1`); optional for `anthropic` (API root, default Anthropic) |
| `providers.<name>.api_key_env` | NAME of the secret holding the key; the value is never in the spec |
| `providers.<name>.self_hosted` | price defaults to `[0, 0, 0]`; the real cost is the GPU job |
| `providers.<name>.max_working` | max agents of the swarm on this provider in a turn at once; enforced at prompt admission (`FLOCK_MAX_WORKING_ENFORCE=0`: report only) |
| `providers.<name>.auth` | `anthropic` only: `x-api-key` (default) or `bearer` (Claude Code's `ANTHROPIC_AUTH_TOKEN`) |
| `providers.<name>.pricing` | `{model id or "*": [input, output, cached]}` USD per 1M tokens |
| `providers.<name>.timeout_s`, `max_retries` | passed to the harness (Codex provider table; Claude `API_TIMEOUT_MS`, `CLAUDE_CODE_MAX_RETRIES`) |
| `models.<alias>.model` | `"<provider>:<model id>"` |
| `models.<alias>.reasoning` | `off`, `low`, `medium`, `high` |
| `models.<alias>.extra` | raw request-body fields (swarmlab's `extra`); sent by the `llm` harness only |
| `models.<alias>.pricing` | `[input, output, cached]`, overrides the provider's |
| `types.<t>.model` | an alias, else `"<provider>:<model id>"`, else a harness-native id (v1) |
| `types.<t>.reasoning` | overrides the alias's |

Resolution of `types.<t>.model`: an alias under `models:`; else `<prefix>:<id>` with the prefix
under `providers:` (split on the first colon, so `hf:Qwen/Qwen3.8-27B:ovhcloud` keeps the router
suffix); else a harness-native id with no provider. A prefix that does not look like a provider
name (`[a-z][a-z0-9_-]*`, e.g. Bedrock's `anthropic.claude-...-v1:0`) is native.
`models.resolve_model(spec, type_spec)` returns the `ResolvedModel` or None.

Checked at apply (HTTP 422 / CLI error naming the type and field): an unknown prefix in a
type or a `models:` entry; a provider-backed model with no price in the alias, the provider's
row for the id, or its `"*"` row, unless `self_hosted`; harness against provider type
(`claude` needs `anthropic`, `codex`, `pi` and `llm` need `openai_compat`; native ids pass,
except on `llm`, which needs a provider).

Pinning: validation writes `types.<t>.resolved_model` (provider, model id, reasoning, extra and
the `[input, output, cached]` price) and the self-hosted `[0, 0, 0]` default into the stored
spec. Workers get the resolved model inside `AssignCommand.spec`; the controller prices with it.

Keys: the controller adds each provider-backed type's `api_key_env` to that type's job secrets
(it must be in the controller's environment, like every secret). The worker reads it from its
own environment and hands it to the harness: under the same name for Codex (`env_key`), as
`ANTHROPIC_API_KEY` for Claude Code (plus `ANTHROPIC_BASE_URL` when `base_url` is set).

Harness configuration: Codex gets `-c` overrides on `codex app-server` (the provider table,
`model_provider`, `model`, `model_reasoning_effort`; `off` is `none`), not a written
`config.toml`, because `CODEX_HOME` is per worker and synced. Reserved ids (`openai`,
`ollama`, `lmstudio`) become `flock-<name>`. Claude Code gets the SDK's `effort` for
low/medium/high and `thinking: disabled` for off.

`max_working`: counted per provider from heartbeats (agents in `working`) and shown in
`GET /swarms/{swarm}` (`status.providers`), `flock status` and the status page. The controller
enforces it at prompt admission: on a command poll, a prompt for an agent that is not in a turn
goes out only while the provider has a free slot; otherwise it stays in `pending_prompts` and
goes out on a later poll (never rejected). Free slots go to pending prompts in FIFO order of
creation across all of the provider's workers (oldest pending prompt first), not to whichever
worker polls first; a worker that is not running (dead, draining, launching) holds no place.
Slots are fresh heartbeat `working` agents, overridden by what the controller learned since:
a prompt handed out takes the slot at once, and a `turn_ended` / `turn_interrupted` /
`prompt_deferred` event releases it at once and wakes the polls of the workers next in line
(pixel run 1k: holding slots until the next heartbeat and first-poller-wins let 384 of 1,000
agents run at `max_working: 128`). Still soft by about one heartbeat: an event batch arrives
up to `flush_interval` (3 s) after the turn ends, and a heartbeat built before the worker
saw a turn change can briefly overrule an event. A capped
provider's `resume_prompt` is queued behind the assign so it is admitted the same way.
`FLOCK_MAX_WORKING_ENFORCE=0` is an escape hatch (report only). Hard limits belong at the endpoint.

Reachability: a provider is a `base_url` every worker reaches over HTTPS. Workers are CPU jobs
and inference runs elsewhere; flock does not colocate workers with GPU jobs.

Preflight: `POST /swarms/{swarm}/preflight` (body: a spec; none = the applied one) probes each
distinct provider-backed model from the controller (which holds the keys) with two streamed
turns: a tool call to a trivial `get_time` tool, then the tool result sent back. The Responses
API for `openai_compat` (what Codex speaks), the Messages API for `anthropic`, non-streamed
Chat Completions (with `extra`) for types on the `llm` harness. Best-effort: a
failed probe refuses `flock launch` (`--skip-preflight` bypasses); a model not probed (key not
in the controller's environment, or preflight skipped) is reported `unverified` with a warning
and applied anyway unless `--strict`. Zero streamed usage is a warning.
`flock update --check` runs it alone on a running swarm.

### Harness environment

A harness process gets an allowlist of the worker's environment (`PATH`, `HOME`, `USER`,
`SHELL`, `TERM`, `LANG`, `LC_*`, `XDG_*`, `TZ`, `TMPDIR`, proxy and CA variables, `IS_SANDBOX`,
its config dir) plus what the worker passes: the type's declared `secrets`, `env`, the msgboard
identity, and the provider key. `FLOCK_TOKEN`, the incarnation and other job secrets stay in the
worker (before 2026-10-16 both adapters passed `os.environ` whole). The Claude SDK merges
`os.environ` itself, so its `cli_path` is a small exec wrapper that drops every name the worker
did not mean to pass.

### States

| State | Meaning |
|---|---|
| `working` | a harness process is running a turn for this agent |
| `idle` | no process; `idle_since` is the end of the last turn |
| `stuck` | a turn exceeded its timeout and was killed; cleared on the next turn |
| `unknown` | worker not seen for 90 s, or the agent's own report is older than 90 s |
| `dead` | worker not seen for 180 s, or the job is terminal; or the worker's heartbeats arrive without this agent for 180 s |

"Seen" is the worker's last heartbeat, command poll or event post from its current job
(`workers.last_seen`): a Space edge can throttle heartbeats (HTTP 429) while other requests
get through, so a worker is declared dead, cancelled and relaunched only after the full
180 s with none of them arriving. Workers retry 429 and 5xx inside their controller client
with exponential backoff and full jitter, never sooner than `Retry-After` (at most 30 s a
wait; a heartbeat retries for up to 20 s, polls and event posts for 30 s), and keep at most
three requests in flight: one poll, one heartbeat, one event batch (events are batched every
3 s; polls are spaced at least 1 s apart even when commands keep coming).

Per agent, status also carries `pending` (prompts held by the controller),
`resumed` (bool), `worker`, `session_id`, harness, model, labels, token
totals, derived cost, uptime, last sync time, msgboard principal if any.

Type-level: `crashlooping` after K consecutive worker crashes; `launch_refused` when the
backend refuses a launch with a 4xx (not 408/429; e.g. an unknown or unplaceable flavor), with
its message in `launch_error`. Either pauses launching until the spec is re-applied. Transient
launch errors back off 30 s doubling to 5 min, last error in `launch_error`.

### API (controller)

| Method | Path | Purpose |
|---|---|---|
| PUT | `/swarms/{swarm}` | apply a spec |
| POST | `/swarms/{swarm}/tools` | upload a tools bundle (tar.gz, docs/tools.md); stored under its content hash |
| GET | `/swarms/{swarm}/tools`, `/swarms/{swarm}/tools/{version}` | stored versions and the current one; one bundle (workers fetch it) |
| POST | `/swarms/{swarm}/preflight` | probe each provider-backed model with one tool-calling request; body: a spec (none: the applied one) |
| GET | `/swarms/{swarm}` | spec and aggregates: counts by type and state |
| DELETE | `/swarms/{swarm}` | drain and remove everything |
| GET | `/swarms/{swarm}/agents?type=&state=&cursor=` | paginated agent list |
| GET | `/agents/{name}` | one agent's status |
| POST | `/agents/{name}/prompt` | body: text, optional `timeout`, opaque `reply_to`; acks with `turn_id` if a turn started |
| POST | `/swarms/{swarm}/prompts` | body: `type`, optional `state`, text, `timeout`, `reply_to`; one pending prompt per matching agent (agents being removed skipped), admitted as single prompts are (`max_working`); `skip_pending` (default on with `state: idle`) skips agents that already have an undelivered prompt, so repeated calls never stack; `state` reads idle/working corrected by turn events and handed-out prompts since the last heartbeat; acks counts `matched`, `queued`, `pending`, `skipped_state`, `skipped_removing`, `skipped_pending`. Controller token only |
| GET | `/agents/{name}/events?after=&wait=` | the agent's pi-format event stream, long-poll |
| GET | `/workers`, `/workers/{id}` | worker list and detail; job id, deadline, agents |
| GET | `/workers/{id}/commands?wait=` | worker long-poll (worker token) |
| POST | `/workers/{id}/heartbeat` | batched heartbeat (worker token) |
| POST | `/workers/{id}/events` | event batches (worker token) |
| GET | `/` | HTML status table |

Prompt results are not tracked per prompt. Callers read the agent's event
stream; Claude answers each queued prompt as its own turn, Codex and Pi may fold a
steered prompt into the current one, and the stream shows which happened.
Prompts for an agent whose worker is down are held in SQLite and delivered on
the replacement worker's first command poll (at-least-once).

Per-prompt limits: Claude, Codex, Pi and `llm` all enforce `max_turns` (model calls) and
`max_budget_usd` (spend, priced from the flock table). Claude gets both as SDK options.
Codex and Pi have no such option, so the worker adapter counts model calls (Codex: one
`thread/tokenUsage/updated` per call) and prices each call; before the call that would
pass a limit it stops the harness (Codex `turn/interrupt`, Pi `abort`, `llm` simply does not
call) and the turn ends
`error` with `reason` max_turns|max_budget_usd, `subtype` error_max_turns|error_max_budget_usd
and `num_turns`. The session stays usable. A Codex steer into a turn that has used its
allowance is held and sent as the next turn once that turn ends.

Each launch gets an incarnation token (job env `WORKER_INCARNATION`); workers send
it as `?incarnation=` on command polls, heartbeats and event posts (events from a replaced
job are still stored but do not count as seen), and the controller answers 409
when it is not the slot's current job, including to a poll that was parked when the
slot was relaunched. A heartbeat that omits agents placed on that worker re-queues
their assign. Both came from the e2e run (experiments/e2e/RESULTS.md, check 6).

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

The worker also sends `spec_rev`, a hash of the type spec it has. The response carries
`tools` and `comms` (docs/tools.md, docs/comms.md), `type_spec` and `spec_rev` when the type
spec changed. The worker applies a new type spec at each agent's next turn boundary
(docs/live.md).

Telemetry, all optional (older workers send none; the controller stores the last values and
`GET /workers`, `GET /agents/{name}` return them):

- `mem` `{current_mb, limit_mb, nonreclaim_mb, slope_mb_h, eta_limit_s, psi_some_avg10,
  oom_kills}`: the worker's cgroup, sampled every 15 s (`flock.worker.cgroup`). The limit is the
  smallest `memory.max` up the tree, else `FLOCK_MEMORY_LIMIT_MB`; `nonreclaim_mb` is anon +
  shmem + slab_unreclaimable + kernel_stack + pagetables; the slope is over the last hour.
- `restore` `{agents, files_done, files_total, started}`: restores in flight, else null.
- `tools` `{name: {n, err, refused, timeouts, sum_s, max_s}}`: world tool calls since the worker
  process started; `refused` is an HTTP 429 or a busy/budget error code.
- `knobs_rev`: a hash of the worker's own settings (drain budget, sync interval, comms rates,
  sync excludes).
- `wants_rotation`: nonreclaimable memory passed `FLOCK_MEM_ROTATE` of the limit; the controller
  drains the worker within its rotation budget (docs/slurm.md "Memory").
- per agent `rev` (the type spec revision applied at its last turn boundary) and `start_rev` (the
  one its running harness process started with).

Each push also writes `<agent>/meta.json` `{files, bytes, ts}` in the bucket when the workdir's
size changed, and every FLOCK_MEMORY_LOG_S (600 s) the worker logs one memory line per agent and
an `agent_memory` event (`pi_pss_mb`, `tools_pss_mb`, `tools_by_comm`, `tmpfs_root_mb`, `tmp_mb`,
`workdir_files`).

`tokens` are cumulative per agent across workers (the worker restores them from the bucket
state). An empty map means "not known yet", for an agent still restoring on a relaunched
worker, and leaves the controller's totals alone. Folding there double-counted cost in the 1k
run (experiments/load-1k/RESULTS.md, bug 2).

### Events and sessions

pi session format v3, one stream per agent, `harness` set to `claude`,
`codex` or `pi`. Produced by the worker from the harness's structured output (SDK
messages, app-server notifications, Pi RPC events, whose messages are already pi v3
and pass through unchanged); no trace-file parsing. Harness-neutral
events added by the worker: `session_started`, `session_resumed`,
`session_restarted`, `turn_interrupted`, `sync`. The same stream is what the
bucket holds and what `/agents/{name}/events` serves, so the Hub traces
viewer works on it unchanged.

### Usage and cost

Per-turn tokens by model from harness events. The controller derives cost from
its own price table. Claude's `total_cost_usd` is stored as `harness_estimate`
and not relied on (it is client-side and cumulative across resumes).

A provider-backed type's model id is priced from its pinned `[input, output, cached]` row
(cache writes at the input price), which wins over the table. Since 2026-10-16 an agent's cost
grows by the price of each heartbeat's new tokens rather than being recomputed from its totals,
so a price changed by a re-apply (or `FLOCK_PRICES`) applies from then on and does not
reprice history.

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
| Launch refused (Hub 4xx) | type marked `launch_refused` with the Hub's message, no retries until re-apply |
| Launch fails transiently (5xx, network) | retry with backoff 30 s doubling to 5 min; last error in status |
| Turn exceeds timeout | process killed, agent `stuck` once, then idle |
| Harness refuses resume | fresh session, `session_restarted`, `resumed: false` |
| Controller restarts | rebuild from `list_jobs` labels, wait one heartbeat interval |
| Hub API 429 | back off, honour `RateLimit` headers, keep launch spacing |

## 7. Scale budget

Measured on cpu-basic (100 agents, experiments/e2e) and cpu-upgrade (1,000 agents,
experiments/load-1k), both 2026-10-14.

| | cpu-basic worker ($0.01/h) | cpu-upgrade worker ($0.03/h) | 10k agents on cpu-upgrade |
|---|---|---|---|
| Agents per worker (mostly idle) | 50 (default) | 200 (all resident is about 26 GB of 32) | 50 workers |
| Concurrent working turns | 20; about 1.35 cold turns/s | 120; 4.2 to 4.9 cold turns/s, turn p95 under 30 s | about 220 cold turns/s |
| Worker cost | $0.01/h | $0.03/h | $1.50/h |
| Cold start to first heartbeat | about 30 s | 28 to 46 s after apply | one launch pass; under 400 per 5 min |
| Heartbeats | 1 per 30 s (mean 30 ms at the controller) | | 1.7 requests/s |
| Jobs API per hour (steady) | about 1 `inspect_job` per worker-state change; no `list_jobs` | | tens of calls |
| Bucket sync, per cycle | 4 Hub calls and about 0.6 s per changed agent, serial | about 120 s for 200 changed agents | 40,000 calls if everyone is active (item 1 in §12) |
| Restore after relaunch | 7 Hub calls per agent | 200 agents in 17 s | 70,000 calls for a mass restart |
| Controller at 1k | 87 MB RSS, under 2% of a core, status page 54 ms | | measured at 10k: 155 MB, status page 0.25 s and 25 KB, 0 5xx of 30,018 requests |
| Heartbeat write at the controller | 30 ms mean at 1k | | 187 ms mean, 10.5 s max at 120 req/s (SQLite single writer) |
| Mass restore | | 200 agents in 17 s | 50 workers x 200 agents in 15 s p50 / 29 s max; 50,000 bucket calls in 2 min, 0 429s |

Inference is the bill. Hosting is noise. The 1k run spent about $2.26 on haiku and about
$0.15 on jobs; the 10k run about $11.3 and $0.53.

The 10k run (experiments/load-10k/RESULTS.md, 2026-10-14) reached 10,000 idle agents 58 s
after apply, delivered 10,000/10,000 prompts and ran at least 9,958 turns, then stopped early:
the Spaces edge throttled the workers' own heartbeats and polls at an aggregate of about 90 to
120 req/s, the controller took the silent workers for dead and crash-relaunched all 50. The
bucket, the Jobs API (50 launches in 4 s, 64 concurrent) and the controller process were not
the limit.

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

Results: all three passed on 2026-10-14 (experiments/resume-gate, experiments/packing,
experiments/e2e). The 1k load test passed the same day after two fixes
(experiments/load-1k/RESULTS.md): a drained worker did not always exit, and cost was
double-counted after a relaunch.

## 10. Unverified assumptions and risks

- Cross-machine session resume: PASSED 2026-10-14 for both harnesses
  (experiments/resume-gate/RESULTS.md). Claude needs only
  `projects/<enc-cwd>/<id>.jsonl` (plus `<id>/` if present); Codex needs only the
  rollout file at its dated relative path and rebuilds its SQLite index. Codex was
  verified locally with two fresh homes, not in a job (no API key here).
- Pi cross-machine resume: PASSED 2026-10-16 in two cpu-basic jobs, as root
  (experiments/resume-gate-pi/RESULTS.md). Packing on cpu-basic and compaction on long
  sessions are unmeasured (docs/pi.md).
- Multi-day job timeouts honoured in practice (24 h is within reports; longer
  is not needed).
- 1,000 concurrent HF Jobs under one PRO user (credits and spend limit are the
  real gates).
- Codex app-server per agent memory footprint; `codex exec` measured locally at
  about 263 MB RSS and 1.5 CPU-s per turn.
- Harness usage fields are cumulative per session (Claude `total_cost_usd`,
  `duration_api_ms`, `modelUsage`; Codex `turn.completed.usage`); the worker
  must difference them per turn. Jobs run as root, so Claude needs `IS_SANDBOX=1`
  for bypassPermissions, and Codex needs `--sandbox danger-full-access` in the
  container. Never ship the Codex OAuth `auth.json` to a job; use an API key.
- Prompt-cache re-writes after idle gaps over 1 h are the main hidden cost of
  long idle agents with large transcripts. This is true of any hosting model.
- Open models (PR A, 2026-10-16): Codex against a real vLLM or HF router Responses endpoint
  (verified only against a local fake server that streams Responses events); whether the HF
  router serves `/v1/responses` for open models; Claude Code against vLLM's Anthropic-compatible
  endpoint (and which background model it calls there, see `ANTHROPIC_SMALL_FAST_MODEL` /
  `ANTHROPIC_DEFAULT_HAIKU_MODEL`, unset by flock); `reasoning: off` sent as effort `none`;
  Codex behaviour for model ids it has no metadata for (context window, tool format).
- msgboard's 2,000 open-poll cap and roughly 300 req/s ingress will matter the
  day agents poll it themselves. Deferred by decision.
- The HF Spaces edge rate-limits traffic to the controller on more than the client IP. One
  client at about 20 req/s is blocked (1k run). Fifty workers at under 1.4 req/s each, from
  distinct egress IPs, were all blocked together once the aggregate reached about 90 to 120
  req/s (10k run), while this workspace's monitor was not. The key (per-Space aggregate or the
  jobs' IP range) and the threshold are not measured. Nothing above roughly 2,000 to 3,000
  active agents can run through the Spaces edge as it is.

## 11. Decision log

Condensed from the grilling session, 2026-10-13/14.

| # | Decision |
|---|---|
| 1 | flock is the lowest layer; swarmlab and msgboard are callers, unchanged in v1 |
| 2 | Coding harnesses first (Claude, Codex); API-loop agents later |
| 3 | No inbox routing in flock; prompt API plus per-agent event stream only |
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
| 23 | Repo `huggingface/flock`, Python, one package, three entry points |
| 24 | Milestone one: resume gate, packing measurement, 100 agents end to end |
| 25 | Open models via a swarmlab-compatible `providers:` block, optional `models:` aliases, `<provider>:<id>` model strings; claude only on `anthropic`, codex only on `openai_compat` (Responses API: Codex 0.162.1 dropped chat); price pinned into the stored spec at apply; controller-side preflight; `max_working` reported, enforcement deferred (2026-10-16); superseded by 27 for `max_working` |
| 26 | Harnesses get an allowlisted environment plus explicit additions, never the worker's `os.environ` (2026-10-16) |
| 27 | Review follow-up (2026-10-19): no `network_group`, workers are CPU-only and reach providers over HTTPS; preflight is best-effort (failed probe blocks, unprobed is `unverified`, `--strict` blocks it); a Hub 4xx on launch marks the type `launch_refused` until re-apply; `max_working` enforced at prompt admission for every turn start (API prompt, resume prompt; spawn's queued prompts), on by default, `FLOCK_MAX_WORKING_ENFORCE=0` as escape hatch (Carlos) |
| 28 | Codex enforces `max_turns` / `max_budget_usd` in the worker (2026-10-19): one token count = one model call, flock-priced; `turn/interrupt` before a call past a limit, turn ends `error` with `reason`, as Claude and Pi; app-server has no native per-turn limit |
| 29 | Pi as a third harness (2026-10-16): `pi --mode rpc` per burst, steer-mode prompts, `agent_settled` ends a turn, per-call usage, allow-listed env; the Pi resume gate runs in a job before real use |
| 30 | Spawn: children are agent rows with a parent, placed on their type's workers; `spawn:` policy apart from types; cascade removal; per-agent scoped tokens (§13) |
| 31 | `llm` as a fourth harness (2026-10-19): an in-process Chat Completions loop for small open models (one `bash` tool, JSONL session, compaction past `context_tokens` projected from the server's prompt_tokens (2026-10-20, run 3: chars/4 undercounted JSON 2x; context-length 400 compacts and retries once, else reason `context_overflow`), tool calls written as text accepted; `tool_choice` forcing with a `done` tool for multi-step turns (one action per prompt without it), after the LFM2.5 probe: constrained output 20/20 valid in ~2.5 s vs free-form 7/20 in ~25 s; one system message, first); `openai_compat` provider required; `extra` sent, `reasoning` ignored; limits checked before each call; preflight probes Chat Completions for `llm` types (Carlos) |
| 32 | Controller as an HF Job (job mode, 2026-10-20; supersedes 9 for large swarms, Space mode kept): the controller runs in a network group under the alias `controller`. Workers join the group and use `http://${HF_NETWORK_GROUP_PREFIX}controller:7860` directly, off the Spaces edge (429s at ~44 workers, pixel runs 1k and 2). Humans and the CLI go through the jobs proxy: the API on 7860 needs an HF token, with the flock token in `X-Flock-Token`; a read-only status page is public on 7861. SQLite and event files are synced to a bucket every 60 s and on SIGTERM, then restored at start. One namespace per controller; model jobs stay outside the group (Carlos) |
| 33 | Tools folder (2026-10-21, thread 3063: tools live outside flock, are read at runtime and change while the swarm runs): public documented functions in `tools/*.py` are tools (schema from type hints, injected `Context`); `flock launch` ships the folder as a tar.gz named by its content hash; workers fetch it at assign and from heartbeat responses and switch at the next turn; one `python -I` subprocess per call (60 s, 1 MB); `llm` offers them as function tools, CLI harnesses through `flock-tool` and a generated Tools section |
| 34 | Comms in the spec (2026-10-21, design note §10, threads 3063 and 3144): agents join the board under their flock name at their first prompt; the token stays in the worker (never in the agent env, workdir or pin); one activity long-poll per worker fans messages out as `comms_mention` / `comms_thread_reply` events; built-in `post` / `read` / `search` tools with rules enforced before the board call; search cut 1 is the board's own FTS5 search with query relaxation, no msgboard change; the kickoff is posted by `<swarm>-host`, never an agent (Carlos) |

## 12. Road to 10k

What the 1k and 10k runs say must change, in order. Item 1 gates everything else.

1. **Take the control plane off the Spaces edge limit, then measure the limit.** Done for the
   worker path by job mode (decision 32, [deploy-job.md](deploy-job.md)); the ramp below is
   still to do, now against the controller job itself. The options were, in
   order of how much they keep: a private Space with workers sending an HF token (needs the
   swarm token out of `Authorization`; whether authenticated traffic has a separate limit is
   unknown); the controller as an HF Job with `--expose` (its own ingress, $0.01/h); or another
   host. Then a controlled ramp: N cpu-basic jobs at a fixed rate each, to find the key and the
   threshold, written into §7.
2. **Liveness that survives a throttled ingress.** Do not cancel a `RUNNING` job only because
   its heartbeats stopped when many workers of a swarm fell silent together; cap cancels per
   pass; never relaunch into a block the controller can observe. Worker side: retry a failed
   heartbeat within seconds, honour `Retry-After`, back the command poll off on 429.
   Partly done after the pixel run 1k (29 of 42 workers declared dead): workers retry 429/5xx
   with backoff and jitter honouring `Retry-After`, and dead needs silence on every route
   (heartbeat, poll, events) for 180 s. Still open: a cap on cancels per pass.
3. **Fewer requests per agent-turn.** The 10k send phase was 120 req/s: prompters 50/s and the
   workers' immediate re-polls another 50/s. A batch `POST /prompts`, a swarm-level events
   stream, and command polls that batch for a few hundred ms would put the same round under
   20 req/s.
4. **Sync and restore per worker, not per agent**, then sync on turn end so a crash loses at
   most one turn instead of a worker's last three minutes (110 transcripts lost on one
   cancelled worker at 10k).
5. **A launch budget per swarm.** The crash path spent 64 launches in ten minutes with nothing
   wrong on the workers; a per-swarm cap on relaunches per window, and a crashloop rule that
   counts the swarm rather than the worker.
6. **Persistent controller state** and a **spend ledger** that survive redeploys, scale-down
   and remove.
7. Smaller: a 200-row heartbeat should not take 10 s (fewer statements, counters out of the
   hot path); the drain should not wait on serial unassign syncs; `max_working_per_worker`
   defers starts instead of warning; explicit command acks; prebuilt worker image; Codex in a
   job with an API key; the `asyncio` child-reaping warning; the kill and deadline-drain checks
   still need a run at 10k.

## 13. Spawn

Added 2026-10-16 (board thread, msg 388). User guide: [spawn.md](spawn.md).

A leader agent creates child agents of other types through the controller, with the `flock`
CLI inside its harness. A child is an ordinary agent row with `parent` set and no replica slot,
named `<parent>-<type>-<NNNN>` from a per-swarm counter that is never reused. Reconcile places
it on its own type's workers with the fill rule of §3 (a job carries one type's image, secrets
and flavor), so nothing about workers changes. Scaling `replicas` ignores spawned rows; they go
with their parent (cascade, children first), their type, the swarm, or an explicit
`DELETE /agents/{name}`.

What may spawn is run policy, kept apart from what a type is: `spawn.<type>` lists the child
types (a list with one `max`, or a map of per-type caps), and `depth`, the generations allowed
below the agent. A child's depth is `min(parent's - 1, its type's spawn.depth)`. `apply` rejects
cycles, so recursion ends by construction. Each spawnable type declares `workers: {initial,
max_extra}`, a cap on its worker slots that `apply` (replicas must fit), spawn admission and
placement all respect. `limits` caps the swarm
(`alive`, `spawned_total`, `budget_usd`). A refusal is a 409 with a `reason`.

Tokens: v1 handed the swarm's `FLOCK_TOKEN` to every harness through the inherited environment.
Now each agent gets `FLOCK_AGENT_TOKEN`, an HMAC of its name and a nonce under a key derived from
the controller token. The controller stores only a hash, re-derives the same value for every
assign, and deletes it with the agent. That token may act on the agent itself, its subtree, and
prompt its parent. The worker blanks `FLOCK_TOKEN` in the harness env; the allow-list from the
providers work removes the rest of the inherited environment.

Persistence: spawned rows are the first agents not derivable from the spec, so the controller
needs its SQLite to survive. It lives in `FLOCK_DATA` (`/data` in the controller image); a Space
with persistent storage keeps it across redeploys. No bucket mirror yet.

Defaults chosen where the thread left them open: `spawn.depth` 1; caps count existing children;
the list form needs `max`; `budget_usd` is checked at spawn time only and counts removed agents'
cost; replicas cannot be removed with `DELETE` (`not_spawned`); an agent cannot prompt or remove
itself, nor read its parent's status. `spawn.warm` (pre-created idle children) was dropped
(Carlos, 2026-10-16). Workers open only for agents that exist, so the first spawn of a type with
no agents pays the worker's boot time; pre-warming is a possible later feature.

## 14. Loops and the budget stop

Added 2026-10-21 (worlds design note §9 and §13). User guide: [loop.md](loop.md).

A type's `loop:` (`TypeSpec.loop`, `LoopSpec` in models.py) keeps its agents prompted: the
loop prompt on `every` (`turn_end` or a duration) while `while` holds, then one
`final` prompt per live agent. The controller's loop engine (`controller/loop.py`) ticks once a
second and sends every prompt through the bulk-prompt path with `state: idle, skip_pending:
true`, so loops never stack prompts and stay under `max_working` admission. `every: turn_end` on
an uncapped provider runs on the worker (`loop_start` / `loop_stop` commands; the worker
re-prompts when a turn completes), under `max_working` on the controller. Prompts are templates
the worker renders (`PromptCommand.loop`, `{round}`, `{spend}`, `{last_summary}`, ...).
`limits.budget_usd` becomes a stop, not only a spawn check: at `budget_usd` minus the final
reserves loops stop, pending prompts are dropped and final rounds run; at `budget_usd` every
agent is stopped and prompts are refused. World-backed wakes (`on: world.*`) are in the
model and refused at apply until a world exists.

### Run-4 additive contracts (2026-10-26)

`comms.extended` adds optional channel-aware tools and per-identity private-message
polling. Legacy specs retain their default tools and event payloads. The rule
fields `max_channels_per_agent` and `max_channels` bound channel creation (the
board count check is advisory across workers: with W workers creating at the same moment
the total can exceed `max_channels` by at most W-1). In extended mode a group mention
(`@all`, `@agents`, ...) counts as a mention only when a human posts it. See [comms.md](comms.md).

Worlds advertising `workspace-upload-v1` receive worker-captured clone snapshots
outside the model's arguments. Other World API v1 worlds keep their dispatch and
timeouts. See [world.md](world.md). Pi selects an endpoint per agent by the same
stable hash used by the llm harness; multi-endpoint Pi workers use separate agent
config directories so models.json cannot race between different endpoints.

Transport failures (`model_transport`) and turn timeouts retry without incrementing
the loop's consecutive error count, after 2 s doubling to 120 s with jitter; after 20 in a
row each further one counts as an ordinary error. Ordinary errors still stop after three failed
turns; intentional interruptions retain their existing behavior. Controller state
sync uploads changed event files incrementally after its first complete sync.
