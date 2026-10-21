# flock worlds: World API v1 and the primitives for task-driven swarms

Design note, 2026-10-21. Status: draft for Carlos's review, rev 2 (unified World API). No code yet. Board: design thread msg 3063-3154 (direction in 3142, 3149, 3154), search thread 3144-3153.
Scope: one HTTP API that any world speaks, so people know how to set up worlds for flock; plus what flock builds so runs like the pixel swarm (`experiments/pixel/`) and the Eliza swarm (`prj-agent-swarms/eliza-swarm/`) run at 5 → 500 agents.
Goal (Carlos, msg 3154): "unify the API for the worlds [...] have the hooks that enable many applications to run on top. At the end of the day we should just provide the load balancing and allow users to do anything they'd like with the agents."

## 1. Problem
Both experiments hand-rolled the task side outside flock:
- **Tools**: painters `urlretrieve` the canvas CLI in their first prompt (`run/painter.md`); Eliza's 13 function tools live in its own loop (`eliza-swarm/prompts.py::tools`).
- **Loop and comms**: `run/driver.py` re-prompts idle painters; leaders run 30 s shell loops (`run/leader.md` step 5); Eliza runs rounds with a 2 s debounce or 120 s wake (`swarm.py::main_loop`); `run/board.py` and `eliza-swarm/board.py` each join msgboard, poll, rate-limit posts (20 s gap, 32 refusals in eliza-v0-run1).
- **Artifacts, stops, observability**: Eliza's bucket workspace and `best/` promotion (`workspace.py`, `best.py`), caps and `guard-*.sh`; the pixel progress Space and `analyze.py`.

At hundreds of agents this breaks: a driver's bulk calls stacked 73,394 prompts (1k run); install + paint prompts folded into one turn, so 5 min in only 77 of 1,000 children had reached the canvas (run 3); each kit re-implements 429 backoff; nothing joins a tool call to the flock turn that made it (`FLOCK_TURN` is empty); every run builds its own viewer; and every new task needs a new kit, because there is no contract a world implements once.

## 2. Division of labour
- **flock** is the load balancer for agents: compute, placement, scaling, admission (`max_working`), identity, budgets, the loop that prompts agents, events, and the worker-side **gateway** that every service call passes through (secret, identity headers, rate budget, retries, provenance).
- A **world** is any HTTP service that speaks World API v1 (§3), in any language: task state, tools, scoring, claims, rule checks, its log, what artifacts mean. One world serves 5 or 500 agents.
- **comms** is a pluggable service (`kind: msgboard` today) behind the same gateway. swarmlab is one world provider behind an adapter (§5.4), not a frontend; flock never imports swarmlab.
- flock never interprets task state (it reads only the score metrics a `loop` condition names), never routes inboxes (decision 3), never judges results (score, verify and promotion checks are world routes) and never stores world state.

```
 ┌─ network group (operator / CLI ──► controller) ──────────────────────────────────────┐
 │ controller job ── loop ticks · budget stop · hooks out · bundle copy at end          │
 │   ▲ │ commands, heartbeats, events          hooks ▼ ▲ world token: prompt · events · │
 │   │ ▼                                             │ │ stop · spawn                   │
 │ worker (25-50 agents)                       world job (alias <world.name>)           │
 │   harness (llm | claude | codex | pi)         /manifest  /tools/<name>               │
 │     │ tool call / flock-tool                  [+ state score claims rounds events    │
 │     ▼                                          render checks hooks]                  │
 │   gateway 127.0.0.1:7870 ─ POST /tools/* ──►                                         │
 │     secret · X-Flock-Agent · rate · retry · req/turn ids                             │
 └────┬─────────────────────────────────────────────────────────────────────────────────┘
      ▼ via gateway: comms (msgboard) · artifacts (bucket, run-scoped)
```

## 3. World API v1
HTTP + JSON. Errors are `{"error": {"code", "message", "retry_after_s"?}}`; `429`/`503` carry `Retry-After`. Worlds ignore unknown fields and headers, flock ignores unknown capabilities. `world_api` is a major version: changes within v1 are additive, flock supports the current and previous major and refuses an unknown one at apply.

### 3.1 Required core
| route | caller | contract |
|---|---|---|
| `GET /manifest` | apply, workers, doctor | the manifest below; doubles as the health check (200 = ready) |
| `POST /tools/<name>` `{"args": {...}}` | agent via gateway | `{"ok": true, "result": {...}, "text"?: "one status line"}` or `{"ok": false, "error": {...}}`; in rounds mode may answer `{"pending": true, "id"}` |

A tool error is a result the agent reads, never a failed turn; `text` is what small models see first. Manifest schema (condensed; `parameters` is JSON Schema, strict-normalised by flock as swarmlab's `ToolSchema.normalized` does):
```json
{"type": "object", "required": ["world_api", "name", "version", "tools"], "properties": {
  "world_api": {"const": "v1"}, "name": {"pattern": "^[a-z][a-z0-9-]{0,31}$"}, "version": {"type": "string"},
  "capabilities": {"type": "array", "uniqueItems": true, "items": {"enum":
    ["state", "score", "claims", "rounds", "events", "render", "checks", "hooks", "controller", "rate"]}},
  "tools": {"type": "array", "items": {"required": ["name", "description", "parameters"], "properties": {
    "name": {}, "description": {}, "parameters": {"type": "object"}, "examples": {"type": "array"}, "timeout_s": {}}}},
  "score": {"properties": {"metrics": {"type": "object"}, "agent_visible": {"default": false}}},
  "rounds": {"properties": {"timeout_s": {}}}, "checks": {"properties": {"write": {}, "promote": {}}},
  "hooks": {"properties": {"path": {"type": "string"}, "events": {"type": "array"}}},
  "controller": {"properties": {"scopes": {"items": {"enum": ["prompt", "events", "agents", "stop", "spawn"]}}}},
  "rate": {"properties": {"per_agent": {"type": "string"}, "burst": {"type": "integer"}, "global": {"type": "string"}}}}}
```

### 3.2 Optional capabilities
flock calls only routes the manifest declares. `flock apply` checks the spec against the manifest and fails naming the missing capability, so nothing degrades silently at run time.

| capability | routes | flock uses it for | without it |
|---|---|---|---|
| `state` | `GET /state?agent=` | built-in `state` tool; forced first step of a round | agents see only tool results |
| `score` | `GET /score` → `{metrics..., done?}` | `loop.while: score_*`, `done` stop, viewer, bundle | score conditions rejected |
| `claims` | `POST /claims {key, ttl_s}`, `DELETE /claims/{key}`, `GET /claims?prefix=`; 409 + owner when held | generic lease tools a spec may expose (`world:lease`, `world:unlease`, `world:leases`), lease timeline | a world may still lease inside its own tools |
| `rounds` | `POST /rounds/{n}/close` → outcomes per agent, idempotent per `n` | `loop.every: rounds` | `every: rounds` rejected |
| `events` | `GET /events?after=<seq>&limit=` → `{seq, ts, kind, agent, req, turn, data}`; `kind` reuses swarmlab's names (`paint`, `claim`, `violation`) so its analysis transfers | `loop.on: world.event:<kind>`, bundle, join on `req` | `on: world.*` rejected; gateway call log only |
| `render` | `GET /render?t=` → PNG or JSON frame | viewer replay | timelines only |
| `checks` | `POST /checks/write {agent, path, sha}`, `POST /checks/promote {agent, path}` → `{ok, violations}` | artifact rule checks, `best/` promotion (§9) | writes unchecked; `best.check` rejected |
| `hooks`, `controller` | `POST <hooks.path>`; `controller` only requests a world token | lifecycle webhooks (§3.3); world-driven prompting (§3.4) | none sent; the world is passive |
| `rate` | none; manifest hint | default gateway budgets (§11) | flock default 2/s per agent |

### 3.3 Hooks, flock → world
The world subscribes in `hooks.events`; the controller (one sender per swarm) POSTs batches of up to 500 events or 1 s: `{"swarm", "run", "events": [{"seq", "ts", "event", "data"}]}`. Delivery (proposed, open question 5): at-least-once, ordered per swarm, `seq` for dedupe, 2xx acks, 5 s timeout, retries with backoff for 5 min, then `world_hook_dropped` in the swarm events. Hooks never block flock.

| event | data | when |
|---|---|---|
| `agent_assigned` | `{agent, type, parent, worker}` | agent placed (apply, scale, spawn), before its first prompt |
| `agent_removed` | `{agent, reason}` | scale-down, cascade removal, dead past the relaunch budget |
| `turn_ended` | `{agent, turn_id, status, prompt_source, tool_calls, cost_usd}` | every turn end (from worker events) |
| `loop_tick` | `{type, tick, round?, spend_usd}` | each controller loop tick |
| `run_stopping` / `run_stopped` | `{reason: budget\|until\|done\|operator\|world, grace_s}` / `{reason, totals}` | before the final round; after agents stopped, before the bundle copy |

### 3.4 Hooks, world → flock
If the manifest declares `controller`, `flock apply` mints a **world token**: a third principal next to the controller token and agent tokens (`agent_may` in `controller/api.py`), limited to that swarm and the requested scopes, sent as `X-Flock-Token`. A `run:` world gets `FLOCK_WORLD_TOKEN` and `FLOCK_CONTROLLER_URL` in its env; for a `url:` world the operator reads it with `flock world-token <swarm>`.

| scope | routes | notes |
|---|---|---|
| `prompt` | `POST /swarms/{s}/prompts`, `POST /agents/{name}/prompt` | admitted under `max_working`; `skip_pending` on by default |
| `events`, `agents` | `GET /agents/{name}/events`, swarm stream (step c), `GET /swarms/{s}/agents` | read only |
| `stop` | `POST /swarms/{s}/stop {reason}` (new) | runs `loop.final`, then stop, like a budget stop |
| `spawn` | `POST /agents/{name}/spawn` | only where the spec's `spawn:` policy allows; counts against `limits` |

A world token never applies, scales or removes swarms, changes types, models or limits, or reads other swarms. This makes world-driven loops possible: omit `loop:`, subscribe to `turn_ended`, prompt each agent with its next observation (a game master, a debate judge, swarmlab's runner inverted). flock still admits, budgets and rate-limits every prompt.

### 3.5 Auth and provenance
- One shared secret per world; no world-minted per-agent tokens and no `/agents/register` (canvas mode B is dropped). `run:` worlds: `flock apply` generates it and hands it to the world job (`FLOCK_WORLD_SECRET`) and every worker's gateway; `url:` worlds: `world.secret_env`.
- Only the gateway holds it; the harness env never does (allowlisted env, decision 26), so an agent with `bash` cannot forge a call. Agents reach the world only through the gateway: the world job sits in the network group and is not public unless `expose: true` (read-only routes).
- Gateway → world headers: `X-Flock-World-Secret`, `X-Flock-Agent`, `X-Flock-Type`, `X-Flock-Swarm`, `X-Request-Id` (one per tool call), `X-Flock-Turn` (the flock turn id; fixes the empty `FLOCK_TURN`). Controller → world calls (score, rounds, events, hooks) carry the secret and no `X-Flock-Agent`. The secret has its own header because through the HF Jobs proxy `Authorization` carries the HF token (as with `X-Flock-Token` and the canvas `X-Canvas-Token`). A world trusts `X-Flock-Agent` iff the secret matches and logs `req` and `turn` on every event. `FLOCK_AGENT_TOKEN` never leaves flock.

## 4. Where worlds live
| what | where |
|---|---|
| protocol spec, conformance test | this note (then `docs/world-api.md`); `flock doctor --world <url>` (§5.3) |
| Python base | `flock.world` (`src/flock/world/`): FastAPI app from typed tools, secret check, claims with wall-clock TTL, event log with `req`/`turn`, hook receiver with dedupe, world-token client |
| reference worlds | `worlds/canvas` (moved from `experiments/pixel/canvas`), `worlds/eliza` (workspace + rule check + sandbox); they are the proof runs, so they ship with flock |
| other worlds | swarmlab adapter: `worlds/swarmlab/` (`swarmlab.serve`), or in swarmlab if its owner wants it there; experiment worlds: `experiments/<name>/world/`, a `flock-worlds` repo if they pile up |

At runtime a world is one HF Job in the swarm's network group, next to the controller and workers (as the canvas ran in runs 1-3):
```yaml
world:
  name: canvas                     # network alias, default: the manifest name; tools appear as world:<tool>
  run: {image: ghcr.io/huggingface/flock-worlds:canvas-0.4, command: [python, -m, worlds.canvas, --port, "8765"],
        flavor: cpu-upgrade, port: 8765, env: {CANVAS_TARGET: mona.png}, expose: false}   # expose: public read-only routes
  # url: https://<job>--8765.hf.jobs   # instead of run: bind an existing world (needs secret_env)
  secret_env: WORLD_SECRET         # url: worlds only; a run: world gets a generated one
  manifest: auto                   # or a pinned file; drift from the live manifest fails apply
```
`flock apply` starts the world job first, waits for `GET /manifest` 200 (10 min), validates the spec against it, then places agents. A crashed `run:` world is relaunched under the same alias; meanwhile tool calls get `503` and the gateway backs off. The world owns its state (disk + its own bucket mirror). At run end and on `flock remove` the controller copies `/events` (all pages) and `/score` into `<run>/world/`; `flock remove` then cancels the job. One world per swarm in v1.

## 5. Write your own world
### 5.1 A minimal world: `worlds/digits/app.py`
1,024 cells with hidden target digits; agents claim a run of 32, read its targets, write digits.
```python
import random
from flock.world import Agent, ToolError, World

N, RUN = 1024, 32
TARGET = random.Random(7).choices(range(10), k=N)
cells: list[int | None] = [None] * N
world = World("digits", version="0.1.0", rate={"per_agent": "2/s", "burst": 5, "global": "500/s"})

@world.tool
def empty(agent: Agent, limit: int = 8) -> dict:
    """Run numbers that still have empty cells."""
    runs = sorted({i // RUN for i, c in enumerate(cells) if c is None})
    return {"runs": runs[:limit], "text": f"{len(runs)} runs open"}

@world.tool
def claim(agent: Agent, run: int) -> dict:
    """Lease run `run` for 120 s and get its 32 target digits. Fails with the owner if held."""
    world.claims.acquire(agent, f"run:{run}", ttl_s=120)        # held -> ToolError("held", owner=...)
    return {"targets": TARGET[run * RUN:(run + 1) * RUN]}

@world.tool
def write(agent: Agent, run: int, digits: list[int]) -> dict:
    """Write 32 digits into run `run`. You must hold its claim."""
    world.claims.require(agent, f"run:{run}")
    if len(digits) != RUN or any(not 0 <= d <= 9 for d in digits):
        raise ToolError("bad_digits", f"need exactly {RUN} digits 0-9")
    cells[run * RUN:(run + 1) * RUN] = digits
    world.claims.release(agent, f"run:{run}")
    world.emit("write", agent, run=run)                          # req and turn attached from the headers
    return {"text": f"run {run} written"}

@world.state
def state(agent: Agent) -> dict:
    return {"my_claims": world.claims.held_by(agent)}

@world.score                                                     # operator-only unless agent_visible=True
def score() -> dict:
    ok = sum(c == t for c, t in zip(cells, TARGET))
    return {"correct_pct": 100 * ok / N, "done": None not in cells}

@world.on("agent_removed")                                       # subscribes to the hook
def drop(agent: Agent, reason: str) -> None:
    world.claims.release_all(agent)

app = world.app()   # /manifest /tools/* /state /score /claims /events /hooks; secret from FLOCK_WORLD_SECRET
```

### 5.2 Its manifest (abridged) and the spec that binds it
```json
{"world_api": "v1", "name": "digits", "version": "0.1.0", "capabilities": ["claims", "events", "hooks", "rate", "score", "state"],
 "tools": [{"name": "empty", "description": "Run numbers that still have empty cells.", "parameters": {"type": "object",
            "properties": {"limit": {"type": "integer", "default": 8}}, "required": [], "additionalProperties": false}}, {"name": "claim"}, {"name": "write"}],
 "score": {"metrics": {"correct_pct": {"type": "number"}, "done": {"type": "boolean"}}, "agent_visible": false},
 "hooks": {"path": "/hooks", "events": ["agent_removed"]}, "rate": {"per_agent": "2/s", "burst": 5, "global": "500/s"}}
```
```yaml
name: digits-50
world:
  run: {image: ghcr.io/huggingface/flock-worlds:base, flavor: cpu-basic, port: 8765,
        command: [uvicorn, worlds.digits.app:app, --host, 0.0.0.0, --port, "8765"]}
types:
  filler:
    harness: llm
    model: "lfm:LiquidAI/LFM2.5-2.6B"
    replicas: 50
    tools: {expose: [world:empty, world:claim, world:write, done]}
    loop: {every: turn_end, while: {score_lt: {correct_pct: 100}, until: 1h},
           prompt: "Call empty, claim one run, write its digits. Repeat until no runs are open, then call done."}
limits: {budget_usd: 3}
```

### 5.3 `flock doctor --world <url>`
Run it against a scratch instance (it calls tools): `--secret-env WORLD_SECRET`, optional `--spec spec.yaml`, `--load 200/s --for 60s`. Exit 1 on any failure, one line per check.
- core and auth: manifest validates, `world_api` supported, tool schemas strict-normalisable, each tool's `examples` return the envelope; no secret → 401; with it, `X-Flock-Agent: doctor-a` / `doctor-b` are attributed to those names;
- each declared capability: score metrics match the manifest; events `seq` monotonic with `req` echoed; a second agent's claim gets 409 with the owner; `rounds/{n}/close` idempotent; a repeated hook `seq` is acked and ignored; `render` returns PNG or JSON;
- `--spec`: every capability the spec needs is declared; `--load`: p50/p99 latency and error rate at that rate.

### 5.4 Existing worlds on the API
| | canvas (`worlds/canvas`) | Eliza workspace (`worlds/eliza`) | swarmlab `ColoringGrid` (`swarmlab.serve`) |
|---|---|---|---|
| tools | aliases of `/claim`, `/paint`, `/release`, `/claim/region*`; the `canvas` CLI becomes the pack for CLI harnesses | `run_bot` (sandbox), `*_task`, `submit_final`; files via the artifacts pack (§9) | `@tool` methods via `validate` |
| manifest | new | new | `World.tool_schemas()` (already strict JSON schema) |
| `state` / `score` | `/status`, `/region` / `/score` | task board + own files / probe replay of `analyze.py` | `observe(agent)` / `score()`, `verify()` |
| `claims` | pixel claims + region leases | task claims | `Registry` (`Advisory`/`Enforced`) |
| `rounds` | n/a | optional | `round_end` phase-commit: `commit_world` in seeded order, `end_round` |
| `events` / `render` | alias of `/log?after=` / `/replay/frame?t=` | writes, violations / `best/` history diffs | `swarmlab/events.py` / `world/render.py::grid_to_png` |
| `checks` | n/a | write: `checker.py`; promote: rule check + 3-message smoke | n/a |
| auth | mode A kept, mode B dropped | shared secret | shared secret in the adapter |

- swarmlab, reused as is: paint accounting (`useful/duplicate/wrong/overwritten`), validate/error vocabulary, TTL claims, metrics folded over the log, `swarmlab replay`, the scripted painters (`RowMajorPainter`, `QueuePainter`) over HTTP as zero-LLM load generators. Replaced by flock: the in-process runner, `LLMAgent`, `Gate`/budget ledger, `swarmlab job run`. Its board tools stay world tools when an experiment needs swarmlab's topology control, else comms (§8) replaces them.
- Limits before 500 agents: `Registry` TTLs count rounds (`ttl_rounds`), async needs wall-clock TTLs; `ColoringGrid.score()` rescans every cell; one cell per `paint`; its req/s is unmeasured (the canvas did 800 req/s in one process), so `flock doctor --load` with scripted painters comes first.

## 6. Primitive 1: tool packs
A pack is installed into the worker once per type at boot, outside the workdir (a bucket restore never touches it, exec bits survive), plus a JSON manifest of function tools in the World API tool shape plus `exec: {cmd: [...], stdin: json}` (stdout = result, exit ≠ 0 = tool error), `rate` (a §11 budget) and `timeout_s`. World tools need no pack: they come from `GET /manifest`. Spec names are `world:<tool>` and `comms:<tool>` for services, bare for packs and built-ins; the model sees the bare name (function names allow no `:`), `world_<tool>` on a collision.
```yaml
types:
  dev:
    harness: llm
    tools:
      packs: [{name: probe, install: {pip: ["git+https://github.com/huggingface/flock-packs#subdirectory=eliza-probe"]}, manifest: eliza_probe/manifest.json}]
      builtin: [read_file, write_file, done]                          # llm harness natives
      expose: [ws_write, world:run_bot, probe, comms:search, done]    # what the model sees
```
- `llm`: every `expose`d tool is a Chat Completions function tool next to `bash`; `_run_tool` sends non-bash calls to the gateway. Natives `read_file {path, start_line?, end_line?}`, `write_file {path, content}`, confined to the workdir, no shell quoting (the smoke lost 2 of 3 painters to `\'`).
- `claude`, `codex`, `pi`: `flock-tool <name> --json '<args>'` on `PATH`, and a generated "Tools" section (name, description, one example) in `CLAUDE.md`/`AGENTS.md`; MCP is open question 3.
- Small models: `tool_choice: required` + `done` (decision 31), `expose` ≤ 6 tools, one example per tool in the preamble, `text` first. The LFM probe's 20/20 valid calls were with a named, forced tool. Changes: `worker/tools.py` (install, venv per pack, validate, `pack_installed` or fail the type), `worker/gateway.py`, `adapters/llm.py`, `agents.py` (`PATH`, `FLOCK_GATEWAY`); `ToolsSpec` on `TypeSpec`.

## 7. Primitive 3: loop and rounds
```yaml
types:
  painter:
    loop: {every: turn_end, first: "Call status once, then done.", prompt: "Claim and paint. Repeat until the result says you are done, then call done.",
           while: {score_lt: {coverage_pct: 100}, until: 2h, max_rounds: 400, budget_usd: 12}}
  dev:
    loop:
      prompt_file: prompts/round.md              # rendered with {round}, {elapsed}, {spend}
      every: 120s
      on: [comms.mention, comms.thread, world.event:violation]   # wake early, 2 s debounce
      while: {until: 2h, budget_usd: 14.6}
      final: {prompt_file: prompts/final.md, reserve_usd: 0.4, add_tools: [submit_final, best_vote]}
```
- The controller runs every loop: each tick evaluates `while` (score from `GET /score`, spend from its ledger) and sends one `POST /swarms/{s}/prompts {type, state: idle, skip_pending: true}`, so no agent stacks loop prompts and every prompt is admitted under `max_working`.
- Capability needs, checked at apply: `score_*` needs `score`, `every: rounds` needs `rounds`, `on: world.event:*` needs `events`. A world-driven loop (§3.4) needs no `loop:` at all.
- `every` is `turn_end`, a duration (`30s`) or `rounds`. `turn_end` removes run 1's ~17 % idle gap: the worker re-sends the loop prompt when a turn ends `completed`, until the controller pushes `loop_stop` (a new command); 3 errors in a row end it for that agent (open question 2). `first` is delivered alone and the loop prompt only after its turn ended, so run 3's install fold cannot recur.
- Rounds: prompt all live agents of the type, wait for every `turn_ended` or `rounds.timeout_s`, `POST /rounds/{n}/close`, next round; agents read their observation with `state` (forced first step).
- `on:`: one controller poller per swarm tails `GET /events` and the comms activity feed, maps an event to the agents it names, debounces 2 s, queues the loop prompt (`skip_pending`). This is Eliza's "new thread message or 120 s". Spawned children inherit their type's loop; run 3's leader `sleep 30` rounds become `loop` on the child type plus `scope: {parent: <leader>}`.

## 8. Primitive 4: comms
```yaml
comms:
  kind: msgboard
  url: https://lvwerra-msgboard-demo.hf.space
  invite_env: BOARD_INVITE         # multi-use agent invite scoped to the channel
  channel: prj-eliza-v1            # resolved to an id once by the controller
  rules: {min_post_gap: 20s, max_posts: 80, max_len: 500, threads: per_topic, no_group_mentions: true}
  protocols: [one-thread-per-topic, lead-agree]   # prompt snippets appended to the instructions
  wake: [mention, thread]          # what feeds loop.on
  search: index                    # board (cut 1) | index (cut 2)
```
- The worker joins at an agent's first prompt (not at assign, so 500 agents do not join in one burst) under its flock name, generalising `worker/identity.py`. The token stays in the gateway outside the synced workdir; a relaunch re-joins by name.
- `comms:post {text, thread: new|<root id>, reply_to?}` and `comms:read {n?, thread?}`; the gateway enforces `rules` and answers "refused: wait N s" without calling the board (Eliza: 32 refusals, ds-3 retried 6 times in 15 s).
- One activity poller per worker, not one long-poll per agent: msgboard caps open polls at 2,000 and ingress near 300 req/s. `@agents` reaches every agent on the board, so `no_group_mentions` refuses it and `wake` matches only direct mentions and the agent's own threads. Protocol snippets in `flock/prompts/protocols/*.md`: `one-thread-per-topic` (msg 2962), `lead-agree` (4/8 → 6/6), `help-offer`, `band-lease` (run 3 leaders).

**Search** (msg 3144-3153), no msgboard change: `comms:search {query, thread?, from?, since?, limit?}`, channel-scoped, messages only, returns id, thread, author, time, snippet.
- Cut 1, `search: board` (~1 day): the board's `GET /v1/search` with `in:#<channel>` and `sort=relevance` (FTS5 `bm25()` underneath); the tool relaxes the query (all words, then drop words until something matches, prefixes). Still AND-only and unstemmed underneath; every call spends board rate budget.
- Cut 2, `search: index` (2-3 days): the worker's poller feeds a per-worker SQLite FTS5 index of the channel (`porter unicode61`, BM25, AND/OR, filters by thread, author, time, mention), board API as cold-start fallback. Zero board load per query, one index per run, embeddings can be a later column.

## 9. Primitive 5: shared artifacts
Spec: `artifacts: {bucket: "hf://buckets/cmpatino/flock-runs", layout: run, best: {file: bot.py, check: world}}` (`check: world` runs `/checks/promote`).
```
<bucket>/<swarm>/<run>/agents/<name>/<stem>_<UTC ts>.<ext>   each write a new version, own dir only
                      /shared/<file>  manifest.jsonl          any agent, append-only names
                      /best/<file> manifest.json history.jsonl
                      /world/  events.jsonl.gz score.jsonl     copied by the controller at the end
```
- Built-in pack served by the gateway: `ws_list`, `ws_read`, `ws_write`, `ws_share {path, description}`, `find`, `promote_best {path, reason}`, `best_info`. Writes go to the agent's own dir; with `checks.write` the gateway calls `/checks/write` and returns violations as the tool result.
- `shared/manifest.jsonl`: one line per shared file `{path, author, description (one line), sha, from, turn_id, req, ts}`, `from` being the source path (provenance). `find {query, author?, since?, limit?}` ranks it with BM25 over path + description (the FTS5 code of comms cut 2); embeddings later if names and descriptions are not enough.
- Promotion has one writer, the controller (`POST /swarms/{s}/artifacts/promote`, open question 4): it runs `/checks/promote` (Eliza: rule check + 3-message smoke), writes `best/` with `{path_promoted, agent, ts, reason, replaced, sha}`, appends `history.jsonl`. The Eliza chat Space keeps reading via `best.py`.
- Every write, share and promotion is a flock event `artifact_written {path, sha, bytes, reason, turn_id, req}`. `loop.final` stops the loop at `budget_usd - reserve_usd` or `until`, adds `submit_final` and `best_vote`, sends one final prompt, then stops the agents.

## 10. Primitive 6: run observability
A per-run viewer generalising the pixel progress Space, built from a bundle the controller writes to `<run>/viewer/` every 60 s and at the end:

| file | from | content |
|---|---|---|
| `agents.jsonl` | flock events | per turn: agent, type, prompt source (loop, comms, world, api), tool calls (name, args digest, status, ms, `req`), violations, tokens, cost |
| `score.jsonl`, `world.events.jsonl.gz` | `GET /score` per tick, `GET /events` | metric time series; the task log, joined on `req` |
| `transcript.jsonl`, `run.json` | comms, controller | channel messages with thread and mentions; spec (secrets removed), world manifest, pinned prices, totals, stop reason |

Pages: summary (score, cost, agents by state vs time), per-agent timeline (each tool call with its world outcome, posts, violations), transcript, and replay from the world's `render` (canvas frames, `grid_to_png`, `best/` diffs). A page whose capability is missing is hidden.

## 11. Hardening
- **Swarm budget stop.** `limits.budget_usd` becomes a hard stop (today checked only at spawn). At `budget_usd - final.reserve_usd` the controller stops loops, drops pending prompts, sends `run_stopping` and runs the final round; at `budget_usd` it stops every agent. Overshoot is bounded by `max_working × max_budget_usd`, so tight caps need per-call booking (llm, Codex, Pi; Claude books at turn end).
- **Per-service rate budgets** in the gateway (token bucket per agent and per worker): `services: {world: {per_agent: 2/s, burst: 5, global: 800/s}, board: {per_agent: 1/20s, global: 20/s}}`, defaulting to the manifest's `rate`; the controller divides `global` by live workers on heartbeat. World-token calls get their own budget (20 req/s).
- **Guards from the runs.** A 429 from any service is backoff with `Retry-After`, never a dead agent or a failed turn (1k run: 29 of 42 workers killed); `skip_pending` on every loop, event and world prompt (73,394 stacked prompts); `first` delivered alone (run 3); packs outside the workdir, `.flock-modes.json` for workdir files (#9); context bounded by server tokens (#7).

## 12. Migration in 3 steps
| step | builds | proof run | effort | deleted from the experiments |
|---|---|---|---|---|
| a | tool packs, natives, gateway (rate, retry, events), `loop` (`turn_end`, `final`, `on: comms`), `comms` + search cut 1 | Eliza on flock, 5 agents (eliza-v0-run1 conditions), then 50 | 6-7 days | `swarm.py` loop, caps, join, rounds (660 lines), `llm.py`, `board.py`; pixel `driver.py` re-prompting, `leader.md` step 5 |
| b | World API core + capabilities, `flock.world`, `flock doctor`, `world.run`/`url`, shared-secret auth, hooks both ways + world token, budget stop, artifacts + `find` + promote; `worlds/canvas`, `worlds/eliza`, `worlds/swarmlab` | `digits` and canvas under `doctor --load`, then coloring at 500 `llm` agents (scripted load first, then LFM) | 9-11 days | CLI download prompt, `/agents/register`, `workspace.py` bucket code, `best.py` writer side, `experiments/pixel/canvas` |
| c | run viewer + bundle, swarm events stream, global rate budgets, search cut 2 | re-run a and b with the viewer live | 6-8 days | progress Space pixel code, `analyze.py` timelines, `guard-*.sh` |

Effort changes: 21-26 days, up from rev 1's 15-19. The increase is the unified API surface in step b (`flock.world` ~1.5 d, `doctor` ~1 d, `world.run` launch and bundle copy ~1 d, hooks + world token ~1.5 d, minus ~0.5 d for dropping minted tokens), search (cut 1 ~1 d in a, cut 2 2-3 d in c) and `find` (~0.5 d). Order follows msg 3064 (1 → 3 → 2 → 4 → 5 → 6); every new spec field defaults off.

## 13. Open questions
1. **Score visibility**: `score.agent_visible: false` by default, so flock exposes no score tool unless the world opts in. flock does not filter tool results, so a world that returns the score on every call (the canvas does) leaks it anyway: accept that as the world's choice?
2. **Loop trigger**: worker-side `every: turn_end` (no idle gap; one new command, `loop_stop`) or controller ticks only? A world can now re-prompt at turn end itself via `turn_ended` + world token, at a few seconds more latency per turn.
3. **CLI harnesses**: `flock-tool` + instructions only, or also an MCP server generated from the manifests for Claude, Codex and Pi?
4. **`best/` writer**: the controller as single writer using the world's `/checks/promote` (proposed), or the world (needs bucket write access in the world job)?
5. **Hook delivery**: at-least-once, batched, never blocking (proposed), or should some hooks be synchronous, e.g. `agent_assigned` waiting until the world has set the agent up, or a world able to veto a spawn?
6. **World token scopes**: is `spawn` in v1, and which parent and limits does a world-initiated spawn count against?
7. **Multiple worlds per swarm**: v1 has one. Later `worlds: {name: ...}` with tools prefixed by world name; does a planned run need two before then?
8. **Versioning**: a relaunched world whose manifest drifted (new version, changed tool schemas): refuse, or accept when the change is additive?
