# swarmlab worlds + flock agents: primitives for task-driven swarms at hundreds of agents

Design note, 2026-10-21. Status: draft for Carlos's review (board msg 3064, 3074). No code yet.
Scope: what flock must build so runs like the pixel swarm (`experiments/pixel/`) and the Eliza swarm
(`prj-agent-swarms/eliza-swarm/`) run on flock at 5 → 500 agents, reusing swarmlab's world machinery.

## 1. Problem

Both experiments hand-rolled the task side outside flock:
- **Tools**: painters `urlretrieve` the canvas CLI in their first prompt (`run/painter.md`); Eliza's 13 function tools live in its own loop (`eliza-swarm/prompts.py::tools`, `swarm.py::Agent.tool`).
- **Loop**: `run/driver.py` re-prompts idle painters; leaders run 30 s shell loops (`run/leader.md` step 5); Eliza runs rounds with a 2 s debounce or 120 s wake (`swarm.py::main_loop`).
- **Comms**: `run/board.py` and `eliza-swarm/board.py` each join msgboard, poll, rate-limit posts (20 s gap, 32 refusals in eliza-v0-run1).
- **Artifacts**: Eliza's bucket workspace, `best/` slot and promotion (`workspace.py`, `best.py`); the canvas paint log copied to a bucket by `job.sh`.
- **Stops and observability**: Eliza caps + `guard-*.sh` (spend, 429s, fast posts); the pixel progress Space and `analyze.py` timelines.

At hundreds of agents this breaks: a driver's bulk calls stacked 73,394 prompts (1k run); install + paint prompts folded into one turn, so 5 min in only 77 of 1,000 children had reached the canvas (run 3); each kit re-implements 429 backoff and rate rules; nothing joins a tool call to the flock turn that made it (`FLOCK_TURN` is empty, pixel run kit); every run builds its own viewer.

## 2. Division of labour

- **flock** owns agents, compute, scaling, admission (`max_working`), identity, budgets, the loop that prompts agents, events, and the worker-side **gateway** through which every tool call to a service passes (auth, rate budget, retries, provenance).
- A **world** owns task state, scoring, claims/leases, rule checks, the task's log and its artifacts' meaning. One world serves 5 or 500 agents; flock never interprets task state beyond the numbers a `loop` condition reads.
- **comms** is a pluggable service (`kind: msgboard` today), reached through the same gateway.

```
 operator / CLI ──► controller (job, network group) ── loop ticks, budget stop, swarm events
                        │ commands / heartbeats / events
              ┌─────────▼──────────── worker (cpu-basic, 25-50 agents) ─────────────┐
              │ harness (llm | claude | codex | pi) ── tool call / `flock-tool` CLI   │
              │        │                                                              │
              │   gateway 127.0.0.1:7870 ─ auth inject · rate budget · retry · events │
              └────────┼───────────────┬──────────────────┬───────────────────────────┘
                       ▼               ▼                  ▼
                 world (HTTP)    comms (msgboard)   artifacts (bucket, run-scoped)
             tools·claims·score   join · post · read    agents/<name>/ · shared/ · best/
             ·rule checks·log
```

## 3. Primitive 1: tool packs

A pack is something installed into the worker once per type at boot, outside the workdir (so a
bucket restore never touches it and exec bits survive), plus a JSON manifest of function tools.

```yaml
types:
  dev:
    harness: llm
    tools:
      packs:
        - name: eliza-ws
          install: {pip: ["git+https://github.com/huggingface/flock-packs#subdirectory=eliza"]}
          manifest: eliza_ws/manifest.json        # inside the installed package
        - name: canvas
          install: {script: "python3 -m pip install canvas-cli==0.3"}
      builtin: [read_file, write_file, done]       # llm harness natives
      expose: [ws_list, ws_read, ws_write, run_bot, read_file, done]   # what the model sees
```

Manifest entry (one per tool; `exec` is how the gateway runs it):
```json
{"name": "ws_write", "description": "Save a new timestamped version of a file into YOUR directory.",
 "parameters": {"type": "object", "properties": {"filename": {"type": "string"},
   "content": {"type": "string"}, "reason": {"type": "string"}},
   "required": ["filename", "content", "reason"], "additionalProperties": false},
 "exec": {"cmd": ["python3", "-m", "eliza_ws", "write"], "stdin": "json"},
 "examples": [{"filename": "bot.py", "content": "class Bot: ...", "reason": "first version"}],
 "rate": "eliza-ws", "timeout_s": 30}
```
`exec` is either a command (args as JSON on stdin, stdout = result, exit code ≠ 0 = tool error) or
`{"world": "<tool>"}` / `{"comms": "post"}` for the services of §4 and §6. Schemas are strict-normalised as swarmlab's `ToolSchema.normalized` does.

How each harness sees a pack:
- `llm`: every `expose`d tool becomes a Chat Completions function tool next to `bash`; the adapter's `_run_tool` sends non-bash calls to the gateway. Natives: `read_file {path, start_line?, end_line?}` and `write_file {path, content}`, confined to the workdir, no shell quoting (the smoke lost 2 of 3 painters to `\'` in a one-liner).
- `claude`, `codex`, `pi`: the gateway's CLI `flock-tool <name> --json '<args>'` (or `- < args.json`) is on `PATH`; the worker appends a generated "Tools" section (name, one-line description, one example) to `CLAUDE.md`/`AGENTS.md`. An MCP server generated from the same manifest is an option (open question 4).
- Small models: `tool_choice: required` + `done` (decision 31), `expose` kept to ≤6 tools, one example call per tool rendered into the preamble, results as one short status line then JSON. The LFM probe's 20/20 valid calls were with a named, forced tool.

Changes in `src/flock/worker`: `tools.py` (install packs per type into `/opt/flock/packs/<type>/<pack>`, a venv per pack, load and validate manifests, emit `pack_installed` or fail the type with `reason`); `gateway.py` (§9); `adapters/llm.py` (tool list from manifests, natives, dispatch); `agents.py` (Tools section, `PATH`, `FLOCK_GATEWAY` in the harness env). `models.py`: `ToolsSpec` on `TypeSpec`.

## 4. Primitive 2: world binding

```yaml
world:
  url: http://${HF_NETWORK_GROUP_PREFIX}canvas:8765   # in the group, or https://<job>--8765.hf.jobs
  admin_token_env: CANVAS_ADMIN_TOKEN    # controller/worker secret; agents never see it
  auth: minted                           # minted (default) | shared | none
  manifest: auto                         # GET /manifest at apply; or a pinned file
  expose: [claim, paint, done]           # per type, under types.<t>.tools.expose
```

The **world protocol** flock expects (HTTP + JSON; errors `{"error": {"code", "message"}}`):

| route | who | what |
|---|---|---|
| `GET /manifest` | apply, workers | `{name, version, modes: [async, rounds], tools: [...schemas], score: {metrics}, rate: {per_agent, burst, global}}` |
| `POST /agents/register {name}` | worker, admin token | world-minted agent token (canvas mode B, today done by hand) |
| `POST /tools/<name> {args, req}` | agent via gateway | `{ok, result}` or `{ok: false, error}`; in rounds mode `{pending: true, id}` |
| `GET /state?agent=` | agent | the agent-visible observation (swarmlab `observe`) |
| `GET /score` | controller, viewer | evaluator metrics; agents only through a declared tool |
| `POST /claims {key, ttl_s}`, `DELETE /claims/{key}`, `GET /claims?prefix=` | agent | TTL leases; `409` with owner when held |
| `POST /rounds/{n}/close` | controller | rounds mode: commit buffered actions, return outcomes per agent |
| `GET /events?after=<seq>&limit=` | controller, viewer | the world log, JSONL, each entry with `agent`, `req`, `seq`, `ts` |

Auth: by default the worker registers each agent at first use with the world's admin token and keeps
the minted token in the gateway; the harness never holds it, and flock's `FLOCK_AGENT_TOKEN` never
leaves flock (a world holding it could act on the controller as that agent). `shared` keeps canvas
mode A (`X-Flock-Agent` header). Every call carries `X-Request-Id` = `req` and `X-Flock-Turn`, which
the gateway sets (fixes the empty `FLOCK_TURN`).

What existing services change:
- **Canvas** (`experiments/pixel/canvas`): already has claims, region leases, score, rate limit, mode-B register and `GET /log?after=`. Add `GET /manifest` (claim, paint, release, score, region tools), `POST /tools/<name>` as aliases of its routes, `/events` as an alias of `/log`. The `canvas` CLI becomes the pack for CLI harnesses; LFM painters call `claim`/`paint` as function tools.
- **Eliza workspace**: `workspace.py` + `checker.py` + `sandbox.py` + the task board become a small FastAPI world: tools `ws_*`, `run_bot`, `*_task`, `submit_final`; rule check on every write (`violation` in `/events`); the score is the probe replay of `analyze.py`.

**How swarmlab worlds map.** A thin adapter `swarmlab.serve` (in swarmlab or flock `contrib/`, open question 6) wraps one `World` instance, its `Registry` and claim policy:
- `World.tool_schemas()` → `GET /manifest` (already JSON schema, strict-normalised); `validate` + `@tool` methods → `POST /tools/<name>`; `observe(agent)` → `GET /state`; `score()`/`verify()` → `GET /score`; the event log (`swarmlab/events.py`) → `GET /events`; `end_turn` → flock's `done`.
- Modes: `commit: async` (exists: each action commits when called) is the default under flock; `round_end` phase-commit is the optional **rounds** mode: tools buffer and answer `pending`, `POST /rounds/{n}/close` runs `commit_world` in seeded order and `end_round`.
- `ColoringGrid` (`paint`, `my_status`, `collective_status`, zones) is the first world served; its board tools (`read_board`/`post`) stay available as world tools when an experiment needs swarmlab's topology and delivery control (crossed zones), else comms §6 replaces them.
- Reused as is: paint accounting (`useful/duplicate/wrong/overwritten`), validate/error vocabulary, `Registry` TTL claims with `Advisory`/`Enforced`, metrics folded over the log, `swarmlab replay`, `world/render.py::grid_to_png`, and the scripted painters (`RowMajorPainter`, `QueuePainter`) driven over HTTP by a remote `AgentTools` shim as zero-LLM load generators.
- Replaced: the in-process runner (flock's loop drives turns and rounds), `LLMAgent` (flock's `llm` harness), `Gate`/budget ledger (flock providers and `limits`), `swarmlab job run` (flock workers + a world job).
- Known limits to fix before 500 agents: `Registry` TTLs are in rounds (`ttl_rounds`), so async mode needs a wall-clock TTL; `ColoringGrid.score()` rescans every cell; one cell per `paint` call. Measure with scripted painters first (the canvas did 800 req/s in one process; ColoringGrid is unmeasured).

## 5. Primitive 3: loop and rounds

```yaml
types:
  painter:
    loop:
      prompt: "Claim and paint. Repeat until the result says you are done, then call done."
      first: "Call status once, then done."        # sent alone, before `prompt` (run 3's fold bug)
      every: turn_end                              # turn_end | 30s | rounds
      while: {score_lt: {coverage_pct: 100}, until: 2h, max_rounds: 400, budget_usd: 12}
  dev:
    loop:
      prompt_file: prompts/round.md                # rendered with {round}, {elapsed}, {spend}
      every: 120s
      on: [comms.mention, comms.thread, world.event:violation]   # wake early, 2 s debounce
      while: {until: 2h, budget_usd: 14.6}
      final: {prompt_file: prompts/final.md, reserve_usd: 0.4, add_tools: [submit_final, best_vote]}
```

- The controller runs every loop. Each tick evaluates `while` (score from the world's `GET /score`, spend from its ledger) and sends one `POST /swarms/{s}/prompts {type, state: idle, skip_pending: true}`, so a type never has more than one pending prompt per agent (no stacking) and every prompt is admitted under `max_working` like any other.
- `every: turn_end` is the cadence that removes run 1's ~17 % idle gap and ~6 prompts per turn: the worker re-sends the loop prompt itself when a turn ends `completed`, until the controller pushes `loop_stop` (a new command). Errors end the loop for that agent after 3 in a row.
- `first` is delivered alone and the loop prompt only after its turn ended: the install-fold failure of run 3 cannot recur. With packs installed at boot, most `first` steps disappear.
- **Rounds** (`every: rounds`): prompt all live agents of the type, wait for every `turn_ended` or `round_timeout`, then `POST /rounds/{n}/close` on the world, then the next round. Agents read their observation with a `state` tool (forced first step) since the bulk route sends one text.
- **Event-driven** (`on:`): the controller tails `GET /events?after=` of the world and the comms activity feed (one poller per swarm), maps an event to the agents it names, debounces 2 s, and queues the loop prompt (`skip_pending`). This is Eliza's "new thread message or 120 s".
- Spawned children inherit their type's loop; a leader's run 3 step 5 becomes `loop` on the child type plus `scope: {parent: <leader>}` set at spawn, so a leader spends one turn, not 15 rounds of `sleep 30`.

## 6. Primitive 4: comms

```yaml
comms:
  kind: msgboard
  url: https://lvwerra-msgboard-demo.hf.space
  invite_env: BOARD_INVITE           # multi-use agent invite scoped to the channel
  channel: prj-eliza-v1              # resolved to an id once by the controller
  rules: {min_post_gap: 20s, max_posts: 80, max_len: 500, threads: per_topic, no_group_mentions: true}
  protocols: [one-thread-per-topic, lead-agree]   # prompt snippets appended to the instructions
  wake: [mention, thread]            # what feeds loop.on
```

- The worker joins at an agent's first prompt (not at assign, so 500 agents do not join in one burst) under the flock agent name, by generalising `worker/identity.py` (today's `msgboard_invite`). The token stays in the gateway, in a per-agent file outside the synced workdir; a relaunch re-joins by name (same principal, rotated token).
- Tools `board_post {text, thread: new|<root id>, reply_to?}` and `board_read {n?, thread?}` are built-in pack tools; the gateway enforces `rules` and returns "refused: wait N s" instead of calling the board (Eliza's 20 s gap, 32 refusals, ds-3 retried 6 times in 15 s).
- One activity poller per worker, not one long-poll per agent: msgboard caps open polls at 2,000 and ingress near 300 req/s (design §2).
- Group mentions: `@agents` expands to every agent on the board, not the channel's members, and wakes non-members through the activity feed. With `no_group_mentions` the gateway refuses posts containing it; `wake` only matches direct mentions of the agent and threads it posted in.
- Protocol snippets in `flock/prompts/protocols/*.md`, versioned with flock: `one-thread-per-topic` (Carlos, msg 2962), `lead-agree` (chat-test: first-named agent leads, partner answers `agree [id]`, act only after agree; 4/8 → 6/6), `help-offer` and `band-lease` (run 3 leaders: lease, announce, progress line every ~3 min, `STALLED` help request with a grant, `done, free capacity`).

## 7. Primitive 5: shared artifacts

```yaml
artifacts:
  bucket: hf://buckets/cmpatino/flock-runs
  layout: run                  # <bucket>/<swarm>/<run>/...
  best: {file: bot.py, check: world:promote_check}   # optional promoted slot
```
```
<bucket>/<swarm>/<run>/agents/<name>/<stem>_<UTC ts>.<ext>   each write a new version, own dir only
                      /shared/                                any agent, append-only names
                      /best/<file> manifest.json history.jsonl
                      /world/  events.jsonl.gz score.jsonl     copied by the controller at the end
```
- Tools `ws_list`, `ws_read`, `ws_write`, `promote_best {path, reason}`, `best_info` are a built-in pack served by the gateway; writes go to the agent's own dir only.
- Promotion has one writer: the controller (`POST /swarms/{s}/artifacts/promote`), which runs the world's check (Eliza: rule check + 3-message smoke), writes `best/` with `{path_promoted, agent, ts, reason, replaced, sha}` and appends `history.jsonl`. Readers such as the Eliza chat Space keep using `best.py`.
- Provenance: every write and promotion is a flock event `artifact_written {path, sha, bytes, reason, turn_id, req}` in the agent's stream.
- Final round: `loop.final` (§5) stops the loop at `budget_usd - reserve_usd` or `until`, adds `submit_final` (copies any file to `agents/<name>/final.<ext>`) and `best_vote`, sends one final prompt, then stops the agents.

## 8. Primitive 6: run observability

A per-run viewer, generalising the pixel progress Space (`experiments/pixel/progress/`), built from a run bundle the controller writes to `<run>/viewer/` every 60 s and at the end:

| file | from | content |
|---|---|---|
| `agents.jsonl` | flock events | per turn: agent, type, prompt source (loop, comms, api), tool calls (name, args digest, status, ms, `req`), violations, tokens, cost |
| `score.jsonl` | world `GET /score` per loop tick | metric time series |
| `world.events.jsonl.gz` | world `GET /events` | the task log, joinable on `req` |
| `transcript.jsonl` | comms | channel messages with thread, mentions |
| `run.json` | controller | spec (secrets removed), pinned prices, totals, stop reason |

Pages: run summary (score vs time, cost vs time, agents by state), per-agent timeline (what it chose, each tool call with its world outcome, posts, violations), board transcript, and replay: the world supplies a render route (canvas `/replay/frame?t=`; swarmlab `render_state()`; Eliza a file diff of `best/` history). Each side emits: flock the gateway events and `req` ids; the world `agent`, `req`, `seq` on every log entry and a `render` route; comms nothing new.

## 9. Hardening

- **Swarm budget stop.** `limits.budget_usd` becomes a hard stop (today it is checked only at spawn). At `budget_usd - final.reserve_usd` the controller stops loops, drops pending prompts and runs the final round; at `budget_usd` it sends `stop` to every agent. Overshoot is bounded by `max_working × max_budget_usd`, so per-call booking (llm, Codex, Pi do it; Claude books at turn end) is required for tight caps.
- **Per-service rate budgets** enforced in the gateway (token bucket per agent and per worker), declared per service: `services: {world: {per_agent: 2/s, burst: 5, global: 800/s}, board: {per_agent: 1/20s, global: 20/s}}`. The controller divides `global` by live workers and sends each worker its share on heartbeat. The world's `rate` from `/manifest` is the default.
- **Guards from the runs.** A 429 from any service is backoff with `Retry-After`, never a dead agent or a failed turn (1k run: 29 of 42 workers killed); `skip_pending` on every loop and event prompt (73,394 stacked prompts); `first` delivered alone (run 3); packs installed outside the workdir, `.flock-modes.json` kept for workdir files (#9, 10 agents on "Permission denied"); context bounded by server tokens (#7).

## 10. Migration in 3 steps

| step | builds | proof run | effort | deleted from the experiments |
|---|---|---|---|---|
| a | tool packs, natives, gateway (auth, rate, events), `loop` (incl. `turn_end`, `final`, `on: comms`), `comms` | Eliza on flock: 5 agents (repeat eliza-v0-run1 conditions), then 50 | 5–6 days | `swarm.py` loop, caps, join, rounds (660 lines), `llm.py`, `board.py`; pixel `driver.py` re-prompting and `leader.md` step 5 loops |
| b | world binding + protocol, `limits.budget_usd` stop, artifacts + promote; canvas and Eliza workspace as worlds; `swarmlab.serve` for `ColoringGrid` | coloring at 500 `llm` agents on flock (scripted load first, then LFM) | 6–8 days | the CLI download prompt, operator-side `/agents/register`, `workspace.py` bucket code, `best.py` writer side |
| c | run viewer + bundle, swarm events stream, per-service global budgets | re-run a and b with the viewer live | 4–5 days | the progress Space's pixel-specific code, `analyze.py` timelines, `guard-*.sh` |

Order inside the steps follows msg 3064 (1 → 3 → 2 → 4 → 5 → 6). Every step lands behind spec fields
that default off; no change to existing specs.

Open questions:
1. **World auth**: world-minted tokens registered by the worker (proposed), or worlds verifying flock agent tokens against the controller's `/whoami`?
2. **Score visibility**: swarmlab forbids agent-visible correctness; the canvas returns the score on every paint. Default `GET /score` to operator-only and let a world expose a score tool explicitly?
3. **Loop trigger**: worker-side `every: turn_end` (no idle gap, one more command) or controller ticks only?
4. **CLI harnesses**: `flock-tool` + instructions only, or also an MCP server generated from the manifest for Claude/Codex/Pi?
5. **`best/` ownership**: controller as the single writer (proposed) or the world?
6. **Where `swarmlab.serve` lives**: in swarmlab (a swarmlab change; decision 1 kept swarmlab unchanged in v1) or in flock `contrib/`?
