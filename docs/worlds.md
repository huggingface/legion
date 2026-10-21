# flock worlds: one spec from laptop to 1k, World API v1, and the primitives for task-driven swarms

Design note, 2026-10-21. Status: draft for Carlos's review, rev 3 (principles + on-ramp). No code yet. Board: design thread msg 3063-3167 (rev 3 direction in 3159-3167, rev 2 in 3142-3156), search thread 3144-3153.
Scope: what flock needs so an experiment like the Eliza swarm (`prj-agent-swarms/eliza-swarm/`) or the pixel swarm (`experiments/pixel/`) is worth running on flock at 5 agents on a laptop and at 1k on HF Jobs, with the same spec; plus one HTTP API any world can speak.
Goal (Carlos): "flock should be attractive regardless of the scale. if the task stays the same, it should be worth it for users to use flock whether it's with 5 or 1k agents" (msg 3161); "tools and any other customizable things should be outside flock and read at runtime. they should also be changeable once the swarm is running" (msg 3166); "leave the world implementation totally open and [...] focus on what's absolutely necessary to drive the agents at scale" (msg 3159).

## 1. Principles
1. **One spec from laptop to 1k.** The same spec, tools, prompts and events run 5 agents on a laptop (`flock run`) and 1k agents on HF Jobs; `compute: hf-jobs` (with the `namespace` and `bucket` jobs need) and `replicas` are the only lines that change. A 5-agent experiment on flock must be less work than a standalone harness; §4.7 is the test.
2. **Everything customizable lives outside flock, is read at runtime, and can change while the swarm runs.** Tools, prompts and protocol snippets, the world binding, comms settings, loop settings, limits, and capabilities granted later are files next to the spec or services the spec names. flock's code and wheel never change for an experiment. Everything loaded is versioned by content hash and every turn records the version it ran on.
3. **flock never interprets task state.** flock drives agents: compute, placement, admission, identity, budgets, the loop, the gateway, events. Task state, scoring and rule checks belong to the experiment's tools or to a world (any service, any language, optional). flock reads only the score metrics a `loop` condition names.

Where rev 2 contradicted them (tools installed as packs only, protocol snippets inside the flock wheel, the spec as the only thing read at apply), rev 3 changes it.

## 2. Problem
Both experiments hand-rolled the task side outside flock. **Tools**: painters `urlretrieve` the canvas CLI in their first prompt (`run/painter.md`); Eliza's 16 function tools live in its own loop (`eliza-swarm/prompts.py::tools`). **Loop and comms**: `run/driver.py` re-prompts idle painters; leaders run 30 s shell loops (`run/leader.md` step 5); Eliza runs rounds with a 2 s debounce or 120 s wake (`swarm.py::main_loop`); `run/board.py` and `eliza-swarm/board.py` each join msgboard, poll, rate-limit posts (20 s gap, 32 refusals in eliza-v0-run1). **Artifacts, stops, observability**: Eliza's bucket workspace and `best/` promotion (`workspace.py`, `best.py`), caps and `guard-*.sh`; the pixel progress Space and `analyze.py`. At hundreds of agents this breaks: a driver's bulk calls stacked 73,394 prompts (1k run); install + paint prompts folded into one turn, so 5 min in only 77 of 1,000 children had reached the canvas (run 3); each kit re-implements 429 backoff; nothing joins a tool call to the flock turn that made it (`FLOCK_TURN` is empty); every run builds its own viewer.

**The on-ramp is the same at any scale** (msg 3160, 3163). Deploy a controller (Space or job), launch worker jobs, create a bucket and tokens, rebuild the worker wheel when a tool changes, and get an agent that only has `bash` (`adapters/llm.py` offers `bash` and `done`). For a 2-hour, 5-agent experiment that is about a day of setup, so Eliza was built standalone: 2,326 lines, of which ~800 (loop with rounds, debounce and caps, router client, board client) overlap flock. What flock lacked on 21 Oct: function tools; a way to put task code (sandbox, rule check) on the worker; the loop, wake on mention, per-agent caps and final round; comms tools and post rules; `limits.budget_usd` as a stop (checked only at spawn); and iteration speed: `promote_best` was added at minute 49 with one file edit and a restart, which on flock is a wheel rebuild and a redeploy. The scale machinery (placement, admission, resume, 429 survival, bulk prompts, spawn, per-agent cost) is fine; the on-ramp is the design flaw.

## 3. Division of labour
- **flock** drives agents at any scale: compute (`local` or `hf-jobs`), placement, admission (`max_working`), identity, budgets, the loop, events, loading the experiment folder at runtime (§4), and the worker-side **gateway** every tool and service call passes through (secret, identity headers, rate budget, retries, provenance). The **experiment folder** holds everything customizable: spec, `tools/`, `prompts/`, optionally `world/`. Read at apply, shipped to workers as a versioned bundle, changeable live.
- A **world** is optional: any HTTP service that speaks World API v1 (§5), for shared state that tools alone cannot hold (claims, a canvas, a task board across workers). One world serves 5 or 500 agents.
- **comms** is a pluggable service (`kind: msgboard` today) behind the gateway. swarmlab is one world provider behind an adapter (§7.4); flock never imports swarmlab. flock never routes inboxes (decision 3), never judges results (checks are experiment tools or world routes), never stores world state.

```
 network group (CLI -> controller); with compute: local, the same picture as child processes of `flock run`
 controller ── loop ticks · budget stop · bundles · grants · hooks out · run bundle
   ▲ ▼ commands (assign, prompt, update), heartbeats, events         hooks ▼ ▲ world token
 worker (25-50 agents; local: one per type)                    world (job | child process), optional
   harness (llm | claude | codex | pi)                           /manifest  /tools/<name>  [+ capabilities]
     │ function call | flock-tool
     ▼
   gateway 127.0.0.1:7870 ── world:<tool> ──────────────────────►
     │ secret · X-Flock-Agent · rate · retry · req/turn ids
     ├─ <tool>: tool host (python -I, bundle <sha>, ctx) ── calls comms · artifacts · world via the gateway
     └─ comms:<tool> (msgboard) · artifacts (bucket | local dir)
```

## 4. On-ramp
### 4.1 `flock run`
```
$ cd experiments/eliza && flock run spec.yaml        # 5 agents on a laptop, no Space, no job, no bucket
$ flock apply spec.yaml                              # second terminal: live change (§4.4); after setting compute: hf-jobs,
                                                     # namespace, bucket and replicas: 500, the same command runs it on jobs
```
`flock run` starts the controller in-process on `127.0.0.1` (state in `.flock/<swarm>/`, a generated token written to `.flock/current` so `flock status`, `prompt`, `events`, `apply`, `grant` in another terminal reach it), applies the spec, starts the world if `world.dir` is set, launches workers as child processes, shows the terminal view (§4.5) and writes the run bundle to `runs/<swarm>-<UTC ts>/`. Ctrl-C runs `loop.final` and stops; a second Ctrl-C stops at once. Spec, tools, prompts, events and CLI are identical at both scales.

| | `compute: local` | `compute: hf-jobs` |
|---|---|---|
| controller | in the `flock run` process | Space or job (`docs/deploy-job.md`) |
| workers | child `flock-worker` processes, same env as a job (`FLOCK_URL`, `FLOCK_TOKEN`, `WORKER_ID`, `SWARM`, `TYPE`, `DEADLINE`) | HF Jobs, `agents_per_worker` per job |
| world (`world.dir`) | child process running `world.command` | one job in the network group (§6) |
| bucket, artifacts | `.flock/<swarm>/store/` (local dir store) unless `bucket:` is set | `bucket:` (required) |
| secrets | the shell env, or `.env` next to the spec (gitignored) | job secrets named in the spec |
### 4.2 The experiment folder
```
experiments/eliza/
  spec.yaml            types, loop, comms, artifacts, limits, compute
  tools/               *.py: every public function is a candidate tool; _*.py are helpers; requirements.txt optional
  prompts/             system.md, round.md, final.md, kickoff.md, protocols/*.md
  world/               optional: a World API v1 service (app.py), started by flock run or as a world job
```
Paths in the spec are relative to the spec file. `flock apply` (and `flock run`) packs the files the spec references into a **bundle**: a content-addressed tarball (`sha256` over sorted path + file hash), uploaded to the controller (`PUT /swarms/{s}/bundles/{sha}`, 5 MB cap, larger via the bucket) and fetched by workers. This generalises what `load_spec` does today for `system_prompt_file` (inline the file because neither controller nor workers can read the caller's files).
### 4.3 The tools-folder contract
```python
# tools/bots.py
from flock.tools import Ctx, ToolError
from ._sandbox import run_conversation

def run_bot(ctx: Ctx, path: str, messages: list[str]) -> dict:
    """Run a bot file in the sandbox on one fresh conversation and return its replies.

    Args:
        path: any workspace file, e.g. agents/ds-2/bot_2026-10-21T13-05-12Z.py or best/bot.py
        messages: user messages, in order (at most 20)
    """
    if len(messages) > 20: raise ToolError("too_many", "at most 20 messages per conversation")
    replies = run_conversation(ctx.artifacts.read(path), messages, mem_mb=64, turn_timeout_s=2.0)
    return {"replies": replies, "text": f"{len(replies)} replies"}
```
What the model sees (`llm` harness: a Chat Completions function tool; CLI harnesses: the same in the generated Tools section):
```json
{"type": "function", "function": {"name": "run_bot", "description": "Run a bot file in the sandbox on one fresh conversation and return its replies.",
 "parameters": {"type": "object", "additionalProperties": false, "required": ["path", "messages"], "properties": {
   "path": {"type": "string", "description": "any workspace file, e.g. agents/ds-2/bot_2026-10-21T13-05-12Z.py or best/bot.py"},
   "messages": {"type": "array", "items": {"type": "string"}, "description": "user messages, in order (at most 20)"}}}}}
```
- Name = function name; description = first docstring paragraph; parameters from type hints via pydantic (`Literal` → enum, defaults → optional, `list[str]`, `TypedDict`), descriptions from the `Args:` section, strict-normalised as world tool schemas are (§5.1). A parameter typed `Ctx` is hidden from the model.
- `ctx`: `agent`, `type`, `swarm`, `turn_id`, `bundle`; `ctx.artifacts` (§11), `ctx.comms` (§10), `ctx.world.call(tool, **args)`, `ctx.emit(kind, **data)` (a flock event with `req` and `turn`), `ctx.secret(name)`. Every service call goes through the gateway, so rate budgets and provenance apply to tools as to agents. Return a dict or a string; `text` is what small models see first. `ToolError(code, message)` and any exception become `{"ok": false, "error": ...}`, a result the agent reads, never a failed turn. Sync or async; `timeout_s` defaults to 60, override with `@flock.tools.tool(timeout_s=..., rate="1/20s")`.
- Execution: one **tool host** per type per worker, `python -I -m flock.toolhost <bundle dir>`, outside the agent workdir and the harness env (decision 26), in a venv built from `tools/requirements.txt` (uv, cached by hash). Module state is shared by every agent of the type on that worker; per-agent state goes through `ctx.artifacts` or a world. Non-Python tools: `tools/<name>.tool.json` with the World API tool shape plus `exec: {cmd: [...], stdin: json}` (stdout = result, exit ≠ 0 = tool error), for CLIs like the canvas one.
### 4.4 Live changes, versioning, `flock grant`
- `flock apply` on a running swarm re-reads the spec and the folders, uploads the new bundle, and the controller sends `update {type, spec, bundle}` (a new command) to that type's workers. Locally `flock run` does this on file save (1 s debounce); `--no-watch` turns it off. A worker fetches the bundle and starts a new tool host next to the old one. An agent mid-turn finishes on the version it started with; its next turn starts on the new one; the old tool host stops when its last call returns. Prompts and system prompt: the `llm` harness rebuilds its system message at the next turn start; CLI harnesses get the rewritten `CLAUDE.md`/`AGENTS.md` and a one-line "tools changed" note in the next prompt (open question 9).
- Controller-side settings (`loop`, `comms.rules`, `limits`, `world.url`) apply at the next tick or call. `world.dir`/`command` changes restart a local world (open question 15). Refused on a running swarm: `compute`, `swarm`, `namespace`, a type's `harness`. Versioning: every `turn_started` event carries `bundle: <sha12>` and the active `grants`; the run bundle keeps each version under `runs/<run>/bundles/<sha>/`, so any turn can be traced to the exact tools and prompts it ran with, and the viewer can diff versions.
- **Capabilities granted to running agents:**
```
flock grant eliza-v1 --type ds --tools promote_best,best_info         # every ds agent, from its next turn
flock grant eliza-v1 --agent ds-2 --tools ws_edit --note "You may now edit others' files into your dir."
```
A grant is an overlay on the spec, stored by the controller, numbered, evented (`grant {id, target, tools, note, by}`) and kept across re-applies (`flock grant <swarm> --list`, `--revoke <id>`). It names tools already in the bundle (tools folder, pack, world, comms); code arrives by `flock apply`, access by `grant`. A world unlocks tools during a run with the world-token scope `grant` (§5.4), limited to the type's `grantable:` list.
### 4.5 The terminal view
`flock run` in the foreground, and `flock status --watch <swarm>` for any swarm, local or on jobs:
```
eliza-v1  local  00:41:12 / 2:00:00  $3.12 / $15.00 (final reserve $0.60)  bundle 3f9a2c1 (+1 grant)
score     no world score · best/bot.py: agents/ds-1/bot_14-17-59Z.py (4 promotions)
agents    working 2 · idle 3 · stuck 0 · error 0      tool calls 742 (9 errors) · posts 61 (7 refused)
  ds-1  working  r21  $0.91  ws_write ok · run_bot ok          ds-2  idle  r19  $0.84  done "integrated memory"  ...
41:02 ds-3 comms:post [integration] ds-1's merged bot passes 9/10 probes; @glm-2 fallback ladder breaks on "ok"
```
Refresh 2 s from the controller's status and event stream. The run bundle (§12) is written to `runs/<swarm>-<ts>/` every 60 s and at the end.
### 4.6 New code in flock vs what exists
| piece | exists today | new |
|---|---|---|
| controller API, reconcile, admission, spawn, prices, events | yes | none |
| compute backend | `JobsClient` protocol, `HubJobsClient`, `FakeJobsClient` (tests, `FLOCK_FAKE_JOBS=1`) | `LocalJobsClient`: `launch` = child `flock-worker` with the job env, `inspect`/`cancel`/`logs_tail`/`list` on processes; `compute:` field, `namespace`/`bucket` optional when local |
| worker, storage | a process configured from env, pointed at `FLOCK_URL`; `BucketSync` over `HubBucketStore` | `update` command, bundle fetch; a local dir store behind the same seam |
| spec files | `load_spec` inlines `system_prompt_file` | folder loader + bundle (hash, upload, fetch) |
| tools | `llm`: `bash`, `done` | tool host, schema from signatures, hot reload, `ctx`, function tools in `llm`, `flock-tool` for CLI harnesses |
| live change | `apply` updates replicas and what new assigns carry | live `update` to running agents, versions on turn events, `flock grant` |
| CLI | apply, preflight, status, agents, agent, prompt, events, scale, remove, spawn, whoami, workers | `run`, `grant`, `status --watch`, `doctor`, `world-token` |
### 4.7 Yardstick: Eliza on flock
Today: 2,326 lines of Python in one process. On flock: one experiment folder, zero loop code.

| file | lines (est.) | from |
|---|---|---|
| `spec.yaml` | ~45 | `config.yaml` (agents, models, caps, loop, board) |
| `prompts/system.md`, `round.md`, `final.md`, `kickoff.md` | ~80 (markdown) | `prompts.py` `SYSTEM`, `ROUND`, `FINAL`; `config.yaml` kickoff |
| `tools/checks.py`, `tools/bots.py` | ~30 + ~30 | `check_write` (rule check), `check_promote` (rule check + 3-message smoke); `run_bot`, `submit_final`, `best_vote` |
| `tools/_sandbox.py`, `_sandbox_runner.py`, `_checker.py`, `rules.yaml` | 216 + 128 + 44 | unchanged |
| `world/app.py` | ~60 | `workspace.py::TaskBoard` on `flock.world` claims, task board as `state` |
| **Python task code** | **~464** | of which 344 copied unchanged |

Disappears: `swarm.py` (660: loop, rounds, debounce, caps, join, kickoff, final round, resume), `llm.py` (126), `board.py` (113), `common.py` (93: events, spend ledger, stops), ~300 lines of `workspace.py` (bucket layout, batched uploads, index, `best/` manifest and history become built-in artifacts tools), the 65-line tool schema list in `prompts.py` (generated from signatures): about 1,360 lines. Kept beside the experiment: `best.py` (the chat Space reads the same `best/` layout), `analyze.py` until the viewer (step c), `mock_llm.py` and `fake_board.py` for offline runs.
```yaml
swarm: eliza-v1
compute: local
providers: {router: {type: openai_compat, base_url: https://router.huggingface.co/v1, api_key_env: HF_TOKEN}}
models: {deepseek: {model: "router:deepseek-ai/DeepSeek-V4.1-Flash-1017:novita", pricing: [0.30, 1.20, 0]},
         glm: {model: "router:zai-org/GLM-5.3-Flash:novita", pricing: [0.15, 0.50, 0]}}   # + extra: {max_tokens: 32000}
world: {dir: world/, command: [uvicorn, app:app, --port, "8765"], port: 8765}
comms: {kind: msgboard, url: https://lvwerra-msgboard-demo.hf.space, invite_env: BOARD_INVITE, channel: prj-eliza-v1,
        kickoff_file: prompts/kickoff.md, wake: [mention, thread],
        rules: {min_post_gap: 20s, max_posts: 80, threads: per_topic, no_group_mentions: true}}
artifacts: {layout: run, checks: {write: checks.check_write}, best: {file: bot.py, check: checks.check_promote}}
types:
  ds: &dev
    harness: llm
    model: deepseek
    replicas: 3
    system_prompt_file: prompts/system.md
    tool_choice: required
    max_turns: 14                         # Eliza's tool calls per round
    tools: [ws_list, ws_read, ws_write, run_bot, comms:read, {use: comms:post, as: board_post},
            world:list_tasks, world:add_task, world:claim_task, world:update_task, promote_best, best_info, done]
    grantable: [ws_edit]
    loop:
      prompt_file: prompts/round.md         # {round} {elapsed} {spend} {cap} {why} {last_summary} + context below
      context: [world.state, artifacts.recent:15, comms.threads, comms.recent:20]
      every: 120s
      on: [comms.mention, comms.thread]
      stagger: 4s
      while: {until: 2h, agent_usd: {deepseek: 3.60, glm: 2.10}}   # per-agent cap by model
      final: {prompt_file: prompts/final.md, add_tools: [submit_final, best_vote]}
  glm: {<<: *dev, model: glm, replicas: 2}
limits: {budget_usd: 15, final_reserve_usd: 0.60}
```
Acceptance, all four before step (a) goes to jobs; if (a1) is not clearly easier than the standalone harness, that is the signal to stop investing (msg 3160):
1. Python task code under 500 lines (estimate ~464); prompts (~80 lines of markdown) and the spec are counted separately; zero lines of loop, LLM client or board client in the folder.
2. `flock run spec.yaml` on a laptop reaches the first agent turn in under a minute (no image build, no job, no bucket).
3. Same behaviour as eliza-v0-run1 at 5 agents (3 DeepSeek + 2 GLM, $15 cap, 2 h): every model call a tool call (run1: 2,291 of 2,291, 9 malformed arguments), 5/5 final submissions, rule-check rejections and post refusals logged as tool results, `best/` promotions with history, cost within ±20 % of run1's $9.72 for comparable activity (run1: 172 thread messages, 129 files).
4. The minute-49 change reproduced live: start without `best:`, then add `check_promote` to `tools/checks.py`, `best:` to `artifacts` and `promote_best`, `best_info` to the type's tools, `flock apply` (or save under `flock run`); agents use them from their next turn and the turn events show the bundle switch.

## 5. World API v1
HTTP + JSON. Errors are `{"error": {"code", "message", "retry_after_s"?}}`; `429`/`503` carry `Retry-After`. Worlds ignore unknown fields and headers, flock ignores unknown capabilities. `world_api` is a major version: changes within v1 are additive, flock supports the current and previous major and refuses an unknown one at apply. A world is optional; tools alone (§4.3) cover experiments without shared state.
### 5.1 Required core
| route | caller | contract |
|---|---|---|
| `GET /manifest` | apply, workers, doctor | the manifest below; doubles as the health check (200 = ready) |
| `POST /tools/<name>` `{"args": {...}}` | agent or tool via gateway | `{"ok": true, "result": {...}, "text"?: "one status line"}` or `{"ok": false, "error": {...}}`; in rounds mode may answer `{"pending": true, "id"}` |

A tool error is a result the agent reads, never a failed turn. Manifest schema (condensed; `parameters` is JSON Schema, strict-normalised by flock as swarmlab's `ToolSchema.normalized` does):
```json
{"type": "object", "required": ["world_api", "name", "version", "tools"], "properties": {
  "world_api": {"const": "v1"}, "name": {"pattern": "^[a-z][a-z0-9-]{0,31}$"}, "version": {"type": "string"},
  "capabilities": {"type": "array", "uniqueItems": true, "items": {"enum":
    ["state", "score", "claims", "rounds", "events", "render", "checks", "hooks", "controller", "rate"]}},
  "tools": {"type": "array", "items": {"required": ["name", "description", "parameters"], "properties": {
    "name": {}, "description": {}, "parameters": {"type": "object"}, "examples": {"type": "array"}, "timeout_s": {}}}},
  "score": {"properties": {"metrics": {"type": "object"}, "agent_visible": {"default": false}}}, "rounds": {"properties": {"timeout_s": {}}},
  "checks": {"properties": {"write": {}, "promote": {}}}, "hooks": {"properties": {"path": {"type": "string"}, "events": {"type": "array"}}},
  "controller": {"properties": {"scopes": {"items": {"enum": ["prompt", "events", "agents", "stop", "spawn", "grant"]}}}},
  "rate": {"properties": {"per_agent": {"type": "string"}, "burst": {"type": "integer"}, "global": {"type": "string"}}}}}
```
### 5.2 Optional capabilities
flock calls only routes the manifest declares. `flock apply` checks the spec against the manifest and fails naming the missing capability, so nothing degrades silently at run time.

| capability | routes | flock uses it for | without it |
|---|---|---|---|
| `state` | `GET /state?agent=` | built-in `state` tool; `loop.context: world.state`; forced first step of a round | agents see only tool results |
| `score` | `GET /score` → `{metrics..., done?}` | `loop.while: score_*`, `done` stop, viewer, bundle | score conditions rejected |
| `claims` | `POST /claims {key, ttl_s}`, `DELETE /claims/{key}`, `GET /claims?prefix=`; 409 + owner when held | generic lease tools (`world:lease`, `world:unlease`, `world:leases`), lease timeline | a world may still lease inside its own tools |
| `rounds` | `POST /rounds/{n}/close` → outcomes per agent, idempotent per `n` | `loop.every: rounds` | `every: rounds` rejected |
| `events` | `GET /events?after=<seq>&limit=` → `{seq, ts, kind, agent, req, turn, data}`; `kind` reuses swarmlab's names (`paint`, `claim`, `violation`) | `loop.on: world.event:<kind>`, bundle, join on `req` | `on: world.*` rejected; gateway call log only |
| `render` | `GET /render?t=` → PNG or JSON frame | viewer replay | timelines only |
| `checks` | `POST /checks/write {agent, path, sha}`, `POST /checks/promote {agent, path}` → `{ok, violations}` | `artifacts.checks: world`, `best.check: world` (§11) | use a tools-folder check, or none |
| `hooks`, `controller` | `POST <hooks.path>`; `controller` only requests a world token | lifecycle webhooks (§5.3); world-driven prompting (§5.4) | none sent; the world is passive |
| `rate` | none; manifest hint | default gateway budgets (§13) | flock default 2/s per agent |
### 5.3 Hooks, flock → world
The world subscribes in `hooks.events`; the controller (one sender per swarm) POSTs batches of up to 500 events or 1 s: `{"swarm", "run", "events": [{"seq", "ts", "event", "data"}]}`. Delivery (proposed, open question 5): at-least-once, ordered per swarm, `seq` for dedupe, 2xx acks, 5 s timeout, retries with backoff for 5 min, then `world_hook_dropped` in the swarm events. Hooks never block flock.

| event | data | when |
|---|---|---|
| `agent_assigned` / `agent_removed` | `{agent, type, parent, worker}` / `{agent, reason}` | agent placed (apply, scale, spawn), before its first prompt / scale-down, cascade removal, dead past the relaunch budget |
| `turn_ended` | `{agent, turn_id, status, prompt_source, tool_calls, cost_usd, bundle}` | every turn end (from worker events) |
| `loop_tick`, `bundle_changed`, `grant` | `{type, tick, round?, spend_usd}`, `{type, bundle}`, `{id, target, tools}` | each loop tick; a live apply reached the type; a grant or revoke |
| `run_stopping` / `run_stopped` | `{reason: budget\|until\|done\|operator\|world, grace_s}` / `{reason, totals}` | before the final round; after agents stopped, before the bundle copy |
### 5.4 Hooks, world → flock
If the manifest declares `controller`, `flock apply` mints a **world token**: a third principal next to the controller token and agent tokens (`agent_may` in `controller/api.py`), limited to that swarm and the requested scopes, sent as `X-Flock-Token`. A started world gets `FLOCK_WORLD_TOKEN` and `FLOCK_CONTROLLER_URL` in its env; for a `url:` world the operator reads it with `flock world-token <swarm>`.

| scope | routes | notes |
|---|---|---|
| `prompt` | `POST /swarms/{s}/prompts`, `POST /agents/{name}/prompt` | admitted under `max_working`; `skip_pending` on by default |
| `events`, `agents` | `GET /agents/{name}/events`, swarm stream (step c), `GET /swarms/{s}/agents` | read only |
| `stop` | `POST /swarms/{s}/stop {reason}` (new) | runs `loop.final`, then stop, like a budget stop |
| `spawn` | `POST /agents/{name}/spawn` | only where the spec's `spawn:` policy allows; counts against `limits` |
| `grant` | `POST /swarms/{s}/grants {target, tools, note?}`, `DELETE /swarms/{s}/grants/{id}` (new) | only tools in the target type's `grantable:`; e.g. unlock `submit_final` at a milestone |

A world token never applies, scales or removes swarms, changes types, models, limits or bundles, or reads other swarms. World-driven loops: omit `loop:`, subscribe to `turn_ended`, prompt each agent with its next observation (a game master, a debate judge, swarmlab's runner inverted). flock still admits, budgets and rate-limits every prompt.
### 5.5 Auth and provenance
- One shared secret per world; no world-minted per-agent tokens and no `/agents/register` (canvas mode B is dropped). Started worlds (`dir`/`run`): flock generates it and hands it to the world (`FLOCK_WORLD_SECRET`) and every worker's gateway; `url:` worlds: `world.secret_env`. Only the gateway holds it; the harness env never does (allowlisted env, decision 26), so an agent with `bash` cannot forge a call. Agents reach the world only through the gateway: a world job sits in the network group and is not public unless `expose: true` (read-only routes); a local world binds `127.0.0.1`. Gateway → world headers: `X-Flock-World-Secret`, `X-Flock-Agent`, `X-Flock-Type`, `X-Flock-Swarm`, `X-Request-Id` (one per tool call), `X-Flock-Turn` (fixes the empty `FLOCK_TURN`), `X-Flock-Bundle`. Controller → world calls carry the secret and no `X-Flock-Agent`. The secret has its own header because through the HF Jobs proxy `Authorization` carries the HF token. A world trusts `X-Flock-Agent` iff the secret matches and logs `req` and `turn` on every event. `FLOCK_AGENT_TOKEN` never leaves flock.

## 6. Where worlds live
| what | where |
|---|---|
| protocol, conformance, Python base | this note (then `docs/world-api.md`); `flock doctor --world <url>` (§7.3); `flock.world` (`src/flock/world/`): FastAPI app from typed tools (same signature rules as §4.3), secret check, claims with wall-clock TTL, event log with `req`/`turn`, hook receiver with dedupe, world-token client |
| experiment worlds | `experiments/<name>/world/`, shipped in the bundle like `tools/`; a `flock-worlds` repo if they pile up |
| reference worlds | `worlds/canvas` (moved from `experiments/pixel/canvas`); the swarmlab adapter in `worlds/swarmlab/` (`swarmlab.serve`), or in swarmlab if its owner wants it there |

One `world:` block for both scales:
```yaml
world:
  name: canvas                     # network alias, default: the manifest name; tools appear as world:<tool>
  dir: world/                      # shipped in the bundle; local: child process, jobs: one job running the same command
  command: [python, -m, canvas, --port, "8765"]
  port: 8765                       # env: {CANVAS_TARGET: mona.png} for both
  run: {image: ghcr.io/huggingface/flock-worlds:base, flavor: cpu-upgrade, expose: false}   # jobs only
```
Instead of `dir`/`command`, `url:` binds an existing world (with `secret_env`); `manifest: auto` (default) or a pinned file, whose drift from the live manifest fails apply (open question 8).
`flock apply` starts the world first, waits for `GET /manifest` 200 (local 30 s, jobs 10 min), validates the spec against it, then places agents. The controller re-reads `/manifest` on every apply. A crashed world is relaunched under the same alias; meanwhile tool calls get `503` and the gateway backs off. The world owns its state (disk + its own bucket mirror). At run end and on `flock remove` the controller copies `/events` (all pages) and `/score` into `<run>/world/`, then stops the world. One world per swarm in v1.

## 7. Write your own world
### 7.1 A minimal world: `experiments/digits/world/app.py`
1,024 cells with hidden target digits; agents claim a run of 32, read its targets, write digits.
```python
import random
from flock.world import Agent, ToolError, World

N, RUN = 1024, 32
TARGET = random.Random(7).choices(range(10), k=N)
cells: list[int | None] = [None] * N
world = World("digits", version="0.1.0", rate={"per_agent": "2/s", "burst": 5, "global": "500/s"})

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

@world.score                                                     # operator-only unless agent_visible=True
def score() -> dict:
    ok = sum(c == t for c, t in zip(cells, TARGET))
    return {"correct_pct": 100 * ok / N, "done": None not in cells}

@world.on("agent_removed")                                       # subscribes to the hook
def drop(agent: Agent, reason: str) -> None:
    world.claims.release_all(agent)

app = world.app()   # /manifest /tools/* /score /claims /events /hooks; secret from FLOCK_WORLD_SECRET
```
An `empty` tool (open runs) and a `@world.state` function complete it (~55 lines). `GET /manifest` then declares `capabilities: [claims, events, hooks, rate, score]`, the tool schemas generated from the signatures, `score.metrics` and `hooks.events: [agent_removed]`.
### 7.2 The spec that binds it
```yaml
swarm: digits-50
compute: local                                   # hf-jobs (+ namespace, bucket) for 500 fillers
world: {dir: world/, command: [uvicorn, app:app, --host, 0.0.0.0, --port, "8765"], port: 8765, run: {flavor: cpu-basic}}
types:
  filler: {harness: llm, model: "lfm:LiquidAI/LFM2.5-2.6B", replicas: 50, tools: [world:empty, world:claim, world:write, done],
           loop: {every: turn_end, while: {score_lt: {correct_pct: 100}, until: 1h},
                  prompt: "Call empty, claim one run, write its digits. Repeat until no runs are open, then call done."}}
limits: {budget_usd: 3}
```
### 7.3 `flock doctor --world <url>`
Run it against a scratch instance (it calls tools): `--secret-env WORLD_SECRET`, optional `--spec spec.yaml`, `--load 200/s --for 60s`. Exit 1 on any failure, one line per check.
- core and auth: manifest validates, `world_api` supported, tool schemas strict-normalisable, each tool's `examples` return the envelope; no secret → 401; with it, `X-Flock-Agent: doctor-a` / `doctor-b` are attributed to those names; each declared capability: score metrics match the manifest; events `seq` monotonic with `req` echoed; a second agent's claim gets 409 with the owner; `rounds/{n}/close` idempotent; a repeated hook `seq` is acked and ignored; `render` returns PNG or JSON; `--spec`: every capability and tool the spec needs exists (world, tools folder, packs); `--load`: p50/p99 latency and error rate at that rate. `flock doctor spec.yaml` without `--world` checks the folder: every listed tool loads and has a valid schema.
### 7.4 Existing worlds on the API
| | canvas (`worlds/canvas`) | Eliza (`experiments/eliza/world`) | swarmlab `ColoringGrid` (`swarmlab.serve`) |
|---|---|---|---|
| tools | aliases of `/claim`, `/paint`, `/release`, `/claim/region*`; the `canvas` CLI as an `exec` tool for CLI harnesses | task board only; files, `run_bot`, checks are tools-folder (§4.7) | `@tool` methods via `validate` |
| manifest / `state` / `score` | new / `/status`, `/region` / `/score` | new / task board / none (operator reads `best/`) | `World.tool_schemas()` / `observe(agent)` / `score()`, `verify()` |
| `claims` / `rounds` | pixel claims + region leases / n/a | task claims / n/a | `Registry` (`Advisory`/`Enforced`) / `round_end` phase-commit |
| `events` / `render` | alias of `/log?after=` / `/replay/frame?t=` | task changes / n/a | `swarmlab/events.py` / `world/render.py::grid_to_png` |

- swarmlab, reused as is: paint accounting (`useful/duplicate/wrong/overwritten`), validate/error vocabulary, TTL claims, metrics folded over the log, `swarmlab replay`, the scripted painters (`RowMajorPainter`, `QueuePainter`) over HTTP as zero-LLM load generators. Replaced by flock: the in-process runner, `LLMAgent`, `Gate`/budget ledger, `swarmlab job run`. Limits before 500 agents: `Registry` TTLs count rounds (`ttl_rounds`), async needs wall-clock TTLs; `ColoringGrid.score()` rescans every cell; one cell per `paint`; its req/s is unmeasured (the canvas did 800 req/s in one process), so `flock doctor --load` with scripted painters comes first.

## 8. Primitive 1: tools (folder first, packs second)
A type's `tools:` is one list; each entry resolves in this order, and the model sees the bare name (function names allow no `:`), or `as:` when set:

| entry | resolves to | shipped how |
|---|---|---|
| `run_bot` | a public function in `tools/*.py`, or `tools/run_bot.tool.json` (`exec`) | in the bundle; hot-reloaded (§4.4) |
| `read_file`, `write_file`, `done`, `state`, `ws_*`, `promote_best`, `best_info`, `find` | built-ins (natives, artifacts §11, world `state`) | in flock |
| `world:<tool>`, `comms:<tool>` | the world's manifest, the comms service (§10) | services, via the gateway |
| `<pack>:<tool>` | a pip-installed pack | `packs:` below |

- **Packs** are the secondary mechanism, for tools shared across experiments: a pip package exposing functions with the §4.3 contract through the entry point `flock.tools`. `packs: {probe: "flock-pack-eliza-probe==0.2"}` at the top of the spec; installed into the tool host venv (locally at bundle build, on jobs at worker boot), `pack_installed` event or the type fails with the pip error. The version pin is part of the bundle hash, so a pack upgrade is a live change like any other.
- `llm`: every listed tool is a Chat Completions function tool next to `bash` (droppable with `bash: false`); `_run_tool` sends non-bash calls to the gateway, which routes them to the tool host, the world or comms. Natives `read_file {path, start_line?, end_line?}`, `write_file {path, content}` are confined to the workdir, no shell quoting (the smoke lost 2 of 3 painters to `\'`). Parity items for Eliza: streaming (the router returns 504 near 120 s on whole-file writes) and an empty-output retry with thinking off (GLM).
- `claude`, `codex`, `pi`: `flock-tool <name> --json '<args>'` on `PATH` (it asks the gateway, so it always sees the current bundle and grants), and a generated Tools section (name, description, one example) in `CLAUDE.md`/`AGENTS.md`; MCP is open question 3. Small models: `tool_choice: required` + `done` (decision 31), few tools, one example per tool in the preamble, `text` first. The LFM probe's 20/20 valid calls were with a named, forced tool; Eliza's flash models made a tool call on all 2,291 forced calls with 13 tools (9 malformed arguments).
- The **gateway** stays the only path to services: tools-folder functions, world tools and comms tools all get the identity headers, rate budgets, retries and `req`/`turn` provenance. Changes: `src/flock/tools/` (loader, schema, `Ctx`), `src/flock/toolhost.py`, `worker/gateway.py`, `adapters/llm.py`, `agents.py` (`PATH`, `FLOCK_GATEWAY`); `tools: list` on `TypeSpec`, `packs` on `SwarmSpec`.

## 9. Primitive 3: loop and rounds
Eliza's loop is in §4.7; a turn-driven painter is `loop: {every: turn_end, first: "Call status once, then done.", prompt: "Claim and paint. Repeat until the result says you are done, then call done.", while: {score_lt: {coverage_pct: 100}, until: 2h, max_rounds: 400, budget_usd: 12}}`.
- The controller runs every loop: each tick evaluates `while` (score from `GET /score`, spend from its ledger, `agent_usd` per agent) and sends one `POST /swarms/{s}/prompts {type, state: idle, skip_pending: true}`, so no agent stacks loop prompts and every prompt is admitted under `max_working`.
- Rendering on the worker: the controller sends the template id; the worker fills built-ins (`round`, `elapsed`, `wall`, `spend`, `cap`, `why`, `last_summary` from `done`) and each `context` source through the gateway, so Eliza's per-agent state prompt (task board, newest files, threads, last 20 messages) needs no code. A tools-folder function `def x(ctx: Ctx) -> str` can be a context source. Capability needs, checked at apply: `score_*` needs `score`, `every: rounds` needs `rounds`, `on: world.event:*` needs `events`, `context: world.state` needs `state`. A world-driven loop (§5.4) needs no `loop:` at all.
- `every` is `turn_end`, a duration (`30s`) or `rounds`. `turn_end` removes run 1's ~17 % idle gap: the worker re-sends the loop prompt when a turn ends `completed`, until the controller pushes `loop_stop` (a new command); 3 errors in a row end it for that agent (open question 2). `first` is delivered alone and the loop prompt only after its turn ended, so run 3's install fold cannot recur. Rounds: prompt all live agents of the type, wait for every `turn_ended` or `rounds.timeout_s`, `POST /rounds/{n}/close`, next round; agents read their observation with `state` (forced first step). `on:`: one controller poller per swarm tails `GET /events` and the comms activity feed, maps an event to the agents it names, debounces 2 s, queues the loop prompt (`skip_pending`). This is Eliza's "new thread message or 120 s". Spawned children inherit their type's loop; run 3's leader `sleep 30` rounds become `loop` on the child type plus `scope: {parent: <leader>}`.

## 10. Primitive 4: comms
The `comms:` block is in §4.7. Fields: `kind` (`msgboard`), `url`, `invite_env` (multi-use agent invite scoped to the channel), `channel` (resolved to an id once by the controller), `kickoff_file` (posted once by the swarm's host identity `<swarm>-host`, never an agent), `rules` (`min_post_gap`, `max_posts`, `max_len`, `threads: per_topic`, `no_group_mentions`), `protocols` (prompt files appended to the instructions), `wake` (what feeds `loop.on`), `search` (`board` | `index`).
- The worker joins at an agent's first prompt (not at assign, so 500 agents do not join in one burst) under its flock name, generalising `worker/identity.py`. The token stays in the gateway outside the synced workdir; a relaunch re-joins by name. `comms:post {text, thread: new|<root id>, reply_to?}` and `comms:read {n?, thread?}`; the gateway enforces `rules` and answers "refused: wait N s" without calling the board (Eliza: 32 refusals, ds-3 retried 6 times in 15 s). `rules` and `protocols` are live settings (§4.4): Eliza's two mid-run rule relays become an edit and an apply.
- One activity poller per worker, not one long-poll per agent: msgboard caps open polls at 2,000 and ingress near 300 req/s. `@agents` reaches every agent on the board (eliza-v0-run1: group mentions reached 21 non-team board members), so `no_group_mentions` refuses it and `wake` matches only direct mentions and the agent's own threads. Protocol snippets are experiment files (principle 2); flock ships starters in `examples/protocols/` (`one-thread-per-topic` msg 2962, `lead-agree` 4/8 → 6/6, `help-offer`, `band-lease`) to copy, never loaded from the wheel.

**Search** (msg 3144-3153), no msgboard change: `comms:search {query, thread?, from?, since?, limit?}`, channel-scoped, messages only, returns id, thread, author, time, snippet.
- Cut 1, `search: board` (~1 day): the board's `GET /v1/search` with `in:#<channel>` and `sort=relevance` (FTS5 `bm25()` underneath); the tool relaxes the query (all words, then drop words until something matches, prefixes). Still AND-only and unstemmed underneath; every call spends board rate budget. Cut 2, `search: index` (2-3 days): the worker's poller feeds a per-worker SQLite FTS5 index of the channel (`porter unicode61`, BM25, AND/OR, filters by thread, author, time, mention), board API as cold-start fallback.

## 11. Primitive 5: shared artifacts
Spec: `artifacts: {bucket: "hf://buckets/cmpatino/flock-runs", layout: run, checks: {write: world | <tools fn>}, best: {file: bot.py, check: world | <tools fn>}}`, a tools function written `<module>.<fn>` (`checks.check_write`); with `compute: local` and no bucket, the same layout under `.flock/<swarm>/store/`.
```
<store>/<swarm>/<run>/agents/<name>/<stem>_<UTC ts>.<ext>   each write a new version, own dir only
                     /shared/<file>  manifest.jsonl          any agent, append-only names
                     /best/<file> manifest.json history.jsonl
                     /world/  events.jsonl.gz score.jsonl     copied by the controller at the end
```
- Built-in tools served by the gateway: `ws_list`, `ws_read`, `ws_write`, `ws_edit {src_path, old, new, reason}`, `ws_share {path, description}`, `find`, `promote_best {path, reason}`, `best_info`; the same operations on `ctx.artifacts` for tools-folder code. Writes go to the agent's own dir; a configured write check runs first and its violations are the tool result. `shared/manifest.jsonl`: one line per shared file `{path, author, description, sha, from, turn_id, req, ts}`. `find {query, author?, since?, limit?}` ranks it with BM25 over path + description (the FTS5 code of comms cut 2).
- Promotion has one writer, the controller (`POST /swarms/{s}/artifacts/promote`, open question 4): it runs the promote check (Eliza: rule check + 3-message smoke, now `tools/checks.py`), writes `best/` with `{path_promoted, agent, ts, reason, replaced, sha}`, appends `history.jsonl`. The Eliza chat Space keeps reading via `best.py`. Every write, share and promotion is a flock event `artifact_written {path, sha, bytes, reason, turn_id, req, bundle}`. `loop.final` stops the loop at `budget_usd - reserve_usd` or `until`, adds `submit_final` and `best_vote`, sends one final prompt, then stops the agents.

## 12. Primitive 6: run observability
The terminal view (§4.5) is the first viewer, at every scale. The run bundle behind it and the web viewer, written to `runs/<swarm>-<ts>/` (local) or `<run>/viewer/` in the bucket every 60 s and at the end. Web pages (step c): summary (score, cost, agents by state vs time), per-agent timeline (each tool call with its outcome, posts, violations, bundle switches), transcript, and replay from the world's `render`; a page whose capability is missing is hidden.

| file | from | content |
|---|---|---|
| `agents.jsonl` | flock events | per turn: agent, type, bundle, grants, prompt source (loop, comms, world, api), tool calls (name, args digest, status, ms, `req`), violations, tokens, cost |
| `score.jsonl`, `world.events.jsonl.gz` | `GET /score` per tick, `GET /events` | metric time series; the task log, joined on `req` |
| `transcript.jsonl`, `run.json`, `bundles/<sha>/` | comms, controller | channel messages; spec (secrets removed), world manifest, pinned prices, totals, stop reason; every version loaded |

## 13. Hardening
- **Swarm budget stop.** `limits.budget_usd` becomes a hard stop (today checked only at spawn). At `budget_usd - final_reserve_usd` the controller stops loops, drops pending prompts, sends `run_stopping` and runs the final round; at `budget_usd` it stops every agent. Overshoot is bounded by `max_working × max_budget_usd`, so tight caps need per-call booking (llm, Codex, Pi; Claude books at turn end). **Per-service rate budgets** in the gateway (token bucket per agent and per worker): `services: {world: {per_agent: 2/s, burst: 5, global: 800/s}, board: {per_agent: 1/20s, global: 20/s}}`, defaulting to the manifest's `rate`; per-tool `rate` from `@tool(rate=...)`; the controller divides `global` by live workers on heartbeat. World-token calls get their own budget (20 req/s).
- **Guards from the runs.** A 429 from any service is backoff with `Retry-After`, never a dead agent or a failed turn (1k run: 29 of 42 workers killed); `skip_pending` on every loop, event and world prompt (73,394 stacked prompts); `first` delivered alone (run 3); tool hosts and packs outside the workdir, `.flock-modes.json` for workdir files (#9); context bounded by server tokens (#7); a bundle that fails to load (import error, bad schema) is refused at apply and never replaces a working one on the workers.

## 14. Migration in 3 steps
| step | builds | proof run | effort | deleted from the experiments |
|---|---|---|---|---|
| a1 | `local` backend + local store, `flock run` + terminal view + run bundle, tools folder (loader, schema, tool host, hot reload, `ctx`), `llm` function tools + natives + streaming, gateway, World API required core + minimal `flock.world` (tools, claims, secret) + `world.dir`, `loop` (`every`, `on: comms`, `context`, `stagger`, `final`), `comms` post/read/rules/kickoff | Eliza on a laptop, 5 agents, eliza-v0-run1 conditions; §4.7 acceptance 1-3 | ~9 days | `swarm.py` (660), `llm.py`, `board.py`, `common.py`, most of `workspace.py`; pixel `driver.py` re-prompting, `leader.md` step 5 |
| a2 | bundle versioning + live `update`, `flock grant`, search cut 1, `compute: hf-jobs` for the same folder | §4.7 acceptance 4 locally, then the same spec on jobs at 50 agents | ~2.5 days | Eliza's mid-run relaunches and rule relays |
| b | full World API capabilities, `flock doctor`, `world.run` on jobs, hooks both ways + world token (incl. `grant`), budget stop, artifacts `find` + promote on jobs, pip packs; `worlds/canvas`, `worlds/swarmlab` | `digits` and canvas under `doctor --load`, then coloring at 500 `llm` agents (scripted load first, then LFM) | 8.5-10.5 days | CLI download prompt, `/agents/register`, `experiments/pixel/canvas` |
| c | web viewer, swarm events stream, global rate budgets, search cut 2 | re-run a and b with the viewer live | 6-8 days | progress Space pixel code, `analyze.py` timelines, `guard-*.sh` |

**What changed in step (a) and why.** Rev 2's (a) was 6-7 days: packs, natives, gateway, `loop`, `comms` + search cut 1, proven on jobs. Rev 3's (a) is 10.5-11.5 days (a1 ~9, a2 ~2.5) and starts on a laptop, because principle 1 makes the 5-agent laptop run the first proof and the cheapest test of whether flock is worth maintaining (msg 3160, 3163):
- +2 days: `local` backend, `flock run`, terminal view, run bundle (the "about 2 extra days" of msg 3163);
- +1.5 days: bundle versioning, live `update` to running agents, `flock grant` (principle 2, msg 3166, which came after that estimate);
- +1 day moved from (b): World API required core and a minimal `flock.world`, because Eliza's task board is shared state across agents and principle 3 keeps it out of flock, so a world is needed in (a1) already;
- ±0: the tools folder replaces pip packs as the (a) mechanism (similar cost); pip packs (~0.5 day) move to (b), which Eliza does not need.

Totals: 25-30 days, up from rev 2's 21-26 (+2 local mode, +1.5 live changes, +0.5 pip packs kept as their own item in (b); the world core moves, it is not added). Every new spec field defaults off; a spec without `compute:` behaves as today (`hf-jobs`). Order of primitives inside (a1): local backend → tools folder → loop → world core → comms, so a 5-agent `flock run` with tools-folder tools and no board is usable after the first ~5 days.

## 15. Open questions
1. **Score visibility**: `score.agent_visible: false` by default, so flock exposes no score tool unless the world opts in. flock does not filter tool results, so a world that returns the score on every call (the canvas does) leaks it anyway: accept that as the world's choice?
2. **Loop trigger**: worker-side `every: turn_end` (no idle gap; one new command, `loop_stop`) or controller ticks only? A world can also re-prompt at turn end via `turn_ended` + world token, at a few seconds more latency per turn.
3. **CLI harnesses**: `flock-tool` + instructions only, or also an MCP server generated from the bundle and manifests for Claude, Codex and Pi?
4. **`best/` writer**: the controller as single writer using the configured promote check (proposed), or the world (needs bucket write access in the world)?
5. **Hook delivery**: at-least-once, batched, never blocking (proposed), or should some hooks be synchronous, e.g. `agent_assigned` waiting until the world has set the agent up, or a world able to veto a spawn?
6. **World token scopes**: are `spawn` and `grant` in v1, and which parent and limits does a world-initiated spawn count against?
7. **Multiple worlds per swarm**: v1 has one. Later `worlds: {name: ...}` with tools prefixed by world name; does a planned run need two before then?
8. **World versioning**: a relaunched world whose manifest drifted (new version, changed tool schemas): refuse, or accept when the change is additive?
9. **Live change for CLI harnesses mid-session**: Claude, Codex and Pi read `CLAUDE.md`/`AGENTS.md` at session start, and their processes live across turns. `flock-tool` sees new tools at once, but the model does not know them. Inject a "tools changed" note into the next prompt (proposed), restart the harness with resume at the next idle (cold start, lost prompt cache), or rely on MCP `tools/list_changed` (needs question 3)?
10. **Local workers vs job workers**: child `flock-worker` processes with the job env (proposed: same code path, harness processes run exactly as on jobs, needs the pinned Claude/Codex/Pi CLIs installed locally, `flock run` checks versions), or the worker inside the `flock run` process (faster start, diverges from jobs), or a local container for exact parity?
11. **Tool isolation on shared workers**: the tool host runs as the same user as the harnesses, so an agent with `bash` can read the bundle's tool source and possibly the tool host's env. A separate user for the tool host (needs the image to allow it), or accept in v1 and document that tools for `bash` agents hold no secrets the agents must not see? Module state is shared by all agents of a type on a worker: enough, or one tool host per agent?
12. **Secrets for tools**: names only in the spec (a type's `tool_secrets: [NAME]`, next to the existing `secrets:`), values from the shell env or `.env` locally and from job secrets on jobs, only the tool host gets them (proposed). Is reading `.env` acceptable, or must locals be exported explicitly?
13. **Grants from worlds**: limited to the type's `grantable:` list (proposed), or any tool in the bundle? Do grants expire (`ttl`)?
14. **Yardstick counting**: is "under 500 lines of task code" Python only (estimate ~464, 344 of it the unchanged sandbox and checker) or including prompts (~545)? Including prompts it is not reachable without changing the task.
15. **Live world changes**: a changed `world.dir` or `command` on a running swarm restarts the world (its in-memory state is lost unless it persists it): restart on apply, or refuse without `--restart-world`?
16. **Local run identity**: does a second `flock run spec.yaml` resume the previous run from `.flock/` (sessions, artifacts, board identities) or start a new run id unless `--resume` (proposed)?
