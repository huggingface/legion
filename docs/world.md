# Worlds

A world is shared state behind tools: a task board, a canvas, a game. It is an HTTP service
outside flock, in any language. The spec names it, and its agents get the world's tools next to
`bash` and their folder tools. Each call reaches the world with the calling agent's name and a
shared secret. flock starts the world (`compute: local`) or connects to one already running
(`url:`). It fetches the world's manifest at every apply and hands the tools to the agents. flock
never reads or stores the world's state.

Use a [tools folder](tools.md) when each worker can do the job alone. Use a world when the agents
must see the same state, across workers (claims, a shared board), at 5 agents or 1,000.

```yaml
# spec.yaml
world:
  dir: ../../worlds/taskboard                       # where `command` runs, relative to the spec
  command: [flock, world, serve, "taskboard:TaskBoard"]
types:
  worker:
    harness: llm
    model: fake:fake-model
    replicas: 2                                     # every tool of the world, or:
    # tools: [world:list_tasks, world:claim_task]   # only these
```

A runnable version is [`examples/world-demo/`](../examples/world-demo/spec.yaml). It works offline
with the fake model server from the tests. Its world is [`worlds/taskboard/`](../worlds/taskboard/taskboard.py):
a shared task board in SQLite with `list_tasks`, `add_task`, `claim_task`, `finish_task` and
`release_task`, where a claim is exclusive.

## Spec

| key | what |
|---|---|
| `world.command` | Starts the world (a list, or a string split like a shell). flock fills in `{port}` (a free port) and `{python}` (flock's own interpreter). `compute: local` only. |
| `world.dir` | Where `command` runs, relative to the spec file. Defaults to the spec's directory. |
| `world.url` | A world that is already running (a Job, a Space, your laptop). Required with `compute: hf-jobs`. With `compute: local` it is used when there is no `command`. |
| `world.token_env` | Names the secret sent as `X-World-Token`. It is read from the controller's environment, like provider keys, and handed to the workers. A started world without it gets a generated secret. |
| `types.<t>.tools: [world:<name>, ...]` | Only these world tools for the type. Without `world:` entries, the type gets every tool in the manifest. |
| `types.<t>.world: false` | No world tools for this type. |

The controller writes `world.endpoint` (the URL it calls) and `world.manifest` into the spec it
stores. `GET /swarms/{swarm}` shows them, and the run bundle (`run.json`) keeps them. The secret is never
part of the spec.

One block works at both scales. `command` is used locally and `url` on HF Jobs, so a spec can
carry both, and `compute:` is the only line that changes.

## What flock does

**At apply** (`flock apply`, `flock run`, any re-apply):

1. With `compute: local` and a `command`, the controller starts the world as a child process. It
   runs in `dir` with these variables set: `FLOCK_WORLD_PORT`, `FLOCK_WORLD_HOST=127.0.0.1`,
   `FLOCK_WORLD_TOKEN` (the secret) and `FLOCK_SWARM`. It also inherits the controller's
   environment, minus every `FLOCK_*` variable. The controller waits up to 30 s for
   `GET /manifest`. A re-apply keeps the running world unless `command`, `dir`, the secret or a
   `*.py` file under `dir` changed. In that case the replacement starts next to the running world
   on another port, and the running world stops only after the replacement passed every check
   below and the spec was stored. A world that exited is started again.
2. It calls `GET /manifest` with the secret and validates it: `world_api: v1`, a name, a version,
   unique tool names that are not `bash` or `done`, and object schemas.
3. It checks the types. Every listed `world:<name>` must exist, and no world tool a type gets may
   share a name with one of its folder or comms tools.

If any step fails, the apply is refused with HTTP 400 and nothing is applied. A replacement
that was started is stopped, and the running world and the stored spec stay as they were. The
message says why, for example:

```
world: cannot reach http://127.0.0.1:8770/manifest: ConnectError: [Errno 111] Connection refused
world: http://w:8000 refused the token (401): check world.token_env
world: invalid manifest from http://w:8000: tools.0.name: Value error, tool name 'done' is reserved (...)
world.command exited with 1 before serving GET /manifest; its log (.flock/<run>/logs/world-<swarm>.log) ends with: ...
types.worker.tools: unknown world tool(s) fly; world taskboard has: list_tasks, add_task, ...
```

A started world stops when its swarm is removed (`flock remove`), when its block is removed, and
at the end of `flock run`, after the workers. Stopping sends SIGTERM to the world's whole process
group (it runs in its own session). After 5 s, SIGKILL goes to whatever is left in the group,
including helpers that outlived the main process. Its output goes to
`<run>/logs/world-<swarm>.log` under `flock run`; a standalone local controller lets it inherit
its own output.

**In the workers.** The world block, with its manifest, comes with every assign and heartbeat
response.

- **The model sees** the world's tools exactly like folder tools:
  - the `llm` harness offers them as function tools next to `bash` (and `done`);
  - the CLI harnesses list them in the generated Tools section of `CLAUDE.md` / `AGENTS.md`, and
    the agent calls them with `flock-tool <name> --json '<args>'` through the worker's local
    gateway.
- **The World section.** A `## World: <name> (version X)` section in the instructions names the
  world and its tools, with the manifest's `description`.
- **Calls.** The arguments are checked against the tool's schema before the world is called. A
  call goes to `POST <endpoint>/tools/<name>` with the secret and the agent's identity. A `429`
  or `503` is retried twice (`Retry-After`, at most 10 s). There is a 60 s timeout and a 1 MB
  result limit. Every failure is a tool result the model reads (`{"error": "..."}`); the turn
  goes on.
- **Changes reach the next turn.** A re-apply re-fetches the manifest, and the next turn sees
  the new tools. If the version changed, that turn's World section also carries a note: "the
  world changed since your last turn (version 0.1.0 -> 0.2.0)".
- **A turn is pinned.** A turn keeps the configuration it started with until it ends: the
  endpoint, the manifest's schemas, the selection, and whether the worker's secret belongs to
  them. This holds for in-process calls and for `flock-tool` calls through the gateway (the
  pinned tool names the configuration).
- **The secret stays in the worker.** The worker gets it as the job secret `FLOCK_WORLD_TOKEN`,
  which every worker of a swarm with a world receives at launch. It is removed from the harness
  environment, even if a type lists it in `secrets`, and every tool call is denied it.
- **One secret per world.** The world block never carries the secret. It names the secret by
  `credential_origin` and `credential_id`:
  - `credential_origin` is the endpoint, or `local:<swarm>` for a world flock started, which is
    always on 127.0.0.1;
  - `credential_id` is HMAC(secret, origin).

  A worker sends its secret only when its own HMAC for that origin matches. After a re-apply
  that points the world at another endpoint or secret, the worker's calls under the new
  configuration are refused ("world ... now uses another secret or endpoint; this worker
  restarts ..."), and nothing is sent. The worker then drains, and the controller relaunches it
  with the new secret. Turns pinned to the old configuration finish against the old world.
  Restarting a started world keeps its origin and secret, so its workers keep running.

## Protocol: World API v1

HTTP and JSON. A world is any server that answers these routes:

| route | body | answer |
|---|---|---|
| `GET /manifest` | | the manifest (below); also the health check |
| `POST /tools/<name>` | `{"args": {...}}` | `{"ok": true, "result": <any JSON>}` or `{"ok": false, "error": {"code": "...", "message": "..."}}` |
| `POST /hooks` | `{"event", "agent", "type", "swarm", "turn", "data"}` | `{"ok": true}`; sent only for the events the manifest lists |

| header | sent | meaning |
|---|---|---|
| `X-World-Token` | every request | the shared secret; answer `401` if it is wrong |
| `X-Flock-Agent` | tool calls | the calling agent (`worker-0001`); trust it when the secret matches |
| `X-Flock-Type`, `X-Flock-Swarm` | tool calls | its type and swarm |
| `X-Flock-Turn` | tool calls | the agent's turn (`t-3`), when it is in one |
| `X-Request-Id` | tool calls | unique per call; log it with what the call changed |

The manifest:

```json
{"world_api": "v1", "name": "taskboard", "version": "0.1.0",
 "description": "A shared task board. Claim a task before you work on it ...",
 "tools": [{"name": "claim_task", "description": "Claim an open task ...",
            "parameters": {"type": "object", "properties": {"name": {"type": "string"}},
                           "required": ["name"], "additionalProperties": false}}],
 "capabilities": ["hooks"], "hooks": ["agent_done"]}
```

- `name` matches `[a-z][a-z0-9-]{0,31}`. `version` is any string; change it when the tools
  change, and the agents are told.
- Each tool is a Chat Completions function: `name`, `description`, and `parameters` (a JSON
  schema of type object).
- Unknown fields are ignored. flock ignores capabilities it does not use, so a world can declare
  more.
- A tool's error is a `200` with `ok: false`; the agent reads `message`. Use `4xx` and `5xx` for
  protocol failures: `401` for the secret, `404` for an unknown tool, `429` or `503` with
  `Retry-After` when overloaded.
- `result` goes to the model as is if it is a string, otherwise as JSON. If `result` is absent,
  a `text` string is used instead.
- Hooks are sent by the worker, one request per event, in the background, with a 5 s timeout and
  no retry:
  - `agent_start`: an agent's first prompt on a worker (again after a relaunch);
  - `turn`: every turn end, with `{status, tool_calls}`;
  - `agent_done`: the agent was unassigned (scale down, removal), with `{reason}`.

  Treat them as hints. A world must stay correct if one is lost.

## Write your own world

**In Python**, with the `flock.world` base. Mark tool methods with `@tool`. Schemas come from the
type hints and the docstring, with the [tools folder rules](tools.md#contract). A parameter named
`agent` (or annotated `Agent`) receives the caller and is never shown to the model.

```python
# worlds/counter/counter.py
from flock.world import Agent, ToolError, World, tool


class Counter(World):
    name, version = "counter", "0.1.0"
    description = "One shared counter."

    def __init__(self) -> None:
        self.n = 0

    @tool
    def bump(self, agent: Agent, by: int = 1) -> dict:
        """Add `by` to the shared counter and return the new value.

        Args:
            by: how much to add (at least 1)
        """
        if by < 1:
            raise ToolError("by must be >= 1")      # the agent reads this as the error
        self.n += by
        return {"n": self.n, "last": agent.name}

    def on_agent_done(self, agent: Agent, data: dict) -> None:   # optional hook
        ...
```

```sh
cd worlds/counter && FLOCK_WORLD_TOKEN=dev flock world serve counter:Counter   # :8765
curl -s -H 'X-World-Token: dev' localhost:8765/manifest
curl -s -H 'X-World-Token: dev' -H 'X-Flock-Agent: me' -d '{"args": {"by": 2}}' \
  localhost:8765/tools/bump        # {"ok":true,"result":{"n":2,"last":"me"}}
```

`flock world serve module:Class` imports the module from the current directory. It serves on
`$FLOCK_WORLD_HOST:$FLOCK_WORLD_PORT` (default `127.0.0.1:8765`, or `--host` / `--port`) and
requires `$FLOCK_WORLD_TOKEN` (`--token-env NAME` for another variable; `--insecure` for none).
Synchronous tools and hooks run one at a time, so the world needs no locks of its own;
`async def` tools and hooks run concurrently. A tool that raises anything other than `ToolError`
returns `{"code": "internal"}` with the exception text. Overriding `manifest()` and `call(tool,
args, agent)` serves tools that are not methods. The server checks each call against
`manifest()` as served at that moment, so what `GET /manifest` advertises is exactly what
`POST /tools/<name>` accepts, also after the manifest changes. It needs the `controller` extra (FastAPI, uvicorn),
which `flock run` needs too.

**In any other language:** serve the three routes above. A world without hooks needs only two.

## Local and HF Jobs

| | `compute: local` (`flock run`) | `compute: hf-jobs` |
|---|---|---|
| who runs the world | the controller, from `command` in `dir` | you: a Job, a Space or any host the workers can reach; give `url` |
| port | chosen by flock (`{port}`, `$FLOCK_WORLD_PORT`), on 127.0.0.1 | yours |
| secret | generated (or `token_env`) | `token_env`, set on the controller; the controller passes it to the workers as a job secret |
| lifetime | stops with the swarm or the run | yours |

Starting the world as an HF Job from `command` (with the job's image and flavor, in the
controller's network group) is not in this version. Run it yourself and set `url`. A world in a
Space is public, so the secret is its only protection.

## Not in this version

- Starting a world as an HF Job.
- `flock doctor --world <url>` (a conformance and load check).
- The world → flock direction: a world token to prompt, stop or grant agents.
- Optional capabilities with routes of their own (`state`, `score`, `claims`, `rounds`,
  `events`, `render`, `checks`). Loop settings that need them (`loop.while.score_*`,
  `every: rounds`, `on: world.*`) are still refused at apply.
- Per-service rate budgets: a world call is not rate-limited by flock beyond the 429 / 503
  retries.
- Batching and delivery guarantees for hooks.

## Local clone uploads (workspace-upload-v1)

A world can advertise `workspace-upload-v1` and implement
`call_workspace(name, args, agent, snapshot)`. For `open_pr(path=...)` and
`run_bot(ref_or_path=<local path>)`, the worker captures a git clone rooted inside
the caller's workdir. It forwards `{args: {...}, workspace: {base_sha, files}}` to
the existing authenticated tool route; the file map contains base64 bytes. The
upload is outside the model's tool arguments, so the model supplies neither a
server path nor a base SHA. `main` and `pr:<number>` use no upload.

Only tracked and nonignored files are captured; missing tracked files represent
deletions. Symlinks, paths outside the agent workspace, git metadata, more than
10,000 files or more than 128 MiB are refused. The base is the merge base of local
HEAD and origin/main; the world validates it against its own history. Commit
local edits and fetch/rebase before refreshing a stale PR. Upload-capable calls
have a 330-second HTTP read timeout so a 300-second conversation can finish.
Other worlds keep their existing timeout and dispatch behavior.

The chatbot4 world preserves the seed input/output and module loader, runs the
bot under stdlib/import and side-effect restrictions with resource limits, and
requires a different agent's review of the exact PR head. It refuses stale bases
instead of rebasing automatically; refreshing a PR always requires fresh review.
`prs.jsonl` records PR events with both the main parent and PR head for merges.
`eval-queue.jsonl` is an append-only hook for the separate held-out evaluator;
this implementation does not start an evaluator.
