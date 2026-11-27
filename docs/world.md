# Worlds

A world is shared state behind tools: a task board, a canvas, a game. It is an HTTP service
outside legion, in any language. Each call reaches the world with the calling agent's name and a
secret. legion never reads or stores the world's state.

Harbor agents reach a world through the `world` resource and its `world.*` tools: the run's
`resources: {world: {url, token_env}}` and `tools: [world.*]`
(experiments/pybot/spec.yaml; `legion.harbor.builtin.resources.world`). The resource
adds the secret and the agent's name on the worker, so no world credential enters a Trial's
container (on hf-sandbox each agent gets its own agent token instead; with a gateway, see "Gateway").

The swarm-level `world:` block is the controller's side: it starts the world (`compute: local`)
or connects to one already running (`url:`), and fetches and checks its manifest at every apply.
A Harbor spec has no `world:` block: it is reachable only from a hand-written swarm spec. The rest of this page up to "Protocol" is about that block.

```yaml
# spec.yaml
world:
  dir: counter                                      # where `command` runs, relative to the spec
  command: ["{python}", -m, legion.world, serve, "counter:Counter"]
```

`counter/counter.py` is the one-tool world under "Write your own world" below.

## Spec

| key | what |
|---|---|
| `world.command` | Starts the world (a list, or a string split like a shell). legion fills in `{port}` (a free port) and `{python}` (legion's own interpreter). The controller starts it with `compute: local`; with `compute: slurm` the run's world job does (docs/slurm.md "Deploying"), next to `url`. |
| `world.dir` | Where `command` runs, relative to the spec file. Defaults to the spec's directory. |
| `world.url` | A world that is already running (a Job, a Space, your laptop). Required with `compute: hf-jobs`. With `compute: local` it is used when there is no `command`. |
| `world.token_env` | Names the secret sent as `X-World-Token`. It is read from the controller's environment, like provider keys; workers do not get this one (the Harbor world resource's own `token_env` is read on the worker). A started world without it gets a generated secret. |

The controller writes `world.endpoint` (the URL it calls) and `world.manifest` into the spec it
stores. `GET /swarms/{swarm}` shows them, and the run bundle (`run.json`) keeps them. The secret is never
part of the spec.

One block works at both scales. `command` is used locally and `url` on HF Jobs, so a spec can
carry both, and `compute:` is the only line that changes.

## What legion does

**At apply** (`legion launch`, any re-apply):

1. With `compute: local` and a `command`, the controller starts the world as a child process. It
   runs in `dir` with these variables set: `LEGION_WORLD_PORT`, `LEGION_WORLD_HOST=127.0.0.1`,
   `LEGION_WORLD_TOKEN` (the secret) and `LEGION_SWARM`. It also inherits the controller's
   environment, minus every `LEGION_*` variable. The controller waits up to 30 s for
   `GET /manifest`. A re-apply keeps the running world unless `command`, `dir`, the secret or a
   `*.py` file under `dir` changed. In that case the replacement starts next to the running world
   on another port, and the running world stops only after the replacement passed every check
   below and the spec was stored. A world that exited is started again.
2. It calls `GET /manifest` with the secret and validates it: `world_api: v1`, a name, a version,
   unique tool names, and object schemas.
3. A type's `tools` entries `world:<name>` must name tools of the manifest.

If any step fails, the apply is refused with HTTP 400 and nothing is applied. A replacement
that was started is stopped, and the running world and the stored spec stay as they were. The
message says why, for example:

```
world: cannot reach http://127.0.0.1:8770/manifest: ConnectError: [Errno 111] Connection refused
world: http://w:8000 refused the token (401): check world.token_env
world: invalid manifest from http://w:8000: tools.0.name: Value error, tool name '1x' must match (...)
world.command exited with 1 before serving GET /manifest; its log (.legion/<run>/logs/world-<swarm>.log) ends with: ...
```

A started world stops when its swarm is removed (`legion stop --delete`), when its block is removed, and
when the local controller ends (`legion stop --release`), after the workers. Stopping sends SIGTERM to the world's whole process
group (it runs in its own session). After 5 s, SIGKILL goes to whatever is left in the group,
including helpers that outlived the main process. Its output goes to
`<run>/logs/world-<swarm>.log` under a `legion launch` run; a standalone local controller lets it inherit
its own output.

## Protocol: World API v1

HTTP and JSON. A world is any server that answers these routes:

| route | body | answer |
|---|---|---|
| `GET /manifest` | | the manifest (below); also the health check |
| `POST /tools/<name>` | `{"args": {...}}` | `{"ok": true, "result": <any JSON>}` or `{"ok": false, "error": {"code": "...", "message": "..."}}` |
| `POST /hooks` | `{"event", "agent", "type", "swarm", "turn", "data"}` | `{"ok": true}`; not sent by legion now (below) |
| `GET /metrics` | | optional, capability `metrics-v1`: load and health numbers (below) |
| `GET /healthz` | | optional, no secret: `{"status": "ready"}` or `{"status": "parked"}` |
| `/admin/*` | | optional, capability `admin-v1`, the admin secret: knobs and lane pauses (below) |

| header | sent | meaning |
|---|---|---|
| `X-World-Token` | every request | the shared secret; answer `401` if it is wrong. `legion.world.server` also takes, on `/manifest` and tool calls, the agent's own token `agent_token(secret, X-Legion-Agent)` (protocol.py), so an agent that holds it acts only as itself |
| `X-Legion-Agent` | tool calls | the calling agent (`worker-0001`); trust it when the secret matches |
| `X-Request-Id` | tool calls | unique per call; log it with what the call changed |

The manifest:

```json
{"world_api": "v1", "name": "counter", "version": "0.1.0",
 "description": "One shared counter.",
 "tools": [{"name": "bump", "description": "Add `by` to the shared counter ...",
            "parameters": {"type": "object", "properties": {"by": {"type": "integer"}},
                           "additionalProperties": false}}],
 "capabilities": ["hooks"], "hooks": ["agent_done"]}
```

- `name` matches `[a-z][a-z0-9-]{0,31}`. `version` is any string; change it when the tools
  change.
- Each tool is a Chat Completions function: `name`, `description`, and `parameters` (a JSON
  schema of type object). Optional `annotations: {"readOnlyHint": true}` marks a tool that
  changes nothing.
- Unknown fields are ignored. legion ignores capabilities it does not use, so a world can declare
  more.
- A tool's error is a `200` with `ok: false`; the agent reads `message`. An optional
  `error.retry_after_s` says the call cannot succeed for that many seconds (a budget, a busy
  backend). Use `4xx` and `5xx` for protocol failures: `401` for the secret, `404` for an
  unknown tool, `429` or `503` with `Retry-After` when overloaded (`python -m legion.world serve` answers
  `{"ok": false, "error": {"code": "overloaded", "message", "retry_after_s"}}`).
- `result` goes to the model as is if it is a string, otherwise as JSON. If `result` is absent,
  a `text` string is used instead.
- Hooks (`agent_start`, `turn`, `agent_done`) were sent by the worker's world hub, which is
  gone: Harbor's world resource sends tool calls only. A world must stay correct without them.

## Write your own world

**In Python**, with the `legion.world` base. Mark tool methods with `@tool`. Schemas come from the
type hints and the docstring (`legion.tools.loader`). A parameter named
`agent` (or annotated `Agent`) receives the caller and is never shown to the model.

```python
# counter/counter.py
from legion.world import Agent, ToolError, World, tool


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
cd counter && LEGION_WORLD_TOKEN=dev python -m legion.world serve counter:Counter   # :8765
curl -s -H 'X-World-Token: dev' localhost:8765/manifest
curl -s -H 'X-World-Token: dev' -H 'X-Legion-Agent: me' -d '{"args": {"by": 2}}' \
  localhost:8765/tools/bump        # {"ok":true,"result":{"n":2,"last":"me"}}
```

`python -m legion.world serve module:Class` imports the module from the current directory. It serves on
`$LEGION_WORLD_HOST:$LEGION_WORLD_PORT` (default `127.0.0.1:8765`, or `--host` / `--port`) and
requires `$LEGION_WORLD_TOKEN` (`--token-env NAME` for another variable; `--insecure` for none).
Synchronous tools and hooks run one at a time, so the world needs no locks of its own
(`concurrent = True`: they run at once, and the world guards its own state); `async def` tools
and hooks run concurrently. Synchronous tools run in one of two thread pools:
`@tool(pool="slow")` for tools that block for seconds or minutes, the "fast" pool (default) for
the rest and the hooks, so slow calls never hold up quick ones. `@tool(read_only=True)` marks a
tool that changes nothing (`annotations.readOnlyHint`). `GET /manifest` is served from a cached
copy on the event loop, rebuilt when the manifest's version changes, so it answers while every
thread is busy. A tool that raises anything other than `ToolError`
returns `{"code": "internal"}` with the exception text. Overriding `manifest()` and `call(tool,
args, agent)` serves tools that are not methods. The server checks each call against
`manifest()` as served at that moment, so what `GET /manifest` advertises is exactly what
`POST /tools/<name>` accepts, also after the manifest changes. It needs the `controller` extra (FastAPI, uvicorn),
which `legion launch` needs too.

`raise ToolError(message, code, retry_after_s=N)` adds `retry_after_s` to the error, for a
refusal that will not change for a while (a used-up budget, a busy backend). The message stays what the agent reads.

**Admission control.** The server refuses a tool call with `429`, `Retry-After` and
`code: "overloaded"` when its pool is overloaded: the fast pool's oldest queued call has
waited more than `LEGION_WORLD_FAST_MAX_WAIT_S` (2 s), or the slow pool's running and queued
calls reach its 64 threads plus `LEGION_WORLD_SLOW_QUEUE` (8). `retry_after_s` is
`LEGION_WORLD_RETRY_AFTER_S` (30). The decision is made from the path before the body is read,
so a refused upload is drained and never decoded. `GET /manifest`, `GET /metrics` and the
hooks are never refused.

**Metrics.** `GET /metrics` (the same token) answers JSON, or Prometheus text with
`Accept: text/plain` (`legion_world_<name>`, labels `tool`, `pool`, `code`). Windowed values
cover the last `window_s` (300 s):

```json
{"uptime_s": 812.4, "rss_mb": 412.0, "cpu_pct": 37.5,
 "loop_lag_s": 0.002, "loop_lag_max_s": 0.08, "overloaded_429_total": 3,
 "upload_v1_total": 2, "upload_v2_total": 408, "window_s": 300.0,
 "tools": {"open_pr": {"calls": 41, "errors_by_code": {"open_budget": 6}, "latency_p50_s": 9.1,
                       "latency_p95_s": 31.0, "latency_max_s": 48.2, "bytes_in": 120000000,
                       "bytes_out": 52000, "calls_total": 310, "errors_total": 44}},
 "pools": {"fast": {"in_flight": 2, "queued": 0, "limit": 40, "queue_wait_oldest_s": 0.0,
                    "queue_wait_p95_s": 0.0, "queue_wait_max_s": 0.01, "overloaded_429_total": 0},
           "slow": {"in_flight": 70, "queued": 6, "limit": 64, "queue_wait_oldest_s": 4.2,
                    "queue_wait_p95_s": 3.0, "queue_wait_max_s": 9.5, "overloaded_429_total": 3}},
 "world": {"open_prs": 57}}
```

`tools` lists the tools called since start: `calls`, `errors_by_code` (refusals included as
`overloaded`), latencies (admitted calls only, queue wait included) and bytes are over the
window; `calls_total` and `errors_total` since start, as are `upload_v1_total` and `upload_v2_total`
(workspace uploads by format). `pools.*.in_flight` counts running and
queued calls; `pools.*.overloaded_429_total` the refusals since start, by pool (each pool also
logs one WARNING at most every 10 s with the reason, the tool and how many it refused since). `loop_lag_s` is how late a once-a-second timer fired. `world` is the world's own
`metrics() -> dict[str, float]` (non-numbers are dropped), called in a thread on every scrape,
concurrently with tools: keep it cheap.

**Saving state.** A world that keeps its state in memory can save it in O(changes) with
`legion.world.store`, a `(kind, key) -> JSON` table on SQLite in WAL mode:

```python
from legion.world.store import Store

store = Store(state_dir / "state.db")
prs = dict(store.iter("prs"))                   # hot rows, at start
prs["7"] = {...}; store.dirty("prs", "7")       # at every change
store.flush(lambda kind, key: prs.get(key), lock)  # saver: dirty rows, one transaction; None deletes
store.put("prs", "3", rec, cold=True)           # out of the hot set; get() and iter(cold=True) still read it
store.backup_to(path)                           # consistent copy while others write (archives)
```

`lock` (what guards your dicts) is held only while the dirty rows are read; the commit runs
after, and a row changed meanwhile stays dirty. A failed flush writes nothing and keeps the
rows dirty for the next one. `close()` checkpoints the WAL into the database file. `backup_to(db_file,
path)` (module function) copies a database another process is writing, for archive scripts.

**Lanes for external writes.** A world that writes to an outside service with one token (a
Git host, an API) can give each kind of write its own lane, from `legion.world.lanes`:

```python
from legion.world.lanes import Lane, LaneBusy, Lanes

lanes = Lanes({"push": Lane(slots=1), "merge": Lane(slots=1),
               "comment": Lane(slots=1, gap_s=1.0)}, dir=state_dir, paused=lambda lane: ...)
try:
    with lanes.hold("push", deadline=time.monotonic() + 20) as slot:
        ...                                  # the write
        slot.ok()                            # or slot.fail("transient") / slot.fail("final")
except LaneBusy as busy:                     # no slot by the deadline: busy.reason, busy.retry_after_s
    raise ToolError("busy; retry later", "busy", retry_after_s=busy.retry_after_s)
lanes.outbox("comment", drain=post).put(key=pr, payload=text)   # post(key, payloads) -> "ok"|"transient"|"final"
```

- A lane runs at most `slots` writes at once, `gap_s` apart. The world decides which failures
  are `transient` (the service is overloaded) and which `final` (retrying cannot help).
- Breaker: 3 transient failures within 15 minutes trip the lane to one slot and at least 5 s
  between writes; after 10 minutes without one it returns to its settings
  (`Lane(trip_after=, trip_window_s=, trip_gap_s=, quiet_s=)`). `Lanes(breaker_groups=[("push",
  "merge")])` gives lanes one shared breaker: a failure in either trips both.
- Pause: `lane.paused = True` or the `paused(lane)` callback; `hold()` then waits until its
  deadline and raises `LaneBusy` (`pausable=False` ignores the pause).
- `lane.slots`, `lane.gap_s` and `lane.paused` are plain attributes; `lane.configure(slots=,
  gap_s=)` changes them and wakes waiters.
- Outbox: writes that must happen eventually. Stored in `<dir>/outbox-<lane>.jsonl` (archive it
  with the world state), drained by a background thread holding the lane's slots; payloads
  queued for the same key while it waits go out as one write (`drain` gets the list, up to
  `max_batch` payloads and `max_chars` characters). Transient failures retry forever with
  doubling backoff (5 s to 10 min); only "ok" and "final" remove payloads; "split" retries
  that key's payloads one per write. `lanes.close()` stops the thread; the file keeps the rest.
- `lanes.metrics()` per lane: `slots`, `effective_slots`, `in_flight`, `waits_total`,
  `wait_s_total`, `ok`, `transient`, `final`, `tripped`, `paused`, `outbox_depth`,
  `outbox_oldest_age_s`.

**Parked mode.** `python -m legion.world serve` binds its port first and builds the world in a background
thread. While that raises (a failed restore, the Hub unreachable, a bad config), it is
*parked*: it logs the error and builds again after 30 s, doubling up to 10 min, forever.
`GET /manifest`, tool calls and hooks then answer `503` with `Retry-After` and

```json
{"ok": false, "parked": true, "error": "OSError: restore failed: ...", "attempt": 3, "next_retry_s": 120}
```

(`error`: the exception's first line, secrets redacted). `GET /healthz` (no secret) says
`parked` or `ready`; `/metrics` adds `parked` (0 or 1), `attempt` and `paused` (paused lanes);
`/admin` works. At an apply, a parked world at the endpoint the stored spec already has keeps
the stored manifest; one that was never applied refuses the apply with `world parked ...`.
The SDK has no separate startup phase: do restores and first fetches in `__init__`, and a
failure there parks the world.

**Admin API: knobs and lane pauses.** A world registers runtime knobs in `__init__` and reads
them later; a lane is a kind of work operators may pause (for example `push` and `merge`):

```python
class Shop(World):
    def __init__(self) -> None:
        # int, float, bool or str; returns the current value (the stored one if set, else default)
        self.slots = self.knob("hub.push.slots", int, int(os.environ.get("SLOTS", "1")),
                               min=0, max=4, doc="Concurrent Hub writes.",
                               on_change=lambda v: setattr(self, "slots", v))

    @tool
    def push(self) -> str:
        if self.paused("push"):  # paused through POST /admin/pause
            raise ToolError("pushes are paused; keep working locally", "paused")
        return f"pushing with {self.knobs['hub.push.slots']} slots"
```

`on_change(value)` runs in a thread (raise to refuse); without one, read `self.knobs[name]`.
Values set through the API and the paused lanes are written to `knobs.json` in the world's
state dir (`World.state_dir()`: `$LEGION_WORLD_STATE_DIR` by default) and read back at the next start, so a requeued world keeps them;
the defaults a world passes (usually from its environment) apply only to knobs never set.
The manifest lists the knob names under `knobs` and adds the capability `admin-v1`.

| route | body | answer |
|---|---|---|
| `GET /admin/knobs` | | `{name: {value, default, type, min, max, doc}}` |
| `PATCH /admin/knobs` | `{name: value, ...}` | `{"ok": true, "changes": [{name, old, new}], "knobs": {...}}`; `400` if any value is unknown, of the wrong type or out of range (nothing applied) |
| `GET /admin/paused` | | `{"parked": false, "paused": {lane: reason}}` |
| `POST /admin/pause` | `{"lanes": [...], "reason": "..."}` | `{"ok": true, "paused": {lane: reason}}` |
| `POST /admin/resume` | `{"lanes": [...]}` | the same |

`/admin/*` takes the admin secret in `X-World-Token`, never the world secret (that gets `401`):
`$LEGION_WORLD_ADMIN_TOKEN` in the world's environment; unset, `/admin` answers `403`. It is
never derived from the world secret. Bodies are capped at 64 KB, and
admission control never applies to `/admin`. Operators go through the controller, which sends
the same secret and records each knob change as a swarm event (`world_knob {name, old, new,
by, reason}`):

- a world legion starts (`command`, `compute: local`): the controller derives the secret from
  its own token and the swarm and starts the world with it in `$LEGION_WORLD_ADMIN_TOKEN`;
- any other world (`url`; Slurm, a Space): set `LEGION_WORLD_ADMIN_TOKEN` in the world's
  environment and in the controller's secrets (the run's secrets file on Slurm,
  `legion init --secrets`), and keep it from the workers.

```bash
python -m legion.world knobs <swarm>                                   # values, defaults, ranges, paused lanes
python -m legion.world set <swarm> pr.open_floor=10 --reason "hub 429s"
```

**In any other language:** serve the three routes above. A world without hooks needs only two.

## Gateway (e2b, hf-sandbox)

On e2b and hf-sandbox, `resources.gateway` keeps every service key on the worker: a Trial's
sandbox holds only a random token of its own and the gateway's URL, and the gateway adds the keys.

```yaml
compute: hf-jobs
environment:
  backend: e2b             # or hf-sandbox
resources:
  gateway: {}             # port: 8790, the worker job's port the controller exposes publicly
  world:
    url: '${WORLD_URL}'
    transport: gateway    # or an env world ("Env worlds" below)
```

- It runs in each worker job, at `https://<job id>--8790.hf.jobs`. The spec allows it only with
  a sandbox backend (`e2b`, `hf-sandbox`) and `compute: hf-jobs`, and `transport: gateway` only
  with a gateway.
- Each Trial gets a new token when it starts; its end, the next Trial's token and the agent
  leaving the worker revoke it. The gateway names the agent from the token, never from its headers. Agents
  send it as `X-World-Token` (Pi too, through its models.json headers): the jobs proxy drops
  `Authorization` on public ports (docs/open-models.md).
- `POST /inference/chat/completions` goes to the agent's own endpoint (`inference`) with its key
  and the router's bill-to header. Only the agent's model is allowed (403), in at most 16 MiB
  (413). The answer streams through; the gateway asks for usage on the last chunk and counts each call into the agent's
  tokens as it ends, so `legion status` and `limits.budget_usd` see a turn's cost while it runs
  (Pi's turn-end usage is then not counted again); a call the agent drops (a killed turn) ends
  upstream too, its usage unknown. Once the agent's cost reaches its
  `loop.while.agent_usd`, further calls get 402, except in its final round.
- `GET /world/manifest` and `POST /world/tools/<name>` go to the world with the world secret and
  the agent's name, after the socket proxy's checks (no `/hooks`, `/metrics`, `/admin`, dot
  segments or queries; the same size cap), or to an env world ("Env worlds" below).
- Any other path is 404, a call without a live token 401, a ninth call in flight 429. One log
  line per call (refusals at WARNING); the export `resources/gateway/usage.json` has each agent's
  calls and tokens.
- Sandboxes keep the open web. No board route: the board stays on `transport: http`.

### Env worlds

An upstream HTTP API that gives each player its own credential (a game, a simulator) can be the
world without code: the world resource's `env` block, in place of `url`, turns each route it
names into a tool, and the gateway forwards an agent's call with that agent's own credential.
There is no world server; agents keep the `world.*` tools.

```yaml
naming: consecutive-id                     # agent-001 has seat 0, agent-002 seat 1, ...
resources:
  gateway: {}
  world:
    transport: gateway
    env:
      name: antbench
      description: You control one ant of a swarm that must carry a load past two walls together.
      upstream: 'http://$${HF_NETWORK_GROUP_PREFIX}antbench:8000'   # in the run's network group
      secret_env: ANTBENCH_ENV_SECRET      # the run's secret, a new one per run
      credential_env: 'ANT_TOKEN_{seat}'   # how the launcher hands each seat its credential
      headers:
        Authorization: Bearer {credential}
      tools:
        observe:
          method: GET
          path: /v1/observe
          query: [run_id, after_seq, limit]  # the rest of the arguments go in a JSON body
          read_only: true
          description: Your ant's latest observation, phase and run_id ...
          parameters:
            type: object
            properties:
              run_id:
                type: string
              after_seq:
                type: integer
              limit:
                type: integer
        # ready, action, advertise: the same shape
```

You run the env process (the upstream) yourself, through legion's launcher:

```sh
ANTBENCH_ENV_SECRET=... python -m legion.harbor.env spec.yaml -- python ant_service.py
```

- Credentials: an agent's is an HMAC of the run's secret (`secret_env`) and its name. Each
  gateway derives it when needed. The launcher reads the env block and the roster from the spec,
  sets `credential_env` for each seat (`{seat}` 0, 1, ...), drops the secret from the command's
  environment and runs it. Workers get the secret as a job secret; give it to the env process
  and nothing else. Use a new secret per run: the same secret and name give the same credential.
  An agent spawned later has a credential the env never got, so its calls fail upstream.
- The env must not be public. On HF Jobs, run it in the run's network group
  (`--network-group $LEGION_NETWORK_GROUP --network-alias <alias>`, no exposed port) and point
  `upstream` at `http://$${HF_NETWORK_GROUP_PREFIX}<alias>:<port>`: `$${` keeps the `${...}`
  through `legion launch`, and the worker fills it from its own environment.
- Each header value may hold `{credential}`; one must. Paths are fixed: agents choose only the
  arguments their schema lists, never the route, a header or the seat. Routes the block does not
  name (for example a control route) are not reachable. Arguments are checked against the
  schema (`bad_args`), other tools are `unknown_tool`, and a body is at most 1 MiB.
- A 2xx answer is the tool's result as the upstream sent it (JSON, else text). Any other status
  is the error `upstream_<status>` with the upstream's body; `Retry-After` becomes
  `retry_after_s`. The secret and the worker's agents' credentials are redacted from results
  and errors, and none is logged.
- The manifest's version is a hash of the tools, so it changes when they do. An env world has
  no health check, metrics or hooks; anything else the env needs (a start deadline, saving a
  result) is the env job's own, as in the AntBench experiment's world job
  (experiments/antbench/README.md).

## Checklist for world authors

1. List the agent's workflow verbs (find, inspect, act, verify, follow up). Each needs a tool, and each write a read that shows its result.
2. Serve reads from the world's own mirror, with a freshness field. Agents never need the service's credentials or a fragile transport.
3. Mark read tools `read_only` (manifest `annotations.readOnlyHint`): they go to the fast pool, and admission control treats them separately.
4. Bound results: byte caps, pagination, narrowing hints.
5. Stable error codes plus `retry_after_s`; make writes idempotent on `X-Request-Id` where the backend allows.
6. Inject the backend, and test with a fake.
7. Pass `python -m legion.world doctor <url> [--token T] [--calls N]`. It warns about reads without the hint, writes with no read on the same noun and read tools with no limit/page argument, lists the capabilities, checks `/healthz` and `/metrics`, and with `--calls N` runs N concurrent calls of each read-only tool with empty args (p50/p95, error codes, `/manifest` latency meanwhile). It exits 1 when the manifest is unreachable or a read-only tool errors; warnings exit 0. Tools with required arguments are not called.

## Local and HF Jobs

| | `compute: local` (`legion launch`) | `compute: hf-jobs` |
|---|---|---|
| who runs the world | the controller, from `command` in `dir` | you: a Job, a Space or any host the workers can reach; give `url` (and the run's `resources.world.url`) |
| port | chosen by legion (`{port}`, `$LEGION_WORLD_PORT`), on 127.0.0.1 | yours |
| secret | generated (or `token_env`) | `token_env`, set on the controller; the world resource reads its own `token_env` on the worker |
| lifetime | stops with the swarm or the run | yours |

Starting the world as an HF Job from `command` (with the job's image and flavor, in the
controller's network group) is not in this version. Run it yourself and set `url`. A world in a
Space is public, so the secret is its only protection.

## Not in this version

- Starting a world as an HF Job.
- `legion doctor --world <url>` (a conformance and load check).
- The world → legion direction: a world token to prompt, stop or grant agents.
- Optional capabilities with routes of their own (`state`, `score`, `claims`, `rounds`,
  `events`, `render`, `checks`). Loop settings that need them (`loop.while.score_*`,
  `every: rounds`, `on: world.*`) are still refused at apply.
- Per-service rate budgets: the world resource caps one agent's calls in flight (8), nothing
  more.

## Local clone uploads (workspace-upload-v1, -v2)

A world can advertise `workspace-upload-v1` and implement
`call_workspace(name, args, agent, snapshot)`. A tool opts in with
`x-legion-workspace` on its manifest entry:
`{"arg": "<argument>", "default": "<value when absent>", "skip": ["<value>", ...], "skip_prefix": "<prefix>"}`
(only `arg` is required). For a call to that tool, the agent's `world.*` tool captures the
git clone named by that argument, rooted inside the agent's workspace, unless the value is in
`skip` or starts with `skip_prefix`. It forwards `{args: {...}, workspace: {base_sha, files}}` to
the existing authenticated tool route; the file map contains base64 bytes. The
upload is outside the model's tool arguments, so the model supplies neither a
server path nor a base SHA. For example, a tool that opens a pull request from the agent's
clone declares it on its `path` argument.

Only tracked and nonignored files are captured; missing tracked files represent
deletions. Symlinks, paths outside the agent workspace, git metadata, more than
10,000 files or more than 128 MiB are refused. The base is the merge base of local
HEAD and origin/main; the world validates it against its own history. Commit
local edits and fetch/rebase before refreshing a stale PR.

**Patch uploads (workspace-upload-v2).** A world that also advertises
`workspace-upload-v2` gets the same files as a patch instead: the tool hashes
them into a temporary index (`GIT_INDEX_FILE`, no clean filters or other
commands from the clone's config) and sends
`workspace: {base_sha, patch}`, where `patch` is the base64 of
`git diff --cached --binary --find-renames <base_sha>` (binary files, deletions,
renames and mode changes). A 3-file change in a 5,000-file clone is about 1 KB
instead of 3.8 MB. A patch over 20 MiB, or a clone v2 cannot capture, goes as v1.
The world applies it with `legion.world.workspace.apply_patch(snapshot, tree)` on
its own checkout of `base_sha` (worktree only, so its excludes apply when it
stages the tree); it refuses symlinks, submodules, paths outside the tree or in
`.git`, and more than 10,000 files or 128 MiB declared (`git apply` runs under
`prlimit`: 128 MiB per file, 2 GiB of memory). A world that cannot use a patch
but could use the full tree answers the error code `need_full` before doing
anything; the tool then resends the call once as v1. `GET /metrics` counts
`upload_v1_total` and `upload_v2_total`.

A tool's optional
`x-legion-timeout_s` raises the tool's HTTP read timeout for its calls (for example
330 seconds for a tool that may run for 300).
