# Pi harness

`harness: pi` runs [Pi](https://github.com/earendil-works/pi)
(`@earendil-works/pi-coding-agent`, pinned to **1.2.0**) as a flock agent. Adapter:
`src/flock/worker/adapters/pi.py`. Status: verified locally against a fake OpenAI-compatible
server; the resume gate passed in two cpu-basic HF Jobs (2026-10-16,
experiments/resume-gate-pi/RESULTS.md); no real model yet (see "Not verified").

## Running a Pi type

A pi type names its model as design.md section 5 says: a `<provider>:<id>` string
or a `models:` alias, resolved at apply; Pi needs an `openai_compat` provider.

```yaml
providers:
  gpu:
    type: openai_compat
    base_url: https://my-endpoint.example/v1
    api_key_env: VLLM_API_KEY      # the job receives it; the worker hands it to Pi by name
    self_hosted: true
types:
  analyst:
    harness: pi
    model: "gpu:Qwen/Qwen3.8-27B-FP8"
    reasoning: low                 # optional: --thinking level
    system_prompt_file: prompts/analyst.md   # written to the workdir as AGENTS.md
```

A model without a provider prefix is passed to `pi --model` as given (one of Pi's built-in
providers, e.g. `anthropic/claude-haiku-5-5`; its key must be in `secrets`). Pi refuses to
prompt when the key variable is unset, so for a keyless endpoint set it to any value (e.g. in
`env`); the adapter reports that refusal as a failed turn.

## How it runs

- **Process**: one `pi --mode rpc` per burst, cwd = `/work/agents/<agent>`:
  `pi --mode rpc --session-dir /state/pi/sessions/<agent> --session-id <uuid> --model <provider>/<id>
  [--thinking <level>] --no-extensions --no-mcp --no-skills --no-prompt-templates --no-approve`.
  After spawning: `set_steering_mode: all`, then `get_state` (its `sessionFile` must be the
  expected file on resume).
- **Home**: `PI_CODING_AGENT_DIR=/state/pi`, one per worker (every agent on a worker has the
  same type). The adapter writes `/state/pi/models.json` atomically on every start: one provider
  (the flock provider name), `api: openai-completions`, `apiKey: "$<VAR>"`. The key is read from the
  environment at request time; it never lands on disk (`auth.json` stays `{}`).
  With multiple provider `base_urls`, each agent selects an endpoint by a stable
  name hash and uses `/state/pi/agents/<name>` as its own config home. This prevents
  different agents from overwriting each other's endpoint in models.json; session
  files are still synced relative to the adapter's home for restore.
- **System prompt**: `AGENTS.md` in the workdir. Pi also reads AGENTS.md/CLAUDE.md from parent
  directories and its agent dir, so `/work`, `/work/agents` and `/state/pi` must not have one.
- **Prompts**: `prompt` with `streamingBehavior: "steer"`. Idle: starts a run
  (`disposition: started`). Busy: queued (`queued`) and delivered after the running tool calls,
  in the same run; with steering mode `all`, several queued prompts go out in one model call.
- **Turns**: a flock turn opens on the prompt response `started` (or `agent_start`) and ends on
  `agent_settled`: `interrupted` if `aborted`, `error` if the last assistant message has
  `stopReason: error` (its `errorMessage` becomes the error), else `completed`.
- **Stop**: `abort` if a run is in flight (Pi writes "Command aborted" as the tool result and
  `agent_settled {aborted: true}`), close stdin, wait 5 s, then SIGKILL. Closing stdin makes Pi
  exit at once and kill the tool's process group, backgrounded children included. A SIGKILLed Pi
  does not: the tool's bash runs in its own session and survives, so `stop()` snapshots Pi's
  descendants from `/proc` first and SIGKILLs any that are still alive afterwards. A process
  that double-forks out of the tree before the snapshot is not caught.
- **Resume**: `session_files()` is the one file `/state/pi/sessions/<agent>/<ts>_<sid>.jsonl`.
  Before spawning with a session id, the adapter requires a file for that id whose header `cwd`
  is the workdir; otherwise it raises `ResumeError`, because Pi would silently create a second
  file with the same id (it does so whenever the cwd differs). The worker then starts a fresh
  session and points the agent at the old transcript, as for the other harnesses.
- **Limits**: `max_turns` and `max_budget_usd` are enforced by the adapter, per flock turn
  (which may fold several steered prompts). It counts model calls (`turn_start`) and prices each
  assistant `message_end` with the flock price table (`price_for`, the same lookup the
  controller bills with; unknown models at the high default row). When Pi would call the model
  again past a limit (an answer that asks for tools, or a `turn_start` for a steered prompt
  after a final answer) it sends `abort`: the tools of that answer do not run (or are killed),
  and the turn ends `error` with `data.reason` `max_turns` or `max_budget_usd` (`subtype`
  `error_max_turns` / `error_max_budget_usd`). A limit reached on a
  final answer ends the turn normally. A steered prompt that arrives after the last allowed
  answer can have its request already on the wire when the abort lands (one extra call, at
  most). Pinned provider prices: the worker passes `type_price_table(spec)`, so a
  provider-backed type is priced at its pinned price.
- **Environment**: the shared `minimal_env` (`adapters/base.py`: `PATH HOME USER LOGNAME SHELL
  TERM LANG LANGUAGE TZ TMPDIR IS_SANDBOX LC_* XDG_*`, CA and proxy variables) plus Pi's own
  variables, then what `AgentRuntime.harness_env` selects for every harness: the names in the
  type's `secrets`, the provider's key variable, the type's `env` and the msgboard identity.
  `PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0` unless set; `PI_CODING_AGENT_DIR`.
  `FLOCK_TOKEN` is always removed, even when listed.
  With isolation on, the allow-list applies to the inherited worker environment only: the
  type's `env` keys pass as written, except names ending in `TOKEN` or `SECRET`, `HF_TOKEN`,
  `FLOCK_TOKEN`, `FLOCK_WORLD_TOKEN` and the worker's `FLOCK_HARNESS_ENV_DROP` (put credentials in
  `secrets`, which stay allow-listed). The dropped names (never values) are logged and emitted
  once per agent per spec revision as an `env_dropped {names}` event, and `flock update` warns
  `types.<T>.env.<KEY>: dropped by pi isolation` for the type's `env` keys that would be dropped
  (by the same function; names only the worker's `FLOCK_HARNESS_ENV_DROP` lists are not previewed).
- **Controller calls**: Pi's `flock` CLI (`flock-agent spawn`, `flock prompt`, ...) goes through the
  worker gateway named in the pin file (`FLOCK_TOOLS_PIN`), not to `FLOCK_URL`: the worker
  sends it with the agent's token to wherever the controller is now, so a controller move no
  longer breaks a Pi that started before it (docs/spawn.md, "CLI inside an agent").
  `FLOCK_URL` and `FLOCK_AGENT_TOKEN` are still in Pi's environment for the direct fallback;
  they can be dropped once every worker opens the gateway.

## Memory

A Pi process otherwise lives as long as the agent stays on the worker (a loop's next prompt
arrives before `idle_grace`), so its heap, session and sandbox root grow for the whole job.

- **Recycle**: the runtime restarts Pi at a turn boundary, resuming the same session
  (`--session-id`): the next prompt goes to a fresh process, the event is
  `harness_recycled {reason, pss_mb}`, and no turn is interrupted. Reasons: `memory` from the worker's
  relief ladder (docs/slurm.md "Memory"), and `endpoint`: a live change of the model endpoint,
  model id or reasoning level (`PiAdapter.update_spec`) applies within one turn.
- **OOM**: a Pi SIGKILLed mid-turn (exit -9/137: the OOM killer) ends the turn
  `turn_interrupted {reason: oom_killed}`; a loop retries it as a transient failure (backoff,
  not counted toward the three failed turns).
- **Tool limits**: with `tool_memory_mb` (or `node_heap_mb`) set, the adapter writes a
  read-only file `<pi home>/../<agent>.bash_env` (bound read-only inside the sandbox) and
  passes it as `BASH_ENV`, which every non-interactive bash (Pi's bash tool) sources:
  `ulimit -d <tool_memory_mb × 1024>` (RLIMIT_DATA: V8 and Python reserve far more virtual
  memory than they use, so not `-v`) and `unset NODE_OPTIONS`. Every tool command and its
  children (git, pytest, node) inherit the limit; Pi does not. `BASH_ENV` passes the isolation
  allow-list.
- **Pi's heap**: `node_heap_mb` sets `NODE_OPTIONS=--max-old-space-size=<MB>` for Pi only.
- **OOM order**: the sandbox writes 800 to `/proc/self/oom_score_adj` before exec (raising
  needs no privilege; the worker keeps its own score), so the kernel kills a tool or a Pi
  before the worker. Under `memory.oom.group=1` the kernel kills the whole cgroup anyway.
- **Per-agent cgroups**: not used. Under cgroup v2 they need the worker to move itself into a
  leaf cgroup first.

## Workdir

The workdir is mirrored to the bucket and restored whenever the agent moves to another worker,
so its size is restore time. Regenerable trees are never synced (`node_modules/`, `.venv/`,
`venv/`, `.cache/`, `.tox/`, `.nox/`, `.gradle/`, `.next/`, `.ipynb_checkpoints/`,
`__pycache__/`, `*.pyc`, `*.o`, ...; `.git/` and worktrees are). The type's `workdir:` block
gives the agent feedback instead of refusing anything:

```yaml
types:
  coder:
    workdir:
      soft_files: 50000   # default; above it (or soft_mb) the agent's next prompt starts with
      soft_mb: 2048       # "your workspace has 182k files, 3100 MB; largest: snaps/ 140k, ...
                          # (soft limit 50k files, 2048 MB); delete clones you no longer need",
                          # at most hourly
      max_files: null     # default 5 x soft: above it (or max_mb) a `workdir_over_quota
      max_mb: null        # {files, bytes, top_dirs}` event too, same hourly limit
      ephemeral: [scratch/]
      base: false             # true: restore from a tar plus per-file deltas (below)
      base_interval_s: 3600   # a new base at most this old ...
      base_delta_files: 5000  # ... or when this many files changed or were deleted since
```

The workdir still syncs above every limit. `ephemeral` lists directories relative to the
workdir that are never synced nor restored, and are emptied (or created) at every agent setup,
so a move or a worker restart starts them empty. Tell the agents in their system prompt, e.g.
"clone repositories you only need for this session into `scratch/`; it is not saved when you
move". A loop prompt can show the current size with `{workspace}` (docs/loop.md).

A restore downloads every mirrored file, about 85 files/s per agent in run 4, so a 100k-file
workdir took around 20 minutes. With `base: true` the worker keeps a tar of the workdir
(`workdir.base.tar`, with modes, symlinks and mtimes) plus only the files changed since it, and a
restore downloads the tar, then those files. Periodic syncs refresh the tar hourly or after 5000
changed files; drain and preempt syncs only send the changes, as before. Agents whose bucket has
no tar yet restore file by file and get one at their first periodic sync; turning `base` off
again goes back to the plain mirror at the next sync. The tar is written to local disk before
upload and downloaded before extraction, so a worker needs free disk for one workdir's size per
agent being synced or restored. Details: `flock.worker.sync`.

## Event mapping

| Pi record | AdapterEvent |
|---|---|
| `prompt` response `disposition: started`, or `agent_start`, with no flock turn open | `turn_started` |
| `message_end` (system, user, assistant, toolResult) | `message`, the Pi message unchanged (already pi v3) |
| `message_end`, role assistant, non-zero usage | `usage`, model `<provider>/<id>`, per call (`cumulative=False`) |
| `compaction_end` with `result.usage` | `usage`, per call (from the docs; not observed) |
| `agent_settled` | `turn_ended` (completed, error or interrupted; error with `reason` after a limit abort) |
| `auto_retry_end {success: false}`, `extension_error`, an unsolicited failed `response` | `error` |
| a `prompt` Pi rejects (e.g. no API key) | `turn_started` + `turn_ended` status `error` |
| stdout EOF | `exited` |
| `turn_start` (one per model call) | counted for `max_turns`; no event |
| `turn_end`, `message_start`/`message_update`, `tool_execution_*`, `queue_update`, `agent_end` | ignored |

Usage: Pi's `input` excludes cached tokens and `reasoning` is already inside `output`, which is
what `TokenUsage` expects. Pi's `cost` is ignored; the controller prices tokens. The usage key
stays `<provider>/<id>`; `price_for` tries the full id, then the part after the first `/`
(exact, then prefix), so `anthropic/claude-haiku-5-5` gets the `claude-haiku-5-5` row and a price
pinned under the bare model id wins.

## Verified (Pi 1.2.0, Node 22.23.3, local, fake server)

Recorded in `tests/fixtures/pi_rpc.jsonl` (re-record with `tests/fixtures/record_pi_rpc.py`) and
exercised by `tests/test_worker_pi.py` (run with `FLOCK_PI_CLI=<path to pi> pytest`):
start, a tool call, two prompts queued while a tool runs (one turn, one model call), a 400 from
the endpoint (`turn_ended` error, not retried), abort mid-tool, resume from a fresh agent dir
holding only the copied session file (the next request carries the full history; the same file
is appended to), `ResumeError` for a missing file or one from another cwd, `--thinking high`
reaching the endpoint as `reasoning_effort: high`, the key arriving as `Bearer <value>` from the
env reference, AGENTS.md in the system prompt, no `FLOCK_TOKEN` in the tool's environment, tool
processes gone after `stop()` (graceful and after a hung Pi is killed), `max_turns=1` and
`max_budget_usd` stopping a tool prompt after one model call (the session stays usable).
`experiments/resume-gate-pi` passed as a local dry run and in two cpu-basic jobs; it and the
tests share one fake server, `tests/fake_openai.py` (`common/launch.sh` copies it next to the
job scripts). Idle VmRSS of one RPC process: 117 MB; startup to the first RPC
response 0.4 to 3.3 s on a busy shared dev box.

## Not verified

- **Workers in an HF Job**: only the resume gate ran (as root, startup 0.4 s, idle VmRSS 107 MB);
  packing (`agents_per_worker` and `max_working_per_worker` for pi types still use defaults
  measured on Claude Code) is unmeasured.
- **Compaction**: never triggered (no long session); its usage mapping follows the docs. Pi's
  default `contextWindow` for a custom model was 128000 in `get_state`.
- `auto_retry_*` events (the fake server's 400 was not retried; a 5xx/429 path was not tried).
- A `prompt` that lands between `agent_end` and `agent_settled`.
- A real model or vLLM endpoint; tool-call parsing is the server's job.
- Pi 1.3.0 (released 2026-10-15) exists and is untested (not upgraded); the version is part of the resume contract, so a
  bump needs the fixture re-recorded and the gate re-run.
