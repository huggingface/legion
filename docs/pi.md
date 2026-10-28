# Pi harness

`harness: pi` runs [Pi](https://github.com/earendil-works/pi)
(`@earendil-works/pi-coding-agent`, pinned to **1.2.0**) as a flock agent. Adapter:
`src/flock/worker/adapters/pi.py`. Status: verified locally against a fake OpenAI-compatible
server; the resume gate passed in two cpu-basic HF Jobs (2026-10-16,
experiments/resume-gate-pi/RESULTS.md); no real model yet (see "Not verified").

## Running a Pi type

A pi type names its model the way codex does (design.md section 5): a `<provider>:<id>` string
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
  `error_max_turns` / `error_max_budget_usd`, as Claude reports them). A limit reached on a
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
  `types.<T>.env.<KEY>: dropped by pi isolation` for the type's `env` keys that would be dropped.

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
  packing (`agents_per_worker` and `max_working_per_worker` for pi types still use the
  Claude-measured defaults) is unmeasured.
- **Compaction**: never triggered (no long session); its usage mapping follows the docs. Pi's
  default `contextWindow` for a custom model was 128000 in `get_state`.
- `auto_retry_*` events (the fake server's 400 was not retried; a 5xx/429 path was not tried).
- A `prompt` that lands between `agent_end` and `agent_settled`.
- A real model or vLLM endpoint; tool-call parsing is the server's job.
- Pi 1.3.0 (released 2026-10-15) exists and is untested (not upgraded); the version is part of the resume contract, so a
  bump needs the fixture re-recorded and the gate re-run.
