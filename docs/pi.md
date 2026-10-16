# Pi harness

`harness: pi` runs [Pi](https://github.com/earendil-works/pi)
(`@earendil-works/pi-coding-agent`, pinned to **1.2.0**) as a flock agent. Adapter:
`src/flock/worker/adapters/pi.py`. Status: verified locally against a fake OpenAI-compatible
server; **not yet run in an HF Job** (see "Not verified").

## Running a Pi type

Until PR A lands `models:` entries and `resolve_model()`, a pi type names its endpoint with a
temporary convention (`pi_model_config` in `worker/agents.py`, marked `TODO(PR A)`):

```yaml
types:
  analyst:
    harness: pi
    model: "openai_compat:https://my-endpoint.example/v1|Qwen/Qwen3.8-27B-FP8"
    env:
      PI_API_KEY_ENV: VLLM_API_KEY   # env var holding the key (default OPENAI_API_KEY)
      PI_REASONING: low              # optional: --thinking level
    secrets: [VLLM_API_KEY]          # the job receives it; only this name reaches Pi
    system_prompt_file: prompts/analyst.md   # written to the workdir as AGENTS.md
```

Equivalent: `env: {PI_BASE_URL: ..., PI_MODEL: ...}` with any `model`. Without either form,
`model` is passed to `pi --model` as given (one of Pi's built-in providers, e.g.
`anthropic/claude-haiku-5-5`; its key must be in `secrets`). Pi refuses to prompt when the key
variable is unset, so for a keyless endpoint set it to any value (e.g. in `env`); the adapter
reports that refusal as a failed turn.

## How it runs

- **Process**: one `pi --mode rpc` per burst, cwd = `/work/agents/<agent>`:
  `pi --mode rpc --session-dir /state/pi/sessions/<agent> --session-id <uuid> --model <provider>/<id>
  [--thinking <level>] --no-extensions --no-mcp --no-skills --no-prompt-templates --no-approve`.
  After spawning: `set_steering_mode: all`, then `get_state` (its `sessionFile` must be the
  expected file on resume).
- **Home**: `PI_CODING_AGENT_DIR=/state/pi`, one per worker (every agent on a worker has the
  same type). The adapter writes `/state/pi/models.json` atomically on every start: one provider
  (`flock`), `api: openai-completions`, `apiKey: "$<VAR>"`. The key is read from the
  environment at request time; it never lands on disk (`auth.json` stays `{}`).
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
- **Environment**: an allow-list, not the worker's environment: `PATH HOME USER SHELL LANG LC_*
  TERM TZ TMPDIR`, CA and proxy variables, `PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0`,
  `PI_CODING_AGENT_DIR`, the names in the type's `secrets`, the API key variable, then the type's
  `env` and the msgboard identity. `FLOCK_TOKEN` is always removed.

## Event mapping

| Pi record | AdapterEvent |
|---|---|
| `prompt` response `disposition: started`, or `agent_start`, with no flock turn open | `turn_started` |
| `message_end` (system, user, assistant, toolResult) | `message`, the Pi message unchanged (already pi v3) |
| `message_end`, role assistant, non-zero usage | `usage`, model `<provider>/<id>`, per call (`cumulative=False`) |
| `compaction_end` with `result.usage` | `usage`, per call (from the docs; not observed) |
| `agent_settled` | `turn_ended` (completed, error or interrupted) |
| `auto_retry_end {success: false}`, `extension_error`, an unsolicited failed `response` | `error` |
| a `prompt` Pi rejects (e.g. no API key) | `turn_started` + `turn_ended` status `error` |
| stdout EOF | `exited` |
| `turn_start`/`turn_end` (one per model call), `message_start`/`message_update`, `tool_execution_*`, `queue_update`, `agent_end` | ignored |

Usage: Pi's `input` excludes cached tokens and `reasoning` is already inside `output`, which is
what `TokenUsage` expects. Pi's `cost` is ignored; the controller prices tokens.

## Verified (Pi 1.2.0, Node 22.23.3, local, fake server)

Recorded in `tests/fixtures/pi_rpc.jsonl` (re-record with `tests/fixtures/record_pi_rpc.py`) and
exercised by `tests/test_worker_pi.py` (run with `FLOCK_PI_CLI=<path to pi> pytest`):
start, a tool call, two prompts queued while a tool runs (one turn, one model call), a 400 from
the endpoint (`turn_ended` error, not retried), abort mid-tool, resume from a fresh agent dir
holding only the copied session file (the next request carries the full history; the same file
is appended to), `ResumeError` for a missing file or one from another cwd, `--thinking high`
reaching the endpoint as `reasoning_effort: high`, the key arriving as `Bearer <value>` from the
env reference, AGENTS.md in the system prompt, no `FLOCK_TOKEN` in the tool's environment, tool
processes gone after `stop()` (graceful and after a hung Pi is killed). `experiments/resume-gate-pi`
passed as a local dry run. Idle VmRSS of one RPC process: 117 MB; startup to the first RPC
response 0.4 to 3.3 s on a busy shared dev box.

## Not verified

- **Anything in an HF Job**: the resume gate (`experiments/resume-gate-pi`, ready, not run), Pi
  as root, startup and memory on cpu-basic, packing (`agents_per_worker` and
  `max_working_per_worker` for pi types still use the Claude-measured defaults).
- **Compaction**: never triggered (no long session); its usage mapping follows the docs. Pi's
  default `contextWindow` for a custom model was 128000 in `get_state`.
- `auto_retry_*` events (the fake server's 400 was not retried; a 5xx/429 path was not tried).
- A `prompt` that lands between `agent_end` and `agent_settled`.
- A real model or vLLM endpoint; tool-call parsing is the server's job.
- Pi 1.3.0 (released 2026-10-15) is untested; the version is part of the resume contract, so a
  bump needs the fixture re-recorded and the gate re-run.
