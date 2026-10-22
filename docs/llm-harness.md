# The `llm` harness

`harness: llm` is a plain API loop for small open models: no coding-agent CLI, no external
process. The worker calls an OpenAI-compatible Chat Completions endpoint, runs the model's
`bash` tool calls in the agent's workdir, and sends the results back until the model answers in
text. Code: `src/flock/worker/adapters/llm.py`. Design inspired by swarmlab's `LLMAgent`; no
swarmlab dependency. Example: `examples/llm-small-model.yaml` (LiquidAI/LFM2.5-2.6B on vLLM).

## When to use it

- Small models (2-10B) that Claude Code, Codex or Pi drown in long system prompts and many tools.
  The `llm` system prompt is a short preamble plus your `AGENTS.md`; there is one tool.
- High call volume (e.g. a pixel swarm doing about 1M short calls): no process start per burst
  (an idle agent costs a JSONL file; a working one an asyncio task), so a worker holds many more
  agents than with a CLI harness.
- Servers that only speak Chat Completions (Codex needs the Responses API).

Use Claude Code, Codex or Pi when the agent must edit code well: they have file-edit tools,
model-written compaction, parallel tool calls, prompt caching strategies and years of prompt
tuning that this loop does not.

## Spec

```yaml
providers:
  lfm: {type: openai_compat, base_url: "https://lfm.example.org/v1", api_key_env: VLLM_API_KEY,
        self_hosted: true, timeout_s: 60, max_retries: 4}
models:
  lfm-small: {model: "lfm:LiquidAI/LFM2.5-2.6B",
              extra: {max_tokens: 256, chat_template_kwargs: {enable_thinking: false}}}
types:
  pixel: {harness: llm, model: lfm-small, tool_choice: bash, max_turns: 8, max_budget_usd: 0.01,
          context_tokens: 16000}
```

- A provider is required (`openai_compat` only; `anthropic` or a native model id is rejected at
  apply). The key stays in the provider's `api_key_env` secret, never in the spec.
- `max_turns`: model calls per flock prompt. `max_budget_usd`: per prompt, priced with the
  type's flock price table (the pinned provider price), as for Codex and Pi.
- `context_tokens` (new, `llm` only, default 32000, minimum 256): the prompt size (tokens) the
  harness keeps each request under, by compacting before the call (see "Context bounding").
  Set it well below the server's `--max-model-len` minus `max_tokens`.
- `chars_per_token` (new, `llm` only, default 2.5, 0.5-8): the starting characters-per-token
  for size estimates; calibrated online from the server's reported prompt tokens.
- `tool_output_chars` (new, `llm` only, default 8000, minimum 200): bash output beyond this is
  cut (head and tail kept, with a `[... N chars truncated ...]` marker) before it is stored or
  sent.
- `extra` from the alias is merged into every request body (this harness can; the others
  cannot). It may override the defaults `temperature: 0` and `max_tokens: 4096`. `stream` is
  dropped: requests are never streamed.
- `tool_choice` (new, `llm` only): `auto` (default), `required`, `none`, or `bash` (force that
  tool by name). It wins over a `tool_choice` in `extra`. A forcing value (`required`, `bash`)
  makes every prompt **one action**: one forced call, its tool run, turn `completed` (plus one
  more forced step per prompt steered in mid-turn), because a forced model can never answer in
  text. Use it for small models: on LFM2.5-2.6B (vLLM, L4) a named tool choice gave 20/20 valid
  calls in ~2.5 s and ~22 output tokens, against 7-15/20 in ~25 s free-form
  (flock-board/probe-reachability-lfm.md).
- `done_tool` (new, `llm` only; default on when `tool_choice` is `required`, off otherwise): offer
  a second tool, `done {summary?: string}`. With it, a forced turn runs several steps: the model
  must call `bash` or `done` on every request, and calling `done` ends the turn `completed` with
  the summary as the final assistant message (calls after `done` in the same reply are skipped).
  `max_turns` still caps model calls (`reason: max_turns`); with no `max_turns`, a forced turn
  with `done` is capped at 50 calls so a model that never calls `done` cannot loop until
  `turn_timeout`. If a reply in such a turn has no tool call at all (a provider dropped it: an
  empty assistant message, seen with Qwen3.5-9B right after a tool result), the harness adds one
  user message, "Reply with exactly one tool call.", and asks again. The retry is a model call
  and counts toward `max_turns`. A second miss in a row ends the turn `completed`, as before.
  `tool_choice: bash` offers only `bash` (one action per prompt) unless
  `done_tool: true`. A `{"tool": "done", ...}` reply written as text counts too. With the type's
  `loop.stop_when: {all_agents_done: true}` (docs/loop.md), `done` is offered whatever the
  `tool_choice` (unless `done_tool: false`, which `flock apply` refuses with it) and takes
  `finished?: boolean`: `done(finished: true)` takes the agent out of its loop.
- `bash` (`llm` only; default on): `false` takes the `bash` tool away, leaving the folder and
  comms tools (and `done`); the preamble then says the agent has no shell. Usually switched
  live with `flock grant <target> bash=off` ([docs/live.md](live.md)); refused with
  `tool_choice: bash`.
- `response_format` goes in `extra` and is sent as is, nested objects included. A JSON schema
  shaped like `{"tool": "bash", "command": ...}` makes every reply a call through the text
  fallback below; any other schema'd reply is the turn's final answer.
- `reasoning` is ignored. Use `extra` with the server's own field: for thinking models on vLLM
  (LFM2.5, Qwen3) `chat_template_kwargs: {enable_thinking: false}`; `reasoning_effort` on the
  router.
- `timeout_s` (default 120) is the per-request timeout; `max_retries` (default 3) bounds retries
  on HTTP 429, 5xx and transport errors, with jittered exponential backoff from 1 s (capped at
  30 s; `Retry-After` honoured up to that cap). Other 4xx fail the turn at once.

### Painter pattern (multi-step forced turns)

```yaml
pixel: {harness: llm, model: lfm-small, tool_choice: required, max_turns: 12}  # done_tool on
```

In `AGENTS.md`: "Each step is one chained shell command (read the canvas, pick a pixel, paint
it, e.g. `curl -s .../canvas && curl -s -X POST .../paint -d ...`). Call `done` when the canvas
says nothing is left, or after N paints." The model can never ramble in text (every reply is a
call, ~25 tokens), and the turn ends on `done` or at `max_turns`.

## Thinking models (LFM2.5, Qwen3)

LFM2.5's chat template ends the prompt with `<think>`, so the model always thinks first; free
form, it spends 800-1024 tokens (~25 s) before answering. Serve it with vLLM
`--reasoning-parser qwen3` (Liquid's recommendation; also `--tool-call-parser lfm2`) so the
reasoning arrives as `reasoning_content`, and switch thinking off or force a tool as above. The
harness tolerates both forms: `reasoning_content` / `reasoning`, or inline `<think>...</think>`
in the content (also `...</think>` without the opening tag, and a reply cut off inside
`<think>`). The thinking becomes a pi `thinking` block in the event stream; it is stripped from
the stored transcript and never sent back. Its tokens are counted (they are in the server's
`completion_tokens`).

The LFM template also needs exactly one system message, first. The harness keeps it that way:
the system prompt is the only system message; compaction notes and steered prompts are user
messages.

## How a turn runs

A flock prompt is one turn. The loop appends the prompt as a user message, then repeats: check
limits, compact if needed, call the model, store and emit the reply, run its tool calls, append
any steered prompts. It stops when a reply has no tool call and nothing was steered in.

| Adapter event | When |
|---|---|
| `turn_started` | the loop takes a prompt while idle |
| `message` (pi format) | each user prompt, assistant reply (`toolCall` blocks, `thinking` from `reasoning_content` or `<think>`) and `toolResult` |
| `usage` | after every model call, per call (not cumulative); `input` excludes cached prompt tokens, keyed by the model id |
| `turn_ended completed` | a reply without tool calls and no queued prompt; a `done` call (its summary is emitted first as the final assistant message, `fromDoneTool: true`); or, forced `tool_choice` without `done`, after the forced call's tool ran. `data.num_turns` = model calls |
| `turn_ended error`, `reason: max_turns` / `max_budget_usd` | before a call past a limit (the last tool results are kept; the next prompt continues the transcript) |
| `turn_ended error`, `reason: api_error` | a 4xx, or retries exhausted |
| `turn_ended error`, `reason: context_overflow` | a context-length 400 that compaction plus one retry could not fix |
| `turn_ended interrupted` | `stop` mid-turn (timeout, stop, drain); the tool's process group is killed |
| `error` | every non-2xx reply from the model server: `HTTP <status> from <url>: <first 500 chars of the body>` (keys and bearer tokens redacted), `data` = `{status, attempt, retrying}`; retries and turn handling are unchanged. Also an internal failure, which ends the burst |
| `exited` | once per burst, on `stop` (or after an internal failure) |

Steering: a prompt sent during a turn is queued and becomes the next user message after the
current tool results, inside the same turn (several queued prompts go in together).

## Tool

Task tools from the swarm's tools folder (`types.<t>.tools`, [docs/tools.md](tools.md)) are
offered next to `bash` as function tools, taken once per turn, and run in a subprocess each;
the rest of this section is about `bash`.

One tool, `bash` (plus `done` when `done_tool` is on, see above) (`command`, optional `timeout` seconds, default 120, max 600): `bash -c` in the
workdir, stdin closed, stdout and stderr combined, output truncated to `tool_output_chars`
(default 8000) characters (head and tail) plus `[exit code N]`; a non-zero exit is `isError`. It runs in its own process group,
killed on timeout or stop. Its environment is the shared allow-list (`PATH`, `HOME`, locale,
proxies, ...) plus what the worker passes (the type's `secrets`, `env`, msgboard identity,
`FLOCK_AGENT*`), minus `FLOCK_TOKEN` and minus the provider key.

Small models often write the call as text instead of `tool_calls`. The loop accepts a reply
that is only (after stripping a code fence or `<tool_call>` / `<|tool_call_start|>` tags):

- `{"tool": "bash", "command": "..."}` (what the preamble asks for), or
  `{"name": "bash", "arguments": {"command": "..."}}`;
- a pythonic call list, `[bash(command="...")]` (LFM2's syntax; parsed with `ast`, literals
  only, nothing evaluated).

Anything else, including prose that contains such JSON, is a final answer. A parsed call is
stored as a native `tool_calls` entry (`flock_from_text` keeps the original text), so later
requests show the model the proper form; the pi message carries `toolCallsFromText: true`.

## Session, resume, compaction

- Session file: `$FLOCK_LLM_HOME/sessions/<agent>/<session id>.jsonl` (`/state/llm` in the
  worker image), one Chat Completions message per line, appended as the turn runs. It is the
  adapter's only `session_files()` entry, synced to the bucket like Pi's.
- Resume reloads it (a missing file is a `ResumeError`: fresh session plus the restart prompt).
  The system message is rebuilt on every start from the preamble and the workdir's `AGENTS.md`,
  so prompt changes apply to old sessions. Tool calls left without a result by an interrupted
  turn get a `[interrupted: no result]` tool message, so the server accepts the transcript.
- Compaction (see "Context bounding" below) rewrites the file; the full history stays in the
  agent's flock session log (`_sessions/<agent>.jsonl`).

## Context bounding

Pixel run 3 (2026-10-20) had `context_tokens: 3000`, yet painter prompts reached 7,937 tokens
and vLLM refused 2,376 calls (0.64% of 369k) with "maximum context length is 8192". There were
two causes. The size estimate was characters / 4, but JSON tool output runs about 2
characters per token (a 616-character paint result cost ~312 tokens). And compaction could not
shrink a single long turn: it only dropped whole earlier prompts. Now:

1. **Projection from the server's count.** Before each call the next prompt is projected as the
   previous call's reported `prompt_tokens` (cached included) plus the characters added since,
   divided by `chars_per_token`. With no previous call in this burst, or after a compaction, it
   is the characters / `chars_per_token` estimate. Over `context_tokens`, the transcript is
   compacted before sending.
2. **Calibration.** Each reply's `prompt_tokens` against the characters sent updates
   `chars_per_token`: an exponential moving average, weight 0.3, bounded to 1-6. It starts at
   the type's `chars_per_token` (default 2.5).
3. **Compaction, to half of `context_tokens`.** It keeps the system message and one note (a user
   message saying how many earlier messages were removed). It then drops, in order:
   - whole earlier prompts;
   - the oldest steps of the current turn (an assistant message with its tool results, never
     split; the newest two steps and the turn's prompt are always kept);
   - text from old tool outputs, cut to 1000 characters.
   There is no model-written summary.
4. **Context-length errors.** A 400 whose body mentions the context or maximum length triggers
   an aggressive compaction (only the newest prompt and the newest two steps), and the call is
   retried once. If that is impossible or fails again, the turn ends `error` with reason
   **`context_overflow`**. Other 4xx still end with `api_error`.
5. **Capped tool output.** Tool output is capped at `tool_output_chars` before it is stored.

## Preflight

Types on `llm` are probed on `POST {base_url}/chat/completions`, non-streamed, with the alias's
`extra` (minus `response_format`, which describes the agent's output, not the probe's) and the
type's `tool_choice` (a named tool forces the probe's `get_time`): turn one asks for a tool
call, turn two sends the tool result back with `tool_choice: auto` and must complete. A tool
call written as text (after any `</think>`) passes with a warning. The same model on a Codex type is
probed separately on the Responses API.

## What it deliberately lacks

No streaming, no parallel tool execution, no file read/edit tools (bash covers `cat`, heredocs,
`sed`), no model-written compaction, no prompt-cache control, no images, no MCP, no reasoning
setting, no tokenizer (the context projection uses the server's reported counts plus a
calibrated chars/token estimate). Packing (agents per worker) is not
measured yet.

## What a small model still needs from the prompt

The harness cannot make a 2.6B model agentic on its own. In `AGENTS.md`: say exactly what one
prompt should achieve and what the final text answer looks like; give literal command examples
for the few operations the agent needs (e.g. the exact `curl` to post to msgboard); say to stop
after the task and answer in one line; keep it short (it is resent on every call). Set
`max_turns` low (5-10) so a looping model fails fast, cap `extra.max_tokens`, and prefer
`tool_choice: bash` (or a `response_format` schema) over free-form replies: constrained output
is what makes a 2.6B thinking model fast and reliable. With a forced tool, the prompt must say
which single command to run for this step; the agent never gets to "think out loud" in text.
