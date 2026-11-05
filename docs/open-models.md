# Open models in flock

A swarm can run its agents on open-weight models: vLLM in a GPU job, the HF router, or any
server that speaks the API the harness speaks. You declare where models are served in a
`providers:` block and refer to them from a type's `model`. The contract is in
`docs/design.md` section 5 ("Providers"); this page is the how-to.

## The spec

```yaml
providers:                       # where models are served
  hf:   {type: openai_compat, base_url: https://router.huggingface.co/v1, api_key_env: HF_TOKEN,
         pricing: {"Qwen/Qwen3.8-27B:deepinfra": [0.2, 2.5, 0.2]}}  # USD per 1M: input, output, cached
  gpu1: {type: openai_compat, base_url: "https://qwen.example.org/v1",  # your vLLM, over HTTPS
         api_key_env: VLLM_API_KEY,
         self_hosted: true, max_working: 256}                 # self-hosted: price defaults to 0
  anth: {type: anthropic, api_key_env: ANTHROPIC_API_KEY}     # base_url optional, default Anthropic
models:                          # optional aliases and per-model settings
  qwen-27b: {model: "gpu1:Qwen/Qwen3.8-27B-FP8", reasoning: medium}
types:
  worker:  {harness: codex, model: "hf:Qwen/Qwen3.8-27B:deepinfra"}  # provider-prefixed, router provider pinned
  analyst: {harness: codex, model: qwen-27b}                  # alias into models:
  writer:  {harness: claude, model: claude-opus-5-5}          # harness-native id, as before
```

`examples/open-models.yaml` is a complete spec. A type's `model` is looked up as an alias first,
then as `<provider>:<model id>` (split on the first colon, so `hf:Qwen/Qwen3.8-27B:ovhcloud`
sends `Qwen/Qwen3.8-27B:ovhcloud` to the router), then taken as a harness-native id. Field names
match swarmlab's providers, so a `providers:` block can be copied between the two.

`flock launch` rejects, naming the type and field: an unknown provider prefix, a `models:` entry
whose prefix is unknown, a provider-backed model without a price (unless `self_hosted`), and a
harness on the wrong kind of provider.

## Which harness on which provider

| harness | provider `type` | API the server must serve | `base_url` |
|---|---|---|---|
| `codex` | `openai_compat` | OpenAI **Responses** (`POST {base_url}/responses`, streamed) | required, ends in `/v1` |
| `claude` | `anthropic` | Anthropic Messages (`POST {base_url}/v1/messages`) | optional API root, no `/v1` |
| `pi` | `openai_compat` | OpenAI Chat Completions (Pi's `openai-completions`) | required, ends in `/v1` |
| `llm` | `openai_compat` (required) | OpenAI **Chat Completions** (`POST {base_url}/chat/completions`, not streamed) | required, ends in `/v1` |
| claude, codex, pi | none (native id) | the harness's own vendor | - |

Codex 0.162.1 (the pinned version) refuses `wire_api = "chat"`, so a server that only has
`/v1/chat/completions` will not work with Codex. vLLM serves `/v1/responses`; for the HF router,
run the preflight to find out for your model.

## Pointing a swarm at vLLM

1. Serve the model with tool calling on and a key, e.g.
   `vllm serve Qwen/Qwen3.8-27B-FP8 --api-key "$VLLM_API_KEY" --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
   (the parsers depend on the model family).
2. Make it reachable from the worker jobs (see "Reaching a self-hosted server" below).
3. Put the key in the controller's environment (a Space secret) under the name you give
   `api_key_env`, e.g. `VLLM_API_KEY`. The controller sends it to that type's jobs as a secret.
4. Declare the provider with `self_hosted: true` (price `[0, 0, 0]`, pinned into the spec) and
   `max_working` to cap how many agents are in a turn on it at once.
5. `flock launch spec.yaml` (it probes the provider first; `flock update --check` probes again).

For the HF router use `base_url: https://router.huggingface.co/v1`, `api_key_env: HF_TOKEN` and
the router's per-token prices in `pricing` (they are not fetched for you; `GET /v1/models` lists
them per provider). Pin the router provider in the model id (`Qwen/Qwen3.8-27B:deepinfra`, with
`pricing` keyed on that id): on 2026-10-16 the router's automatic choice for Qwen3.8-27B
(cerebras) failed every turn after a tool call, while deepinfra, novita and ovhcloud worked.
Preflight catches this: it sends a tool result back.

## Reaching a self-hosted server

A self-hosted provider is just a `base_url` that every worker must be able to reach over
HTTPS, with the key sent as the provider says. flock does not place workers next to the
server: workers are CPU jobs and inference runs elsewhere. `127.0.0.1` works only if the server
runs inside the worker job. Options: a public endpoint behind its own key (e.g. vLLM's
`--api-key` behind a TLS proxy or on a cloud VM; the simplest, and preflight can probe it from
the controller); HF Inference Endpoints, which serve an HTTPS URL that takes an HF token as
`Authorization: Bearer` (`api_key_env: HF_TOKEN`); or the HF Jobs proxy, untested: start the
GPU job with `--expose 8000` (URL `https://<job id>--8000.hf.jobs`, which checks an HF token),
serve vLLM without `--api-key`, set `api_key_env: HF_TOKEN`, and use `auth: bearer` on
`anthropic` providers because Claude Code's default `x-api-key` is not accepted. On 2026-10-16
`--expose-public 8000` with vLLM's own `--api-key` returned vLLM's 401 for every keyed request
(the proxy most likely did not pass the key through).

## Keys

The spec only ever names a key (`api_key_env`). The value lives in three places, each by that
name:

| where | who puts it there | used for |
|---|---|---|
| controller environment (Space secrets) | you | preflight; sent to jobs |
| worker job secrets | controller, added automatically for provider-backed types | read by the worker |
| harness process | worker | Codex: same name (`env_key`); Claude Code: `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN` with `auth: bearer` |

How the key travels: Codex always sends `Authorization: Bearer <key>`. Claude Code sends
`x-api-key: <key>` by default; set `auth: bearer` on an `anthropic` provider to send
`Authorization: Bearer <key>` instead (Claude Code's `ANTHROPIC_AUTH_TOKEN`), for gateways and
proxies that only accept a bearer token. Preflight sends the key the same way.

Controller secrets to set: every `api_key_env` your providers name (e.g. `HF_TOKEN`,
`VLLM_API_KEY`), plus the `secrets` your types list, plus `FLOCK_TOKEN`.

The harness gets nothing else from the worker's environment: an allowlist (`PATH`, `HOME`,
`LANG`, `LC_*`, `TERM`, proxies and CA bundles, its config dir) plus the type's `env`, its
declared `secrets`, the msgboard identity and the provider key. `FLOCK_TOKEN` stays in the
worker.

## Reasoning

| `reasoning` | Codex (`model_reasoning_effort`, sent as Responses `reasoning.effort`) | Claude Code (Agent SDK) |
|---|---|---|
| `off` | `none` | `thinking: {type: disabled}` |
| `low` / `medium` / `high` | same word | `effort: low/medium/high` |

A type's `reasoning` overrides its alias's. Whether a server honours the effort is up to the
server; Qwen3 on vLLM turns thinking off through the chat template, not the effort. `extra`
(raw request-body fields such as `chat_template_kwargs`) is accepted for swarmlab compatibility
but Claude Code, Codex and Pi cannot send body fields, so for them it is not used; set such
defaults on the server (vLLM can set default chat-template arguments at startup). The `llm`
harness does send `extra`, merged into every request body, and ignores `reasoning`
(docs/llm-harness.md).

## Cost

Tokens are reported by model id; a provider-backed type's model id is priced from the
`[input, output, cached]` row pinned at apply (alias `pricing`, else the provider's row for the
id, else its `"*"` row, else `[0, 0, 0]` when self-hosted). Cache writes count at the input
price. A re-apply with a new price changes the cost of new tokens only. Self-hosted cost is the
GPU job's time, which flock does not count.

Cost is only as good as the usage the server reports. A provider-backed turn that completes with
zero tokens is flagged: the worker logs a warning (once per agent), the `turn_ended` event carries
`usage_missing: true`, and `flock status` / the status page count such agents per provider
(`zero_usage_agents`) with a "cost unknown" warning. The HF router streams zero usage for some
providers (deepinfra, novita and ovhcloud for Qwen3.8-27B on 2026-10-16), so Codex on the router
shows $0 today.

## Preflight

`flock update --check` on a running swarm (or `POST /swarms/{swarm}/preflight` with the spec as the body)
makes the controller probe each distinct provider-backed model the way a harness starts: two
turns, streamed on the Responses API for `openai_compat` (shaped like Codex's requests:
`store: false`, encrypted reasoning, parallel tool calls), non-streamed Chat Completions with
`extra` for types on the `llm` or `pi` harness (a tool call written as text passes with a warning,
since that harness accepts it), or the Messages API for `anthropic`.
Turn one asks for a call to a trivial `get_time` tool; turn two sends the model's output back
with the tool's result and must complete. Per model it reports `ok`, `tool_call_seen`,
`follow_up_ok`, `usage_reported`, `latency_ms`, `error` and `warning`. `ok` needs both turns: a
server that answers turn one and fails the follow-up (the HF router's automatic provider choice
did, see below) fails preflight. A server that streams zero token usage passes with a warning,
since flock then cannot count cost. `flock launch` runs preflight whenever the spec has
provider-backed models and refuses to apply on a failure; `--skip-preflight` bypasses it.

Preflight never passes silently. A model that was not probed (its `api_key_env` is not in the
controller's environment, or `--skip-preflight`) is reported as `unverified` with a warning
line. Validation is best-effort: `launch` blocks on definite problems (a failed probe) and
proceeds on missing evidence (unverified), unless `--strict` is passed, which counts
unverified as failure (`flock update --check --strict` exits 1 the same way).

## `max_working`

The number of agents of the swarm on that provider that may be in a turn at once, counted per
agent (an agent is one slot whether its turn makes one request or fifty). The controller
enforces it, on by default:

- **Queue, not reject.** A prompt for an agent that is not in a turn is accepted as usual
  (`POST /agents/{name}/prompt` never refuses it) but stays queued in the controller while the
  provider is at its cap. A prompt for an agent already in a turn takes no new slot.
- **Oldest prompt first.** Free slots go to pending prompts in the order they were created,
  across all workers of the provider, so a worker that polls often cannot keep re-taking
  freed slots. The worker owning the next prompt in line is woken as soon as a slot frees.
- **Every turn start is admitted the same way.** Admission happens where the controller hands
  queued prompts to workers, so it covers API prompts and `resume_prompt` after a crash (for a
  capped provider it is queued behind the assign instead of riding inside it). Anything that
  queues a prompt in the controller is covered too, e.g. a spawned child's first prompt.
- **Counts come from worker heartbeats, corrected by turn events.** A prompt handed out
  takes its slot at once (two polls between heartbeats cannot both take the last slot), and
  the agent's `turn_ended` event releases it at once (events reach the controller every 3 s),
  not at the next heartbeat. The cap is still **soft by about one heartbeat**: a heartbeat
  built before the worker saw a turn change can briefly overrule an event.
- **The real limit belongs at the endpoint.** `max_working` keeps a swarm from overrunning a
  server it shares; it does not protect the server from other clients, and it counts agents,
  not requests. Set the server's own concurrency limits (vLLM `--max-num-seqs`, a gateway
  rate limit) for a hard bound.

`flock status`, `GET /swarms/{swarm}` (`status.providers`) and the status page show
`working / max_working` per provider. `FLOCK_MAX_WORKING_ENFORCE=0` on the controller is an
escape hatch: report only, no queueing.

## Status

`GET /agents/{name}` and `flock status --agent` show the provider of a provider-backed agent;
`session_started` events carry `provider` and `model_id`.

## Not verified yet

- Codex against a real vLLM Responses endpoint. Against the HF router it works with a pinned
  provider (smoke run 2026-10-16: 10/10 turns with tool calls, cost unknown because usage is zero).
- Claude Code against vLLM's Anthropic-compatible endpoint. Against a fake server, Claude Code
  2.1.296 sent every request, background ones included, to the type's model id, so
  `ANTHROPIC_SMALL_FAST_MODEL` was not needed.
- Workers reaching a GPU job through the HF Jobs proxy (`--expose` with an HF token).
- Codex with model ids it has no built-in metadata for (context window, auto-compaction).
- Cost of Codex on the HF router: the working router providers stream zero usage, so it shows
  as $0 with the zero-usage warning.


Pi also supports `base_urls` for multiple servers of one model. Agents are spread
by a stable name hash, and their model config directories are separate when more
than one endpoint is configured. See [pi.md](pi.md).
