# Open models in flock

A swarm can run its agents on open-weight models: vLLM in a GPU job, the HF router, or any
server that speaks OpenAI Chat Completions. A Harbor spec names the server in `inference:`;
this page is the how-to.

## The spec

A Harbor spec names one model server, `inference:`, and the run's `model` (a role may set its
own `model` and `pricing`):

```yaml
model: huggingface/Qwen/Qwen3.8-27B:deepinfra    # the router provider pinned
pricing: [0.2, 2.5, 0.2]                         # USD per 1M: input, output, cached
inference:
  base_url: https://router.huggingface.co/v1     # or your vLLM: https://qwen.example.org/v1
  api_key_env: HF_INFERENCE_TOKEN                # inference only: never the workers' HF_TOKEN
  api_format: openai_chat_completions
  max_working: 256                               # optional: agents in a turn at once
```

`flock launch` compiles it into the swarm spec's `providers:` block (one `openai_compat`
provider, `inference`, with `pricing` keyed by model id) and each type's `model` as
`inference:<model id>`. The contract of `providers:` and `models:` is in docs/design.md
"Providers"; field names match swarmlab's.

Harbor's Pi speaks OpenAI **Chat Completions** (`POST {base_url}/chat/completions`); the
provider must be `openai_compat` and `base_url` end in `/v1`.

## Pointing a swarm at vLLM

1. Serve the model with tool calling on and a key, e.g.
   `vllm serve Qwen/Qwen3.8-27B-FP8 --api-key "$VLLM_API_KEY" --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
   (the parsers depend on the model family).
2. Make it reachable from the worker jobs (see "Reaching a self-hosted server" below).
3. Put the key in the controller's environment (a Space secret) under the name you give
   `inference.api_key_env`, e.g. `VLLM_API_KEY`. The controller sends it to the workers as a secret.
4. Set `pricing: [0, 0, 0]` (the GPU job is the real cost) and `inference.max_working` to cap
   how many agents are in a turn on it at once.
5. `flock launch spec.yaml` (it probes the provider first; `flock update --check` probes again).

For the HF router use `base_url: https://router.huggingface.co/v1`, an inference-only token
and the router's per-token prices in `pricing` (they are not fetched for you; `GET /v1/models` lists
them per provider). Pin the router provider in the model id (`Qwen/Qwen3.8-27B:deepinfra`, with
`pricing` keyed on that id): some router providers fail every turn after a tool call.
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
serve vLLM without `--api-key` and set `api_key_env: HF_TOKEN`. `--expose-public 8000` with
vLLM's own `--api-key` has returned vLLM's 401 for every keyed request (the proxy most likely
does not pass the key through).

## Keys

The spec only ever names a key (`api_key_env`). The value lives in three places, each by that
name:

| where | who puts it there | used for |
|---|---|---|
| controller environment (Space secrets) | you | preflight; sent to jobs |
| worker job secrets | controller, added automatically for provider-backed types | read by the worker |
| the agent's Trial | worker (Harbor) | the same name |

How the key travels: `Authorization: Bearer <key>`. Preflight sends the key the same way.

Controller secrets to set: every `api_key_env` your providers name (e.g. `HF_TOKEN`,
`VLLM_API_KEY`), plus the `secrets` your types list, plus `FLOCK_TOKEN`.

A Trial's environment is the task's plus the inference key, the tools and the resources;
`FLOCK_TOKEN` and the workers' `HF_TOKEN` stay in the worker.

## Reasoning

A type's `reasoning` (`off`, `low`, `medium`, `high`) overrides its alias's. Whether a server
honours the effort is up to the server; Qwen3 on vLLM turns thinking off through the chat
template, not the effort. `extra` (raw request-body fields such as `chat_template_kwargs`) is
accepted for swarmlab compatibility but not sent (Harbor's Pi cannot; `flock launch` warns), so
set such defaults on the server (vLLM can set default chat-template arguments at startup).

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
providers, so such a model shows $0.

## Preflight

`flock update --check` on a running swarm (or `POST /swarms/{swarm}/preflight` with the spec as the body)
makes the controller probe each distinct provider-backed model the way a harness starts: two
non-streamed Chat Completions turns, as Pi calls it (no `extra`; a tool call written as text is
no tool call, since Pi takes only `tool_calls`).
Turn one asks for a call to a trivial `get_time` tool; turn two sends the model's output back
with the tool's result and must complete. Per model it reports `ok`, `tool_call_seen`,
`follow_up_ok`, `usage_reported`, `latency_ms`, `error` and `warning`. `ok` needs both turns: a
server that answers turn one and fails the follow-up (some HF router providers do,
see below) fails preflight. A server that streams zero token usage passes with a warning,
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

- Workers reaching a GPU job through the HF Jobs proxy (`--expose` with an HF token).
- Cost on the HF router: the working router providers stream zero usage, so it shows as $0 with
  the zero-usage warning.

