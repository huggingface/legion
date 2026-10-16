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
  gpu1: {type: openai_compat, base_url: "http://${HF_NETWORK_GROUP_PREFIX}vllm:8000/v1",
         api_key_env: VLLM_API_KEY, network_group: qwen-gpu,
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

`flock apply` rejects, naming the type and field: an unknown provider prefix, a `models:` entry
whose prefix is unknown, a provider-backed model without a price (unless `self_hosted`), and a
harness on the wrong kind of provider.

## Which harness on which provider

| harness | provider `type` | API the server must serve | `base_url` |
|---|---|---|---|
| `codex` | `openai_compat` | OpenAI **Responses** (`POST {base_url}/responses`, streamed) | required, ends in `/v1` |
| `claude` | `anthropic` | Anthropic Messages (`POST {base_url}/v1/messages`) | optional API root, no `/v1` |
| either | none (native id) | the harness's own vendor | - |

Codex 0.162.1 (the pinned version) refuses `wire_api = "chat"`, so a server that only has
`/v1/chat/completions` will not work with Codex. vLLM serves `/v1/responses`; for the HF router,
run the preflight to find out for your model.

## Pointing a swarm at vLLM

1. Serve the model with tool calling on and a key, e.g.
   `vllm serve Qwen/Qwen3.8-27B-FP8 --api-key "$VLLM_API_KEY" --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`
   (the parsers depend on the model family).
2. Make it reachable from the worker jobs. `127.0.0.1` does not work unless the workers run
   inside the same job. HF Jobs offers two paths:
   - **Network group (recommended).** Start the server job with
     `hf jobs run --network-group <group> --network-alias vllm ...` and set
     `network_group: <group>` on the provider. flock launches the workers of every type on that
     provider into the same group, and they reach the server directly on any port. Write the
     URL with the alias, `base_url: http://${HF_NETWORK_GROUP_PREFIX}vllm:8000/v1`. HF Jobs sets
     `HF_NETWORK_GROUP_PREFIX` inside group members only, and the worker expands it. vLLM keeps
     its own `--api-key`. Both jobs must be in the same namespace (and resource group). The
     controller is not in the group, so preflight skips these providers (reported as `skipped`).
     **Not yet working end to end:** a group is placed where its first member runs, and later
     members must use a flavor available there. On 2026-10-16 a group started by an
     `a100-large` vLLM job refused `cpu-basic` and `cpu-upgrade` workers ("Flavor 'cpu-basic'
     for arch 'amd64' is not available where network group ... already runs"). Untested ways
     out: start the workers first, or give the worker types a flavor from the GPU's pool.
   - **Jobs proxy.** `--expose 8000` publishes `https://<job id>--8000.hf.jobs`, which needs an
     HF token as `Authorization: Bearer`. Serve without `--api-key`, set
     `api_key_env: HF_TOKEN`, and use `auth: bearer` on `anthropic` providers, because Claude
     Code's default `x-api-key` is not accepted. In the 2026-10-16 smoke run,
     `--expose-public 8000` with vLLM's `--api-key` returned vLLM's 401 for every keyed request:
     the proxy did not pass the key through.
3. Put the key in the controller's environment (a Space secret) under the name you give
   `api_key_env`, e.g. `VLLM_API_KEY`. The controller sends it to that type's jobs as a secret.
4. Declare the provider with `self_hosted: true` (price `[0, 0, 0]`, pinned into the spec) and,
   if you want the count in status, `max_working`.
5. `flock preflight spec.yaml`, then `flock apply spec.yaml`.

For the HF router use `base_url: https://router.huggingface.co/v1`, `api_key_env: HF_TOKEN` and
the router's per-token prices in `pricing` (they are not fetched for you; `GET /v1/models` lists
them per provider). Pin the router provider in the model id (`Qwen/Qwen3.8-27B:deepinfra`, with
`pricing` keyed on that id): on 2026-10-16 the router's automatic choice for Qwen3.8-27B
(cerebras) failed every turn after a tool call, while deepinfra, novita and ovhcloud worked.
Preflight catches this: it sends a tool result back.

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
but neither harness can send body fields, so it is not used; set such defaults on the server
(vLLM can set default chat-template arguments at startup).

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

`flock preflight spec.yaml` (or `POST /swarms/{swarm}/preflight` with the spec as the body)
makes the controller probe each distinct provider-backed model the way a harness starts: two
streamed turns on the Responses API for `openai_compat` (shaped like Codex's requests:
`store: false`, encrypted reasoning, parallel tool calls) or the Messages API for `anthropic`.
Turn one asks for a call to a trivial `get_time` tool; turn two sends the model's output back
with the tool's result and must complete. Per model it reports `ok`, `tool_call_seen`,
`follow_up_ok`, `usage_reported`, `latency_ms`, `error` and `warning`. `ok` needs both turns: a
server that answers turn one and fails the follow-up (the HF router's automatic provider choice
did, see below) fails preflight. A server that streams zero token usage passes with a warning,
since flock then cannot count cost. `flock apply` runs preflight whenever the spec has
provider-backed models and refuses to apply on a failure; `--skip-preflight` bypasses it.

## `max_working`

The number of agents of the swarm on that provider that may be in a turn at once, counted per
agent from heartbeats (an agent is one slot whether its turn makes one request or fifty). Today
it is reported, not enforced: `flock status`, `GET /swarms/{swarm}` (`status.providers`) and the
status page show `working / max_working` per provider. Enforcement would hold prompts for idle
agents while the provider is at its cap; the count lags by up to one heartbeat (30 s), so the
cap would be soft by that much.

## Status

`GET /agents/{name}` and `flock agents` show the provider of a provider-backed agent;
`session_started` events carry `provider` and `model_id`.

## Not verified yet

- Codex against a real vLLM Responses endpoint. Against the HF router it works with a pinned
  provider (smoke run 2026-10-16: 10/10 turns with tool calls, cost unknown because usage is zero).
- Claude Code against vLLM's Anthropic-compatible endpoint. Against a fake server, Claude Code
  2.1.296 sent every request, background ones included, to the type's model id, so
  `ANTHROPIC_SMALL_FAST_MODEL` was not needed.
- Workers reaching a GPU job through a network group (see the placement note above).
- Codex with model ids it has no built-in metadata for (context window, auto-compaction).
