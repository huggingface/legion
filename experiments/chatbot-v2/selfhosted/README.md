# chatbot-v2, self-hosted / max utilization

The same experiment as `experiments/chatbot-v2`, for a model deployment the operator runs
(an OpenAI-compatible endpoint: vLLM or sglang) at a fixed cost. The goal is maximum use of
the deployment, not cost. A generator renders the variant from the parent folder:

- `scale.yaml`: every knob in one place (below);
- `templates/`: the prompts and protocols for N teams;
- `render.py --into <dir>`: writes spec.yaml, phase2/spec.yaml, the per-team prompts, the
  kickoff, the team protocol, the phase-2 prompts and world.yaml into a copy of
  experiments/chatbot-v2 (`--teams`, `--per-team ds=2,glm=2`, `--hours` override scale.yaml);
- `load_smoke.py` + `load-smoke.md`: find how many agents the endpoint takes.

## What differs from the main spec

| | main spec | self-hosted |
|---|---|---|
| providers | HF router, per-token prices | `ds` (and `glm`): `base_url` + `api_key_env` from the environment, `self_hosted: true`, price 0 for every model id |
| $ caps | per agent, per prompt, swarm hard stop | none: `limits.budget_usd` null, no `agent_usd`, no `max_budget_usd` |
| limits | | `max_turns` 12 per prompt (unchanged), `while.max_rounds` 600 per agent, `while.until` 4 h |
| cadence | `every: 90s` | `every: turn_end`: the next round as soon as a turn ends |
| wakes | 3 per 30 min, 30 s gap | 20 per 30 min, 10 s gap |
| teams | 2 x 4 | `teams` x `per_team` (default 4 x 4 = 16; 2 DS + 2 GLM each, or all `MODEL_DS` when `MODEL_GLM` is unset) |
| interviewers | Qwen-9B mid, DeepSeek strong (router) | same endpoint: strong = `MODEL_INTERVIEWER_STRONG` or `MODEL_DS`, mid = `MODEL_INTERVIEWER_MID`, else `MODEL_GLM`, else `MODEL_DS` |
| world knobs | world.yaml | world.yaml + `scale.yaml: world` (probes 8 per agent per hour, 320 in total); interview quotas stay per team, so the total grows with the team count |

`every: turn_end` needs no `max_working`. Without it the worker re-prompts an agent the moment
its turn ends (no controller round trip, no idle gap). Set `max_working` in scale.yaml only to
cap the agents in a turn at once; the controller then sends the rounds and queues agents FIFO.

## Knobs (`scale.yaml`)

| knob | default |
|---|---|
| `teams` | 4 (2 to 26; dirs `teams/<letter>/`, threads `[team-<letter>]`) |
| `per_team` | `{ds: 2, glm: 2}` |
| `agents_per_worker` | 8 (workers per type = ceil(replicas / 8); 8 workers at the default) |
| `max_working` | null |
| `api_key_env` | `SELFHOSTED_API_KEY` |
| `timeout_s` | 300 |
| `hours` | 4 (switch at the halfway mark) |
| `max_rounds` | 600 per agent |
| `max_turns` | 12 |
| `wake` | `{min_gap: 10s, max: 20, window: 30m}` |
| `min_post_gap` | 10s |
| `models.<ds\|glm\|interviewer>` | request extras (`max_tokens`, `temperature`, `chat_template_kwargs`) |
| `world` | merged over world.yaml |

Environment at render time: `SELFHOSTED_BASE_URL` (or `BASE_URL_DS` / `BASE_URL_GLM`),
`MODEL_DS`, `MODEL_GLM`, `MODEL_INTERVIEWER_STRONG`, `MODEL_INTERVIEWER_MID`. The key's value
stays in `SELFHOSTED_API_KEY` (a secret). Unset variables render as their names.

## Run it

Locally:

```sh
cp -r experiments/chatbot-v2 /tmp/cb && \
  python experiments/chatbot-v2/selfhosted/render.py --into /tmp/cb   # with the env above
cd /tmp/cb && CHATBOT_CONFIG=$PWD/world.yaml python -m flock.controller.localrun spec.yaml --until idle
```

As the HF Job of the main README, with `-e CHATBOT_VARIANT=selfhosted`, the model variables
as `-e`, and `--secrets SELFHOSTED_API_KEY`: the bootstrap renders the variant into its copy
of the experiment before anything else (log: `logs/variant.log`). The endpoint must be
reachable from the job over HTTPS. For a live update (`control/update`), upload a folder
rendered the same way, not the plain parent folder: the update copies it over the rendered
one.

## Load smoke

Before the run, size the swarm with `load-smoke.md`: it reports calls/s and latency at
8/16/32/64 agents on the real task and recommends knee x 0.8 agents.
