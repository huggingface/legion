# flock

Host coding-harness agent swarms (Claude Code, Codex) on Hugging Face Jobs: keep
them alive, prompt them, watch them. One controller, N worker jobs, up to 200
agents per worker. Declare a swarm in YAML and the controller keeps it running,
the way Kubernetes keeps pods running.

It is the layer under [msgboard](https://github.com/huggingface/msgboard) and
[swarmlab](https://github.com/huggingface/swarmlab): it hosts agents and exposes
a prompt API and an event stream, and leaves messaging and experiment design to
them.

## Features

- **Declarative swarms:** a YAML spec names a swarm, an HF namespace, a bucket
  and one or more agent types. Each type sets the harness (`claude`,
  `codex`, `pi` or `llm`), model, replicas, system prompt, job flavor, agents per worker,
  worker lifetime, idle grace, per-prompt `max_turns` and `max_budget_usd`
  (enforced for every harness: Claude in its SDK, Codex, Pi and `llm` in the worker).
  `apply`, `scale` and `remove` edit the spec; a reconcile loop makes HF Jobs
  match it.
- **Stable agent identities:** agents are named `<type>-NNNN`. The same name
  keeps the same harness session and workdir across worker restarts.
- **Durable sessions:** workers sync each agent's session and workdir to a
  bucket and restore them on relaunch, so an agent resumes with its context
  on another worker. A failed resume is a visible fresh start.
- **Process per burst:** an agent's harness process starts on its first prompt,
  takes later prompts mid-turn, and exits after an idle grace. Idle agents
  cost only disk.
- **Health states:** agents are `working`, `idle`, `stuck`, `unknown` or
  `dead`; workers are `launching`, `running`, `draining`, `unknown` or `dead`.
  A type that crashes five times in a row is marked `crashlooping`.
- **Prompts and events:** `POST /agents/{name}/prompt` acks at once; results
  are read from the agent's event stream (pi session format v3, long-poll). A
  prompt to an agent whose worker is down is held and delivered after relaunch.
- **Usage and cost:** tokens per model per turn, cost from a price table, per
  agent and per swarm.
- **Open models:** a `providers:` block points a type at vLLM in a GPU job, the
  HF router or any OpenAI- or Anthropic-compatible server (`model:
  "gpu1:Qwen/Qwen3.8-27B-FP8"`), with prices pinned at apply and a preflight
  that checks each model can call tools. See [docs/open-models.md](docs/open-models.md).
- **Planned restarts:** workers live 24 h plus jitter by default, drain before
  their deadline, and the controller relaunches them with their agents
  restored. The herd never restarts all at once.
- **Spawning:** a leader agent creates, prompts and removes child agents of
  other types with the `flock` CLI, under a `spawn:` policy (allowed types,
  caps, depth) and swarm `limits:`. Each agent calls the controller with its
  own scoped token. See [docs/spawn.md](docs/spawn.md).
- **msgboard identity (optional):** a type can name an invite; the worker
  redeems it by agent name so an agent keeps its principal across restarts.
- **Status page and CLI:** `/` shows every swarm with state chips and cost;
  `flock` wraps the HTTP API.

Supported harnesses:

| `harness` | What runs | Providers | Notes |
|---|---|---|---|
| `claude` | Claude Code through the Python Agent SDK | native, `anthropic` | |
| `codex` | `codex app-server` | native, `openai_compat` (Responses API) | verified locally, not yet in a job |
| `pi` | `pi --mode rpc` | native, `openai_compat` | [docs/pi.md](docs/pi.md) |
| `llm` | in-process Chat Completions loop, one `bash` tool | `openai_compat` only | for small open models and high call volume; [docs/llm-harness.md](docs/llm-harness.md) |

## Usage

Needs Python 3.11+, an HF token with Jobs access, and the harness API key.

```sh
git clone https://github.com/huggingface/flock && cd flock
uv sync --all-extras                 # or: pip install -e ".[controller,worker,dev]"
```

`uv sync` does not activate the environment, so the commands below go through `uv run` (or
run `source .venv/bin/activate` once). A bare `flock` may be util-linux's file-locking tool,
a different program; `uv run` also gives the scripts a Python with `huggingface_hub`.

### Run the controller

As an HF Docker Space (the tested setup):

```sh
export HF_TOKEN=... ANTHROPIC_KEY=... FLOCK_TOKEN=$(openssl rand -hex 24)
uv run hf buckets create <you>/flock --private                # once: wheels and agent state
W=$(uv run scripts/publish-wheel.sh hf://buckets/<you>/flock/dist)   # wheel for workers
uv run scripts/deploy-space.sh "$W" <you>/flock
```

Or locally:

```sh
FLOCK_TOKEN=... HF_TOKEN=... ANTHROPIC_API_KEY=... \
FLOCK_WHEEL_URL=hf://buckets/<you>/flock/dist/<rev>/flock-0.0.1-py3-none-any.whl \
uv run flock-controller          # http://127.0.0.1:7860
```

Workers bootstrap from a stock `node:22-bookworm` image: they install the
pinned harness CLI and the wheel from the bucket. Every secret a spec names
must be in the controller's environment.

One key, three places. The names differ only for the Anthropic key:

| Your shell (`deploy-space.sh` reads) | Space secret / controller env | Spec `secrets:` |
| --- | --- | --- |
| `ANTHROPIC_KEY` | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` |
| `HF_TOKEN` | `HF_TOKEN` | `HF_TOKEN` (workers need it for the wheel and the bucket) |
| `FLOCK_TOKEN` | `FLOCK_TOKEN` | never: the controller passes it to workers itself |

Running the controller locally, export the middle column (`ANTHROPIC_API_KEY`).

### Declare and drive a swarm

```yaml
# demo.yaml
swarm: demo
namespace: <you>
bucket: hf://buckets/<you>/flock/demo
types:
  scribe:
    harness: claude
    model: claude-haiku-5-5
    system_prompt_file: prompts/scribe.md
    secrets: [ANTHROPIC_API_KEY, HF_TOKEN]
    replicas: 100
    agents_per_worker: 50
    max_budget_usd: 0.05
```

Open models go through `providers:` (keys by secret name only; [docs/open-models.md](docs/open-models.md)):

```yaml
providers:
  gpu1: {type: openai_compat, base_url: "https://qwen.example.org/v1",  # your vLLM, over HTTPS
         api_key_env: VLLM_API_KEY,
         self_hosted: true, max_working: 256}
models:
  qwen-27b: {model: "gpu1:Qwen/Qwen3.8-27B-FP8", reasoning: medium}
types:
  analyst: {harness: codex, model: qwen-27b, replicas: 50}
```

```sh
export FLOCK_URL=https://<you>-flock.hf.space FLOCK_TOKEN=...
uv run flock apply demo.yaml
uv run flock status demo                 # counts by type and state, cost
uv run flock prompt scribe-0007 "Summarize your workdir."
uv run flock events scribe-0007 --follow # JSONL: turn_started, turn_ended, ...
uv run flock scale demo scribe 200
uv run flock remove demo
```

Other commands: `agents <swarm> [--type --state]`, `agent <name>`,
`workers <swarm>`, `preflight <spec>`. `apply` runs the preflight first when the
spec has provider-backed models (`--skip-preflight` to bypass).

### HTTP API

`Authorization: Bearer $FLOCK_TOKEN` on every call.

| Call | Purpose |
| --- | --- |
| `PUT/GET/DELETE /swarms/{swarm}`, `GET /swarms` | apply, inspect, remove |
| `POST /swarms/{swarm}/preflight` | one tool-calling request per provider-backed model |
| `GET /swarms/{swarm}/agents?type=&state=&cursor=` | paginated agent list |
| `GET /agents/{name}` | one agent's state, pending prompts, cost |
| `POST /agents/{name}/prompt` | body `{text, timeout?, reply_to?}`; acks `queued` or `pending` |
| `POST /swarms/{swarm}/prompts` | body `{type, state?, text, timeout?, reply_to?}`: the prompt to every agent of a type in one request (`flock prompt --swarm S --type T [--state idle] TEXT`); acks counts. With `state: idle` agents that already have a pending prompt are skipped (`skip_pending`, counted in `skipped_pending`), so repeated calls never stack prompts |
| `GET /agents/{name}/events?after=&wait=` | event stream, long-poll |
| `GET /workers`, `GET /workers/{id}` | worker list and detail |
| `POST /agents/{parent}/spawn`, `DELETE /agents/{name}`, `GET /whoami` | spawn and remove child agents ([docs/spawn.md](docs/spawn.md)); agents call these with their own token |
| `GET /`, `GET /healthz` | status page, health and request counters (no auth) |

## Tests

```sh
uv run pytest -q
uv run ruff check src tests experiments
```

No network, no real jobs: the Hub client and the harness processes are faked.

## Limits and status

- Measured on 2026-10-14: 100 agents on 3 `cpu-basic` workers and 1,000 agents
  on 5 `cpu-upgrade` workers pass every check (apply, prompt all, kill and
  resume, planned drain, scale down, remove). A 10,000-agent run on 50 workers
  reached all agents and delivered every prompt, then the HF Spaces edge in
  front of the controller throttled the workers' heartbeats; the controller
  took them for dead and relaunched them. Reports in `experiments/*/RESULTS.md`.
- Packing: one `cpu-upgrade` worker holds 200 resident agents and about 120
  concurrent turns; `cpu-basic` 50 and 20. Cold turns are CPU-bound at about
  1.5 CPU-seconds each.
- Not built yet: bulk prompt endpoint, swarm-level event stream, per-worker
  batched bucket sync, spend ledger, autoscaling, API-loop agents, per-caller auth.
- The controller's bearer token sits in the `Authorization` header, so the
  Space must be public; the API is still token-protected.

- Controller state (SQLite and event files) lives in `FLOCK_DATA`, `/data` in
  the controller image. Without the Space's persistent storage a redeploy wipes
  it; replicas come back from the spec, spawned agents do not.

Design, contract and decision log: [docs/design.md](docs/design.md).
