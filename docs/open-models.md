# Open models in flock

A spec names its model with `model`, a Hub id, and says where it runs with one `inference` block:

- **`hf_router`**: the [HF router](https://huggingface.co/docs/inference-providers/index)
  (Inference Providers). This is the default when the spec has no `inference` block.
- **`urls`**: OpenAI-compatible servers you run, on a Slurm cluster, a cloud box or your own
  machine.

The block has the same shape in a Harbor spec (the run's, or a role's own) and in a swarm spec
(the swarm's, or a type's own). This page is the how-to; the contract is in
[design.md](design.md) "Inference".

## The spec

```yaml
model: Qwen/Qwen3.8-27B            # a Hub id; with urls, the name the servers serve

# The HF router (the default)
inference:
  hf_router:
    provider: cerebras             # a provider, or a policy: fastest (the router's default), cheapest, preferred
    bill_to: huggingface           # optional: bill an org (X-HF-Bill-To)
  api_key_env: HF_INFERENCE_TOKEN  # default: HF_TOKEN

# Servers you run
inference:
  urls: [http://10.53.196.166:8000/v1, http://10.53.196.167:8000/v1]
  api_key_env: VLLM_API_KEY        # optional: no key by default

# Settings for either
inference:
  max_working: 3                   # agents of the swarm in a turn at once on these endpoints
  context_window: 131072           # used only when the servers report none
  reasoning_effort: low            # Pi's --thinking level
  metrics: true                    # urls only: scrape each server's vLLM /metrics (docs/health.md)
```

| Field | Meaning |
|---|---|
| `hf_router.provider` | A router provider (`cerebras`, `deepinfra`, ...), or a selection policy: `fastest`, `cheapest`, `preferred`. Unset: the router's default (`fastest`). flock sends `<model>:<provider>`. |
| `hf_router.bill_to` | An org the router bills instead of the key's owner: Pi and preflight send `X-HF-Bill-To`. |
| `urls` | One or more base URLs ending in `/v1`, all serving `model`. A comma-separated string works too (`urls: ${VLLM_URLS}`). |
| `api_key_env` | The env var that holds the key (never the key itself). Default: `HF_TOKEN` for `hf_router`, none for `urls`. |
| `max_working` | See [`max_working`](#max_working). |
| `context_window` | The model's context window in tokens, for servers that do not report one. |
| `reasoning_effort` | Pi's `--thinking` level (`off` ... `max`). |
| `metrics` | `urls` only: scrape each server's vLLM `/metrics` into the health snapshots. |

A role (Harbor spec) or a type (swarm spec) may set its own `model`, on the run's endpoints, and
its own `inference`, which replaces the run's block as a whole. `pricing` stays where it was: the
run's and a role's `[input, output, cached]` in USD per 1M tokens (see [Cost](#cost)).

`flock launch` compiles a Harbor spec's block into the swarm spec as it is. Each type's
`resolved_model` then pins the block it runs on, its price and, at apply, the context window
(below). Harbor's Pi speaks OpenAI **Chat Completions** (`POST {url}/chat/completions`).

## The HF router (`hf_router`)

`model` is the Hub id the router lists (`Qwen/Qwen3.8-27B`, not `huggingface/...` and no
`:provider` suffix: the provider goes in `hf_router.provider`). Pin a provider that serves the
model with tool calls: some router providers fail every turn after a tool call, and preflight
catches that, since it sends a tool result back. `GET https://router.huggingface.co/v1/models`
lists each model's providers with their context length, tool support and prices.

The key defaults to `HF_TOKEN`. Every agent's container gets it, so with HF Sandboxes, HF Jobs or
Slurm workers the spec compiler refuses `HF_TOKEN` (the workers' own Hub token): create a
[fine-grained token](https://huggingface.co/settings/tokens/new?tokenType=fineGrained) with only
**Make calls to Inference Providers** and name it, e.g. `api_key_env: HF_INFERENCE_TOKEN`
(experiments/mini-swarm/README.md).

## Servers you run (`urls`)

Any OpenAI-compatible server with tool calling works: vLLM or SGLang on a Slurm cluster,
llama.cpp's `llama-server`, Ollama, a box on the LAN, or another hosted router. Turn tool calling
on, e.g. vLLM's `--enable-auto-tool-choice --tool-call-parser <parser>` (the parser depends on the
model family; `qwen3_coder` for Qwen3.5 and Qwen3.8).

- **Several servers.** Each agent calls one of the `urls`, picked by rendezvous hashing on its
  name: the agents spread over the servers, an agent keeps its server while the list is the same,
  and a server added or removed moves only the agents that must move. Preflight probes every
  server, and with `metrics: true` the health snapshots cover each one (docs/health.md).
- **Reachability.** The agents must reach every URL; the controller's checks use the URLs as
  written.
  - A private cluster address (`10.x`) needs the swarm on the same cluster (`compute: slurm`,
    docs/slurm.md).
  - `localhost` is the machine that runs flock. With `environment.backend: podman` flock gives
    the agents `host.containers.internal` instead, the name podman puts in every container's
    `/etc/hosts` for its host, and allowlists it. The server must listen on the host's address,
    not only on `127.0.0.1`: vLLM does by default, Ollama with `OLLAMA_HOST=0.0.0.0`.
  - HF Sandboxes and HF Jobs workers cannot reach your machine or a private network: the spec
    compiler refuses `localhost` and private addresses there. Serve the model at a public URL.
    Under `compute: slurm`, `localhost` would be each worker's own node and is refused too.
- **Keys.** A server without a key needs no `api_key_env`; Pi then gets a placeholder key.
- **A server that moves.** When a server restarts on another node, update `urls` and run
  `flock update`: each agent calls its new server from its next turn.
- **Cost.** $0 unless the spec sets `pricing`; the real cost is the GPUs' time, which flock does
  not count.

### Your own machine

Serve the model, then run experiments/mini-swarm/urls.yaml: mini-swarm's three agents in
rootless podman containers, on `Qwen/Qwen3.5-0.8B` at `http://localhost:8000/v1`, with a local
board.

```sh
vllm serve Qwen/Qwen3.5-0.8B --port 8000 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3
# or Ollama (MODEL is its own name):
#   OLLAMA_HOST=0.0.0.0 ollama serve & ollama pull qwen3.5:2b
#   MODEL=qwen3.5:2b SERVER_URL=http://localhost:11434/v1 uv run flock launch ...
uv run flock launch experiments/mini-swarm/urls.yaml
uv run flock status mini-swarm-urls
```

### On the HF cluster

The server runs on a GPU node, in Pyxis; the agents run in rootless podman on a CPU node and
reach it by the GPU node's address. The GPU nodes' podman (3.4, with CNI plugins that refuse the
networks it writes) cannot start the containers on the network flock gives each Trial; the CPU
nodes' (4.9) can. `scripts/slurm/podman-node.sh` sets up podman in a shell on a node as
`worker.sbatch` does for Slurm workers (storage under `/tmp`, no user systemd).

```sh
sbatch -p hopper-b200 serve-vllm.sbatch Qwen/Qwen3.5-0.8B
# its log ends with: serving Qwen/Qwen3.5-0.8B at http://<node address>:8000/v1
srun -p hopper-cpu -n 1 -c 8 --mem=16G --pty bash
source .venv/bin/activate && source scripts/slurm/podman-node.sh
SERVER_URL=http://<node address>:8000/v1 flock launch experiments/mini-swarm/urls.yaml
flock status mini-swarm-urls
```

`serve-vllm.sbatch`, one vLLM server on one GPU (the Slurm recipe below uses it too):

```bash
#!/bin/bash
#SBATCH --job-name=vllm
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=%x-%j.log
set -uo pipefail
MODEL=${1:-Qwen/Qwen3.5-0.8B}
PORT=${PORT:-8000}
IMAGE=${IMAGE:-/fsx/anton/shared/images/vllm-openai-v0.32.0.sqsh}
export HF_HOME=/hf
mkdir -p "/scratch/$USER/hf"  # the weights, on the node's own disk
srun --overlap --container-image="$IMAGE" --no-container-mount-home \
  --container-mounts="/scratch/$USER/hf:/hf" --container-env=HF_HOME \
  vllm serve "$MODEL" --port "$PORT" \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 &
until curl -sf -o /dev/null "localhost:$PORT/v1/models"; do
  sleep 10
  kill -0 $! 2>/dev/null || { echo "vLLM exited during startup"; exit 1; }
done
echo "serving $MODEL at http://$(hostname -I | awk '{print $1}'):$PORT/v1"
wait
```

### Slurm

Big models are servers you run: start them with your own job script (one per server, e.g.
`serve-vllm.sbatch` above, or a multi-node one behind a router), then list their URLs from a
`compute: slurm` spec (docs/slurm.md). Its workers run on the cluster, so they reach the nodes'
addresses:

```yaml
compute: {kind: slurm}
model: Qwen/Qwen3.5-0.8B
inference:
  urls: ${VLLM_URLS}               # run.env: VLLM_URLS=http://10.53.1.2:8000/v1,http://10.53.1.3:8000/v1
environment: {backend: podman}
```

`flock init` checks that the servers answer a tool call (its `providers` check), and
`flock launch` checks the model and pins its context window. A server that restarts on another
node gets a new address: update `VLLM_URLS` in run.env and run `flock update`.

## Model check and context window

At apply the controller reads `GET <url>/models` from each endpoint, as written:

- **The router** must list `model`. A named `provider` must serve it with tool support; the
  apply lists the providers that do when it does not.
- **Each server you run** must serve `model`; the apply lists what it serves when it does not.

It then pins the context window the model is served with into the spec
(`types.<t>.resolved_model.context_window`): the smallest over the servers, or the router
providers a request may reach (the named one, or under a policy every one with tool support).
Each counts with its own window (vLLM's and SGLang's `max_model_len`, the router's
`context_length`), or `inference.context_window` when it reports none; one without either
refuses the apply, naming it. Pi's model config gets it as `contextWindow`, so Pi
compacts before the server refuses a request, and `flock status` shows it per model. A re-apply
with the same block and model keeps the pinned value without asking the servers again.

## Keys

The spec only ever names a key (`api_key_env`). The value lives in three places, each by that
name:

| where | who puts it there | used for |
|---|---|---|
| controller environment (Space secrets, run.env) | you | the model check, preflight; sent to jobs |
| worker job secrets | the controller, for the types on that block | read by the worker |
| the agent's Trial | the worker (Harbor) | Pi's requests |

How the key travels: `Authorization: Bearer <key>`. Controller secrets to set: every
`api_key_env` your blocks name (or `HF_TOKEN` for the router by default), plus the `secrets` your
types list, plus `FLOCK_TOKEN`. A Trial's environment is the task's plus the inference key, the
tools and the resources; `FLOCK_TOKEN` and the workers' `HF_TOKEN` stay in the worker.

## Reasoning

`inference.reasoning_effort` is Pi's `--thinking` level. Whether a server honours it is up to
the server; Qwen3 on vLLM turns thinking off through the chat template, not the effort, so set
such defaults on the server (vLLM can set default chat-template arguments at startup).

## Cost

Tokens are reported under the type's `model` and priced from the `[input, output, cached]` row
pinned at apply: the role's `pricing` for its own model, else the run's `pricing`, else
`[0, 0, 0]`. The router's prices are not fetched for you (`GET /v1/models` lists them per
provider). Cache writes count at the input price. A re-apply with a new price changes the cost
of new tokens only.

Cost is only as good as the usage the server reports. A turn that completes with zero tokens is
flagged: the worker logs a warning (once per agent), the `turn_ended` event carries
`usage_missing: true`, and `flock status` and the status page count such agents per inference
block (`zero_usage_agents`) with a "cost unknown" warning.

## Preflight

`flock update --check` on a running swarm (or `POST /swarms/{swarm}/preflight` with the spec as
the body) makes the controller probe each distinct model and inference block the way a harness
starts. First the model check above, then two non-streamed Chat Completions turns, as Pi calls
it (a tool call written as text is no tool call, since Pi takes only `tool_calls`): turn one asks
for a call to a trivial `get_time` tool; turn two sends the model's output back with the tool's
result and must complete. Per model it reports `ok`, `tool_call_seen`, `follow_up_ok`,
`usage_reported`, `latency_ms`, `error` and `warning`. `ok` needs both turns: a server that
answers turn one and fails the follow-up (some router providers do) fails preflight, with a hint
at the likely cause (another `hf_router.provider`, or the server's tool-calling flags). A server
that reports zero token usage passes with a warning, since flock then cannot count cost. `flock
launch` runs preflight and refuses to apply on a failure; `--skip-preflight` bypasses it.

Preflight never passes silently. A model that was not probed (its key is not in the
controller's environment, or `--skip-preflight`) is reported as `unverified` with a warning
line. `launch` blocks on definite problems (a failed probe) and proceeds on missing evidence
(unverified), unless `--strict` is passed, which counts unverified as failure (`flock update
--check --strict` exits 1 the same way).

## `max_working`

The number of agents of the swarm on an inference block that may be in a turn at once, counted
per agent (an agent is one slot whether its turn makes one request or fifty). The controller
enforces it, on by default:

- **Queue, not reject.** A prompt for an agent that is not in a turn is accepted as usual
  (`POST /agents/{name}/prompt` never refuses it) but stays queued in the controller while the
  block is at its cap. A prompt for an agent already in a turn takes no new slot.
- **Oldest prompt first.** Free slots go to pending prompts in the order they were created,
  across all workers on the block, so a worker that polls often cannot keep re-taking
  freed slots. The worker owning the next prompt in line is woken as soon as a slot frees.
- **Every turn start is admitted the same way.** Admission happens where the controller hands
  queued prompts to workers, so it covers API prompts and `resume_prompt` after a crash (on a
  capped block it is queued behind the assign instead of riding inside it). Anything that
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

`FLOCK_MAX_WORKING_ENFORCE=0` on the controller is an escape hatch: report only, no queueing.

## Status

`flock status`, `GET /swarms/{swarm}` (`status.providers`) and the status page show each
inference block by name (`inference`, or `types.<t>.inference` for a type's own): where it
points, each model's context window, and `working / max_working`. `GET /agents/{name}` and
`flock status --agent` show an agent's block (`provider`) and the server it calls
(`endpoint`); `session_started` events carry `provider` and `model_id`.

## Not verified yet

- Cost on the HF router: some router providers stream zero usage, so their tokens show as $0
  with the zero-usage warning.
