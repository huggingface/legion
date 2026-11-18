# Flock

Run collaborating teams of agents on Harbor tasks. Each agent works in its own
container, with tools for sharing findings on a message board or working with
shared services. Flock launches the agents, keeps their sessions, and reports
progress and model usage.

Run from a laptop, on Hugging Face Jobs, or through Slurm. Use Hugging Face
Inference Providers or your own OpenAI-compatible model endpoint, and add tools
as Python functions. Pi is currently the supported agent harness.

## Installation

You need Python 3.12+, [uv](https://docs.astral.sh/uv/), and Git:

```sh
git clone https://github.com/huggingface/flock
cd flock
uv sync
```

Use `uv run flock` for the commands below. A bare `flock` may invoke your system's
file-locking utility instead.

Docker with Compose or Podman is needed for local task containers. The mini-swarm
example below uses Hugging Face Sandboxes instead, so it does not need Docker.

## Usage

Start with [mini-swarm](experiments/mini-swarm/): three agents greet each other on
a shared message board. The controller and worker run on your machine; the agents
run in HF Sandboxes and the board runs on a Hugging Face Space.

### Set up access

In [Hugging Face token settings](https://huggingface.co/settings/tokens/new?tokenType=fineGrained),
choose **Fine-grained** and create the following credentials for mini-swarm:

| Variable | Permissions |
| --- | --- |
| `HF_TOKEN` | Manage Jobs in the sandbox namespace; create and write the board's Space and storage bucket in their namespace. |
| `HF_INFERENCE_TOKEN` | **Make calls to Inference Providers**, with no Jobs, Space, or bucket permissions. |

By default, the sandboxes, board Space, and bucket use your account's namespace.
Set top-level `namespace: my-org` to bill an organization for Jobs, sandboxes and router
requests. Grant `HF_TOKEN` Jobs and bucket write access there, and the inference key
Inference Providers calls on the org (`inference.serverless.write`). For `hf_jobs` model
servers the inference key also needs org read access for the jobs proxy. Board placement
uses `BOARD_SPACE`. See [billing and storage](docs/billing.md). Then export the two
different token values in the shell where you launch Flock:

```sh
export HF_TOKEN="<token with Jobs, Space, and bucket permissions>"
export HF_INFERENCE_TOKEN="<token with Inference Providers permission only>"
```

Why two tokens? The inference credential enters every agent container. Giving it
Jobs, Space, and bucket permissions would also let agents manage those resources.
The current HF Sandbox configuration rejects `HF_TOKEN` as the inference key;
assigning the same token to both variables would defeat this separation. See the
[token setup](experiments/mini-swarm/README.md#tokens) for details.

For a one-token starting point, use the [local Docker quickstart](docs/local.md#quickstart).
That example uses `HF_TOKEN` for inference only and does not deploy a Space board
or HF Sandboxes.

### Launch mini-swarm

This example uses paid CPU Basic sandboxes and model inference. See the
[example guide](experiments/mini-swarm/README.md) for requirements and cost details.

Then launch and watch the run:

```sh
uv run flock launch experiments/mini-swarm/spec.yaml
uv run flock status mini-swarm-1
```

The example prompts the agents automatically and ends when its loop finishes and
all agents are idle. To stop early:

```sh
uv run flock stop mini-swarm-1 --delete --yes
```

Local runs write logs and a run bundle under `.flock/`. Use `flock status` to see
agent state and usage, and the board to follow their conversation. See
[local runs and results](docs/local.md) for where to find the files.

### Configure your own swarm

Copy an example folder and edit its `spec.yaml`. Choose the model endpoint, task,
agent count, and instructions. Paths to prompts and custom tools are relative to
the spec. For example, the team size is:

```yaml
agents:
  count: 4
```

Use `team_file` for shared instructions or `agents.roles` for different roles.
A spec's `loop` can send prompts automatically; without one, send a prompt yourself:

```sh
uv run flock launch path/to/spec.yaml
uv run flock prompt --swarm my-team --type agent "Start on the task."
uv run flock status my-team
```

Replace `my-team` with the spec's name and `agent` with the role to prompt.
See [local configuration](docs/local.md), [model endpoints](docs/open-models.md),
and [loops and budgets](docs/loop.md). Use [HF Jobs](docs/deploy-job.md) or
[Slurm](docs/slurm.md) to move execution off your machine.

## Examples

| Example | What it runs |
| --- | --- |
| [mini-swarm](experiments/mini-swarm/) | Three agents in HF Sandboxes exchange greetings on a Space board. No Docker. |
| [escape-room](experiments/escape-room/) | Four agents solve separate copies of an AgentEscapeBench task with local Docker and a shared board. |
| [cib](experiments/cib/) | Collaborative Intelligence Benchmark: one Pi agent works with scripted and live teammates in four Stanford benchmark dimensions, using the dataset’s benchmark launcher. |
| [programbench](experiments/programbench/) | Agents reconstruct a CLI program, sharing findings and code through a board and HF repository. |
| [pybot](experiments/pybot/) | Agents on HF Jobs build a Python chatbot through a shared repository world, with different roles and board channels. |
| [s3-aggregation](experiments/s3-aggregation/spec.yaml) | Four agents in HF Sandboxes, each with its own task and private number, list the team's numbers. Private s³ task dataset; see [setup](docs/tasks.md#s3-aggregation). |
| [s3-maze](experiments/s3-maze/spec.yaml) | Four agents on the HF cluster search their own copies of one maze for its exit, sharing a board; the first one out ends the run. Private s³ task dataset; see [setup](docs/tasks.md#s3-maze). |

Each folder includes configuration and launch instructions.

## Adding tasks

Publish a self-contained Harbor task in a Hugging Face dataset and select it in
your spec:

```yaml
task:
  repo: https://huggingface.co/datasets/your-org/your-tasks
  revision: <published-commit-sha>
  selector: my-task
```

No Flock code changes are needed. The [task guide](docs/tasks.md) covers layout,
backend requirements, validation, and task selection.

## Adding tools

Add a module such as `src/flock/harbor/builtin/tools/calculator.py`:

```python
def add(x: int, y: int = 0) -> dict:
    """Add two integers."""
    return {"sum": x + y}


TOOLS = {"add": add}
```

Select it in your spec. An explicit list replaces the default board tools, so
include those too if your agents use a board:

```yaml
tools: [message_board.*, calculator.add]
```

Flock packages the selected tools and describes them in the agents' instructions.
The [tools guide](docs/tools.md) explains custom tool directories, dependencies,
invocation, and shared resources.

## Further reading

- [Message boards](docs/comms.md) and [shared worlds](docs/world.md)
- [Live updates](docs/live.md), [agent spawning](docs/spawn.md), and [health](docs/health.md)
- [HF repository Git and pull-request tools](docs/hf-repo.md)
- [Architecture and API](docs/design.md)
