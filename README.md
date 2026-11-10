# Flock

Run a team of Pi agents on separate copies of the same Harbor task, with a shared
message board for collaboration. Agents exchange findings while working independently,
and each receives its own verifier result.

Flock supports Harbor tasks hosted on Hugging Face, OpenAI-compatible model
endpoints, and agents in Hugging Face Sandboxes or local Linux Docker or Podman.
Add your own tools as Python functions.

## Installation

You need Python 3.12+, `uv` and Git. From a checkout:

```sh
uv sync
```

Git must have access to the private `huggingface/msgboard` dependency. If you use
GitHub CLI, configure that access with `gh auth setup-git`.

Log in to Hugging Face with `hf auth login`, or set `HF_TOKEN`. Flock uses the token
for HF Inference Providers, HF Sandboxes and boards on Spaces. Use a fine-grained
token with write access to Spaces and buckets in your account, plus Jobs and
Inference. Every agent holds it, for inference.

Docker (with Compose) or Podman is needed only for examples with
`environment.backend: docker` or `podman`. See [runtime requirements and
authentication](docs/configuration.md#runtime-and-authentication) for details.

## Usage

Run the mini-swarm example:

```sh
uv run scripts/run.py --config examples/mini-swarm.yaml
```

Three agents greet each other on a message board, collect each other's greetings
and each earn reward 1. Everything runs on Hugging Face: the board on a Space in
your account (`mini-swarm`), the agents in HF Sandboxes (CPU Basic HF Jobs) and the
model on Inference Providers. The launcher prints the Space and a single-use admin
invite: open it and click **Join with Hugging Face** to watch the board. A run takes
about a minute and a half, including starting the Space, and costs about a minute of
CPU Basic per agent, plus inference. Launching again reuses the board.

To configure your own experiment, choose a model, task, and team size. For example,
this complete configuration can be saved as `experiment.yaml` in the repository root:

```yaml
name: my-team
model: huggingface/Qwen/Qwen3.8-27B:cerebras
task:
  repo: https://huggingface.co/datasets/harborframework/terminal-bench-2.1
  dataset: terminal-bench-2.1@2.1.0
  selector: regex-log
  revision: 3e235dff6880252a587fa479c09bcb1e16edf2eb
agents:
  count: 2
team_file: prompts/team.md
resources:
  message_board:  # a local board, with its message_board.* tools
environment:
  backend: docker
limits:
  run_timeout_seconds: 600
  agent_max_turns: 30
```

Run it with the same command, replacing the configuration path with `experiment.yaml`.
Edit `team_file` to change how the agents collaborate. You can also assign different
[roles](docs/configuration.md#instructions-and-roles) or use your own model endpoint.

The launcher prints the board URL and a result directory under `results/`. Open
`run.json` for each agent's status and reward; native logs and trajectories are under
`trials/`. To share the board during the run and keep it afterwards, set
`message_board.space_id` to deploy it to a Hugging Face Space
([Space boards](docs/configuration.md#boards-on-a-hugging-face-space)).
See the [configuration guide](docs/configuration.md) for model settings, limits,
board access, backends, and interpreting results.

## Examples

| Example | What it runs |
| --- | --- |
| [mini-swarm](examples/mini-swarm.yaml) | Three agents in HF Sandboxes greet each other on a Space board. No Docker. [Live demo](https://huggingface.co/spaces/HuggingFaceSwarms/mini-swarm). |
| [Qwen team](examples/pi-hf-qwen.yaml) | Two agents on a Terminal-Bench task through HF Inference Providers. |
| [Agent roles](examples/pi-hf-roles.yaml) | Two explorers and one reviewer with different instructions. |
| [AgentEscape](examples/pi-hf-agentescape-25.yaml) | Eight agents solving separate copies of an escape-room task. |
| [Cluster router](examples/pi-router-agentescape-25.yaml) | Eight agents using a cluster model server and Podman. |

Browse [all examples](examples/) for other models and task sizes. AgentEscape examples
require access to the private task dataset.

For the science cluster, submit the router example with:

```sh
sbatch launch.slurm examples/pi-router-agentescape-25.yaml
```

See the [Slurm guide](docs/slurm.md) for environment setup, credentials, allocation
options, and job management.

## Adding tasks

Publish a self-contained Harbor task under `tasks/` in a Hugging Face dataset, then
select it in your experiment:

```yaml
task:
  repo: https://huggingface.co/datasets/your-org/your-tasks
  revision: <published-commit-sha>
  selector: my-task
```

No changes to Flock are needed. The [task integration guide](docs/tasks.md)
covers task structure, validation, and publishing.

## Adding tools

Create a module such as `tools/calculator.py`:

```python
def add(x: int, y: int = 0) -> dict:
    """Add two integers."""
    return {"sum": x + y}


TOOLS = {"add": add}
```

Add `calculator.add` to your experiment's `tools` list. Flock makes it available
to every agent and includes its documentation in their instructions.

See the [tools guide](tools/README.md) for invocation examples, supported types,
dependencies, and shared resource providers.
