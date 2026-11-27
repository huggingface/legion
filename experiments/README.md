# Experiments

Each folder is one swarm: a `spec.yaml` (plus its prompts) and, for most, a README with what it
needs, what it costs and how to launch it.

- [aops-math](aops-math/README.md): four agents solve one AoPS competition problem as a team sharing a message board; the container grades each agent's answer (a number, set, tuple, interval or expression) exactly.
- [cib](cib/README.md): one agent evaluated as a teammate on the Collaborative Intelligence Benchmark.
- [escape-room](escape-room/README.md): four agents solve one AgentEscapeBench room in local Docker, sharing a board.
- [mini-swarm](mini-swarm/README.md): three agents greet each other on a Space board, each in an HF Sandbox.
- [programbench](programbench/README.md): two agents implement csview and exchange source through a private HF dataset.
- [pybot](pybot/README.md): a swarm in ten lanes builds a Python chatbot through pull requests, on HF Jobs.
- [s3-aggregation](s3-aggregation/spec.yaml): agents each hold a private number and list the team's numbers in order ([docs](../docs/tasks.md#s3-aggregation)).
- [s3-maze](s3-maze/spec.yaml): agents search their own copies of one tree of rooms for its exit ([docs](../docs/tasks.md#s3-maze)).

## What goes in a folder

Everything legion-specific for one experiment lives in `experiments/<exp>/`; the dataset it runs
holds plain Harbor tasks only ([docs/tasks.md](../docs/tasks.md#where-things-live)).

```text
experiments/<exp>/
  spec.yaml        # the entry point: `legion launch experiments/<exp>`; variants as more specs
  team.md          # the spec's team_file, if any
  prompts/         # role and loop prompts, if any
  tools/           # the spec's tools_dir: legion tools only this experiment uses
  score.py         # scorers or metrics over a run's results, and sweep scripts, if any
  README.md        # what it needs, what it costs, how to launch it
```

- The spec loads with `Spec.load` and launches with `legion launch`. No other config format or
  a dataset's own CLI as the entry point.
- Pin the dataset with `task.revision`. Never point `tools_dir`, `team_file` or another path
  into a dataset download (`../../<dataset>/...`): the experiment then depends on where someone
  downloaded it and on files the pinned revision doesn't cover. Copy what legion needs here.
- Task text and `team.md` don't name tool commands (`legion-tool ...`); legion's agent
  instructions describe the tools.

## Where agents run

`environment.backend` picks where each agent's Harbor Trial runs; `compute` picks where legion's
workers run ([docs/local.md](../docs/local.md)). A spec has one `environment` block; uncomment the one you want:

```yaml
# Rootless podman on the worker's machine (the default): the one container engine on Slurm and
# the HF cluster (hopper-cpu nodes).
environment:
  backend: podman

# One HF Sandbox (an HF Job) per agent. Needs HF_TOKEN on the workers and a task with a
# public docker_image; billed to `namespace`, else HF_TOKEN's account.
# environment:
#   backend: hf-sandbox
#   flavor: cpu-basic   # HF Jobs hardware; cpu-basic is the default

# One E2B sandbox per agent. Needs E2B_API_KEY on the workers; billed to that key's
# E2B account. Not on Slurm.
# environment:
#   backend: e2b
```

The two sandbox backends have no mounts or local network: give their agents a Space or HTTP
board, and a model at a public URL. `compute: hf-jobs` workers need one of them.

### Local runs

`backend: docker` runs the Trials on a Docker engine with Compose, for `compute: local` on a
machine that already runs dockerd (escape-room, programbench). It is refused on Slurm, and the
HF cluster's nodes have no Docker. A spec without `backend` gets podman, so set it:

```yaml
environment:
  backend: docker
```
