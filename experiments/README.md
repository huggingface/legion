# Experiments

Each folder is one swarm: a `spec.yaml` (plus its prompts) and, for most, a README with what it
needs, what it costs and how to launch it.

- [aops-math](aops-math/README.md): agents solve one AoPS competition problem, alone or as a team sharing a message board; the team's majority-vote answer is compared with the same number of independent agents. Each final answer (a number, set, tuple, interval or expression) is graded by an exact grader, plus an LLM judge for structured answers it rejects.
- [cib](cib/README.md): one agent evaluated as a teammate on the Collaborative Intelligence Benchmark.
- [escape-room](escape-room/README.md): four agents solve one AgentEscapeBench room in local Docker, sharing a board.
- [mini-swarm](mini-swarm/README.md): three agents greet each other on a Space board, each in an HF Sandbox.
- [programbench](programbench/README.md): two agents implement csview and exchange source through a private HF dataset.
- [pybot](pybot/README.md): a swarm in ten lanes builds a Python chatbot through pull requests, on HF Jobs.
- [s3-aggregation](s3-aggregation/spec.yaml): agents each hold a private number and list the team's numbers in order ([docs](../docs/tasks.md#s3-aggregation)).
- [s3-maze](s3-maze/spec.yaml): agents search their own copies of one tree of rooms for its exit ([docs](../docs/tasks.md#s3-maze)).

## Where agents run

`environment.backend` picks where each agent's Harbor Trial runs; `compute` picks where legion's
workers run ([docs/local.md](../docs/local.md)). A spec has one `environment` block; uncomment the one you want:

```yaml
# Docker on the worker's machine (the default). Needs Docker with Compose; not on Slurm.
environment:
  backend: docker

# Podman on the worker's machine, rootless: the one container engine on Slurm.
# environment:
#   backend: podman

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
