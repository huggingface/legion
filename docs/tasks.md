# Adding Harbor tasks

Legion runs tasks from Hugging Face dataset repositories through Harbor. Each agent
gets its own task environment. A shared board or world is configured separately in
the Legion spec.

## Format references

Use the upstream documentation for the format and Hub conventions:

- [Harbor task format](https://docs.harborframework.com/tasks/overview): task files,
  environment configuration, reference solutions, and verifier rewards.
- [Hub RL environment dataset metadata](https://huggingface.co/docs/hub/datasets-cards#declare-an-rl-environment-dataset):
  the `rl-environment` and `harbor` tags for discoverability and framework compatibility.
- [TRL's Harbor dataset layout](https://huggingface.co/docs/trl/harbor#the-dataset):
  the `tasks/<task_id>/` layout in a Hugging Face dataset repository.

These are shared conventions; the sections below cover selecting and running a
task with Legion. Check compatibility with Legion's pinned Harbor version when using
features from newer upstream documentation.

## Publish a task

Put each self-contained task under `tasks/<task-name>/` in your dataset:

```text
tasks/my-task/
  instruction.md
  task.toml
  environment/
    Dockerfile
  tests/
    test.sh
  solution/           # optional reference solution
```

`instruction.md` explains the task to the agent. `task.toml` defines Harbor's
agent, environment, and verifier settings. The environment contains the task's
runtime; the verifier checks the submission and writes Harbor's reward output.
Use the Harbor task format supported by the version pinned in `pyproject.toml`.

Add a dataset card (`README.md` at the repository root) with this YAML metadata:

```yaml
---
tags:
  - rl-environment
  - harbor
---
```

The tags describe compatibility and help users find the dataset; they do not run
or validate its tasks. Document the task's requirements and license in the card.

Keep required files in the task or its declared image. A task must not depend on
paths from the publishing machine. Build and verify it with Harbor before
publishing, including a known successful submission and an incorrect submission.
Commit the task to the dataset and record the full commit SHA.

## Select it in Legion

Add this block to a working experiment spec:

```yaml
task:
  repo: https://huggingface.co/datasets/your-org/your-tasks
  revision: <published-commit-sha>
  selector: my-task
```

Without `dataset`, Harbor discovers tasks in the repository. For a Harbor registry,
set `dataset: name@version`; registry entries may point to their own task commits.
`selector` matches task names, by exact name or glob, and must select exactly one
task: every agent runs it. Pin `revision` to a commit to make the selection reproducible.

To give each agent its own task, make `selector` a list with one entry per agent. Each
entry must match exactly one task, and the first agent (`agent-001`) runs the first
entry's, `agent-002` the second's, and so on, in the list's order. This needs
`naming: consecutive-id`, `agents.count` equal to the list's length, and no roles. Use it
for private information per agent, or to run a few tasks of a multi-task dataset side by
side, one agent each:

```yaml
naming: consecutive-id
agents:
  count: 2
task:
  repo: https://huggingface.co/datasets/your-org/your-tasks
  revision: <published-commit-sha>
  selector: [task-a, task-b]  # agent-001 runs task-a, agent-002 task-b
```

The team stays one type on one worker and board, and can't be resized in a run. Each
agent's Trial gets only its own task (instruction, files and tests).

For a private dataset, the process resolving and downloading the task needs Git
credentials with read access. Configure your existing HF login for Git, for
example with `hf auth login --add-to-git-credential`. Do not put tokens in the URL
or commit them in the spec.

## Choose a compatible environment

- `environment.backend: docker` or `podman` runs task containers on the worker's
  machine. Install the corresponding container runtime and Compose support.
- `environment.backend: hf-sandbox` needs a prebuilt public `docker_image` in the
  task configuration. Local mounts and multi-container Compose tasks are not
  supported. Use a Space or existing HTTP board rather than a local socket board.
- `environment.backend: e2b` runs each Trial in an [E2B](https://e2b.dev/) sandbox.
  Export `E2B_API_KEY` where the workers run. As on hf-sandbox, there are no mounts
  or local networks: use a Space or existing HTTP board.
- HF Jobs workers use HF Sandboxes or E2B. Slurm workers use Podman or HF Sandboxes.
  `compute` chooses where workers run; `environment.backend` chooses where their
  task environments run.

Start with one agent and the smallest useful task before scaling up. Check task
startup, a model turn, tool access, and the task's verifier output. Legion keeps a
Trial alive across prompts; task completion and grading follow the task and Trial
lifecycle, while a spec's loop controls when agents receive more prompts.

See [local usage](local.md) for launching and inspecting a run, and
[escape-room](../experiments/escape-room/) for an existing task integration.

## s3-aggregation

The private [s3-aggregation dataset](https://huggingface.co/datasets/HuggingFaceSwarms/s3-aggregation)
adapts the information-aggregation task of [s³](https://arxiv.org/abs/2611.08615): each of N
agents holds a private number, and the team must list them all in ascending order. A team has
one task per slot, `agg-pPP-nNN-sKK`, whose instruction states that slot's number, and the
[example](../experiments/s3-aggregation/spec.yaml) gives each agent its own: its selector
lists the team's N slots, `<team>-s01` to `<team>-sNN` (see above). It runs one team on HF
Sandboxes, with the board on a public Space chosen for that run. `HF_TOKEN` needs dataset read
access, Jobs permission, and create/write access to the board's Space and storage bucket.
`HF_INFERENCE_TOKEN` is the agents' inference key. `TEAM` picks another team of four (e.g.
`TEAM=agg-p07-n04`); for another size, set `N` and list that many slots. The dataset README
covers the cluster, the board rules, the scaling
sweep and the metrics.

The example omits `namespace`: sandboxes use `HF_TOKEN`'s personal account and router calls
use the inference key's default billing account with no bill-to header. `BOARD_SPACE` is
required: choose a fresh bare name for each independent run; it resolves under `HF_TOKEN`'s
account. Dataset read access to `HuggingFaceSwarms/s3-aggregation` is separate from personal
Jobs, Space and bucket permissions. See [billing and storage](billing.md).
The default local workers do not need a run bucket. For HF Jobs or Slurm workers, set
`bucket_id: legion-runs`: launch qualifies it as `<your-username>/legion-runs` and creates
the bucket private if missing. A qualified ID selects storage explicitly. Use
`[namespace/]name`, without a bucket URL or path prefix.

```sh
export BOARD_SPACE="s3-aggregation-$(uv run python -c 'import uuid; print(uuid.uuid4().hex[:12])')"
uv run legion launch experiments/s3-aggregation/spec.yaml
```

Retain that value when resuming the same run. After all its workers have stopped, you may
delete a board newly created for this run; leave pre-existing and shared boards in place.

## s3-maze

The private [s3-maze dataset](https://huggingface.co/datasets/HuggingFaceSwarms/s3-maze) adapts
the parallel-exploration task of [s³](https://arxiv.org/abs/2611.08615): each of N agents
searches its own copy of one tree of rooms for the exit with `maze look` and `maze move ROOM`,
on a budget of 210 moves that one agent rarely finds it in, and the board lets the team split
the search. Each agent's Trial is a compose stack, its container and a `maze` service that alone
holds the tree, so it runs on docker or podman, not HF Sandboxes. The
[example](../experiments/s3-maze/spec.yaml) runs one team (`TREE`, default `maze-t01`, and `N`,
default 4) on the HF cluster: one Slurm worker on hopper-cpu in rootless podman, with a local
board, and `stop_on_success` with the service's `success_check`, so the first agent out ends
the run. The example includes a protocol hint in its team text (`team.md`, the spec's
`team_file`), so each agent reads the board and claims a door before it goes through one; remove
`team_file` for the neutral condition of the earlier sweep. It needs the dataset on the shared
disk for its `registries.conf` (podman pulls Docker Hub images through `mirror.gcr.io`),
`HF_TOKEN` with read access to `HuggingFaceSwarms/s3-maze` and bucket create/write access in
your personal account, a separate `HF_INFERENCE_TOKEN` with personal Inference Providers
permission for the agents, and
the Git credential helper for a private task repo ([Slurm](slurm.md), "Containers").

The example omits `namespace`, so router calls use the inference key's default personal
billing account with no org bill-to header. Set `BUCKET_ID` to a bare name such as
`legion-runs` (resolved under `HF_TOKEN`'s account) or a qualified `<your-username>/<name>`.
Launch creates a missing bucket private and retains an existing bucket's visibility.
See [billing and storage](billing.md) for personal permissions and optional org billing.

```sh
DS=/fsx/$USER/s3-maze
hf download HuggingFaceSwarms/s3-maze --repo-type dataset --revision 3b6c363d92626bd4d97a7ca89964fb850982d9ab --local-dir $DS
printf 'SLURM_PARTITION=hopper-cpu\nSLURM_MAX_MEM_PER_CPU_MB=2000\nVENV=%s\nCONTAINERS_REGISTRIES_CONF=%s\n' \
  "$PWD/.venv" "$DS/legion/registries.conf" > cluster.env
# secrets.env (mode 600): HF_TOKEN, HF_INFERENCE_TOKEN, BUCKET_ID
uv run legion init $PWD/runs/maze-t01-n04 --from experiments/s3-maze --profile cluster.env --secrets secrets.env
echo RUN_NAME=maze-t01-n04 >> runs/maze-t01-n04/run.env
export LEGION_RUN_DIR=$PWD/runs/maze-t01-n04
uv run legion launch $LEGION_RUN_DIR/experiment
uv run legion status maze-t01-n04            # STOPPED ON SUCCESS once an agent is out
uv run legion stop maze-t01-n04 --release    # ends the controller, or a run nobody won
uv run python $DS/eval/metrics.py $LEGION_RUN_DIR
```

The worker copies each agent's grading, with its move log, and the board export to
`<run dir>/results/`, where `metrics.py` reads them. The dataset README covers the move budget,
the grader, the checks and the metrics.
