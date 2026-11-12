# Adding Harbor tasks

Flock runs tasks from Hugging Face dataset repositories through Harbor. Each agent
gets its own task environment. A shared board or world is configured separately in
the Flock spec.

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

Keep required files in the task or its declared image. A task must not depend on
paths from the publishing machine. Build and verify it with Harbor before
publishing, including a known successful submission and an incorrect submission.
Commit the task to the dataset and record the full commit SHA.

## Select it in Flock

Add this block to a working experiment spec:

```yaml
task:
  repo: https://huggingface.co/datasets/your-org/your-tasks
  revision: <published-commit-sha>
  selector: my-task
```

Without `dataset`, Harbor discovers tasks in the repository. For a Harbor registry,
set `dataset: name@version`; registry entries may point to their own task commits.
`selector` matches task names and must select exactly one task. Pin `revision` to
a commit to make the selection reproducible.

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
- HF Jobs workers use HF Sandboxes. Slurm workers use Podman or HF Sandboxes.
  `compute` chooses where workers run; `environment.backend` chooses where their
  task environments run.

Start with one agent and the smallest useful task before scaling up. Check task
startup, a model turn, tool access, and the task's verifier output. Flock keeps a
Trial alive across prompts; task completion and grading follow the task and Trial
lifecycle, while a spec's loop controls when agents receive more prompts.

See [local usage](local.md) for launching and inspecting a run, and
[escape-room](../experiments/escape-room/) for an existing task integration.
