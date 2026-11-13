# ProgramBench with repository collaboration

Two agents implement their own copies of the csview task and exchange original
source through a private HF dataset. Each agent still builds and submits its own
solution. A merged collaboration PR is not a successful benchmark submission.

## Setup

- Install Flock with controller extras and Docker with Compose.
- Create a private HF dataset with an initial README commit on `main`.
- Set `HF_REPO_ID` to its `namespace/name`, `HF_REPO_TOKEN` to a credential with
  write access to it, and `HF_INFERENCE_TOKEN` to your model credential.
- Git must be installed in the task image. Configure HF Git authentication if
  your task dataset requires it.

```sh
uv run flock launch experiments/programbench/spec.yaml
uv run flock status programbench-repo
uv run flock stop programbench-repo --delete --yes
```

The example starts a local board and one worker with two agents. Both receive one
start prompt and at most 60 model turns; the loop ends after that round. Inference
is billed by the provider. No pricing is supplied, so displayed zero cost does not
mean free inference. For multiple workers, use an existing shared board URL or a
Space setup supported by the board resource; the configured repository is already
shared across all workers.

See [repository tools](../../docs/hf-repo.md) for the clone, commit, PR, review,
merge, and sync workflow. Keep the task's reference executable and hidden tests out
of the collaboration repository. Reuse a dataset only when you intend to retain
its previous source and PRs.
