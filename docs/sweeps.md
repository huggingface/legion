# Sweeps on Slurm

`legion sweep` runs an independent matrix of pinned Harbor experiments. It uses
the regular `legion init`, launch and release lifecycle for each cell. Run the
driver on a compute node when the cluster limits long-running login processes.

Use a cluster profile and a private bucket as for a [single Slurm run](slurm.md).
The [s³ aggregation example](../experiments/s3-aggregation/sweep.yaml) uses two
team sizes, exact per-agent task selectors and an existing vLLM endpoint.

```sh
export BUCKET_ID=your-namespace/private-run-bucket
export VLLM_URL=http://model-host:8000/v1
uv run legion sweep experiments/s3-aggregation/sweep.yaml --profile cluster.env --plan
uv run legion sweep experiments/s3-aggregation/sweep.yaml --profile cluster.env \
  --secrets secrets.env --root runs/aggregation-sweep
```

The YAML names a base `spec` relative to the matrix file, defaulting to
`spec.yaml`. Common `overrides` apply first. Each `matrix` axis names one or
more choices, each of which can override multiple spec fields. Axes are applied
in name order. Mappings merge, lists and scalars replace, and `null` removes a
field from the base. The matrix expands into one cell per combination.

Give each role a loop and pin `task.revision` to a commit SHA. The sweep waits
for every planned agent to finish and grade. `max_runs` bounds concurrent
attempts and `max_agents` optionally bounds their total agent reservation.
`run_timeout` includes startup and defaults to one hour.

Use an output directory outside the experiment folder. Run the same command
again to resume. Legion checks the inputs against the saved plan, adopts
existing attempts before starting new cells, and skips completed cells. It
pauses if a controller submission or shutdown cannot be verified. Check the
attempt's Slurm state and logs before resuming.

`infrastructure_retries` defaults to zero. A confirmed infrastructure failure
can get a new attempt directory and run name with an incremented `-rN` suffix.
Legion checks worker accounting and verifier artifacts after release. A graded
zero counts as a completed result. Attempts with replaced workers are kept
separate because their artifact copies can be incomplete. [Slurm result
preservation](https://github.com/huggingface/legion/issues/121)

`rows.jsonl` has one row per cell with its task pin, input fingerprint, attempt
history and the selected completed artifact directory. Each experiment can
read the verifier files there to compute its team and sweep scores.
