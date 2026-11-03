# Run 4 Slurm templates

Start with ../README.md and run.env.example. These scripts are the liaison's actual
run3 scripts adapted to chatbot4, the Pi-only type `agent`, 54-hour lifetimes,
per-role code pins and one-archive world snapshots. They have not been submitted.

Use `run3-reference.md` only for the proven signal/requeue procedures and cluster
constraints. Its old run3 paths, world/probe settings and vLLM address are historical.
Copy these templates to the cluster's operational slurm directory before use;
the scripts source `/fsx/carlos_patino/prj-eliza/slurm/common.sh` at runtime.

## Resilience and placement

- `world.sbatch` requeues on a crash only when it was up less than `WORLD_STABLE_S` (600 s); a
  longer run resets `<run>/world/crash-requeues`. After 5 fast crashes it sleeps with backoff
  (5 min doubling to `WORLD_RETRY_MAX_S`, 30 min) and requeues itself again instead of ending.
- `controller.sbatch` does not wait for the world on start or requeue. Only the first apply waits
  (`WORLD_WAIT_S`); if the world is still down it logs and leaves the controller running.
- Spot-only capacity is enough. If the cluster has a non-preemptible partition or QOS, set
  `WORLD_PARTITION`/`WORLD_QOS` and `CTL_PARTITION`/`CTL_QOS` in run.env and pass them on the
  sbatch line (see the top of each script); unset keeps the `#SBATCH` header.

## Ramp

The controller job applies `experiments/$RUN3_EXPERIMENT/spec.yaml` only when the swarm is not on
the controller (404); a requeue (200) applies nothing, so it never undoes a pause or resets live
counts. The controller then raises replicas along the spec's `replicas_ramp` (phases, `stagger`,
`hold_if`: docs/live.md "Ramp"); its clock is in the controller state, so a requeue continues it.
