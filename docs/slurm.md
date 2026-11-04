# Running on Slurm

[Back to the README](../README.md)

On the science cluster, keep the checkout and installed `.venv` on `/fsx`. Set
`environment.backend: podman`, export `HF_TOKEN` for private task access and the
inference credential named in the YAML, then submit from the repository root:

```sh
sbatch launch.slurm examples/pi-router-agentescape-25.yaml
```

The launcher uses one `hopper-cpu` node with 28 CPUs, 54 GiB, and a two-hour allocation
by default. Agents share the CPUs. Override the allocation with standard Slurm flags:

```sh
sbatch --cpus-per-task=8 --mem=24G --time=01:15:00 \
  launch.slurm examples/pi-router-agentescape-25.yaml
```

Edit the router example's endpoint for your model server. Credentials are inherited
from the submitting shell; `HF_TOKEN` is supplied to Git for Hugging Face access.
The launcher uses the existing environment and isolates Podman storage in job-local
`/tmp`. It schedules native health checks without requiring a user systemd session.

Logs go to `flock-JOBID.log`; experiment artifacts go to the configured
`output_dir`. Inspect jobs with `squeue --me` and `sacct -j JOBID`, or cancel with
`scancel JOBID`. CPU nodes use Spot instances: reclamation interrupts the run.
Automatic requeue and experiment resume are not supported; partial artifacts on
shared storage are retained. Multi-node experiments are not supported.


See the [configuration guide](configuration.md) for credentials, results, and failure handling.
