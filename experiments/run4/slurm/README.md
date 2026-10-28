# Run 4 Slurm templates

Start with ../README.md and run.env.example. These scripts are the liaison's actual
run3 scripts adapted to chatbot4, the Pi-only type `agent`, 54-hour lifetimes,
per-role code pins and one-archive world snapshots. They have not been submitted.

Use `run3-reference.md` only for the proven signal/requeue procedures and cluster
constraints. Its old run3 paths, world/probe settings and vLLM address are historical.
Copy these templates to the cluster's operational slurm directory before use;
the scripts source `/fsx/carlos_patino/prj-eliza/slurm/common.sh` at runtime.
