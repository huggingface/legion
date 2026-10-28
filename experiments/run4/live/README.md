# run4 live infra snapshot (28 Oct 2026, ~20:00 UTC)

What run 4 was actually running on the science cluster, beyond the code on this branch. Copied from
`/fsx/carlos_patino/prj-eliza` so the setup can be replicated from git. No secrets here: they live in
`secrets/secrets.env` (and the liaison's board `token` file) and are only referenced by name.

Code in use: world `bee9cd1`, controller `60700c8`, worker `551df92` (all on `run4-setup`). The Slurm
scripts the jobs run (`slurm/*.sbatch`, `common.sh`, `*-inner.sh`) are identical to `experiments/run4/slurm/`.

| File | What it is |
|---|---|
| `run.env` | The live non-secret settings, sourced by `common.sh`. Differs from `slurm/run.env.example` on 14 keys: Hub slots (`CHATBOT_HUB_WRITE_SLOTS=1`, `CHATBOT_MERGE_HUB_SLOTS=1`), worker size (`WORKER_CPUS=6`, `WORKER_MEM=11G`; the example's 4 CPU / 7G OOMs), polling/sync (`WS_SYNC_S=60`, `FLOCK_COMMS_RATE=0.15`, `FLOCK_DRAIN_BUDGET_S=60`, `FLOCK_SYNC_EXCLUDE`), `CHATBOT_GET_PR_DIFF_BYTES`, the PR pause file, waves, model and vLLM URLs. |
| `serve-dsv41-flash-run4.slurm` | The vLLM job (2 × `hopper-b300` nodes, DeepSeek-V4.1-Flash, DP8 + EP, dspark speculative decoding, router). Nothing else in the repo starts inference. Open question: the metrics show all requests on DP engine 5 on both nodes. |
| `spec/spec.slurm.local.yaml` | The rendered spec applied with `flock update` (render.py + `waves.py pin`, replicas pinned agent 300 / agent-spawner 20 / agent-child 0, last update 08:52 from `bee9cd1`). |
| `spec/waves-hold.yaml`, `spec/waves-smoke.yaml` | The hold and smoke wave schedules used alongside `waves.yaml`. |
| `notices/*.txt` | Texts sent to agents with `flock prompt --swarm run4 --type <type> "<text>"`. Today: `open-budget-prompt.txt` (17:38) and `open-budget-prompt2.txt` (18:18), both to all three types. |
| `liaison/` | Operator tooling (plain bash/python over `runs/run4/logs/`). `job-monitor.sh`: job states, failed jobs, 429s, vLLM errors. `mem-watch.sh`: per-worker cgroup memory. `world-watch.sh`: 5-min `/manifest` probe, vote-failure trip-wire, relaunch setup. `hourly-stats.sh`: the hourly status numbers. `post.sh`, `post-once.sh`, `listen.py`: board I/O (read `liaison/token`). |

Operating rules used in run 4 (from `run-notes.txt`):

- **Worker drain:** at anon+shmem ≥ 9000 MB (or current ≥ 11000 MB with file ≤ 1200 MB), `scancel --signal=USR1 --full <job>`, sent right after a `sync cycle: 12 of 12` line. The controller relaunches the worker; restores take 18 min–2 h.
- **World trip-wire:** ≥ 3 `vote failed` in 15 min → touch the pause file, set 1 Hub write slot, snapshot, USR1 the world.
- **World restart:** when `/manifest` stays slow (≥ ~20 s) or memory climbs toward the limit. Snapshot `run4/world.tar.gz` to `run4-snapshots/`, then `scancel --signal=USR1 --full <world job>`; back in ~3 min.
- **Shutdown:** `flock loop --stop` per type, the pause file, `flock remove`, then TERM (signal only) to controller and world, then a final snapshot.
