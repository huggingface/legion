# Packing measurement on cpu-upgrade

Question: how many concurrent working turns can one `cpu-upgrade` worker (8 vCPU, 32 GB,
$0.03/h) sustain, and does 200 agents per worker fit in memory? This repeats the
[cpu-basic run](RESULTS.md) for the 1k load test. Run 2026-10-14, branch `load-1k`.

## What was run

One HF Job, `6acf64235ccb47ba0f799ce4` (`cpu-upgrade`, `node:22-bookworm`, Claude Code
2.1.296, label `experiment=swarm-harness-1k`, timeout 30 m). Bootstrap took 8 s. The
container sees 64 host CPUs and 505 GB, but its cgroup gives `cpu.max = 800000 100000`
(8 vCPU) and `memory.max = 32 GB`.

```sh
cd experiments
FLAVOR=cpu-upgrade TIMEOUT=30m LABEL=experiment=swarm-harness-1k ./common/launch.sh packing-cpu-upgrade \
  packing/job.sh -e LEVELS=10,20,40,80,120 -e IDLE=60 -e DEST=hf://buckets/cmpatino/swarm-harness/packing/cpu-upgrade
```

`launch.sh` gained `FLAVOR` and `LABEL`. Otherwise this is the same `packing.py`: N concurrent
cold `claude -p --model claude-haiku-5-5 "Reply with the single word OK."` turns, each with its
own workdir and a new session. The script samples the process tree and the cgroup every 0.5 s.
The summary is in
[`claude-job-summary-cpu-upgrade.json`](claude-job-summary-cpu-upgrade.json); raw samples are
in `hf://buckets/cmpatino/swarm-harness/packing/cpu-upgrade/`.

## Claude Code, concurrent cold `claude -p` turns on cpu-upgrade

| N | ok | batch wall s | turns/s | turn wall p50 / p95 / max s | first event p50 / max s | duration_api_ms p50 / p95 | CPU cores mean / max | CPU-s per turn (rusage / 8 x wall / N) | peak sum-RSS MB (per turn) | peak cgroup mem MB (per turn above baseline) | peak procs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 10 | 2.3 | 4.4 | 2.2 / 2.3 / 2.3 | 0.8 / 0.8 | 898 / 993 | 5.7 / 8.3 | 1.12 / 1.82 | 2,524 (252) | 1,852 (136) | 34 |
| 20 | 20 | 4.3 | 4.7 | 3.8 / 4.3 / 4.3 | 2.1 / 2.3 | 1,024 / 1,510 | 6.2 / 8.3 | 1.28 / 1.71 | 4,598 (230) | 2,749 (113) | 80 |
| 40 | 40 | 8.1 | 4.9 | 7.6 / 8.0 / 8.1 | 5.6 / 6.2 | 1,101 / 1,501 | 7.8 / 8.6 | 1.54 / 1.62 | 8,973 (224) | 4,770 (107) | 159 |
| 80 | 80 | 18.9 | 4.2 | 16.9 / 18.0 / 18.7 | 12.2 / 16.7 | 1,699 / 3,696 | 7.8 / 13.2 | 1.80 / 1.89 | 16,976 (212) | 8,213 (96) | 247 |
| 120 | 120 | 28.3 | 4.2 | 22.1 / 27.1 / 27.5 | 17.4 / 26.0 | 1,695 / 3,202 | 6.4 / 9.5 | 1.82 / 1.89 | 23,019 (192) | 10,483 (83) | 258 |

The CPU max above 8 is a sampling artefact: ticks are read per process, 0.5 s apart.
"rusage" is the children's CPU time divided by N. "8 x wall / N" is the cpu-basic table's
definition, which assumes the cores are saturated for the whole batch.

## Idle resident Claude session (same job)

| init event | result | duration_api_ms | RSS idle (mean / end) | cgroup mem above baseline | CPU while idle | procs | exit after stdin close |
|---|---|---|---|---|---|---|---|
| 0.91 s | 1.92 s | 978 | 246 / 268 MB | ~129 MB (484 -> 613 MB) | 0.025 cores | 1 | 0.46 s |

## Against the scaling estimate

- **Throughput is 4.2 to 4.9 cold turns/s, not the 5.4 predicted** from 4 x the cpu-basic
  1.35/s. It peaks at N=40 (4.9/s) and settles at 4.2/s for N >= 80. CPU per cold turn is
  1.6 to 1.9 CPU-s here against about 1.5 on cpu-basic. The likely cause is contention between
  8 cgroup CPUs on a 64-CPU host (scheduling and cache effects with 250 processes). This is not
  verified.
- Below N=20 the worker is not CPU-bound: 10 turns finish in 2.3 s on 5.7 cores, so latency is
  the API round trip plus CLI startup.
- Latency grows by about 0.22 s per extra concurrent cold start, against 0.7 s on cpu-basic.
  The 4x core count shows up as about 3.2x here.
- Memory is not the limit. 120 concurrent turns used 10.5 GB of cgroup memory (83 MB per turn
  above baseline). Every turn succeeded.

## Defaults for cpu-upgrade

- **`max_working_per_worker = 120`.** Turn p95 is 27.1 s at N=120, the largest measured N with
  p95 under 30 s. Extrapolating at 0.22 s per extra turn puts the 30 s line at about N=130.
  First event p50 is 17 s at that load. If a caller cares about time to first token, 40
  (first event p95 about 6 s) is the comfortable value. The setting still only warns; the
  worker does not queue.
- **`agents_per_worker = 200` is sane by memory.** An idle agent has no process. Within
  `idle_grace`, a resident session costs about 129 MB of cgroup memory, and a working turn
  83 to 136 MB. Worst case, all 200 agents resident at once is 200 x 129 MB, about 26 GB, which
  fits in 32 GB with about 5 GB left for the worker and tool processes. 120 working plus 80
  resident is about 10.5 + 10.3 = 21 GB. CPU sets the real ceiling: cold-starting all 200 at
  once takes about 200 / 4.2 = 48 s of saturated CPU, so first events p95 would be about 45 s.
  Real turns that run tools (tests, builds) add their own memory, which this experiment does not
  measure. At those workloads 200 per worker leaves little headroom.
- `idle_grace`: as on cpu-basic, residency costs memory and saves CPU (about 0.1 CPU-s for a
  warm turn against about 1.8 cold). At 200 agents per worker, keep `idle_grace` short enough
  that the resident set stays well under about 200 (`60s` in the load test).
- Cost per cold-turn capacity: cpu-upgrade gives 3.1 to 3.6x the cold-turn throughput of
  cpu-basic for 3x the price ($0.03/h vs $0.01/h). It also holds 4x the agents per job, so a
  10k swarm needs 4x fewer jobs, fewer Hub calls and fewer heartbeats.
