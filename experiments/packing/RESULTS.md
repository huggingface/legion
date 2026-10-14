# Packing measurement (milestone 1, step 2)

Question: how many concurrent working turns can one `cpu-basic` worker (2 vCPU, 16 GB) sustain,
and how much does an idle resident session cost? Run 2026-10-14.

## What was run

One HF Job, `6acf468dc66288145223d03c` (`cpu-basic`, `node:22-bookworm`, Claude Code 2.1.296).
Scheduling took 2 s, the run 201 s. The container sees 16 host CPUs and 126 GB, but its cgroup
gives `cpu.max = 200000 100000` (2 vCPU) and `memory.max = 16 GB`.

```sh
cd experiments
./common/launch.sh packing-claude packing/job.sh -e LEVELS=5,10,20,40,60 -e IDLE=60
# job.sh -> python3 packing/packing.py --harness claude --levels 5,10,20,40,60 --idle 60 --base /work/pack --out /out
# each turn: claude -p --model claude-haiku-5-5 --max-budget-usd 0.05 --permission-mode bypassPermissions \
#            --output-format stream-json --verbose "Reply with the single word OK."   (own workdir, own new session)
```

`packing.py` starts N turns at once and walks its process tree in `/proc` every 0.5 s, recording
summed RSS, CPU ticks converted to cores, and process count. It also reads the container's
`memory.current`. Per turn it records wall latency, time to first stream event, and the
`result` event. N=60 was added beyond the requested 5–40 to look for the memory wall, which did
not appear. Raw samples and per-turn records are in `out/job/` (gitignored); level summaries are
in `claude-job-summary.json`.

## Claude Code, concurrent cold `claude -p` turns on cpu-basic

| N | ok | batch wall s | turn wall p50 / p95 / max s | first event p50 / max s | duration_api_ms p50 / p95 | CPU cores mean / max | CPU-s per turn | peak sum-RSS MB (per turn) | peak cgroup mem MB (per turn above baseline) | peak procs |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | 5 | 4.0 | 3.7 / 3.9 / 3.9 | 2.2 / 2.4 | 821 / 1,154 | 1.70 / 2.0 | 1.58 | 1,142 (228) | 1,031 (111) | 15 |
| 10 | 10 | 7.5 | 6.2 / 7.5 / 7.5 | 4.5 / 5.6 | 856 / 1,015 | 1.96 / 2.1 | 1.51 | 2,164 (216) | 1,471 (99) | 32 |
| 20 | 20 | 14.9 | 13.5 / 14.4 / 14.4 | 10.1 / 12.2 | 1,202 / 2,393 | 1.96 / 2.2 | 1.49 | 4,266 (213) | 2,408 (97) | 67 |
| 40 | 40 | 29.5 | 26.6 / 29.0 / 29.5 | 20.4 / 28.0 | 1,961 / 5,939 | 1.95 / 2.3 | 1.48 | 8,323 (208) | 4,074 (90) | 96 |
| 60 | 60 | 44.3 | 40.0 / 42.5 / 43.9 | 30.5 / 41.1 | 2,506 / 5,488 | 1.79 / 2.7 | 1.48 | 11,626 (194) | 5,560 (85) | 128 |

CPU-s per turn on the job = 2 vCPU × batch wall / N. The sum of per-process RSS counts shared
pages (the node binary and libraries) once per process. The cgroup figure is the real memory
cost and is about half of it.

## Idle resident Claude session (same job)

`claude -p --input-format stream-json --output-format stream-json --verbose`, sent one prompt,
then left alive for 60 s with stdin open:

| init event | result | duration_api_ms | RSS idle (mean / end) | cgroup mem above baseline | CPU while idle | procs | exit after stdin close |
|---|---|---|---|---|---|---|---|
| 0.84 s | 1.74 s | 873 | 230 / 229 MB | ~116 MB | 0.005 cores | 1 | 0.52 s |

## Cold vs resident CPU per turn (local, N=1, sequential)

`resident_cpu.py` ran 3 cold `claude -p` turns, then one streaming session with 4 prompts.
The local box has faster cores than cpu-basic.

| | CPU-s per turn | wall s |
|---|---|---|
| cold `claude -p` (3 turns) | 0.98, 0.94, 0.92 | 2.0–2.7 |
| resident, first prompt (most of startup included) | 0.85 | 2.0 |
| resident, prompts 2–4 | 0.07, 0.15, 0.08 | 0.9–7.1 (API latency) |

In the streaming session `duration_api_ms` and `total_cost_usd` are cumulative as well (2,020 ->
9,044 -> 10,573 ms).

## Codex, local only (N=3)

Ran with `CODEX_HOME` set to a fresh copy of `auth.json` and `config.toml`:
`python3 packing.py --harness codex --levels 3`, which runs
`codex exec --json -m gpt-6-luna -c model_reasoning_effort="low" --sandbox danger-full-access ...`.
This shared box has 8 vCPU, and its cgroup memory includes other agents' processes, so only the
per-tree numbers mean anything.

| N | ok | turn wall s | first event (`thread.started`) s | CPU-s per turn | peak sum-RSS MB (per turn) | procs |
|---|---|---|---|---|---|---|
| 3 | 3 | 6.8–7.3 | 3.7–3.9 | 1.48 | 788 (263) | 10 |

Codex in an HF Job is untested because there is no `OPENAI_API_KEY` here. `codex exec` reports
no API duration or cost, only tokens.

## What this means for worker defaults

- **CPU limits a cpu-basic worker; memory does not.** A cold Claude turn costs about 1.5 CPU-s,
  mostly CLI startup. Two vCPUs saturate from N=5 up, and throughput stays flat at about
  1.35 cold turns/s whatever N is. Latency grows linearly: about 0.7 s per extra concurrent
  cold start. 60 concurrent turns all succeeded, using 5.6 GB of cgroup memory.
- **Residency saves CPU, not prompt cache.** A turn sent to a live process costs about
  0.1 CPU-s against about 1 CPU-s cold (8–12x less). An idle resident process costs about
  116 MB of real memory and 0.005 cores. The design's "residency buys nothing" holds for model
  cost and does not hold for worker CPU. A long `idle_grace` is cheap, and memory allows about
  100 idle residents per worker with headroom.
- Per-process memory is lower than the design's 300 MB. Measured: 85–111 MB cgroup and
  190–230 MB RSS per working process. These are trivial turns; real turns that run tools (tests,
  builds) add whatever those tools use, which this experiment does not measure.
- Suggested defaults:
  - `agents_per_worker = 50`: 50 × 116 MB ≈ 6 GB resident, leaving about 10 GB for tool
    processes. Cold-restoring all 50 after a relaunch takes about 37 s of CPU.
  - `max_working_per_worker = 20`: 20 simultaneous cold starts reach first event in about 10 s
    p50. Above that, startup latency dominates.
  - `idle_grace` can be raised well above 120 s while memory allows it.
- Job overhead: scheduling took 2 s and bootstrap (`npm i -g` claude + uv + hf) took 6–9 s. A
  worker is ready about 10 s after launch, before any agent restore.

## Unverified
- CPU and memory of real working turns with tool calls. These numbers come from the
  smallest possible turn.
- Codex packing on cpu-basic, and the memory of a Codex app-server per agent.
- Whether API latency growth at high N (p95 5.9 s at N=40) is server-side or client event-loop
  starvation on a saturated CPU. The second is likely, since `duration_api_ms` is measured
  client-side.
