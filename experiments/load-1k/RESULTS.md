# 1k load test

Question: does swarm-harness hold up at 1,000 agents on five `cpu-upgrade` workers, and what
must change before 10k? Run 2026-10-14, 11:30 to 12:39 UTC, branch `load-1k`. There were 1,000
Claude haiku agents of two types, 200 per worker. All 1,000 were prompted twice from one client.
One worker was killed by hand, four workers drained at their deadline, then came a scale-down,
a remove, and a short re-run of the drain after two fixes.

## Verdict

| # | Check | Result | Key numbers |
|---|---|---|---|
| 1 | Apply: 5 workers in one pass, heartbeats, bootstrap, 1,000 idle | **PASS** | 5 `run_job` in the first pass (all created 11:30:34); bootstrap 12-34 s; first heartbeat +28.5 to +46.4 s after apply; 1,000 idle at +52 s |
| 2 | Prompt all 1,000 at about 50 in flight; latency, request rate, cost, status page | **FAIL (infra), PASS on round 2** | Round 1: the HF Spaces edge rate-limited the client IP (2,242 x 429; 839 prompts acked, 161 never sent). Round 2 with a leaner client: 1,000/1,000 `turn_ended`, 999/1,000 correct (1 model misread), 0 HTTP errors; first event p50 1.2 / p95 2.3 s, turn p50 2.9 / p95 4.9 s; 13.6 req/s at the controller, 0 5xx; controller $2.000099 vs Claude $2.000101; status page 0.054 s |
| 3 | Bucket after the first sync; per-worker sync cost | **PASS** | 1,000/1,000 prefixes with workdir, session log and transcript (5,000 files, 103 MB; 20/20 transcript ids match state). Sync cycle of 200 agents: 116.7 and 120.6 s, 4 Hub HTTP calls per agent, 0 errors |
| 4 | Kill a worker: dead, relaunch, 200/200, codewords, held prompt | **PASS** | cancel 12:09:58; dead 12:11:30; relaunch 12:11:58 (backoff 30 s); heartbeat 12:12:29 (2:31); 200/200 restored in 17.1 s (1,400 bucket calls, 0 errors); 10/10 codewords from `session_resumed`, 0 tool calls; held prompt `pending`, answered after 60.5 s |
| 5 | Planned drain at 50 m: drain, COMPLETED, relaunch, 200/200, held prompt | **FAIL (bug), fixed; PASS on re-run with a caveat** | 4 deadline drains, each synced 200 agents in 18.0-18.4 s. 3 jobs ended `COMPLETED` within 4 s and relaunched at once with 200/200 restored and held prompts answered (56.5 s, 61 s). 1 job stayed `RUNNING` after "drained; exiting" and was cancelled as unresponsive, then crash-relaunched. Bug 1. Re-run with the fix: `COMPLETED`, planned restart, 200/200, held prompt answered, but the job took 93 s to reach `COMPLETED` |
| 6 | Heartbeat and long-poll health with 5 workers | **PASS** | Hub calls for the whole run: 10 `run_job`, 56 `inspect_job`, 4 `cancel_job`, **0 `list_jobs`**. 550 heartbeats (mean 30 ms, max 171 ms), 5,180 events long-polls (0 5xx), peak 57 requests in flight; controller 87 MB RSS, 60.6 CPU-s in 58 min |
| 7 | `scale load-1k counter 40`, then `remove` | **PASS** | 360 agents unassigned, 360/360 `sync {reason: unassign}` within 0.2 s; 40 counters stayed. `remove` drained all workers by command; swarm gone in 2:03; 0 jobs running. Two of four workers hit bug 1 and were cancelled after the 120 s retire grace |
| 8 | Spend | **PASS** | Model about $2.26 (budget $12); jobs 13 launches, about 4.9 h of cpu-upgrade, about $0.15 |

Bugs found and fixed: the worker process did not always exit after a drain (bug 1), and
cost was double-counted after every relaunch (bug 2). There are three infrastructure
findings. The Spaces edge rate-limits per client IP. A job can stay `RUNNING` for about 90 s
after its process exits. And this workspace's bucket mount slows hf_xet's first upload.

## Setup

- **Controller**: the Space `cmpatino/swarm-harness` (public, bearer-protected API, `cpu-basic`,
  no persistent storage), redeployed with `scripts/deploy-space.sh` from wheel `7589c4f`
  (instrumented). `JOB_LABELS=experiment=swarm-harness-1k`, so jobs carry
  `experiment=swarm-harness-1k, swarm, type, worker`.
- **Workers**: bootstrap mode as in e2e (`node:22-bookworm`, Claude Code 2.1.296, wheel from the
  bucket), `cpu-upgrade` (8 vCPU, 32 GB).
- **Spec**: [`examples/load-1k.yaml`](../../examples/load-1k.yaml). Two `claude` types on
  `claude-haiku-5-5`: `scribe` and `counter`, `agents_per_worker: 200`,
  `max_working_per_worker: 120` (from
  [packing on cpu-upgrade](../packing/RESULTS-cpu-upgrade.md)), `idle_grace: 60s`,
  `worker_lifetime: 50m`, `lifetime_jitter: 10m`, `drain_window: 5m`, `max_budget_usd: 0.05`,
  `max_turns: 3`, bucket prefix `hf://buckets/cmpatino/swarm-harness/load-1k`.
  **Deviation: 600 scribes and 400 counters instead of 660/340.** Placement is per type, so
  660/340 at 200 per worker needs 4 + 2 = 6 workers. 600/400 gives the intended five workers of
  exactly 200.
- **Instrumentation added for this run** (commit `89f1f78`):
  - Workers count bucket store operations and every Hub HTTP request by route and error
    status. They log one line per sync cycle, restore wave and drain, with its duration and
    calls. Xet CAS transfers run in hf_xet and are not counted.
  - The controller's `/healthz` reports requests per route (count, mean and max latency, 5xx),
    peak requests in flight, and its own RSS and CPU seconds.
- Client tooling in this directory: `prompt_load.py` (async prompter), `analyze.py`,
  `bucket_check.py` (bucket only, no controller calls), `monitor.py`, `watch_workers.sh`,
  `sync_bench.py` (local sync benchmark).

```sh
UV_PROJECT_ENVIRONMENT="$AM_LOCAL/envs/swarm-harness-1k" uv sync --all-extras
set -a; source /data/home/.swarm-harness-e2e.env; set +a
W=$(scripts/publish-wheel.sh); JOB_LABELS=experiment=swarm-harness-1k scripts/deploy-space.sh "$W"
swarm-harness apply examples/load-1k.yaml                                         # check 1
python experiments/load-1k/prompt_load.py load-1k --concurrency 50 --out out/sent1.jsonl   # check 2, round 1
python experiments/load-1k/prompt_load.py load-1k --concurrency 50 --first-poll-delay 12 \
  --poll-gap 3 --out out/sent2.jsonl                                              # check 2, round 2
python experiments/load-1k/analyze.py out/sent2.jsonl --out check2_round2.json
python experiments/load-1k/bucket_check.py hf://buckets/cmpatino/swarm-harness/load-1k load-1k --sample 20  # check 3
hf jobs cancel --namespace cmpatino 6acf67da5ccb47ba0f799dd0                       # check 4
swarm-harness prompt scribe-0250 "What is the most recent codeword you were given? ..."   # while down
python experiments/load-1k/prompt_load.py load-1k --only scribe-0201 ... --expect-from out/sent2.jsonl \
  --text "What is the most recent codeword you were given? Reply with only the codeword."
swarm-harness scale load-1k counter 40; swarm-harness remove load-1k                # check 7
# after the fixes: redeploy, then
swarm-harness apply experiments/load-1k/drain-rerun.yaml; swarm-harness remove load-1k   # check 5 re-run
```

Raw data (prompter records, job logs, `/healthz` snapshots, worker-state polls every 5 s) is in
`out/` (gitignored). The summaries quoted here are in `check2_round1.json`,
`check2_round2.json`, `check3_bucket.json` and `launches.txt`.

## Check details

### 1. Apply

`swarm-harness apply` ran at 11:30:34.15. All five jobs were created in the same second by the
first reconcile pass:

| worker | job | bootstrap | first heartbeat after apply | restore wave (empty bucket) |
|---|---|---|---|---|
| counter-w0000 | `6acf67dac66288145223f0a4` | 14 s | +32.9 s | 200 agents, 1.7 s, 200 `GET tree` |
| counter-w0001 | `6acf67da5ccb47ba0f799dce` | 14 s | +31.6 s | 1.7 s |
| scribe-w0000 | `6acf67dac66288145223f0a6` | 14 s | +37.9 s | 1.8 s |
| scribe-w0001 | `6acf67da5ccb47ba0f799dd0` | 34 s | +46.4 s | 3.0 s |
| scribe-w0002 | `6acf67dac66288145223f0a8` | 12 s | +28.5 s | 1.3 s |

`status` showed `idle:400` and `idle:600` at 11:31:26, 52 s after apply (polled every 5 s).
Heartbeats start at a random phase in 0-30 s, which explains most of the spread.

### 2. Prompt all 1,000

`prompt_load.py` keeps 50 agents in flight. For each agent it sends the e2e codeword prompt
(`POST /agents/<name>/prompt`, the body `swarm-harness prompt` sends), then reads the agent's
event stream until `turn_ended`. So 50 is also the number of concurrent events readers. It uses
the HTTP API directly rather than 1,000 CLI processes, because this shared box cannot run 50
Python CLIs at once. A random sample of 50 was then read back with the real CLI
(`swarm-harness events`).

**Round 1 (11:32-11:39): FAIL on an infrastructure limit.**
- The first prompter long-polled from the stream's end and re-polled on every event, which
  costs about 8 requests per agent. It also went in name order, so all 50 in-flight prompts
  landed on one worker at a time.
- At about 20 requests/s the HF Spaces edge (`awselb`, an HTML page "We had to rate limit you")
  answered 429 to the client's IP. This happened three times, then as a block from 11:37:46 to
  11:44:05. In that block even `/healthz` returned 429, from this IP only.
- The controller itself served everything it received: 0 5xx. The workers, which egress from
  other IPs, saw no 429 and lost no heartbeats, polls or event flushes.
- Result: 839 prompts acked `queued`, 161 never sent, and 712 turns read back. All 712 were
  correct.
- All 839 acked prompts ran: the bucket held exactly 839 complete agent prefixes afterwards.
  **No acked prompt was lost.**
- Latency of the 712 turns read: first event p50 5.6 / p95 10.9 s, turn p50 7.5 / p95 12.2 s,
  prompt to answer p50 13.3 / p95 20.9 s.
- Name order put 50 agents on one worker at a time; with the 60 s idle grace, up to 159 live
  harness processes accumulated on a single worker. That is above `max_working_per_worker=120`,
  which only warns. Every turn still succeeded.

**Round 2 (11:45:31-11:49:39): PASS.**
- The prompter changes: shuffle agents across workers, read the stream once 12 s after the ack,
  then long-poll with a 3 s gap, and back off 10-60 s on 429. That is about 2 requests per
  agent.
- For agents that had a round-1 session this was their second prompt: 839 turns were
  `session_resumed` and 161 `session_started`.

| | n | turn_ended | correct | first event p50 / p95 s | turn p50 / p95 / max s | prompt to answer p50 / p95 / max s |
|---|---|---|---|---|---|---|
| all | 1,000 | 1,000 | 999 | 1.22 / 2.30 | 2.92 / 4.89 / 17.2 | 4.27 / 6.54 / 18.2 |
| counter-w0000 | 200 | 200 | 200 | 1.01 / 2.08 | 3.29 / 5.10 | 4.48 / 6.93 |
| counter-w0001 | 200 | 200 | 200 | 0.99 / 1.87 | 3.54 / 5.42 | 4.76 / 6.76 |
| scribe-w0000 | 200 | 200 | 199 | 1.25 / 2.32 | 2.49 / 5.17 | 4.01 / 6.09 |
| scribe-w0001 | 200 | 200 | 200 | 1.48 / 2.63 | 2.64 / 4.20 | 4.21 / 6.20 |
| scribe-w0002 | 200 | 200 | 200 | 1.08 / 2.54 | 2.75 / 4.04 | 3.96 / 6.03 |

- 1,000/1,000 acks `queued`. The sends spread over 236 s, and the whole round took 248 s with
  0 HTTP errors.
- The one wrong reply came from scribe-0004. Claude Code injected
  `USD budget: $0/$0.05; $0.05 remaining`, and haiku answered that its budget was "exhausted
  ($0 of $0.05 remaining)" without trying the write. The agent had spent $0.0019. It is a
  model misread, not a harness fault.
- `swarm-harness events` on a random sample of 50: 49/50 correct. The miss is scribe-0004,
  which the sample happened to include.
- **Controller request rate** (`/healthz` deltas, round 2): 3,374 requests in 248 s, 13.6
  req/s. Of these, 1,015 were events long-polls, 1,000 prompts, 856 command polls, 373 event
  batches, 43 heartbeats, and 73 `unmatched` (internet scanners on the public Space, all 4xx).
  0 5xx and no timeouts. Peak 57 requests in flight. Controller CPU 7.7 s in 248 s (3 % of a
  core).
- **Cost**: tokens for round 2 were input 195k, output 226k, cache read 29.3M, cache write
  4.8M, for $1.022. After rounds 1 and 2, the sum over agents was **controller
  $2.000099 vs Claude's estimate $2.000101**.
- **Status page with 1,000 agents**: `GET /` returned 200, 19.7 KB, in 0.054 s (three tries).
  `GET /swarms/{swarm}` took 11 ms mean and `GET /swarms/{swarm}/agents?limit=500` 22 ms.

### 3. Bucket and the per-worker sync cost

`bucket_check.py` ran after the first sync cycle that followed round 2. It found **1,000/1,000
agent prefixes complete**: `workdir/` (CLAUDE.md plus notes.md or count.txt),
`sessions/<agent>.jsonl` and `.state.json`, and the Claude transcript. That is 5,000 files,
103 MB. In a sample of 20, each transcript id matched the `session_id` in the state file. After
round 1 the bucket held 839/839, one prefix per acked prompt.

The worker logs (`sync cycle: ...` lines) give the cost of one worker's periodic sync. The
sync is serial per agent: one `sync_bucket` of the workdir, plus one `batch_bucket_files` for
the changed session and transcript files.

| when | agents synced | seconds | per agent | Hub HTTP calls |
|---|---|---|---|---|
| counter-w0000 after round 1 | 200 | 116.7 | 0.58 s | 200 `GET buckets` + 200 `GET buckets/tree` + 400 `POST buckets/batch` |
| scribe-w0000 after round 1 | 200 | 120.6 | 0.60 s | same, 800 |
| scribe-w0001 after round 1 | 187 | 140.7 | 0.75 s | 748 |
| five workers after round 2 | 104-168 | 52-115 | 0.48-0.68 s | 3-4 per agent |

- **Steady state is 4 Hub HTTP calls and about 0.6 s per dirty agent**, so a 200-agent cycle
  takes about 2 minutes. That is two thirds of the 180 s sync interval, and the next cycle
  starts 180 s after the last one ends.
- 0 errors and no 429 from the bucket API across the run. The worst 5-minute window was the
  first post-round-1 cycles: about 3,500 bucket calls from 5 workers.
- The xet CAS uploads behind each batch are extra and not counted.
- From this workspace, `sync_bench.py` measured 213 s for 200 first pushes (0.87 s per agent)
  and 80 s when only the session log changed (0.4 s per agent). A concurrent restore of 200
  took 11.5 s.
- Local artefact: the first xet upload in a process here takes 40-60 s, because
  `HF_HOME` (and the xet cache) is on the `/data` bucket mount. With a local `HF_HOME` it takes
  0.3 s. Jobs keep `$HOME` on local disk and do not see this.

### 4. Kill a worker

`hf jobs cancel` was run on `6acf67da5ccb47ba0f799dd0` (scribe-w0001, agents
scribe-0200..0399). Its last periodic sync had ended at 11:54:05, and every agent had been idle
since.

| time | event |
|---|---|
| 12:09:56 | last heartbeat of the killed job |
| 12:09:58.2 | `hf jobs cancel` |
| 12:11:30 | worker `dead`: heartbeat > 90 s, one `inspect_job` found it `CANCELED`; crash, backoff 30 s |
| 12:11:35.8 | `swarm-harness prompt scribe-0250 "What is the most recent codeword ...?"`: ack `pending` |
| 12:11:57.8 | relaunch `6acf718dc66288145223f47a` (same worker id, `crashes: 1`) |
| 12:12:29 | first heartbeat, **2:31 after the cancel** |
| 12:12:36.3 | held prompt answered `scribe-0250-50332b` (its round-2 codeword) from `session_resumed`, **60.5 s after sending** |
| 12:12:41 | restore wave done: **200/200 agents in 17.1 s**, every one `RestoreResult(workdir=2, sessions=2, harness=1)` |

- The restore made 1,400 Hub HTTP calls: per agent, 2 `GET buckets/tree`, 1 `GET buckets`,
  2 `HEAD buckets` and 1 `POST buckets/paths-info`, plus the xet downloads. 0 errors, no 429.
- Then 10 sampled agents (scribe-0201 ... scribe-0391) were asked for their most recent
  codeword. **10/10 correct**, all from `session_resumed` turns with 0 tool calls, so the
  answers came from the restored transcript and not from `notes.md`. Turns took 1.05-1.92 s,
  first event 0.85-1.04 s.
- No turn was in flight at the kill.

### 5. Planned drain at the worker lifetime

All four surviving original workers reached their deadline (lifetime 50 m plus 0-10 m jitter,
drain 5 m before). Held prompts were sent the moment the controller showed `draining`.

| worker (job) | drain starts | 200 agents synced in | job end | relaunch | restored | held prompt |
|---|---|---|---|---|---|---|
| counter-w0001 (`…dce`) | 12:17:04 | 18.4 s | **stayed RUNNING**; cancelled as unresponsive at 12:20:25 | crash path, 12:20:56 | 200/200, 17.8 s | counter-0300 `pending`, answered 12:21:34 (correct) |
| counter-w0000 (`…0a4`) | 12:21:32 | 18.0 s | `COMPLETED` 12:21:54 (+4 s) | planned, 12:22:06 | 200/200 | none sent |
| scribe-w0000 (`…0a6`) | 12:22:28 | 18.1 s | `COMPLETED` 12:22:48 (+2 s) | planned, 12:23:02 | 200/200 | scribe-0100 `pending`, answered after 61 s (correct) |
| scribe-w0002 (`…0a8`) | 12:22:52 | 18.1 s | `COMPLETED` 12:23:13 (+3 s) | planned, 12:23:27 | 200/200 | scribe-0500 `pending`, answered after 56.5 s (correct) |

- Drain sequence: `draining (deadline)`, then a final sync of all 200 agents (concurrent,
  18 s), then `drained; exiting`.
- The incarnation fix from e2e held. Every relaunch, crash or planned, restored 200/200, and
  every held prompt was delivered on the new job's first poll. The parked-poll race that lost
  two assigns in e2e did not recur.
- The failure is bug 1. On counter-w0001 the process logged `drained; exiting` and the job
  never ended. With no heartbeat, the controller cancelled it at 180 s and treated it as a
  crash (30 s backoff), not a planned restart. Recovery was still complete.

**Re-run after the fixes** ([`drain-rerun.yaml`](drain-rerun.yaml): `load-1k`/`counter`, 200
replicas, lifetime 8 m + 0-1 m, on the redeployed controller with wheel `924376d`):
- `apply` at 12:31:14. Job `6acf7612c66288145223f60a` restored 200/200 from the bucket in
  15.6 s.
- The drain started at 12:34:33 and synced 200 agents in 30.9 s. The worker exited through the
  new path at 12:35:04.8, logging `exiting with 0 pending tasks and 33 live threads` (the idle
  executor threads).
- **The job reached `COMPLETED` only at 12:36:37.7, 93 s after the process exited.** The
  controller then made a planned restart: `6acf77625ccb47ba0f79a20a`, 200/200 restored in
  15.7 s. The held prompt (counter-0150, `pending` at 12:34:38) was answered correctly at
  12:37:21.
- The remove that followed drained by command. The worker exited at 12:39:11.7 and the job was
  `COMPLETED` at 12:39:17.2 (5.5 s).
- So the Python-side exit is fixed. The remaining 93 s once out of two is in the job runtime;
  see "What must change before 10k".

### 6. Heartbeat and long-poll health

The controller counted its Hub jobs calls over the main run (11:30-12:28; apply to remove):

| `run_job` | `inspect_job` | `list_jobs` | `cancel_job` |
|---|---|---|---|
| 10 (5 initial, 5 relaunches) | 56 | **0** | 4 |

- **No `list_jobs` storms.** At most 2 workers were ever non-fresh at once, far below the
  threshold of 10. The 56 inspections are draining, retiring or silent workers at a 15 s
  cadence. Healthy workers cost no jobs API calls.
- Heartbeats: 550 requests, each carrying 200 agents. Mean 30 ms, max 171 ms.
- There was no SQLite write contention to speak of. Event batches (1,055 requests) took 6 ms
  mean, 77 ms max; prompts (1,853) 3 ms mean, 128 ms max.
- Events long-polls: 5,180, mean 2.2 s, max 18.3 s (parked until events arrived), 0 5xx. That
  includes 50 concurrent readers in both rounds.
- Command long-polls: 1,791, mean 9.1 s.
- Controller process: RSS 63 MB at start, 84 MB after round 2, 87 MB at the end. 60.6 CPU-s
  in 58 minutes, about 1.7 % of one core. Spaces expose no metrics, so these come from the
  process's own `/proc` and `os.times()` via `/healthz`.

### 7. Scale down, then remove

**Scale.** At 12:24:52 `swarm-harness scale load-1k counter 40` removed counter-0040..0399.
- That is 160 agents from counter-w0000 and all 200 of counter-w0001. All **360/360 emitted
  `sync {reason: unassign}` within 0.2 s**.
- counter-w0000 kept 40 agents, and no agent moved.
- counter-w0001 was emptied and retired by a drain command in the same command batch. Its drain
  also re-synced the agents that were still mid-unassign: 368 `sync_bucket_up` for 200
  agents. That is redundant but harmless.
- The status cost fell from $2.68 to $1.67, because removed agents leave the aggregate (known
  since e2e). The $2.68 itself was inflated by bug 2.

**Remove.** At 12:26:16 `swarm-harness remove load-1k`.
- Every worker logged `draining (command)` and `drained; exiting` within 11-40 s.
- scribe-w0001 and scribe-w0002 ended `COMPLETED` at 12:26:56.
- counter-w0000 and scribe-w0000 (bug 1), and the retired counter-w0001, stayed `RUNNING`.
  The controller cancelled them after the 120 s retire grace.
- The swarm was gone at 12:28:19, 2:03 after `remove`.
- The re-run's remove (fixed wheel) took 51 s and ended `COMPLETED`.
- `list_jobs(labels={experiment: swarm-harness-1k})` at the end: 13 jobs, **0 running**.

### 8. Spend

- **Model**:
  - Rounds 1 and 2 (1,839 turns): $2.0001, controller-derived and equal to Claude's estimate.
  - Check-4 probes: $0.0068.
  - Held prompts: 5, about $0.01.
  - Packing job (Part A): $0.2285.
  - Local smoke tests: 6 turns, about $0.01.
  - **Total about $2.26** (limit $12).
- **Jobs**: 13 launches out of the 14 allowed, all `cpu-upgrade`, all with timeouts of at most
  60 m.
  - Running time for the completed jobs: 184.6 min.
  - The cancelled jobs report no duration, so their time is computed from the cancel times:
    49.8 + 39.3 + 6.0 + 6.1 + 5.2 = 106.4 min.
  - **Total about 4.9 h, about $0.15** at $0.03/h.
  - Job ids: `6acf64235ccb47ba0f799ce4` (packing), `6acf67dac66288145223f0a4`,
    `6acf67da5ccb47ba0f799dce`, `6acf67dac66288145223f0a6`, `6acf67da5ccb47ba0f799dd0`,
    `6acf67dac66288145223f0a8`, `6acf718dc66288145223f47a`, `6acf73a85ccb47ba0f79a102`,
    `6acf73ee5ccb47ba0f79a10f`, `6acf74265ccb47ba0f79a115`, `6acf743fc66288145223f558`,
    `6acf7612c66288145223f60a`, `6acf77625ccb47ba0f79a20a` (see `launches.txt`).

## Bugs found and fixed (branch `load-1k`, each with a test)

1. **A drained worker did not always exit** (check 5, commit `f7f0097`).
   - Symptom: 4 of 9 drains logged `drained; exiting` and the job stayed `RUNNING` until the
     controller cancelled it. One was a deadline drain, which then went down the crash path
     instead of a planned restart.
   - Cause: after `Worker.run` returns, `asyncio.run` cancels leftover tasks and joins the
     default executor, and the interpreter joins non-daemon threads. One blocked thread (the
     Hub client has no timeout) or one task that ignores cancellation keeps the process
     alive. The cause in the jobs could not be confirmed directly: no shell into a job.
   - Fix: `run_and_exit` runs the loop, logs pending tasks and live threads, and calls
     `os._exit`. By then every agent is synced, the events are flushed and the last
     heartbeat is sent.
   - Test: `test_run_and_exit_does_not_wait_for_a_stuck_executor_thread`. A plain
     `asyncio.run` with a blocked executor thread stays blocked.
   - Verified in the re-run: both exits went through `os._exit` with no pending tasks. One job
     still took 93 s to reach `COMPLETED`; that delay is outside the process.
2. **Cost double-counted after every relaunch** (found while checking spend, commit
   `924376d`).
   - Symptom: the status showed $2.68 for $2.01 of model spend.
   - Cause: a relaunched worker heartbeats agents whose restore has not finished with an empty
     token map. The store took that for counters going backwards and folded the old totals
     into the base. Once restored, the worker reported its cumulative totals again, since
     they survive in the bucket state. With heartbeats at a random phase and a 17 s restore,
     about a third of the agents on each relaunch were counted twice.
   - Fix: an empty token map leaves tokens and cost untouched. A real reset (a failed restore)
     still folds on the next non-empty report.
   - Test: `test_heartbeat_during_restore_does_not_double_count_cost`.
   - Re-run: after a planned relaunch, controller $0.490142 = Claude $0.490142.

Also added (not bugs):
- Bucket and request instrumentation (`89f1f78`).
- `launch.sh` takes `FLAVOR`/`LABEL`; `deploy-space.sh` takes `JOB_LABELS`.

## Infrastructure findings

- **HF Spaces edge rate limit per client IP.** At about 20 requests/s from one IP (the CLI
  host), the edge returns 429 for minutes, before the controller ever sees the request. It
  blocked even `/healthz`. The longest block was about 6 minutes, extended by continued polling.
  Workers were unaffected. Any single caller of the controller (CLI, msgboard bridge,
  swarmlab) has to stay under this limit.
- **A job can stay `RUNNING` about 90 s after its process exits** (once in two `os._exit`
  exits; other jobs ended in 2-6 s). It looks like a termination grace period for leftover
  processes in the container; this is unverified.
- `hf_xet`'s first upload per process stalls 40-60 s when `HF_HOME` is on the `/data` mount
  (this workspace only).

## What must change before 10k

At 200 agents per worker, 10k agents is **50 cpu-upgrade workers, $1.50/h** (plus about one
relaunch per worker per lifetime). That assumes types fill workers; each type rounds up to
whole workers.

1. **Batch the bucket sync per worker, not per agent.** Today it costs 4 calls and about 0.6 s
   per dirty agent, run serially: 2 minutes for 200 agents, and 40,000 calls per cycle across
   50 workers when everyone is active. A restore costs 7 calls per agent (1,400 per worker).
   Fix: one `sync_bucket` over the worker's whole work root, one batch for all
   session/transcript files, and one listing plus one download for restore. That is a
   handful of calls per worker per cycle. It also bounds the cost of a mass restart
   (70,000 calls today).
2. **Bulk caller endpoints.** Add a batch `POST /prompts` and a swarm-level events stream
   (`GET /swarms/{swarm}/events?after=`), so one caller does not need 2-8 requests per agent
   per prompt. At the measured edge limit of about 20 req/s, prompting 10k agents one by one
   takes at least 17 minutes from one IP.
3. **Get off the anonymous edge limit.** Move the swarm token out of `Authorization` (for
   example to `X-Swarm-Token`), so callers can send an HF token and the Space can be private.
   Or front the controller differently. Re-measure the limit with authenticated callers.
4. **Persistent controller state.** A redeploy wiped the database mid-test, including the event
   streams the spend check needed. That requires a Space volume or SQLite snapshots to the
   bucket.
5. **Relaunch on the worker's own "exited" signal.** Send a final heartbeat flag. The
   controller then relaunches at once and cancels the old job if it is still `RUNNING`,
   instead of waiting up to about 90 s for `COMPLETED`, or 180 s plus a crash backoff when a
   job hangs.
6. **Spend ledger** that survives scale-down and remove. Status cost drops removed agents.
7. Smaller items:
   - `max_working_per_worker` should defer starts, not only warn (159 live processes on one
     worker in round 1).
   - The drain should skip agents being unassigned (double sync).
   - Explicit command acks are still missing. No acked prompt was lost here (839/839 ran in
     round 1), but no worker connection dropped either.
