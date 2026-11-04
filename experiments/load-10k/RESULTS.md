# 10k load test

Question: does flock hold up at 10,000 agents on fifty `cpu-upgrade` workers, and what
must change next? Run 2026-10-14, 13:04 to 13:30 UTC, branch `load-10k`, controller and workers
on the unchanged main wheel `0cb4258`. There were 10,000 Claude haiku agents of two types, 200 per
worker. All 10,000 were prompted once from five cpu-basic prompter jobs. The run ended early: the
HF Spaces edge throttled the workers' own control traffic, the controller took the silent workers
for dead and crash-relaunched all fifty, a second wave followed, and with 122 of the 130 allowed
launches spent the swarm was removed at 13:25, before the kill test and the deadline drains.

## Verdict

| # | Check | Result | Key numbers |
|---|---|---|---|
| 1 | Apply: 50 workers, 10,000 idle | **PASS** | 50 `run_job` in the first pass, created 13:04:53.8 to 13:04:57.9 (4.1 s), all accepted, no refusal or limit; bootstrap 11 to 24 s (median 13); first heartbeats +21 to +55 s after apply; 10,000 idle and 50 running at +58 s |
| 2 | Prompt all 10,000 once | **FAIL (infra)** | 10,000/10,000 acked (9,357 `queued`, 643 `pending`). At least 9,958 turns ran: 9,869 transcripts in the bucket plus 89 `turn_ended` read back from agents whose transcript died with their worker; 42 unverified. The prompters could read back only 5,525 turns (5,525/5,525 correct) because the edge throttled their reads (1,060 x 429 on POST, 5,194 on GET). Before the first block: first event p50 2.9 s, turn p50 4.0 / p95 17.3 s, prompt to answer p50 7.1 s. Controller: peak about 120 req/s, 0 5xx in 30,018 requests; status page 24.6 KB in 0.23 to 0.26 s; RSS 155 MB (max 167), CPU 13.5 % of a core on average, 70 % peak; controller cost $11.30 at the end of the round |
| 3 | Bucket after the first sync; sync and restore cost | **PASS** | 9,999/10,000 prefixes, 9,869 complete (49,607 files, 895 MB), 30/30 sampled transcript ids match the state. Sync 0.54 s and 3.85 Hub calls per dirty agent (serial), 200-agent cycles 75 to 114 s. Mass restore: 50 workers x 200 agents in 15.4 s p50 / 29.3 s max, about 1,005 calls per worker, about 50,000 bucket calls in two minutes, **0 bucket 429s**; 1 `push_failed` in 11,517 agent syncs |
| 4 | Kill a worker by hand | **NOT RUN** (an involuntary mass kill was measured instead) | last heartbeat to `dead` 183 s p50 (180 to 186); dead to relaunch 47 s p50 (42 to 80; 30 s backoff); relaunch to first heartbeat 186 s p50 (42 to 281) because the new jobs were throttled too; every relaunch restored 200/200; all 643 held prompts were delivered after relaunch (643/643 transcripts). Codeword recall was not probed |
| 5 | Deadline drains at 60 m | **NOT RUN** | The 64 crash relaunches consumed the launch budget (122 of 130) and the swarm was removed at 13:25, 45 min before the first deadline. The command drains at `remove` stand in: 40/40 drained workers reached `COMPLETED`, 39 logged `drained; exiting`, 38 exited with 0 pending tasks and 1 with 1 |
| 6 | Controller health | **PASS, with findings** | 30,018 requests, 0 5xx, peak 89 in flight. Heartbeats 1,001 at 187 ms mean, **10.5 s max** (30 ms / 171 ms at 1k). Hub calls: 114 `run_job`, 603 `inspect_job`, **34 `list_jobs`** (the >10 non-fresh workers fallback, by design), 74 `cancel_job` |
| 7 | `scale counter 400`, then `remove` | **PARTIAL** | Scale skipped (the remove could not wait). `remove` at 13:25:25: 40 workers drained by command in 42 to 48 s each (all agents already unassigning), swarm gone at 13:27:44 (2:19), 24 workers that could not heartbeat were cancelled, zero jobs at 13:30:16 |
| 8 | Spend | **PASS** | Model about $11.30 (limit $45 for prompting); jobs 122 launches of 130, cpu-upgrade 17.1 h about $0.51, cpu-basic 1.5 h about $0.02. Total about $11.9 of $60 |

No code bug was found in the harness paths that ran: launches, placement, restore, held prompts,
drains, exits and removal all did what the design says. The run failed on one infrastructure
limit and one design assumption built on it: the control plane goes through the HF Spaces edge,
the edge rate-limits that traffic as a whole once a few thousand agents are active, and the
controller reads a throttled worker as a dead one. Three findings follow from that; see
"Infrastructure findings" and "What must change".

## Setup

- **Controller**: the Space `cmpatino/flock` (public, bearer-protected API, `cpu-basic`,
  no persistent storage), running main `0cb4258` from the published wheel
  `hf://buckets/cmpatino/flock/dist/0cb4258/flock-0.0.1-py3-none-any.whl`.
  Redeployed with `JOB_LABELS=experiment=flock-10k` before `apply` (no source change, so
  no new commit; the variable change restarted the Space).
- **Workers**: bootstrap mode as in load-1k (`node:22-bookworm`, Claude Code 2.1.296, the wheel
  from the bucket), `cpu-upgrade` (8 vCPU, 32 GB).
- **Spec**: [`examples/load-10k.yaml`](../../examples/load-10k.yaml): `scribe` 6,000 and
  `counter` 4,000 on `claude-haiku-5-5`, `agents_per_worker: 200`, `max_working_per_worker: 120`,
  `idle_grace: 60s`, `worker_lifetime: 60m`, `lifetime_jitter: 15m`, `drain_window: 5m`,
  `max_budget_usd: 0.05`, `max_turns: 3`, bucket prefix `hf://buckets/cmpatino/flock/load-10k`.
  30 + 20 = 50 workers.
- **Prompting from jobs, not from this machine.** The 1k run found the edge limits one client IP
  at about 20 req/s, so the 10k prompt came from five `cpu-basic` jobs, each a slice of 2,000
  agents (every fifth agent, so each slice touches every worker), each paced at 10 req/s with 20
  requests in flight. Phase A sends the codeword prompt (one POST per agent); phase B, 20 s after
  the last send, reads each agent's whole event stream once (one GET per agent) and retries
  unfinished agents every 15 s until 600 s after their send. Records are uploaded to
  `hf://buckets/cmpatino/flock/load-10k-runs/round1/`.
- Tooling in this directory: `prompt_job.py` (the prompter, a uv script run with
  `hf jobs uv run`), `run_prompters.sh` (launches N slices), `fetch_runs.py` (downloads and
  concatenates the slices), `jobs_dump.py` (job metadata and logs by label). The 1k scripts
  `monitor.py` (15 s), `watch_workers.sh` (5 s), `analyze.py` and `bucket_check.py` were reused as
  they are.

```sh
git worktree add -b load-10k ../flock-wt-load-10k main
uv sync --all-extras
set -a; source /data/home/.flock-e2e.env; set +a
JOB_LABELS=experiment=flock-10k scripts/deploy-space.sh hf://buckets/cmpatino/flock/dist/0cb4258/flock-0.0.1-py3-none-any.whl
python experiments/load-1k/monitor.py load-10k out/monitor.jsonl --every 15 &
bash experiments/load-1k/watch_workers.sh load-10k out/workers.tsv 5 &
flock apply examples/load-10k.yaml                                        # check 1, 13:04:52.9
python experiments/load-10k/prompt_job.py load-10k --slice 0/5 --limit 3 --out out/smoke.jsonl   # 3-agent smoke test
bash experiments/load-10k/run_prompters.sh load-10k round1 5 10                   # check 2, 13:06:46
hf jobs run --flavor cpu-basic ... alpine:3.20 sh -c 'wget -qO- https://api.ipify.org'  # x3, egress-IP probes
flock remove load-10k                                                     # 13:25:25, budget stop
python experiments/load-10k/jobs_dump.py out                                      # 122 jobs, 122 logs
python experiments/load-10k/fetch_runs.py round1 out/sent1.jsonl
python experiments/load-1k/analyze.py out/sent1.jsonl --swarm load-10k --out check2_round1.json
python experiments/load-1k/bucket_check.py hf://buckets/cmpatino/flock/load-10k load-10k --sample 30  # check 3
```

Raw data (prompter records, all 122 job logs, `/healthz` snapshots, the status page HTML,
worker-state polls every 5 s, controller polls every 15 s) is in `out/` (gitignored). The
summaries quoted here are in `check2_round1.json`, `check3_bucket.json`, `healthz-final.json`
and `launches.txt`.

## Timeline

| time (UTC) | event |
|---|---|
| 13:04:53 | `apply`; 50 jobs created by 13:04:58 |
| 13:05:50 | 50 running, 10,000 idle |
| 13:06:46 | 5 prompter jobs launched; sends start 13:07:11 |
| 13:07:26 to 13:09:43 | 50 to 120 req/s at the controller; first 429s in worker logs at 13:07:46 |
| 13:08:42 | first workers `unknown` (4), recover within a minute |
| 13:12:31 to 13:17:10 | **block**: the controller's inbound rate falls to about 0.5 req/s (only this machine's monitor gets through); 50 workers go `unknown`, then `dead` from 13:13:48 |
| 13:14:28 to 13:16:04 | the controller cancels and relaunches all 50 (generation 2) |
| 13:17:22 to 13:20:10 | generation 2 heartbeats, restores 200/200 each, delivers the 643 held prompts and the rest of the sends; prompters finish sending at 13:20:04 |
| 13:21:11 to 13:25:01 | **second block** (prompters reading at 40 to 76 req/s plus workers): 14 workers `dead`, cancelled and relaunched (generation 3); 2 more `launching` at 13:25 |
| 13:25:25 | `remove` (122 launches used); 40 workers drain by command |
| 13:27:44 | swarm gone; 13:30:16 zero jobs running |

## Check details

### 1. Apply

`flock apply` returned in 0.29 s at 13:04:52.9. The first reconcile pass launched all 50
workers: `/healthz` showed `run_job: 20` at 13:04:55, 43 at 13:04:57 and 50 at 13:04:59; the
jobs API reports creation times from 13:04:53.8 to 13:04:57.9 and start times 13:04:57 to
13:05:07. Nothing was refused: no `cancelReason`, no credit or concurrency message, 50 `RUNNING`.
Bootstrap took 11 to 24 s (median 13 s, 114 jobs). The first heartbeats arrived between +21 s and
+55 s after apply (heartbeats start at a random phase in 0 to 30 s), and `status` showed 10,000
idle on 50 running workers at 13:05:50, 58 s after apply. The restore waves on the empty bucket
were 200 agents in 1.2 to 1.7 s (200 `GET buckets/tree` each).

### 2. Prompt all 10,000

The five prompters listed the 10,000 agents (20 pages of 500, 8 to 11 s each) and started
sending at 13:07:11. Their own view of the round:

| slice | acks queued / pending | turns read back | correct | 429 on POST | 429 on GET | agents never read | wall |
|---|---|---|---|---|---|---|---|
| 0 | 1,988 / 12 | 1,889 | 1,889 | 0 | 160 | 0 | 816 s |
| 1 | 1,772 / 228 | 434 | 434 | 360 | 1,846 | 1,566 | 1,375 s |
| 2 | 1,790 / 210 | 457 | 457 | 360 | 1,823 | 1,543 | 1,376 s |
| 3 | 1,820 / 180 | 1,396 | 1,396 | 260 | 573 | 503 | 891 s |
| 4 | 1,987 / 13 | 1,349 | 1,349 | 80 | 792 | 492 | 865 s |
| all | 9,357 / 643 | 5,525 | **5,525** | 1,060 | 5,194 | 4,104 | |

- **Every prompt was acked.** 643 came back `pending` because the agent's worker was `unknown`
  or `dead` at that moment; the controller holds those in `pending_prompts` and delivers them
  to the worker's next job. All 643 of those agents have a transcript in the bucket, so all were
  delivered.
- **How many turns ran.** The prompters read back 5,525 `turn_ended` events, all with the right
  codeword (there was not one model misread this time). They could not read the other 4,392
  streams: once the edge started answering 429 to the prompters' GETs, the 10 to 60 s backoffs
  pushed them past the 600 s deadline (4,104 records end in `GET ... failed`, 288 were read
  before the turn had started). The bucket gives the fuller count: 9,869 agents have a Claude
  transcript, so ran at least one turn. Of the 131 without one, 110 sat on `counter-w0014`,
  whose generation-1 job was cancelled before its next periodic sync; the prompters had read
  `turn_ended` for 89 of the 131. So **at least 9,958 of 10,000 prompts ran**, and 42 cannot be
  verified either way (they were acked `queued`, and the pending-prompts table means a job
  ending does not drop them).
- **Latency**, measured on the 5,887 prompts sent before 13:10 (before the first block), by server
  timestamps: first event p50 2.9 s / p95 77 s, turn p50 4.0 s / p95 17.3 s / max 39 s, prompt
  to answer p50 7.1 s / p95 88 s. The p95s are the throttle: command polls that got 429 delayed
  delivery by the worker's backoff. The turn p50 of 4.0 s compares with 2.9 s at 1k; up to 150
  agents were working at once across 50 workers, three per worker.
- 112 turns were interrupted by the mass cancel (`turn_interrupted` events); all 112 agents have
  a transcript, and none was re-run (the types have no `resume_prompt`).
- **Controller request rate**: 86 to 120 req/s during the send phase (prompters 50/s, worker
  command polls returning immediately while prompts flowed, event flushes every 3 s, heartbeats
  1.7/s), 0 5xx. Over the whole run, 30,018 requests: 10,003 prompts (11 ms mean), 7,799 command
  polls (3.7 s mean), 6,033 event reads (6 ms), 4,194 event batches (19 ms, 1.9 s max), 1,001
  heartbeats (187 ms mean, 10.5 s max), 65 4xx (scanners and the post-remove 404s), peak 89 in
  flight. CPU: 230 s in 28 min (13.5 % of a core), 70 % at the peak of the send phase. RSS 130 to
  155 MB, max 167 MB.
- **Status page with 10,000 agents**: `GET /` returned 24,571 B in 0.235 to 0.265 s (three tries
  idle, 0.24 s under load); 54 ms and 19.7 KB at 1k. `GET /swarms/load-10k` 0.19 s,
  `GET /swarms/load-10k/agents?limit=500` 0.18 s (251 KB).
- **Cost**: the controller showed $11.30 at 13:21 and 13:25 (the Claude estimate per agent was
  not captured: the swarm was removed before the agent list could be read, and removed agents
  leave the aggregate). The 5,525 observed turns cost $6.27 from their tokens (input 6.5M, output
  2.0M, cache read 171M, cache write 23M), $0.00113 per turn, which extrapolates to $11.3 for the
  9,958 turns that ran. The two agree.

### 3. Bucket, sync cost, and the mass restore

`bucket_check.py` after the run: **9,999 agent prefixes, 9,869 complete** (workdir, session log,
state and transcript), 49,607 files, 895 MB, listed in 2.6 s; 30/30 sampled transcript ids match
the `session_id` in the state file. One agent (`scribe-5128`) has no prefix at all and 130 have a
session log and state but no transcript or workdir (see check 2: a cancelled job's unsynced
state).

From the 114 worker logs:

| | n | agents | seconds | per agent | Hub HTTP calls | errors |
|---|---|---|---|---|---|---|
| sync cycles, generation 1 | 97 | p50 55, max 166 | | 0.54 s | 3.85 per agent | 0 |
| sync cycles, generation 2 | 86 | p50 32, max 80 | | 0.55 s | 5.23 per agent | 1 `push_failed` |
| cycles of 150+ agents | 21 | | 75 to 114 | | | |
| restore waves, generation 1 (empty bucket) | 50 | 200 | p50 1.3, max 1.7 | | 200 | 0 |
| restore waves, generation 2 (full restore) | 50 | 200 | **p50 15.4, max 29.3** | 0.08 s | p50 1,005 | 0 |
| restore waves, generation 3 | 4 | 200 | p50 15.8 | | p50 1,268 | 0 |
| drains at `remove` | 40 | 0 (all unassigning) | p50 42, max 48 | | | 0 |

- The per-agent sync cost is the same as at 1k (0.6 s, 4 calls), so a worker with all 200 agents
  dirty spends 75 to 114 s of every 180 s interval syncing. 11,517 agent syncs ran in 26 minutes.
- **The main unknown is answered: the bucket API did not throttle.** The mass restore made about
  50,000 Hub calls (list, paths-info, HEAD, download) from 50 workers inside two minutes,
  on top of the sync cycles, with zero 429s and zero errors. Restoring 200 agents took 15 s per
  worker, as at 1k.
- The 180 s periodic sync is the crash exposure: a cancelled job loses whatever happened since
  its last cycle. Here that was one worker's 110 transcripts.

### 4. Kill a worker

Not run by hand. The edge block killed all 50 instead (13:13:48 to 13:16:04), and the 5 s worker
poll measured the same quantities:

- last heartbeat to `dead`: 183 s p50 (180 to 186 s, 64 transitions);
- `dead` to the relaunch being visible: 47 s p50 (42 to 80 s; 30 s crash backoff plus the 5 s
  reconcile tick);
- relaunch to first heartbeat: 186 s p50 (42 to 281 s). The best cases match the 1k figure
  (bootstrap plus the heartbeat phase); the median is long because the new jobs' first heartbeats
  were throttled as well, which is the strongest evidence that the limit is not per job IP;
- every relaunch logged `restore wave: 200 agents set up`, 200/200, and all 643 held prompts
  were answered after the relaunch (every one of those agents has a transcript).

The codeword-recall probe (ten agents asked for their most recent codeword after a restore) was
not run.

### 5. Planned drains at the worker lifetime

Not run. The first deadline drain would have started at about 14:10 (60 m plus 0 to 15 m jitter,
drain 5 m before), and the crash waves had used 114 of the 130 launches by 13:25 with a third wave
forming; letting 50 deadline relaunches happen was not possible within the limit, so the swarm
was removed first. What the command drains at `remove` showed: all 40 workers that could still
reach the controller logged `draining (command)`, 39 `drained; exiting`, 38 `exiting with 0
pending tasks and 33 live threads` and 1 with 1 pending task; all 40 jobs reached `COMPLETED`
(36 of generation 2, 4 of generation 3), the first within 50 s of the command, the last at
13:27:24. The exit fix from the 1k run held.

### 6. Controller health

Hub jobs calls over the run: **114 `run_job`, 603 `inspect_job`, 34 `list_jobs`, 74
`cancel_job`.** The 34 listings are the `LIST_THRESHOLD` fallback: when more than 10 workers of
a swarm are non-fresh, one pass uses one `list_jobs` instead of N inspects. That is the designed
behaviour and it kept the jobs API traffic at tens of calls per minute during the blocks.

- Heartbeats: 1,001 requests carrying 200 agents each, 187 ms mean, **10.5 s max**. At 1k the
  same route was 30 ms mean, 171 ms max. The heartbeat writes 200 agent rows in one transaction
  and competes with the prompt inserts and event batches; at 120 req/s the SQLite writer became
  the queue. No 5xx and no request timed out, but a 10 s heartbeat is a third of the interval.
- Event batches: 4,194, 19 ms mean, 1.9 s max. Prompts: 10,003, 11 ms mean, 0.83 s max. Event
  reads: 6,033, 6 ms mean (one read per agent rather than long-polls).
- Command long-polls: 7,799, 3.7 s mean, 25 s max.
- The controller never saw the 429s (they are answered by the edge); its own 4xx count is 65.
- Process: RSS 62 MB at start, 134 MB during the send phase, 155 MB at the end (max 167 MB);
  230 CPU-s in 28 minutes. The Space stayed responsive from this machine throughout, including
  during both blocks.

### 7. Scale down, then remove

`scale` was skipped: the relaunch waves were spending launches every pass and the remove could
not wait. `flock remove load-10k` at 13:25:25:

- the 10,000 agents were deleted and unassigned from their workers; 39 workers were `draining`
  within 7 s, and the 11 whose relaunched jobs were still bootstrapping got the drain on their
  first poll;
- each drain logged `0 agents finished and synced in 42 to 48 s`: by then every agent was mid
  unassign, and the drain waited for those serial unassign syncs (about 0.2 s per agent) before
  exiting. 40/40 `COMPLETED`;
- the 24 jobs that could not heartbeat (14 of generation 2 and 10 of generation 3, all logging
  429s to the end) were cancelled by the controller;
- the swarm was gone at 13:27:44, **2:19 after the command**; the last prompter job ended at
  13:30:16 and `hf jobs ls` shows no running job.

### 8. Spend

- **Model**: the controller's aggregate was $11.30 at 13:21 and 13:25 (the round-1 turns, the
  resumed turns after the relaunch, no re-run of interrupted turns); the 3-agent smoke test
  about $0.003. The token-derived cost of the 5,525 observed turns is $6.27, $0.00113 per turn.
  **Total about $11.3** (prompting was to stop at $45).
- **Jobs**: 122 launches of the 130 allowed, every one labelled `experiment=flock-10k`,
  every timeout at most 75 m (workers 60 to 75 m, prompters 40 m, probes 5 m): 50 initial
  workers, 64 crash relaunches, 5 prompters, 3 egress-IP probes. Running time: generation 1
  7.92 h, generation 2 8.58 h, generation 3 0.62 h, all `cpu-upgrade`, 17.1 h about **$0.51**; the
  prompters and probes 1.5 h of `cpu-basic`, about $0.02. Cancelled jobs report no end time, so
  theirs is the last time the 5 s worker poll saw the job (74 of 122).
- Total about $11.9 of the $60 limit.

## Bugs

None found in the paths that ran, and no code was changed. Two behaviours are design problems
rather than defects, recorded here because they decided the run:

1. **A throttled worker looks like a dead worker.** The controller's only liveness signal is the
   heartbeat, which travels through the same edge as everything else. When the edge blocked all
   50 workers for 4.5 minutes, every heartbeat failed for more than 180 s, `_refresh_worker`
   inspected each job, found it `RUNNING` and silent, cancelled it as unresponsive and relaunched
   it on the crash path. That cost 64 launches, 10,000 restores, 112 interrupted turns and one
   worker's unsynced transcripts, for workers that were healthy. The worker side compounds it:
   `heartbeat_loop` sends once per 30 s and does not retry a failed heartbeat until the next
   interval, so six consecutive 429s are a death sentence.
2. **A dead worker's unsynced state is lost** (by design, 180 s periodic sync). 110 agents on
   `counter-w0014` lost their first transcript.

Also seen, not diagnosed: 95 `asyncio: Unknown child process pid N, will report returncode 255`
warnings across the worker logs (a harness child reaped outside asyncio's watcher; no visible
effect on turns), and one `push_failed` bucket error in 11,517 syncs.

## Infrastructure findings

1. **The Spaces edge limit is not only per client IP.** At 1k, one client at 20 req/s was
   blocked while five workers at about 1 req/s each were not. Here: 50 workers each sending at
   most 1.4 req/s (command polls at the 1 s floor, event flushes every 3 s, one heartbeat per
   30 s) got 429s on every route (4,231 warnings: 1,025 heartbeats, 1,938 polls, 1,268 event
   flushes; every one of the 114 worker jobs saw at least one, up to 16 heartbeat failures in
   one job), while the aggregate at the controller was 86 to 120 req/s. The blocks were
   simultaneous across workers and prompters (13:12:31 to 13:17:10 and 13:21:11 to 13:25:01),
   and fresh jobs launched during a block were throttled from their first request. Three probe
   jobs (two cpu-basic, one cpu-upgrade) each had a distinct egress IP, so jobs are not behind
   one NAT. This machine's monitor (0.2 req/s) was never blocked. The limiter therefore keys on
   something all jobs share and this Space does not (a per-Space aggregate, or the jobs' IP
   range), with a threshold somewhere between the 14 req/s of the 1k run and the 90 to 120 req/s
   here. The exact key and threshold are unmeasured.
2. **The bucket API is not the bottleneck at this scale.** About 50,000 restore calls in two
   minutes plus 44,000 sync calls over the run, 0 429s, 1 error.
3. **HF Jobs accepted 50 `cpu-upgrade` launches in 4 s and 64 more over ten minutes**, up to 64
   concurrent jobs of this flavor plus five cpu-basic, with no concurrency, credit or
   resource-group refusal. Cancelled jobs still report no `finished_at` or durations.
4. **A 200-row heartbeat takes up to 10.5 s at the controller under a 120 req/s mixed load** (187
   ms mean). SQLite's single writer is where the controller will saturate first, not CPU or RAM.
5. The status page scales linearly: 0.25 s and 24.6 KB at 10k.

## What must change before the next step

Item 1 gates everything else: nothing above about 2,000 to 3,000 active agents can be run again
through the Spaces edge as it is.

1. **Take the control plane off the Spaces edge limit, then measure the limit.** Options, in
   order of how much they keep: a private Space with workers sending an HF token (requires moving
   the swarm token out of `Authorization`; whether authenticated traffic has a higher or separate
   limit is unknown); the controller as an HF Job with `--expose` (its own ingress, $0.01/h); or
   the controller behind another host. Whatever the choice, run a controlled ramp first: N
   cpu-basic jobs at a fixed rate each, find the key (per IP, per range, per Space) and the
   threshold, and write it into §7 of the design.
2. **Liveness that survives a throttled ingress.** The controller should not cancel a job that is
   `RUNNING` only because its heartbeats stopped: require a longer silence when many workers of
   a swarm fall silent together (that is the controller's ingress failing, not 50 workers), cap
   cancels per pass, and never relaunch into a block it can observe (the heartbeat route's own
   request count dropping to zero). On the worker: retry a failed heartbeat after a few seconds
   with jitter, honour `Retry-After`, and back the command poll off on 429 instead of the 1 s
   floor.
3. **Fewer requests per agent-turn** (unchanged from the 1k list, now with numbers): the send
   phase was 120 req/s for 10,000 prompts, of which the prompters were 50/s and the workers'
   immediate re-polls another 50/s. A batch `POST /prompts`, a swarm-level event stream, and
   command polls that batch for a few hundred ms before answering would put the same round under
   20 req/s.
4. **Sync on turn end, not only every 180 s**, once the per-worker batched sync (item 1 of the 1k
   list) makes it cheap; a crash then loses at most one turn, not a worker's last three minutes.
5. **A launch budget per swarm.** The crash path spent 64 launches in ten minutes with nothing
   wrong on the workers; a per-swarm cap on relaunches per window (and a crashloop rule that
   counts the swarm, not the worker) would have stopped it after the first wave.
6. **Spend ledger** (from the 1k list): the only model-cost figure left after `remove` is a status
   poll taken before it. A ledger that outlives agents is needed for any accounting.
7. Smaller: one heartbeat per worker should not take 10 s (write the 200 agent rows in fewer
   statements, or move per-agent counters out of the hot path); the drain should not wait on
   serial unassign syncs (42 s with zero agents); the `asyncio` child-reaping warning wants a
   look; the deadline-drain and kill checks (4 and 5) still need a run at this scale.
