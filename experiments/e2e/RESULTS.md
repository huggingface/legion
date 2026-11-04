# End to end (milestone 1, step 3)

Question: does the design work end to end on real HF infrastructure? Run 2026-10-14, 09:54 to
10:44 UTC, branch `e2e`. 100 Claude agents of two types on three `cpu-basic` workers, prompted
from the CLI, with a worker killed by hand, a pending prompt held across the relaunch, planned
drains at a shortened lifetime, a scale-down and a remove.

## Verdict

| # | Check | Result | Key numbers |
|---|---|---|---|
| 1 | Space deploy, `/healthz`, CLI `status` with token | **PASS** | healthz `{"ok":true}`; `GET /swarms` without token 401; `flock status` exit 0 |
| 2 | `apply` -> 3 workers, heartbeats < 3 min, 100 idle | **PASS** | 3 `run_job` in one pass at 09:54:26; first heartbeats +33/+46/+60 s; 100 idle at +64 s; bootstrap 14-23 s |
| 3 | Prompt all 100 from the CLI | **PASS** | 100/100 acks `queued`, 100/100 `turn_ended`, 100/100 correct codewords; first event p50 17.8 s / p95 27.3 s; turn p50 15.2 s / p95 23.7 s; prompt to answer p50 32.5 s / p95 43.1 s; controller $0.112523 vs Claude $0.112526 |
| 4 | Bucket: workdir, session log, harness transcript per agent | **PASS** | 100/100 complete at 09:58:53 (first sync tick after the turns), 500 files, 9.0 MB |
| 5 | Kill a worker with `hf jobs cancel`; resume elsewhere; pending prompt | **PASS** | cancel 09:59:06, crash detected 10:00:17, relaunch (same worker id, new job) 10:00:47, heartbeat 10:01:10 (2:04); 32/32 restored; 5/5 codewords from `session_resumed` with 0 tool calls; held prompt ack `pending`, answered 54 s after sending |
| 6 | Planned drain at a shortened lifetime | **FAIL, fixed, PASS on re-run** | Run 1: drain sequence, `COMPLETED`, immediate `planned_restart` all correct, but the relaunch lost 2 of 34 assigns and the held prompt (bug 1). Run 2 with the fix: 34/34 restored, held prompt answered after 52.9 s, every probed agent resumed |
| 7 | `scale e2e counter 10`, then `remove e2e` | **PASS** | 24 agents (counter-0010..0033) unassigned with a final sync, state kept in the bucket, no agent moved; `remove` drained all workers by command, swarm gone in 20 s, no cancels needed, 0 jobs running |
| 8 | Hub API calls and spend | **PASS** | Jobs API from the controller: run 1 6 `run_job` + 12 `inspect_job`, run 2 4 + 9, 0 `list_jobs` / `cancel_job`. Model spend $0.129 from events (119 turns), about $0.14 with local smoke tests; jobs about 2.2 h of cpu-basic, about $0.02 |

## Setup

- **Controller**: public Docker Space [`cmpatino/flock`](https://huggingface.co/spaces/cmpatino/flock)
  (`https://cmpatino-flock.hf.space`), `cpu-basic`, no persistent storage, built from
  `Dockerfile.controller` by `scripts/deploy-space.sh`. Space secrets `FLOCK_TOKEN`,
  `HF_TOKEN`, `ANTHROPIC_API_KEY` (from `ANTHROPIC_KEY`); variables
  `FLOCK_WORKER_BOOTSTRAP=1`, `FLOCK_WHEEL_URL`,
  `FLOCK_JOB_LABELS=experiment=flock-m1`.
- **Public Space, bearer-protected API.** A private Space needs an HF bearer token on every
  request, and that collides with our own `Authorization: Bearer <swarm token>` header. So the
  Space is public: every API route needs the swarm token (an unauthenticated
  `GET /swarms` returns 401), but `/` (the HTML status table: swarm, type and agent names,
  counts, cost) and `/healthz` (including Hub call counts) are readable by anyone. Before real
  use, either accept that or move our token to a different header so the Space can be private.
- **Workers**: bootstrap mode (no Docker daemon here, so `Dockerfile.worker` was not built).
  Each job runs `node:22-bookworm` with a bash script that installs
  `@anthropic-ai/claude-code@2.1.296`, `@openai/codex@0.162.1`, uv, `huggingface_hub==2.2.0`
  and the flock wheel from `hf://buckets/cmpatino/flock/dist/<rev>/`
  (`cd97485` for run 1, `f33ad53` with the fix for run 2)
  (pinned `claude-agent-sdk==0.2.166`), sets `IS_SANDBOX=1`, `CLAUDE_CONFIG_DIR=/state/claude`,
  `CODEX_HOME=/state/codex`, and execs `flock-worker`. Bootstrap took 14 to 23 s.
- **Spec**: [`examples/e2e.yaml`](../../examples/e2e.yaml). Two types, both `harness: claude`,
  `model: claude-haiku-5-5`: `scribe` (66 replicas, system prompt: keep `notes.md`) and
  `counter` (34 replicas, system prompt: keep `count.txt`). `agents_per_worker: 34`,
  `max_working_per_worker: 20`, `idle_grace: 60s`, `worker_lifetime: 40m`,
  `lifetime_jitter: 5m`, `drain_window: 5m`, `max_budget_usd: 0.05`, `max_turns: 3`.
  **Codex is out of scope** (no OpenAI key here). Heterogeneity is two Claude types with
  different system prompts. It shows in behaviour: scribes made 1.0 tool calls per turn,
  counters 1.9 (read, then write `count.txt`).
- Bucket prefix `hf://buckets/cmpatino/flock/e2e`, so agent state lives under
  `e2e/e2e/<type>/<agent>/`.

Commands, in order:

```sh
uv sync --all-extras
W=$(scripts/publish-wheel.sh)          # hf://buckets/cmpatino/flock/dist/<rev>/flock-0.0.1-py3-none-any.whl
set -a; source ~/.flock-e2e.env; set +a   # FLOCK_TOKEN, FLOCK_URL (never printed)
scripts/deploy-space.sh "$W"
curl -s $FLOCK_URL/healthz; flock status                  # check 1
flock apply examples/e2e.yaml                                    # check 2
experiments/e2e/prompt_all.sh e2e out/sent1.jsonl                        # check 3 (25 CLI processes at a time)
python experiments/e2e/measure.py out/sent1.jsonl --out out/check3.json
python experiments/e2e/bucket_check.py hf://buckets/cmpatino/flock/e2e e2e   # check 4
hf jobs cancel --namespace cmpatino 6acf5152c66288145223dfd9             # check 5
flock prompt scribe-0045 "What is your codeword? Reply with only the codeword."   # while down
flock prompt scribe-00{34,40,50,60,65} "What is your codeword? Reply with only the codeword."
hf jobs logs --namespace cmpatino <job>                                  # check 6
flock scale e2e counter 10                                       # check 7
flock remove e2e
# fix, then run 2 on the redeployed (empty) controller
W=$(scripts/publish-wheel.sh); scripts/deploy-space.sh "$W"
flock apply experiments/e2e/e2e-rerun.yaml                       # check 6, re-run
flock remove e2e
python experiments/e2e/spend.py scribe-00{00..65} counter-00{00..33}     # check 8
```

Raw logs (job logs, controller log, status poll every 15 s, per-agent measurements) are in
`out/` (gitignored). The summaries quoted below are in `check3.json`, `check5.json`, `check6.json`,
`spend.json` and `timeline.txt`.

## Check details

### 1. Deploy

`scripts/deploy-space.sh` stages `Dockerfile.controller` as `Dockerfile` with `pyproject.toml`,
`uv.lock`, `src/` and a Space README, then sets secrets and variables through the hub API and
uploads. The Space built in about 3 minutes. `curl $FLOCK_URL/healthz` returned
`{"ok":true,"hub_calls":{}}`, `GET /swarms` without a token returned 401, and
`flock status` printed `no swarms` with exit 0.

### 2. Apply

`flock apply examples/e2e.yaml` at 09:54:26. Placement filled workers in order:
`e2e-scribe-w0000` (34), `e2e-scribe-w0001` (32), `e2e-counter-w0000` (34). All three
`run_job` calls ran in the first reconcile pass. Jobs `6acf51525ccb47ba0f798f47`,
`6acf5152c66288145223dfd9`, `6acf5152c66288145223dfd7` carried labels
`experiment=flock-m1, swarm=e2e, type, worker`. Bootstrap took 14, 20 and 23 s. The
first heartbeats arrived at 09:54:59, 09:55:12 and 09:55:26. Heartbeats start at a random phase
in 0-30 s, which explains the spread. All 100 agents were `idle` at 09:55:30, 64 s after apply.

### 3. Prompt all 100

`prompt_all.sh` ran `flock prompt <name> "Your codeword is <name>-<hex6>. Record it in
your file and reply with only the codeword."` for all 100 agents, 25 CLI processes at a time. All
100 acks were `queued`; the sends took 09:55:42-09:55:46. `measure.py` read every agent's event
stream back. "First event" is the first event after the send (`session_started`, i.e. the prompt
was delivered and the harness process was up). "Turn" is `turn_started` to `turn_ended`.

| | n | first event p50 / p95 s | turn p50 / p95 / max s | prompt to answer p50 / p95 / max s | tool calls per turn |
|---|---|---|---|---|---|
| all | 100 | 17.8 / 27.3 | 15.2 / 23.7 / 26.7 | 32.5 / 43.1 / 52.3 | 1.3 |
| scribe | 66 | 16.8 / - | 13.6 / 18.7 | 31.2 / 33.2 / 42.3 | 1.0 |
| counter | 34 | 21.8 / - | 18.7 / 25.7 | 41.3 / 43.7 / 52.3 | 1.9 |

- The packing measurement predicted this latency. Each worker cold-started 32 to 34 Claude
  processes at once on 2 vCPUs, and first events spread over about 10-27 s.
  `max_working_per_worker: 20` only warns; it does not queue (design: the worker does no
  scheduling). The warning `N live harness processes exceed max_working_per_worker=20` appeared
  33, 33 and 31 times, once per burst start above the limit.
- Tokens for the 100 turns: input 119,047, output 37,951, cache read 2,984,918, cache write
  414,372.
- **Cost: controller-derived $0.112523, Claude's own estimate $0.112526.** This needed bug 2:
  before the fix, the controller's placeholder table priced haiku 10x too high ($1.12).
- Status page and `status e2e` at 09:57: `counter idle:34, running:1, $0.04`;
  `scribe idle:66, running:2, $0.07`; `100 agents · 3 workers · 0 pending prompts · $0.11`.
- One worker logged one `asyncio: Unknown child process pid 440, will report returncode 255`. It
  had no visible effect.

### 4. Bucket

`bucket_check.py` at 09:58:53 (the first 180 s sync tick after the bursts closed) found
**100/100 agents complete, 500 files, 9.0 MB**. Each agent had exactly:

```
harness/projects/-work-agents-counter-0000/1981b420-….jsonl   # Claude transcript (resume set)
sessions/counter-0000.jsonl, sessions/counter-0000.state.json # pi v3 stream + worker state
workdir/CLAUDE.md, workdir/count.txt                          # system prompt + the agent's file
```

`count.txt` of counter-0007 held `1\ncounter-0007-d86a26`, and `notes.md` of scribe-0045 held
`- Codeword: scribe-0045-18ca26`.

### 5. Kill a worker

`hf jobs cancel --namespace cmpatino 6acf5152c66288145223dfd9` (`e2e-scribe-w0001`, agents
scribe-0034..0065) at 09:59:06. The job ended `CANCELED`; SIGTERM has no handler, so this is a
crash, as intended.

| time | event |
|---|---|
| 09:58:44 | last heartbeat of the killed job |
| 10:00:16 | worker `unknown` (heartbeat > 90 s); prompt to scribe-0045 -> ack `pending` |
| 10:00:17 | controller: `crash e2e-scribe-w0001 CANCELED: None; backoff 30s` (first `inspect_job`) |
| 10:00:47 | `launch e2e-scribe-w0001 6acf52cf5ccb47ba0f7992e6` (same worker id, `crashes: 1`) |
| 10:01:10 | first heartbeat of the new job: 2:04 after the cancel; 32/32 `restored RestoreResult(workdir=2, sessions=2, harness=1)` |
| 10:01:10 | held prompt delivered on the first poll; scribe-0045 answered `scribe-0045-18ca26` (`session_resumed`, 0 tool calls), 54.2 s after sending |
| 10:01:18 | "What is your codeword? Reply with only the codeword." to scribe-0034/0040/0050/0060/0065 |

All 5 replied with their exact codeword from a `session_resumed` turn with **no tool calls**:
the answer came from the restored transcript, not from `notes.md`. Turns took 0.9-1.0 s, first
event 0.6-1.0 s. No turn was in flight at the kill, so no `turn_interrupted` was expected. A
manual cancel counts as a crash, so the 30 s backoff applies. Detection is bounded by the
90 s unknown threshold plus the 15 s inspect cadence.

### 6. Planned drain (`worker_lifetime: 40m`, `drain_window: 5m`)

**Run 1 (bug found).** `e2e-counter-w0000` (deadline 10:34:40) drained on schedule. The worker
log (`hf jobs logs`) showed `10:29:40 draining (deadline)`, then `10:29:55 drained; exiting`,
and the job ended `COMPLETED`. A multi-step turn on counter-0002, sent at 10:29:25 (three
`sleep 8` Bash calls), finished inside the drain at 10:29:53. Its transcript was synced
(`sync {reason: drain}` at 10:29:54) before exit. The status is `error_max_turns` from
`max_turns: 3`, as expected. While the worker was draining, a prompt to counter-0003 returned
`pending`. The controller logged `planned_restart e2e-counter-w0000 COMPLETED` and
`launch … 6acf59b65ccb47ba0f7996bd` in the same pass at 10:30:14, 19 s after exit, with no
backoff. The new job heartbeated at 10:30:52. Agents resumed: counter-0004 and counter-0005
answered their codewords from `session_resumed` turns. `e2e-scribe-w0000` drained the same way
at 10:31:12 and relaunched at 10:31:44 with 34/34 agents.

But the new counter job restored only **32 of 34** agents. counter-0000 and counter-0001 stayed
`unknown`, and the held prompt to counter-0003 was never answered (pending count 0, no
`prompt_deferred`). See bug 1. **FAIL.**

**Run 2 (after the fix, `e2e-rerun.yaml`).** The fix required a redeploy, and the Space has no
persistent storage, so this run also tested recovery after losing the controller's database.
Same spec, except the counter worker's lifetime is cut to 8 m (+0-1 m) so it drains about 3 min
after launch.
- `apply` at 10:38:11 on the empty controller: 3 jobs, all 100 agents idle at 10:39:00, and
  100/100 restored from the bucket (34 + 34 + 32 `RestoreResult(workdir=2…)`).
  scribe-0001, scribe-0040, counter-0005, and counter-0020 and counter-0033 (both scaled away
  in check 7) all answered their run-1 codewords from `session_resumed` turns with 0 tool calls.
- Drain at 10:41:33. The in-flight turn finished, the worker exited at 10:41:50, and the job
  ended `COMPLETED`. `planned_restart` and `launch` came at 10:42:06; the new job heartbeated at
  10:42:41 with **34/34** agents restored.
- The prompt held during the drain (counter-0003, ack `pending`) was delivered on the first
  poll and answered correctly after 52.9 s (`session_resumed`). counter-0000 and counter-0001,
  the two agents lost in run 1, answered correctly. **PASS.**

### 7. Scale down, then remove

At 10:33:25, `flock scale e2e counter 10`. The controller deleted counter-0010..0033
(highest slots first) and queued `unassign` commands. Each of those agents logged a final
`sync {reason: unassign}` at 10:33:25. The bucket kept their state (counter-0033: transcript,
session log and state, `CLAUDE.md`, `count.txt`), and run 2 showed it is usable. Following the
design (no moving live agents, no bin packing), `e2e-counter-w0000` kept running with its 10
remaining agents and no agent moved. Status: `counter idle:8 unknown:2` (the two agents from
bug 1), `scribe idle:66`, 76 agents. The status cost fell from $0.12 to $0.09 because the
aggregate covers only existing agents (see "Spend").

At 10:34:00, `flock remove e2e`. Every agent was unassigned and every worker sent
`DrainCommand`. The workers logged `draining (command)` and exited 2-8 s later (`COMPLETED`).
The controller logged `remove_worker` for each and `remove_swarm e2e` at 10:34:20. No
`cancel_job` was needed. Afterwards `list_jobs(labels={experiment: flock-m1})` shows
0 running jobs. Run 2's remove took 20 s the same way.

### 8. Hub API calls and spend

The controller counts its jobs API calls (`/healthz` and a log line every minute):

| | `run_job` | `inspect_job` | `list_jobs` | `cancel_job` |
|---|---|---|---|---|
| run 1 (40 min, 6 launches) | 6 | 12 | 0 | 0 |
| run 2 (6 min, 4 launches) | 4 | 9 | 0 | 0 |

Healthy workers cost no jobs API calls. Inspections happen only for silent, draining or
retiring workers (15 s cadence). `list_jobs` is used only above 10 such workers, and once at
startup per stored swarm. Not instrumented: the workers' bucket API calls (one tree listing per
agent restore; per sync of a dirty agent, one `sync_bucket` plus one `batch_bucket_files`), and
the `hf` CLI calls I made by hand.

**Spend.** `spend.py` sums `turn_ended` tokens from every agent's event stream (removed agents
included) and prices them:
- Run 1: 109 turns, $0.1178. Run 2: 10 turns, $0.0114. Total $0.129.
- Local smoke tests: 6 turns, about $0.006 (Claude's estimate).
- Model total about **$0.14** (budget $5).
- Job launches: **10 of 12**: `6acf5152c66288145223dfd7`, `6acf51525ccb47ba0f798f47`,
  `6acf5152c66288145223dfd9` (cancelled), `6acf52cf5ccb47ba0f7992e6`,
  `6acf59b65ccb47ba0f7996bd`, `6acf5a105ccb47ba0f79973b`, `6acf5b93c66288145223ebe9`,
  `6acf5b935ccb47ba0f799a87`, `6acf5b93c66288145223ebeb`, `6acf5c7e5ccb47ba0f799b5a`.
  About 2.2 job-hours of cpu-basic, about $0.02. Every job had a timeout of at most 45 min.

## Bugs found and fixed (all on branch `e2e`, each with tests)

1. **A parked long-poll from a finished job stole its replacement's commands** (check 6).
   The drained job's last `GET /commands?wait=25` was still parked on the controller when the
   planned relaunch, 19 s after exit, queued the new job's assigns. The parked handler woke,
   returned two assigns and the held prompt to a client that had exited, and the ack
   background task deleted them. The same could happen to a crash relaunch within 25 s, and a
   late heartbeat from an old job could write into the new job's slot. Fix (`f33ad53`):
   - The controller mints an incarnation per launch: job env `WORKER_INCARNATION`, stored on
     the worker row, adopted from the job env on rebuild.
   - Workers send it on polls and heartbeats. A mismatch, including one that occurs while a
     poll is parked, gets 409, which makes a live stale worker abandon its slot.
   - As defence in depth, a heartbeat that omits agents placed on that worker re-queues their
     assign. Assigns are idempotent on the worker.
   - Additive store migration.
   - Tests: `test_parked_poll_of_a_finished_job_gets_nothing_from_its_replacement`,
     `test_heartbeat_missing_assigned_agents_requeues_their_assign`,
     `test_worker_sends_its_incarnation_on_polls_and_heartbeats`, the migration test, and the
     rebuild assertion.
2. **Price table 10x off for Claude 5.5** (check 3). `prices.py` placeholders priced haiku-5-5 at
   $1/$5. The list price is $0.10/$0.50 per MTok (cache read $0.01, 5 m write $0.125). The
   controller now matches Claude's `total_cost_usd` to 6 decimals. Opus 5.5 and Sonnet 5.5 rows
   were updated from the same table; the Codex row is still a placeholder. Test:
   `test_haiku_5_5_matches_claude_code_estimate`.
3. **Worker drained at once if its lifetime was shorter than 10 min** (found in the local
   rehearsal). Until the first assign arrived, the worker used a 10-minute default drain
   window. The controller now passes `DRAIN_WINDOW` in the job env. Test:
   `test_drain_window_from_env_applies_before_any_assign`.
4. **Unreadable worker logs**: one httpx line per poll and heartbeat, plus tqdm bars per upload.
   The worker now quiets httpx and sets `HF_HUB_DISABLE_PROGRESS_BARS=1` (the
   `disable_progress_bars()` call alone did not silence the xet upload bars).

## Other changes (features the run needed)

- `controller/jobs.py` **bootstrap launch mode**: `node:22-bookworm` plus a bash script that
  mirrors `experiments/common/job_bootstrap.sh`. It is used when
  `FLOCK_WORKER_BOOTSTRAP=1` or no worker image is configured; a type's own `image`
  still wins. The wheel comes from `FLOCK_WHEEL_URL`: `hf buckets cp`, falling back to
  the Python API, then `uv tool install`, falling back to `pip`. Tests in `tests/test_jobs.py`.
  `Dockerfile.worker` is unchanged and remains the eventual path.
- `scripts/publish-wheel.sh` (`uv build`, upload to `dist/<git rev>/`) and
  `scripts/deploy-space.sh`.
- **`models.py`: `TypeSpec.system_prompt: str | None` (additive).** `flock apply` inlines
  `system_prompt_file`, relative to the spec file. `AssignCommand.spec` carries the text, and
  the worker writes it as `CLAUDE.md`/`AGENTS.md`, preferring it over the file. Tests in
  `test_cli.py` and `test_worker_agents.py`. This is the only `models.py` change. The
  incarnation is a query parameter and a job env var, not a model field.
- `FLOCK_JOB_LABELS` adds deployment-wide job labels; our own `swarm/type/worker` labels
  cannot be overridden. `/healthz` reports Hub call counts.
- `docs/design.md` §5 gained a note for each contract addition.
- Experiment tooling: `prompt_all.sh`, `measure.py`, `bucket_check.py`, `spend.py`,
  `e2e-rerun.yaml`.

## Deviations from the design

- Workers run in bootstrap mode, not the prebuilt image (no Docker daemon here). This adds
  14-23 s per launch and downloads npm, uv and PyPI packages on every launch.
- The controller is a public Space (see Setup) with no persistent storage. A restart or
  redeploy loses the database. Run 2 showed the recovery path: re-apply the spec and every agent
  restores from the bucket. Live jobs are orphaned, though: they get 404 and exit, and
  `rebuild_from_jobs` cannot adopt them without the stored spec.
- Codex was not exercised (no key). Both types are Claude.

## What remains before 1k

- **Explicit command acks.** Commands are deleted when the poll response is sent, not when the
  worker receives it. A dropped connection to a live worker still loses a prompt. Assigns now
  self-heal through heartbeats; prompts do not. Have the worker ack command ids on its next poll.
- **Persistent controller storage** (a Space volume, or SQLite snapshots to the bucket), so a
  controller restart adopts live jobs as the design intends.
- **The prebuilt worker image** (`Dockerfile.worker` built in CI), to remove the bootstrap's
  per-launch downloads (1k launches means 1k npm/PyPI installs) and its 14-23 s.
- **Cold-start bursts.** Prompting every agent at once gave a first-event p95 of 27 s at 34 agents
  per worker. At the 50-agent default, expect about 40 s. `max_working_per_worker` only warns.
  Decide whether a worker should defer starts above it, or whether callers should spread prompts.
- **Cost accounting for removed agents.** Status aggregates drop removed agents' cost. Keep a
  per-swarm spend ledger in the controller.
- **Instrument the workers' bucket API calls** (they will dominate Hub traffic at 1k: restores
  and per-agent syncs every 180 s). Measure `inspect_job` and `list_jobs` at 1k against the
  5-minute rate window.
- Codex in a job (needs `OPENAI_API_KEY`), and a public versus private Space decision for the
  auth header.

The Space `cmpatino/flock` is left running (`cpu-basic`, free tier) with no swarms. Agent
state from both runs stays in `hf://buckets/cmpatino/flock/e2e/` (500 files, 9.3 MB).

