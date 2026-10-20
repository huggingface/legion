# Controller as an HF Job (job mode)

Added 2026-10-20. Decision 31 in [design.md](design.md). The Space deployment
(`scripts/deploy-space.sh`) stays for small swarms; use job mode once a swarm has
more than a few dozen worker jobs.

## Why

Workers in a Space deployment reach the controller through the HF Spaces edge, which
rate-limits all traffic to one Space at roughly 90-120 req/s in aggregate (design §10).
Two pixel runs hit this with about 44 worker jobs:

- **Run 1k (2026-10-19):** 1,076 HTTP 429 lines in 54 of 71 worker jobs. 29 workers
  (725 agents) were declared dead and relaunched.
- **Run 2 (2026-10-20):** 300 + 329 + 218 throttled requests, the first ~30 s after the
  first worker booted. PR #10's backoff was already in place. 22 workers went silent for
  over 180 s, were declared dead and relaunched, and the leaders' prompts never arrived.

The controller never saw those 429s (12,486 requests, 0 5xx). A shared edge limit is the
ceiling, and retries cannot raise it. In job mode the controller and its workers share an
HF Jobs network group, so worker traffic never crosses an edge or a proxy.

## Shape

```
              humans, flock CLI, leaders outside the group
                 │  https://<job>--7860.hf.jobs  (HF token, jobs proxy)
                 │  https://<job>--7861.hf.jobs  (public, read-only)
┌────────────────▼───────── network group <group> (one namespace) ─┐
│  controller job (cpu-upgrade), alias "controller"                │
│     7860  full API      7861  `/` and `/healthz` only            │
│        ▲                                                         │
│        │ http://${HF_NETWORK_GROUP_PREFIX}controller:7860        │
│  worker jobs (cpu-basic) ... and their agents' `flock` CLI       │
└──────────────────────────────────────────────────────────────────┘
   GPU / model jobs stay outside the group, reached through their own proxy URL.
```

- **Workers.** `jobs.py` launches every worker with `network_group=<group>` and
  `FLOCK_URL=http://${HF_NETWORK_GROUP_PREFIX}controller:7860`. The `${...}` is sent
  literally: HF Jobs sets `HF_NETWORK_GROUP_PREFIX` only inside group members, so the
  worker expands it at start (`flock.worker.main.expand_env`). A worker outside the group
  fails fast and says so. Agents get the expanded URL. The launch budget per 5 minutes is
  unchanged.
- **Namespace.** A network group spans one namespace (and resource group). A job-mode
  controller refuses (`400`) a spec whose `namespace` differs from its own `FLOCK_NAMESPACE`.
- **Model jobs** (vLLM on GPUs) are not in the group. A group started by an A100 job
  refused CPU members (probe-reachability, 2026-10-19), and other GPU pools are untested. Workers keep reaching models
  through the proxy (`--expose`, `auth: bearer` with an HF token), as before.

## Deploy

```sh
export HF_TOKEN=... ANTHROPIC_KEY=... FLOCK_TOKEN=$(openssl rand -hex 24)
W=$(uv run scripts/publish-wheel.sh hf://buckets/<you>/flock/dist)
scripts/deploy-job.sh "$W" --group flock-run3 [--flavor cpu-upgrade] [--timeout 48h] \
    [--secret OPENAI_API_KEY ...] [--dry-run]
# controller job: 6ad7...   API: https://6ad7...--7860.hf.jobs   status: https://6ad7...--7861.hf.jobs
export FLOCK_URL=https://6ad7...--7860.hf.jobs
uv run flock apply spec.yaml          # spec.namespace must be the job's namespace
```

The script (`python -m flock.controller.jobdeploy`) launches `python:3.12-slim`. That
image installs `huggingface_hub`, downloads the wheel from the bucket, installs
`wheel[controller]` and runs `flock-controller`, so no image build is needed. Workers
still bootstrap from the same wheel. The job gets:

| | value |
|---|---|
| network | `--network-group <group> --network-alias controller` |
| ports | `--expose 7860` (API, HF token), `--expose-public 7861` (status) |
| secrets | `FLOCK_TOKEN`, `HF_TOKEN`, `ANTHROPIC_API_KEY` (from `ANTHROPIC_API_KEY` or `ANTHROPIC_KEY`), each `--secret NAME` (spec secrets such as `OPENAI_API_KEY`, `MSGBOARD_INVITE_*`) |
| env | `FLOCK_MODE=job`, `FLOCK_NETWORK_GROUP`, `FLOCK_NAMESPACE`, `FLOCK_STATE_URL`, `FLOCK_WHEEL_URL`, `FLOCK_WORKER_BOOTSTRAP=1`, `FLOCK_JOB_LABELS` (from `JOB_LABELS`) |
| labels | `flock-role=controller`, `flock-group=<group>` (no `swarm` label, so it is never adopted as a worker) |

**Its own URL.** Job env cannot change after launch, but HF Jobs sets `JOB_ID` in every
container and the proxy URL is `https://<JOB_ID>--<port>.hf.jobs`. The controller derives
both URLs itself, with no Hub call and no second launch step. It logs them at start and
reports them in `/healthz` (`public_url`, `status_url`). `FLOCK_PUBLIC_URL` overrides.

## Two URLs, two auth paths

| who | URL | headers |
|---|---|---|
| workers, agents inside the group | `http://${HF_NETWORK_GROUP_PREFIX}controller:7860` | `Authorization: Bearer <flock or agent token>` (direct, no proxy) |
| humans, CLI, anything outside the group | `https://<job>--7860.hf.jobs` | `Authorization: Bearer <HF token>` for the proxy **and** `X-Flock-Token: <flock token>` |
| anyone | `https://<job>--7861.hf.jobs` | none; `GET /` (status page) and `/healthz`, everything else 404 |

The jobs proxy consumes `Authorization` and returns 401 without a valid HF token
(probe-reachability, 2026-10-19). The API therefore takes the flock token, or an agent
token, from `X-Flock-Token` first, then from `Authorization: Bearer`, which keeps Space
mode working unchanged. The `flock` CLI always sends `X-Flock-Token`. When `FLOCK_URL`
ends in `.hf.jobs` or `FLOCK_PROXY_AUTH=hf`, it also sends `HF_TOKEN` (or the token saved by
`hf auth login`) as the bearer; otherwise it sends the flock token as the bearer, as before.
`/` on 7860 still works through the proxy with an HF token.

**Unverified until the smoke run:** whether the proxy forwards `X-Flock-Token` (the probe
showed that extra headers pass harmlessly, but nothing echoed them).

## State

The job's disk is ephemeral. With `FLOCK_STATE_URL` set (the script sets
`hf://buckets/<ns>/<bucket>/controller/<group>`, i.e. the wheel's bucket), the controller:

- **pushes every `FLOCK_STATE_SYNC_INTERVAL` s (60)**, and once more on SIGTERM.
  - `state.db` is a SQLite online-backup copy, consistent with WAL, made under the store lock.
  - Then `events/` is mirrored (`FLOCK_STATE_EVENTS=0` keeps event files local).
  - A push is skipped when nothing was written since the last one.
- **restores on start** (`FLOCK_STATE_RESTORE=1`, the default), before the store opens.
  - It downloads `state.db`, drops any local WAL, then pulls `events/`.
  - `Store.repair_events` cuts each event file back to the restored index, so `after=`
    cursors stay valid.
  - `rebuild_from_jobs` then adopts the live worker jobs by label, as after a Space restart.

**Loss window.**
- A clean stop (cancel, timeout, SIGTERM) loses nothing, provided the final push finishes
  within the platform's grace period. That grace period is not documented and not measured.
- A hard kill loses up to one interval (60 s) of controller writes.
- What is lost:
  - **Prompts.** Ones queued in that window are lost. Ones already delivered are not.
  - **Heartbeat-derived fields** (tokens, cost, state). They come back with the next heartbeats.
  - **Applies, spawns and removes** in that window. They have to be redone.
- Agent sessions are not affected: workers sync those to the swarm bucket themselves.

## Timeout and restart

- **Max timeout.** The Hub's OpenAPI schema bounds `timeoutSeconds` only by 2^53, and the
  docs give no cap (2026-10-20). Workers already run 24 h + jitter. A multi-day controller
  job is **unverified**. The script defaults to `--timeout 48h`.
- **Restart (or renewal before the timeout).** Run `scripts/deploy-job.sh "$W" --group <same> --replace`.
  1. It cancels the running controller of that group. The cancel sends SIGTERM, and the
     old controller makes its final push.
  2. It waits up to 180 s for that job to end.
  3. It launches a new job that claims the same alias.
- **Alias takeover.** The Hub schema says "an alias names a role, not a job: several members
  may claim the same alias, which then resolves to all of them". So a new job can take
  `controller` while workers stay up. For this reason the script refuses to start a second
  controller in a group while one runs: workers would split their traffic between two
  controllers. Whether the alias stops resolving to the old job the moment it ends is
  **unverified** (the smoke run checks it).
- **Workers during a restart.** They retry with backoff. They are declared dead only after
  180 s of silence measured by the new controller, which also resets the clock for the
  workers it adopts. A restart of under about 2 minutes therefore relaunches nothing. A
  longer outage gets the workers relaunched, the same as a crash, with sessions restored
  from the bucket.
- **The proxy URL changes with the job id.** Re-export `FLOCK_URL` after a restart. The
  workers' URL (the alias) does not change.

## Cost

cpu-upgrade is $0.03/h, plus a small flat hourly rate for each exposed port (two here).
Pick cpu-basic ($0.01/h) for small swarms.

## Limits and open questions

- One namespace per controller (the network group's scope).
- GPU flavors in the same group are untested beyond L4 (accepted) and A100 (refused).
- Untested in a real job, to be checked by the smoke run:
  - whether `X-Flock-Token` passes the proxy;
  - SIGTERM grace on cancel;
  - the bootstrap time of the controller job;
  - how alias resolution behaves across a `--replace`.
