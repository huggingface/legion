# Controller as an HF Job (job mode)

The Space deployment (`scripts/deploy-space.sh`) is unsupported: its build cannot install the
private msgboard (Dockerfile.controller).

## Why

The HF Spaces edge rate-limits all traffic to one Space at about 90-120 req/s, and retries
cannot raise that ceiling. In job mode the controller and its workers share an HF Jobs network
group, so worker traffic never crosses an edge or a proxy.

## Shape

```
              humans, legion CLI, leaders outside the group
                 │  https://<job>--7860.hf.jobs  (HF token, jobs proxy)
                 │  https://<job>--7861.hf.jobs  (public, read-only; opt-in)
┌────────────────▼───────── network group <group> (one namespace) ─┐
│  controller job (cpu-upgrade), alias "controller"                │
│     7860  full API      7861  `/` and `/healthz` only            │
│        ▲                                                         │
│        │ http://${HF_NETWORK_GROUP_PREFIX}controller:7860        │
│  worker jobs (cpu-basic) ... and their agents' `legion` CLI      │
└──────────────────────────────────────────────────────────────────┘
   GPU / model jobs stay outside the group, reached through their own proxy URL.
```

- **Workers.** `jobs.py` launches every worker with `network_group=<group>` and
  `LEGION_URL=http://${HF_NETWORK_GROUP_PREFIX}controller:7860`. The `${...}` is sent
  literally: HF Jobs sets `HF_NETWORK_GROUP_PREFIX` only inside group members, so the
  worker expands it at start (`legion.worker.main.expand_env`). A worker outside the group
  fails fast and says so. Agents get the expanded URL. The launch budget per 5 minutes is
  the same as in any mode.
- **Namespace.** A network group spans one namespace (and resource group). A job-mode
  controller refuses (`400`) a spec whose `namespace` differs from its own `LEGION_NAMESPACE`.
  An unset spec namespace resolves to `HF_TOKEN`'s account for this check. If a controller
  was manually deployed with `--namespace my-org` and that differs from the token account,
  set `namespace: my-org` in the spec; a blank namespace is refused before any worker
  launches. This keeps worker placement and run billing consistent. When the accounts
  match, the stored namespace stays empty and router calls have no bill-to header.
- **Model jobs** (vLLM on GPUs) are not in the group. A group started by an A100 job
  refused CPU members, and other GPU pools are untested. Workers reach models through the
  proxy (`--expose`, `auth: bearer` with an HF token).

## Deploy

Replace `<your-username>` with the personal Hub account that owns `HF_TOKEN`.

```sh
export HF_TOKEN=... LEGION_TOKEN=$(openssl rand -hex 24)
uv run hf buckets create "<your-username>/legion" --private  # once, for wheels and state
W=$(uv run scripts/publish-wheel.sh "hf://buckets/<your-username>/legion/dist")
scripts/deploy-job.sh "$W" --group my-run [--flavor cpu-upgrade] [--timeout 48h] \
    [--secret OPENAI_API_KEY ...] [--dry-run]
# controller job: 6ad7...   API: https://6ad7...--7860.hf.jobs   status: not public (--public-status)
export LEGION_URL=https://6ad7...--7860.hf.jobs
uv run legion launch spec.yaml          # omit namespace to use HF_TOKEN's personal account
```

`legion launch` does this itself when no controller answers and `LEGION_WHEEL_URL` is set: the
controller job of group `LEGION_NETWORK_GROUP` (default `legion`; `cpu-upgrade`, 48 h, the spec's
secrets that are set), then `export LEGION_URL=...` (`--run RUN_DIR` records it in
`hosts/controller.host`); `legion stop --release` cancels it after a checkpoint (docs/live.md
"Lifecycle").

The script (`python -m legion.controller.jobdeploy`) launches `python:3.12-slim`. That
image installs `huggingface_hub` and uv, downloads the wheel and `harbor-deps.tar` next to it
from the bucket, installs `wheel[controller]` with that tar's msgboard wheel and uv.lock's
versions, and runs `legion-controller`, so no image build is needed. Workers bootstrap from the
same wheel and tar. The job gets:

| | value |
|---|---|
| network | `--network-group <group> --network-alias controller` |
| ports | `--expose 7860` (API, HF token), `--expose-public 7861` (status; opt-in: `--public-status`, or `LEGION_PUBLIC_STATUS=1` for `legion launch`; off, or with a huggingface_hub without `expose_public`, only the API is exposed and the launch says so) |
| secrets | `LEGION_TOKEN`, `HF_TOKEN`, each `--secret NAME` (spec secrets such as `OPENAI_API_KEY`, `MSGBOARD_INVITE_*`) |
| env | `LEGION_MODE=job`, `LEGION_NETWORK_GROUP`, `LEGION_NAMESPACE`, `LEGION_STATE_URL`, `LEGION_WHEEL_URL`, `LEGION_WORKER_BOOTSTRAP=1`, `LEGION_JOB_LABELS` (from `JOB_LABELS`) |
| labels | `legion-role=controller`, `legion-group=<group>` (no `swarm` label, so it is never adopted as a worker) |

**Its own URL.** Job env cannot change after launch, but HF Jobs sets `JOB_ID` in every
container and the proxy URL is `https://<JOB_ID>--<port>.hf.jobs`. The controller derives
both URLs itself, with no Hub call and no second launch step. It logs them at start and
reports them in `/healthz` (`public_url`, `status_url`). `LEGION_PUBLIC_URL` overrides.

Publishing requires Git access to the private `huggingface/msgboard` dependency.
The publication includes `harbor-deps.tar`, so jobs install the pinned dependencies
without needing your GitHub credentials. Workers bootstrap from a stock
`node:22-bookworm` image and download the wheel from the bucket.

Every secret named by a spec, including model `api_key_env` values, must be in the
controller's environment. Workers need `HF_TOKEN` to access the wheel and state
bucket. Do not list `LEGION_TOKEN` in the spec's secrets; the controller supplies
worker authentication itself.

For a swarm running on your machine, set `compute: local` in the spec (the default)
and use the normal launch command from a shell without `LEGION_URL` set:

```sh
export HF_TOKEN=...
uv run legion launch spec.yaml   # spec with compute: local
```

Legion starts the controller and workers locally and records the run under `.legion/`.
The spec's `environment.backend` still chooses where agent containers run; use
`docker` or `podman` for local containers. See [local usage](local.md) for a complete
spec, model credentials, status commands, and results.

## Two URLs, two auth paths

| who | URL | headers |
|---|---|---|
| workers, agents inside the group | `http://${HF_NETWORK_GROUP_PREFIX}controller:7860` | `Authorization: Bearer <legion or agent token>` (direct, no proxy) |
| humans, CLI, anything outside the group | `https://<job>--7860.hf.jobs` | `Authorization: Bearer <HF token>` for the proxy **and** `X-Legion-Token: <legion token>` |
| anyone, with `--public-status` or `LEGION_PUBLIC_STATUS=1` | `https://<job>--7861.hf.jobs` | none; `GET /` (status page) and `/healthz`, everything else 404 |

The jobs proxy consumes `Authorization` and returns 401 without a valid HF token. The API therefore takes the legion token, or an agent
token, from `X-Legion-Token` first, then from `Authorization: Bearer`. The `legion` CLI always sends `X-Legion-Token`. When `LEGION_URL`
ends in `.hf.jobs` or `LEGION_PROXY_AUTH=hf`, it also sends `HF_TOKEN` (or the token saved by
`hf auth login`) as the bearer; otherwise it sends the legion token as the bearer.
`/` on 7860 still works through the proxy with an HF token.

## State

The job's disk is ephemeral. With `LEGION_STATE_URL` set (the script sets
`hf://buckets/<ns>/<bucket>/controller/<group>`, i.e. the wheel's bucket), the controller:

- **pushes every `LEGION_STATE_SYNC_INTERVAL` s (60)**, and once more on SIGTERM.
  - `state.db` is a SQLite online-backup copy, consistent with WAL, made under the store lock.
  - Then `events/` is mirrored (`LEGION_STATE_EVENTS=0` keeps event files local).
  - A push is skipped when nothing was written since the last one. The periodic push and the
    final push never overlap.
- **restores on start** (`LEGION_STATE_RESTORE=1`, the default), before the store opens.
  - It downloads `state.db` and checks it. **A snapshot with no swarms is not restored**:
    after `legion stop --delete` of the last swarm, the next push writes an empty snapshot, so a later
    controller in the same group starts empty.
  - It drops any local WAL, then pulls `events/`.
  - It logs what it restored: `state: restored N swarms (names), A agents, W workers`.
  - `Store.repair_events` cuts each event file back to the restored index, so `after=`
    cursors stay valid.
  - `rebuild_from_jobs` then adopts the live worker jobs by label, as after a Space restart.
- **Shutdown on SIGTERM**, in this order:
  1. Close the listeners and cancel the reconcile and sync loops.
  2. Push the state. The push is bounded by `LEGION_FINAL_PUSH_TIMEOUT` (30 s) and always
     logs one line: `final push: ok (N files)` or `final push: failed (reason)`.
  3. Give open requests (parked long-polls) 5 s.
  4. Exit, logging `legion controller: exit`.

  The push runs before the drain, so a short platform grace period still gets it: uvicorn may
  wait on parked long-polls before it would reach a push placed after them.
- **Teardown:** `scripts/deploy-job.sh --down --group <g> (<wheel-url> | --state-url URL)`.
  1. It cancels the group's controller and waits for it to end (the final push).
  2. Only then does it delete everything under `controller/<g>/`.
  If the controller does not stop within 180 s, it deletes nothing. Run it after `legion
  remove` at the end of an experiment. To ignore a snapshot once, launch with
  `LEGION_STATE_RESTORE=0`.

**Loss window.**
- A clean stop (cancel, timeout, SIGTERM) loses only what is written during the final push
  and the 5 s drain, provided the push finishes within the platform's grace period. That grace
  period is not documented and not measured.
- A hard kill loses up to one interval (60 s) of controller writes.
- What is lost:
  - **Prompts.** Ones queued in that window are lost. Ones already delivered are not.
  - **Heartbeat-derived fields** (tokens, cost, state). They come back with the next heartbeats.
  - **Applies, spawns and removes** in that window. They have to be redone.
- Agent sessions are not affected: workers sync those to the swarm bucket themselves.

## Timeout and restart

- **Max timeout.** The Hub's OpenAPI schema bounds `timeoutSeconds` only by 2^53, and the
  docs give no cap. Workers already run 24 h + jitter. A multi-day controller
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
  controllers. The alias stops resolving to the old job once it ends; heartbeats pause
  during the replace.
- **Workers during a restart.** They retry with backoff. They are declared dead only after
  180 s of silence measured by the new controller, which also resets the clock for the
  workers it adopts. A restart of under about 2 minutes therefore relaunches nothing. A
  longer outage gets the workers relaunched, the same as a crash, with sessions restored
  from the bucket.
- **The proxy URL changes with the job id.** Re-export `LEGION_URL` after a restart. The
  workers' URL (the alias) does not change.

## Cost

cpu-upgrade is $0.03/h, plus a small flat hourly rate for each exposed port (two here).
Pick cpu-basic ($0.01/h) for small swarms.

## Limits and open questions

- One namespace per controller (the network group's scope).
- GPU flavors in the same group are untested beyond L4 (accepted) and A100 (refused).
- Still unmeasured: the platform's SIGTERM grace period on cancel. The push is bounded at 30 s
  and runs first, before the drain.
