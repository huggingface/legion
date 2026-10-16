# Compute-provider contract and `flock doctor` (design note)

Status: design only, 2026-10-16. For Carlos and Leandro. No code, no jobs launched.
Read against `main` @ 6b15604 and `providers` @ 666867f (PR #2).

## 1. Problem

- flock asks its backend one thing: "run this container" (`JobsClient.launch`, `controller/jobs.py`).
  Everything a swarm depends on beyond that is assumed and never checked: the worker reaches the
  controller and the model endpoint, auth headers survive the path, secrets arrive, refusals can be recognised.
- The 16 Oct smoke runs (about $0.65) found these one paid GPU run at a time:
  - `--expose-public 8000`: vLLM answered every keyed request with its own 401. The cause is inferred
    (the proxy did not pass `Authorization` through) and not confirmed.
  - A network group started by an `a100-large` job refused `cpu-basic` and `cpu-upgrade` workers with
    HTTP 400 ("Flavor ... is not available where network group ... already runs").
  - That 400 was retried every 30 s with no end while type health stayed `ok` (`reconcile.py`, `_launch` except block).
  - Preflight reported network-group providers as `ok` plus `skipped` (`preflight.py:172`), so apply went ahead unchecked.
  - `l40sx1` sat in SCHEDULING for 9 min ("waiting for hardware"), and nothing told flock.
- The harness↔endpoint half is fixed: preflight now sends two streamed turns. This note covers the
  compute half. Decisions in force (Carlos, 16 Oct): workers are CPU-only, and inference runs
  elsewhere. Colocation is an optional capability. HF Jobs is the first backend, not the definition
  of one. The spec stays harness- and provider-agnostic.

## 2. Contract

### 2.1 Protocol (sketch)

```python
class ComputeProvider(Protocol):
    name: str                                            # "hf-jobs"
    def declare(self) -> dict[Cap, bool]: ...            # what the adapter claims; doctor verifies
    def sizes(self) -> dict[str, Size]: ...              # flavor -> vcpu, mem_gb, usd_per_h | None
    def launch(self, req: WorkerLaunch) -> str: ...      # handle; raises LaunchRefused | LaunchDeferred
    def status(self, handle: str) -> WorkerStatus: ...
    def cancel(self, handle: str) -> None: ...           # idempotent; unknown handle is not an error
    def list(self, labels: Mapping[str, str]) -> list[WorkerListing]: ...  # filtered by the backend
    def logs(self, handle: str, tail: int = 200) -> str | None: ...       # None: not supported

@dataclass(frozen=True)
class WorkerLaunch:
    worker_id: str
    image: str                  # bootstrap image or prebuilt worker image
    command: list[str]          # bootstrap script or ["flock-worker"]
    env: dict[str, str]         # non-secret; may be readable back through the backend API
    secrets: list[str]          # names only; the provider resolves values from its SecretSource
    labels: dict[str, str]      # swarm, type, worker + deployment labels
    flavor: str                 # backend size name, validated against sizes()
    timeout_s: int              # hard lifetime; expiry must be reported as `expired`
    network: NetworkOptions | None = None   # optional: group, aliases, expose ports

Phase = Literal["pending", "running", "succeeded", "failed", "expired", "cancelled", "gone"]
class LaunchRefused(Exception): ...   # permanent: bad flavor, placement, auth, quota exhausted (4xx)
class LaunchDeferred(Exception): ...  # transient: 429 (retry_after), 5xx, network error
```

- This replaces `JobsClient`, which takes a `TypeSpec` and an HF `namespace`. The namespace becomes
  provider config. Reconcile converts `TypeSpec` into a `WorkerLaunch`, so providers never see the spec.
- `expired` is part of the contract. Today reconcile string-matches `message == "Job timeout"` to tell a
  planned restart from a crash (`reconcile.py:317`); after this change the provider does the mapping.
- Secrets travel by name. The provider gets values from the `SecretSource` it was built with (today
  the controller's environment). A `WorkerLaunch` never holds a value, so it is safe to log.
- A `pending` phase with no deadline is a failure mode (the l40sx1 case). Each provider declares a
  `pending_timeout`. When it runs out, reconcile cancels the job and backs off, treating it as transient.

### 2.2 Capabilities

| capability | question | needed by |
|---|---|---|
| `boot` | bootstrap image starts; seconds to first callback, per flavor | always |
| `controller_reachable` | worker → `FLOCK_URL` over HTTPS with a bearer token | always |
| `egress_https` | worker → arbitrary public HTTPS with `Authorization` / `x-api-key` intact | always (model APIs, HF bucket) |
| `secrets` | values arrive as env at start; not readable back via API, logs or harness env | always |
| `labels` | list filtered by labels, owned by the backend | always (rebuild after controller restart) |
| `expiry` | hard timeout honoured, reported distinguishably | always (planned restarts) |
| `fail_fast` | permanent refusal is distinct from transient failure at launch | always (else retry forever) |
| `logs` | recent stdout retrievable | optional (debugging) |
| `quota` | max concurrent workers, launch/API rate limit | warn at apply |
| `pricing` | USD/h per flavor known | cost display |
| `expose` | inbound port through a proxy; its auth model; which headers reach the container | optional |
| `colocate` | private network between jobs; placement constraints | optional |

Each result is tri-state, carries the configuration it was tested in, and is stored as JSON:

```json
"colocate": {"state": "unverified", "declared": true,
  "tested": {"backend": "hf-jobs", "client": "huggingface_hub 2.2.0", "namespace": "cmpatino",
             "flavors": ["cpu-basic", "cpu-basic"], "pool": "default", "image": "node:22-bookworm",
             "date": "2026-10-17"},
  "detail": "peer reached by alias, bearer hash matched", "evidence": ["<job ids>"]}
```

Rules:
1. Only a passing probe yields `supported`. A capability that is declared but not probed is `unverified`.
2. A result holds only for the configuration it was tested in. If a spec needs another configuration
   (another flavor, a GPU peer, another namespace or pool), that need is `unverified`. A CPU-only
   doctor run never certifies mixed CPU+GPU placement.
3. `unsupported` requires a definite answer: a 4xx refusal, or a probe that failed for a known reason.
   A timeout or a stuck `pending` is `unverified`, with the reason in `detail`.
4. Results go stale: older than 14 days, a changed client or backend version, or another namespace
   all mean `unverified`.
5. An endpoint preflight that was skipped is `unverified`, never `ok`. This changes PR #2, where
   `preflight.py:172` sets `ok = skipped = True`.

## 3. `flock doctor <backend>`

`flock doctor hf-jobs [--flavors cpu-basic,cpu-upgrade] [--only P] [--expose] [--echo-url URL]`.
The controller runs it, since it holds the backend token and the secrets. Doctor launches only flavors
on a CPU allowlist. Every job gets `timeout=5m` and the label `flock-doctor=<run id>`. Probes 1-4 share
one job: the real bootstrap with `FLOCK_DOCTOR=1`, which runs the checks, reports, and exits.

| # | probe | how | pass | cost |
|---|---|---|---|---|
| 1 | boot | real image, command and wheel, once per flavor in `--flavors` | callback < 300 s; record seconds | ~$0.0005/flavor |
| 2 | controller | worker `GET /healthz`, `POST /doctor/report` with FLOCK_TOKEN | 200 | same job |
| 3 | egress + headers | worker calls the controller's `/doctor/echo` (and `--echo-url` if given) with `Authorization: Bearer <nonce1>` and `x-api-key: <nonce2>`. The echo returns sha256 of each header it received | both hashes match | same job |
| 4 | secrets | launch carries the secret `FLOCK_DOCTOR_SECRET=<nonce3>`. The worker reports its hash and builds a harness env with the real env filter: the nonce and FLOCK_TOKEN must be absent. The controller checks that `status()`, `list()` and `logs()` output do not contain the nonce | all four | same job |
| 5 | fail_fast | launch with flavor `flock-doctor-no-such-flavor`, then pass the exception to reconcile's classifier | `LaunchRefused` within 30 s, classified permanent | $0 (refused) |
| 6 | expiry | `sleep 600` with `timeout_s=60` | phase `expired` within 3 min | ~$0.0003 |
| 7 | colocate (only if declared) | two jobs of the tested flavor in group `flock-doctor-<run>`. A serves a header echo on :8000 with alias `peer`; B calls `http://${HF_NETWORK_GROUP_PREFIX}peer:8000` with a bearer nonce | hash matches; flavors recorded | ~$0.001 |
| 8 | expose (opt-in) | A with `expose=[8000]`. The controller calls the proxy URL with an HF bearer token plus `X-Flock-Probe: <nonce>`; then `update_job_expose` to public and call with `Authorization: Bearer <nonce4>`. The echo says which headers arrived | records passthrough per auth mode (a result, not pass/fail) | ~$0.001 + $0.01/h expose |

All values sent are fresh random nonces; no real secret is used or printed. Only hashes travel back.

Stop conditions:
- If probe 1 or 2 fails, doctor stops. Everything not yet probed stays `unverified`.
- A job pending for more than 3 min is cancelled and marked `unverified` (capacity).
- Doctor cancels every job at 6 min regardless of state, and at the end it sweeps by label, also on Ctrl-C.
- Suite cap: 6 jobs and 30 job-minutes ($0.005 on cpu-basic). There are no GPU flavors, and none can be configured.
- One run per backend at a time (a lock row in SQLite). Whole suite: under $0.01 and about 5 min wall.

Output: `doctor.json` is stored in the controller's SQLite (`doctor_runs`: backend, run id, started, report).
- The latest run is served at `GET /doctor/{backend}`.
- `flock doctor --show` prints the capability table.
- `flock status` adds one line, e.g. `hf-jobs: doctor 2026-10-17 (3 d) 9 supported, 1 unsupported, 2 unverified`.
- Apply reads the report on the server side.

## 4. Apply-time checks

A pure function, `requirements(spec) -> list[Need]`, turns the spec into needs. Each need is a
capability plus a configuration, and each is compared with the latest report.

| spec feature | needs |
|---|---|
| every type | `boot` for its flavor, `controller_reachable`, `secrets`, `labels`, `expiry`, `fail_fast` |
| any type (model APIs, bucket sync) | `egress_https` |
| provider `base_url` on the backend's proxy domain (`*.hf.jobs`) | `expose` with passthrough verified for that provider's auth mode (`bearer` HF token vs own key) |
| provider joined to a group (`network_group` today) | `colocate` for the pair (worker flavor, server flavor); unknown server flavor → unverified |
| `self_hosted` provider with a public `base_url` | `egress_https` only, plus endpoint preflight. **Not** `colocate` (review msg 427) |
| model whose preflight was skipped | endpoint `unverified` |
| total workers incl. `workers.max_extra` above `quota` | warning |
| flavor with no known price | warning: cost unknown |

Outcomes:
- `unsupported`: apply refuses, with no override.
- `unverified`: apply refuses unless `--allow-unverified`. The flag is stored on the swarm, and status
  shows e.g. `applied with unverified: colocate(cpu-basic+a100-large)`.
- No doctor report: everything counts as unverified. The refusal says `run: flock doctor hf-jobs (about $0.01)`.
- The message names the type, the need and what was tested, e.g. `analyst: provider gpu1 needs
  colocate(cpu-basic + a100-large): unverified; tested cpu-basic + cpu-basic on 2026-10-17`.

There is a path from endpoint-unverified to verified without a GPU smoke run: a **worker-side preflight**.
The first worker of an affected type runs the same two-turn probe from the worker network and reports
an event. Agents on that type get prompts only after it passes. This is the "probe from the actual
worker network" in review msg 424.

## 5. HF Jobs today (from smoke-results.md, design §2/§7, HF jobs guide)

Tested configuration unless stated: namespace `cmpatino`, bootstrap image `node:22-bookworm`,
`huggingface_hub` 2.2.0 on the worker, 2026-10-14/16.

| capability | state | evidence / tested config |
|---|---|---|
| boot | supported | cpu-basic: bootstrap 39 s (smoke run 1). cpu-upgrade: 28-46 s to first heartbeat (load-1k) |
| controller_reachable | supported | cpu-basic and cpu-upgrade → Space URL, all runs. The Spaces edge throttles at about 90-120 req/s (10k run); that limit belongs to the controller host, not to this contract |
| egress_https | supported for known hosts | `router.huggingface.co` with a bearer HF token (Codex, run 1); Anthropic API with `x-api-key` (load runs). Arbitrary host with header echo: unverified |
| secrets | delivery supported; isolation unverified | FLOCK_TOKEN and keys arrive as env on every run. The harness env filter has unit tests only. Docs: `inspect_job` returns `env` in plaintext, so sensitive values must go in `secrets` |
| labels | supported | server-side filter, used by `rebuild_from_jobs`; 50-worker and 10k runs |
| expiry | supported | stage `ERROR`, message "Job timeout" (design §2). Multi-week timeouts unverified |
| fail_fast | backend supported; flock unsupported | Hub returned HTTP 400 with a precise message at launch (network group, 2026-10-16); flock retried it every 30 s. Invalid-flavor case unverified. Missing capacity does not fail fast: l40sx1 stayed in SCHEDULING for 9 min |
| logs | unverified | `fetch_job_logs` is documented; flock never calls it |
| quota | partly known | no documented concurrency cap; 64 concurrent and 50 launches in 4 s measured (10k run); API limit 1,000-6,000 calls per 5 min by plan (documented). Above 64: unverified |
| pricing | supported | documented per flavor and via `list_jobs_hardware()`: cpu-basic $0.01/h, cpu-upgrade $0.03/h; expose adds $0.01/h (design §2) |
| expose | reachability supported; header passthrough unverified | `expose_public=[8000]` on a100-large: `/health` 200 through the proxy. Keyed requests got vLLM's own 401, cause unconfirmed. Docs: `expose` requires an HF token with read access to the namespace |
| colocate | unsupported for a100-large first + cpu-basic/cpu-upgrade; otherwise unverified | HTTP 400 on 2026-10-16 (job 6ad214815ccb47ba0f7b219a was the first member). CPU+CPU groups are documented, not tested. CPU first, then a GPU joining: untested |

## 6. What a second backend needs

1. An adapter: the seven protocol methods, with native states mapped to the seven phases (`expired` and
   `pending` included) and a `pending_timeout`.
2. An error classifier from native errors to `LaunchRefused` / `LaunchDeferred`, unit-tested against recorded responses.
3. Secrets transport: how values reach the container, and proof they cannot be read back as plain env.
4. Labels: native, or emulated (name prefix plus a registry the controller can rebuild from after a restart).
5. Bootstrap: the image needs bash, node and python, or use the prebuilt `Dockerfile.worker`. Workers need
   egress to `huggingface.co` (wheel and bucket) and `HF_TOKEN` as a secret: the bucket stays on HF
   whichever backend runs the workers. They also need egress to the controller's public URL.
6. `declare()`, then `flock doctor <backend>`. The first report is committed under `docs/` as the reference.
7. One row in the backends table in the docs.

Rough effort, adapter plus doctor green, unverified estimates:
- Plain Docker host or VM pool: 1-2 days. Labels are native. Expiry needs a timeout wrapper, and there is no pricing (warning only).
- Kubernetes Jobs: 2-3 days. `activeDeadlineSeconds` gives expiry, Secrets are referenced by name,
  labels are native, admission errors are 4xx, and ResourceQuota gives the quota.
- Modal: 2-3 days. Native timeouts and named secrets; label/listing semantics need checking first.

## 7. Migration plan (small PRs)

- **PR 1: protocol, no behaviour change.**
  - Adds `controller/compute.py` (protocol, `WorkerLaunch`, phases, error classes) and `HubJobsBackend`
    wrapping today's `HubJobsClient`.
  - `FakeBackend` replaces `FakeJobsClient`; reconcile builds `WorkerLaunch` from `TypeSpec`.
  - The classifier exists, but reconcile still retries both kinds as today.
  - A contract test suite, parametrised over backends, runs the doctor's assertions offline against the fake.
- **PR 2: doctor.** Probes, `FLOCK_DOCTOR=1` worker mode, `/doctor/echo`, `/doctor/report`,
  `doctor_runs` table, CLI `flock doctor` and `--show`, and the status line. Running it costs about $0.01
  and needs Carlos's OK like any job.
- **PR 3: gating and refusals.**
  - Adds `requirements(spec)`, apply refusal and warnings, and `--allow-unverified`.
  - A `LaunchRefused` sets type health `launch_refused` (like `crashlooping`: shown in status, no
    relaunch until the spec is edited).
  - `pending_timeout` handling; preflight `skipped` becomes unverified.
  - The worker-side preflight is optional here (Q4).

Interactions:
- **PR #2 (providers)** changes `jobs.py` (`network_group()` at :72 and :146) and the preflight skip.
  - Land it first. PR 1 then moves the group into `WorkerLaunch.network`.
  - The `ok = skipped = True` line changes in PR 3, or in #2 if it has not merged yet.
- **PR #1 (Pi)** adds a harness CLI to the bootstrap. The bootstrap script stays backend-independent,
  so the only conflict is a rebase on the version pins in `jobs.py`.
- **Spawn** (`workers: {initial, max_extra}`, declared only): the gating sums `initial + max_extra`
  over types against `quota` and only warns. A spawned child whose worker launch is refused shows as
  `launch_refused` on its type, rather than a silent retry.

## 8. Open questions (Carlos, Leandro)

- Should `unverified` refuse by default (proposed), or only warn for the first weeks?
- Spec naming: keep HF's `network_group` on providers, or move to a generic
  `network: {colocate: <group>, alias}` plus `server_flavor`, so the placement pair can be checked?
- Doctor schedule: manual only, or also at controller start / weekly (about $0.01 per run)?
- Worker-side preflight before prompts: in PR 3, or a separate PR?
- Mixed CPU+GPU on HF: try one cheap probe (start a cpu-basic group, launch a GPU into it, cancel
  while SCHEDULING; billing during SCHEDULING is unverified), or drop HF `colocate` until HF documents placement?
- Is a second backend planned (Docker host, Kubernetes, Modal)? The answer shapes PR 1's fixtures.
