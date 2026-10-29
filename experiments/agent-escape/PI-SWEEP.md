# Pi scaling sweep

Run `bash experiments/agent-escape/setup.sh` and
`bash experiments/agent-escape/setup-pi.sh` from the Flock repository. The latter pins
Pi 1.2.0 and Matplotlib 3.10.8; its wrapper resolves Node's absolute path so Slurm
jobs do not depend on an interactive shell. The benchmark revision is pinned by setup.sh.
`HF_TOKEN` must be available to the submitting process for the existing private bucket.
The serving run supplies the router endpoint and owner-only API-key file.

```sh
uv run --no-sync python experiments/agent-escape/sweep.py submit \
  --server-run .flock/b300-qwen/serving/20261029T104948.409701Z/server
```

The CPU gate runs difficulty 5 with 4 agents, difficulty 10 with 4 agents, and
difficulty 5 with 32 agents sequentially. Each must solve, report complete usage,
use a distinct stable session ID per agent, offer the native puzzle tools without
OS tools, request low reasoning and 4096 output tokens, and verify persisted boards.
Failure prevents sweep submission. Pilots never enter the plotted sample.

Passing the gate submits 75 CPU tasks: difficulties 5/10/15/20/25, agent counts
2/4/8/16/32, three trials per point, scenario index 0 throughout. A seeded shuffle
(default 42) determines task order; the Slurm array allows eight simultaneous tasks.
Each task has 8 CPUs, 16 GB, a 75-minute allocation, and a 60-minute experiment limit.
Each agent has a 200-call Pi turn limit and 32768-token model context.
No GPU allocations are created: all inference uses the existing router.

Pi exposes Flock's pinned puzzle and communication functions as native tools.
Built-in shell and file tools, discovered extensions, MCP servers, skills, and prompt
templates are disabled. The trusted extension invokes Flock's validating tool runner.
Its `X-Session-ID` is the Pi session UUID and survives resume. A first-request barrier
waits for every agent before generation starts, with a five-minute startup timeout.

Boards include harness, model, difficulty, scenario, agent count, trial, UTC timestamp,
and attempt suffix, e.g. `escape-pi-qwen27b-d10-s00-n08-r02-20261029t120000z-a1b2c3d4`.
They retain the existing private bucket/viewer access policy.

Each attempt has `metadata.json`, `result.json`, Flock sessions and events, benchmark
trajectory, and `model-requests.jsonl`. The request journal records settings, offered
tool names, session IDs, and provider usage, without prompts or credentials. Model
admission closes on verified success or shutdown. Already admitted requests drain
before Pi stops; all their tokens count, including compaction and retried calls.
Failed requests or missing usage invalidate the measurement rather than becoming zero.
Input includes cached input once; output already includes reasoning tokens.
Synthetic provider preflight is skipped for measured runs; the launcher checks the
router's model list, and the real pilots validate model/tool compatibility.

The manifest retains all 75 tasks. An `afterany` analysis job writes `results.csv`,
`results.json`, and `normalized-tokens.png`, `.svg`, `.pdf` in the sweep directory.
It can also be run manually:

```sh
uv run --no-sync python experiments/agent-escape/sweep.py analyze PATH/manifest.json
```

There are five panels, one per difficulty, with shared linear axes. Each point shows
successful trials' mean of **all tokens through shutdown / configured agent count**,
with sample standard deviation where at least two trials succeeded. Labels show
successes/3. Zero successes produce no invented point. Failed-trial token costs
remain in the table; incomplete measurements are null, with observed partial counts
retained separately. A missing report is explicitly marked, and no failed trial is
silently replaced. Intermediate analysis is allowed, but pending tasks then appear
as missing reports and must be regenerated after completion.

Slurm does not receive credentials in arguments or exported environment. Each task's
owner-only secrets file is removed when its launcher exits. Tasks cancelled before
starting can leave their secrets file; remove those files after cancellation.
The task launcher checks recorded source hashes before running. Do not edit experiment
or Pi adapter sources while tasks are queued; submit a fresh sweep after changes.
