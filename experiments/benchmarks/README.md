# OpenEnv inference pilot

This package runs **one shared task per Flock swarm**. It does not train models or use TRL.
Pi retains its execution loop, native Flock tools, and message-board communication. The bridge
translates authenticated World API v1 tool calls into one OpenEnv WebSocket session per attempt.

The first adapters are AgentEscapeBench and Terminal-Bench 2.0 `db-wal-recovery`. The latter
uses Harbor's Docker environment and native verifier, not Harbor's autonomous agent harness.
Other Terminal-Bench tasks are deliberately rejected until their runtime requirements are validated.
The existing `experiments/agent-escape/` runner still gives each agent an independent puzzle copy;
its results are a different experiment condition and must not be pooled with `shared_task` results.

## Prepare and review

Run commands from the Flock repository. Runtime dependencies are isolated from Flock in a Python
3.12 environment: OpenEnv Core 0.3.0, Harbor 0.25.0, and exact transitive versions in
`requirements.lock`. The public documentation's development branch may describe newer APIs.

```bash
bash experiments/benchmarks/setup.sh
# Only if the existing Agent Escape checkout/runtime is absent:
bash experiments/agent-escape/setup.sh
.venv/bin/python -m experiments.benchmarks catalog
.venv/bin/python -m experiments.benchmarks prepare experiments/benchmarks/pilot.yaml \
  --output .flock/openenv-pilot-review
```

Preparation performs no inference and starts no containers. It resolves the two tasks into four
attempts (1 and 2 agents, one repetition), saves task and code hashes, and writes a prompt,
ordinary Flock spec, and immutable manifest per attempt. The generated World URL is deliberately
a non-serving placeholder; `run` fills in the trial-local service endpoint.

Inspect `campaign.json` and each `manifest.json` / `spec.yaml`. Agent Escape uses difficulty 5,
scenario index 0; the resolved filename and content hash are recorded. Terminal-Bench is pinned
to the official registry commit in `terminal-source.json`; no missing-task substitution occurs.
Task inputs remain evaluator-side; only filtered observations/instructions reach agents.

The checked-in Pi profile retains the previously supplied router URL as a **reference**, not a
claim that it is currently reachable. It uses low reasoning, 64K context, 16K max new tokens,
5-second comms polling, zero comms rate limits, and private bucket-backed boards. Credentials
are environment-variable references, never generated into the manifests. Agent Escape permits
30 minutes; Terminal-Bench respects its stricter native 900-second agent limit.

## No-model validation

Replace `ATTEMPT/manifest.json` below with a prepared attempt path.

```bash
.venv/bin/python -m experiments.benchmarks validate ATTEMPT/manifest.json
.venv/bin/python -m experiments.benchmarks validate ATTEMPT/manifest.json --runtime
PYTHONPATH=src:. .flock/deps/benchmarks-venv/bin/python -m experiments.benchmarks \
  reference ATTEMPT/manifest.json
```

The reference command is explicit: Agent Escape starts the real upstream server and real OpenEnv
service, then checks a fixed sequence against direct upstream calls. It uses evaluator-only
answer data for the final positive check; those data never enter live agent prompts. The gate
checks API parity, not whether an LLM can solve the puzzle. Terminal-Bench uses two fresh Docker
sandboxes: an invalid submission must fail and Harbor's oracle must pass. It makes no model calls.
Reference artifacts live below the attempt directory and a rerun requires a newly prepared attempt.

Docker readiness must pass on the **execution node**. Installing Docker or having the CLI available
is insufficient. Runtime validation fails explicitly if the user cannot access the daemon. No
sudo, alternate runtime conversion, or remote paid sandbox is silently selected.

## Execute only after config review

No pilot inference is part of implementation validation. For a later authorized local run, supply
`LOCAL_KEY` and `HF_TOKEN` in the process environment, then use:

```bash
.venv/bin/python -m experiments.benchmarks run ATTEMPT/manifest.json
```

Execution checks the router's `/models` endpoint before starting Pi. An attempt may run once;
prepare a new campaign to retry. Services and containers are not resumed after connection loss.
The output contains a separate `effective-spec.yaml` with actual runtime ports.

For Slurm, provide an existing owner-only JSON file containing `LOCAL_KEY` and `HF_TOKEN`.
The file must be readable from the compute node. Values never enter `sbatch` arguments or its
exported environment; only the file path is passed.

```bash
.venv/bin/python -m experiments.benchmarks submit CAMPAIGN/campaign.json \
  --secrets-file /private/path/keys.json --dry-run
.venv/bin/python -m experiments.benchmarks submit CAMPAIGN/campaign.json \
  --secrets-file /private/path/keys.json
.venv/bin/python -m experiments.benchmarks status CAMPAIGN/campaign.json
.venv/bin/python -m experiments.benchmarks cancel CAMPAIGN/campaign.json
.venv/bin/python -m experiments.benchmarks analyze CAMPAIGN/campaign.json
```

One array element owns one allocation/attempt, with `%1` limiting concurrency. Array execution
order is scheduler-defined. The default requests 8 CPUs, 16 GiB, no GPU, and 75 minutes including
setup/grading and usage-drain reserve. There are no automatic retries or requeue. An uncertain
submission receipt leaves an intent record and refuses another submission until an operator
reconciles it with Slurm. Cancellation addresses only the recorded job ID.

## Semantics and limitations

- All configured agents can operate on the shared task. Agent identity comes from authenticated
  Flock headers. Bridge conversation IDs are stable logical `attempt:agent` identifiers; actual
  Pi session records remain in the ordinary Flock bundle.
- The bridge serializes actions and records request IDs and sequence numbers. Repeated request
  IDs return the original result; conflicting reuse fails. If a mutation's outcome is unknown,
  the entire attempt is invalidated rather than automatically retried/reset.
- The control token is distinct from the World token. Reset, privileged OpenEnv routes and
  grading are inaccessible with worker credentials. Services bind only to loopback in this pilot.
- Terminal commands run in the shared container, one at a time, for at most 300 seconds. They
  return handles. Output is spooled in the sandbox and read in 32-KiB pages after completion;
  the command file-size limit is 8 MiB. File operations are bounded and allowed between commands.
  Standard background descendants are terminated with their command group. This is not a claim
  of protection against deliberately daemonized processes; the supplied prompt prohibits them.
- `exec_cancel` tears down the sandbox and cancels the attempt, rather than pretending that
  cancelling a host future killed every remote process. A successful task submission is accepted
  only between commands; the first accepted submission seals mutation access.
- Timeout/idle termination seals actions and stops agents. A running terminal command must drain
  before grading; failure to drain within the grading deadline produces an invalid result.
- Native graders determine success; missing verifier output is an error, not a zero score. The
  existing inference usage proxy records aggregate usage, not training tokens/log probabilities.
  It drains in-flight upstream requests after agent shutdown, up to the configured request bound.
- `result.json` distinguishes task outcome, termination reason, infrastructure validity and token
  completeness. Incomplete totals remain observed partial counts; normalized tokens are null.
  Analysis retains those outcomes but excludes incomplete usage from token averages. It includes
  complete valid failures as well as successes in the reported mean.
- The runtime builds the pinned Terminal-Bench Dockerfile rather than trusting a mutable prebuilt
  task-image tag. The built image ID is recorded in `environment.json`. Base images/package repositories can still change; fully hermetic image builds
  are not claimed. Native Harbor logs and verifier outputs are retained in the attempt directory.
- Preparation pins source/task hashes. Editing those inputs requires a new campaign. No controller
  schema changes, existing experiment migration, broad benchmark sweep, or training integration
  is included.

## Tests

```bash
.venv/bin/python -m pytest -q tests/test_benchmark_bridge.py \
  tests/test_benchmark_campaign.py tests/test_benchmark_runner.py
.venv/bin/ruff check experiments/benchmarks tests/test_benchmark*.py
```

Unit tests fake providers, environments and scheduler calls. Real OpenEnv/upstream and Docker
reference gates are separate opt-in commands, never ordinary pytest side effects.
