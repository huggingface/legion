# Local mode: `flock run`

One spec from a laptop to 1k agents. `compute: local` runs the swarm on your machine:
`flock run spec.yaml` starts the controller in-process on 127.0.0.1, launches each worker
as a subprocess, shows the run in the terminal and writes a run bundle when it stops. The
controller is the same code as on HF Jobs (reconcile loop, placement, heartbeats, admission,
prompts, events); only the backend that starts workers differs. `compute: hf-jobs` (the
default) is the only line you change to run the same spec on HF Jobs.

No HF token, bucket or network is needed when the model endpoint is local.

## Quickstart

Install (Python 3.11+):

```sh
git clone https://github.com/huggingface/flock && cd flock
uv sync --all-extras          # or: pip install -e ".[controller]"
```

Write a spec. This one points 5 `llm` agents at any OpenAI-compatible server on your machine
(Ollama, llama.cpp, vLLM, LM Studio); it is `examples/local.yaml`:

```yaml
swarm: hello
compute: local
namespace: your-hf-name                  # used by hf-jobs only
bucket: hf://buckets/your-hf-name/flock  # used by hf-jobs only
providers:
  local: {type: openai_compat, base_url: "http://localhost:11434/v1", api_key_env: LOCAL_KEY, self_hosted: true}
types:
  writer:
    harness: llm
    model: "local:qwen3:4b"
    replicas: 5
```

`namespace` and `bucket` can be left out locally; keeping them means the switch to jobs is one
line. To use the HF router instead, set `base_url: https://router.huggingface.co/v1`,
`api_key_env: HF_TOKEN`, drop `self_hosted` and add `pricing: {"*": [in, out, cached]}` (USD per
1M tokens, [docs/open-models.md](open-models.md)).

Run it:

```sh
export LOCAL_KEY=unused                      # the key variable must exist; any value for a server without auth
uv run flock run examples/local.yaml --prompt "Introduce yourself in one sentence." --until idle
```

- `--prompt TEXT` sends one prompt to every agent at start (optional; without it the agents
  wait for prompts from a second terminal).
- `--until idle` stops once every agent is idle with nothing pending, after each prompted agent
  finished a turn. `--until 10m` stops after a duration. Without `--until` the run goes until
  Ctrl-C (a second Ctrl-C forces) or the spec's `limits.budget_usd`.
- `--skip-preflight` skips the tool-call probe of each model (on by default for
  provider-backed models, as with `flock apply`); `--grace 10` is how long in-flight turns get
  at stop; `--data-dir .flock` is where runs go.

## What you see

A frame redrawn every 2 s (plain ANSI, no curses):

```
flock run · hello · compute local · 00:00:06 · http://127.0.0.1:46131

agents   3    idle 3
workers  1    running 1
prompts  pending 0 · running 0 · completed 3
cost     $0.0000   fake-model in=378 out=42

type    health  agents  workers    cost
writer  ok      idle 3  running 1  $0.0000

recent events
  writer-0002      16:14:44   t-1 turn_started
  writer-0002      16:14:44   t-1 user     bash: echo hi
  writer-0002      16:14:44   t-1 assistant [bash {"command": "echo hi"}]
  writer-0002      16:14:44   t-1 tool     bash: hi [exit code 0]
  writer-0002      16:14:44   t-1 turn_ended completed fake-model in=126 out=14

second terminal: `flock status`, `flock agents <swarm>`, ... from this directory
Ctrl-C stops the run (twice: force)
```

When stdout is not a terminal (CI, `| tee`), it prints each new event once and one status line
per refresh instead.

**Second terminal.** While the run is up, `.flock/current` (mode 0600) holds the controller URL
and the generated token. Every other command uses it when `FLOCK_URL` is not set, so from the
same directory:

```sh
uv run flock status
uv run flock agents hello
uv run flock prompt writer-0001 "Now write a haiku."
uv run flock prompt --swarm hello --type writer --state idle "Next round."
uv run flock events writer-0001 --pretty --follow
```

From elsewhere: `FLOCK_CURRENT=/path/to/.flock/current flock status`, or set `FLOCK_URL` and
`FLOCK_TOKEN` from that file.

**At the end** the run halts the reconcile loop (nothing relaunches) and drains every worker:
idle agents stop at once, in-flight turns get `--grace` seconds. Then each remaining worker gets
SIGTERM: it interrupts its turns (a `turn_interrupted` event) and its harness kills the tool
processes it started. Up to 5 s later, anything still alive is killed, including tool processes
in their own session. Finally the run waits for the last events and writes the bundle. The
budget check, the view and the bundle use the spec as currently applied, so a re-apply or
`flock scale` during the run counts: `limits.budget_usd`, for example.

```
.flock/hello-20261021-161443/
  run.json             spec (env values with TOKEN/KEY/SECRET/... in the name redacted), stop reason,
                       duration, totals (agents, turns, events, cost, tokens), per agent and per worker
  events.jsonl         every agent's events merged by time (pi session format v3, as `flock events`)
  events/<agent>.jsonl one file per agent
  agents/<agent>/      agent workdirs        state/llm/   llm harness sessions
  logs/                controller.log, <worker>.log      controller/  the controller's SQLite state
```

## Switching to HF Jobs

Change `compute: local` to `compute: hf-jobs` (and give `namespace` and `bucket` if the spec does
not have them), deploy a controller ([README](../README.md#run-the-controller),
[docs/deploy-job.md](deploy-job.md)) and `flock apply spec.yaml`. `flock run` refuses a
`hf-jobs` spec, and a jobs controller refuses a `local` spec (HTTP 400), so a spec never runs
on a backend it did not ask for.

A long-lived local controller without the terminal view also exists:
`FLOCK_TOKEN=... FLOCK_COMPUTE=local uv run flock-controller`, then `flock apply` a `local` spec.

## What differs locally

Same code as on jobs: the spec, the API and CLI, the reconcile loop and placement
(`agents_per_worker`, `workers:`), heartbeats and agent states, prompt admission
(`max_working`), bulk and children prompts, spawn and its limits, events, cost from the pinned
prices, worker lifetime and planned restarts, crash backoff and `crashlooping`. The local tests
exercise placement over two workers, prompts, events, the CLI and stop; spawn is not yet run
locally.

Different:

1. **Workers are processes.** One `python -m flock.worker.main` subprocess per worker slot, with
   the job env contract (`FLOCK_URL`, `FLOCK_TOKEN`, `WORKER_ID`, `WORKER_INCARNATION`, `SWARM`,
   `TYPE`, `DEADLINE`, `DRAIN_WINDOW`, the type's `env`, its secrets). Job ids are `local-NNNN`.
   Exit 0 is `COMPLETED`, any other exit `ERROR` ("exit N"), a process past its lifetime is
   killed and reported as `Job timeout` (a planned restart, as on HF Jobs).
2. **Ignored:** `image`, `flavor`, the network group, the wheel and bootstrap, `namespace` and
   `bucket`. There is no bucket sync: agent workdirs live in `<run>/agents/` (shared by the
   run's workers, so a restarted worker slot finds them on disk) and `llm` sessions in
   `<run>/state/llm`.
3. **Secrets come from your shell.** The names a spec lists (`secrets:`, `api_key_env`,
   `msgboard_invite`) are read from the environment of `flock run` and passed to the workers as
   env, as HF Jobs passes job secrets. Workers inherit your environment minus your own flock
   connection settings (`FLOCK_*`, except `FLOCK_CLAUDE_CLI`, `FLOCK_CODEX_CLI`, `FLOCK_PI_CLI`
   and `FLOCK_PROMPTS_DIR`); the harness still gets only the allow-listed variables plus what the
   worker passes.
4. **Faster loop.** Reconcile every 1 s (jobs: 5 s) and worker heartbeats every 5 s (jobs: 30 s;
   a type's `env: {HEARTBEAT_INTERVAL: ...}` overrides it).
5. **The controller is in-process.** 127.0.0.1 on a free port, a new random token per run, no
   status port, no state sync to a bucket, no jobs proxy auth. Each `flock run` is a new run
   directory: a run does not resume a previous one.
6. **Workers die with the controller.** A worker exits when the `flock run` process is gone
   (`FLOCK_LOCAL_PARENT_PID`), so a killed run leaves no orphan workers. Ctrl-C reaches only
   `flock run`, which drains and stops the workers itself. A local worker handles SIGTERM by
   interrupting its turns (a job worker keeps the default behaviour).
7. **Logs are files.** `<run>/logs/<worker>.log` instead of `hf jobs logs`; `flock workers`
   says so.
8. **No sandbox.** Agents' `bash` runs as you, on your machine, starting in the agent's workdir
   but not confined to it. Use a container or a throwaway user for untrusted tasks.
9. **Harnesses other than `llm`.** `claude`, `codex` and `pi` use the CLIs on your `PATH` (or
   `FLOCK_CLAUDE_CLI`, `FLOCK_CODEX_CLI`, `FLOCK_PI_CLI`). Claude and Codex use your own
   `~/.claude` and `~/.codex`. Pi gets one config dir per worker,
   `<run>/state/pi/<worker>` (`PI_CODING_AGENT_DIR`), because it writes `models.json` for its
   type there; your `~/.pi/agent` is never touched, and its login is not used. These harnesses
   are wired the same way as `llm` but not yet tested locally.

Known gaps: a worker that crashes is noticed by the reconcile loop's usual rules (60 s for a
worker that never heartbeats, 90 s after its last heartbeat), not at once; `limits.budget_usd`
is the only spec limit that stops a run (`alive` and `spawned_total` limit spawning, as on jobs);
tested on Linux, not yet on macOS; Windows is not supported (process groups).
