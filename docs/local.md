# Local mode: `flock launch`

One spec from a laptop to 1k agents. `compute: local` runs the swarm on your machine:
`flock launch spec.yaml` starts the controller on 127.0.0.1 and launches each worker as a
subprocess, both in the background (detached; output in `.flock/launch.log`, the run directory
under `.flock/<swarm>-<UTC time>/`), then returns. `flock status`, `flock prompt`,
`flock update` and `flock stop` find the run through `.flock/current` in the same
directory. The controller writes a run bundle when it ends. The
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
uv run flock launch examples/local.yaml
uv run flock prompt --swarm hello --type writer "Introduce yourself in one sentence."
uv run flock status hello
uv run flock stop hello --release      # drain the workers; `flock launch hello` resumes
uv run flock stop hello --delete --yes # end the controller, write the bundle
```

For tests and dry runs, `python -m flock.controller.localrun examples/local.yaml --prompt "..." --until idle`
runs the same thing in the foreground and exits when done.

- `flock prompt --swarm <swarm> --type <type> "text"` sends a prompt to every agent of a type
  (optional; without it the agents wait for prompts).
- `flock stop <swarm>` holds the swarm; `flock stop <swarm> --release` drains the workers and keeps
  the controller (`flock launch <swarm>` resumes); `flock stop <swarm> --delete` ends the
  controller and writes the bundle. A run also ends at the spec's `limits.budget_usd`. The test harness (`python -m flock.controller.localrun`) additionally has
  `--until idle|10m` and `--grace S`.
- A `tools/` folder next to the spec ([tools.md](tools.md)) is checked, uploaded and named in
  the spec by its hash before the run starts, as `flock launch` does.
- After editing the experiment folder (spec, prompt files, tools), run `flock update <folder>` to
  apply the change to the running swarm; a broken save is shown and not applied
  ([live.md](live.md)).

## Reaching the run

While the run is up, `.flock/current` (mode 0600) holds the controller URL
and the generated token. Every other command uses it when `FLOCK_URL` is not set, so from the
same directory:

```sh
uv run flock status
uv run flock status hello --type writer
uv run flock prompt writer-0001 "Now write a haiku."
uv run flock prompt --swarm hello --type writer --state idle "Next round."
uv run flock status writer-0001 --events --follow
```

From elsewhere: `FLOCK_CURRENT=/path/to/.flock/current flock status`, or set `FLOCK_URL` and
`FLOCK_TOKEN` from that file.

**At the end** the run halts the reconcile loop (nothing relaunches) and drains every worker:
idle agents stop at once, in-flight turns get 10 s (`--grace` of the foreground harness). Then each remaining worker gets
SIGTERM: it interrupts its turns (a `turn_interrupted` event) and its harness kills the tool
processes it started. Up to 5 s later, anything still alive is killed, including tool processes
in their own session. Finally the run waits for the last events and writes the bundle. The
budget check, the view and the bundle use the spec as currently applied, so a
`flock update` during the run counts: `limits.budget_usd`, for example.

```
.flock/hello-20261021-161443/
  run.json             spec (env values with TOKEN/KEY/SECRET/... in the name redacted), stop reason,
                       duration, totals (agents, turns, events, cost, tokens), per agent and per worker,
                       `loops` (per type: state, stop reason, rounds, final round) and `budget`
                       (budget_usd, final reserve, cost, which budget stop fired)
  events.jsonl         every agent's events and the swarm-level ones (loop_*, final_done,
                       budget_stop; agent ""), merged by time (pi session format v3)
  events/<agent>.jsonl one file per agent
  agents/<agent>/      agent workdirs        state/llm/   llm harness sessions
  logs/                controller.log, <worker>.log      controller/  the controller's SQLite state
```

## Switching to HF Jobs

Change `compute: local` to `compute: hf-jobs` (and give `namespace` and `bucket` if the spec does
not have them), deploy a controller ([README](../README.md#run-the-controller),
[docs/deploy-job.md](deploy-job.md)) and `flock launch spec.yaml`. `flock launch` of a
`hf-jobs` spec goes to the jobs controller, and a jobs controller refuses a `local` spec (HTTP 400), so a spec never runs
on a backend it did not ask for.

A long-lived local controller can also be started by hand:
`FLOCK_TOKEN=... FLOCK_COMPUTE=local uv run flock-controller`, then `flock launch` a `local` spec.

## What differs locally

- Agent names follow the spec's `naming:` as on HF Jobs: `<type>-NNNN` by default,
  `<adjective>-<noun>` with `naming: words` (docs/spawn.md "Names"); the run bundle's
  `events/<name>.jsonl` and `agents/<name>/` use the same names.

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
   `msgboard_invite`) are read from the environment of `flock launch` and passed to the workers as
   env, as HF Jobs passes job secrets. Workers inherit your environment minus your own flock
   connection settings (`FLOCK_*`, except `FLOCK_PI_CLI` and `FLOCK_PROMPTS_DIR`); the harness still gets only the allow-listed variables plus what the
   worker passes.
4. **Faster loop.** Reconcile every 1 s (jobs: 5 s; `FLOCK_RECONCILE_INTERVAL` overrides both)
   and worker heartbeats every 5 s (jobs: 30 s; a type's `env: {HEARTBEAT_INTERVAL: ...}`
   overrides it). The foreground run checks its stop conditions every 2 s (`FLOCK_LOCAL_REFRESH`).
5. **The controller runs next to you.** 127.0.0.1 on a free port, a new random token per run, no
   status port, no jobs proxy auth. The controller state syncs to `FLOCK_STATE_URL`, by
   default `<bucket>/controller/<swarm>` when the spec has a bucket and an HF token is found
   (HF_TOKEN or the saved one; else off, said at start); `FLOCK_STATE_URL=` turns it off. Each `flock launch` is a new run
   directory: a run does not resume a previous one.
6. **Workers die with the controller.** A worker exits when the controller process is gone
   (`FLOCK_LOCAL_PARENT_PID`), so a killed run leaves no orphan workers. `flock stop --delete` makes the
   controller drain and stop the workers itself. A local worker handles SIGTERM by
   interrupting its turns (a job worker keeps the default behaviour).
7. **Logs are files.** `<run>/logs/<worker>.log` instead of `hf jobs logs`; `flock status`
   says so.
8. **No sandbox.** Agents' `bash` runs as you, on your machine, starting in the agent's workdir
   but not confined to it. Use a container or a throwaway user for untrusted tasks.
9. **The `pi` harness.** It uses the `pi` CLI on your `PATH` (or `FLOCK_PI_CLI`) and gets one
   config dir per worker, `<run>/state/pi/<worker>` (`PI_CODING_AGENT_DIR`), because it writes
   `models.json` for its type there; your `~/.pi/agent` is never touched, and its login is not
   used. It is wired the same way as `llm` but not yet tested locally.

A worker process that exits is noticed on the next reconcile pass and relaunched after
`compute.crash_backoff` (2 s locally; docs/live.md "Worker liveness").

Known gaps: `limits.budget_usd` is the only spec limit that stops a run (`alive` and
`spawned_total` limit spawning, as on jobs); tested on Linux, not yet on macOS; Windows is not supported (process groups).
