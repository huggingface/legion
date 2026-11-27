# Run results

`legion.results.load(path)` reads a finished run into one typed result: an outcome for each
Trial, each agent and the run, the run and attempt IDs, the spec's limits and task, timing, and
the paths of Harbor's own Trial files and of the resources' exports (the board's). Experiment scorers read this instead of parsing `.legion/`
bundles, `run.json` or Slurm run directories themselves. `legion results PATH` prints the same
result as JSON. Neither needs a controller.

```python
from legion import results

r = results.load("runs/aggregation-n4")  # a Slurm run dir, a local bundle, or a folder holding one
if r.outcome.completed:
    rewards = {a.name: a.reward for a in r.agents}
else:
    print(r.outcome, r.reasons)  # an infrastructure outcome: nothing to score
for a in r.agents:
    trial = a.trials[-1]
    r.root / trial.result_path, r.root / trial.verifier_dir  # Harbor's files, not copies
boards = [r.root / e.path for e in r.exports if e.resource == "message_board"]
```

Every path in a result is relative to `r.root`, so a run directory that was copied or moved
gives the same result, and a saved result stays valid next to its run.

`path` is one of these:

- a local run bundle, `.legion/<swarm>-<time>/` ([local runs](local.md));
- a folder holding exactly one bundle under `.legion/`, such as a CIB case folder;
- a Slurm run directory, `legion init`'s RUN_DIR ([Slurm](slurm.md)).

Read a Slurm run after `legion stop --release`. Its Trials are verified as the workers drain.

## Outcomes

`Outcome` is a string enum. `completed` is the only benchmark result. Every other value is an
infrastructure outcome, so there is nothing to score.

| outcome | meaning |
|---|---|
| `completed` | The verifier gave finite rewards and Harbor recorded no exception. A reward of 0 is completed. |
| `preempted` | The agent's worker exited preempted (Slurm exit 75), whether or not it was relaunched. |
| `worker_replaced` | The agent's worker was relaunched after a crash, a kill or a planned restart. A relaunched worker reuses its Trial names, so its Trial files can be overwritten ([#121](https://github.com/huggingface/legion/issues/121)). |
| `worker_lost` | The agent's worker started and never recorded an exit. |
| `timeout` | Harbor's `AgentTimeoutError`, `AgentSetupTimeoutError`, `VerifierTimeoutError` or `EnvironmentStartTimeoutError`. |
| `harness_error` | Any other exception in the Trial, from Pi or Harbor, including a cancelled Trial. |
| `unknown` | No evidence: no `result.json`, no verifier result, a missing planned agent, a spec that doesn't load, or a run that didn't end. |

- A **Trial**'s outcome comes from its own `result.json`.
- An **agent** takes an infrastructure outcome (`preempted`, `worker_replaced` or
  `worker_lost`) from the worker launches it was placed on, if they have one. An agent placed on
  two launches, the same worker relaunched or another worker, is `worker_replaced`: it starts
  its Trial names again there. Otherwise it takes its latest Trial's outcome, and an agent with
  no Trials is `unknown`. `agent.reward` is its latest Trial's reward when that Trial completed,
  even for an agent with an infrastructure outcome. So the Trial's own result stays visible
  next to the infrastructure outcome.
- The **run** is `completed` when every agent is and nothing else is wrong. Otherwise it takes
  the first outcome found, in table order. `reasons` lists every problem, for example
  `agent-002: worker_replaced, worker tiny-writer-w0001`.

A sweep or a scorer decides what to retry. Every outcome except `completed` is an
infrastructure outcome.

## Fields

- `RunResult`
  - `run_id`: the logical run. A sweep attempt's run directory holds its receipt,
    `attempt.json`. Its `run_id` and `attempt` are used when it records them, else the attempt's
    name without its `-rN` suffix and that `N`. Without a receipt, `run_id` is the swarm's name,
    even one ending in `-rN`, and `attempt` is 0.
  - `attempt_id`: this execution, the run directory's name.
  - `swarm`, `compute` (`local` or `slurm`), `root` (absolute where it was loaded), `outcome`
    and `reasons`.
  - `started_at`, `ended_at`, `duration_s`, and a local run's `stop_reason`.
  - `limits`: the spec's limits, `task` (repo, revision, selector) and `planned_agents`.
  - `agents`, `workers` and `exports`.
- `AgentResult`: `name`, `workers` (every worker it was placed on), `outcome`, `reward` and
  `trials`, oldest first.
- `TrialResult`: `name`, `outcome`, `reward` (`rewards["reward"]`, or the only reward), `rewards`,
  `exception_type`, `started_at`, `finished_at`, `result_path`, `result_sha256` and
  `verifier_dir`. A multi-step Trial's verifier files are under `steps/<name>/verifier/`, next to
  `result_path`.
- `WorkerResult`: `id`, `agents`, `crashes` (local runs) and `incarnations`. Each incarnation
  has `job_id`, `host`, `started_at`, `ended_at`, `exit` (`drained`, `preempted`, or `None`
  when no exit was recorded) and `agents`.
- `ResourceExport`: `worker`, `resource` and `path`, one per exported file, such as the
  message board's `board.jsonl` and `channels.json`. A Slurm worker exports into its own folder
  as it drains. A local run's workers share one folder, so `worker` is `None` there. A preempted
  worker exports nothing ([#121](https://github.com/huggingface/legion/issues/121)).

A local `run.json` records the compiled spec, so its `limits` are those of the compiled spec:
`agent_max_turns`, `agent_timeout_seconds`, `trial_hours` and the budget. A Slurm run's come from
its `experiment/spec.yaml`, resolved with `cluster.env` and `run.env`, and include
`run_timeout_seconds`. Secret values are never part of the result.

## Where it reads from

Nothing is copied or written. The result is read from what the run leaves:

```
local: .legion/<swarm>-<time>/
  run.json                                    the spec, the agents and their worker, workers' crashes
  agents/_harbor/<agent>/trials/<trial>/      Harbor's result.json and verifier/
  agents/_harbor/_worker/resources/<name>/    the resources' exports
  agents/_harbor/_workers/<worker>/<incarnation>.json
slurm: RUN_DIR/
  experiment/spec.yaml, cluster.env, run.env  the spec as the run resolved it
  attempt.json                                a sweep attempt's receipt, if any
  results/<swarm>/<agent>/trials/<trial>/     Harbor's result.json and verifier/
  results/<swarm>/<worker>/resources/<name>/  the resources' exports
  results/<swarm>/_workers/<worker>/<incarnation>.json
  launch/<worker>-<incarnation>.env           one per worker launch
```

Each worker launch writes an incarnation record. It records the Slurm job, host, start time and
the agents placed on the worker as they arrive. It is completed with `ended_at` and `exit` once
the worker has drained (its Trials ended and were verified) or has been preempted. A worker that
was killed leaves its record without an exit. Runs from before these records exist still show
relaunches through their `launch/` env files, but which agents the worker held is unknown, so
the run itself is `worker_replaced`.

Keeping Trial results across a relaunch is still open
([#121](https://github.com/huggingface/legion/issues/121)). Until it lands, the affected agents
are reported as `preempted` or `worker_replaced` rather than scored.
