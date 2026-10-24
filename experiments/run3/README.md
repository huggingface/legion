# run3: one chatbot, no imposed structure

Run 3 of the chatbot experiment. Many agents build ONE natural, friendly chatbot (pure-stdlib
Python, stdin to stdout, a package or one file, NOT a therapist) with no teams, no lead and no
phases. They organize themselves on one board channel: one thread per subtask or approach,
`search` before starting. Agents join in waves (20 every 5 min to 100, then 20 per hour to 300,
then the operator decides; cap 1,000). Promotion and strong interviews are open from the start.

Nothing here launches anything. Every value for the operator is a placeholder.

## Files

| file | what |
|---|---|
| `spec.yaml` | the source spec: types `agent` (thinking off), `agent-think` (thinking on, 0 by default), ephemeral `interviewer`; loop, board, limits |
| `render.py` | fills in the environment and writes the spec that runs; `--variant selfhosted` (default, single job), `smoke` (30-min loop on the endpoint), `jobs` (HF Jobs, world on a Space), `router-smoke` (HF router, $ caps) |
| `spec.smoke.yaml`, `spec.jobs.yaml`, `spec.router-smoke.yaml` | the derived variants, generated (`render.py --variant X --committed`; a test checks they are current) |
| `waves.yaml`, `waves-smoke.yaml`, `waves-router-smoke.yaml` | the wave schedules (run; 30-min smoke 20/20/20 cap 40; router smoke 2/2/2) |
| `load/` | `load_smoke.py` (the endpoint's load smoke) and `job.sh` (it as a cpu-basic job) |
| `waves.py` | the wave runner: `plan`, `status`, `tick`, `run`, `add N`, `set N`, `pin` |
| `world.yaml` | the world's knobs (`$CHATBOT_CONFIG`; re-read on change) |
| `prompts/` | `system.md`, `round.md` (short), `kickoff.md`, `final.md`, `interviewer.md`; the newcomer prompt is `loop.first` in `spec.yaml` |
| `protocols/` | `run3.md` (self-organization), `chatbot.md` (GREEN/RED, RED -> probe -> fix) |
| `tools/interview.py` | `spawn_interviewer(booking, thread)` |
| `job/` | `bootstrap.sh` (the single job), `launch.sh` (payload + the `hf jobs run` line) |
| `world-space/` | the world as a Docker Space (`flock world serve`): `Dockerfile`, `start.sh`, `prepare.sh` |

The world is `worlds/chatbot3` (on `worlds/chatbot2`): `agents/<name>/` per agent (own-dir
writes, everyone reads and runs everything), `best/` (beat-best gate: strictly better on the
live probe set, ties refused, never lower on the seed probes; the qualify bar shared with
`qualify` and `book_interview`), swarm probes with verified evidence, strong-only interview
bookings with no quota (still reserved, frozen, charged on a verdict and released when cut off,
for the records), the `turn`/`agent_done` hooks, `log.jsonl` and the bucket mirror of `best/`.

## Knobs

| where | knob | default |
|---|---|---|
| `waves.yaml` | `waves`, `cap`, `think_share` | 20/5 min to 100, 20/h to 300; cap 1,000; 0 thinking agents |
| `spec.yaml` loop | `every`, `wake`, `while` | `turn_end`; `{min_gap: 10s, max: 20, window: 30m}`; `until: 48h`, `max_rounds: 6000` |
| `spec.yaml` types | `max_turns`, `context_tokens`, `agents_per_worker` | 12, 64000, 25 |
| `spec.yaml` | `spawn.agent.max`, `limits.alive` | 1 live interviewer per agent, 2,000 |
| `spec.yaml` comms | `min_post_gap`, `max_len` | 10 s, 800 |
| `world.yaml` | `probes.per_agent_per_hour`, `max_lines`, `max_total` | 4, 12, 400 |
| `world.yaml` | `runs.parallel` | 0 (= the world's CPU count): bot runs at once, read at the world's start |
| `world.yaml` | `interviews.start_minutes`, `expire_minutes` | 5, 20 (records only) |

## The endpoint: vLLM checklist

Configuration is `OPENAI_BASE_URL` (ending in `/v1`), an optional `OPENAI_API_KEY`, and
`MODEL_WORKER` (the served id; interviewers use it too unless `MODEL_INTERVIEWER` is set).

```sh
vllm serve <hf model path> --served-model-name <MODEL_WORKER> \
  --enable-auto-tool-choice --tool-call-parser <the DeepSeek parser for this model> \
  --reasoning-parser <the DeepSeek reasoning parser> \
  --max-model-len 65536 \
  --max-num-seqs <>= live agents + live interviewers> \
  --enable-prefix-caching \
  [--api-key <OPENAI_API_KEY>]
```

- **Tool calling on, with the DeepSeek tool parser**, and `tool_choice: "required"` accepted
  (every agent request sends it).
- **Context at least 64k** (`--max-model-len`): the harness compacts at 64,000 tokens and asks
  for up to 32,000 output tokens (`max_tokens`).
- **Thinking.** DeepSeek-V4.1-Flash's chat template defaults to thinking ON. The `agent` type
  sends `chat_template_kwargs: {thinking: false}`; `agent-think` sends `{thinking: true}`. The
  server needs the reasoning parser on, so thinking arrives as `reasoning_content`; the
  harness shows it in the transcript and never sends it back (tested).
- **`--max-num-seqs` at or above the live agents** (plus interviewers), or turns queue on the
  server.
- **Prefix caching on**: every turn resends the agent's history.
- Reachable over HTTPS from the job(s); the key, if any, is a job secret.

## Waves

`waves.py` raises `types.agent.replicas` (and `agent-think`, by `think_share`) in the rendered
spec and applies it the way `flock update` does, so the whole folder goes with it. The count on
the controller is the truth: a wave never lowers it, and it never exceeds `cap`. Each change is
a JSON line in `<spec>.waves.jsonl`. The schedule file is re-read at every tick.

- **Single job:** the bootstrap ticks every `CHATBOT_CHECK_S` (30 s) from the job's start.
  `control/add` (content: a number) adds that many now; `control/waves-off` stops the schedule;
  `control/update` re-syncs `<bucket>/experiment/`, re-renders, pins the live count
  (`waves.py pin`) and runs `flock update`.
- **Jobs mode:** the operator session runs `python waves.py run --spec spec.jobs.local.yaml`
  with `FLOCK_URL`/`FLOCK_TOKEN` set; it stops when the schedule is over (300).
  `waves.py status` shows the count and the next wave; `waves.py add 100` or `set 250`
  overrides (`set` may lower). `flock scale run3 agent N` also works; the next wave never
  lowers it.

## Runbook A: single job (smokes, up to ~50 agents)

A single job is for smokes and small runs. Whether one HF Job can run 48 h is not verified
(docs/deploy-job.md: the timeout has no documented cap; workers in jobs mode restart at 24 h
plus jitter by design). Use runbook B for the 48 h run at scale.

1. Commit, then build the payload (it launches nothing):
   `experiments/run3/job/launch.sh hf://buckets/<ns>/<bucket>/<run>`
2. Optional smoke without the endpoint (HF router, $15 cap, 20 min, 2/4/6 agents):
   ```sh
   hf jobs run --detach --flavor cpu-upgrade --timeout 40m --name run3-smoke --label experiment=run3 \
     --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN \
     -e CHATBOT_BUCKET=hf://buckets/<ns>/<bucket>/<run>-smoke -e CHATBOT_BOARD_URL=<board url> \
     -e CHATBOT_CHANNEL=<test channel> -e CHATBOT_RUN=smoke1 \
     -e RUN3_VARIANT=router-smoke -e RUN3_WAVES=waves-router-smoke.yaml \
     python:3.12-slim bash -c "$(cat experiments/run3/job/bootstrap.sh)"
   ```
3. The run (the endpoint; cap the waves with an edited `waves.yaml` at <= 50):
   ```sh
   hf jobs run --detach --flavor cpu-xl --timeout 48h --name run3 --label experiment=run3 \
     --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN --secrets OPENAI_API_KEY \
     -e CHATBOT_BUCKET=hf://buckets/<ns>/<bucket>/<run> -e CHATBOT_BOARD_URL=<board url> \
     -e CHATBOT_CHANNEL=<channel> -e CHATBOT_RUN=<run suffix> \
     -e OPENAI_BASE_URL=<vllm url>/v1 -e MODEL_WORKER=<served model id> \
     python:3.12-slim bash -c "$(cat experiments/run3/job/bootstrap.sh)"
   ```
4. Live: upload `control/add` (a number), `control/waves-off`, or the edited folder to
   `<bucket>/<run>/experiment/` then `control/update`. Results of each in `control/*.log`.
5. Stop: upload `control/stop` (drain, final round, bundle, last sync). Not `hf jobs cancel`.
6. After: `<bucket>/<run>/ws/` (agents, best, probes, interviews, verdicts, log),
   `best/` (mirror, `current.json`), `flock/.flock/<run>/` (bundle), `logs/` (incl.
   `waves.jsonl`).

## Runbook B: jobs mode (50 to 1,000 agents, 48 h)

Controller as an HF Job (docs/deploy-job.md), workers as HF Jobs in its network group at 25
agents each (40 agent workers at the cap, up to 40 interviewer workers), the world on a Space.

1. Wheel: `W=$(uv run scripts/publish-wheel.sh hf://buckets/<ns>/flock/dist)`.
2. World Space: `experiments/run3/world-space/prepare.sh <dir>`; create a public Docker Space,
   set its secrets `CHATBOT_WORLD_TOKEN`, `HF_TOKEN` and the variable
   `CHATBOT_BUCKET=hf://buckets/<ns>/<bucket>/<run>`, pick CPU hardware (8+ vCPU; see risks),
   enable persistent storage if possible, `hf upload <ns>/<space> <dir> --repo-type space`.
   Check `GET https://<space host>/manifest` with the `X-World-Token` header.
3. Controller: `scripts/deploy-job.sh "$W" --group run3 --flavor cpu-upgrade --timeout 50h
   --secret CHATBOT_WORLD_TOKEN --secret CHATBOT_BOARD_INVITE --secret OPENAI_API_KEY`;
   `export FLOCK_URL=https://<job>--7860.hf.jobs FLOCK_TOKEN=<token>`.
4. Spec: `OPENAI_BASE_URL=<vllm>/v1 MODEL_WORKER=<id> CHATBOT_BOARD_URL=<board>
   CHATBOT_CHANNEL=<channel> CHATBOT_RUN=<suffix> RUN3_NAMESPACE=<ns>
   RUN3_BUCKET=hf://buckets/<ns>/<bucket> RUN3_WORLD_URL=https://<space host>
   python experiments/run3/render.py --variant jobs --out experiments/run3/spec.jobs.local.yaml`
5. `flock apply experiments/run3/spec.jobs.local.yaml` (replicas 0: the swarm, the kickoff, the
   world check; no worker yet).
6. Waves, in a session that stays up (tmux):
   `python experiments/run3/waves.py run --spec experiments/run3/spec.jobs.local.yaml`.
   After 300: `waves.py status`, then `waves.py add N` as decided.
7. Live changes: edit prompts/spec, re-render (step 4), `waves.py pin --spec ...` (the live
   count into the file), then `flock update experiments/run3/spec.jobs.local.yaml`.
8. Renewal: workers restart on their own at 24 h + up to 4 h (sessions from the bucket). Renew
   the controller before its timeout: `scripts/deploy-job.sh "$W" --group run3 --replace`
   (heartbeats pause ~1 min; re-export `FLOCK_URL`).
9. Stop: `flock loop run3 --type agent --stop` (and `--type agent-think`; final round), then `flock remove run3`, then
   `scripts/deploy-job.sh --down --group run3 "$W"`; stop the Space.

## Tests (offline, fakes only)

`tests/test_chatbot3_world.py` (the world in process) and `tests/test_run3_experiment.py`: the
schedule parser, the rendered variants, a jobs-mode dry run (controller + `FakeJobsClient`,
the world behind `world.url`, waves 0 -> 40 -> 1,000 with 40 worker slots, `set` lowering), and
the single-job bootstrap end to end (3 waves 0 -> 20 -> 40 with the third capped, the newcomer
prompt first for all 40 agents, a promotion at once and 39 ties, 40 bookings without quota,
verdicts on frozen copies, the stop marker, the bundle in the bucket, no secret in it).

## Pre-launch checks

- **Load smoke** (`load/load_smoke.py`): the run-3 request shape (46k-token prompts, one fresh
  tool result per call, 135-token outputs, `tool_choice: required`) at C = 8 to 128 for 3 min
  each; req/s, p50/p95, errors, and vLLM's `/metrics` if reachable; the knee, the agent cap
  and the first wave for one node. From a shell with `OPENAI_BASE_URL`/`MODEL_WORKER` set:
  `python experiments/run3/load/load_smoke.py --out <dir>`, or as a cpu-basic job with
  `load/job.sh`.
- **30-min smoke**: `RUN3_VARIANT=smoke RUN3_WAVES=waves-smoke.yaml` on runbook A.

The exact commands, markers and the 12-point checklist are in the operator's pre-launch note.
