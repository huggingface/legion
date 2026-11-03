# chatbot-v2

Run 3 of the chatbot experiment: 8 open-model agents (4 DeepSeek-V4.1-Flash, 4 GLM-5.3-Flash)
build ONE conversational chatbot in 4 hours. Phase 1 (2 h): two teams of 4, each in a direction
it chooses, no lead. Phase 2 (2 h): one team of 8 on `best/`, after a short compare round. A
final round ends with GREEN/RED verdicts and strong interviews of the final best. The same
loop, board rules and job shape as chatbot-v1; what changed is the world
([`worlds/chatbot2`](../../worlds/chatbot2/chatbot2.py)) and the prompts.

## What is new against chatbot-v1

| change | where |
|---|---|
| a bot is a file or a **package** (`bot/` with `__main__.py`, one module per feature, each under 20 KB), run like `python -I -m bot` | world: `run_bot`, `qualify`, `promote_best`, `copy_file` (recursive), `best/bot/` + `best/current.json` |
| `grep_file(path, pattern, max_hits)`: `file:line:text` hits in a file or a package | world |
| 2 MB per file; `read_file` keeps its 60 KB window and needs `start`/`end` above it; `best_status` reports files, bytes and lines | world |
| **swarm-written probes**: `add_probe(lines, check, evidence)`, verified by re-running the bot (it must fail); `probes()` lists them with best's pass/fail | world |
| **strict promotion** on the live set (seed probes as the floor + 1 point per swarm probe): a tie does not promote | world |
| **interview bookings**: `book_interview(level, path)` checks the quota, reserves it and freezes a copy; `spawn_interviewer(booking, thread)`; charged on the verdict, released when the interview is cut off; strong and mid quotas apart; the verdict lists evidence for script, then for agent, then the label | world, `tools/interview.py`, `prompts/interviewer.md` |
| **compare round**: promotion locked for `merge.compare_minutes` after the merge | world, `phase2/` prompts |
| prompts: one module per feature; RED becomes a probe before its fix; read the board once per round; `done` only after an edit, a verdict post, a promotion or a probe | `prompts/`, `phase2/`, `protocols/` |
| caps: 4 h at the previous run's measured rate per model, + 30% | `spec.yaml` |

## Files

- `spec.yaml`, `phase2/spec.yaml`: phase 1 and phase 2; only prompt and protocol files differ
  (a test checks it). GLM and DeepSeek types get the same prompts and tools.
- `world.yaml`: the world's knobs, read from `$CHATBOT_CONFIG` and re-read when the file
  changes (the job sets `CHATBOT_CONFIG` to this file, so a live update applies them).
- `prompts/` (phase 1: system, round, final per team; kickoff; interviewer), `phase2/`,
  `protocols/`.
- `tools/interview.py`: `spawn_interviewer(booking, thread)`.
- `job/`: `bootstrap.sh` (the container entrypoint), `switch.sh`, `launch.sh`, `render.py`, `smoke.json`.

## Knobs (`world.yaml`)

| knob | default | what |
|---|---|---|
| `merge.compare_minutes` | 15 | promotion stays locked this long after the merge; 0 skips the compare round |
| `probes.per_agent_per_hour` | 4 | `add_probe` calls per agent per rolling hour |
| `probes.max_lines` | 12 | user lines per probe |
| `probes.max_total` | 150 | swarm probes in the live set |
| `interviews.strong_per_hour` | 2 | strong interviews per team per rolling hour |
| `interviews.mid_per_hour` | 3 | mid interviews per team per rolling hour (a separate quota) |
| `interviews.start_minutes` | 5 | a booking with no chat by then is released |
| `interviews.expire_minutes` | 20 | an interview idle this long is released |

Probe checks: `mentions:<word>` (the reply to the last line holds it, as a whole word),
`mentions_fact:<word>` (the same, and an earlier line must state it), `not_phrase:<text>` (no
reply holds it), `not_repeat`, `not_empty`, `answers_question` (the last line ends with `?`;
its reply has a statement of 3+ words). A crash or a timeout fails every check.

## Run it locally

```sh
cd experiments/chatbot-v2
export HF_TOKEN=... CHATBOT_BOARD_INVITE=... CHATBOT_WORLD_TOKEN=$(openssl rand -hex 16)
export CHATBOT_CONFIG=$PWD/world.yaml
python -m flock.controller.localrun spec.yaml --until idle          # after setting comms.url / comms.channel
# at the 2 h mark, the switch: the world first, then the phase-2 prompts
python ../../worlds/chatbot2/chatbot2.py merge && flock update phase2/spec.yaml
```

## Run it as an HF Job

The same job shape as chatbot-v1: ONE `cpu-upgrade` job running `python -m flock.controller.localrun --until idle` in
local mode, a bucket carrying the payload in and the workspace, bundle and logs out.

```sh
hf buckets create <ns>/<bucket> --private                                  # once
experiments/chatbot-v2/job/launch.sh hf://buckets/<ns>/<bucket>/<run>       # upload; launches nothing
hf jobs run --detach --flavor cpu-upgrade --timeout 270m --name chatbot-v2 \
  --label experiment=chatbot-v2 \
  --secrets HF_TOKEN --secrets CHATBOT_BOARD_INVITE --secrets CHATBOT_WORLD_TOKEN \
  -e CHATBOT_BUCKET=hf://buckets/<ns>/<bucket>/<run> \
  -e CHATBOT_BOARD_URL=<board url> -e CHATBOT_CHANNEL=<channel> -e CHATBOT_RUN=<run> \
  python:3.12-slim bash -c "$(cat experiments/chatbot-v2/job/bootstrap.sh)"
```

Control markers in `<bucket>/<run>/control/` (checked every 30 s):

- `switch`: `job/switch.sh` (world merge, then `flock update phase2/spec.local.yaml`); the
  output lands in `control/switch.log`.
- `update`: re-syncs `<bucket>/<run>/experiment/` over the experiment (spec, prompts,
  `world.yaml`), runs `flock update` for the current phase and deletes the marker; the diff
  lands in `control/update-<n>.log`. To skip the compare round: `compare_minutes: 0` in
  `world.yaml`, then `update`.
- `stop`: flock drains and writes its bundle, then the final sync. Prefer it to
  `hf jobs cancel`, which leaves the container only seconds.

`CHATBOT_SMOKE=1` applies `job/smoke.json` (20 min loop at 60 s, $2.50 per agent, $15 total) for
a smoke job; `-e CHATBOT_SWITCH_AT=600` switches at +10 min. For a 3-min compare round, upload
`job/smoke-world.yaml` as `<bucket>/<run>/experiment/world.yaml` first (applied at the switch).
With `-e CHATBOT_RUN=<name>`, `variants.<name>` in `smoke.json` overrides those settings
(`job/render.py`). `resmoke` is a 10-min interview-path check: $0.80 per agent, $5 total, 1-min
compare round and booking start window (its `world` knobs, merged into `world.yaml` at start),
and a round-prompt line asking each agent to book a mid interview and spawn its interviewer
in round 2. Use it with `-e CHATBOT_SWITCH_AT=240`.

## The chat Space contract

The world mirrors `best/` to `$CHATBOT_BUCKET` after every promotion. `best/current.json` says
what best is: `{"n", "kind": "file" | "package", "entry": "best/bot.py" | "best/bot",
"files": [...]}`. Snapshots are `best/v<n>.py` or `best/v<n>/`. A reader downloads `files`,
then runs a file with `python -I bot.py`, or a package by putting its directory in an empty
dir as `bot/` and running `python -I -c "import sys,runpy; sys.path.insert(0,'.');
runpy.run_module('bot', run_name='__main__', alter_sys=True)"` there (or `python -m bot`
without `-I`). Files a package no longer has are deleted from the bucket.

## Offline tests (no network, no cost)

- `tests/test_chatbot2_world.py`: the world in process.
- `tests/test_chatbot2_experiment.py`: the experiment under `localrun` with the fake model
  server and the fake board (8 agents, both phases, the compare round, bookings, probes,
  promotions, the final round), the phase-2 spec check, and `job/bootstrap.sh` run locally
  (live update, switch marker, stop marker).
