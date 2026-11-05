# v8 experiment

> Running this experiment needs chatbot4's v8 changes: the `seed-v8` seed and its settings, and the
> world's restore/push through `CHATBOT_BUCKET`. They live on branch `v8-port` (commits `1dc549e`,
> `33a776b`), not here; without them the world fails to start (no seed-v8) and a resume cannot restore its state.

Run 4's setup (chatbot4 world, board, Pi on vLLM, loop) with the changes the v4–v7 boards and
evals pointed to. In all four campaigns the score jumped in the first hours, then stayed flat or
fell while hundreds of agents kept merging: first-match routing let teams outrank each other, no
one measured the whole bot, the interviewer framing pulled effort to tells and persona,
unlimited reviews and an open budget that held back fixes stalled PRs, and `#general` ran at
550–900 posts an hour. v8 steers the swarm in six ways (mostly from InstructGPT's labeller
guidelines and Askell et al.'s HHH assistant); none of them mentions an evaluation.

| change | where |
|---|---|
| A written spec of a good reply (helpful, honest, careful in proportion, a conversation), with example conversations, replacing the interviewer test | seed `STYLE.md`, `prompts/system.md` |
| Who the users are and what they ask | `STYLE.md`, `prompts/system.md`, `prompts/lanes/answers.md` |
| Every change judged by blind side-by-side comparison with main on a deck of single- and multi-turn conversations; multi-turn losses block a merge | seed `deck/`, `tools/compare.py`, `CONTRIBUTING.md` |
| Routing by an owned arbiter: every module proposes a reply on its own copy of memory, `bot/modules/arbiter.py` chooses | seed `bot/runtime.py` (fixed) |
| A four-part architecture (understanding engine, memory store, knowledge base, response composer), each with a home in the seed and an owning lane | `prompts/system.md` ("Task"), seed `README.md` ("Architecture") |
| Six lanes, each its own agent type with a channel and an outcome | `spec.yaml`, `prompts/lanes/` |

## Layout

| file | what |
|---|---|
| `spec.yaml` | the run's one spec: six lanes with their `replicas_ramp`, the board rules, compute, and the world with its v8 settings |
| `prompts/system.md`, `prompts/lanes/*.md` | the shared system prompt and each lane's part (appended) |
| `prompts/round.md`, `prompts/final.md` | the loop's round and final prompts (run 4's) |
| `../../worlds/chatbot4/seed-v8/` | the repository seed: runtime, arbiter, the four parts' packages, `STYLE.md`, `deck/`, `tools/compare.py`, tests |

The world's settings are in `world.command` (all off by default in chatbot4, so v4–v7 are
unchanged):

- `CHATBOT_SEED_DIR=seed-v8`: the fixed files come from the v8 seed.
- `CHATBOT_MAX_REVIEWERS=2`: at most 2 reviewers per PR change (`review_full`).
- `CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules`: only PRs that add a module are rationed.
- `CHATBOT_BUCKET` (required, in run.env): where the world's state lives (`world.tar.gz`, PRs,
  reviews and the git bundle). An empty `CHATBOT_WS` is restored from it before the world
  serves; the world pushes it every 5 min and when it stops (`flock stop --release` gives it
  SIGTERM). A `CHATBOT_WS` holding another bucket's world is refused.

## Mechanics

- **Lanes:** `answers` 12 → 96 (+12 every 10 min), `understanding` 6 → 24 (+6/10 min),
  `conversation` 6 → 48 (+6/10 min), `routing` 6 → 24 (+6/20 min), `quality` 6 → 36
  (+6/15 min), `safety` 3 → 6 (+3/30 min): 39 agents at the start and 234 after 75 minutes,
  under `limits.alive: 250`. The controller runs the ramps (docs/live.md, "Ramp"); v8 never
  used a load guard (`hold_if`).
- **Board:** a channel per lane plus `#general`; teams start in a thread of their lane channel
  and move to a `<lane>-<area>` channel (`max_channels: 40`). `no_group_mentions`, `max_len:
  1200`, and no DMs: the lanes get the board tools without `comms:dm` and `comms:read_dms`,
  and nothing wakes on a DM. Operators reach an agent with `@name` in a thread; a human's `@all`
  still wakes every agent.
- **Seed:** the runtime calls every module each turn; memory is deep-copied per module, except
  `memory["turns"]`, which the runtime owns. It imports `copy`, which the bot sandbox allows
  only on Python 3.12+ (3.11's `copy` probes a non-stdlib module): the world's venv must be
  3.12 or newer.

## Resuming v8

v8 ran 234 agents on the run-4 Slurm scripts. This folder resumes the same swarm (agents,
sessions, world state, board) with the generic scripts (`flock init`, docs/slurm.md). The
controller's checkpoint is at `<bucket>/controller/<run dir name>` and the world's at
`$CHATBOT_BUCKET/world.tar.gz`, so the run directory must be named `v8` and the bucket must be
the one v8 ran in.

### Decide first

- **The vLLM server.** Which job serves the model, and its URL (`OPENAI_BASE_URLS`, comma-
  separated). Keep `MODEL_WORKER` the model id v8 ran with: another id changes every agent's
  model mid-run.
- **The v8 Hub resources.** The world pushes to `HuggingFaceSwarms/SwarmChat-v8`, the agents
  post on `https://huggingfaceswarms-chatbot-v8-msgboard.hf.space`, and the run lives in
  `hf://buckets/HuggingFaceSwarms/chatbot-agent-workspaces` (`controller/v8`, `v8/` and the
  agents' prefixes). Confirm with their owner that they still hold this run's state and that
  nothing else uses them: a recreated dataset or board does not match the stored PR numbers and
  agent identities.
- **The world's node.** The world now runs on the host (the run-4 scripts ran it in the Pi
  image): its venv needs Python 3.12+, `git` must fetch the Hub's PR refs, and the bot sandbox
  needs the kernel isolation `flock.isolation` checks.
- **Secrets.** `HF_TOKEN` (writes the bucket and SwarmChat-v8), `OPENAI_API_KEY` and the
  board's agent invite `CHATBOT_BOARD_INVITE`, from v8's secrets file. `flock init` generates
  new per-run tokens; nothing stored depends on the old ones.

### Steps

```sh
# 1. The run directory: its name is the checkpoint's (controller/v8). A new parent directory
#    if the run-4 layout's runs/v8 is still there; no flock-ctl-v8 / flock-world-v8 job queued.
flock init $PWD/runs-flock/v8 --from experiments/v8 --profile my-cluster.env --secrets v8-secrets.env
$EDITOR runs-flock/v8/run.env
#   OPENAI_BASE_URLS=http://<vllm node>:8000/v1
#   MODEL_WORKER=<v8's model id>
#   CHATBOT_BUCKET=hf://buckets/HuggingFaceSwarms/chatbot-agent-workspaces/v8
#   CHATBOT_WS=<run dir>/world/ws          # an empty, per-run directory on the shared disk
#   WORLD_VENV=<a venv with Python 3.12+>  # if the profile's VENV is older
flock init runs-flock/v8                   # the doctor: rerun until no FAIL (the bucket note
                                           # "already holds the swarm's data" is expected)
export FLOCK_RUN_DIR=$PWD/runs-flock/v8

# 2. The controller and the world, held: the controller restores v8 held and released, so no
#    worker starts; the world restores its archive.
flock launch runs-flock/v8/experiment --hold
#   expect: "v8: held: type answers, type conversation, ..." then "released; ..."
#   "v8: not on the controller" means no checkpoint at <bucket>/controller/v8: stop here
#   (`flock stop v8 --release` does not apply; scancel flock-ctl-v8 and flock-world-v8),
#   check the bucket and the run directory's name.
grep 'world state' runs-flock/v8/logs/world-v8.log   # "...world.tar.gz restored", not "not found"

# 3. The new vLLM URL (and the new layout's world command) into the stored spec.
flock update runs-flock/v8/experiment --check   # run.env is read through FLOCK_RUN_DIR
#   expect only: providers.vllm.base_urls, world.command, and maybe namespace or the
#   profile's compute.partition/cpus/mem. Stop if it lists bucket, swarm,
#   models.worker-pi.model, a type's tools, ramp or loop, or "prompts changed".
flock update runs-flock/v8/experiment

# 4. Resume: the holds and the release are lifted, workers start and each agent restores from
#    the bucket by name; the ramps keep their stored schedule.
flock launch runs-flock/v8/experiment
flock status v8                  # the update's generation, 234 agents, workers starting
flock status v8 --watch          # world up, provider errors, crash events
flock status v8 --events         # unreleased, resumed, launch; no launch_refused or crashloop
```

To stop again: `flock stop v8 --release` (agents synced, a final controller checkpoint, the
world's SIGTERM push, then both jobs end).

## Gaps

- **Comparisons are on the honour system.** The world doesn't check that a PR or a review
  carries a comparison, or that its verdicts are honest.
- **The deck is the swarm's, so it can drift.** The quality lane owns it; operators should look
  at it during the run.
- **The arbiter starts as plain priority order.** Until the routing lane improves it,
  overlapping modules still compete by PRIORITY.
- **The budget exemption can be gamed.** New logic can be packed into an existing module's file
  to dodge `open_budget`.
- **Two reviewers per change can stall.** Operators can raise `review.max_reviewers` live
  (docs/world.md, "Admin API").
- **The world's archive lags.** A world killed without SIGTERM loses up to 5 min of PR and
  review state (merged code is on the Hub); a crash restarts from its `CHATBOT_WS`, which is
  newer than the archive.
