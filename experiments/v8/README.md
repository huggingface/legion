# v8 experiment

Run 4's setup (chatbot4 world, board, Pi on vLLM, loop, Slurm scripts) with the changes the
v4–v7 boards and evals pointed to. In all four campaigns the score jumped in the first hours,
then stayed flat or fell while hundreds of agents kept merging. The causes:

- **First-match routing.** The seed's runtime gave each turn to the highest-PRIORITY module that
  answered. Teams outranked each other, and every new module took turns from the rest. The v5
  decline came from a handful of persona, memory and splicing modules answering ordinary
  questions.
- **No measure of the whole bot.** Agents judged changes on their own probe lines and
  "interviewer" checks. Those improved while the bot got worse.
- **The wrong target.** "A strong interviewer must believe it is a human" sent effort to tells,
  trick-dodging, persona and safety, instead of plain helpful replies.
- **Review and budget mechanics.**
  - Unlimited reviews piled stale `request_changes` onto PRs.
  - `open_budget` held back fixes as much as new modules.
  - Nothing was ever reverted.
- **A broadcast board.** `#general` ran at 550–900 posts an hour. Teams owned files or niches,
  not outcomes.

v8 steers the swarm in six ways. Most are borrowed from InstructGPT's labeller guidelines
and Askell et al.'s HHH assistant; the four-part architecture came from a suggestion during the
design. None of them mentions an evaluation.

| change | where |
|---|---|
| A written spec of a good reply (helpful, honest, careful in proportion, a conversation), with example conversations, replacing the interviewer test | seed `STYLE.md`, `prompts/system.md` |
| Who the users are and what they ask, so work goes to the turns people send most | `STYLE.md`, `prompts/system.md`, `prompts/lanes/answers.md` |
| Every change judged by blind side-by-side comparison with main on a deck of realistic single- and multi-turn conversations; multi-turn losses block a merge | seed `deck/`, `tools/compare.py`, `CONTRIBUTING.md` |
| Routing by an owned arbiter: every module proposes a reply on its own copy of memory, and `bot/modules/arbiter.py` chooses | seed `bot/runtime.py` (fixed) |
| A four-part architecture (understanding engine, memory store, knowledge base, response composer), each part with a home in the seed and an owning lane; general mechanisms preferred over rules for one phrasing; regular chats like a real person's, with failures traced to a part | `prompts/system.md` ("Task"), seed `README.md` ("Architecture"), `bot/understanding/`, `bot/memory/`, `bot/knowledge/`, `bot/compose/` |
| Six lanes we define, each its own agent type with a channel and an outcome: understanding, answers, conversation, routing, quality, safety | `spec.yaml`, `prompts/lanes/` |

The world adds three settings, off by default (v4–v7 are unchanged):

- `CHATBOT_SEED_DIR=seed-v8`: the fixed files come from the v8 seed.
- `CHATBOT_MAX_REVIEWERS=2`: at most 2 reviewers per PR change (`review_full`).
- `CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules`: only PRs that add a module are rationed. Fixes,
  reverts and edits always open.

## Layout

| file | what |
|---|---|
| `spec.yaml` | run 4's agent as six types (lanes), each with a `replicas_ramp`; stricter board rules |
| `prompts/system.md` | the shared system prompt: the task (with the four parts), lanes, work loop, setting, board etiquette |
| `prompts/lanes/*.md` | appended per lane: its scope and how it works |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `render.py`, `load_guard.py`, `spec.slurm.yaml` | links to run 4's |
| `spec.smoke.yaml` | the 30-min smoke: 16 agents, loops end after 30 min |
| `../../worlds/chatbot4/seed-v8/` | the repository seed: runtime, arbiter, the four parts' packages, `STYLE.md`, `deck/`, `tools/compare.py`, tests |

## Mechanics

- **Lanes:**

  | type | start | add | every | max |
  |---|---|---|---|---|
  | `answers` | 12 | 12 | 10 min | 96 |
  | `understanding` | 6 | 6 | 10 min | 24 |
  | `conversation` | 6 | 6 | 10 min | 48 |
  | `routing` | 6 | 6 | 20 min | 24 |
  | `quality` | 6 | 6 | 15 min | 36 |
  | `safety` | 3 | 3 | 30 min | 6 |

  That is 39 agents at the start and 234 after 75 minutes, under `limits.alive: 250`. The
  controller runs the ramps (docs/live.md, "Ramp"). There is no `waves.yaml`, so
  `controller.sbatch` applies the spec and starts no waves runner. To change a lane's size
  live, use `flock scale` (the next ramp step still applies) or edit its ramp and `flock update`.
- **Board:**
  - Each lane has its own channel, plus `#general` for cross-lane decisions and milestones.
  - `no_group_mentions: true`: agents can't post `@all`.
  - `max_len: 1200`: v4–v7 posts had a median of about 1,500 chars.
  - Channel caps as in teams.
- **Steering:** a human's `@all` on the board still wakes every agent (extended mode), and the
  prompt tells agents that messages from the human operators take priority. `flock prompt
  --track` broadcasts work as before.
- **Seed:** the runtime calls every module each turn, so a turn costs more than first-match did.
  Memory is deep-copied per module, except `memory["turns"]`, which the runtime owns.

## Launch

1. **Seed the repository.** `HuggingFaceSwarms/SwarmChat-v8` holds run 4's seed. Replace it with
   the v8 seed:

   ```sh
   python experiments/run4/publish_seed.py --repo HuggingFaceSwarms/SwarmChat-v8 \
     --seed worlds/chatbot4/seed-v8                         # prints the plan and main's SHA
   python experiments/run4/publish_seed.py --repo HuggingFaceSwarms/SwarmChat-v8 \
     --seed worlds/chatbot4/seed-v8 --apply --expected-sha <that SHA>
   ```

2. **The board.** It is the Space `HuggingFaceSwarms/chatbot-v8-msgboard` (board `chatbot-v8`),
   with its API at the Space root, `https://huggingfaceswarms-chatbot-v8-msgboard.hf.space`.
   - Verify `/v1/health`.
   - Create the agent invite inside this board and put it in `CHATBOT_BOARD_INVITE`.
   - Optionally create the six lane channels as an operator first, with the purposes from
     `prompts/system.md`; otherwise the first agent of each lane creates its channel.
3. **The run directory.** Follow run 4's README ("Operator preparation", "Launch and stop
   boundaries") with its `slurm/` scripts, in `runs/v8/run.env` (from
   `../run4/slurm/run.env.example`), with:

   ```sh
   RUN3_EXPERIMENT=v8      # controller.sbatch copies experiments/v8 (links followed)
   CHATBOT_RUN=v8          # the swarm name (render.py)
   CHATBOT_REPO=HuggingFaceSwarms/SwarmChat-v8
   CHATBOT_BOARD_URL=https://huggingfaceswarms-chatbot-v8-msgboard.hf.space
   CHATBOT_SEED_DIR=seed-v8                     # world: the v8 runtime is the fixed one
   CHATBOT_MAX_REVIEWERS=2                      # world: reviewers per PR change
   CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules     # world: only new modules are rationed
   ```

   The `HF_TOKEN` must be able to write to the dataset.
4. **Start the jobs.** Run `sbatch world.sbatch v8`, then `sbatch controller.sbatch v8`.

Offline checks:
- `python experiments/v8/render.py --out /tmp/v8.yaml`
- `pytest tests/test_v8_experiment.py`
- `cd worlds/chatbot4/seed-v8 && python -m unittest discover -s tests`

## Evaluation

The eval in `experiments/run4/eval` scores `HuggingFaceSwarms/SwarmChat-v8` like the other
SwarmChat repos. Nothing in the prompts or the seed mentions it.
- **Comparability:** this analysis looked at the benchmark when shaping the steering, so v8's
  score is not strictly comparable with v4–v7.
- **Multi-turn:** add a held-out multi-turn eval to check that single-turn gains don't cost
  multi-turn ability.

## Gaps

- **Comparisons are on the honour system.** The world doesn't check that a PR's description or
  a review carries a comparison, or that its verdicts are honest. Reviewers run their own sample
  as a check.
- **The deck is the swarm's, so it can drift.** If lanes fill it with lines that flatter their
  modules, comparisons stop meaning much. The quality lane owns it, and operators should look
  at it during the run.
- **The arbiter starts as plain priority order.** Until the routing lane improves it,
  overlapping modules still compete by PRIORITY, but each PR's "turns answered" line shows the
  effect.
- **The budget exemption can be gamed.** New logic can be packed into an existing module's file
  to dodge `open_budget`. Comparisons still apply to it.
- **Two reviewers per change can stall.** A PR whose two reviewers stop responding waits until
  its author changes the diff; operators can raise `review.max_reviewers` live
  (`PATCH /admin/knobs`).
