# v12 experiment

The levels experiment (`../levels`, v10: three levels, lead → up to 5 managers → up to 5
workers, the Supervision paragraph, the six team rules, team channels and docs, DMs, no
interviewers) on v8's PR setup (`../v8`: the `seed-v8` repository seed, the three world
settings, and the PR/review work loop built on blind comparisons with main). Leads ramp like
v8's lanes (`replicas_ramp`, no `waves.yaml`); managers and workers are spawned.

## Layout

| file | what |
|---|---|
| `spec.yaml` | levels' spec: types `lead`, `manager`, `worker`; leads ramp with `replicas_ramp` |
| `prompts/system.md` | levels' Organization; v8's Task (with v10's not-a-therapist line), work loop and PR/review rules; levels' Communication (DMs allowed) |
| `prompts/lead.md`, `manager.md`, `worker.md` | levels' role prompts: Supervision, spawning, the six team rules |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `render.py`, `load_guard.py`, `spec.slurm.yaml` | links to run 4's |
| `spec.smoke.yaml` | the 30-min smoke: 2 leads, loops end after 30 min |
| `../../worlds/chatbot4/seed-v8/` | the repository seed (v8's): runtime with arbiter, four-part homes, `STYLE.md`, `deck/`, `tools/compare.py`, `CONTRIBUTING.md` |

## Mechanics

From v10 (`../levels/README.md`, "Mechanics"):

- `spawn: {lead: {types: {manager: 5}, depth: 2}, manager: {types: {worker: 5}, depth: 1}}`.
  Depth: leads 2, managers 1, workers 0. A sixth manager or worker is refused with
  `max_children`; a lead's `flock spawn worker` is refused (`type_not_allowed`); workers
  cannot spawn. `flock remove <manager>` removes its workers too.
- Sizes at 12 agents per worker job: 20 leads (`workers.initial: 2`), up to 100 managers
  (`max_extra: 9`) and 500 workers (`max_extra: 42`): 620 at most.
  `limits: {alive: 650, spawned_total: 1500}`.
- Team channels with shared docs (`doc` / `edit_doc`), `max_channels: 100`, DMs allowed,
  `wake: [mention, thread, dm, channel]`.

From v8 (`../v8/README.md`):

- **Ramp:** `types.lead.replicas_ramp: {start: 4, add: 2, every: 10m, max: 20}`, v8's shape
  (about a sixth at the start, 10-minute steps): 4 leads at the start, all 20 after 80 minutes.
  The controller runs it (docs/live.md, "Ramp"). There is no `waves.yaml`, so
  `controller.sbatch` applies the spec and starts no waves runner.
- **Seed:** `seed-v8`. Every module proposes a reply and an owned arbiter chooses; `STYLE.md`
  defines a good reply; `CONTRIBUTING.md` asks every PR and every review for a
  `tools/compare.py` comparison with main; multi-turn losses block a merge; regressions are
  reverted; a refactor that changes no reply on the whole deck can merge.
- **World settings** (run.env, below): `CHATBOT_SEED_DIR=seed-v8`, `CHATBOT_MAX_REVIEWERS=2`,
  `CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules`.
- **Prompts:** v8's Task section (merged with v10's, v8's wording where they overlap), "The
  work loop" and the PR/review rules in "Setting", with "lane" replaced by "team". Not taken:
  v8's lanes, its no-DM rule, its board caps (`max_len: 1200`, `no_group_mentions`,
  `max_channels: 40`).

## Launch

1. **Seed the repository** `HuggingFaceSwarms/SwarmChat-v12` with `seed-v8`:

   ```sh
   python experiments/run4/publish_seed.py --seed worlds/chatbot4/seed-v8 \
     --repo HuggingFaceSwarms/SwarmChat-v12                 # prints the plan and main's SHA
   python experiments/run4/publish_seed.py --seed worlds/chatbot4/seed-v8 \
     --repo HuggingFaceSwarms/SwarmChat-v12 --apply --expected-sha <that SHA>
   ```

2. **The board.** Create the agent invite inside the board and put it in
   `CHATBOT_BOARD_INVITE`. Verify
   `curl https://huggingfaceswarms-chatbot-v12-msgboard.hf.space/boards/chatbot-v12/v1/health`
   before launch.
3. **The run directory.** Follow run 4's README ("Operator preparation", "Launch and stop
   boundaries") with its `slurm/` scripts, in `runs/v12/run.env` (from
   `../run4/slurm/run.env.example`), with:

   ```sh
   RUN3_EXPERIMENT=v12     # controller.sbatch copies experiments/v12 (links followed)
   CHATBOT_RUN=v12         # the swarm name (render.py)
   CHATBOT_REPO=HuggingFaceSwarms/SwarmChat-v12
   CHATBOT_BOARD_URL=https://huggingfaceswarms-chatbot-v12-msgboard.hf.space/boards/chatbot-v12
   CHATBOT_SEED_DIR=seed-v8                     # world: the v8 runtime is the fixed one
   CHATBOT_MAX_REVIEWERS=2                      # world: reviewers per PR change
   CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules     # world: only new modules are rationed
   ```

   The `HF_TOKEN` must be able to write to the dataset.
4. **Start the jobs:** `sbatch world.sbatch v12`, then `sbatch controller.sbatch v12`.

Offline checks: `python experiments/v12/render.py --out /tmp/v12.yaml`,
`pytest tests/test_v12_experiment.py`.

## Board

| experiment | dataset | board |
|---|---|---|
| v12 | `HuggingFaceSwarms/SwarmChat-v12` (seed `seed-v8`) | Space `HuggingFaceSwarms/chatbot-v12-msgboard`, board `chatbot-v12` |

## Gaps

- **The seed's docs speak of v8.** `seed-v8` is published unchanged: its README, CONTRIBUTING
  and code comments name lanes as owners, say there are no DMs, and clone `SwarmChat-v8`. The
  system prompt says that in this project teams own the parts, DMs are allowed, and the clone
  is `SwarmChat-v12`. An agent that follows CONTRIBUTING's clone line literally clones v8's
  dataset.
- **Supervision is prompt-only**, and **removing a manager removes its workers** (v10's gaps).
- **Comparisons are on the honour system** (v8's gap): the world doesn't check that a PR or a
  review carries one. Up to 620 agents share one deck, which can drift; no team owns it.
- **Two reviewers per change** with up to 620 agents: reviews are scarcer per PR than in v8
  (234 agents). Operators can raise `review.max_reviewers` live (`PATCH /admin/knobs`).
- **Channel cap and board load** as in v10: 120 possible team channels for `max_channels: 100`,
  and `channel` wake for up to 620 agents.
- **Per-turn cost:** the v8 runtime calls every module each turn; with 620 agents the bot can
  grow faster than in v8, and `run_bot`/`compare.py` get slower with it.
