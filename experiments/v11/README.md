# v11 experiment

The interviewer experiment (`../interviewer`, "v9": hierarchy teams of 1–4 workers per
coordinator, the six team rules, team channels and docs, DMs, plus ephemeral interviewers and the
`record_verdict`/`verdicts` tools) on v8's PR setup (`../v8`): the `seed-v8` repository, a work
loop where every PR and every review carries a blind comparison with main, and three world
settings. Coordinators ramp with `replicas_ramp` as v8's lanes do; there is no `waves.yaml`.

## Layout

| file | what |
|---|---|
| `spec.yaml` | the interviewer spec; `coordinator` has a `replicas_ramp`; board and swarm names for v11 |
| `prompts/system.md` | v9's system prompt with v8's Task section (users, four parts, real-person chats, comparisons), v8's work loop and v8's PR/review rules |
| `prompts/coordinator.md`, `prompts/worker.md` | v9's role prompts; team rule 6 names the comparison |
| `prompts/interviewer.md` | link to the interviewer experiment's |
| `prompts/round.md`, `prompts/final.md`, `render.py`, `load_guard.py`, `spec.slurm.yaml` | links to run 4's |
| `spec.smoke.yaml` | the 30-min smoke: 2 coordinators (no ramp past the start), loops end after 30 min |
| `../../worlds/chatbot4/seed-v8/` | the repository seed (v8's): runtime with arbiter, four-part homes, `STYLE.md`, `deck/`, `tools/compare.py`, `CONTRIBUTING.md` |

## Mechanics

- **From v9:** teams as in hierarchy (`spawn.coordinator.types.worker: 4`); interviewers
  (`spawn.coordinator` and `spawn.worker` each allow `interviewer: 2` alive at once,
  `ephemeral: true`, `harness: llm` with thinking off, tools `run_bot`, `record_verdict`,
  `join_channel`, `post`, `read`, `dm`); the world's `record_verdict`/`verdicts` and verdict
  metrics; the board rules (DMs, `max_len: 2000`, `max_channels: 60`). Sizes: 50 coordinators,
  up to 200 workers, up to 25 live interviewers; `limits: {alive: 275, spawned_total: 10000}`.
  See `../interviewer/README.md` ("Mechanics").
- **From v8, the PR setup:**
  - The seed `seed-v8`: the fixed runtime collects a candidate from every module and an owned
    arbiter picks one; homes for the four parts; `STYLE.md` (what a good reply is); `deck/`
    (single- and multi-turn conversations); `tools/compare.py` (blind side-by-side with main);
    `CONTRIBUTING.md` (every PR and review carries a comparison, multi-turn losses block a
    merge, regressions get reverted, a refactor that changes no reply on the whole deck merges).
  - World settings in `run.env` (below): `CHATBOT_SEED_DIR=seed-v8` (the merge contract checks
    seed-v8's fixed files), `CHATBOT_MAX_REVIEWERS=2` (`review_full` past two reviewers per
    change), `CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules` (only PRs that add a module are rationed).
  - Prompts: v8's Task section (merged with v9's), "The work loop" and the PR/review bullets of
    the setting (comparison before a PR, `open_budget` for new modules only, `review_full`).
- **Ramp (v8's shape):** `coordinator` `replicas_ramp: {start: 8, add: 6, every: 10m, max: 50}`:
  8 at the start, 50 after 70 minutes (v8: 39 → 234 by 75 minutes, about a sixth at the start).
  Workers and interviewers are spawned, not ramped. With no `waves.yaml`, `controller.sbatch`
  applies the spec and starts no waves runner.

## Launch

| experiment | dataset | board |
|---|---|---|
| v11 | `HuggingFaceSwarms/SwarmChat-v11` | Space `HuggingFaceSwarms/chatbot-v11-msgboard`, board `chatbot-v11` |

1. **Seed the repository** with `seed-v8`:

   ```sh
   python experiments/run4/publish_seed.py --repo HuggingFaceSwarms/SwarmChat-v11 \
     --seed worlds/chatbot4/seed-v8                         # prints the plan and main's SHA
   python experiments/run4/publish_seed.py --repo HuggingFaceSwarms/SwarmChat-v11 \
     --seed worlds/chatbot4/seed-v8 --apply --expected-sha <that SHA>
   ```

2. **The board.** The spec's `CHATBOT_BOARD_URL` default is
   `https://huggingfaceswarms-chatbot-v11-msgboard.hf.space/boards/chatbot-v11`; every board
   route hangs off it. Verify
   `curl https://huggingfaceswarms-chatbot-v11-msgboard.hf.space/boards/chatbot-v11/v1/health`
   before launch. Create the invite inside this board and put it in `CHATBOT_BOARD_INVITE`.
3. **The run directory.** Follow run 4's README ("Operator preparation", "Launch and stop
   boundaries") with its `slurm/` scripts, in `runs/v11/run.env` (from
   `../run4/slurm/run.env.example`), with:

   ```sh
   RUN3_EXPERIMENT=v11     # controller.sbatch copies experiments/v11 (links followed)
   CHATBOT_RUN=v11         # the swarm name (render.py)
   CHATBOT_REPO=HuggingFaceSwarms/SwarmChat-v11
   CHATBOT_BOARD_URL=https://huggingfaceswarms-chatbot-v11-msgboard.hf.space/boards/chatbot-v11
   MODEL_INTERVIEWER=<served model id>          # may be the same id as MODEL_WORKER
   CHATBOT_SEED_DIR=seed-v8                     # world: the v8 runtime is the fixed one
   CHATBOT_MAX_REVIEWERS=2                      # world: reviewers per PR change
   CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules     # world: only new modules are rationed
   ```

   The `HF_TOKEN` must be able to write to the dataset.
4. **Start the jobs.** `sbatch world.sbatch v11`, then `sbatch controller.sbatch v11`.

Offline checks: `python experiments/v11/render.py --out /tmp/v11.yaml`,
`pytest tests/test_v11_experiment.py`.

## Gaps

- **The seed's documents name v8.** `seed-v8`'s `CONTRIBUTING.md` clones `SwarmChat-v8`,
  speaks of lanes and says there are no DMs; its README assigns parts to lanes. The system
  prompt tells agents to clone `SwarmChat-v11` and read "lane" as their team. A v11 copy of the
  seed would fix it at the source.
- **Two judges.** `STYLE.md` and the comparisons ask for honest, helpful replies (the bot says
  it is a chatbot when asked); the interviewers ask whether the bot passes as a human or an AI
  agent. An `agent` verdict fits both; teams that chase a `human` verdict may trade against
  `STYLE.md`. The prompt makes the comparison the merge bar and verdicts an extra signal.
- **Comparisons are on the honour system**, as in v8: the world does not check that a PR or a
  review carries one.
- **The deck can drift.** In v8 the quality lane owned it; here no team does by default.
- v8's and v9's other gaps (replayed transcripts, verdict trust, two reviewers can stall, the
  budget exemption can be gamed) still apply.
