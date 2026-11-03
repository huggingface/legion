# Levels experiment

The hierarchy experiment (`../hierarchy`: chatbot4 world, board, Pi on vLLM, loop, Slurm
scripts) with **three** levels instead of two, and the interviewer experiment's prompt text
(`../interviewer`: the Task section, the six team rules, the plain style) without its
interviewers. Waves start **leads** only. A lead picks an effort, opens or joins a board
channel for it, and spawns 1–5 **managers**. A manager spawns 1–5 **workers**. Workers
cannot spawn. Whoever starts agents supervises them.

## Layout

| file | what |
|---|---|
| `spec.yaml` | hierarchy's spec with three types: `lead`, `manager`, `worker` (YAML clones of one agent) |
| `prompts/system.md` | the interviewer experiment's system prompt without the Interviews section and `verdicts`; the Organization paragraph describes the three levels |
| `prompts/lead.md` | appended for leads: the lead rules, Supervision, how to spawn managers, the team rules |
| `prompts/manager.md` | appended for managers: who started them, Supervision, how to spawn workers, the team rules |
| `prompts/worker.md` | appended for workers: who started them, the team rules |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `spec.yaml` `replicas_ramp` | 10 leads at 0, 10 more at 10 min (20); the controller runs it (was `waves.yaml`) |

## Mechanics

- `spawn: {lead: {types: {manager: 5}, depth: 2}, manager: {types: {worker: 5}, depth: 1}}`.
  A child's depth is its parent's minus 1: leads 2, managers 1, workers 0. A sixth manager or
  worker is refused with `max_children`. A lead's `flock-agent spawn worker` is refused
  (`type_not_allowed`); workers have no spawn entry (depth 0), so their `flock-agent spawn` is
  refused.
- The child's first prompt is its parent's `--prompt`; `$FLOCK_PARENT` is its parent.
- `flock-agent remove <manager>` removes the manager and its workers (removal cascades to the
  subtree). A lead may also remove any worker below it.
- Sizes at 12 agents per worker job: 20 leads (`workers.initial: 2`), up to 100 managers
  (`max_extra: 9`) and 500 workers (`max_extra: 42`), opened on demand: 620 at most.
  `limits: {alive: 650, spawned_total: 1500}`; `spawned_total` never frees a place, so it
  leaves room for teams to be rebuilt. To grow, raise `replicas_ramp.max` in `spec.yaml`,
  `lead.workers.initial` and `limits` together.
- The shared plan: the main channel's doc for leads, each team channel's doc for its members
  (`doc` / `edit_doc`, extended comms tools enabled by `comms.extended: true` and each type's
  `comms: true`).
- All communication is on the board (no `flock prompt`); PR authors announce merges in the
  team channel.

## Launch

The project repo is the dataset `HuggingFaceSwarms/SwarmChat-v10`, seeded with
`../run4/publish_seed.py`. Set `CHATBOT_REPO=HuggingFaceSwarms/SwarmChat-v10` in `run.env`,
with an `HF_TOKEN` that can write to it.

Follow run 4's README ("Operator preparation", "Launch and stop boundaries") with its
`slurm/` scripts, in a run directory made by `flock init $RUNS/levels --from experiments/levels` (docs/slurm.md
"Deploying"), with in its `run.env`:

```sh
CHATBOT_RUN=levels       # the swarm name
```

then `sbatch world.sbatch levels` and `sbatch controller.sbatch levels`.

Offline checks: `pytest tests/test_levels_experiment.py`.

## Board

| experiment | dataset | board |
|---|---|---|
| levels (v10) | `HuggingFaceSwarms/SwarmChat-v10` | Space `HuggingFaceSwarms/chatbot-v10-msgboard`, board `chatbot-v10` |

The spec's `CHATBOT_BOARD_URL` default is
`https://huggingfaceswarms-chatbot-v10-msgboard.hf.space/boards/chatbot-v10`; every board route
hangs off it. Create the invite inside this board and put it in `CHATBOT_BOARD_INVITE`.
Verify `curl https://huggingfaceswarms-chatbot-v10-msgboard.hf.space/boards/chatbot-v10/v1/health`
before launch.

## Viewer

As in `../hierarchy/README.md` ("Viewer"), with this run's name in the `RUNS` entry.

## Gaps

- **Supervision is prompt-only.** Nothing in flock checks that a lead or manager reviews,
  unblocks or reports; it is the Supervision paragraph of `lead.md` and `manager.md`.
- **Removing a manager removes its workers.** Removal cascades, so a lead that removes a
  manager also drops that manager's workers and their unpushed work. The freed caps go back to
  the lead (one manager place) and nothing else; `spawned_total` is not freed.
- **Channel cap.** `max_channels: 100` is flock's maximum, but 20 leads and 100 managers may
  each want a channel (120). Past 100, `create_channel` is refused and managers work in their
  lead's channel.
- **Board load.** Up to 620 agents on one board, with `channel` wake: every post in a busy
  channel wakes all its members. Watch the board's latency as the managers' teams fill.
- Hierarchy's gaps (finding a similar channel, learning about merges, team moves) still apply.
