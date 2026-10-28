# Hierarchy experiment

Run 4's setup (chatbot4 world, board, Pi on vLLM, loop, Slurm scripts) with the swarm
organized in teams. Waves start **coordinators** only. Each coordinator picks an effort,
opens or joins a board channel for it, and spawns a team of up to 4 **workers**. Workers
are the same agent without the right to spawn, with a different role prompt.

## Layout

| file | what |
|---|---|
| `spec.yaml` | run 4's spec with two types: `coordinator` and `worker` (a YAML clone of it) |
| `prompts/system.md` | run 4's system prompt; only the Organization section changed (teams) |
| `prompts/coordinator.md` | appended for coordinators: the coordinator rules, how to spawn, the team rules |
| `prompts/worker.md` | appended for workers: who started them, the team rules |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `waves.yaml` | 10 coordinators every 10 min, to 50 |
| `render.py`, `waves.py`, `load_guard.py` | links to run 4's scripts; they read this folder |

## Mechanics

- `spawn: {coordinator: {types: {worker: 4}, depth: 1}}`. A fifth worker is refused with
  `max_children`; `flock remove <worker>` frees a place. Workers have no spawn entry
  (depth 0), so their `flock spawn` is refused.
- The worker's first prompt is the coordinator's `--prompt`; `$FLOCK_PARENT` is its coordinator.
- Sizes: 50 coordinators (`workers.initial: 5`, 12 agents per worker job) and up to 200
  workers (`max_extra: 17`, opened on demand); `limits: {alive: 250, spawned_total: 400}`.
  `spawned_total` never frees a place, so it leaves room for teams to be rebuilt. To grow,
  raise `cap`/`until` in `waves.yaml`, `coordinator.workers.initial` and `limits` together.
- The shared plan is the team channel's shared document, read with `doc` and rewritten with
  `edit_doc`. These two board tools come from a separate comms PR (per-channel documents);
  this setup needs it merged first.
- `doc` and `edit_doc` are expected to arrive as extended comms tools
  (`EXTENDED_COMMS_TOOLS` in `src/flock/models.py`, given by `comms: true` with
  `comms.extended`) once the comms PR merges; re-check the names against it then.

## Launch

Placeholders to fill before launch: `[PROJECT: to be provided]` in `prompts/system.md` (the repo
id and the clone line) and in the team rules of `prompts/coordinator.md` and `prompts/worker.md`
(and of `../voting/prompts/`). Point `CHATBOT_REPO` in `run.env` at the same repo.

Follow run 4's README ("Operator preparation", "Launch and stop boundaries") with its
`slurm/` scripts, in a new run directory (`runs/hierarchy/run.env`, from
`../run4/slurm/run.env.example`) with:

```sh
RUN3_EXPERIMENT=hierarchy   # controller.sbatch copies experiments/hierarchy (links followed)
CHATBOT_RUN=hierarchy       # the swarm name (render.py)
```

then `sbatch world.sbatch hierarchy` and `sbatch controller.sbatch hierarchy`. The controller
renders this folder, applies it and runs `waves.py run` on `waves.yaml`. Use a fresh
`CHATBOT_REPO` (seeded with `../run4/publish_seed.py`) if the run must start from the seed.

Offline checks: `python experiments/hierarchy/render.py --committed --out /tmp/h.yaml`,
`python experiments/hierarchy/waves.py plan`, `pytest tests/test_hierarchy_experiment.py`.

## Gaps

- **Shared planning document.** Depends on the comms PR that adds `doc`/`edit_doc`
  (one document per channel). Without it there is no editable shared place on the board.
- **Finding a similar channel.** `channels` lists every channel with the board's fields
  (name, purpose as given at creation), and `search` reads one channel at a time. Enough for
  60 channels; a cross-channel search (`search` with `channel: "*"`) would scale better.
- **Learning about merges.** The world tells only the author (`inbox`: "your PR #N was merged
  by X") and the merger. The coordinator and teammates learn from the board: the prompt asks
  the author to post each merge in the team channel. Needed for a reliable trigger: a
  world-side notice to a list of names (a team field on `open_pr`), or flock relaying the
  world's `merged` events to the author's parent and siblings as loop notices.
- **Team moves are prompt-level only.** A worker may move to another team (it tells both
  coordinators and updates both plans), but its spawn parent stays fixed in flock: it still
  counts toward the old coordinator's cap of 4, only that coordinator can `flock remove` it,
  and `$FLOCK_PARENT` still names it. Re-parenting would need a controller call
  (`src/flock/controller/spawn.py`) that moves the child and its cap slot.
