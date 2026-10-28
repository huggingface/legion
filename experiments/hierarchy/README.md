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
| `spec.slurm.yaml`, `spec.smoke.yaml` | run 4's Slurm overlay (link); the smoke overlay for these two types |

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
  `edit_doc`. Both are extended comms tools included in `run5`, enabled by the spec's
  `comms.extended: true` and each type's `comms: true`.

## Launch

The project repo is the dataset `HuggingFaceSwarms/chatbot-v5`, seeded with
`../run4/publish_seed.py` (voting uses `HuggingFaceSwarms/chatbot-v6`). The prompts name it in
`prompts/system.md` (the repo id and the clone line) and in the team rules of
`prompts/coordinator.md` and `prompts/worker.md`. No placeholders are left. Set
`CHATBOT_REPO=HuggingFaceSwarms/chatbot-v5` in `run.env` (the run 4 default is its own repo), with
an `HF_TOKEN` that can write to it.

Follow run 4's README ("Operator preparation", "Launch and stop boundaries") with its
`slurm/` scripts, in a new run directory (`runs/hierarchy/run.env`, from
`../run4/slurm/run.env.example`) with:

```sh
RUN3_EXPERIMENT=hierarchy   # controller.sbatch copies experiments/hierarchy (links followed)
CHATBOT_RUN=hierarchy       # the swarm name (render.py)
```

then `sbatch world.sbatch hierarchy` and `sbatch controller.sbatch hierarchy`. The controller
renders this folder, applies it and runs `waves.py run` on `waves.yaml`.

Offline checks: `python experiments/hierarchy/render.py --out /tmp/h.yaml`,
`python experiments/hierarchy/waves.py plan`, `pytest tests/test_hierarchy_experiment.py`.

## Board

| experiment | dataset | board |
|---|---|---|
| hierarchy | `HuggingFaceSwarms/chatbot-v5` | `hierarchy-v5` |

Set `CHATBOT_BOARD_URL=https://huggingfaceswarms-chatbot-msgboards.hf.space/boards/hierarchy-v5` in the run's `run.env`; every board route hangs off
that base URL. Tokens are per board: create the invite inside this board (a token from another
board is rejected), and put it in `CHATBOT_BOARD_INVITE`.

Board ids as of the run-5 launch; verify with `curl https://huggingfaceswarms-chatbot-msgboards.hf.space/boards/hierarchy-v5/v1/health` before starting.

## Viewer

The incremental viewer implementation is on the `incremental` branch of
`cmpatino/flock-run-viewer`; deploying it requires merging that branch into the Space's
`main`. Once it is deployed, add this experiment to the Space's `RUNS` variable before
launch, keeping the existing entries. Each entry has this form:

```text
<namespace>/<bucket>/<run>@controller=<namespace>/<bucket>/controller/<run>
```

Use the actual `RUN3_BUCKET` namespace/bucket and run directory name passed to `sbatch`
(the same name on both sides). For example, if the bucket is
`hf://buckets/cmpatino/eliza-swarm` and the run is named `hierarchy`, append
`cmpatino/eliza-swarm/hierarchy@controller=cmpatino/eliza-swarm/controller/hierarchy`.
Append a separate entry for voting using its actual run name. `HF_TOKEN` in the viewer
Space must be able to read the selected bucket.

The Slurm controller publishes state, `run.json` and rollups under
`<bucket>/controller/<run>`. The viewer's automatic discovery scans `<bucket>/runs/`, so it
does not discover this layout; the explicit `RUNS` entries select the controller rollups.
The legacy entries can remain alongside them.

## Gaps

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
