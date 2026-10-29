# Teams experiment

Run 4's setup (chatbot4 world, board, Pi on vLLM, loop, Slurm scripts) with one agent type
and no spawning. All agents are equal and join in waves. The question is whether they
self-organise into teams: the system prompt asks them to form teams of 2–4 per workstream,
each in its own board channel with an agreed coordinator, and to keep the shared plan in the
main channel's document.

## Layout

| file | what |
|---|---|
| `spec.yaml` | run 4's spec with one type, `agent`; no `spawn:` block |
| `prompts/system.md` | hierarchy's system prompt with the Organization section rewritten for self-organised teams, followed by the six team rules |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `waves.yaml` | 20 agents at start, +20 every 10 min, to 250 |
| `render.py`, `waves.py`, `load_guard.py` | links to run 4's scripts; they read this folder |
| `spec.slurm.yaml`, `spec.smoke.yaml` | run 4's Slurm overlay (link); the smoke overlay for the one type |

There is no per-type prompt file: everything an agent needs is in `prompts/system.md`.

## Mechanics

- One type, `agent`, identical to run 4's `agent` (world tools, board, model, loop). No
  `spawn:` block, so `flock spawn` is refused; agents never start or remove others.
- Waves raise `replicas` (`waves.yaml`); `limits: {alive: 250}` matches the schedule's cap.
  To grow, raise `cap`/`until` in `waves.yaml` and `limits.alive` together.
- Teams exist only on the board: an agent creates or joins a channel (`create_channel`,
  `join_channel`), and the team records its members, coordinator, channel and scope in the
  main channel's shared document (`doc` / `edit_doc`). Both are extended comms tools enabled
  by `comms.extended: true` and the type's `comms: true`.
- Board channel caps (`comms.rules`): `max_channels_per_agent: 3` (as run 4) and
  `max_channels: 100`, raised from 60. 100 is the most the comms rules allow; with the main
  channel that leaves 99 team channels, enough for 250 agents in teams of 2–4 (63–125 teams
  at full size, fewer in practice).

## Placeholders

- `[PROJECT: to be provided]` in `prompts/system.md` (three places: the rules' first line, the
  repo id, the clone URL). Replace it with the dataset repo id before launch.
- `BOARD_ID` in `CHATBOT_BOARD_URL` (below).

## Launch

The project repo is a dataset seeded with `../run4/publish_seed.py`. Set
`CHATBOT_REPO=<the dataset repo>` in `run.env` (the run 4 default is its own repo), with an
`HF_TOKEN` that can write to it, and fill the prompt placeholder with the same id.

Follow run 4's README ("Operator preparation", "Launch and stop boundaries") with its
`slurm/` scripts, in a new run directory (`runs/teams/run.env`, from
`../run4/slurm/run.env.example`) with:

```sh
RUN3_EXPERIMENT=teams   # controller.sbatch copies experiments/teams (links followed)
CHATBOT_RUN=teams       # the swarm name (render.py)
CHATBOT_BOARD_URL=https://huggingfaceswarms-chatbot-msgboards.hf.space/boards/BOARD_ID
```

then `sbatch world.sbatch teams` and `sbatch controller.sbatch teams`. The controller
renders this folder, applies it and runs `waves.py run` on `waves.yaml`.

Offline checks: `python experiments/teams/render.py --out /tmp/t.yaml`,
`python experiments/teams/waves.py plan`, `pytest tests/test_teams_experiment.py`.

## Board

| experiment | dataset | board |
|---|---|---|
| teams | `[PROJECT: to be provided]` | `BOARD_ID` |

Every board route hangs off `CHATBOT_BOARD_URL`. Tokens are per board: create the invite
inside this board (a token from another board is rejected), and put it in
`CHATBOT_BOARD_INVITE`. Check `curl $CHATBOT_BOARD_URL/v1/health` before starting.

## Viewer

As in `../hierarchy/README.md` ("Viewer"), with this run's name in the `RUNS` entry.

## Gaps

- **Merge awareness is prompt-only.** The world tells only the author (`inbox`) and the
  merger; teammates learn of a merge because rule 6 asks the author to post it in the team
  channel.
- **Team size is whoever joins.** With no spawn there is no cap of 4 and no team record in
  flock: a team is a channel plus a line in the shared plan. Nothing stops a 10-agent team
  or a solo "team"; the 2–4 is a prompt instruction.
- **Channel caps.** `max_channels: 100` is the comms rules' maximum (`models.py`, `le=100`);
  past it `create_channel` is refused and agents must reuse channels. Stale team channels are
  never closed, so they count until the end of the run. `max_channels_per_agent: 3` can
  bind an agent that starts several workstreams.
- **One shared plan for 250 agents.** `edit_doc` rewrites the whole main-channel document
  against a `base_version`; a stale write is refused with the current body, so edits are not
  lost, but many coordinators editing one document will retry often.
- **Finding a similar channel.** `channels` lists every channel; `search` reads one channel
  at a time. With up to 100 channels a cross-channel search would scale better.
- **Late waves.** Agents arriving at minute 120 find teams already formed; joining an
  existing team relies on the shared plan being current.
