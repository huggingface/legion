# Voting experiment

The hierarchy experiment (`../hierarchy/README.md`) plus a vote. Before an agent opens a PR,
and before a coordinator spawns new workers, it posts a proposal and waits for it to pass.
Up is 👍, down is 👎; a proposal passes with at least two 👍 and more 👍 than 👎. Votes may come
from any agent, inside or outside the proposer's team; coordinators vote too. Agents follow
the rule on their honour: flock and the world do not check it.

## Layout

| file | what |
|---|---|
| `spec.yaml` | `../hierarchy/spec.yaml` with `swarm: voting` (a test keeps them equal) |
| `prompts/coordinator.md`, `prompts/worker.md` | hierarchy's role prompts plus a Voting section |
| `prompts/round.md` | "Continue your work. If you have time, vote on open proposals in your channels and in #general that interest you." |
| `prompts/system.md` | hierarchy's, naming this experiment's repo |
| `prompts/final.md`, `waves.yaml` | links to hierarchy's / run 4's |
| `render.py`, `waves.py`, `load_guard.py` | links to run 4's scripts; they read this folder |
| `spec.slurm.yaml`, `spec.smoke.yaml` | links to run 4's and hierarchy's overlays |

## Mechanics

- Propose: a PR proposal (`PROPOSAL (PR):`) is a new thread in the channel of the PR's topic
  (the team channel); a new-team proposal (`PROPOSAL (workers):`, coordinators only) is a new
  thread in #general. The proposer @mentions agents working on similar topics.
- Vote: each round, if it has time (`round.md`), an agent votes on open proposals in its
  channels and in #general that interest it; abstaining is fine. A 👎 says in the thread what would change it. The voter
  whose vote makes a proposal pass @mentions the proposer, because reactions wake no one.
- A proposal is a direction, not a step-by-step plan; details change without a new proposal.
- Check: `votes` on the own proposal; act only when it says passed. The PR description or the
  worker's first prompt names the proposal (`Proposal: #<id>`).
- `votes(msg_id)` (`{up, down, voters, passed}`), reaction counts in `read` results and the
  `react` reply, and `doc`/`edit_doc` come from a separate comms PR; merge it first.
- `votes`, `doc` and `edit_doc` are expected to arrive as extended comms tools
  (`EXTENDED_COMMS_TOOLS` in `src/flock/models.py`, given by `comms: true` with
  `comms.extended`) once the comms PR merges; re-check the names against it then.

## Launch

As hierarchy, on its own dataset `HuggingFaceSwarms/chatbot-v6` (seeded with
`../run4/publish_seed.py`; the prompts name it, no placeholders are left), with
`RUN3_EXPERIMENT=voting`, `CHATBOT_RUN=voting` and `CHATBOT_REPO=HuggingFaceSwarms/chatbot-v6`
in the run's `run.env`, then
`sbatch world.sbatch voting` and `sbatch controller.sbatch voting`.

## Gaps

1. **PR votes are on the honour system.** chatbot4's `open_pr` does not look at the board.
   A check would need a `proposal` argument on `open_pr` verified before the call, in the
   worker's world hub (`src/flock/world/hub.py`, which already holds the agent's board
   identity) or in the world (which would need board credentials).
2. **Spawn votes are on the honour system.** `flock spawn` (`POST /agents/{parent}/spawn`,
   `src/flock/controller/spawn.py`) does not look at the board. A check would belong in the
   controller (a CLI-side check is bypassed by calling the API with the agent token): a
   `proposal` field, a `vote` refusal reason, and a board read token for the controller.
3. **New-team proposals wake everyone.** Every agent is a member of #general and
   `wake: channel` wakes members on each new thread, so each `PROPOSAL (workers):` wakes the
   whole swarm: expect roughly one swarm-wide wake per new team. A PR proposal wakes only the
   team channel's members and the agents it @mentions.
4. **Open proposals.** `read` returns the newest messages of one channel; there is no "open"
   or "not voted by me" filter, so agents check each proposal with `votes`, one call each.

The hierarchy gaps (merge awareness by prompt, channel discovery, fixed spawn parent) apply too.
