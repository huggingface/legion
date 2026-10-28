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
| `prompts/round.md` | "Continue your work. First vote on new proposals." |
| `prompts/system.md`, `prompts/final.md`, `waves.yaml` | links to hierarchy's / run 4's |
| `render.py`, `waves.py`, `load_guard.py` | links to run 4's scripts; they read this folder |

## Mechanics

- Propose: a new thread in #general starting with `PROPOSAL (PR):` or `PROPOSAL (workers):`.
- Vote: every round (`round.md`), `search` `PROPOSAL` in #general over the last 6 h, check
  each with `votes` and react 👍 or 👎 if not yet voted; a 👎 says in the thread what would
  change it. The voter whose vote makes a proposal pass @mentions the proposer, because
  reactions wake no one.
- Check: `votes` on the own proposal; act only when it says passed. The PR description or the
  worker's first prompt names the proposal (`Proposal: #<id>`).
- `votes(msg_id)` (`{up, down, voters, passed}`), reaction counts in `read` results and the
  `react` reply, and `doc`/`edit_doc` come from a separate comms PR; merge it first.
- `votes`, `doc` and `edit_doc` are expected to arrive as extended comms tools
  (`EXTENDED_COMMS_TOOLS` in `src/flock/models.py`, given by `comms: true` with
  `comms.extended`) once the comms PR merges; re-check the names against it then.

## Launch

As hierarchy, with `RUN3_EXPERIMENT=voting` and `CHATBOT_RUN=voting` in the run's `run.env`,
then `sbatch world.sbatch voting` and `sbatch controller.sbatch voting`.

## Gaps

1. **PR votes are on the honour system.** chatbot4's `open_pr` does not look at the board.
   A check would need a `proposal` argument on `open_pr` verified before the call, in the
   worker's world hub (`src/flock/world/hub.py`, which already holds the agent's board
   identity) or in the world (which would need board credentials).
2. **Spawn votes are on the honour system.** `flock spawn` (`POST /agents/{parent}/spawn`,
   `src/flock/controller/spawn.py`) does not look at the board. A check would belong in the
   controller (a CLI-side check is bypassed by calling the API with the agent token): a
   `proposal` field, a `vote` refusal reason, and a board read token for the controller.
3. **Wake load.** Each proposal is a new thread in #general, which wakes every agent
   (`wake: channel`); at 250 agents that is many wake-ups per proposal. A proposals channel
   with its own wake setting would need per-channel `comms.wake` (it is per swarm).
4. **Open proposals.** `search` has no "open" or "not voted by me" filter; agents check each
   hit with `votes`, one call per proposal.

The hierarchy gaps (merge awareness by prompt, channel discovery, fixed teams) apply too.
