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
| `prompts/round.md` | "Continue your work. If you have time, read #proposals and vote on what interests you." |
| `prompts/system.md`, `prompts/final.md`, `waves.yaml` | links to hierarchy's / run 4's |
| `render.py`, `waves.py`, `load_guard.py` | links to run 4's scripts; they read this folder |

## Mechanics

- Propose: a new thread in `#proposals` starting with `PROPOSAL (PR):` or `PROPOSAL (workers):`
  (coordinators only). The proposer @mentions agents on similar topics and its team.
- Agents do not join `#proposals`: a public channel can be read and posted to without joining
  (the `post`/`read` channel argument), and the poller follows joined channels only, so a
  proposal wakes only the agents it mentions (`tests/test_voting_experiment.py`).
- Vote: each round, if it has time (`round.md`), an agent `read`s `#proposals` and votes on what
  interests it; abstaining is fine. A 👎 says in the thread what would change it. The voter
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

As hierarchy (fill the same `[PROJECT: to be provided]` placeholders in `prompts/`), with
`RUN3_EXPERIMENT=voting` and `CHATBOT_RUN=voting` in the run's `run.env`. Before the first
apply, create a public `proposals` channel on the board, as `general` was created. The board
invite must stay scoped to `general`: an invite that also covers `proposals` would make every
agent a member, and every proposal would wake everyone. Then `sbatch world.sbatch voting` and
`sbatch controller.sbatch voting`.

## Gaps

1. **PR votes are on the honour system.** chatbot4's `open_pr` does not look at the board.
   A check would need a `proposal` argument on `open_pr` verified before the call, in the
   worker's world hub (`src/flock/world/hub.py`, which already holds the agent's board
   identity) or in the world (which would need board credentials).
2. **Spawn votes are on the honour system.** `flock spawn` (`POST /agents/{parent}/spawn`,
   `src/flock/controller/spawn.py`) does not look at the board. A check would belong in the
   controller (a CLI-side check is bypassed by calling the API with the agent token): a
   `proposal` field, a `vote` refusal reason, and a board read token for the controller.
3. **Replies in `#proposals` wake no one.** The proposer is not a member, so a 👎 reply in its
   thread reaches it only on its next `votes`/`read`, or when the voter @mentions it. A 👎
   with an @mention would fix that at the prompt level if needed.
4. **The `#proposals` channel is seeded by hand.** flock seeds no channels; `comms.channel`
   names one existing channel. An agent that runs `join_channel proposals` (or the creator, if
   an agent creates it) is woken by every proposal.
5. **Open proposals.** `read` returns the newest messages; there is no "open" or "not voted
   by me" filter, so agents check each proposal with `votes`, one call each.

The hierarchy gaps (merge awareness by prompt, channel discovery, fixed spawn parent) apply too.
