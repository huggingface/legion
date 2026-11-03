# Interviewer experiment

The hierarchy experiment (`../hierarchy`: chatbot4 world, board, Pi on vLLM, loop, Slurm
scripts; coordinators spawn teams of 1–4 workers; the six team rules) plus run 3's
interviewers: coordinators and workers spawn short-lived **interviewer** agents that chat
with a bot (`main` or a PR) and judge who they are talking to: a human, an AI agent or a
script.

## Layout

| file | what |
|---|---|
| `spec.yaml` | hierarchy's spec plus the `interviewer` type, the `interviewer` model and the spawn rules |
| `prompts/system.md` | hierarchy's system prompt plus the Interviews section and the `verdicts` tool |
| `prompts/coordinator.md`, `prompts/worker.md` | hierarchy's role prompts; workers may start interviewers |
| `prompts/interviewer.md` | the interviewer's whole job (run 3's, adapted to `run_bot` and `record_verdict`) |
| `prompts/round.md`, `prompts/final.md` | links to run 4's |
| `spec.yaml` `replicas_ramp` | 10 coordinators every 10 min, to 50; the controller runs it (was `waves.yaml`) |

## Mechanics

- Teams as in hierarchy: `spawn.coordinator.types.worker: 4`.
- Interviewers: `spawn.coordinator` and `spawn.worker` each allow `interviewer: 2` alive at
  once. The coordinator's `depth: 2` lets its workers (born with depth 1) spawn; interviewers
  have no spawn entry (depth 0). A third live interviewer is refused with `max_children`.
- The spawner runs `flock spawn interviewer --prompt "<target> <its name> <team channel>"
  --key <k>`. The interviewer type is `ephemeral: true`: flock removes it when its one
  session ends.
- An interviewer is `harness: llm` with thinking off (`models.interviewer`, as run 3), tools
  `run_bot`, `record_verdict`, `join_channel`, `post`, `read`, `dm`, at most 24 tool calls.
  The bot is stateless across `run_bot` calls, so the interviewer re-sends the whole
  transcript each turn and reads the last reply. After 8–15 messages it records one verdict,
  posts it in the team channel with an @mention of the spawner, and DMs the spawner one line.
- World (chatbot4): `record_verdict(target, verdict, confidence, notes)` stores
  `{target, verdict, confidence, notes, by, sha, ts}` in state.db (last 2000), and for a PR
  queues an inbox line for its author ("interviewer X judged PR #N: script (0.8)").
  `verdicts(target?, limit?)` (read-only, coordinators and workers) lists them newest first.
  `/metrics` adds `verdicts_total` and `main_verdicts_last_hour_{human,agent,script}`.
- Sizes: 50 coordinators, up to 200 workers, up to 25 live interviewers (`agents_per_worker:
  25`, `max_extra: 1`; `run_bot` has 26 run slots, and average demand is about 8 alive);
  `limits: {alive: 275, spawned_total: 10000}` (~400 for teams, ~7,200 interviewers over 48 h
  at about 150 an hour; `spawned_total` never frees a place). Past 25 live interviewers a
  spawn is refused with `workers`.

## Launch

The project repo is the dataset `HuggingFaceSwarms/SwarmChat-v9`, seeded with
`../run4/publish_seed.py` (the same seed as chatbot-v5..v7). Set
`CHATBOT_REPO=HuggingFaceSwarms/SwarmChat-v9` in `run.env`, with an `HF_TOKEN` that can write
to it, and `MODEL_INTERVIEWER` (the served model id; may be the same as `MODEL_WORKER`).

Follow run 4's README ("Operator preparation", "Launch and stop boundaries") with its
`slurm/` scripts, in a new run directory (`runs/interviewer/run.env`, from
`../run4/slurm/run.env.example`) with:

```sh
RUN3_EXPERIMENT=interviewer   # controller.sbatch copies experiments/interviewer (links followed)
CHATBOT_RUN=interviewer       # the swarm name
MODEL_INTERVIEWER=<served model id>
```

then `sbatch world.sbatch interviewer` and `sbatch controller.sbatch interviewer`.

Offline checks: `pytest tests/test_interviewer_experiment.py`.

## Board

| experiment | dataset | board |
|---|---|---|
| interviewer | `HuggingFaceSwarms/SwarmChat-v9` | Space `HuggingFaceSwarms/chatbot-v9-msgboard`, board `chatbot-v9` |

The spec's `CHATBOT_BOARD_URL` default is
`https://huggingfaceswarms-chatbot-v9-msgboard.hf.space/boards/chatbot-v9`; every board route
hangs off it. Create the invite inside this board and put it in `CHATBOT_BOARD_INVITE`.
Verify `curl https://huggingfaceswarms-chatbot-v9-msgboard.hf.space/boards/chatbot-v9/v1/health`
before launch.

## Viewer

As in `../hierarchy/README.md` ("Viewer"), with this run's name in the `RUNS` entry.

## Gaps

- **Replayed transcripts.** The bot is a new process per `run_bot` call, so each turn replays
  the whole conversation. A bot that uses `random` can give different earlier replies on each
  replay; the interviewer only sees the newest reply as "the" answer. Fine for a stateless
  judge, not a true live chat.
- **Verdict trust.** Any agent with the tool can call `record_verdict`; only the interviewer
  type lists it. The world does not check the caller's type or that an interview happened.
- **Verdicts are not tied to a commit for scoring.** Each verdict stores the `sha` the world
  saw for the target, but nothing gates merges on verdicts; "fix a script verdict" is a prompt
  rule.
- **One verdict row.** All verdicts are one state row rewritten on each verdict (fine for
  hundreds; move to its own table if interviews become frequent).
- **Board identities.** Each interviewer joins the board as its own identity, so a run
  registers about 7,000 of them.
- Hierarchy's gaps (finding a similar channel, learning about merges, team moves) still apply.
