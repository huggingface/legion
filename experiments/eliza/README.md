# Eliza on flock

Five open-model agents (3 `ds`, 2 `glm` on the HF router) write ONE ELIZA-style chatbot together:
a single Python file, standard library only, that reads the user's lines on stdin and writes one
reply line per line. They coordinate on a msgboard channel, talk to each other's candidates with
`run_bot`, and keep the team's best at `best/bot.py` with `promote_best`.

This is the worlds-design §4.7 yardstick: the task is the world in
[`worlds/eliza/`](../../worlds/eliza/eliza.py) (about 300 lines); the loop, rounds, wake-ups,
final round, board and budget are this spec. There is no loop, LLM-client or board code here.

| file | what |
|---|---|
| `spec.yaml` | models, world, comms, the two types with their `loop:`, the budget |
| `prompts/system.md` | the task and the tools (both types) |
| `prompts/round.md`, `prompts/final.md` | the loop prompt (every 90 s and on board wake-ups) and the final round |
| `prompts/kickoff.md` | posted once on the channel by the swarm's host identity |
| `protocols/lead-agree.md`, `protocols/eliza.md` | how the team splits work, reviews and promotes |

The world's tools:
- `read_file(path, start?, end?)`: any file, or a range of its lines; the result says the total
  number of lines;
- `edit_file(path, old, new)` (an exact text that must occur once; refused with the count
  otherwise), `insert_after(path, anchor, text)`, `write_file(path, content)` (new files) and
  `copy_file(src, path)`: only under `agents/<your name>/`. Each write is atomic, and a `.py`
  file that no longer compiles is refused with the error and its line number, keeping the
  previous content. Agents are told to edit, not rewrite: small edits keep model calls short;
- `list_files`, `run_bot(path, lines)`, `promote_best(path, reason)` (only if the candidate
  scores at least as well as the current best on `probes/*.txt`), `best_status`. Agents also get the board tools `post`, `read`, `search` and
`done`; `bash` is off.

## Run it locally

Needs Python 3.11+, `uv sync --all-extras` in the repo, an HF token with router access and a
msgboard channel with a multi-use agent invite.

```sh
cd experiments/eliza
cp spec.yaml spec.local.yaml        # gitignored; set comms.url and comms.channel in it
export HF_TOKEN=<your HF token>
export ELIZA_BOARD_INVITE=<the channel's agent invite>
uv run flock run spec.local.yaml --until idle
```

`flock run` starts the controller, the world (`worlds/eliza`, as a child process) and two worker
processes, posts the kickoff, and stops when both loops and their final rounds are done (2 h at
most, `while.until`), or at `limits.budget_usd`. Ctrl-C runs the final round and stops; twice
stops at once.

From a second terminal in the same directory:

```sh
uv run flock status                 # agents, cost, loops
uv run flock loop eliza             # rounds per agent
uv run flock events ds-0000 --pretty --follow
uv run flock update spec.local.yaml # apply an edited prompt, rule or limit to the running swarm
```

Outputs:
- the run bundle: `.flock/eliza-<UTC ts>/` (`run.json` with the loops and budget summaries,
  `events.jsonl`, `logs/world-eliza.log`);
- the workspace: `worlds/eliza/ws/` (or `$ELIZA_WS`): `agents/<agent>/`, `best/bot.py`,
  `best/history.jsonl` (every promotion with its score), `best/v<n>.py`, `log.jsonl` (every
  write, edit, refusal, syntax error and promotion, with flock's request id and turn, and the
  file's bytes before and after). Copy it next to the
  bundle after the run; delete it to start the next run from scratch.

Talk to the result: `printf 'hello\nI feel sad\n' | python -I worlds/eliza/ws/best/bot.py`.

## Offline dry run

`uv run pytest tests/test_eliza_experiment.py` runs this folder under `flock run` with the fake
model server and the fake board from `tests/`: 5 agents, scripted tool calls, 2 rounds and a
final round, about 20 s, no network, no cost.

## Cost

The caps follow the original run: `while.agent_usd` $3.60 per `ds` agent and $2.10 per `glm`
agent, `limits.budget_usd` $15 (the loops stop at $14.40 to leave the final rounds their
reserve). Expected cost of a 2 h run: _to be filled in by the operator after the first run_.

## Change it

- Probes: `worlds/eliza/probes/*.txt` (one user line per line; ` => kw1|kw2` means the reply must
  contain one of them). They are copied into a fresh workspace; agents can read them.
- Models: `models:` in the spec (any router model with tool calls; add its price to `pricing`).
- The split and the team: the `first:` prompt and `prompts/kickoff.md` name the agents
  (`ds-0000` ... `glm-0001`); change both with `replicas`.
