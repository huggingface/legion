# s3-aggregation

N agents each hold a private number from 1 to 99, written in English words, and the team must
list all N numbers in ascending order. No agent can solve it alone, so it measures how well a
swarm communicates over the board. It adapts the information-aggregation task of
[s³](https://arxiv.org/abs/2611.08615) ([scalingswarms.com](https://scalingswarms.com)); the
tasks, the role files, the grader and the checks are in the dataset
[`HuggingFaceSwarms/s3-aggregation`](https://huggingface.co/datasets/HuggingFaceSwarms/s3-aggregation)
(private for now; its README covers the task and its known limits).

| File | |
| --- | --- |
| `spec.template.yaml` | what every team's run shares: model, the pinned dataset, the board rules, the loop and limits, and where it runs (`where:`) |
| `team.py` | writes the spec for one team: the task `agg-pPP-nNN`, N agents and their role files |
| `../s3/metrics.py` | reads run directories and writes one row per team (below) |

Each agent's number reaches only that agent: `agents.role_files` gives `agent-00k` the dataset's
`roles/<task>/slot-0k.md` ("Your private number: seventy-three"). The agents stay one flock type
on one worker (`agents_per_worker: N`), so they share that worker's board: roles would each run
on their own workers.

**Needs** (as experiments/mini-swarm): `HF_TOKEN`, which reads the private dataset, starts the
sandboxes and deploys the board's Space, and `HF_INFERENCE_TOKEN`, a fine-grained token with
only "Make calls to Inference Providers", which goes into the sandboxes. Never the same token.

## Run one team on HF Sandboxes

The controller and the team's one worker run on this machine, each agent in its own HF Sandbox
(`cpu-basic`) and the board on a Space:

```sh
uv run experiments/s3-aggregation/team.py --problem 7 --agents 4   # out/agg-p07-n04/
BOARD_SPACE=<you>/s3-agg-p07-n04 uv run flock launch experiments/s3-aggregation/out/agg-p07-n04/spec.yaml
uv run flock status agg-p07-n04
uv run experiments/s3/metrics.py .flock/agg-p07-n04-<time>/       # once the run has ended
hf repos delete <you>/s3-agg-p07-n04 --repo-type space -y          # a new board for every team
hf buckets delete <you>/s3-agg-p07-n04-bucket                      # its snapshots
```

The run ends by itself once its loop is over (4 rounds or 15 minutes) and every agent is idle,
then Harbor grades each agent's `/app/answer.json` and the board is exported into the run
directory. The board's Space must be public while the run uses it (the sandboxes hold no Hub
token), and the instruction has no run tag, so give each team a new Space and delete it, and its
snapshot bucket, once the run directory has the export.

`--where cluster` writes the HF cluster's spec: one Slurm worker running the team in rootless
podman, with a local board on that worker (new each run). Set it up with `flock init` as in
docs/slurm.md. It compiles but has not run yet.

## The board rules

The same for every eval run, since they shape how agents talk (`spec.template.yaml`):
one channel, `#general`; an agent is woken by a mention or a reply in its thread; a post is at
most 1000 characters, at least 5 s after the agent's previous one, and at most 40 per agent.
podman enforces them in each agent's rule proxy. HF Sandboxes have no proxy, so there they are
not enforced and each Trial records a `message_board.warning` event.

## Metrics

`uv run experiments/s3/metrics.py RUN_DIR [RUN_DIR ...] [--out rows.jsonl]` writes one JSON line
per run directory (one team):

| field | |
| --- | --- |
| `task`, `agents`, `revision`, `stop_reason`, `duration_s` | the team and how its run ended |
| `success` | any agent's reward is 1 (`null`: no rewards found) |
| `rewards` | each agent's reward: its latest Trial's verifier result |
| `tokens`, `cost_usd` | the team's tokens (input, output, cache, reasoning) and cost |
| `posts`, `other_posts` | the team's board posts in the run; posts by anyone else |
| `per_agent` | each agent's reward, tokens, posts and turns |
| `edges` | the message graph: sender -> each teammate it mentions (a group mention names every member) or messages directly (a DM, or a private channel of the two) |
| `centralization` | the paper's (Appendix M): the sum over agents of (k* - k_j) / (N - 1)^2, k_j the in-degree; 0 when every agent receives from as many teammates, 1 when one receives from all and no one else from anyone; `null` for N = 1 |
| `reciprocity` | the fraction of edges whose reverse exists; `null` without edges |
| `missing` | what the run directory lacks: an agent's reward, the board export |

Only messages posted between the run's start and end count. A post that mentions no one adds no
edge: on a board everyone reads every post, so a broadcast edge (the paper's) would connect
every pair that posts. s3-maze adds its move-log metrics to the same rows (`EXTRAS`).
