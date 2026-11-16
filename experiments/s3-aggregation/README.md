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

## Run one team on the HF cluster

`--where cluster` writes the spec for one Slurm worker running the team in rootless podman
(hopper-cpu has podman 4.9), with a local board on that worker, new each run, so no Space. The
run directory goes on the shared disk (docs/slurm.md "Deploying"): a cluster profile with
`SLURM_PARTITION=hopper-cpu`, `WORKER_CPUS` and `WORKER_MEM` (at most 2000 MB per CPU there)
and `VENV`, and a mode-600 secrets file with `HF_TOKEN`, `HF_INFERENCE_TOKEN`, `NAMESPACE` and
`BUCKET` (a private bucket for the run's state):

```sh
uv run experiments/s3-aggregation/team.py --problem 7 --agents 4 --where cluster --out runs/src/agg-p07-n04
uv run flock init $PWD/runs/agg-p07-n04 --from runs/src/agg-p07-n04 --profile cluster.env --secrets secrets.env
export FLOCK_RUN_DIR=$PWD/runs/agg-p07-n04
uv run flock launch $FLOCK_RUN_DIR/experiment
uv run flock status agg-p07-n04                  # until the loop is done and the agents idle
uv run flock stop agg-p07-n04 --release          # drain: each Trial is graded, then the jobs end
uv run experiments/s3/metrics.py $FLOCK_RUN_DIR
```

A Slurm run doesn't end by itself: `--release` drains its worker, which is when each Trial is
graded. The worker copies the grading and the board export to `<run dir>/results/` (its own
files are in the node's `/tmp`), and `metrics.py` reads them from there.

## The board rules

The same for every eval run, since they shape how agents talk (`spec.template.yaml`):
one channel, `#general`; an agent is woken by a mention or a reply in its thread; a post is at
most 1000 characters, at least 5 s after the agent's previous one, and at most 40 per agent.
podman enforces them in each agent's rule proxy. HF Sandboxes have no proxy, so there they are
not enforced and each Trial records a `message_board.warning` event.

## Metrics

`uv run experiments/s3/metrics.py RUN_DIR [RUN_DIR ...] [--out rows.jsonl]` writes one JSON line
per run directory (one team): a local run's `.flock/<swarm>-<time>/`, or a Slurm run's directory.
A Slurm run has no local bundle, so there the tokens come from Harbor's Trial results (no
reasoning split) and `stop_reason`, `cost_usd` and the turns are `null`:

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
