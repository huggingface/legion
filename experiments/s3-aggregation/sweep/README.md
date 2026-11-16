# s3-aggregation scaling sweep

What the s³ paper's Figure 5 (information aggregation: success against swarm size) looks like
with flock: every problem of `HuggingFaceSwarms/s3-aggregation` at N = 1, 2, 4, 8 and 16 agents,
each team its own flock run on the HF cluster.

| File | |
| --- | --- |
| `serve-vllm.sbatch` | the model server: vLLM with `Qwen/Qwen3.5-27B` on one B200 GPU (docs/open-models.md) |
| `sweep.py` | runs the grid, one command, resumable: a row per team in `<root>/rows.jsonl` |
| `plot.py` | `scaling.png` (Figure 5's style), `teams.csv`, `by_n.csv` and `by_n.md` from the rows |

## Run it

```sh
sbatch -p hopper-b200 -o .flock/s3-sweep/vllm-%j.log experiments/s3-aggregation/sweep/serve-vllm.sbatch
# its log ends with: serving Qwen/Qwen3.5-27B at http://<node address>:8000/v1
uv run experiments/s3-aggregation/sweep/sweep.py --urls http://<node address>:8000/v1 \
  --parallel 8 --dataset HuggingFaceSwarms/s3-aggregation-results --prefix grid
scancel -n s3-sweep-vllm
```

Each team is team.py's cluster spec (`--where cluster`) with the model on the server's URL:
`compute: slurm`, one worker on hopper-cpu running the team in rootless podman with a local
board, sized to the team (a CPU per agent, at least 4). The board rules, loop and limits are
the template's at every N. sweep.py inits and launches a team, waits until its loop is done
and its agents are idle, releases it (`flock stop --release`: each Trial is graded as the
worker drains, and the results land in the run directory), and appends metrics.py's row with
the wall time. A team that has a row is skipped, so the same command resumes. The run state goes
to a private bucket, the run directories (spec, results, logs; never `run.env`) to the private
dataset, and plot.py runs at the end.

## How it differs from the paper's Figure 5

| | s³ (Figure 5, top row) | here |
| --- | --- | --- |
| Agents | Qwen3-0.6B trained with RL at N = 4, and its controls | `Qwen/Qwen3.5-27B`, not trained: one prompted condition |
| Harness | their asynchronous harness: `message(to, text)` and `read_messages()`, notifications that interrupt generation, idle until a message arrives | Pi in a container per agent, with flock's board tools (`flock-tool message_board.*`) through its shell; a shared `#general` channel, mentions and a long-poll `wait`; flock's loop prompts again at each turn's end |
| Messages | to one teammate or broadcast | posts on a board every agent reads; private channels possible |
| Names | sampled from `agent_1`..`agent_64`, teammates in shuffled order | `agent-001`..`agent-00N`, in order |
| Budget | 8,192 work tokens + 1,024 for the answer per agent; reward reduced by tokens used | no token allowance: 60 model calls per prompt, 4 loop rounds, 15 min, 10 min per turn, 262,144-token window |
| Answer | `####` and a JSON array, from any agent active at the end | `/app/answer.json` per agent, graded after the run; the team succeeds if any agent's is right |
| Instruction | Appendix G's, with an example message and the token-cost rule | adapted from Appendix G for the board, with no example message and no token cost |
| Problems | 32 held-out problems x 6 seeds; N = 1, 2, 4, 6, 8, 12, 16 | 32 problems (our seed, the same recipe: 16 numbers in 1..99, the first N) x 1 attempt; N = 1, 2, 4, 8, 16 |
| Intervals | mean ± SE across six seeds | 95% Wilson intervals across problems |
| Right panel | token efficiency (needs the allowance) | generated (output) tokens per team, as the paper's Figure 10 |
