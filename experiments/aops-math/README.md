# aops-math

Four agents solve one AoPS competition problem as a team sharing a message board (one `#general` channel). Each agent
works in its own container and writes its final answer to `/app/answer.txt`: one number, or a set, tuple, interval or
expression. A hidden grader in the container checks it exactly. The tasks are in
`HuggingFaceSwarms/aops-math-bench` (private): 200 problems with a numeric answer and 200 with a structured one.

The folder has the spec and the prompt; the board tools come with the dataset:

- `spec.yaml`: the team, the board rules (posts up to 2000 characters, at least 5 s apart, 40 per agent) and the loop.
  Each agent is prompted once and woken by its teammates' posts. At `until: 30m` the loop stops, and every agent gets
  a final prompt to leave just its answer in `/app/answer.txt`. The run then ends by itself.
- `team.md`: how to work as a team and use the board.
- the board tools (`board.post`, `board.read`, `board.wait`), from the dataset's `setups/tools/` (downloaded below).
  Agents write a post to a file and then post it, so math, quotes and `$` never pass through the shell; reads return
  only unseen posts, one line each.

Each agent may write at most 32,768 output tokens (`loop.while.agent_usd` with `pricing: [0, 1, 0]`, i.e. $1 per 1M
output tokens; the prices are not real). The task revision pinned in the spec blocks the agents' network, except
the model server.

**Needs**

- podman or docker (`environment.backend` in the spec). On the HF cluster: a CPU node with
  `scripts/slurm/podman-node.sh` (see docs/open-models.md, "On the HF cluster").
- An OpenAI-compatible server for `MODEL` with tool calling on; sampling is set in the server deployment. The dataset
  card has the vLLM command we used (Qwen3.5: presence penalty 0, higher values break tool calls).
- `HF_TOKEN` with read access to the dataset. Git must be able to clone it, e.g. `hf auth login --add-to-git-credential`.

**Cost**: nothing beyond your own model server. One task takes about 30 min (the loop's `until`).

**Launch**

From the repository root:

```sh
hf download HuggingFaceSwarms/aops-math-bench --repo-type dataset --revision 3f6f3d5f1f03ea2fc92b303c91036085a957aadb \
  --include "setups/tools/*" --local-dir aops-math-bench
export MODEL=<served model> SERVER_URL=http://<host>:<port>/v1 TASK_ID=1e0f5b785c687e40
uv run legion launch experiments/aops-math/spec.yaml
```

`uv run legion status` shows progress. The run bundle, with each agent's answer and grade, is written under
`.legion/` when the run ends. Scoring the team (majority vote over the agents' answers, plus an optional LLM judge for
structured answers the exact grader rejects), other team sizes, a single agent and one agent given a team's tokens are
in the dataset card (`setups/`).
