# aops-math

Agents solve one AoPS competition problem, alone or as a team sharing a message board (one `#general` channel). Each
agent works in its own container and writes its final answer to `/app/answer.txt`: one number, or a set, tuple,
interval or expression. A hidden grader checks it exactly; for structured answers it rejects, an LLM judge decides.
A team's answer is the majority vote of its agents. We compare a team of N with N agents working alone (same vote) and
with one agent given the team's total tokens. The tasks, every setup and the details are in the dataset card of
`HuggingFaceSwarms/aops-math-bench` (private); `spec.yaml` here is its `setups/team_4.yaml`.

**Needs**

- podman or docker (`environment.backend` in the spec). On the HF cluster: a CPU node with
  `scripts/slurm/podman-node.sh` (see docs/open-models.md, "On the HF cluster").
- An OpenAI-compatible server for `MODEL` with tool calling on; sampling is set in the server deployment. The dataset
  card has the vLLM command we used (Qwen3.5: presence penalty 0, higher values break tool calls).
- `HF_TOKEN` with read access to the dataset. Git must be able to clone it, e.g. `hf auth login --add-to-git-credential`.

**Cost**: the agents cost nothing beyond your own server; `score.py --judge` costs about $0.001 per structured answer
the exact grader rejects. One task with a team of 4 takes about 5-15 min.

**Launch**

```sh
LEGION=$PWD
hf download HuggingFaceSwarms/aops-math-bench --repo-type dataset --include "setups/*" --local-dir aops-math-bench
cd aops-math-bench/setups
export MODEL=<served model> SERVER_URL=http://<host>:<port>/v1 TASK_ID=1e0f5b785c687e40 TASK_REV=<dataset commit>
uv run --project $LEGION python pacer.py --rounds 1 --allowance 32768 &
uv run --project $LEGION python -m legion.controller.localrun team_4.yaml --until idle --grace 150
OPENAI_API_KEY=... uv run --project $LEGION python score.py --judge
```

`pacer.py` ends the run once every agent has been idle for 45 s, with one final prompt; the run and its bundle are
under `.legion/`. `score.py --judge` prints each agent's grade and the team's vote. Other setups (one agent, teams of
2-16, one agent with a team's tokens) are in the card.
