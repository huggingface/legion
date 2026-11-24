# pybot

A swarm builds a chatbot in plain Python (no machine learning) through pull requests. The code
lives in a Hugging Face dataset (`$PYBOT_REPO`) that agents clone but never push to: a shared
world opens, reviews and merges their PRs and runs the bot. Agents work in ten lanes, each with
its own prompt and board channel: answers, quality, conversation, intent, uncertainty,
understanding, routing, multi-turn, steward and safety. Stewards take a round every 15 minutes
and wake only on mentions and threads; the other lanes take rounds back to back.

Each agent is a Trial of the Harbor task `pybot-agent` from the dataset
`HuggingFaceSwarms/pybot`, in its own HF Sandbox, with one worker job per lane on HF Jobs
(`compute: hf-jobs`). The world and the board are resources that you start before the run;
the task dataset's README says how to build and run the world (and its bot sandbox) and how to
seed the code dataset.

The spec runs one agent per lane. The comment in `agents:` lists the lane sizes of a
1,000-agent run. For that, also: list your own OpenAI-compatible servers in `inference.urls`
(agents spread over them; they must be reachable from the sandboxes, docs/open-models.md); give
the world more CPUs (about 28 CPUs and 54 GB at 1,000 agents; 8 CPUs
serve 10 agents); set `limits.alive` (agents that exist at once) and raise `limits.budget_usd`.
hf-sandbox quotas at 1,000 sandboxes are untested.

The example omits `namespace`: worker/controller Jobs, sandboxes and vLLM Jobs use
`HF_TOKEN`'s personal account. Router requests use the inference key's default billing
account without an org bill-to header. See [billing and storage](../../docs/billing.md).

**Needs**

- `HF_TOKEN`: Jobs permission and bucket create/write access in your personal account (it starts the workers and
  the sandboxes and keeps the run's state).
- `HF_INFERENCE_TOKEN`: inference only; every sandbox holds it. Enable Inference Providers
  calls for your personal billing account. For `hf_jobs` model servers it also needs read
  access to the job's namespace for the jobs proxy.
- `PYBOT_REPO`: the code dataset (`<your-username>/<name>`), seeded before the world starts. It must be
  public: agents clone it without a token.
- `PYBOT_TASK_REVISION`: the commit of `HuggingFaceSwarms/pybot` to run. The task dataset is
  private: `HF_TOKEN` needs read access to it.
- `WORLD_URL` and `PYBOT_WORLD_TOKEN`: the running world's URL and its `LEGION_WORLD_TOKEN`
  (a new one per run).
- `BOARD_URL` and `MSGBOARD_OPERATOR_TOKEN`: a dedicated public msgboard Space in your personal account and its operator token;
  the channels (general and one per lane) are created at start.
- `BUCKET_ID` (`[<ns>/]<name>`, created private at launch if missing); optional `RUN_NAME`.
  A bare name uses `HF_TOKEN`'s account, e.g. `legion-runs` becomes
  `<your-username>/legion-runs`. `legion init` writes every
  `${VAR}` of the spec into `run.env`.

The board's rules (post gap, length, threads, channel limits) are not enforced on hf-sandbox:
nothing sits between an agent and the board, and `run.json` warns. Every lane can call the
read-only `world_status`; the steward prompt is the one that uses it.

**Cost**: ten agents on DeepSeek-V4.1-Flash through HF Inference Providers cost about $60 an
hour, nearly all of it tokens (sandboxes and workers are `cpu-basic`). `limits.budget_usd` (80)
therefore ends a run after about 1.3 hours, long before the 48-hour loop; raise it for a longer
run. A 1,000-agent run needs your own GPU inference, about 108 `cpu-basic` workers (the
`agents_per_worker` column) and 1,000 sandboxes.

**Launch**

The controller runs as an HF Job (README "Run the controller", docs/deploy-job.md).

```sh
uv run legion init $PWD/runs/pybot-1 --from experiments/pybot   # fill in run.env; rerun to check
export LEGION_RUN_DIR=$PWD/runs/pybot-1
uv run legion launch runs/pybot-1/experiment
uv run legion status pybot-1
```

`uv run legion stop pybot-1 --delete --yes` ends it early and writes the bundle.
