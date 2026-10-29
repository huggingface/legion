# SwarmChat evaluation

Each chatbot swarm builds a rule-based chatbot in a dataset repo of the
[Chatbot Project](https://huggingface.co/collections/HuggingFaceSwarms/chatbot-project) collection,
e.g. [`HuggingFaceSwarms/SwarmChat-v5`](https://huggingface.co/datasets/HuggingFaceSwarms/SwarmChat-v5).
These scripts score any commit of one, as a model called SwarmChat, and any Hub or API model, on
[SmolTalkBench](https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench) (private): 1,000
single-turn everyday chat prompts with reference replies.

| File | What it does |
|---|---|
| [`smoltalk_bench.py`](smoltalk_bench.py) | the Inspect task `smoltalk_bench_gold` |
| [`judge.py`](judge.py) | AlpacaEval 2.0's judge prompt, order swap, preference and win rates |
| [`swarmchat_server.py`](swarmchat_server.py) | serves a bot checkout as an OpenAI-compatible endpoint |
| [`evaluate.py`](evaluate.py), [`hf_job.py`](hf_job.py) | launch the eval on Hugging Face Jobs, adapted from Carbon-X |
| [`sync.py`](sync.py) | evaluates every new `main` commit of the SwarmChat repos, as a scheduled Job |

The script that builds the dataset lives in the dataset repo, as `scripts/create_dataset.py`.

## Scoring

The scoring follows AlpacaEval 2.0's default annotator (`weighted_alpaca_eval_gpt4_turbo`).

- **The judge.** For each prompt, the judge sees the model's reply and the reference, named
  `m` and `M`. Which one comes first is fixed per prompt, as in AlpacaEval. The judge is
  `deepseek-ai/DeepSeek-V4.1-Flash` on Inference Providers (fireworks-ai), called with
  thinking off.
- **A sample's score.** This is the judge's probability that the model's reply is better,
  read from its top-5 logprobs.
- **The metrics:**
  - `win_rate`, with its `standard_error`;
  - `discrete_win_rate`;
  - `length_controlled_win_rate`, the primary metric. It uses AlpacaEval's
    `length_controlled_minimal` GLM, because the default GLM needs instruction difficulties
    fitted on AlpacaEval's own prompts. Its `lc_standard_error` is a bootstrap.

All numbers are percentages against the gold references, which are Llama-3.1-70B-Instruct's
replies in smoltalk. `tests/test_smoltalk_bench.py` checks the port against values computed
with the [`alpaca_eval`](https://github.com/tatsu-lab/alpaca_eval) package.

SmolLM and SmolLM2 Instruct were trained on the source conversations, so their scores are
inflated.

## Run on Hugging Face Jobs

`evaluate.py` launches one Job, which runs one eval. Run it from the repository root, after
logging in with `hf auth login`:

```sh
# SwarmChat at the head of HuggingFaceSwarms/SwarmChat-v4, on a cpu-upgrade Job with one
# conversation per vCPU
uv run experiments/run4/eval/evaluate.py --model HuggingFaceSwarms/SwarmChat-v4

# Another swarm's chatbot, at an earlier commit
uv run experiments/run4/eval/evaluate.py --model HuggingFaceSwarms/SwarmChat-v5 --revision 62dbbb8

# A Hub model, served with vLLM on an rtx-pro-6000 Job
uv run experiments/run4/eval/evaluate.py --model HuggingFaceTB/SmolLM-135M-Instruct

# An API model, called from a CPU Job
uv run experiments/run4/eval/evaluate.py --backend endpoint \
  --model hf-inference-providers/Qwen/Qwen3-14B:nscale \
  --chat_template_kwargs '{"enable_thinking": false}'
```

- **Smoke tests and dry runs.** `--limit 20` runs a quick smoke test, and `--dry_run`
  prints the Jobs without submitting them.
- **Hub models.** Sampling defaults to the model's `generation_config.json`, and to pure
  sampling (temperature 1, top-p 1, no top-k) for whatever it leaves out. `--temperature`,
  `--top_p` and `--top_k` override it.
- **Where Jobs run.** They run under the token's user, or under `--namespace`. The files
  they need are uploaded to the `jobs-artifacts` bucket of the output bucket's namespace.
- **SwarmChat.** A model ID with `SwarmChat` in it is a swarm's chatbot, served with
  `swarmchat_server.py`; `--backend swarmchat` does the same for a repo without it. Like a Hub
  model, the repo is the model ID and a commit is a revision. Each revision is named by its
  commit time and hash, e.g. `20261028T214140Z-62dbbb8f50a4`, so revisions sort in merge
  order.

Logs go to the private Storage Bucket `HuggingFaceSwarms/swarmchat-evals`, or to the bucket
passed with `--bucket`:

```
<model>/<revision>/smoltalk_bench_gold/<timestamp>/
├── <timestamp>_smoltalk-bench-gold_<id>.eval    # Inspect log
├── summary.json                                 # primary metric and its standard error,
│                                                #   all metrics, sample counts
└── swarmchat_server.log                         # SwarmChat only, if the bot wrote to stderr
```

```sh
inspect view --log-dir hf://buckets/HuggingFaceSwarms/swarmchat-evals/HuggingFaceSwarms/SwarmChat-v4
```

## Keeping SwarmChat evals up to date

`sync.py` launches an eval for every `main` commit of every `HuggingFaceSwarms/SwarmChat-v<N>`
dataset that has none yet, one Job per commit, so many commits of one repo run side by side.

- **What it skips.** Commits with a `summary.json` in the bucket, those with an eval in flight,
  and those whose evals already failed twice (`--max_attempts`).
- **Unchanged bots.** A commit that leaves `bot/` as its parent left it would reply as its
  parent does. It gets `<revision>/skipped.json` in the bucket, naming that parent, instead of
  an eval.
- **Order.** Newest first, round-robin across repos, so new merges are scored within one run
  while a backlog drains.
- **Rate limit.** It stops at 50 evals in flight across all models (`--max_running`), which kept
  the judge free of 429s.
- **No state.** A run keeps no state of its own, so running it by hand is also a backfill.

```sh
# What a run would launch and skip
uv run experiments/run4/eval/sync.py --dry_run

# Run every 10 minutes as a scheduled cpu-basic Job, under the token's user
uv run experiments/run4/eval/sync.py --schedule "*/10 * * * *"
```

The scheduled Job and its evals run the eval code as it was when the schedule was created; rerun
`--schedule` after changing it, which replaces the old schedule. `hf jobs scheduled ps`,
`hf jobs scheduled suspend <id>` and `hf jobs scheduled delete <id>` list, pause and remove it.

## SwarmChat

`swarmchat_server.py` loads the bot once and forks a worker per conversation. Every
conversation therefore starts from the same just-loaded state, and the hash seed and all
random seeds are fixed. A commit gives the same replies whatever the order, the concurrency
or the run. The exceptions are replies that read the clock or seed from an object id.

The bot runs without a sandbox, with an empty environment. HF Jobs do not allow the namespaces
of `flock.isolation`, and the Job's container is disposable. Locally:

```sh
python experiments/run4/eval/swarmchat_server.py <bot checkout> --port 8000 &
SWARMCHAT_BASE_URL=http://127.0.0.1:8000/v1 SWARMCHAT_API_KEY=unused HF_TOKEN=$(hf auth token) \
  inspect eval experiments/run4/eval/smoltalk_bench.py@smoltalk_bench_gold \
  --model openai-api/swarmchat/SwarmChat --max-connections 4
```

The bot reads only the user turns. A multi-turn conversation is replayed from the start on
every request, which gives the same replies, since a conversation is deterministic.

## Development

The Inspect and AlpacaEval code needs the `evals` dependency group. Without it, those tests are
skipped:

```sh
uv sync --all-extras --group evals
pytest tests/test_smoltalk_bench.py
```
