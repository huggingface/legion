# SwarmChat evaluation

The run4 swarm builds a rule-based chatbot in
[`cmpatino/chatbot-swarm`](https://huggingface.co/datasets/cmpatino/chatbot-swarm). These scripts
score any commit of it, as the model `SwarmChat`, and any Hub or API model, on
[SmolTalkBench](https://huggingface.co/datasets/HuggingFaceSwarms/SmolTalkBench) (private): 1,000
single-turn everyday chat prompts with reference replies.

| File | What it does |
|---|---|
| [`smoltalk_bench.py`](smoltalk_bench.py) | the Inspect task `smoltalk_bench_gold` |
| [`judge.py`](judge.py) | AlpacaEval 2.0's judge prompt, order swap, preference and win rates |
| [`swarmchat_server.py`](swarmchat_server.py) | serves a bot checkout as an OpenAI-compatible endpoint |
| [`evaluate.py`](evaluate.py), [`hf_job.py`](hf_job.py) | launch the eval on Hugging Face Jobs, adapted from Carbon-X |

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

`evaluate.py` launches one Job per model revision. Run it from the repository root, after
logging in with `hf auth login`:

```sh
# SwarmChat at two commits: one CPU Job each
uv run experiments/run4/eval/evaluate.py --model SwarmChat --revision main fdef14e

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
- **Where Jobs run.** They run under the token's user, or under `--namespace`.
- **SwarmChat revisions.** Each revision is named by its commit time and hash, e.g.
  `20261028T162718Z-fdef14e9f8e3`, so revisions sort in merge order.

Logs go to the private Storage Bucket `HuggingFaceSwarms/swarmchat-evals`, or to the bucket
passed with `--bucket`:

```
<model>/<revision>/smoltalk_bench_gold/<timestamp>/
├── <timestamp>_smoltalk-bench-gold_<id>.eval    # Inspect log
├── summary.json                                 # primary metric, all metrics, sample counts
└── swarmchat_server.log                         # SwarmChat only, if the bot wrote to stderr
```

```sh
inspect view --log-dir hf://buckets/HuggingFaceSwarms/swarmchat-evals/SwarmChat
```

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
