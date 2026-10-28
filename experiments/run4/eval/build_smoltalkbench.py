# /// script
# requires-python = ">=3.11"
# dependencies = ["datasets>=3.0", "huggingface_hub>=0.30"]
# ///
"""Build SmolTalkBench: single-turn chat prompts with reference replies, AlpacaEval style.

    uv run experiments/run4/eval/build_smoltalkbench.py            # build and print stats
    uv run experiments/run4/eval/build_smoltalkbench.py --push     # also push (private)

Source: the everyday-conversations subset of HuggingFaceTB/smoltalk (train + test), written
whole by Llama-3.1-70B-Instruct. Every conversation opens with a bare greeting and the same
"How can I help you today?", so the prompt is the second user turn and the reference is the
assistant's reply to it. Prompts are deduplicated, then sampled in proportion to their parent
topic (full_topic without its leaf), largest remainders rounding the allocation to exactly N.
Columns follow AlpacaEval's reference outputs (instruction, output, generator, dataset).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from datasets import Dataset, load_dataset
from huggingface_hub import DatasetCard, HfApi

SOURCE = "HuggingFaceTB/smoltalk"
SUBSET = "everyday-conversations"
REVISION = "5feaf2fd3ffca7c237fc38d1861bc30365d48ffa"
GENERATOR = "meta-llama/Meta-Llama-3.1-70B-Instruct"
REPO = "HuggingFaceSwarms/SmolTalkBench"
META = {"license": "apache-2.0", "language": ["en"], "pretty_name": "SmolTalkBench",
        "tags": ["chatbot", "evaluation", "alpaca-eval"]}


def candidates():
    """One row per distinct second user turn, first occurrence kept (train, then test)."""
    ds = load_dataset(SOURCE, SUBSET, revision=REVISION)
    rows, seen = [], set()
    for split in ("train", "test"):
        for index, record in enumerate(ds[split]):
            msgs = record["messages"]
            assert [m["role"] for m in msgs[:4]] == ["user", "assistant", "user", "assistant"]
            prompt = msgs[2]["content"].strip()
            key = " ".join(prompt.casefold().split())
            if key in seen:
                continue
            seen.add(key)
            topic = record["full_topic"]
            rows.append({"id": f"{split}-{index:04d}", "instruction": prompt,
                         "output": msgs[3]["content"].strip(), "generator": GENERATOR,
                         "dataset": f"{SOURCE}/{SUBSET}", "topic": topic,
                         "category": topic.split("/")[0],
                         # Capitalised three-level topics are everyday requests; lowercase
                         # two-level ones are grade-school science and general knowledge.
                         "group": "everyday" if topic.count("/") == 2 else "school",
                         "source_split": split, "source_index": index})
    return rows


def stratified(rows, n, seed):
    """Proportional allocation over parent topics, rounded by largest remainder."""
    strata = defaultdict(list)
    for row in rows:
        strata[row["topic"].rsplit("/", 1)[0]].append(row)
    names = sorted(strata)
    quotas = {s: n * len(strata[s]) / len(rows) for s in names}
    alloc = {s: int(q) for s, q in quotas.items()}
    for s in sorted(names, key=lambda s: (alloc[s] - quotas[s], s))[:n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = [row for s in names for row in rng.sample(strata[s], alloc[s])]
    return sorted(picked, key=lambda row: (row["source_split"] != "train", row["source_index"]))


def card(n, seed, total):
    return f"""# SmolTalkBench

{n} single-turn everyday chat prompts with reference replies, for AlpacaEval-style pairwise
win rates. Built to compare the rule-based chatbot from the run4 agent swarm
(`cmpatino/chatbot-swarm`) at different commits, and small LMs, against one fixed reference.

## Construction

- Source: the `{SUBSET}` subset of [`{SOURCE}`](https://huggingface.co/datasets/{SOURCE})
  at revision `{REVISION[:12]}`, train and test splits. Each conversation was written whole by
  [`{GENERATOR}`](https://huggingface.co/{GENERATOR}).
- Every source conversation opens with a bare greeting ("Hi", "Hello", ...) answered by
  "Hello! How can I help you today?". The greeting is dropped: `instruction` is the second
  user turn and `output` the assistant's reply to it.
- Prompts are deduplicated (case and whitespace insensitive), leaving {total} rows.
- {n} rows are sampled with seed {seed}, in proportion to each parent topic (`topic` without
  its last level), so every subcategory keeps its share.

## Columns

| column | description |
|---|---|
| `id` | `<source split>-<source index>` |
| `instruction` | the prompt (second user turn) |
| `output` | reference reply by `generator` |
| `generator`, `dataset` | reference model and source, as in AlpacaEval's reference outputs |
| `topic` | the source `full_topic` |
| `category` | first level of `topic` |
| `group` | `everyday` (three-level topics) or `school` (two-level grade-school knowledge topics) |
| `source_split`, `source_index` | the row in the source subset |

Note that SmolLM and SmolLM2 Instruct models were trained on the source conversations.

Built by `experiments/run4/eval/build_smoltalkbench.py` in the flock repository.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repo", default=REPO)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--public", action="store_true", help="push as a public dataset")
    args = parser.parse_args()
    rows = candidates()
    picked = stratified(rows, args.n, args.seed)
    assert len(picked) == args.n and len({r["id"] for r in picked}) == args.n
    print(f"{len(rows)} distinct prompts -> {len(picked)} sampled")
    print("groups:", dict(Counter(r["group"] for r in picked)))
    print("categories:", dict(Counter(r["category"] for r in picked).most_common()))
    if not args.push:
        return
    Dataset.from_list(picked).push_to_hub(args.repo, split="test", private=not args.public)
    # push_to_hub wrote the card's dataset_info and configs; keep them, add ours.
    pushed = DatasetCard.load(args.repo, repo_type="dataset")
    for key, value in META.items():
        pushed.data[key] = value
    pushed.text = card(args.n, args.seed, len(rows))
    pushed.push_to_hub(args.repo, repo_type="dataset")
    print(f"pushed https://huggingface.co/datasets/{args.repo} @ "
          f"{HfApi().dataset_info(args.repo).sha}")


if __name__ == "__main__":
    main()
