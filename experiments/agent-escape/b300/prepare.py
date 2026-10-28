"""Freeze real UltraChat prompts and reconstructed AgentEscape contexts; no inference."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from importlib.metadata import version
from pathlib import Path

MODEL = "Qwen/Qwen3.8-27B"
DATASET = "HuggingFaceH4/ultrachat_200k"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def token_count(encoded):
    """Transformers 5 can return BatchEncoding rather than a bare token list."""
    if hasattr(encoded, "keys"):
        encoded = encoded["input_ids"]
    return len(encoded)


def wire_message(message):
    """Convert saved harness events back to OpenAI messages, without executing tools."""
    role = message["role"]
    blocks = message.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    if role == "toolResult":
        return {"role": "tool", "tool_call_id": message["toolCallId"], "content": text}
    result = {"role": role, "content": text or None}
    if role == "assistant":
        calls = [
            {
                "id": b["id"],
                "type": "function",
                "function": {"name": b["name"], "arguments": json.dumps(b["arguments"])},
            }
            for b in blocks
            if b.get("type") == "toolCall"
        ]
        if calls:
            result["tool_calls"] = calls
        thinking = "".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
        if thinking:
            result["reasoning_content"] = thinking
    return result


def escape_contexts(summary, tokenizer):
    from flock.worker.adapters.llm import DONE_PREAMBLE, DONE_TOOL, NO_BASH_PREAMBLE

    result = []
    for row in json.loads(summary.read_text()):
        candidates = []
        attempt = Path(row["attempt"])
        for session in sorted(attempt.glob("flock/*/agents/_sessions/*.jsonl")):
            agent = session.stem
            agents = session.parent.parent
            pin = json.loads((agents / "_tools_pins" / f"{agent}.json").read_text())
            tools = [DONE_TOOL] + [
                {
                    "type": "function",
                    "function": {k: t[k] for k in ("name", "description", "parameters")},
                }
                for t in pin["toolset"]["tools"]
            ]
            system = NO_BASH_PREAMBLE.format(agent=agent, workdir=pin["workdir"])
            system += DONE_PREAMBLE + "\n\n" + (agents / agent / "AGENTS.md").read_text()
            messages = [{"role": "system", "content": system}]
            for line in session.read_text().splitlines():
                event = json.loads(line)
                # Compacted contexts cannot be reconstructed exactly from public events.
                if "compact" in event.get("customType", ""):
                    break
                m = event.get("message")
                if not m or m.get("role") not in ("user", "assistant", "toolResult"):
                    continue
                if m["role"] == "assistant":
                    if not m.get("usage"):  # synthetic final summary
                        continue
                    rendered = copy.deepcopy(messages)
                    for message in rendered:
                        for call in message.get("tool_calls", []):
                            call["function"]["arguments"] = json.loads(
                                call["function"]["arguments"]
                            )
                    ids = tokenizer.apply_chat_template(
                        rendered,
                        tools=tools,
                        tokenize=True,
                        add_generation_prompt=True,
                        reasoning_effort="low",
                    )
                    if token_count(ids) <= 16384:
                        payload = {
                            "messages": messages.copy(),
                            "tools": tools,
                            "tool_choice": "required",
                        }
                        candidates.append(
                            {
                                "id": f"{row['difficulty']}:{agent}:{event['id']}",
                                "difficulty": row["difficulty"],
                                "payload": payload,
                                "prompt_tokens": token_count(ids),
                                "sha256": digest(payload),
                                "source": str(session),
                                "source_sha256": hashlib.sha256(session.read_bytes()).hexdigest(),
                            }
                        )
                messages.append(wire_message(m))
        random.Random(42 + row["difficulty"]).shuffle(candidates)
        if len(candidates) < 20:
            raise RuntimeError(f"Only {len(candidates)} eligible contexts at {row['difficulty']}")
        result.extend(candidates[:20])
    assert len(result) == 100
    return result


def main():
    from datasets import load_dataset
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoTokenizer

    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--escape-summary", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    model_rev = api.model_info(MODEL).sha
    dataset_rev = api.dataset_info(DATASET).sha
    model_path = snapshot_download(
        MODEL, revision=model_rev, cache_dir=args.cache, ignore_patterns=["*.md", "*.png", "*.jpg"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    low = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="low",
    )
    high = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        add_generation_prompt=True,
        reasoning_effort="xhigh",
    )
    if low == high:
        raise RuntimeError("Chat template ignores reasoning_effort; refusing mislabeled run")
    dataset = load_dataset(DATASET, split="test_sft", revision=dataset_rev)
    indices = list(range(len(dataset)))
    random.Random(42).shuffle(indices)
    rows = []
    for i in indices:
        m = next(m for m in dataset[i]["messages"] if m["role"] == "user")
        payload = {"messages": [{"role": "user", "content": m["content"]}]}
        ids = tokenizer.apply_chat_template(
            payload["messages"], tokenize=True, add_generation_prompt=True, reasoning_effort="low"
        )
        if token_count(ids) > 8192:
            continue
        rows.append(
            {
                "id": str(i),
                "payload": payload,
                "prompt_tokens": token_count(ids),
                "sha256": digest(payload),
            }
        )
        if len(rows) == 520:
            break
    if len(rows) != 520:
        raise RuntimeError("Insufficient eligible dataset rows")
    escape = escape_contexts(args.escape_summary, tokenizer)
    for name, data in [("ultrachat", rows[:512]), ("warmup", rows[512:]), ("agent_escape", escape)]:
        (args.output / f"{name}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in data)
        )
    manifest = {
        "model": MODEL,
        "model_revision": model_rev,
        "model_path": model_path,
        "dataset": DATASET,
        "dataset_revision": dataset_rev,
        "split": "test_sft",
        "seed": 42,
        "tokenizer_revision": model_rev,
        "reasoning_effort": "low",
        "reasoning_template_verified": True,
        "escape_note": "Reconstructed event prefixes, not captured HTTP requests; "
        "pre-compaction, <=16384 prompt tokens, original schemas.",
        "versions": {n: version(n) for n in ("transformers", "datasets", "huggingface_hub")},
        "files": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.glob("*.jsonl")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
