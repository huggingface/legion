"""Export faithful multi-turn trajectories from verified winning SFT swarms."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from task_splits import select_tasks


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assistant(message):
    if message.get("errorMessage") or message.get("stopReason") not in ("stop", "toolUse"):
        raise ValueError("incomplete_or_errored_response")
    text, calls = [], []
    for part in message["content"]:
        if part["type"] == "text":
            text.append(part["text"])
        elif part["type"] == "toolCall":
            if not isinstance(part["arguments"], dict):
                raise TypeError("malformed_tool_arguments")
            calls.append(
                {
                    "id": part["id"],
                    "type": "function",
                    "function": {
                        "name": part["name"],
                        "arguments": json.dumps(part["arguments"], ensure_ascii=False),
                    },
                }
            )
        elif part["type"] == "thinking" and part.get("thinking"):
            raise ValueError("unexpected_reasoning")
        else:
            raise ValueError("unsupported_response_content")
    if not text and not calls:
        raise ValueError("empty_response")
    result = {"role": "assistant", "content": "\n".join(text) or None}
    if calls:
        result["tool_calls"] = calls
    return result


def validate_tools(messages, tools):
    names = {t["function"]["name"] for t in tools}
    outstanding = set()
    for message in messages:
        role = message["role"]
        if role == "tool":
            ident = message["tool_call_id"]
            if ident not in outstanding:
                raise ValueError("orphan_tool_result")
            outstanding.remove(ident)
        else:
            if outstanding:
                raise ValueError("missing_tool_result")
            for call in message.get("tool_calls", []):
                if call["function"]["name"] not in names or call["id"] in outstanding:
                    raise ValueError("invalid_tool_call")
                args = call["function"]["arguments"]
                if not isinstance(json.loads(args) if isinstance(args, str) else args, dict):
                    raise TypeError("malformed_tool_arguments")
                outstanding.add(call["id"])
    # The final assistant target may issue calls whose results are not in this example.


def pairs(path, rejected):
    requests, seen = {}, set()
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    for index, line in enumerate(lines):
        try:
            event = json.loads(line)
        except ValueError:
            rejected["malformed_capture_line"] += 1
            continue
        ident = event.get("id")
        if event["type"] == "request":
            if not ident or ident in requests:
                raise ValueError("duplicate_request_id")
            requests[ident] = event
        elif event["type"] == "response":
            message = event.get("message", {})
            # Pi emits this event when its turn-budget hook aborts before a new
            # provider request. It contains no model output and is not a pair.
            if (
                ident is None
                and index == len(lines) - 1
                and requests
                and requests.keys() == seen
                and message.get("role") == "assistant"
                and message.get("content") == []
                and message.get("stopReason") == "error"
                and message.get("errorMessage") == "This operation was aborted"
            ):
                rejected["empty_terminal_abort"] += 1
                continue
            if ident not in requests or ident in seen:
                rejected["unpaired_response"] += 1
                continue
            seen.add(ident)
            yield requests[ident], event
    rejected["unanswered_request"] += len(requests.keys() - seen)


def latest(root, row):
    dirs = [root / (row["prefix"] + f"-a{i}") for i in (0, 1)]
    return next((p for p in reversed(dirs) if (p / "owner.json").exists()), dirs[0])


def rendered_length(messages, tools, tokenizer):
    if tokenizer is None:
        return None
    return len(
        tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=True,
            return_dict=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    )


def trajectories(examples, tokenizer=None, max_tokens=65536):
    """Join only exact prefixes; context rewrites and unvalidated assistants start a new row.

    The initial context stays in prompt, including any assistants retained by compaction.
    Every assistant in completion is a validated target, and intervening observations are
    copied from the next actual request. No terminal tool result or final answer is invented.
    """
    groups = []
    current = []
    reason = "agent_start"
    for example in examples:
        boundary = None
        if current:
            previous = current[-1]
            history = previous["prompt"] + previous["completion"]
            prompt = example["prompt"]
            if example["tools"] != previous["tools"]:
                boundary = "tools_changed"
            elif prompt[: len(history)] != history:
                boundary = "context_changed"
            elif any(m["role"] == "assistant" for m in prompt[len(history) :]):
                boundary = "unvalidated_assistant_context"
        if boundary:
            groups.append((current, reason))
            current, reason = [], boundary
        current.append(example)
    if current:
        groups.append((current, reason))

    def emit(group, boundary_reason):
        first, last = group[0], group[-1]
        messages = last["prompt"] + last["completion"]
        length = rendered_length(messages, first["tools"], tokenizer)
        if length is not None and length > max_tokens:
            # Splitting rows cannot shorten this last target's actual conditioning
            # history. Fail rather than silently truncate or invent a new context.
            raise ValueError("over_context_budget")
        prefix_length = len(first["prompt"])
        completion = messages[prefix_length:]
        indices = [len(e["prompt"]) - prefix_length for e in group]
        assert [i for i, m in enumerate(completion) if m["role"] == "assistant"] == indices
        for example, index in zip(group, indices, strict=True):
            assert messages[: prefix_length + index] == example["prompt"]
            assert completion[index] == example["completion"][0]
        metadata = dict(first["metadata"])
        for key in ("request_id", "turn_index", "response_stop_reason"):
            metadata.pop(key)
        metadata.update(
            start_turn_index=first["metadata"]["turn_index"],
            end_turn_index=last["metadata"]["turn_index"],
            request_ids=[e["metadata"]["request_id"] for e in group],
            response_ids=[e["id"] for e in group],
            response_stop_reasons=[e["metadata"]["response_stop_reason"] for e in group],
            assistant_message_indices=indices,
            assistant_responses=len(group),
            segment_start_reason=boundary_reason,
            tokens=length,
        )
        ident = hashlib.sha256(f"trajectory-v1/{first['id']}".encode()).hexdigest()
        yield {
            "id": ident,
            "prompt": first["prompt"],
            "completion": completion,
            "tools": first["tools"],
            "metadata": metadata,
        }

    for group, boundary_reason in groups:
        yield from emit(group, boundary_reason)


def export(root, destination, tokenizer=None, pilots=False, format="trajectories"):
    if format not in ("trajectories", "responses"):
        raise ValueError("Unknown export format")
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    study = read(root / "study.json")
    manifest = read(root / "splits.json")
    selected = select_tasks(
        manifest, read(root / "registry.json"), "sft", [5, 10, 15, 20, 25], "evaluate"
    )
    allowed = {r["scenario_id"]: r for r in selected}
    rows = list(read(root / "smokes.json").values()) if pilots else read(root / "manifest.json")
    counts, rejects, difficulties = Counter(), Counter(), {}
    ids = set()
    output = destination / "examples.jsonl"
    temporary = output.with_suffix(".tmp")
    with temporary.open("w") as stream:
        for row in rows:
            expected = allowed.get(row["id"])
            if expected is None or row["scenario_sha256"] != expected["scenario_sha256"]:
                raise ValueError("cross_split_or_changed_scenario")
            run = latest(root, row)
            if not (run / "completion.json").exists():
                counts["pending"] += 1
                continue
            counts["attempted"] += 1
            result = read(run / "completion.json")
            if result["status"] != "finished" or not result.get("cleanup_complete"):
                counts["infrastructure_or_incomplete"] += 1
                continue
            counts["valid"] += 1
            level = difficulties.setdefault(str(row["difficulty"]), Counter())
            level["valid_teams"] += 1
            winners = set()
            for trial in result["trials"]:
                evidence = run / "native" / trial["trial"] / "artifacts/evidence/final.json"
                if trial["rewards"].get("reward") == 1 and evidence.exists():
                    e = read(evidence)
                    if e.get("won") is True and e.get("scenario_sha256") == row["scenario_sha256"]:
                        winners.add(trial["trial"])
            if not result["verified_success"] or not winners:
                counts["unsolved"] += 1
                continue
            counts["winning_teams"] += 1
            level["winning_teams"] += 1
            level["individual_winners"] += len(winners)
            for trial in sorted((run / "native").iterdir()):
                capture = trial / "agent/sft-capture.jsonl"
                if not capture.exists():
                    rejects["missing_agent_capture"] += 1
                    continue
                contributed = False
                examples = []
                source_hash = sha(capture)
                request_ids, response_ids, last_stop = set(), set(), None
                for line in capture.open():
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if event.get("type") == "request":
                        request_ids.add(event.get("id"))
                    elif event.get("type") == "response":
                        response_ids.add(event.get("id"))
                        last_stop = event.get("message", {}).get("stopReason")
                native_result = (
                    read(trial / "result.json") if (trial / "result.json").exists() else {}
                )
                for turn_index, (request, response) in enumerate(pairs(capture, rejects)):
                    try:
                        settings = request["settings"]
                        if (
                            settings["model"] != study["model"]
                            or settings["enable_thinking"] is not False
                        ):
                            raise ValueError("wrong_model_settings")
                        target = assistant(response["message"])
                        prompt, tools = request["messages"], request["tools"]
                        validate_tools(prompt + [target], tools)
                        # Decode wire-format argument strings for training templates;
                        # raw requests remain unchanged in the capture artifact.
                        rendered_messages = json.loads(json.dumps(prompt + [target]))
                        for message in rendered_messages:
                            for call in message.get("tool_calls", []):
                                arguments = call["function"]["arguments"]
                                if isinstance(arguments, str):
                                    call["function"]["arguments"] = json.loads(arguments)
                        prompt, target = rendered_messages[:-1], rendered_messages[-1]
                        token_count = None
                        if tokenizer is not None and format == "responses":
                            token_count = rendered_length(rendered_messages, tools, tokenizer)
                            if token_count > 65536:
                                raise ValueError("over_context_budget")
                    except (ValueError, KeyError, TypeError) as exc:
                        rejects[str(exc)] += 1
                        continue
                    ident = hashlib.sha256(
                        f"{run.name}/{trial.name}/{request['id']}".encode()
                    ).hexdigest()
                    if ident in ids:
                        raise ValueError("duplicate_example")
                    ids.add(ident)
                    metadata = {
                        "scenario_id": row["id"],
                        "difficulty": row["difficulty"],
                        "swarm": run.name,
                        "agent": trial.name,
                        "attempt": int(run.name[-1]),
                        "request_id": request["id"],
                        "turn_index": turn_index,
                        "team_success": True,
                        "individual_success": trial.name in winners,
                        "team_termination_reason": result["reason"],
                        "agent_last_response_stop_reason": last_stop,
                        "agent_unanswered_requests": len(request_ids - response_ids),
                        "agent_exception_type": native_result.get("exception_type"),
                        "response_stop_reason": response["message"]["stopReason"],
                        "model_revision": study["model_revision"],
                        "registry_revision": study["revision"],
                        "split": "sft",
                        "split_digest": sha(root / "splits.json"),
                        "scenario_sha256": row["scenario_sha256"],
                        "capture_sha256": source_hash,
                        "tokens": token_count,
                        "pilot": pilots,
                    }
                    examples.append(
                        {
                            "id": ident,
                            "prompt": prompt,
                            "completion": [target],
                            "tools": tools,
                            "metadata": metadata,
                        }
                    )
                exported = (
                    trajectories(examples, tokenizer) if format == "trajectories" else examples
                )
                for example in exported:
                    stream.write(json.dumps(example, ensure_ascii=False) + "\n")
                    contributed = True
                    counts["examples"] += 1
                    responses = example["metadata"].get("assistant_responses", 1)
                    counts["assistant_responses"] += responses
                    level = difficulties.setdefault(str(row["difficulty"]), Counter())
                    level["examples"] += 1
                    level["assistant_responses"] += responses
                    level["tokens"] += example["metadata"]["tokens"] or 0
                if contributed:
                    counts["contributing_agents"] += 1
                    level["contributing_agents"] += 1
    temporary.replace(output)
    # Arrow's JSON inference cannot represent arbitrary tool parameter schemas reliably.
    # Store variable JSON columns as strings in Parquet; decode with json.loads on load.
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = pa.schema(
        [(key, pa.string()) for key in ["id", "prompt", "completion", "tools", "metadata"]]
    )
    with pq.ParquetWriter(destination / "examples.parquet", schema) as writer:
        batch = []
        for line in output.open():
            value = json.loads(line)
            batch.append(
                {k: v if k == "id" else json.dumps(v, ensure_ascii=False) for k, v in value.items()}
            )
            if len(batch) == 128:
                writer.write_table(pa.Table.from_pylist(batch, schema=schema))
                batch = []
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=schema))
    report = {
        "format": format,
        "format_version": 1,
        "counts": dict(counts),
        "rejections": dict(rejects),
        "difficulty": difficulties,
        "planned": len(rows),
        "pilot": pilots,
        "render_validated": tokenizer is not None,
        "sha256": sha(output),
        "parquet_json_columns": ["prompt", "completion", "tools", "metadata"],
    }
    (destination / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--pilots", action="store_true")
    parser.add_argument("--format", choices=["trajectories", "responses"], default="trajectories")
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    print(json.dumps(export(args.root, args.destination, tok, args.pilots, args.format)))
