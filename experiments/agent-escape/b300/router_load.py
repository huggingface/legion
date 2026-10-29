"""Ramp real dataset requests through the Qwen router; never submits Slurm jobs."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import random
import re
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx
from benchmark import MODEL, measure, summarize

METRICS = re.compile(
    r'^(vllm:(?:num_requests_running|num_requests_waiting|kv_cache_usage_perc|'
    r'prefix_cache_hits_total|prefix_cache_queries_total|generation_tokens_total|'
    r'prompt_tokens_total))(?:\{.*\})?\s+(\S+)'
)


def ramp_plan(requests, levels):
    if not levels or levels != sorted(set(levels)) or levels[0] < 1:
        raise ValueError("Concurrency levels must be distinct positive integers in ascending order")
    counts = [2 * level for level in levels]
    remaining = requests - sum(counts)
    if remaining < 0:
        raise ValueError(f"At least {sum(counts)} requests are needed for two waves at every level")
    for i in range(len(counts)):
        extra = min(remaining, max(0, 128 - counts[i]))
        counts[i] += extra
        remaining -= extra
    counts[-1] += remaining
    return list(zip(levels, counts))


def load_rows(data, workload):
    manifest = json.loads((data / "manifest.json").read_text())
    name = workload + ".jsonl"
    raw = (data / name).read_bytes()
    if manifest["model"] != MODEL:
        raise ValueError("Dataset tokenizer/model does not match Qwen3.8-27B")
    if hashlib.sha256(raw).hexdigest() != manifest["files"][name]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if not rows:
        raise ValueError("Empty dataset")
    for row in rows:
        digest = hashlib.sha256(json.dumps(row["payload"], sort_keys=True).encode()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Prompt checksum mismatch: {row['id']}")
    return manifest, rows


def select_rows(rows, count, seed):
    rng = random.Random(seed)
    selected = []
    while len(selected) < count:
        cycle = list(rows)
        rng.shuffle(cycle)
        selected.extend(cycle[:count - len(selected)])
    return selected


def parse_metrics(text):
    result = {}
    for line in text.splitlines():
        match = METRICS.match(line)
        if match:
            value = float(match[2])
            if math.isfinite(value):
                result[match[1]] = result.get(match[1], 0) + value
    return result


async def sample_metrics(client, instances):
    async def one(instance):
        try:
            response = await client.get(instance["url"] + "/metrics", timeout=5)
            response.raise_for_status()
            return {"url": instance["url"], "metrics": parse_metrics(response.text)}
        except (httpx.HTTPError, ValueError) as error:
            return {"url": instance["url"], "error": str(error)[:500]}

    return await asyncio.gather(*(one(instance) for instance in instances))


async def run_level(client, base, rows, concurrency, output, *, context, timeout, instances,
                    save_responses=False):
    queue = asyncio.Queue()
    for index, row in enumerate(rows):
        queue.put_nowait((index, row))
    results = []
    active = peak = 0
    started, cpu_started = time.perf_counter(), time.process_time()
    stop_metrics = asyncio.Event()

    async def metrics_loop():
        with (output / "metrics.jsonl").open("w") as log:
            while not stop_metrics.is_set():
                sample = {"time": time.time(), "concurrency": concurrency,
                          "instances": await sample_metrics(client, instances)}
                log.write(json.dumps(sample) + "\n")
                log.flush()
                try:
                    await asyncio.wait_for(stop_metrics.wait(), timeout=5)
                except TimeoutError:
                    pass

    output.mkdir()
    metrics_task = asyncio.create_task(metrics_loop())
    try:
        with (output / "requests.jsonl").open("w") as log:
            async def worker():
                nonlocal active, peak
                while not queue.empty():
                    index, row = queue.get_nowait()
                    active += 1
                    peak = max(peak, active)
                    started_at = time.time()
                    try:
                        result = await measure(client, base, row, timeout, context=context)
                    finally:
                        active -= 1
                    result.update(request_index=index, concurrency=concurrency,
                                  started_at=started_at, completed_at=time.time())
                    if not save_responses:
                        result.pop("response", None)
                        result.pop("chunk_gaps_s", None)
                    log.write(json.dumps(result) + "\n")
                    log.flush()
                    results.append({k: v for k, v in result.items()
                                    if k not in ("response", "chunk_gaps_s")})
                    if len(results) % 32 == 0 or len(results) == len(rows):
                        print(json.dumps({"concurrency": concurrency, "done": len(results),
                                          "requests": len(rows), "inflight": active}), flush=True)

            async with asyncio.TaskGroup() as tasks:
                for _ in range(concurrency):
                    tasks.create_task(worker())
    finally:
        elapsed = time.perf_counter() - started
        stop_metrics.set()
        await metrics_task
    summary = summarize(results, elapsed, len(instances))
    summary.update(concurrency=concurrency, client_inflight_peak=peak,
                   client_cpu_seconds=time.process_time() - cpu_started)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


async def wait_endpoint(run, timeout):
    deadline = time.monotonic() + timeout
    while not (run / "endpoint.json").is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Router did not become ready: {run}")
        await asyncio.sleep(2)
    endpoint = json.loads((run / "endpoint.json").read_text())
    if endpoint["model"] != MODEL or not endpoint["instances"]:
        raise ValueError("Expected Qwen3.8-27B and a nonempty instance list")
    key = (run / "api-key").read_text().strip()
    if not key:
        raise ValueError("Empty router API key")
    config = json.loads((run / "config.json").read_text())
    return endpoint, key, config["context"]


def ensure_file_limit(concurrency):
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    needed = 2 * concurrency + 256
    if soft != resource.RLIM_INFINITY and soft < needed:
        if hard != resource.RLIM_INFINITY and hard < needed:
            raise ValueError(f"Client file-descriptor hard limit {hard} is below {needed}")
        resource.setrlimit(resource.RLIMIT_NOFILE, (needed, hard))


async def run(args):
    levels = [int(value) for value in args.concurrency.split(",")]
    plan = ramp_plan(args.requests, levels)
    if args.timeout <= 0 or args.wait_ready <= 0:
        raise ValueError("Timeouts must be positive")
    ensure_file_limit(max(levels))
    manifest, rows = load_rows(args.data, args.workload)
    endpoint, key, context = await wait_endpoint(args.server_run, args.wait_ready)
    if any(not 0 <= row["prompt_tokens"] < context for row in rows):
        raise ValueError("Dataset contains prompts exceeding the server's context limit")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"endpoint": endpoint, "dataset": manifest, "workload": args.workload,
                "requests": args.requests, "plan": plan, "seed": args.seed, "context": context,
                "unique_prompts": len(rows), "request_timeout_s": args.timeout,
                "sampling": "Shuffle each full dataset cycle; reuse verbatim prompts when exhausted",
                "cache": "Prefix caching and cache-aware routing remain enabled throughout",
                "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes())
                                  .hexdigest() for name in ("router_load.py", "benchmark.py")}}
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output), "plan": plan}), flush=True)
    summaries = []
    async with httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"}, timeout=None, trust_env=False,
        limits=httpx.Limits(max_connections=max(levels) + 16,
                           max_keepalive_connections=max(levels) + 16),
    ) as client:
        response = await client.get(endpoint["base_url"].rstrip("/") + "/models", timeout=10)
        response.raise_for_status()
        base = endpoint["base_url"].rstrip("/").removesuffix("/v1")
        for concurrency, count in plan:
            selected = select_rows(rows, count, args.seed + concurrency)
            summary = await run_level(
                client, base, selected, concurrency, output / f"concurrency-{concurrency}",
                context=context, timeout=args.timeout, instances=endpoint["instances"],
                save_responses=args.save_responses,
            )
            summaries.append(summary)
            (output / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
            print(json.dumps(summary), flush=True)
    conclusion = {
        "requests": sum(s["requests"] for s in summaries),
        "completed": sum(s["completed"] for s in summaries),
        "highest_tested_concurrency": max(levels),
        "highest_zero_error_concurrency": max(
            (s["concurrency"] for s in summaries if s["error_rate"] == 0), default=None),
        "first_error_concurrency": next(
            (s["concurrency"] for s in summaries if s["error_rate"] > 0), None),
        "best_output_throughput_concurrency": max(summaries, key=lambda s: s["output_tokens_s"])
        ["concurrency"],
        "hard_limit_established": False,
        "note": "Finite client-concurrency ramp; inspect throughput, latency and server queues. "
                "Repeated real prompts benefit from prefix caching; not a cold-cache benchmark.",
    }
    (output / "completion.json").write_text(json.dumps(conclusion, indent=2) + "\n")
    print(json.dumps(conclusion), flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--server-run", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(".flock/b300-qwen/data"))
    parser.add_argument("--workload", choices=("ultrachat", "agent_escape"), default="ultrachat")
    parser.add_argument("--requests", type=int, default=5000)
    parser.add_argument("--concurrency", default="16,32,64,128,256,512,1024")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--wait-ready", type=float, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-responses", action="store_true")
    parser.add_argument("--output", type=Path, default=Path(".flock/b300-qwen/load") /
                        datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"))
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
