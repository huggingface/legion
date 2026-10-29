"""Deadline-bounded screening: all GPU layouts first, then balanced agent replay."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import benchmark as bench
import httpx
from analyze import analyze


def stages():
    return [
        (dataset, layout, gpus)
        for dataset in ("ultrachat", "agent_escape")
        for layout, gpus in bench.MATRIX
    ]


def select_rows(datasets, dataset):
    if dataset == "ultrachat":
        return datasets[dataset][:32]
    return [
        row
        for difficulty in (5, 10, 15, 20, 25)
        for row in [r for r in datasets[dataset] if r["difficulty"] == difficulty][:4]
    ]


def reuse_case(prior, selected, dataset, layout, gpus, concurrency, destination):
    """Reuse one complete case only when every prompt hash and order matches."""
    if prior is None or not (prior / "summary.jsonl").exists():
        return None
    for line in (prior / "summary.jsonl").read_text().splitlines():
        result = json.loads(line)
        if (result["dataset"], result["layout"], result["gpus"], result["concurrency"]) != (
            dataset,
            layout,
            gpus,
            concurrency,
        ):
            continue
        source = prior / f"{layout}{gpus}-{dataset}" / f"c{concurrency}-r{result['repeat']}.jsonl"
        if not source.exists():
            continue
        requests = [json.loads(line) for line in source.read_text().splitlines()]
        # Concurrent completions may be out of input order. IDs plus hashes identify the set.
        expected = sorted((r["id"], r["sha256"]) for r in selected)
        actual = sorted((r["id"], r["prompt_sha256"]) for r in requests)
        if actual != expected or any(r["status"] != "completed" for r in requests):
            continue
        shutil.copy2(source, destination)
        return {**result, "repeat": 0, "reused_from": str(source)}
    return None


def save_result(output, result):
    with (output / "summary.jsonl").open("a") as f:
        f.write(json.dumps(result) + "\n")
    bench.export_results(output)
    report = analyze(output)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("RESULT", json.dumps(result), flush=True)


async def run_stage(args, client, datasets, dataset, layout, gpus, visible):
    directory = args.output / f"{layout}{gpus}-{dataset}"
    directory.mkdir(exist_ok=True)
    selected = select_rows(datasets, dataset)
    cs = (4, 32) if dataset == "ultrachat" else (4,)
    pending = []
    for concurrency in cs:
        result = reuse_case(
            args.prior,
            selected,
            dataset,
            layout,
            gpus,
            concurrency,
            directory / f"c{concurrency}-r0.jsonl",
        )
        if result is not None:
            save_result(args.output, result)
        else:
            pending.append(concurrency)
    if not pending:
        return
    command = bench.server_command(
        args.model_path, layout, gpus, dataset == "agent_escape", args.port
    )
    (directory / "command.json").write_text(json.dumps(command, indent=2))
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(visible[:gpus])}
    proc = None
    try:
        with (directory / "server.log").open("w") as log:
            proc = subprocess.Popen(  # noqa: ASYNC220 -- startup outside timed measurement
                command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
        url = f"http://127.0.0.1:{args.port}"
        await bench.readiness(client, url, proc, timeout=420)
        warmup = datasets["warmup"][:1]  # one disjoint real prompt, once per server
        await bench.tokenize_rows(client, url, selected + warmup)
        warmed = await bench.run_case(client, url, warmup, 1, directory / "warmup.jsonl", gpus)
        if warmed["completed"] != len(warmup):
            raise RuntimeError("Warmup failed")
        print("READY", dataset, layout, gpus, flush=True)
        for concurrency in pending:
            if dataset == "agent_escape":
                response = await client.post(url + "/reset_prefix_cache")
                response.raise_for_status()
            name = f"c{concurrency}-r0"
            before = await client.get(url + "/metrics")
            (directory / f"{name}-metrics-before.txt").write_text(before.text)
            result = await bench.run_case(
                client, url, selected, concurrency, directory / f"{name}.jsonl", gpus
            )
            result.update(
                layout=layout,
                gpus=gpus,
                dataset=dataset,
                concurrency=concurrency,
                repeat=0,
                profile="quick",
            )
            after = await client.get(url + "/metrics")
            (directory / f"{name}-metrics-after.txt").write_text(after.text)
            save_result(args.output, result)
    finally:
        bench.stop_process(proc)


async def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    if args.prior is not None:
        previous = json.loads((args.prior / "runtime.json").read_text())
        if (
            previous["dataset_manifest"]["model_revision"] != manifest["model_revision"]
            or previous["dataset_manifest"]["files"] != manifest["files"]
            or previous["context"] != bench.CONTEXT
            or previous["sampling"] != {"temperature": 0, "seed": 42, "reasoning_effort": "low"}
        ):
            raise ValueError("Prior run has incompatible model, dataset, context or sampling")
    datasets = {
        name: [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines()]
        for name in ("ultrachat", "warmup", "agent_escape")
    }
    runtime = {
        "profile": "quick",
        "expected_cases": 21,
        "required_repetitions": 1,
        "deadline_epoch": args.deadline_epoch,
        "dataset_manifest": manifest,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "transformers", "httpx", "jsonschema")
        },
        "context": bench.CONTEXT,
        "sampling": {"temperature": 0, "seed": 42, "reasoning_effort": "low"},
        "note": "Single-repetition screening, not a saturation or variance study. "
        "Natural EOS; deadline-interrupted requests are incomplete.",
        "selected_ids": {
            name: [r["id"] for r in select_rows(datasets, name)]
            for name in ("ultrachat", "agent_escape")
        },
    }
    (args.output / "runtime.json").write_text(json.dumps(runtime, indent=2))
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    if len(visible) < 8:
        raise RuntimeError("The complete sweep needs eight visible GPUs")
    monitor = subprocess.Popen(  # noqa: ASYNC220 -- telemetry setup
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw",
            "--format=csv",
            "-l",
            "2",
        ],
        stdout=(args.output / "gpu.csv").open("w"),
        start_new_session=True,
    )
    started = time.time()
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": "Bearer " + os.environ["VLLM_API_KEY"]},
            timeout=900,
            trust_env=False,
            limits=httpx.Limits(max_connections=150),
        ) as client:
            for dataset, layout, gpus in stages():
                remaining = args.deadline_epoch - time.time() - 70  # reserve process cleanup
                if remaining <= 0:
                    break
                print(
                    "START",
                    dataset,
                    layout,
                    gpus,
                    "remaining_seconds",
                    round(remaining),
                    flush=True,
                )
                try:
                    # Prevent a bad topology from consuming the whole comparison budget.
                    async with asyncio.timeout(min(600, remaining)):
                        await run_stage(args, client, datasets, dataset, layout, gpus, visible)
                except (TimeoutError, RuntimeError, ValueError, httpx.HTTPError, OSError) as exc:
                    directory = args.output / f"{layout}{gpus}-{dataset}"
                    directory.mkdir(exist_ok=True)
                    (directory / "failure.json").write_text(
                        json.dumps(
                            {
                                "error": str(exc) or type(exc).__name__,
                                "partial_requests": "Any incomplete case JSONL is excluded from summaries.",
                            }
                        )
                    )
                    print("FAILED", dataset, layout, gpus, repr(exc), flush=True)
    finally:
        bench.stop_process(monitor)
        summary = args.output / "summary.jsonl"
        count = len(summary.read_text().splitlines()) if summary.exists() else 0
        (args.output / "completion.json").write_text(
            json.dumps(
                {
                    "completed_cases": count,
                    "expected_cases": 21,
                    "complete": count == 21,
                    "elapsed_seconds": time.time() - started,
                    "deadline_epoch": args.deadline_epoch,
                    "finished_epoch": time.time(),
                },
                indent=2,
            )
        )
        if count:
            (args.output / "report.json").write_text(json.dumps(analyze(args.output), indent=2))
        print("FINISHED", count, "/21 cases", flush=True)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--deadline-epoch", type=float, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        task = loop.create_task(run(args))
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        loop.add_signal_handler(signal.SIGUSR1, task.cancel)

        async def wait_task():
            await task

        try:
            runner.run(wait_task())
        except asyncio.CancelledError:
            raise SystemExit(143) from None


if __name__ == "__main__":
    main()
