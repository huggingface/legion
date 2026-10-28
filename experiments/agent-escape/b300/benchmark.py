"""EOS-respecting streaming benchmarks and sequential vLLM TP/DP orchestration."""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import hashlib
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx
import jsonschema

MODEL = "Qwen/Qwen3.8-27B"
CONTEXT = 32768
MATRIX = [("tp", n) for n in (1, 2, 4, 8)] + [("replicas", n) for n in (2, 4, 8)]


def payload_for(row):
    # No ignore_eos, min_tokens or artificial output target. The context is the safety cap.
    return {
        **row["payload"],
        "model": MODEL,
        "temperature": 0,
        "seed": 42,
        "reasoning_effort": "low",
        "chat_template_kwargs": {"reasoning_effort": "low"},
        "max_tokens": CONTEXT - row["prompt_tokens"],
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def tool_error(calls, tools):
    schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
    if not calls:
        return "missing_tool_call"
    for call in calls.values():
        fn = call["function"]
        if fn["name"] not in schemas:
            return "unknown_tool"
        try:
            args = json.loads(fn["arguments"])
        except ValueError:
            return "malformed_tool_arguments"
        try:
            jsonschema.validate(args, schemas[fn["name"]])
        except jsonschema.ValidationError:
            return "invalid_tool_arguments"
    return None


async def measure(client, url, row, timeout=900):
    started = time.perf_counter()
    result = {
        "id": row["id"],
        "prompt_sha256": row["sha256"],
        "status": "incomplete",
        "finish_reason": None,
        "prompt_tokens": 0,
        "output_tokens": 0,
        "ttft_s": None,
        "tpot_s": None,
    }
    first = None
    last = None
    gaps = []
    calls = {}
    texts = {"content": "", "reasoning_content": ""}
    usage = None
    done = False
    try:
        async with asyncio.timeout(timeout):
            async with client.stream(
                "POST", url + "/v1/chat/completions", json=payload_for(row)
            ) as response:
                if response.status_code != 200:
                    result.update(
                        status="http_error",
                        http_status=response.status_code,
                        error=(await response.aread()).decode()[:2000],
                    )
                else:
                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            done = True
                            break
                        event = json.loads(data)
                        if event.get("error"):
                            raise ValueError(str(event["error"]))
                        if event.get("usage"):
                            usage = event["usage"]
                        for choice in event.get("choices", []):
                            delta = choice.get("delta", {})
                            token_bearing = any(
                                delta.get(k)
                                for k in ("content", "reasoning_content", "reasoning", "tool_calls")
                            )
                            if token_bearing:
                                now = time.perf_counter()
                                if first is None:
                                    first = now
                                if last is not None:
                                    gaps.append(now - last)
                                last = now
                            for key in texts:
                                texts[key] += delta.get(key) or ""
                            if delta.get("reasoning"):
                                texts["reasoning_content"] += delta["reasoning"]
                            for call in delta.get("tool_calls", []):
                                saved = calls.setdefault(
                                    call["index"],
                                    {"id": "", "function": {"name": "", "arguments": ""}},
                                )
                                saved["id"] += call.get("id") or ""
                                for key in ("name", "arguments"):
                                    saved["function"][key] += (
                                        call.get("function", {}).get(key) or ""
                                    )
                            if choice.get("finish_reason"):
                                result["finish_reason"] = choice["finish_reason"]
                    if done and usage is not None:
                        result["prompt_tokens"] = usage["prompt_tokens"]
                        result["output_tokens"] = usage["completion_tokens"]
                        result["usage"] = usage
                        finish = result["finish_reason"]
                        result["status"] = (
                            "completed"
                            if finish in ("stop", "tool_calls")
                            else "context_limit"
                            if finish == "length"
                            else "incomplete"
                        )
                        if result["status"] == "completed" and row["payload"].get("tools"):
                            result["status"] = (
                                tool_error(calls, row["payload"]["tools"]) or "completed"
                            )
                    else:
                        result["status"] = "missing_usage" if done else "stream_interrupted"
    except TimeoutError:
        result["status"] = "timeout"
    except (httpx.HTTPError, ValueError, KeyError) as exc:
        result.update(status="stream_error", error=str(exc)[:2000])
    result["latency_s"] = time.perf_counter() - started
    result["ttft_s"] = first - started if first is not None else None
    if first is not None and last is not None and result["output_tokens"] > 1:
        result["tpot_s"] = (last - first) / (result["output_tokens"] - 1)
    result["chunk_gaps_s"] = gaps  # chunks can contain multiple tokens; not token-level ITL
    result["response"] = {**texts, "tool_calls": list(calls.values())}
    return result


def quantile(values, q):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return values[round((len(values) - 1) * q)]


def summarize(results, elapsed, gpus):
    good = [r for r in results if r["status"] == "completed"]
    output = sum(r["output_tokens"] for r in good)
    total = output + sum(r["prompt_tokens"] for r in good)
    summary = {
        "requests": len(results),
        "completed": len(good),
        "seconds": elapsed,
        "output_tokens_s": output / elapsed,
        "total_tokens_s": total / elapsed,
        "requests_s": len(good) / elapsed,
        "output_tokens_s_gpu": output / elapsed / gpus,
        "observed_output_tokens": sum(r["output_tokens"] for r in results),
        "error_rate": 1 - len(good) / len(results),
        "statuses": dict(collections.Counter(r["status"] for r in results)),
        "finish_reasons": dict(collections.Counter(r["finish_reason"] for r in results)),
    }
    for name in ("ttft_s", "latency_s", "tpot_s", "output_tokens"):
        for label, q in (("p50", 0.5), ("p95", 0.95)):
            summary[f"{name}_{label}"] = quantile([r[name] for r in good], q)
    return summary


async def run_case(client, url, rows, concurrency, output, gpus):
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    results = []
    started = time.perf_counter()
    with output.open("w") as log:

        async def worker():
            while not queue.empty():
                row = queue.get_nowait()
                result = await measure(client, url, row)
                results.append(result)
                log.write(json.dumps(result) + "\n")
                log.flush()

        await asyncio.gather(*(worker() for _ in range(min(concurrency, len(rows)))))
    return summarize(results, time.perf_counter() - started, gpus)


def server_command(model_path, layout, gpus, prefix_cache, port):
    return [
        "vllm",
        "serve",
        model_path,
        "--served-model-name",
        MODEL,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(CONTEXT),
        "--gpu-memory-utilization",
        "0.9",
        "--max-num-seqs",
        "128",
        "--max-num-batched-tokens",
        "8192",
        "--tensor-parallel-size",
        str(gpus if layout == "tp" else 1),
        "--data-parallel-size",
        str(gpus if layout == "replicas" else 1),
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--language-model-only",
        "--default-chat-template-kwargs",
        '{"reasoning_effort":"low"}',
        "--enable-prefix-caching" if prefix_cache else "--no-enable-prefix-caching",
    ]


def stop_process(proc):
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    except ProcessLookupError:
        pass


async def readiness(client, url, proc, timeout=1800):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Server exited during startup ({proc.returncode})")
        try:
            r = await client.get(url + "/v1/models", timeout=5)
            if r.status_code == 200:
                return
        except httpx.HTTPError:
            pass
        await asyncio.sleep(5)
    raise TimeoutError("Server readiness timed out")


async def tokenize_rows(client, url, rows):
    for row in rows:
        body = {
            **row["payload"],
            "model": MODEL,
            "add_generation_prompt": True,
            "chat_template_kwargs": {"reasoning_effort": "low"},
        }
        body.pop("tool_choice", None)
        r = await client.post(url + "/tokenize", json=body)
        r.raise_for_status()
        row["prompt_tokens"] = r.json()["count"]
        if row["prompt_tokens"] >= CONTEXT:
            raise ValueError(f"Prompt exceeds context: {row['id']}")


async def smoke(client, url, rows):
    for row in rows:
        r = await measure(client, url, row)
        if r["status"] != "completed":
            raise RuntimeError(f"Smoke failed: {r['status']}: {r.get('error', '')}")
    # Exercise the non-streaming interface used by Flock, too.
    body = payload_for(rows[-1])
    body["stream"] = False
    body.pop("stream_options")
    response = await client.post(url + "/v1/chat/completions", json=body)
    response.raise_for_status()
    data = response.json()
    calls = {i: c for i, c in enumerate(data["choices"][0]["message"].get("tool_calls", []))}
    if tool_error(calls, rows[-1]["payload"]["tools"]):
        raise RuntimeError("Nonstreaming tool-call smoke failed")


def export_results(output):
    rows = [json.loads(line) for line in (output / "summary.jsonl").read_text().splitlines()]
    fields = [k for k in rows[0] if k not in ("statuses", "finish_reasons")]
    with (output / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    try:
        plotting = os.environ.get("BENCH_PLOT_PATH")
        if plotting and plotting not in sys.path:
            sys.path.insert(0, plotting)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, dataset in zip(axes, ("ultrachat", "agent_escape")):
            for layout, gpus in MATRIX:
                subset = [
                    r
                    for r in rows
                    if r["dataset"] == dataset and r["layout"] == layout and r["gpus"] == gpus
                ]
                cs = sorted({r["concurrency"] for r in subset})
                if not cs:
                    continue
                ys = [
                    sum(r["output_tokens_s"] for r in subset if r["concurrency"] == c)
                    / sum(r["concurrency"] == c for r in subset)
                    for c in cs
                ]
                ax.plot(cs, ys, marker="o", label=f"{layout} {gpus} GPU")
            ax.set(title=dataset, xlabel="Concurrent requests", ylabel="Output tokens/s")
            ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output / "throughput.png", dpi=160)
        plt.close(fig)
    except ImportError:
        pass


async def orchestrate(args):
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if hashlib.sha256((args.data / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    datasets = {
        name: [json.loads(line) for line in (args.data / f"{name}.jsonl").read_text().splitlines()]
        for name in ("ultrachat", "warmup", "agent_escape")
    }
    versions = {
        name: importlib.metadata.version(name)
        for name in ("vllm", "transformers", "httpx", "jsonschema")
    }
    (args.output / "runtime.json").write_text(
        json.dumps(
            {
                "versions": versions,
                "dataset_manifest": manifest,
                "context": CONTEXT,
                "allocation_gpus": 8,
                "routing": "vLLM native data-parallel frontend, pinned with engine",
                "sampling": {"temperature": 0, "seed": 42, "reasoning_effort": "low"},
            },
            indent=2,
        )
    )
    key = os.environ["VLLM_API_KEY"]
    url = f"http://127.0.0.1:{args.port}"
    env = os.environ.copy()
    visible = env.get("CUDA_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7").split(",")
    matrix = MATRIX if args.mode == "benchmark" else [(args.layout, args.gpus)]
    allocation_start = time.monotonic()
    active_gpu_seconds = 0
    monitor = subprocess.Popen(  # noqa: ASYNC220 -- setup outside measured requests
        [
            "nvidia-smi",
            (
                "--query-gpu=timestamp,index,uuid,name,"
                "utilization.gpu,memory.used,memory.total,power.draw"
            ),
            "--format=csv",
            "-l",
            "2",
        ],
        stdout=(args.output / "gpu.csv").open("w"),
        start_new_session=True,
    )
    try:
        async with httpx.AsyncClient(
            headers={"Authorization": f"Bearer {key}"},
            timeout=900,
            limits=httpx.Limits(max_connections=150),
        ) as client:
            for layout, gpus in matrix:
                if len(visible) < gpus:
                    raise ValueError(f"Requested {gpus} GPUs, only {visible} visible")
                for dataset in ("ultrachat", "agent_escape"):
                    if args.mode == "serve" and dataset == "ultrachat":
                        continue
                    label = f"{layout}{gpus}-{dataset}"
                    directory = args.output / label
                    directory.mkdir(exist_ok=True)
                    command = server_command(
                        args.model_path, layout, gpus, dataset == "agent_escape", args.port
                    )
                    (directory / "command.json").write_text(json.dumps(command, indent=2))
                    env["CUDA_VISIBLE_DEVICES"] = ",".join(visible[:gpus])
                    proc = None
                    begun = time.monotonic()
                    try:
                        with (directory / "server.log").open("w") as log:
                            proc = subprocess.Popen(  # noqa: ASYNC220 -- server startup
                                command,
                                env=env,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        await readiness(client, url, proc)
                        await tokenize_rows(client, url, datasets[dataset] + datasets["warmup"])
                        # First real replay is a tools smoke; all sampling is low reasoning.
                        await tokenize_rows(client, url, datasets["agent_escape"][:1])
                        await smoke(
                            client, url, [datasets["warmup"][0], datasets["agent_escape"][0]]
                        )
                        endpoint = {
                            "base_url": f"http://{os.uname().nodename}:{args.port}/v1",
                            "model": MODEL,
                            "layout": layout,
                            "gpus": gpus,
                        }
                        (args.output / "endpoint.json").write_text(json.dumps(endpoint, indent=2))
                        print("READY", label, endpoint, flush=True)
                        if args.mode == "serve":
                            while proc.poll() is None:
                                await asyncio.sleep(5)
                            raise RuntimeError("Serving process exited")
                        cs = (1, 4, 16, 32, 64, 128) if dataset == "ultrachat" else (4, 16)
                        for concurrency in cs:
                            selected = (
                                datasets[dataset][: max(32, 4 * concurrency)]
                                if dataset == "ultrachat"
                                else datasets[dataset]
                            )
                            for repeat in range(3):
                                if dataset == "agent_escape":
                                    reset = await client.post(url + "/reset_prefix_cache")
                                    reset.raise_for_status()
                                await run_case(
                                    client,
                                    url,
                                    datasets["warmup"],
                                    4,
                                    directory / "warmup.jsonl",
                                    gpus,
                                )
                                # Warmup must not populate measured replay caches.
                                if dataset == "agent_escape":
                                    reset = await client.post(url + "/reset_prefix_cache")
                                    reset.raise_for_status()
                                metrics_before = await client.get(url + "/metrics")
                                name = f"c{concurrency}-r{repeat}"
                                (directory / f"{name}-metrics-before.txt").write_text(
                                    metrics_before.text
                                )
                                result = await run_case(
                                    client,
                                    url,
                                    selected,
                                    concurrency,
                                    directory / f"{name}.jsonl",
                                    gpus,
                                )
                                result.update(
                                    layout=layout,
                                    gpus=gpus,
                                    dataset=dataset,
                                    concurrency=concurrency,
                                    repeat=repeat,
                                )
                                metrics_after = await client.get(url + "/metrics")
                                (directory / f"{name}-metrics-after.txt").write_text(
                                    metrics_after.text
                                )
                                with (args.output / "summary.jsonl").open("a") as f:
                                    f.write(json.dumps(result) + "\n")
                                export_results(args.output)
                                print("RESULT", json.dumps(result), flush=True)
                    except Exception as exc:
                        (directory / "failure.json").write_text(json.dumps({"error": str(exc)}))
                        print("FAILED", label, str(exc), flush=True)
                        if args.mode == "serve":
                            raise
                    finally:
                        stop_process(proc)
                        active_gpu_seconds += (time.monotonic() - begun) * gpus
                        (args.output / "endpoint.json").unlink(missing_ok=True)
                        (args.output / "allocation.json").write_text(
                            json.dumps(
                                {
                                    "allocated_gpu_hours": (time.monotonic() - allocation_start)
                                    * 8
                                    / 3600,
                                    "active_gpu_hours": active_gpu_seconds / 3600,
                                },
                                indent=2,
                            )
                        )
    finally:
        stop_process(monitor)
    if args.mode == "benchmark":
        summary = args.output / "summary.jsonl"
        completed = len(summary.read_text().splitlines()) if summary.exists() else 0
        (args.output / "completion.json").write_text(
            json.dumps(
                {"completed_cases": completed, "expected_cases": 168, "complete": completed == 168},
                indent=2,
            )
        )
        if completed != 168:
            raise RuntimeError(f"Only {completed}/168 cases completed; inspect failure logs")


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--mode", choices=("serve", "benchmark"), default="benchmark")
    parser.add_argument("--layout", choices=("tp", "replicas"), default="tp")
    parser.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory must be new; refusing to mix benchmark runs")
    # SIGTERM cancels orchestration so finally blocks clean up process groups.
    with asyncio.Runner() as runner:
        loop = runner.get_loop()
        task = loop.create_task(orchestrate(args))
        loop.add_signal_handler(signal.SIGTERM, task.cancel)
        loop.add_signal_handler(signal.SIGUSR1, task.cancel)
        try:

            async def wait_task():
                await task

            runner.run(wait_task())
        except asyncio.CancelledError:
            raise SystemExit(143) from None


if __name__ == "__main__":
    main()
