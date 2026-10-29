"""Run a SmolTalkBench task inside a Hugging Face Job launched by evaluate.py.

Installs the `evals` dependency group, serves the model (SwarmChat with swarmchat_server.py, a
Hub model with vLLM), runs the Inspect task, writes `summary.json` beside the log and uploads
the folder to the output bucket. Adapted from Carbon-X's evaluation/job.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

MOUNTED_ROOT = Path("/workspace/flock")
RUNTIME_ROOT = Path("/tmp/flock")
EVAL_DIR = RUNTIME_ROOT / "experiments" / "run4" / "eval"
VENV = Path("/tmp/evals-venv")
RUNTIME_ENV_VAR = "SMOLTALK_EVAL_RUNTIME"
LOG_DIR = Path("/tmp/inspect-logs")
PORT = 8000
STARTUP_TIMEOUT = 900
SWARMCHAT_REPO = "cmpatino/chatbot-swarm"
SCORER, PRIMARY_METRIC = "alpaca_eval_judge", "length_controlled_win_rate"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True,
                        help="SwarmChat, a Hub repo ID for vllm or an Inspect model string")
    parser.add_argument("--backend", choices=["swarmchat", "vllm", "endpoint"], required=True)
    parser.add_argument("--revision", help="chatbot-swarm commit for swarmchat")
    parser.add_argument("--task", required=True, help="Inspect task, e.g. smoltalk_bench.py@...")
    parser.add_argument("--output_dir", required=True, help="hf://buckets/... folder")
    parser.add_argument("--eval_kwargs", type=json.loads, required=True,
                        help="JSON keyword arguments for eval()")
    parser.add_argument("--workers", type=int, default=4, help="SwarmChat conversations at once")
    parser.add_argument("--vllm_args", nargs="+", default=[], metavar="KEY=VALUE")
    return parser.parse_args()


def bootstrap() -> None:
    """Installs the evals dependency group into a venv and re-executes this script with it."""
    # Copy out of the read-only mount so uv can write next to the project
    shutil.copytree(MOUNTED_ROOT, RUNTIME_ROOT)
    env = os.environ.copy()
    env[RUNTIME_ENV_VAR] = "1"
    env.pop("VIRTUAL_ENV", None)
    # A venv of its own: vLLM keeps the image's Python and packages
    subprocess.run(["uv", "sync", "--frozen", "--only-group", "evals"], cwd=RUNTIME_ROOT,
                   env=env | {"UV_PROJECT_ENVIRONMENT": str(VENV)}, check=True)
    python = str(VENV / "bin" / "python")
    os.execve(python, [python, str(EVAL_DIR / "job.py"), *sys.argv[1:]], env)


def wait_healthy(server: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"the model server exited with status {server.returncode}")
        try:
            with urlopen(url, timeout=2) as response:
                if response.status == 200:
                    return
        except (TimeoutError, URLError, ConnectionError):
            pass
        time.sleep(5)
    stop(server)
    raise TimeoutError(f"the model server was not healthy within {STARTUP_TIMEOUT} s")


def stop(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def start_swarmchat(revision: str, workers: int) -> subprocess.Popen:
    from huggingface_hub import snapshot_download

    tree = snapshot_download(SWARMCHAT_REPO, repo_type="dataset", revision=revision,
                             allow_patterns=["bot/**"], local_dir="/tmp/chatbot-swarm")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen(
        [sys.executable, str(EVAL_DIR / "swarmchat_server.py"), tree, "--port", str(PORT),
         "--workers", str(workers), "--log", str(LOG_DIR / "swarmchat_server.log")],
        start_new_session=True)
    wait_healthy(server, f"http://127.0.0.1:{PORT}/health")
    # Inspect's openai-api/swarmchat provider reads these
    os.environ["SWARMCHAT_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    os.environ["SWARMCHAT_API_KEY"] = "unused"
    return server


def start_vllm(repo_id: str, vllm_args: list[str]) -> subprocess.Popen:
    from huggingface_hub import snapshot_download

    arguments = dict(arg.split("=", 1) for arg in vllm_args)
    # Download first so the transfer does not count against the startup timeout
    snapshot_download(repo_id, revision=arguments.get("revision"))
    command = ["vllm", "serve", repo_id]
    for key, value in arguments.items():
        flag = key.replace("_", "-")
        if value == "true":
            command.append(f"--{flag}")
        elif value == "false":
            command.append(f"--no-{flag}")
        else:
            command.extend([f"--{flag}", value])
    command.extend(["--host", "127.0.0.1", "--port", str(PORT)])
    server = subprocess.Popen(command, start_new_session=True)
    wait_healthy(server, f"http://127.0.0.1:{PORT}/health")
    # Inspect's vLLM provider connects to the server through VLLM_BASE_URL
    os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    return server


def _json_value(value):
    return None if isinstance(value, float) and not math.isfinite(value) else value


def build_summary(log) -> dict:
    """The primary metric, all metrics and sample counts of an Inspect log."""
    samples = log.samples or []
    scores = [s.scores.get(SCORER) for s in samples if s.scores]
    judged = [s for s in scores if s is not None and isinstance(s.value, float)
              and not math.isnan(s.value)]
    metadata = dict(log.eval.metadata or {})
    metadata.update({
        "task": log.eval.task,
        "task_args": dict(log.eval.task_args),
        "model": log.eval.model,
        "model_roles": {role: [c.model for c in config] if isinstance(config, list)
                        else config.model for role, config in (log.eval.model_roles or {}).items()},
        "started_at": log.stats.started_at,
        "completed_at": log.stats.completed_at,
        "status": log.status,
        "total_samples": log.results.total_samples,
        "completed_samples": log.results.completed_samples,
        "judged_samples": len(judged),
        "empty_outputs": sum(not s.output.completion for s in samples),
        "stop_reasons": dict(Counter(s.output.stop_reason for s in samples).most_common()),
        "error_samples": sum(s.error is not None for s in samples),
    })
    metrics = {m.name: _json_value(m.value) for score in log.results.scores
               if score.name == SCORER for m in score.metrics.values()}
    return {"metadata": metadata,
            "primary_metric": {"name": PRIMARY_METRIC, "value": metrics.get(PRIMARY_METRIC)},
            "metrics": metrics}


def main():
    if os.environ.get(RUNTIME_ENV_VAR) != "1":
        bootstrap()

    from huggingface_hub import HfApi
    from inspect_ai import eval as inspect_eval
    from inspect_ai.log import read_eval_log

    args = parse_args()
    os.chdir(RUNTIME_ROOT)
    server = None
    status = None
    try:
        model = args.model
        if args.backend == "swarmchat":
            server = start_swarmchat(args.revision, args.workers)
            model = "openai-api/swarmchat/SwarmChat"
        elif args.backend == "vllm":
            server = start_vllm(args.model, args.vllm_args)
            model = f"vllm/{args.model}"
        log = inspect_eval(tasks=str(EVAL_DIR / args.task), model=model, log_dir=str(LOG_DIR),
                           display="plain", **args.eval_kwargs)[0]
        if server is not None and server.poll() is not None:
            raise RuntimeError(f"the model server exited during the eval ({server.returncode})")
        status = log.status
        if status == "success":
            summary = build_summary(read_eval_log(log.location))
            (LOG_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"{log.eval.task}: {PRIMARY_METRIC} = {summary['primary_metric']['value']}")
    finally:
        if server is not None:
            stop(server)
        # Keep empty files, such as a server log nothing was written to, out of the bucket
        for path in LOG_DIR.glob("*"):
            if path.is_file() and path.stat().st_size == 0:
                path.unlink()
        if LOG_DIR.is_dir() and any(LOG_DIR.iterdir()):
            HfApi().sync_bucket(str(LOG_DIR), args.output_dir)
            print(f"Uploaded the Inspect log to {args.output_dir}")
    if status != "success":
        sys.exit(f"Inspect evaluation finished with status {status!r}")


if __name__ == "__main__":
    main()
