"""Evaluate SwarmChat or a Hub model on SmolTalkBench with Hugging Face Jobs, one Job per revision.

`--model SwarmChat` serves commits of cmpatino/chatbot-swarm with swarmchat_server.py on a CPU
Job. A revision is named by its commit's time and hash, e.g. 20261028T162718Z-fdef14e9f8e3, so
revisions sort in merge order. A Hub repo ID is served with vLLM on a GPU Job, and with
`--backend endpoint` any Inspect model string is called from a CPU Job. Inspect logs and a
`summary.json` go to a Storage Bucket, in `<model>/<revision>/<task>/<timestamp>/`.

Usage:

uv run experiments/run4/eval/evaluate.py --model SwarmChat --revision main fdef14e9f8e3
uv run experiments/run4/eval/evaluate.py --model HuggingFaceTB/SmolLM-135M-Instruct
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

from huggingface_hub import HfApi, Volume, get_token, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from huggingface_hub.utils import disable_progress_bars

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = "experiments/run4/eval"
JOB_ROOT = "/workspace/flock"
VLLM_IMAGE = "vllm/vllm-openai:v0.25.1"
CPU_IMAGE = "ghcr.io/astral-sh/uv:python3.12-bookworm"
DEFAULT_BUCKET = "HuggingFaceSwarms/swarmchat-evals"
DEFAULT_GPU_FLAVOR = "rtx-pro-6000"
DEFAULT_CPU_FLAVOR = "cpu-basic"
SWARMCHAT = "SwarmChat"
SWARMCHAT_REPO = "cmpatino/chatbot-swarm"
TASKS = {"smoltalk_bench_gold": "smoltalk_bench.py@smoltalk_bench_gold"}
JUDGE = "hf-inference-providers/deepseek-ai/DeepSeek-V4.1-Flash:fireworks-ai"
# Prompt tokens reserved when deriving max_tokens from a Hub model's context length
PROMPT_MARGIN = 512
PROVIDER_API_KEYS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY")
JOB_ENV = {"HF_HUB_DISABLE_PROGRESS_BARS": "1", "TOKENIZERS_PARALLELISM": "false"}
VLLM_JOB_ENV = {"TORCH_COMPILE_DISABLE": "1", "VLLM_USE_AOT_COMPILE": "0",
                "VLLM_WORKER_MULTIPROC_METHOD": "spawn"}


def _key_value(value: str) -> tuple[str, str]:
    key, separator, argument_value = value.partition("=")
    if not key or not separator:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {value!r}")
    return key, argument_value


def _json_object(value: str) -> dict:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(f"expected a JSON object, got {value!r}")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        help=f"{SWARMCHAT}, a Hub repo ID or an Inspect model string")
    parser.add_argument("--backend", choices=["swarmchat", "vllm", "endpoint"],
                        help="default: swarmchat for SwarmChat, vllm otherwise")
    parser.add_argument("--revision", nargs="+", default=["main"],
                        help="chatbot-swarm commits for SwarmChat (one Job each), or the Hub "
                             "model revision for vllm")
    parser.add_argument("--task", choices=list(TASKS), default="smoltalk_bench_gold")
    parser.add_argument("--judge", default=JUDGE, help="Inspect model string of the judge")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Storage Bucket for the logs")
    parser.add_argument("--namespace", help="who runs the Jobs (default: the token's user)")
    parser.add_argument("--flavor", help=f"Job hardware (default: {DEFAULT_GPU_FLAVOR} for "
                                         f"vllm, {DEFAULT_CPU_FLAVOR} otherwise)")
    parser.add_argument("--timeout", default="6h", help="maximum Job duration")
    parser.add_argument("--limit", type=int, help="evaluate only the first N samples")
    parser.add_argument("--max_connections", type=int,
                        help="concurrent requests to the model (default: 4 for SwarmChat, "
                             "128 for vllm, 50 for endpoint)")
    parser.add_argument("--max_tokens", type=int,
                        help="default for vllm: the context length minus a prompt margin")
    parser.add_argument("--temperature", type=float, help="default for vllm: generation_config")
    parser.add_argument("--top_p", type=float, help="default for vllm: generation_config")
    parser.add_argument("--top_k", type=int, help="default for vllm: generation_config")
    parser.add_argument("--chat_template_kwargs", type=_json_object,
                        help="JSON object for extra_body.chat_template_kwargs")
    parser.add_argument("--model_base_url", help="base URL of an endpoint")
    parser.add_argument("--vllm_args", nargs="+", type=_key_value, default=[],
                        metavar="KEY=VALUE", help="overrides for `vllm serve`")
    parser.add_argument("--secrets", nargs="+", default=[], metavar="NAME",
                        help="extra local environment variables to forward")
    parser.add_argument("--dry_run", action="store_true",
                        help="print the Jobs without uploading or submitting")
    return parser.parse_args()


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def swarmchat_revision(sha: str, committed: datetime) -> str:
    """A commit's revision name, sortable by time: 20261028T162718Z-fdef14e9f8e3."""
    return f"{committed.astimezone(UTC):%Y%m%dT%H%M%SZ}-{sha[:12]}"


def log_model_id(model: str) -> str:
    """The model's folder in the bucket."""
    return model.removeprefix("hf-inference-providers/").split(":", 1)[0]


def _hub_json(repo_id: str, filename: str, revision: str) -> dict:
    try:
        return json.loads(Path(hf_hub_download(repo_id, filename, revision=revision)).read_text())
    except EntryNotFoundError:
        return {}


def context_length(repo_id: str, revision: str) -> int:
    config = _hub_json(repo_id, "config.json", revision)
    for config_object in (config, config.get("text_config") or {}):
        if max_position_embeddings := config_object.get("max_position_embeddings"):
            return int(max_position_embeddings)
    raise ValueError(f"no context length for {repo_id}; pass --vllm_args max_model_len=...")


def gpu_count(api: HfApi, flavor: str) -> int:
    hardware = next((h for h in api.list_jobs_hardware() if h.name == flavor), None)
    if hardware is None or hardware.accelerator is None:
        raise ValueError(f"--backend vllm needs a GPU flavor, got {flavor!r}")
    return int(hardware.accelerator.quantity)


def snapshot_volume(api: HfApi) -> Volume:
    """Uploads the files the Job needs to a content-addressed folder of the jobs-artifacts bucket."""
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
         "pyproject.toml", "uv.lock", EVAL_DIR],
        cwd=ROOT, check=True, capture_output=True)
    paths = sorted(p for p in completed.stdout.decode().split("\0") if (ROOT / p).is_file())
    digest = sha256()
    for path in paths:
        digest.update(path.encode())
        digest.update((ROOT / path).read_bytes())
    with tempfile.TemporaryDirectory() as staging_dir:
        for path in paths:
            destination = Path(staging_dir) / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / path, destination)
        return api.sync_job_volume(staging_dir, JOB_ROOT,
                                   remote_name=f"swarmchat-evals/{digest.hexdigest()[:16]}")


def main():
    args = parse_args()
    token = get_token()
    if token is None:
        raise RuntimeError("HF Jobs need a Hugging Face token; run `hf auth login` first")
    api = HfApi(token=token)
    namespace = args.namespace or api.whoami()["name"]
    backend = args.backend or ("swarmchat" if args.model == SWARMCHAT else "vllm")
    if (backend == "swarmchat") != (args.model == SWARMCHAT):
        raise ValueError(f"--backend swarmchat goes with --model {SWARMCHAT}")
    if backend != "swarmchat" and len(args.revision) > 1:
        raise ValueError("several revisions are only supported for SwarmChat")
    if backend == "endpoint" and args.revision != ["main"]:
        raise ValueError("--revision does not apply to --backend endpoint")
    flavor = args.flavor or (DEFAULT_GPU_FLAVOR if backend == "vllm" else DEFAULT_CPU_FLAVOR)
    sampling = {"temperature": args.temperature, "top_p": args.top_p, "top_k": args.top_k}
    max_tokens = args.max_tokens
    vllm_args = {}
    if backend == "vllm":
        revision = args.revision[0]
        user_vllm_args = dict(args.vllm_args)
        max_model_len = int(user_vllm_args.get("max_model_len")
                            or context_length(args.model, revision))
        tensor_parallel_size = int(user_vllm_args.get("tensor_parallel_size", 1))
        vllm_args = {"revision": revision, "max_model_len": max_model_len,
                     "data_parallel_size": gpu_count(api, flavor) // tensor_parallel_size,
                     "disable_uvicorn_access_log": "true", **user_vllm_args}
        generation_config = _hub_json(args.model, "generation_config.json", revision)
        sampling = {k: v if v is not None else generation_config.get(k)
                    for k, v in sampling.items()}
        if max_tokens is None:
            max_tokens = max_model_len - PROMPT_MARGIN
    max_connections = args.max_connections or {"swarmchat": 4, "vllm": 128,
                                               "endpoint": 50}[backend]

    secrets = {"HF_TOKEN": token}
    for name in args.secrets:
        if name not in os.environ:
            raise ValueError(f"--secrets {name} is not set in the local environment")
    for name in (*PROVIDER_API_KEYS, *args.secrets):
        if name in os.environ:
            secrets[name] = os.environ[name]

    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]
    jobs = []
    for requested in args.revision:
        commit = None
        if backend == "swarmchat":
            info = api.dataset_info(SWARMCHAT_REPO, revision=requested)
            commit, revision = info.sha, swarmchat_revision(info.sha, info.last_modified)
        else:
            revision = requested if backend == "vllm" else "main"
        model_id = SWARMCHAT if backend == "swarmchat" else log_model_id(args.model)
        metadata = {
            "model_id": model_id, "revision": revision, "backend": backend,
            "commit": commit, "commit_url": commit and
            f"https://huggingface.co/datasets/{SWARMCHAT_REPO}/tree/{commit}",
            "judge": args.judge, **sampling, "max_tokens": max_tokens,
            "chat_template_kwargs": args.chat_template_kwargs,
        }
        eval_kwargs = {
            "limit": args.limit, "max_connections": max_connections, "max_tokens": max_tokens,
            **sampling,
            "extra_body": {"chat_template_kwargs": args.chat_template_kwargs}
            if args.chat_template_kwargs else None,
            "model_base_url": args.model_base_url,
            "model_roles": {"grader": args.judge},
            "max_retries": 10, "timeout": 600,
            # Score samples that still error after a retry so metrics cover the whole dataset
            "retry_on_error": 1, "fail_on_error": False,
            "metadata": metadata,
        }
        eval_kwargs = {k: v for k, v in eval_kwargs.items() if v is not None}
        output_path = f"{model_id}/{revision}/{args.task}/{timestamp}"
        command = ["python3", f"{JOB_ROOT}/{EVAL_DIR}/job.py", "--model", args.model,
                   "--backend", backend, "--task", TASKS[args.task],
                   "--output_dir", f"hf://buckets/{args.bucket}/{output_path}",
                   "--eval_kwargs", json.dumps(eval_kwargs)]
        if commit:
            command += ["--revision", commit, "--workers", str(max_connections)]
        if vllm_args:
            command += ["--vllm_args", *(f"{k}={v}" for k, v in vllm_args.items())]
        jobs.append({
            "name": _safe_name(f"{args.task}-{model_id}-{revision}"),
            "image": VLLM_IMAGE if backend == "vllm" else CPU_IMAGE,
            "command": command,
            "env": JOB_ENV | (VLLM_JOB_ENV if backend == "vllm" else {}),
            "flavor": flavor, "timeout": args.timeout, "namespace": namespace,
            "output_url": f"https://huggingface.co/buckets/{args.bucket}/tree/{output_path}",
        })

    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return

    api.create_bucket(args.bucket, private=True, exist_ok=True)
    disable_progress_bars()
    volume = snapshot_volume(api)
    for job in jobs:
        output_url = job.pop("output_url")
        job_info = api.run_job(**job, secrets=secrets, volumes=[volume])
        print(f"{job['name']}\n  Job: {job_info.url}\n  Logs: {output_url}")


if __name__ == "__main__":
    main()
