"""One CPU Slurm allocation runs bounded AgentEscape attempts sequentially."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import httpx
import yaml

from flock.models import SwarmSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


def load_launcher():
    spec = importlib.util.spec_from_file_location("agent_escape_run", HERE / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    launcher = load_launcher()
    for difficulty in args.difficulties:
        launcher.select_case(ROOT / ".flock/deps/agent-escape", difficulty, args.scenario_index)
    if args.timeout <= 0:
        raise ValueError("Timeout must be positive")
    endpoint = read_json(args.server_run / "endpoint.json")
    key = (args.server_run / "api-key").read_text().strip()
    if not key or not os.environ.get("HF_TOKEN"):
        raise ValueError("Server api-key and HF_TOKEN are required")
    spec = yaml.safe_load(args.spec.read_text())
    spec["providers"]["local"]["base_url"] = endpoint["base_url"]
    spec["models"]["qwen-27b"]["model"] = "local:" + endpoint["model"]
    for definition in spec["types"].values():
        definition["system_prompt_file"] = str(
            (args.spec.parent / definition["system_prompt_file"]).resolve())
    spec["comms"]["storage"]["mount_root"] = f"/tmp/flock-boards-{os.getuid()}"
    SwarmSpec.model_validate(spec)
    if not 1 <= getattr(args, "agents", 4) <= 32:
        raise ValueError("Agent count must be between 1 and 32")
    output = args.output.resolve() if args.output else ROOT / ".flock/agent-escape-slurm" / (
        datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    (output / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    write_json(output / "secrets.json", {"HF_TOKEN": os.environ["HF_TOKEN"], "LOCAL_KEY": key})
    sources = [HERE / name for name in ["run.py", "world.py", "prompt.md", "slurm.py", "run.slurm", "usage_proxy.py", "prompt-pi.md"]]
    sources += [ROOT / "src/flock/worker/adapters" / name
                for name in ("pi.py", "pi_tools.mjs")]
    config = {"output": str(output), "spec": str(output / "spec.yaml"),
              "endpoint": endpoint, "difficulties": args.difficulties,
              "scenario_index": args.scenario_index, "timeout": args.timeout,
              "agents": getattr(args, "agents", 4), "trial": getattr(args, "trial", 1),
              "secrets_file": str(output / "secrets.json"),
              "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=ROOT, text=True).strip()}
    write_json(output / "config.json", config)
    return output


def submit(args):
    output = prepare(args)
    command = ["sbatch", "--parsable", "--export=NONE", "--partition", args.partition,
               "--cpus-per-task", str(args.cpus), "--mem", args.mem, "--time", args.time,
               "--output", str(output / "slurm-%j.log"), str(HERE / "run.slurm"),
               str(output / "config.json")]
    if args.dry_run:
        print(json.dumps({"command": command, "output": str(output)}, indent=2))
        (output / "secrets.json").unlink()
        return 0
    # No credentials in sbatch's environment, arguments, config, or job metadata.
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "USER", "LOGNAME",
           "LANG", "TZ", "SHELL"} or k.startswith("LC_")}
    try:
        job = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(";")[0]
    except BaseException:
        (output / "secrets.json").unlink(missing_ok=True)
        raise
    write_json(output / "submission.json", {"job_id": job, "command": command})
    print(json.dumps({"job_id": job, "output": str(output)}, indent=2))
    return 0


def read_secrets(path):
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Secrets must be an owner-only regular file")
    return read_json(path)


def sample_metrics(client, instances):
    def one(instance):
        try:
            response = client.get(instance["url"] + "/metrics", timeout=5)
            response.raise_for_status()
            return {"url": instance["url"], "metrics": response.text}
        except httpx.HTTPError as exc:
            return {"url": instance["url"], "error": type(exc).__name__}
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(one, instances))


def run(config_path):
    config = read_json(config_path)
    output = Path(config["output"])
    process = None
    stopping = False
    stop_at = None
    results = []

    def stop(*_):
        nonlocal stopping, stop_at
        if stopping:
            return
        stopping = True
        stop_at = time.monotonic()
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGINT)  # run.py drains Flock and verifies the board

    def wait_for_stop():
        if process is None or process.poll() is not None:
            return
        remaining = max(0, 180 - (time.monotonic() - stop_at)) if stop_at is not None else 180
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        secrets = read_secrets(config["secrets_file"])
        for filename, expected in config["source_sha256"].items():
            if hashlib.sha256(Path(filename).read_bytes()).hexdigest() != expected:
                raise ValueError("Experiment source changed after submission; submit a fresh job")
        env = {**os.environ, **secrets, "HF_HUB_DISABLE_PROGRESS_BARS": "1",
               "PYTHONUNBUFFERED": "1"}
        env["PATH"] = os.pathsep.join([str(ROOT / ".flock/tools"), str(ROOT / ".venv/bin"),
                                      env.get("PATH", "/usr/bin:/bin")])
        write_json(output / "runtime.json", {"job_id": os.getenv("SLURM_JOB_ID"),
                   "hostname": socket.gethostname(), "started": time.time()})
        with httpx.Client(headers={"Authorization": "Bearer " + secrets["LOCAL_KEY"]},
                          timeout=15) as client:
            response = client.get(config["endpoint"]["base_url"] + "/models")
            response.raise_for_status()
            if config["endpoint"]["model"] not in [m["id"] for m in response.json()["data"]]:
                raise ValueError("Router does not serve the configured model")
            for index, difficulty in enumerate(config["difficulties"]):
                if stopping:
                    break
                label = f"{index:02d}-d{difficulty}"
                command = [sys.executable, str(HERE / "run.py"), "--spec", config["spec"],
                           "--difficulty", str(difficulty), "--scenario-index",
                           str(config["scenario_index"]), "--timeout", str(config["timeout"]),
                           "--agents", str(config.get("agents", 4)),
                           "--trial", str(config.get("trial", 1))]
                attempt = None
                print(f"START difficulty={difficulty} scenario={config['scenario_index']}", flush=True)
                with (output / f"{label}.log").open("w") as log, (
                        output / f"{label}-metrics.jsonl").open("w") as metrics:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, start_new_session=True)
                    deadline = time.monotonic() + config["timeout"] + 240
                    while True:
                        metrics.write(json.dumps({"time": time.time(), "backends":
                            sample_metrics(client, config["endpoint"].get("instances", []))}) + "\n")
                        metrics.flush()
                        # run.py prints its exact unique path before any services start.
                        if attempt is None:
                            for line in (output / f"{label}.log").read_text().splitlines():
                                if line.startswith("Attempt artifacts: "):
                                    attempt = Path(line.split(": ", 1)[1])
                                    write_json(output / f"{label}-attempt.json", {"path": str(attempt)})
                        if process.poll() is not None:
                            break
                        if time.monotonic() > deadline or (
                                stop_at is not None and time.monotonic() - stop_at > 180):
                            stop()
                            wait_for_stop()
                            break
                        try:
                            process.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            pass
                # Never guess an attempt from other concurrent experiments.
                report = read_json(attempt / "result.json") if attempt and (
                    attempt / "result.json").is_file() else {"error": "No completed attempt report"}
                results.append({"difficulty": difficulty, "scenario_index": config["scenario_index"],
                                "attempt": str(attempt) if attempt else None,
                                "exit_code": process.returncode, "result": report})
                write_json(output / "summary.json", results)
                print(f"FINISHED difficulty={difficulty} exit={process.returncode} "
                      f"reason={report.get('reason')}", flush=True)
                process = None
    except Exception as exc:  # noqa: BLE001 — never log credentials from HTTP exceptions
        write_json(output / "error.json", {"error_type": type(exc).__name__})
        print(f"Launcher failed: {type(exc).__name__}; inspect per-attempt logs", flush=True)
        stop()
        wait_for_stop()
        return 1
    finally:
        Path(config["secrets_file"]).unlink(missing_ok=True)
        write_json(output / "summary.json", results)
    return 0 if not stopping and len(results) == len(config["difficulties"]) and all(
        r["exit_code"] == 0 for r in results) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--server-run", type=Path, required=True)
    s.add_argument("--spec", type=Path, default=HERE / "spec-router-qwen27b.yaml")
    s.add_argument("--difficulties", type=int, nargs="+", choices=[5, 10, 15, 20, 25], default=[5, 10])
    s.add_argument("--scenario-index", type=int, default=0)
    s.add_argument("--agents", type=int, default=4)
    s.add_argument("--trial", type=int, default=1)
    s.add_argument("--timeout", type=int, default=900)
    s.add_argument("--partition", default="hopper-cpu")
    s.add_argument("--cpus", type=int, default=8)
    s.add_argument("--mem", default="16G")
    s.add_argument("--time", default="01:00:00")
    s.add_argument("--output", type=Path)
    s.add_argument("--dry-run", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("config", type=Path)
    args = parser.parse_args()
    return submit(args) if args.mode == "submit" else run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())
