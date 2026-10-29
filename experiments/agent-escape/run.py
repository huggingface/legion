"""Run one bounded attempt, retaining trajectories and a normal Flock run bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REVISION = "03742c79f48646e0130886e18e3e79798cf8edd7"
DIFFICULTIES = (5, 10, 15, 20, 25)


def select_case(source: Path, difficulty: int, scenario_index: int) -> Path:
    """Resolve a zero-based index in the pinned benchmark's sorted narrative files."""
    if difficulty not in DIFFICULTIES:
        raise ValueError(f"Unsupported difficulty {difficulty}; choose from {DIFFICULTIES}")
    directory = source / "questions" / str(difficulty) / "narrative"
    if not directory.is_dir():
        raise ValueError(f"Scenario directory is missing: {directory}; run setup.sh first")
    cases = sorted(path for path in directory.glob("*.json") if path.is_file())
    if not cases:
        raise ValueError(f"No scenario JSON files found in {directory}")
    if not 0 <= scenario_index < len(cases):
        raise ValueError(
            f"Scenario index {scenario_index} is invalid for difficulty {difficulty}; "
            f"valid range is 0–{len(cases) - 1} ({len(cases)} scenarios)"
        )
    return cases[scenario_index]


def stop(process, grace=120):
    if process is None or process.poll() is not None:
        return
    process.send_signal(signal.SIGINT)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise RuntimeError("Process did not shut down gracefully; inspect logs and mount state")


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=HERE / "spec.yaml")
    parser.add_argument("--timeout", type=float, default=1800, help="Attempt limit in seconds")
    parser.add_argument("--difficulty", type=int, choices=DIFFICULTIES, default=10,
                        help="Puzzle node count (default: 10)")
    parser.add_argument("--scenario-index", type=int, default=0,
                        help="Zero-based index in sorted scenario filenames (default: 0)")
    parser.add_argument("--agents", type=int, default=None)
    parser.add_argument("--trial", type=int, default=1)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    source = ROOT / ".flock/deps/agent-escape"
    try:
        case_file = select_case(source, args.difficulty, args.scenario_index)
    except ValueError as exc:
        parser.error(str(exc))
    selection = {
        "difficulty": args.difficulty,
        "scenario_index": args.scenario_index,
        "case": case_file.relative_to(source).as_posix(),
        "case_sha256": hashlib.sha256(case_file.read_bytes()).hexdigest(),
        "benchmark_revision": REVISION,
    }
    python = ROOT / ".flock/deps/agent-escape-venv/bin/python"
    if not python.exists():
        parser.error("Run bash experiments/agent-escape/setup.sh first")
    revision = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"], text=True)
    if revision != REVISION or dirty:
        parser.error("Benchmark checkout must be clean and pinned to " + REVISION)
    run_id = uuid.uuid4().hex
    output = ROOT / ".flock/agent-escape-attempts" / run_id
    output.mkdir(parents=True, mode=0o700)
    spec_path = args.spec.resolve()
    spec = yaml.safe_load(spec_path.read_text())
    workers = spec.get("types", {}).get("worker", {})
    agents = args.agents if args.agents is not None else workers.get("replicas", 4)
    if not 1 <= agents <= 32:
        parser.error("Agent count must be between 1 and 32")
    workers["replicas"] = agents
    for provider in spec.get("providers", {}).values():
        provider["max_working"] = agents
    stamp = time.strftime("%Y%m%dt%H%M%Sz", time.gmtime())
    spec["swarm"] = (f"escape-{workers['harness']}-qwen27b-d{args.difficulty}-s{args.scenario_index:02d}"
                     f"-n{agents:02d}-r{args.trial:02d}-{stamp}-{run_id[:8]}")
    spec["comms"]["kickoff"] = (f"{agents} workers have independent copies of the same puzzle. "
                                "Share exact discoveries and inputs in general.")
    if (
        set(spec.get("types", {})) != {"worker"}
        or workers.get("replicas", 0) < 1
        or workers.get("agents_per_worker") != 1
    ):
        parser.error("This experiment requires worker replicas, one agent per worker")
    shutil.copyfile(HERE / "world.py", output / "world.py")
    spec["world"]["dir"] = str(output)
    for name, definition in spec["types"].items():
        prompt = (spec_path.parent / definition["system_prompt_file"]).resolve()
        copy = output / f"{name}-prompt.md"
        shutil.copyfile(prompt, copy)
        definition["system_prompt_file"] = str(copy)
    snapshot = output / "spec.yaml"
    snapshot.write_text(yaml.safe_dump(spec, sort_keys=False))
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(ROOT / ".flock/tools"), str(ROOT / ".venv/bin"), env.get("PATH", "")]
    )
    env.setdefault("LOCAL_KEY", "unused")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    url = f"http://127.0.0.1:{port}"
    env.update(
        ESCAPE_AGENTS=str(agents),
        ESCAPE_SERVER_URL=url,
        ESCAPE_CASE_FILE=str(case_file),
        ESCAPE_OUTPUT=str(output),
        ESCAPE_RUN_ID=run_id,
    )
    proxy_url = upstream = None
    if workers["harness"] == "pi":
        with socket.socket() as proxy_socket:
            proxy_socket.bind(("127.0.0.1", 0))
            proxy_port = proxy_socket.getsockname()[1]
        proxy_url = f"http://127.0.0.1:{proxy_port}"
        upstream = spec["providers"]["local"]["base_url"]
        spec["providers"]["local"]["base_url"] = proxy_url + "/v1"
        snapshot.write_text(yaml.safe_dump(spec, sort_keys=False))
    metadata = {
        "run_id": run_id,
        "agents": agents,
        "trial": args.trial,
        "harness": workers["harness"],
        "upstream": upstream,
        **selection,
        "world_sha256": hashlib.sha256((output / "world.py").read_bytes()).hexdigest(),
        "launcher_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started": time.time(),
        "timeout": args.timeout,
        "spec": spec,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Selected difficulty {args.difficulty}, scenario index {args.scenario_index}: "
          f"{case_file.name}", flush=True)
    print(f"Attempt artifacts: {output}", flush=True)
    backend = flock = proxy = None
    reason = "error"
    failure = None
    started = time.monotonic()
    try:
        if proxy_url:
            with (output / "usage-proxy.log").open("w") as log:
                proxy = subprocess.Popen([str(ROOT / ".venv/bin/python"),
                    str(HERE / "usage_proxy.py"), "--port", str(proxy_port), "--agents", str(agents),
                    "--upstream", upstream, "--output", str(output)], cwd=ROOT,
                    env={k: env[k] for k in ["PATH", "HOME", "LANG", "LOCAL_KEY"] if k in env},
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            for _ in range(100):
                try:
                    if httpx.get(proxy_url + "/health", timeout=1).is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("Usage proxy startup failed")
        # The benchmark needs no model or bucket credentials.
        backend_env = {key: env[key] for key in ("PATH", "HOME", "LANG") if key in env}
        with (output / "benchmark.log").open("w") as log:
            backend = subprocess.Popen(
                [
                    str(python),
                    "-m",
                    "uvicorn",
                    "server:app",
                    "--app-dir",
                    str(source),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                ],
                cwd=output,
                env=backend_env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        with httpx.Client(timeout=2) as client:
            for _ in range(100):
                if backend.poll() is not None:
                    raise RuntimeError("Benchmark failed to start; see benchmark.log")
                try:
                    if client.get(url + "/health").is_success:
                        break
                except httpx.HTTPError:
                    pass
                time.sleep(0.1)
            else:
                raise RuntimeError("Benchmark startup timed out")
        with (output / "flock.log").open("w") as log:
            flock = subprocess.Popen(
                [
                    str(ROOT / ".venv/bin/flock"),
                    "run",
                    str(snapshot),
                    *(["--skip-preflight"] if proxy_url else []),
                    "--data-dir",
                    str(output / "flock"),
                    "--until",
                    "idle",
                    "--prompt",
                    (
                        "First post a short introduction on general in thread 1 and read(n=12). "
                        "Remember the highest message ID read. Then get_observation and claim an "
                        "unclaimed branch. Every three puzzle actions, read(n=12, since=LAST_READ_ID) "
                        "and update the cursor from returned messages only. Share only new verified "
                        "findings, failed inputs, or requests in thread 1, at most 500 characters. "
                        "Acknowledge teammate message IDs when reusing findings. Submit the victory "
                        "string with submit_answer; do not stop before verified success or limits."
                    ),
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while True:
            if (output / "success.json").exists():
                reason = "verified_success"
                break
            if backend.poll() is not None:
                raise RuntimeError("Benchmark exited during attempt")
            if flock.poll() is not None:
                reason = "idle" if flock.returncode == 0 else "flock_error"
                break
            if time.monotonic() - started >= args.timeout:
                reason = "timeout"
                break
            time.sleep(1)
    except KeyboardInterrupt:
        reason = "interrupted"
    except (OSError, RuntimeError, httpx.HTTPError) as exc:
        failure = str(exc)
    finally:
        if proxy_url and proxy and proxy.poll() is None:
            try:
                with httpx.Client(timeout=5) as client:
                    client.post(proxy_url + "/close").raise_for_status()
                    deadline = time.monotonic() + 180
                    while client.get(proxy_url + "/health").json()["active"]:
                        if time.monotonic() >= deadline:
                            failure = "Model calls did not drain; accounting may be incomplete"
                            break
                        time.sleep(0.5)
                    time.sleep(1)  # let Pi consume the final usage records
            except httpx.HTTPError:
                failure = "Usage proxy drain failed"
        for process in (flock, backend, proxy):
            try:
                stop(process)
            except (OSError, RuntimeError, httpx.HTTPError) as exc:
                failure = str(exc)
        events = []
        trajectory = output / "trajectory.jsonl"
        if trajectory.exists():
            events = [json.loads(line) for line in trajectory.read_text().splitlines()]
        success = output / "success.json"
        report = {
            **selection,
            "agents": agents,
            "trial": args.trial,
            "harness": workers["harness"],
            "reason": reason,
            "success": json.loads(success.read_text()) if success.exists() else None,
            "error": failure,
            "elapsed_seconds": time.monotonic() - started,
            "flock_exit_code": flock.returncode if flock else None,
            "actions_by_agent": dict(
                Counter(e["agent"] for e in events if e["action"] != "initialize")
            ),
            "actions_by_type": dict(Counter(e["action"] for e in events)),
        }
        if proxy_url:
            from usage_proxy import summarize
            report["accounting"] = summarize(output / "model-requests.jsonl")
            if not report["accounting"]["complete"]:
                report["measurement_error"] = "Incomplete upstream token accounting"
        bundles = list((output / "flock").glob("*/run.json"))
        if len(bundles) == 1:
            bundle = json.loads(bundles[0].read_text())
            report["flock_bundle"] = str(bundles[0].relative_to(output))
            report["totals"] = bundle.get("totals", {})
            board_path = bundles[0].parent / "board/metadata.json"
            if board_path.exists():
                board = json.loads(board_path.read_text())
                report["board"] = {
                    "location": board.get("data_dir"),
                    "verified_at": board.get("storage", {}).get("verified_at"),
                }
        if report["success"]:
            report["time_to_success_seconds"] = report["success"]["time"] - metadata["started"]
        (output / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    return 0 if report["success"] and not failure and report["flock_exit_code"] == 0 and not report.get("measurement_error") else 1


if __name__ == "__main__":
    sys.exit(main())
