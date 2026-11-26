"""Prepare, launch, and supervise a frozen SFT collection; owns only its submitted jobs."""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from task_splits import REPO, TASK_REVISION, select_tasks, verify_task

REGISTRY = "94f1508ff870559cbacc53fa469ab3395c2a48d9"
MODEL = "Qwen/Qwen3.5-27B"
MODEL_REVISION = "fc05daec18b0a78c049392ed2e771dde82bdf654"
PY = sys.executable

TERMINAL = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def prepare(root, secrets, bucket, network_helpers, agents=16):
    from huggingface_hub import snapshot_download

    if agents < 1:
        raise ValueError("agents must be positive")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if (root / "study.json").exists() or (root / "jobs.json").exists():
        raise ValueError("Run is already prepared; choose a new directory")
    source = Path(__file__).resolve().parent
    for name in ("registries.conf", "compose_compat.py", "legion-run", "resources/capture.js"):
        if not (source / name).is_file():
            raise ValueError(f"Missing runtime asset: {name}")
    for p in source.iterdir():
        if p.is_file():
            shutil.copy2(p, root / p.name)
    shutil.copytree(
        source / "resources",
        root / "resources",
        ignore=shutil.ignore_patterns("__pycache__"),
        dirs_exist_ok=True,
    )
    snapshot = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=REGISTRY,
            allow_patterns=["registry.json", "splits/v1.json"],
        )
    )
    shutil.copy2(snapshot / "registry.json", root / "registry.json")
    shutil.copy2(snapshot / "splits/v1.json", root / "splits.json")
    manifest = read(root / "splits.json")
    selected = select_tasks(
        manifest, read(root / "registry.json"), "sft", [5, 10, 15, 20, 25], "evaluate"
    )
    tasks = Path(
        snapshot_download(
            REPO,
            repo_type="dataset",
            revision=TASK_REVISION,
            allow_patterns=[r["task_path"] + "/*" for r in selected],
        )
    )
    for row in selected:
        verify_task(tasks / row["task_path"], row)
    model = Path(
        snapshot_download(
            MODEL,
            revision=MODEL_REVISION,
            local_files_only=True,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.model", "*.txt"],
        )
    )
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ").lower()
    rows = []
    for i, r in enumerate(selected):
        rows.append(
            {
                **r,
                "id": r["scenario_id"],
                "index": i,
                "scenario_index": i // 5,
                "agents": agents,
                "prefix": f"sft35-{stamp}-{i:03d}",
            }
        )
    # Deterministic scenario order is inherited from the frozen split helper.
    smokes = {}
    for level in [5, 10, 15, 20, 25]:
        row = next(r for r in rows if r["difficulty"] == level)
        smokes["smoke" + str(level)] = {
            **row,
            "index": "smoke" + str(level),
            "prefix": f"sft35-{stamp}-pilot{level}",
            "smoke": True,
        }
    smokes["smoke2"] = {
        **rows[0],
        "index": "smoke2",
        "prefix": f"sft35-{stamp}-capture2",
        "agents": 2,
        "smoke": True,
        "startup_only": True,
        "work_seconds": 120,
    }
    save(root / "manifest.json", rows)
    save(root / "smokes.json", smokes)
    study = {
        "model": MODEL,
        "model_revision": MODEL_REVISION,
        "model_snapshot": str(model),
        "dataset": REPO,
        "revision": REGISTRY,
        "task_snapshot": str(tasks),
        "bucket": bucket,
        "repo": str(source.parents[1]),
        "network_helpers": str(network_helpers.resolve()),
        "export_python": os.environ.get("SFT_EXPORT_PYTHON", PY),
        "template_secrets": str(Path(secrets).resolve()),
        "key_file": str(root / "server/api-key"),
        "concurrency": 8,
        "agents": agents,
        "split": "sft",
        "seed": 42,
    }
    save(root / "study.json", study)
    (root / "server").mkdir()
    (root / "launch.slurm").write_text(f"""#!/bin/bash
#SBATCH -p hopper-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1900M
#SBATCH --time=06:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
set -eu
umask 077
exec {PY} {root}/run_array.py "${{1:-$SLURM_ARRAY_TASK_ID}}"
""")
    (root / "supervisor.slurm").write_text(f"""#!/bin/bash
#SBATCH -p hopper-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
set -eu
umask 077
exec {PY} {root}/pipeline.py supervise {root}
""")
    save(
        root / "source-hashes.json",
        {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*")
            if p.is_file()
        },
    )
    return root


def submit(root, resume=False):
    root = Path(root).resolve()
    study = read(root / "study.json")
    if resume:
        previous = read(root / "jobs.json")
        identifiers = [previous["router"], previous["supervisor"], *previous["children"]]
        previous_states = states(identifiers)
        if any(previous_states.get(job) not in TERMINAL for job in identifiers):
            raise ValueError("Resume requires all previous jobs to be terminal")
        if any(root.glob("*/reservation.json")):
            raise ValueError("Unreleased attempt reservation; inspect cleanup before resuming")
        # This recovery path reuses completed pilots, never resamples them.
        for row in read(root / "smokes.json").values():
            gate(root, row)
        archive = root / "history" / datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
        archive.mkdir(parents=True)
        for name in ("server", "jobs.json", "final-status.json", "progress.json", "supervisor.log"):
            path = root / name
            if path.exists():
                path.rename(archive / name)
        (root / "server").mkdir()
    elif (root / "jobs.json").exists():
        raise ValueError("Already submitted; do not duplicate this collection")
    env = {**os.environ, "QWEN_LAUNCHER": str(root / "qwen_serve.py"), "MAX_MODEL_LEN": "65536"}
    router = (
        subprocess.check_output(
            [
                "sbatch",
                "--parsable",
                "--job-name=sft35-router",
                "-o",
                str(root / "server/slurm.log"),
                str(root / "serve.slurm"),
                study["model_snapshot"],
                str(root / "server"),
            ],
            env=env,
            text=True,
        )
        .strip()
        .split(";")[0]
    )
    save(root / "jobs.json", {"router": router, "children": []})
    try:
        supervisor = (
            subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--job-name=sft35-supervisor",
                    "-o",
                    str(root / "supervisor.log"),
                    str(root / "supervisor.slurm"),
                ],
                text=True,
            )
            .strip()
            .split(";")[0]
        )
    except Exception:
        subprocess.run(["scancel", router], check=False)
        raise
    jobs = read(root / "jobs.json")
    jobs["supervisor"] = supervisor
    save(root / "jobs.json", jobs)
    return jobs


def states(ids):
    out = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobID%64,State%32"],
        text=True,
        timeout=30,
    )
    return {
        line.split("|")[0]: line.split("|")[1].split()[0]
        for line in out.splitlines()
        if "|" in line
    }


def gate(root, row):
    from collections import Counter

    from export import assistant, latest, pairs, validate_tools

    d = latest(root, row)
    result = read(d / "completion.json")
    if result["status"] != "finished" or not result["cleanup_complete"]:
        raise ValueError(f"Pilot failed: {d.name}: {result['reason']}")
    if not (d / "board.db").exists() or result.get("board_messages", 0) < 1:
        raise ValueError("Missing messageboard evidence")
    if result["native_results"] != row["agents"]:
        raise ValueError("Missing native pilot results")
    barrier = read(d / "barrier.json")
    if len(barrier["ready"]) != row["agents"]:
        raise ValueError("Missing synchronized starts")
    captures = list((d / "native").glob("*/agent/sft-capture.jsonl"))
    if len(captures) != row["agents"]:
        raise ValueError("Missing agent capture")
    for capture in captures:
        rejects = Counter()
        valid = 0
        for request, response in pairs(capture, rejects):
            settings = request["settings"]
            expected = {
                "model": MODEL,
                "max_tokens": 16384,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "enable_thinking": False,
            }
            if any(settings[k] != v for k, v in expected.items()):
                raise ValueError("Wrong generation settings")
            try:
                target = assistant(response["message"])
            except ValueError as exc:
                if str(exc) == "incomplete_or_errored_response":
                    continue
                raise
            validate_tools(request["messages"] + [target], request["tools"])
            valid += 1
        if not valid:
            raise ValueError("No complete captured response from " + capture.parent.parent.name)
        if rejects["unpaired_response"] or rejects["malformed_capture_line"]:
            raise ValueError("Capture pairing failed")
    return {
        "run": d.name,
        "passed": True,
        "agents": len(captures),
        "verified_success": result["verified_success"],
    }


def supervise(root):
    root = Path(root).resolve()
    jobs = read(root / "jobs.json")
    router = jobs["router"]
    rows = read(root / "manifest.json")
    smokes = read(root / "smokes.json")
    reason = "failed"
    last_report = 0
    from export import latest

    def interrupted(sig, frame):
        raise SystemExit(128 + sig)

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)

    def launch(label, args):
        job = (
            subprocess.check_output(
                [
                    "sbatch",
                    "--parsable",
                    "--job-name=sft35-" + label,
                    "-o",
                    str(root / (label + "-%A_%a.log")),
                    *args,
                ],
                text=True,
            )
            .strip()
            .split(";")[0]
        )
        jobs["children"].append(job)
        save(root / "jobs.json", jobs)
        return job

    def wait(job, phase):
        nonlocal last_report
        while True:
            current = states([router, job])
            state = current.get(job)
            if current.get(router) in TERMINAL:
                raise RuntimeError("Router terminated")
            if state in TERMINAL or (
                state is None
                and any(k.startswith(job + "_") for k in current)
                and all(v in TERMINAL for k, v in current.items() if k.startswith(job + "_"))
            ):
                return current
            if time.time() - last_report > 600:
                completed = [
                    read(p) for row in rows if (p := latest(root, row) / "completion.json").exists()
                ]
                progress = {
                    "at": time.time(),
                    "phase": phase,
                    "job": job,
                    "states": current,
                    "planned": len(rows),
                    "finished": sum(c["status"] == "finished" for c in completed),
                    "winning_teams": sum(c.get("verified_success", False) for c in completed),
                }
                save(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
                last_report = time.time()
            time.sleep(30)

    try:
        while not (root / "server/endpoint.json").exists():
            if states([router]).get(router) in TERMINAL:
                raise RuntimeError("Router failed before readiness")
            time.sleep(30)
        import re
        import socket
        from urllib.parse import urlsplit

        endpoint = read(root / "server/endpoint.json")
        url = urlsplit(endpoint["base_url"])
        info = subprocess.check_output(["scontrol", "show", "job", router, "-o"], text=True)
        study = read(root / "study.json")
        study.update(
            router_job=router,
            router_url=f"http://{socket.gethostbyname(url.hostname)}:{url.port}/v1",
            router_end=re.search(r"EndTime=(\S+)", info).group(1) + "+00:00",
        )
        save(root / "study.json", study)
        if not (latest(root, smokes["smoke2"]) / "completion.json").exists():
            job = launch("capture2", [str(root / "launch.slurm"), "smoke2"])
            wait(job, "capture_smoke")
        gates = [gate(root, smokes["smoke2"])]
        save(root / "gates.json", gates)
        pilot_jobs = [
            (launch(f"pilot{d}", [str(root / "launch.slurm"), f"smoke{d}"]), smokes[f"smoke{d}"])
            for d in [5, 10, 15, 20, 25]
            if not (latest(root, smokes[f"smoke{d}"]) / "completion.json").exists()
        ]
        for job, row in pilot_jobs:
            wait(job, "native_pilots")
        gates = [gate(root, row) for row in smokes.values()]
        save(root / "gates.json", gates)
        cmd = [
            study["export_python"],
            str(root / "export.py"),
            str(root),
            str(root / "pilot-export"),
            "--tokenizer",
            study["model_snapshot"],
            "--pilots",
        ]
        subprocess.run(cmd, check=True, timeout=7200)
        pilot_report = read(root / "pilot-export/report.json")
        if pilot_report["counts"].get("examples", 0) == 0:
            raise RuntimeError("No winning pilot examples to validate export")
        job = launch("collection", ["--array=0-111%8", str(root / "launch.slurm")])
        wait(job, "collection")
        completions = [read(latest(root, row) / "completion.json") for row in rows]
        subprocess.run(
            [
                study["export_python"],
                str(root / "export.py"),
                str(root),
                str(root / "dataset-v1"),
                "--tokenizer",
                study["model_snapshot"],
            ],
            check=True,
            timeout=43200,
        )
        reason = (
            "complete"
            if all(c["status"] == "finished" and c["cleanup_complete"] for c in completions)
            else "incomplete_infrastructure"
        )
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if jobs["children"]:
            subprocess.run(["scancel", *jobs["children"]], check=False)
        prefixes = tuple(row["prefix"] + "-a" for row in [*rows, *smokes.values()])
        live = subprocess.check_output(["squeue", "--me", "-h", "-o", "%i|%j"], text=True)
        owned = [
            line.split("|")[0]
            for line in live.splitlines()
            if line.split("|")[1].removeprefix("legion-ctl-").startswith(prefixes)
        ]
        if owned:
            subprocess.run(["scancel", *owned], check=False)
        subprocess.run(["scancel", router], check=False)
        save(root / "final-status.json", {"at": time.time(), "reason": reason})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "submit", "resume", "supervise", "validate"])
    parser.add_argument("root", type=Path)
    parser.add_argument("--secrets", type=Path)
    parser.add_argument("--agents", type=int, default=16, help="Agents per production and difficulty-pilot team")
    parser.add_argument("--bucket", help="Writable Hub bucket, namespace/name")
    parser.add_argument("--network-helpers", type=Path, help="Directory containing netavark and aardvark-dns")
    args = parser.parse_args()
    if args.command == "prepare":
        if args.secrets is None or not args.bucket or "/" not in args.bucket or not args.network_helpers:
            parser.error("prepare requires --secrets, --bucket namespace/name and --network-helpers")
        print(prepare(args.root, args.secrets, args.bucket, args.network_helpers, args.agents))
    elif args.command in ("submit", "resume"):
        print(json.dumps(submit(args.root, resume=args.command == "resume")))
    elif args.command == "validate":
        root = args.root.resolve()
        study = read(root / "study.json")
        rows = select_tasks(
            read(root / "splits.json"),
            read(root / "registry.json"),
            "sft",
            [5, 10, 15, 20, 25],
            "evaluate",
        )
        for row in rows:
            verify_task(Path(study["task_snapshot"]) / row["task_path"], row)
        print(
            json.dumps(
                {
                    "verified_scenarios": len(rows),
                    "gates": [gate(root, row) for row in read(root / "smokes.json").values()],
                }
            )
        )
    else:
        supervise(args.root)
