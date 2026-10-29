"""Gated Pi sweep: three pilots, then 75 array tasks capped at eight concurrent runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

import slurm

HERE, ROOT = slurm.HERE, slurm.ROOT
DIFFICULTIES = [5, 10, 15, 20, 25]
COUNTS = [2, 4, 8, 16, 32]
PILOTS = [(5, 4), (10, 4), (5, 32)]


def matrix(seed=42):
    rows = [{"difficulty": d, "agents": n, "trial": r}
            for d in DIFFICULTIES for n in COUNTS for r in range(1, 4)]
    random.Random(seed).shuffle(rows)
    return rows


def prepare_task(root, server, row, label):
    args = argparse.Namespace(server_run=Path(server), spec=HERE / "spec-router-pi.yaml",
        difficulties=[row["difficulty"]], agents=row["agents"], trial=row["trial"],
        scenario_index=0, timeout=3600, output=root / label)
    return slurm.prepare(args) / "config.json"


def batch(root, mode, argument, *, array=None, dependency=None):
    command = ["sbatch", "--parsable", "--export=NONE", "--partition=hopper-cpu",
        "--cpus-per-task=8", "--mem=16G", "--time=04:00:00" if mode == "gate" else "--time=01:15:00",
        "--output=" + str(root / (mode + "-%A_%a.log"))]
    if array:
        command.append("--array=" + array)
    if dependency:
        command.append("--dependency=afterany:" + dependency)
    command += [str(HERE / "sweep.slurm"), mode, str(argument)]
    env = {k: v for k, v in os.environ.items() if k in {"PATH", "HOME", "USER", "LOGNAME", "LANG"}}
    job = subprocess.check_output(command, cwd=ROOT, env=env, text=True).strip().split(";")[0]
    return {"job_id": job, "command": command}


def gate_errors(result, agents, attempt):
    errors = []
    accounting = result.get("accounting", {})
    if not result.get("success") or result.get("error") or result.get("flock_exit_code") != 0:
        errors.append("pilot did not finish successfully")
    if not accounting.get("complete") or len(accounting.get("session_ids", [])) != agents:
        errors.append("incomplete usage or missing agent conversation IDs")
    if not result.get("board", {}).get("verified_at"):
        errors.append("board persistence not verified")
    journal = attempt / "model-requests.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    starts = [r for r in rows if r["event"] == "start"]
    expected = {"get_observation", "investigate", "tool_call", "use_item", "submit_answer"}
    for row in starts:
        offered = set(row.get("tools", []))
        if not expected <= offered or offered & {"bash", "write", "edit", "find", "grep", "ls"}:
            errors.append("incorrect native tool selection")
            break
        settings = row["settings"]
        if settings.get("reasoning_effort") != "low" or max(
                settings.get("max_tokens", 0), settings.get("max_completion_tokens", 0)) != 4096:
            errors.append("incorrect reasoning or generation cap")
            break
    if not starts:
        errors.append("no model requests")
    return errors


def gate(config_path):
    cfg = slurm.read_json(config_path)
    root = Path(cfg["output"])
    secret = Path(cfg["secrets_file"])
    try:
        os.environ.update(slurm.read_secrets(secret))
        for i, (difficulty, agents) in enumerate(PILOTS):
            row = {"difficulty": difficulty, "agents": agents, "trial": 0}
            task = prepare_task(root, cfg["server_run"], row, f"pilot-{i}-d{difficulty}-n{agents}")
            code = slurm.run(task)
            summary = slurm.read_json(task.parent / "summary.json")
            errors = ["pilot launcher failed"] if code or not summary else []
            if summary:
                entry = summary[0]
                errors += gate_errors(entry["result"], agents, Path(entry["attempt"]))
            slurm.write_json(root / f"pilot-{i}-validation.json", {"passed": not errors, "errors": errors})
            if errors:
                slurm.write_json(root / "gate.json", {"passed": False, "pilot": i, "errors": errors})
                return 1
        rows = matrix(cfg["seed"])
        for i, row in enumerate(rows):
            row["config"] = str(prepare_task(root, cfg["server_run"], row, f"task-{i:03d}"))
        manifest = root / "manifest.json"
        slurm.write_json(manifest, {"seed": cfg["seed"], "scenario_index": 0, "tasks": rows})
        slurm.write_json(root / "gate.json", {"passed": True, "pilots": len(PILOTS)})
        submission = batch(root, "task", manifest, array=f"0-{len(rows)-1}%8")
        slurm.write_json(root / "array-submission.json", submission)
        analysis = batch(root, "analyze", manifest, dependency=submission["job_id"])
        slurm.write_json(root / "analysis-submission.json", analysis)
        print(json.dumps({"array": submission["job_id"], "analysis": analysis["job_id"]}), flush=True)
        return 0
    finally:
        secret.unlink(missing_ok=True)


def collect(manifest):
    rows = []
    for task in slurm.read_json(manifest)["tasks"]:
        path = Path(task["config"]).parent / "summary.json"
        summary = slurm.read_json(path) if path.exists() else []
        entry = summary[0] if summary else {}
        result = entry.get("result", {})
        accounting = result.get("accounting", {})
        valid = accounting.get("complete", False)
        success = bool(result.get("success") and entry.get("exit_code") == 0 and valid)
        total = accounting.get("total_tokens") if valid else None
        rows.append({**task, "success": success, "measurement_valid": valid,
            "reason": result.get("reason", "missing_report"), "attempt": entry.get("attempt"),
            "total_tokens": total, "observed_tokens": accounting.get("total_tokens"),
            "normalized_tokens": total / task["agents"] if total is not None else None,
            "elapsed_seconds": result.get("elapsed_seconds"), "error": result.get("error"),
            "measurement_error": result.get("measurement_error")})
    return rows


def analyze(manifest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = manifest.parent
    rows = collect(manifest)
    slurm.write_json(root / "results.json", rows)
    with (root / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4.8), sharex=True, sharey=True)
    for ax, difficulty in zip(axes, DIFFICULTIES, strict=True):
        for n in COUNTS:
            values = [r["normalized_tokens"] for r in rows if r["difficulty"] == difficulty
                      and r["agents"] == n and r["success"]]
            if values:
                mean = statistics.mean(values)
                sd = statistics.stdev(values) if len(values) > 1 else None
                ax.errorbar(n, mean, yerr=sd, fmt="o", color="tab:blue", capsize=4)
            ax.text(n, .98, f"{len(values)}/3", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=8)
        ax.set_title(f"Difficulty {difficulty}")
        ax.set_xticks(COUNTS)
        ax.set_xlabel("Number of agents")
        ax.grid(alpha=.2)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Total tokens / configured agents")
    fig.suptitle("Pi · Qwen3.8-27B · scenario 0 · successful trials: mean ± sample SD")
    fig.tight_layout()
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(root / f"normalized-tokens.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps({"runs": len(rows), "successes": sum(r["success"] for r in rows),
                      "output": str(root)}))
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--server-run", type=Path, required=True)
    submit.add_argument("--seed", type=int, default=42)
    for mode in ("gate", "task", "analyze"):
        sub.add_parser(mode).add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "submit":
        root = ROOT / ".flock/agent-escape-sweeps" / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        root.mkdir(parents=True, mode=0o700)
        slurm.write_json(root / "secrets.json", {"HF_TOKEN": os.environ["HF_TOKEN"]})
        slurm.write_json(root / "gate-config.json", {"output": str(root), "seed": args.seed,
            "server_run": str(args.server_run.resolve()), "secrets_file": str(root / "secrets.json")})
        submission = batch(root, "gate", root / "gate-config.json")
        slurm.write_json(root / "gate-submission.json", submission)
        print(json.dumps({"output": str(root), **submission}, indent=2))
        return 0
    if args.mode == "gate":
        return gate(args.path)
    if args.mode == "task":
        task = slurm.read_json(args.path)["tasks"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
        return slurm.run(task["config"])
    return analyze(args.path)


if __name__ == "__main__":
    sys.exit(main())
