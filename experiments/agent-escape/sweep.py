"""Gated Pi sweep: three pilots, then 75 array tasks capped at eight concurrent runs."""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
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
import yaml

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
        scenario_index=0, timeout=3600, output=root / label,
        attempt_number=row.get("attempt_number", 1))
    config = root / label / "config.json"
    if config.exists():
        return config
    return slurm.prepare(args) / "config.json"


def batch(root, mode, argument, *, array=None, dependency=None):
    command = ["sbatch", "--parsable", "--export=NONE", "--partition=hopper-cpu",
        "--cpus-per-task=8", "--mem=16G", "--time=05:00:00" if mode == "gate" else "--time=01:30:00",
        "--output=" + str(root / (mode + "-%A_%a.log")),
        "--job-name=" + root.name + "-" + mode]
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
    delivery = result.get("prompt_delivery")
    if delivery is not None and (delivery["missing_agents"] or
                                 len(delivery["prompted_agents"]) != agents):
        errors.append("initial prompts not delivered to full roster")
    journal = attempt / "model-requests.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
    starts = [r for r in rows if r["event"] == "start"]
    spec = yaml.safe_load((attempt / "spec.yaml").read_text())
    max_tokens = spec["models"]["qwen-27b"]["extra"]["maxTokens"]
    expected = {"get_observation", "investigate", "tool_call", "use_item", "submit_answer"}
    for row in starts:
        offered = set(row.get("tools", []))
        if not expected <= offered or offered & {"bash", "write", "edit", "find", "grep", "ls"}:
            errors.append("incorrect native tool selection")
            break
        settings = row["settings"]
        if settings.get("reasoning_effort") != "low" or max(
                settings.get("max_tokens", 0), settings.get("max_completion_tokens", 0)) != max_tokens:
            errors.append("incorrect reasoning or generation cap")
            break
    if not starts:
        errors.append("no model requests")
    return errors


def source_hashes():
    paths = sorted((ROOT / "src/flock").rglob("*.py"))
    paths += sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.yaml"))
    paths += [ROOT / "scripts/check_bucket_board.py", ROOT / "tests/test_reliability_concurrency.py",
              ROOT / "src/flock/worker/adapters/pi_tools.mjs", HERE / "sweep.slurm"]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def check_sources(cfg):
    if cfg.get("source_sha256") != source_hashes():
        raise RuntimeError("Validation source changed after submission; submit a fresh gate")


def gate(config_path):
    cfg = slurm.read_json(config_path)
    root = Path(cfg["output"])
    secret = Path(cfg["secrets_file"])
    try:
        check_sources(cfg)
        os.environ.update(slurm.read_secrets(secret))
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        check = subprocess.run([sys.executable, str(ROOT / "scripts/check_bucket_board.py"),
            "--bucket", "HuggingFaceSwarms/Experiments", "--mount-root", "/tmp/flock-board-gate",
            "--data-dir", str(root / "storage-gate"), "--duration", "1800", "--faults"],
            cwd=ROOT, check=False)
        if check.returncode:
            slurm.write_json(root / "gate.json", {"passed": False, "stage": "storage"})
            return 1
        check = subprocess.run([sys.executable, "-m", "pytest", "-q",
            "tests/test_reliability_concurrency.py"], cwd=ROOT, check=False)
        if check.returncode:
            slurm.write_json(root / "gate.json", {"passed": False, "stage": "concurrency"})
            return 1
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
        check_sources(cfg)
        rows = matrix(cfg["seed"])
        for i, row in enumerate(rows):
            row["config"] = str(prepare_task(root, cfg["server_run"], row, f"task-{i:03d}"))
        manifest = root / "manifest.json"
        slurm.write_json(manifest, {"seed": cfg["seed"], "scenario_index": 0, "tasks": rows,
            "server_run": cfg["server_run"], "secrets_file": cfg["secrets_file"]})
        slurm.write_json(root / "gate.json", {"passed": True, "pilots": len(PILOTS)})
        submission = batch(root, "task", manifest, array=f"0-{len(rows)-1}%8")
        slurm.write_json(root / "array-submission.json", submission)
        analysis = batch(root, "finalize", manifest, dependency=submission["job_id"])
        slurm.write_json(root / "analysis-submission.json", analysis)
        print(json.dumps({"array": submission["job_id"], "analysis": analysis["job_id"]}), flush=True)
        return 0
    finally:
        # The finalizer needs the authorized token to prepare at most one fresh retry.
        if not (root / "array-submission.json").exists():
            secret.unlink(missing_ok=True)


def scheduler_states(job):
    if not job:
        return {}
    output = subprocess.check_output(["sacct", "-n", "-P", "-j", str(job),
        "--format=JobID%100,State"], text=True)
    return {line.split("|")[0]: line.split("|")[1].split()[0].rstrip("+")
            for line in output.splitlines() if "|" in line}


def read_report(path, default):
    try:
        return slurm.read_json(path)
    except (OSError, ValueError):
        return default


def attempt_result(task, state=None):
    path = Path(task["config"]).parent / "summary.json"
    summary = read_report(path, [])
    entry = summary[0] if summary else {}
    result = entry.get("result", {})
    # Node death can leave an attempt report but prevent the launcher's summary write.
    if not result:
        pointers = list(path.parent.glob("*-attempt.json"))
        if len(pointers) == 1:
            pointer = read_report(pointers[0], {})
            if pointer.get("path"):
                attempt = Path(pointer["path"])
                result = read_report(attempt / "result.json", {})
                if not result and (attempt / "model-requests.jsonl").exists():
                    from usage_proxy import summarize
                    partial = summarize(attempt / "model-requests.jsonl")
                    partial["complete"] = False  # no final report means termination is unverified
                    result = {"accounting": partial, "reason": "missing_report"}
                entry = {"attempt": str(attempt), "result": result}
    accounting = result.get("accounting", {})
    infrastructure = result.get("infrastructure_outcome", "unknown")
    if state == "NODE_FAIL":
        infrastructure = "node_failure"
    elif state and state.startswith("CANCELLED"):
        infrastructure = "cancelled"
    valid = bool(accounting.get("complete"))
    success = bool(result.get("success") and entry.get("exit_code") == 0 and valid)
    total = accounting.get("total_tokens") if valid else None
    return {**task, "success": success, "verified_solution": bool(result.get("success")),
        "measurement_valid": valid, "infrastructure_outcome": infrastructure,
        "puzzle_outcome": result.get("puzzle_outcome", "unknown"), "scheduler_state": state,
        "persistence_status": result.get("persistence_status", "unknown"),
        "reason": result.get("reason", "missing_report"), "attempt": entry.get("attempt"),
        "total_tokens": total, "observed_tokens": accounting.get("total_tokens"),
        "normalized_tokens": total / task["agents"] if total is not None else None,
        "elapsed_seconds": result.get("elapsed_seconds"), "error": result.get("error"),
        "measurement_error": result.get("measurement_error")}


def collect(manifest):
    doc = slurm.read_json(manifest)
    root = manifest.parent
    states = slurm.read_json(root / "scheduler-states.json") if (
        root / "scheduler-states.json").exists() else {}
    retries = {r["original_index"]: r for r in doc.get("retries", [])}
    rows = []
    for i, task in enumerate(doc["tasks"]):
        original = attempt_result(task, states.get(str(i)))
        attempts = [original]
        if i in retries:
            attempts.append(attempt_result(retries[i], states.get("retry-" + str(i))))
        selected = attempts[-1]
        rows.append({**selected, "attempt_count": len(attempts),
            "discarded_observed_tokens": sum(a["observed_tokens"] or 0 for a in attempts[:-1]),
            "discarded_accounting_complete": all(a["measurement_valid"] for a in attempts[:-1]),
            "attempt_history": attempts})
    return rows


def finalize(manifest):
    root = manifest.parent
    # One finalizer owns retry planning/submission and analysis at a time.
    with (root / ".finalize.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        doc = slurm.read_json(manifest)
        if (root / "finalized.json").exists():
            return 0
        submission = slurm.read_json(root / "array-submission.json")
        states = scheduler_states(submission["job_id"])
        mapped = {str(i): states.get(f"{submission['job_id']}_{i}")
                  for i in range(len(doc["tasks"]))}
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "NODE_FAIL", "TIMEOUT",
                    "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
        if any(state not in terminal for state in mapped.values()):
            raise RuntimeError("Primary scheduler states are missing or nonterminal; retry finalizer later")
        retry_submission = root / "retry-submission.json"
        if retry_submission.exists():
            retry_job = slurm.read_json(retry_submission)["job_id"]
            retry_states = scheduler_states(retry_job)
            for i, row in enumerate(doc.get("retries", [])):
                mapped["retry-" + str(row["original_index"])] = retry_states.get(f"{retry_job}_{i}")
            if any(state not in terminal for state in mapped.values()):
                raise RuntimeError("Retry scheduler states are missing or nonterminal; retry finalizer later")
            slurm.write_json(root / "scheduler-states.json", mapped)
            analyze(manifest)
            slurm.write_json(root / "finalized.json", {"complete": True})
            Path(doc["secrets_file"]).unlink(missing_ok=True)
            return 0
        slurm.write_json(root / "scheduler-states.json", mapped)
        if "retries" not in doc:
            os.environ.update(slurm.read_secrets(doc["secrets_file"]))
            retries = []
            for i, task in enumerate(doc["tasks"]):
                outcome = attempt_result(task, mapped[str(i)])
                if outcome["infrastructure_outcome"] not in {"node_failure", "startup_failure"}:
                    continue
                row = {k: task[k] for k in ("difficulty", "agents", "trial")}
                row.update(original_index=i, attempt_number=2)
                row["config"] = str(prepare_task(root, doc["server_run"], row, f"retry-{i:03d}"))
                retries.append(row)
            doc["retries"] = retries
            slurm.write_json(manifest, doc)
        if doc["retries"]:
            # A durable intent prevents duplicate submissions if the scheduler accepts a job
            # but the process dies before recording its ID. Ambiguity stops for reconciliation.
            intent = root / "retry-submit-intent.json"
            if intent.exists():
                raise RuntimeError("Retry submission is ambiguous; reconcile scheduler job name before retrying")
            slurm.write_json(intent, {"job_name": root.name + "-retry"})
            retry = batch(root, "retry", manifest, array=f"0-{len(doc['retries'])-1}%8")
            slurm.write_json(retry_submission, retry)
            follow = batch(root, "finalize", manifest, dependency=retry["job_id"])
            slurm.write_json(root / "retry-finalizer.json", follow)
            return 0
        analyze(manifest)
        slurm.write_json(root / "finalized.json", {"complete": True})
        Path(doc["secrets_file"]).unlink(missing_ok=True)
        return 0


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
    for mode in ("gate", "task", "retry", "finalize", "analyze"):
        sub.add_parser(mode).add_argument("path", type=Path)
    args = parser.parse_args()
    if args.mode == "submit":
        root = ROOT / ".flock/agent-escape-sweeps" / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        root.mkdir(parents=True, mode=0o700)
        slurm.write_json(root / "secrets.json", {"HF_TOKEN": os.environ["HF_TOKEN"]})
        slurm.write_json(root / "gate-config.json", {"output": str(root), "seed": args.seed,
            "server_run": str(args.server_run.resolve()), "secrets_file": str(root / "secrets.json"),
            "source_sha256": source_hashes()})
        submission = batch(root, "gate", root / "gate-config.json")
        slurm.write_json(root / "gate-submission.json", submission)
        print(json.dumps({"output": str(root), **submission}, indent=2))
        return 0
    if args.mode == "gate":
        return gate(args.path)
    if args.mode == "finalize":
        return finalize(args.path)
    if args.mode in {"task", "retry"}:
        task = slurm.read_json(args.path)["tasks" if args.mode == "task" else "retries"][int(os.environ["SLURM_ARRAY_TASK_ID"])]
        return slurm.run(task["config"])
    return analyze(args.path)


if __name__ == "__main__":
    sys.exit(main())
