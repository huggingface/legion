# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml"]
# ///
"""Read-only capture of valid maze teams and exact exported generated-token evidence.

uv run capture.py --runs FINAL --hint TEAM_MD --out OUTPUT
Every output count is checked against the row, agent and team totals. Missing fields fail
unless the hashed Trial proves setup failed before execution; those Trials generated zero.
Explicit zero-output teams remain in both panels' cohort.
"""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

GRIDS = [("27b-hint", "Qwen3.5-27B"), ("2b-hint", "Qwen3.5-2B")]
SOURCE = "03ec1dd9be659bde3751d37e393d014cdd8e5061"
REVISION = "3b6c363d92626bd4d97a7ca89964fb850982d9ab"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def count(value, where):
    if type(value) is not int or value < 0:
        raise ValueError(f"Missing or invalid generated-token count: {where}")
    return value


def trial_output(result, path):
    value = (result.get("agent_result") or {}).get("n_output_tokens")
    if type(value) is int and value >= 0:
        return value, "explicit_n_output_tokens"
    error = result.get("exception_info") or {}
    trace = error.get("exception_traceback", "")
    # Harbor initializes agent_result and agent_execution before calling agent.run.
    # A failed _prepare/_setup_agent with neither field never executed the model.
    if (result.get("agent_result") is None and result.get("agent_execution") is None
            and error.get("exception_type") and "_prepare" in trace and "_setup_agent" in trace):
        return 0, "proven_pre_execution_setup_failure"
    raise ValueError(f"Missing generated-token evidence for an executed/unknown Trial: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--rows", type=Path, help="fixed read-only row snapshot, otherwise read runs")
    parser.add_argument("--hint", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    hint = args.hint.read_text().strip()
    snapshot, teams, audit, attempts, seen = {}, [], [], [], set()
    metadata = {
        "evaluated_flock_revision": SOURCE, "evaluated_dataset_revision": REVISION,
        "render_slurm_job": os.environ.get("SLURM_JOB_ID"),
        "collaboration": {"instructions": hint, "max_channels": 1, "channels": ["general"],
                          "wake": ["mention", "thread"], "agent_dms": False},
        "enable_thinking": False, "raw_grid_to_display_model": dict(GRIDS),
        "source_row_files_sha256": {}, "discard_accounting": {},
        "output_source": "Harbor result.json agent_result.n_output_tokens; exact per-Trial/agent/team checks",
        "missing_output_policy": "abort unless hashed pre-execution setup failure proves zero; never exclude teams",
        "zero_output_policy": "include explicit zeros; abort only if a whole group's mean is zero",
    }
    # Read the two accepted-row files once before auditing any source; later appends are excluded.
    row_source = args.rows or args.runs
    for grid, _ in GRIDS:
        raw = (row_source / grid / "rows.jsonl").read_bytes()
        snapshot[grid] = [json.loads(line) for line in raw.splitlines() if line.strip()]
        metadata["source_row_files_sha256"][f"{grid}/rows.jsonl"] = digest(raw)
    frozen = row_source / "snapshot.json"
    metadata["at"] = (json.loads(frozen.read_text())["at"] if frozen.exists() else
                      datetime.now(ZoneInfo("Europe/Zurich")).isoformat())
    for grid, model in GRIDS:
        kept = {row["run"] for row in snapshot[grid]}
        raw = (row_source / grid / "rows-discarded.jsonl").read_bytes()
        discarded = [json.loads(line) for line in raw.splitlines() if line.strip()]
        unique = {row["run"]: row for row in discarded}
        metadata["source_row_files_sha256"][f"{grid}/rows-discarded.jsonl"] = digest(raw)
        metadata["discard_accounting"][grid] = {
            "discard_records": len(discarded), "distinct_runs_in_discard_log": len(unique),
            "repeated_discard_records": len(discarded) - len(unique),
            "eventually_accepted_runs": sorted(set(unique) & kept),
            "distinct_set_aside_runs": len(set(unique) - kept),
        }
        for name, row in sorted(unique.items()):
            attempts.append({"model": model, "grid": grid, "run": name, "tree": row["tree"],
                             "n": row["agents"], "eventually_accepted": int(name in kept),
                             "reason": row.get("set_aside", ""), "outcome": row.get("outcome", "")})
        for row in snapshot[grid]:
            key = model, row["tree"], row["agents"]
            if key in seen or row["missing"] or row["revision"] != REVISION:
                raise ValueError(f"Duplicate, missing or wrong-revision team: {key}")
            seen.add(key)
            if row["agents"] not in [1, 2, 4, 8, 16] or not 1 <= row["tree"] <= 32:
                raise ValueError(f"Unexpected team: {key}")
            run = args.runs / grid / "runs" / row["run"]
            spec = yaml.safe_load((run / "experiment/spec.yaml").read_text())
            board = spec["resources"]["message_board"]
            if (spec["model"] != f"Qwen/{model}" or spec["agents"]["count"] != row["agents"]
                    or spec["task"]["selector"] != f"maze-t{row['tree']:02d}"
                    or spec["task"]["revision"] != REVISION
                    or board["channels"] != ["general"] or board["rules"]["max_channels"] != 1
                    or (run / "experiment/team.md").read_text().strip() != hint
                    or len(list((run / "launch").glob("*.env"))) != 1):
                raise ValueError(f"Unexpected source configuration: {key}")
            outputs, trial_hashes = defaultdict(int), {}
            for path in sorted((run / "results").rglob("trials/*/result.json")):
                agent, trial = path.parent.parent.parent.name, path.parent.name
                raw = path.read_bytes()
                identity = agent, trial
                if identity in trial_hashes:
                    if trial_hashes[identity] != digest(raw):
                        raise ValueError(f"Conflicting duplicate Trial: {path}")
                    continue
                trial_hashes[identity] = digest(raw)
                result = json.loads(raw)
                output, evidence = trial_output(result, path)
                outputs[agent] += output
                audit.append({"model": model, "grid": grid, "run": row["run"], "agent": agent,
                              "trial": trial, "source_path": str(path.relative_to(args.runs)),
                              "source_sha256": digest(raw), "output_tokens": output,
                              "output_evidence": evidence,
                              "agent_execution_present": int(result.get("agent_execution") is not None),
                              "exception_type": (result.get("exception_info") or {}).get("exception_type", "")})
            expected_agents = {f"agent-{i:03d}" for i in range(1, row["agents"] + 1)}
            if set(outputs) != expected_agents or set(row["per_agent"]) != expected_agents:
                raise ValueError(f"Missing or unplanned agent token source: {key}")
            for agent, output in outputs.items():
                if output != count(row["per_agent"][agent]["tokens"].get("output"), (key, agent)):
                    raise ValueError(f"Per-agent output mismatch: {key}, {agent}")
            total = sum(outputs.values())
            if total != count(row["tokens"].get("output"), key):
                raise ValueError(f"Team output mismatch: {key}")
            if type(row["success"]) is not bool:
                raise ValueError(f"Unknown success outcome: {key}")
            teams.append({"model": model, "grid": grid, "tree": row["tree"], "n": row["agents"],
                          "success": int(row["success"]), "run": row["run"], "output_tokens": total,
                          "audited_trials": len(trial_hashes)})
    teams.sort(key=lambda row: (row["model"] == "Qwen3.5-2B", row["n"], row["tree"]))
    write_csv(args.out / "teams.csv", teams)
    write_csv(args.out / "token-audit.csv", audit)
    write_csv(args.out / "attempts.csv", attempts)
    metadata["audited_teams"] = len(teams)
    metadata["audited_trials"] = len(audit)
    metadata["proven_pre_execution_zero_trials"] = sum(
        row["output_evidence"] == "proven_pre_execution_setup_failure" for row in audit)
    metadata["zero_output_teams"] = [row["run"] for row in teams if row["output_tokens"] == 0]
    metadata["cohort_sha256"] = digest((args.out / "teams.csv").read_bytes())
    metadata["token_audit_sha256"] = digest((args.out / "token-audit.csv").read_bytes())
    (args.out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({key: metadata[key] for key in ["at", "audited_teams", "audited_trials",
                                                   "zero_output_teams", "cohort_sha256"]}))


if __name__ == "__main__":
    main()
