# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8", "pyyaml"]
# ///
"""Hint-only maze results: `uv run report.py --runs FINAL_RUNS --out RESULTS`.

`FINAL_RUNS` contains 27b-hint/ and 2b-hint/. With no --runs, redraw from the saved teams.csv
and capture.json beside this script. Produces maze-preliminary.png, plain points and lines without
intervals, and each model's best-of-N from its own N=1 rate when at least 20 teams support it.
"""

import argparse
import csv
import json
import os
import statistics
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, NullFormatter

GRIDS = [("27b-hint", "Qwen3.5-27B + hint", "#2a78d6"),
         ("2b-hint", "Qwen3.5-2B + hint", "#eb6834")]
SIZES = [1, 2, 4, 8, 16]


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def capture(root, out):
    teams, discarded, revisions = [], [], set()
    accounting = {}
    for grid, label, _ in GRIDS:
        source = root / grid
        rows = [json.loads(line) for line in (source / "rows.jsonl").read_text().splitlines() if line]
        kept = set()
        for row in rows:
            if row["missing"]:
                raise ValueError(f"{row['run']}: incomplete evidence")
            run = source / "runs" / row["run"]
            spec = yaml.safe_load((run / "experiment/spec.yaml").read_text())
            tokens = row["tokens"]
            teams.append({
                "grid": grid, "model": label, "tree": row["tree"], "n": row["agents"],
                "success": int(bool(row["success"])), "run": row["run"],
                "input_tokens": tokens["input"] + tokens["cache_read"],
                "uncached_input_tokens": tokens["input"], "cached_input_tokens": tokens["cache_read"],
                "output_tokens": tokens["output"], "board_posts": row["posts"],
                "room_overlap": row.get("room_overlap"),
                "first_exit_s": (row.get("first_exit") or {}).get("after_s"),
                "wall_s": row.get("wall_s"), "worker_cpus": spec["compute"]["cpus"],
            })
            kept.add(row["run"])
            revisions.add(row["revision"])
        records = [json.loads(line) for line in (source / "rows-discarded.jsonl").read_text().splitlines()
                   if line]
        unique = {row["run"]: row for row in records}
        accounting[grid] = {
            "discard_records": len(records), "distinct_runs_in_discard_log": len(unique),
            "repeated_discard_records": len(records) - len(unique),
            "eventually_accepted_runs": sorted(set(unique) & kept),
            "distinct_set_aside_runs": len(set(unique) - kept),
        }
        for name, row in sorted(unique.items()):
            discarded.append({"grid": grid, "run": name, "tree": row["tree"], "n": row["agents"],
                              "eventually_accepted": int(name in kept),
                              "reason": row.get("set_aside", ""), "outcome": row.get("outcome", "")})
    teams.sort(key=lambda row: (row["grid"], row["n"], row["tree"]))
    write_csv(out / "teams.csv", teams)
    if discarded:
        write_csv(out / "attempts.csv", discarded)
    snapshot_path = root / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text()) if snapshot_path.exists() else {}
    metadata = {"at": snapshot.get("at", datetime.now(ZoneInfo("Europe/Zurich")).isoformat()),
                "evaluated_flock_revision": "03ec1dd9be659bde3751d37e393d014cdd8e5061",
                "render_slurm_job": os.environ.get("SLURM_JOB_ID"), "source_snapshot": snapshot,
                "evaluated_dataset_revisions": sorted(revisions), "discard_accounting": accounting}
    (out / "capture.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return teams, metadata


def read_capture(out):
    with (out / "teams.csv").open(newline="") as handle:
        teams = list(csv.DictReader(handle))
    integer = ["tree", "n", "success", "input_tokens", "uncached_input_tokens",
               "cached_input_tokens", "output_tokens", "board_posts", "worker_cpus"]
    for row in teams:
        for key in integer:
            row[key] = int(row[key])
        for key in ["room_overlap", "first_exit_s", "wall_s"]:
            row[key] = float(row[key]) if row[key] else None
    return teams, json.loads((out / "capture.json").read_text())


def summarise(teams):
    seen = set()
    allowed = {grid for grid, _, _ in GRIDS}
    for row in teams:
        key = (row["grid"], row["tree"], row["n"])
        if key in seen or row["grid"] not in allowed or row["n"] not in SIZES or not 1 <= row["tree"] <= 32:
            raise ValueError(f"Unexpected or duplicate team: {key}")
        seen.add(key)
    summaries = []
    for grid, label, _ in GRIDS:
        first = [row for row in teams if row["grid"] == grid and row["n"] == 1]
        p1 = sum(row["success"] for row in first) / len(first) if len(first) >= 20 else None
        for n in SIZES:
            group = [row for row in teams if row["grid"] == grid and row["n"] == n]
            won, count = sum(row["success"] for row in group), len(group)
            summaries.append({
                "grid": grid, "model": label, "n": n, "successes": won, "completed": count,
                "unresolved": 32 - count, "success_rate": won / count if count else None,
                "best_of_n": 1 - (1 - p1) ** n if p1 is not None else None,
                "median_input_tokens": statistics.median(row["input_tokens"] for row in group) if group else None,
                "median_output_tokens": statistics.median(row["output_tokens"] for row in group) if group else None,
            })
    return summaries


def figure(summary, at, out):
    fig = Figure(figsize=(8.2, 5.5), facecolor="#fcfcfb")
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    partial = any(row["completed"] != 32 for row in summary)
    for grid, label, colour in GRIDS:
        rows = [row for row in summary if row["grid"] == grid and row["completed"]]
        is_partial = any(row["completed"] != 32 for row in rows) or len(rows) != len(SIZES)
        total = sum(row["completed"] for row in rows)
        suffix = f" (partial: {total}/160 teams)" if is_partial else ""
        ax.plot([row["n"] for row in rows], [100 * row["success_rate"] for row in rows],
                color=colour, marker="o", linewidth=2, markersize=6, alpha=0.6 if is_partial else 1,
                label=label + suffix)
        if is_partial:
            for row in rows:
                ax.annotate(f"n={row['completed']}/32", (row["n"], 100 * row["success_rate"]),
                            textcoords="offset points", xytext=(4, 7), fontsize=8, color=colour)
        reference = [row for row in summary if row["grid"] == grid and row["best_of_n"] is not None]
        if reference:
            first = next(row for row in summary if row["grid"] == grid and row["n"] == 1)
            ax.plot([row["n"] for row in reference], [100 * row["best_of_n"] for row in reference],
                    color=colour, linestyle="--", linewidth=1.2,
                    label=f"{label.split(' +')[0]}: best-of-N ({first['successes']}/{first['completed']} at N=1)")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(SIZES))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(n) for n in SIZES])
    ax.set_xlim(0.8, 20)
    ax.set_ylim(0, 106)
    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Success (%): any agent reaches the exit")
    ax.set_facecolor("#fcfcfb")
    ax.grid(axis="y", color="#e4e3df")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    stamp = datetime.fromisoformat(at).astimezone(ZoneInfo("Europe/Zurich")).strftime("%d %b %Y, %H:%M %Z")
    ax.set_title("s³ maze (preliminary): protocol hint, thinking off, max_channels: 1\n" + stamp,
                 fontsize=11, loc="left")
    high_n = next(row for row in summary if row["grid"] == "2b-hint" and row["n"] == 16)
    note = (f"2B N=16: {high_n['completed']}/32 teams recorded; completion is selective, and unresolved teams are excluded.\n"
            "Dashed: independent attempts at each model's own N=1 rate; these are descriptive references." if partial else
            "32 teams per point, one per tree. Dashed: independent attempts at each model's own N=1 rate.")
    fig.text(0.02, 0.01, note, fontsize=8, color="#52514e")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "maze-preliminary.png", dpi=200, facecolor="#fcfcfb")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    teams, metadata = capture(args.runs, args.out) if args.runs else read_capture(args.out)
    summary = summarise(teams)
    write_csv(args.out / "by_n.csv", summary)
    table = ["| Model | N | Successes/completed | Best-of-N | Median input / output tokens |",
             "|---|---:|---:|---:|---:|"]
    for row in summary:
        reference = f"{100 * row['best_of_n']:.0f}%" if row["best_of_n"] is not None else "—"
        median = (f"{row['median_input_tokens'] / 1e6:.1f}M / {row['median_output_tokens'] / 1e3:.0f}k"
                  if row["completed"] else "—")
        table.append(f"| {row['model']} | {row['n']} | {row['successes']}/{row['completed']} | {reference} | {median} |")
    (args.out / "by_n.md").write_text("\n".join(table) + "\n")
    figure(summary, metadata["at"], args.out)
    print(metadata["at"])
    print("\n".join(table))


if __name__ == "__main__":
    main()
