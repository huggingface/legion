# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Maze success and reference-normalized output-token efficiency from one exact cohort.

uv run efficiency.py teams.csv --capture capture.json --out success-efficiency.png
Matches aggregation 19eca5c, experiments/s3-aggregation/sweep/efficiency.py.
The curated README/provenance is maintained separately and is never overwritten by redraws. The scale
9216 is a reference, not an enforced budget. Successes and failures share the same mean.
"""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODELS = [("Qwen3.5-27B", "#2a78d6"), ("Qwen3.5-2B", "#eb6834")]
SIZES = [1, 2, 4, 8, 16]


def summarize(path, reference_tokens):
    if reference_tokens <= 0:
        raise ValueError("Reference tokens must be positive")
    groups, seen = defaultdict(list), set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            model, n, tree = row["model"], int(row["n"]), int(row["tree"])
            success, output = int(row["success"]), int(row["output_tokens"])
            key = model, n, tree
            if (key in seen or model not in dict(MODELS) or n not in SIZES or not 1 <= tree <= 32
                    or success not in [0, 1] or output < 0):
                raise ValueError(f"Invalid or duplicate team: {key}")
            seen.add(key)
            groups[model, n].append((success, output))
    summary = []
    for model, _ in MODELS:
        first = groups[model, 1]
        p1 = sum(success for success, _ in first) / len(first) if len(first) >= 20 else None
        for n in SIZES:
            cohort = groups[model, n]
            if not cohort or len(cohort) > 32:
                raise ValueError(f"Empty or oversized cohort: {model}, {n}")
            successes = sum(success for success, _ in cohort)
            output_sum = sum(output for _, output in cohort)
            if output_sum <= 0:
                raise ValueError(f"Undefined efficiency: all explicit output counts are zero at {model}, {n}")
            mean_output = output_sum / len(cohort)
            success_rate = successes / len(cohort)
            summary.append({"model": model, "n": n, "teams": len(cohort), "successes": successes,
                            "failed_teams": len(cohort) - successes, "unresolved": 32 - len(cohort),
                            "success_rate": success_rate, "sum_team_output_tokens": output_sum,
                            "mean_team_output_tokens": mean_output,
                            "reference_tokens_per_agent": reference_tokens,
                            "token_efficiency": success_rate * n * reference_tokens / mean_output,
                            "best_of_n": 1 - (1 - p1) ** n if p1 is not None else None})
    return summary


def figure(path, summary, metadata, reference_tokens):
    with rc_context({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                     "svg.hashsalt": "s3-maze-efficiency"}):
        fig = Figure(figsize=(11.8, 5.5), facecolor=SURFACE)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        for model, color in MODELS:
            points = [row for row in summary if row["model"] == model]
            ns = [row["n"] for row in points]
            partial = any(row["unresolved"] for row in points)
            style = {"marker": "o", "linewidth": 2.3, "markersize": 7,
                     "color": color, "label": model, "alpha": 0.6 if partial else 1}
            axes[0].plot(ns, [100 * row["success_rate"] for row in points], **style)
            axes[1].plot(ns, [row["token_efficiency"] for row in points], **style)
            reference = [row for row in points if row["best_of_n"] is not None]
            if reference:
                axes[0].plot([row["n"] for row in reference],
                             [100 * row["best_of_n"] for row in reference],
                             linestyle="--", linewidth=1.3, color=color)
            if partial:
                for row in points:
                    label = f"n={row['teams']}/32" + ("*" if row["unresolved"] else "")
                    for ax, value in [(axes[0], 100 * row["success_rate"]),
                                      (axes[1], row["token_efficiency"])]:
                        ax.annotate(label, (row["n"], value), textcoords="offset points",
                                    xytext=(-6, 9) if row["n"] == 16 else (3, 8),
                                    ha="right" if row["n"] == 16 else "left", fontsize=8, color=color)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(SIZES, labels=[str(n) for n in SIZES])
            ax.set_xlabel("Swarm size N", labelpad=8)
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, 108)
        axes[0].set_ylabel("Success (%)")
        axes[0].set_title("Success (%) ↑", loc="left", pad=14, fontsize=13)
        axes[0].legend(handles=[Line2D([0], [0], color=MUTED, linestyle="--", linewidth=1.3,
                                      label="Best-of-N (each model's own N=1)")],
                       loc="upper left", frameon=False, fontsize=8)
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel("Reference-normalized efficiency")
        axes[1].set_title("Token efficiency ↑", loc="left", pad=14, fontsize=13)
        partial = any(row["unresolved"] for row in summary)
        title = "s3-maze: success and token efficiency" + (" (preliminary)" if partial else "")
        fig.suptitle(title, x=0.07, ha="left",
                     fontsize=16, y=0.985)
        stamp = datetime.fromisoformat(metadata["at"]).astimezone(ZoneInfo("Europe/Zurich")).strftime("%d %b %Y, %H:%M:%S %Z")
        totals = {model: sum(row["teams"] for row in summary if row["model"] == model) for model, _ in MODELS}
        fig.text(0.07, 0.914, f"{stamp} · 27B {totals['Qwen3.5-27B']}/160 teams · 2B {totals['Qwen3.5-2B']}/160 teams",
                 fontsize=10, color=MUTED)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(0.98, 0.937), ncol=2, frameon=False, fontsize=10)
        high = next(row for row in summary if row["model"] == "Qwen3.5-2B" and row["n"] == 16)
        fig.text(0.07, 0.108,
                 f"Efficiency = success rate × N × {reference_tokens:,} / mean team output tokens. "
                 "Same valid teams in both panels, successes and failures.", fontsize=9, color=MUTED)
        fig.text(0.07, 0.065,
                 f"{reference_tokens:,} tokens/agent is a reference scale, not an enforced budget. "
                 "Thinking off · max_channels: 1 · no intervals.", fontsize=9, color=MUTED)
        caveat = (f"*2B N=16: {high['teams']}/32 trees recorded; selective/incomplete in BOTH panels. "
                  "Unresolved trees are excluded, not failures." if high["unresolved"] else
                  "32 teams per point, one per tree; both panels include successes and failures.")
        fig.text(0.07, 0.022, caveat, fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0.015, 0.17, 0.995, 0.865), w_pad=3.5)
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        svg = path.with_suffix(".svg")
        fig.savefig(svg, facecolor=SURFACE, metadata={"Date": metadata["at"]})
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("teams", type=Path)
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--reference-tokens", type=int, default=9216)
    args = parser.parse_args()
    metadata = json.loads(args.capture.read_text())
    if hashlib.sha256(args.teams.read_bytes()).hexdigest() != metadata["cohort_sha256"]:
        raise ValueError("Cohort hash differs from audited capture")
    summary = summarize(args.teams, args.reference_tokens)
    figure(args.out, summary, metadata, args.reference_tokens)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)
    print(metadata["at"])
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
