# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""Plot success and reference-normalized token efficiency from a finalized teams.csv.

The s³ paper defines efficiency as success_rate * N * L / mean_team_generated_tokens,
including failed teams. These Flock runs have no fixed token allowance: L is an explicitly
labelled reference scale, not an enforced budget. Only generated (output) tokens count.

    uv run experiments/s3-aggregation/sweep/efficiency.py \
        experiments/s3-aggregation/results/latest/teams.csv \
        --out experiments/s3-aggregation/results/latest/success-efficiency.png \
        --reference-tokens 9216
"""

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from matplotlib import rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ("#2a78d6", "#eb6834", "#1baf7a")


def summarize(path: Path, reference_tokens: int) -> list[dict]:
    if reference_tokens <= 0:
        raise ValueError("reference tokens must be positive")
    groups = defaultdict(list)
    seen = set()
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["model"], row["task"])
            if key in seen:
                raise ValueError(f"duplicate team: {key}")
            seen.add(key)
            n, success, output = int(row["agents"]), int(row["success"]), int(row["output"])
            if n <= 0 or success not in (0, 1) or output <= 0 or row["missing"]:
                raise ValueError(f"invalid or incomplete counted team: {key}")
            groups[row["model"], n].append((success, output))
    if not groups:
        raise ValueError("no teams found")
    summary = []
    for (model, n), teams in groups.items():
        successes = sum(success for success, _ in teams)
        mean_output = statistics.fmean(output for _, output in teams)
        success_rate = successes / len(teams)
        summary.append({
            "model": model, "agents": n, "teams": len(teams), "successes": successes,
            "success_rate": success_rate, "mean_output_tokens_per_team": mean_output,
            "mean_output_tokens_per_agent": mean_output / n,
            "reference_tokens_per_agent": reference_tokens,
            "token_efficiency": success_rate * n * reference_tokens / mean_output,
        })
    return summary


def figure(path: Path, summary: list[dict], reference_tokens: int) -> None:
    with rc_context({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK}):
        fig = Figure(figsize=(11.8, 4.8), facecolor=SURFACE)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        for index, model in enumerate(dict.fromkeys(row["model"] for row in summary)):
            points = sorted((row for row in summary if row["model"] == model),
                            key=lambda row: row["agents"])
            ns = [row["agents"] for row in points]
            style = {"marker": "o", "linewidth": 2.3, "markersize": 7,
                     "color": COLORS[index % len(COLORS)], "label": model.split(",", 1)[0]}
            axes[0].plot(ns, [100 * row["success_rate"] for row in points], **style)
            axes[1].plot(ns, [row["token_efficiency"] for row in points], **style)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ns = sorted({row["agents"] for row in summary})
            ax.set_xticks(ns, labels=[str(n) for n in ns])
            ax.set_xlabel("Swarm size N", labelpad=8)
            ax.set_facecolor(SURFACE)
            ax.grid(axis="y", color=GRID, linewidth=0.8)
            ax.set_axisbelow(True)
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylim(0, 105)
        axes[0].set_ylabel("Success (%)")
        axes[0].set_title("Success (%) ↑", loc="left", pad=14, fontsize=13)
        axes[1].set_ylim(bottom=0)
        axes[1].set_ylabel("Reference-normalized efficiency")
        axes[1].set_title("Token efficiency ↑", loc="left", pad=14, fontsize=13)
        fig.suptitle("s3-aggregation: success and token efficiency", x=0.07, ha="left",
                     fontsize=16, y=0.98)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right",
                   bbox_to_anchor=(0.98, 0.99), ncol=2, frameon=False, fontsize=10)
        fig.text(0.07, 0.075,
                 f"Efficiency = success rate × N × {reference_tokens:,} / mean team output tokens. "
                 "All 32 teams per N included, successes and failures.", fontsize=9, color=MUTED)
        fig.text(0.07, 0.028,
                 f"{reference_tokens:,} tokens/agent is a reference scale, not an enforced budget. "
                 "Thinking off · max_channels: 1 · no intervals.", fontsize=9, color=MUTED)
        fig.tight_layout(rect=(0.015, 0.13, 0.995, 0.89), w_pad=3.5)
        fig.savefig(path, dpi=200, facecolor=SURFACE)
        svg = path.with_suffix(".svg")
        fig.savefig(svg, facecolor=SURFACE)
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("teams", type=Path, help="finalized per-team CSV, including failures")
    parser.add_argument("--out", required=True, type=Path, help="combined PNG; also writes SVG/CSV")
    parser.add_argument("--reference-tokens", type=int, default=9216,
                        help="reference tokens per agent, not an enforced allowance (default: 9216)")
    args = parser.parse_args()
    summary = summarize(args.teams, args.reference_tokens)
    figure(args.out, summary, args.reference_tokens)
    with args.out.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary)


if __name__ == "__main__":
    main()
