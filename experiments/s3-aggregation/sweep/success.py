# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""One panel for a summary: success (%) against swarm size N per model, the share of teams with
any agent right, as plain points and lines (plot.py draws the full panels, with 95% intervals,
from the same rows):

    uv run experiments/s3-aggregation/sweep/success.py --series Qwen3.5-2B=<root> \
        --series Qwen3.5-27B=<root> --out success.png
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--series", action="append", required=True, help="MODEL=ROOT, repeatable")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="s3-aggregation: success against swarm size")
    parser.add_argument("--note", default="", help="added to the footnote")
    args = parser.parse_args(argv)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": plot.GRID, "axes.labelcolor": plot.INK_2,
                         "xtick.color": plot.INK_2, "ytick.color": plot.INK_2,
                         "text.color": plot.INK})
    fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor=plot.SURFACE)
    sources = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.series]
    for k, (model, root) in enumerate(sources):
        _, summary = plot.load(root)
        ax.plot([s["agents"] for s in summary], [100 * s["success_rate"] for s in summary],
                marker="o", linewidth=2.2, markersize=7, color=plot.SERIES[k], label=model)
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16], labels=["1", "2", "4", "8", "16"])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Swarm size N")
    ax.set_ylabel("Success (%)")
    ax.set_facecolor(plot.SURFACE)
    ax.grid(axis="y", color=plot.GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="center right")
    ax.set_title(args.title, fontsize=10.5, color=plot.INK, loc="left")
    fig.text(0.01, 0.01, "32 problems per N, one attempt each." + (f" {args.note}" if args.note else ""),
             fontsize=8, color=plot.INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(args.out, dpi=200, facecolor=plot.SURFACE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
