"""Compare latest valid trial token measurements with total elapsed time."""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter


def ranks(values):
    values = np.asarray(values)
    return np.array([np.sum(values < v) + (np.sum(values == v) - 1) / 2 for v in values])


def correlations(rows):
    x = np.array([r["elapsed_seconds"] / 60 for r in rows])
    y = np.array([r["normalized_tokens"] for r in rows])
    return {"n": len(rows), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "spearman_rho": float(np.corrcoef(ranks(x), ranks(y))[0, 1])}


def generated_rows(rows):
    """Replace the plotted metric with reported completion tokens, never input tokens."""
    converted = []
    for row in rows:
        output = None
        if row["measurement_valid"]:
            report = json.loads((Path(row["attempt"]) / "result.json").read_text())
            accounting = report["accounting"]
            assert accounting["complete"]
            output = accounting["output_tokens"]
            assert output >= 0
            assert accounting["input_tokens"] + output == row["total_tokens"]
        converted.append({**row, "generated_tokens": output,
                          "normalized_tokens": output / row["agents"]
                          if output is not None else None})
    return converted


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--generated-only", action="store_true")
    args = parser.parse_args()
    rows = json.loads(args.results.read_text())
    if args.generated_only:
        rows = generated_rows(rows)
    metric = "Generated tokens" if args.generated_only else "Total tokens"
    prefix = "generated-tokens-vs-time" if args.generated_only else "tokens-vs-time"
    valid = [r for r in rows if r["measurement_valid"] and r["normalized_tokens"] is not None
             and r["elapsed_seconds"] is not None]
    solved = [r for r in valid if r["success"]]
    stats = {"all_valid": correlations(valid), "successful": correlations(solved),
             "successful_by_difficulty": {
                 str(d): correlations([r for r in solved if r["difficulty"] == d])
                 for d in [5, 10, 15, 20, 25]},
             "excluded": len(rows) - len(valid),
             "time_metric": "elapsed_seconds including shutdown/draining",
             "token_metric": metric + " / configured agents including shutdown/draining"}
    colors = {5: "#217dc0", 10: "#0d9488", 15: "#d97706", 20: "#9333ea", 25: "#db2777"}
    ink, muted = "#1e293b", "#64748b"
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(left=.085, right=.98, bottom=.22, top=.72, wspace=.26)
    group, title = valid, "All runs with complete accounting"
    for d, color in colors.items():
        for success, marker in [(True, "o"), (False, "X")]:
            subset = [r for r in group if r["difficulty"] == d and r["success"] == success]
            ax.scatter([r["elapsed_seconds"] / 60 for r in subset],
                       [r["normalized_tokens"] for r in subset], color=color,
                       marker=marker, s=55 if success else 85, alpha=.8,
                       edgecolors="white", linewidths=.5, zorder=3)
    c = correlations(group)
    ax.set_title(title, loc="left", color=ink, fontsize=15, weight="bold", pad=32)
    ax.text(0, 1.035, f"n = {c['n']}  ·  Pearson r = {c['pearson_r']:.3f}  ·  "
            f"Spearman ρ = {c['spearman_rho']:.3f}",
            transform=ax.transAxes, color=muted, fontsize=10)
    ax.set_xlabel("Elapsed time including shutdown (minutes)", color=muted)
    ax.set_ylabel(metric + " / configured agents", color=muted)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.grid(color="#dde5ef", alpha=.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=muted)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#cbd5e1")
    fig.suptitle(f"AgentEscapeBench: {metric.lower()} per agent vs. time", x=.085,
                 y=.97, ha="left", fontsize=19, weight="bold", color=ink)
    fig.text(.085, .9, "Pi · Qwen3.8-27B · low reasoning · scenario 0 · latest attempt per trial",
             color=muted, fontsize=11)
    handles = [Line2D([0], [0], marker="o", linestyle="none", color=c,
                      label=f"Difficulty {d}") for d, c in colors.items()]
    handles.append(Line2D([0], [0], marker="X", linestyle="none", color=muted,
                          label="Time budget exhausted"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.078, .87),
               ncols=6, frameon=False, fontsize=9)
    fig.text(.085, .12, "Pearson measures linear association; Spearman measures rank association. "
             f"Each point is one trial; {len(rows) - len(valid)} incomplete measurements are excluded.",
             color=muted, fontsize=8)
    fig.text(.085, .075, "Pooled correlations mix difficulty and agent count. "
             "Budget-exhausted runs are time-censored; they are not times to solution.",
             color=muted, fontsize=9)
    root = args.results.parent
    for ext in ["png", "pdf", "svg"]:
        fig.savefig(root / f"{prefix}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)
    (root / f"{prefix}-statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
