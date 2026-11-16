# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The sweep's figure, CSVs and table from `<root>/rows.jsonl` (sweep.py's rows, one per team):

    uv run experiments/s3-aggregation/sweep/plot.py <root>

- `scaling.png`, five panels against the swarm size N: success (%) in the style of the s³
  paper's Figure 5 (information aggregation), with 95% Wilson intervals over the problems; then
  what the team costs: input and generated tokens per team and board posts per agent (means
  with 95% intervals), and wall time per team (median and interquartile range: a team in flight
  across a driver restart waited for it). The paper's right panel is token efficiency, which
  needs a token allowance our agents don't have; its Figure 10 shows tokens per team;
- `teams.csv`: one row per team; `by_n.csv` and `by_n.md`: per N, the teams, success rate and
  interval, mean reward, tokens per team (input, cached, output), board posts per agent and
  the median wall time.
"""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

SURFACE, INK, INK_2, GRID, SERIES = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    m = statistics.fmean(values)
    half = 1.96 * statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return (m, m - half, m + half)


def team_row(r: dict) -> dict:
    rewards = [v for v in r["rewards"].values() if v is not None]
    t = r["tokens"]
    return {"task": r["task"], "problem": r.get("problem"), "agents": r["agents"],
            "success": int(bool(r["success"])), "mean_reward": statistics.fmean(rewards)
            if rewards else None, "input_uncached": t["input"], "cached": t["cache_read"],
            "output": t["output"], "posts": r["posts"],
            "posts_per_agent": r["posts"] / r["agents"], "wall_s": r.get("wall_s"),
            "duration_s": r["duration_s"], "outcome": r.get("outcome"),
            "reruns": r.get("reruns", 0), "incarnations": r.get("incarnations"),
            "missing": ";".join(r["missing"])}


def by_n(teams: list[dict]) -> list[dict]:
    out = []
    for n in sorted({t["agents"] for t in teams}):
        group = [t for t in teams if t["agents"] == n]
        k = sum(t["success"] for t in group)
        low, high = wilson(k, len(group))
        rewards = [t["mean_reward"] for t in group if t["mean_reward"] is not None]
        out.append({
            "agents": n, "teams": len(group), "successes": k,
            "success_rate": k / len(group), "ci_low": low, "ci_high": high,
            "mean_reward": statistics.fmean(rewards) if rewards else math.nan,
            "input_uncached_per_team": statistics.fmean(t["input_uncached"] for t in group),
            "cached_per_team": statistics.fmean(t["cached"] for t in group),
            "output_per_team": statistics.fmean(t["output"] for t in group),
            "posts_per_agent": statistics.fmean(t["posts_per_agent"] for t in group),
            "wall_s": statistics.median(t["wall_s"] for t in group if t["wall_s"] is not None),
            "incomplete": sum(bool(t["missing"]) for t in group),
            "reruns": sum(t["reruns"] for t in group)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def table(summary: list[dict]) -> str:
    head = ("| N | teams | success | 95% CI | mean reward | input / cached / output tokens per"
            " team | posts per agent | wall time |")
    lines = [head, "|---|---|---|---|---|---|---|---|"]
    for s in summary:
        lines.append(
            f"| {s['agents']} | {s['teams']} | {s['successes']}/{s['teams']} "
            f"({100 * s['success_rate']:.0f}%) | {100 * s['ci_low']:.0f}-{100 * s['ci_high']:.0f}% "
            f"| {s['mean_reward']:.2f} | {s['input_uncached_per_team'] / 1e3:.0f}k / "
            f"{s['cached_per_team'] / 1e3:.0f}k / {s['output_per_team'] / 1e3:.1f}k "
            f"| {s['posts_per_agent']:.1f} | {s['wall_s'] / 60:.1f} min (median) |")
    return "\n".join(lines) + "\n"


def quartiles(values: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return (statistics.median(values), q[0], q[2])


def figure(path: Path, teams: list[dict], summary: list[dict], title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    ns = [s["agents"] for s in summary]
    by_n = {n: [t for t in teams if t["agents"] == n] for n in ns}
    rate = [(100 * s["success_rate"], 100 * s["ci_low"], 100 * s["ci_high"]) for s in summary]
    panels = [
        ("Success (%)", rate, (0, 105)),
        ("Input tokens per team (M)",
         [tuple(v / 1e6 for v in mean_ci([t["input_uncached"] + t["cached"] for t in by_n[n]]))
          for n in ns], None),
        ("Generated tokens per team (k)",
         [tuple(v / 1e3 for v in mean_ci([t["output"] for t in by_n[n]])) for n in ns], None),
        ("Wall time per team (min)",
         [tuple(v / 60 for v in quartiles([t["wall_s"] for t in by_n[n] if t["wall_s"]]))
          for n in ns], None),
        ("Board posts per agent", [mean_ci([t["posts_per_agent"] for t in by_n[n]]) for n in ns],
         None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.7), facecolor=SURFACE)
    for ax, (label, points, ylim) in zip(axes, panels, strict=True):
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xticks(ns)
        ax.set_xlabel("Swarm size N")
        ax.set_ylabel(label)
        middle = [p[0] for p in points]
        ax.errorbar(ns, middle, yerr=[[p[0] - p[1] for p in points], [p[2] - p[0] for p in points]],
                    color=SERIES, linewidth=2, marker="o", markersize=7, capsize=3, elinewidth=1.5)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(bottom=0)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "One attempt per problem. Success: share of teams with any agent right, "
             "95% Wilson intervals. Tokens and posts: means, 95% intervals. Wall time (init to "
             "release): median and interquartile range.", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def main(argv=None) -> int:
    root = Path((argv or sys.argv[1:])[0])
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines() if line]
    teams = sorted((team_row(r) for r in rows), key=lambda t: (t["agents"], t["problem"]))
    summary = by_n(teams)
    write_csv(root / "teams.csv", teams)
    write_csv(root / "by_n.csv", summary)
    (root / "by_n.md").write_text(table(summary))
    figure(root / "scaling.png", teams, summary,
           "s3-aggregation with flock: success and cost against swarm size (Qwen3.5-27B)")
    print(table(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
