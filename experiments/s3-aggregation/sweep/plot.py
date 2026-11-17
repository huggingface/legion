# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The sweep's figure, CSVs and table from sweep.py's rows (`<root>/rows.jsonl`, one per team):

    uv run experiments/s3-aggregation/sweep/plot.py <root>                  # one grid, into <root>
    uv run experiments/s3-aggregation/sweep/plot.py --series Qwen3.5-27B=<root> \
        --series Qwen3.5-2B=<root> --out <dir>                             # one series per model

- `scaling.png`, five panels against the swarm size N: success (%) in the style of the s³
  paper's Figure 5 (information aggregation), with 95% Wilson intervals over the problems; then
  what the team costs: input and generated tokens per team and board posts per agent (means
  with 95% intervals), and wall time per team (median and interquartile range: a team in flight
  across a driver restart waited for it). The paper's right panel is token efficiency, which
  needs a token allowance our agents don't have; its Figure 10 shows tokens per team;
- `teams.csv`: one row per team; `by_n.csv` and `by_n.md`: per N, the teams, success rate and
  interval, mean reward, tokens per team (input, cached when the server reports it, output),
  board posts per agent and the median wall time.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # the reference palette's first slots, in order


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


def table(series: list[tuple[str, list[dict], list[dict]]]) -> str:
    # a server that reports no cache hits (vLLM by default) gets no cached column
    cached = any(s["cached_per_team"] for _, _, summary in series for s in summary)
    tokens = "input / cached / output" if cached else "input / output"
    head = (f"| model | N | teams | success | 95% CI | mean reward | {tokens} tokens per team "
            "| posts per agent | median wall time |")
    lines = [head, "|---|---|---|---|---|---|---|---|---|"]
    for model, _, summary in series:
        lines += [row(model, s, cached) for s in summary]
    return "\n".join(lines) + "\n"


def row(model: str, s: dict, cached: bool) -> str:
    tokens = (f"{s['input_uncached_per_team'] / 1e3:.0f}k / "
              + (f"{s['cached_per_team'] / 1e3:.0f}k / " if cached else "")
              + f"{s['output_per_team'] / 1e3:.1f}k")
    return (f"| {model} | {s['agents']} | {s['teams']} | {s['successes']}/{s['teams']} "
            f"({100 * s['success_rate']:.0f}%) | {100 * s['ci_low']:.0f}-{100 * s['ci_high']:.0f}% "
            f"| {s['mean_reward']:.2f} | {tokens} | {s['posts_per_agent']:.1f} "
            f"| {s['wall_s'] / 60:.1f} min |")


def quartiles(values: list[float]) -> tuple[float, float, float]:
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return (statistics.median(values), q[0], q[2])


def panels(teams: list[dict], summary: list[dict]) -> list[tuple]:
    """(label, [(middle, low, high) per N], y limits or "log") for each panel: tokens on a log
    scale, since two models' teams can differ by more than an order of magnitude."""
    ns = [s["agents"] for s in summary]
    by_n = {n: [t for t in teams if t["agents"] == n] for n in ns}
    return [
        ("Success (%)", [(100 * s["success_rate"], 100 * s["ci_low"], 100 * s["ci_high"])
                         for s in summary], (0, 105)),
        ("Input tokens per team",
         [mean_ci([t["input_uncached"] + t["cached"] for t in by_n[n]]) for n in ns], "log"),
        ("Generated tokens per team", [mean_ci([t["output"] for t in by_n[n]]) for n in ns],
         "log"),
        ("Wall time per team (min)",
         [tuple(v / 60 for v in quartiles([t["wall_s"] for t in by_n[n] if t["wall_s"]]))
          for n in ns], None),
        ("Board posts per agent", [mean_ci([t["posts_per_agent"] for t in by_n[n]]) for n in ns],
         None),
    ]


def tokens(value: float, _=None) -> str:
    """A log axis tick as a token count: 100, 10k, 1M."""
    for size, unit in ((1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


def figure(path: Path, series: list[tuple[str, list[dict], list[dict]]], title: str,
           faint: frozenset[str] = frozenset(), note: str = "") -> None:
    """One line per series, a colour per model (a label's first word); a `faint` series (an
    earlier run, for comparison) is drawn lighter and dashed or dotted."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, axes = plt.subplots(1, 5, figsize=(17, 4.4), facecolor=SURFACE)
    models = list(dict.fromkeys(label.split()[0] for label, _, _ in series))
    dashes = iter(["--", ":", "-."] * len(series))
    for k, (model, teams, summary) in enumerate(series):
        ns = [s["agents"] for s in summary]
        offset = (k - (len(series) - 1) / 2) * 0.14  # side by side, so the bars don't hide
        light = model in faint
        style = {"color": SERIES[models.index(model.split()[0])], "label": model,
                 "linewidth": 1.4 if light else 2.2, "markersize": 5 if light else 7,
                 "alpha": 0.45 if light else 1.0, "linestyle": next(dashes) if light else "-",
                 "zorder": 2 if light else 3}
        for ax, (label, points, scale) in zip(axes, panels(teams, summary), strict=True):
            ax.errorbar([n + offset for n in ns], [p[0] for p in points],
                        yerr=[[max(0.0, p[0] - p[1]) for p in points],  # Wilson at 0/n: 1e-15
                              [max(0.0, p[2] - p[0]) for p in points]],
                        marker="o", capsize=3, elinewidth=1.2 if light else 1.5, **style)
            ax.set_ylabel(label)
            ax.set_xticks(ns)
    for ax, (_, _, scale) in zip(axes, panels(*series[0][1:]), strict=True):  # every series in
        if scale == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(tokens))
        elif scale:
            ax.set_ylim(*scale)
        else:
            ax.set_ylim(bottom=0)
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Swarm size N")
    if len(series) > 1:
        fig.legend(*axes[0].get_legend_handles_labels(), loc="upper left", ncol=len(series),
                   bbox_to_anchor=(0.01, 0.935), frameon=False, fontsize=9)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.01, "One attempt per problem. Success: share of teams with any agent right, "
             "95% Wilson intervals. Tokens (log scale) and posts: means, 95% intervals. Wall time (init "
             "to release): median and interquartile range." + (f" {note}" if note else ""),
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.87))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def load(root: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines() if line]
    teams = sorted((team_row(r) for r in rows), key=lambda t: (t["agents"], t["problem"]))
    return teams, by_n(teams)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The sweep's figure, CSVs and table.")
    parser.add_argument("root", nargs="?", type=Path, help="one grid; outputs go there too")
    parser.add_argument("--series", action="append", default=[], help="MODEL=ROOT, repeatable")
    parser.add_argument("--out", type=Path, help="where --series outputs go")
    parser.add_argument("--faint", action="append", default=[],
                        help="a --series' MODEL drawn lighter, for comparison; repeatable")
    parser.add_argument("--title", help="the figure's title")
    parser.add_argument("--note", default="", help="added to the figure's footnote")
    args = parser.parse_args(argv)
    if args.root:  # the grid's model, from one of its team specs
        spec = next(args.root.glob("src/*/spec.yaml"), None)
        model = next((x.split(": ", 1)[1].strip() for x in spec.read_text().splitlines()
                      if x.startswith("model: ")), args.root.name) if spec else args.root.name
        sources, out = [(model.split("/")[-1], args.root)], args.root
    else:
        sources = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.series]
        out = args.out
        out.mkdir(parents=True, exist_ok=True)
    series = [(model, *load(root)) for model, root in sources]
    write_csv(out / "teams.csv", [{"model": m} | t for m, teams, _ in series for t in teams])
    write_csv(out / "by_n.csv", [{"model": m} | s for m, _, summary in series for s in summary])
    (out / "by_n.md").write_text(table(series))
    figure(out / "scaling.png", series, args.title or "s3-aggregation with flock: success and "
           f"cost against swarm size ({', '.join(m for m, _, _ in series)})",
           frozenset(args.faint), args.note)
    print(table(series))
    return 0


if __name__ == "__main__":
    sys.exit(main())
