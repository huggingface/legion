# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The s3-maze sweep's figure, CSVs and table from sweep.py's rows (`<root>/rows.jsonl`):

    uv run flock/sweep/plot.py --series Qwen3.5-27B=<root> --series Qwen3.5-2B=<root> --out <dir>

- `scaling.png`, eight panels, one series per model, after the s³ paper's maze results
  (Figure 4, bottom: success against swarm size; Figure 5c: success by the branch that holds
  the exit; Appendix M: room overlap):
  - success (%) against N, with 95% Wilson intervals over the trees, and best-of-N from the
    model's own N = 1 rate (1 - (1 - p1)^N, the paper's independent-attempts baseline) dashed;
  - the team's moves and the time (from its Trials' start) to the first exit, over the teams
    that got out: median and interquartile range;
  - room overlap, (V - U) / V, mean and 95% interval (N >= 2);
  - input and generated tokens per team (log scale): median and interquartile range;
  - board posts per agent: mean and 95% interval;
  - success by which of the start room's three doors, in the listed order, leads to the exit,
    over every N >= 2 team (95% Wilson intervals).
- `teams.csv` (one row per team), `by_n.csv` and `by_n.md` (per model and N).
"""

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
LIGHT: set[str] = set()  # the series drawn lighter (`--series NAME=ROOT@light`)


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


def quartiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return (math.nan, math.nan, math.nan)
    q = statistics.quantiles(values, n=4) if len(values) > 1 else [values[0]] * 3
    return (statistics.median(values), q[0], q[2])


def team_row(r: dict) -> dict:
    t, first = r["tokens"], r.get("first_exit") or {}
    return {"task": r["task"], "tree": r["tree"], "agents": r["agents"],
            "exit_branch": r["exit_branch"], "success": int(bool(r["success"])),
            "exit_by": first.get("agent"), "exit_after_s": first.get("after_s"),
            "exit_team_moves": first.get("team_moves"),
            "exit_agent_moves": first.get("agent_moves"),
            "moves": sum((r.get("moves") or {}).values()),
            "refused_moves": sum((r.get("refused_moves") or {}).values()),
            "room_overlap": r.get("room_overlap"), "rooms_union": r.get("rooms_union"),
            "input_uncached": t["input"], "cached": t["cache_read"], "output": t["output"],
            "posts": r["posts"], "posts_per_agent": r["posts"] / r["agents"],
            "edges": len(r["edges"]), "outcome": r.get("outcome"), "wall_s": r.get("wall_s"),
            "duration_s": r["duration_s"], "reruns": r.get("reruns", 0),
            "worker_wait_s": r.get("worker_wait_s"), "run": r["run"]}


def by_n(teams: list[dict]) -> list[dict]:
    """Per N. A tree with no team that counted at that N (every attempt set aside) is
    `missing`: left out of the rates, not a failure."""
    out, p1 = [], None
    trees = {t["tree"] for t in teams}
    for n in sorted({t["agents"] for t in teams}):
        group = [t for t in teams if t["agents"] == n]
        missing = sorted(trees - {t["tree"] for t in group})
        won = [t for t in group if t["success"]]
        k = len(won)
        low, high = wilson(k, len(group))
        if n == 1:
            p1 = k / len(group)
        overlap = [t["room_overlap"] for t in group if t["room_overlap"] is not None]
        moves, minutes = (quartiles([t["exit_team_moves"] for t in won]),
                          quartiles([t["exit_after_s"] / 60 for t in won]))
        out.append({
            "agents": n, "teams": len(group), "missing": len(missing),
            "missing_trees": " ".join(f"t{x:02d}" for x in missing),
            "successes": k, "success_rate": k / len(group),
            "ci_low": low, "ci_high": high,
            "best_of_n_from_n1": 1 - (1 - p1) ** n if p1 is not None else math.nan,
            "exit_team_moves_median": moves[0], "exit_team_moves_q1": moves[1],
            "exit_team_moves_q3": moves[2],
            "exit_agent_moves_median": quartiles([t["exit_agent_moves"] for t in won])[0],
            "exit_minutes_median": minutes[0], "exit_minutes_q1": minutes[1],
            "exit_minutes_q3": minutes[2],
            "moves_per_agent": statistics.fmean(t["moves"] / n for t in group),
            "room_overlap": statistics.fmean(overlap) if overlap else math.nan,
            "input_per_team": statistics.fmean(t["input_uncached"] + t["cached"] for t in group),
            "output_per_team": statistics.fmean(t["output"] for t in group),
            "input_per_team_median": statistics.median(t["input_uncached"] + t["cached"]
                                                       for t in group),
            "output_per_team_median": statistics.median(t["output"] for t in group),
            "posts_per_agent": statistics.fmean(t["posts_per_agent"] for t in group),
            "wall_minutes_median": statistics.median(t["wall_s"] / 60 for t in group
                                                     if t["wall_s"] is not None),
            "reruns": sum(t["reruns"] for t in group)})
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def table(series) -> str:
    lines = [("| model | N | teams | success | 95% CI | best-of-N from N=1 | team moves to exit "
              "| minutes to exit | moves per agent | median input / output tokens per team "
              "| posts per agent | room overlap |"),
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for model, _, summary in series:
        for s in summary:
            def num(x, f="{:.0f}"):
                return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f.format(x)
            gone = f" ({s['missing']} missing)" if s["missing"] else ""
            lines.append(
                f"| {model} | {s['agents']} | {s['teams']}{gone} | {s['successes']}/{s['teams']} "
                f"({100 * s['success_rate']:.0f}%) | {100 * s['ci_low']:.0f}-"
                f"{100 * s['ci_high']:.0f}% | {num(100 * s['best_of_n_from_n1'], '{:.0f}%')} "
                f"| {num(s['exit_team_moves_median'])} | {num(s['exit_minutes_median'], '{:.1f}')} "
                f"| {s['moves_per_agent']:.0f} | {s['input_per_team_median'] / 1e6:.1f}M / "
                f"{s['output_per_team_median'] / 1e3:.0f}k | {s['posts_per_agent']:.1f} "
                f"| {num(s['room_overlap'], '{:.2f}')} |")
    return "\n".join(lines) + "\n"


def panels(teams: list[dict], summary: list[dict]) -> list[tuple]:
    """(label, x values, [(middle, low, high)], y scale, dashed reference or None)."""
    ns = [s["agents"] for s in summary]
    group = {n: [t for t in teams if t["agents"] == n] for n in ns}
    won = {n: [t for t in group[n] if t["success"]] for n in ns}
    multi = [n for n in ns if n > 1]
    branches = [1, 2, 3]
    by_branch = [[t for t in teams if t["agents"] > 1 and t["exit_branch"] == b] for b in branches]
    return [
        ("Success (%)", ns, [(100 * s["success_rate"], 100 * s["ci_low"], 100 * s["ci_high"])
                             for s in summary], (0, 105),
         [100 * s["best_of_n_from_n1"] for s in summary]),
        ("Team moves to the first exit", ns,
         [quartiles([t["exit_team_moves"] for t in won[n]]) for n in ns], None, None),
        ("Minutes to the first exit", ns,
         [quartiles([t["exit_after_s"] / 60 for t in won[n]]) for n in ns], None, None),
        ("Room overlap", multi,
         [mean_ci([t["room_overlap"] for t in group[n] if t["room_overlap"] is not None])
          for n in multi], (0, 1), None),
        ("Input tokens per team", ns,
         [quartiles([t["input_uncached"] + t["cached"] for t in group[n]]) for n in ns], "log",
         None),
        ("Generated tokens per team", ns, [quartiles([t["output"] for t in group[n]]) for n in ns],
         "log", None),
        ("Board posts per agent", ns, [mean_ci([t["posts_per_agent"] for t in group[n]])
                                       for n in ns], None, None),
        ("Success (%) by the exit's branch, N ≥ 2", branches,
         [(100 * sum(t["success"] for t in g) / len(g) if g else math.nan,
           *(100 * v for v in wilson(sum(t["success"] for t in g), len(g)))) for g in by_branch],
         (0, 105), None),
    ]


def tokens(value: float, _=None) -> str:
    for size, unit in ((1e6, "M"), (1e3, "k")):
        if value >= size:
            return f"{value / size:g}{unit}"
    return f"{value:g}"


def figure(path: Path, series, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, grid = plt.subplots(2, 4, figsize=(17, 7.8), facecolor=SURFACE)
    axes = grid.flatten()
    xs_by_panel: dict[int, set] = {}
    colours, c = [], -1  # a light series (`@light`) takes the colour of the series before it
    for model, _, _ in series:
        c += model not in LIGHT or c < 0
        colours.append(SERIES[c])
    for k, (model, teams, summary) in enumerate(series):
        light = model in LIGHT
        look = ({"alpha": 0.4, "linewidth": 1.3, "markersize": 4, "elinewidth": 1.0} if light
                else {"linewidth": 2, "markersize": 6, "elinewidth": 1.4})
        step = (k - (len(series) - 1) / 2)  # side by side, so the bars don't hide each other
        for i, (ax, (label, xs, points, scale, reference)) in enumerate(
                zip(axes, panels(teams, summary), strict=True)):
            xs_by_panel.setdefault(i, set()).update(xs)
            # N on a log2 axis: shift by a factor; the branches (panel 7) by a step
            moved = [x + 0.08 * step if i == 7 else x * 2 ** (0.07 * step) for x in xs]
            ax.errorbar(moved, [p[0] for p in points],
                        yerr=[[max(0.0, p[0] - p[1]) if not math.isnan(p[1]) else 0 for p in points],
                              [max(0.0, p[2] - p[0]) if not math.isnan(p[2]) else 0 for p in points]],
                        color=colours[k], marker="o", capsize=3, label=model,
                        linestyle="-" if i != 7 else "none", **look)
            if reference and not light:
                ax.plot(xs, reference, color=colours[k], linewidth=1.3, linestyle="--",
                        label=f"{model}: best-of-N from N = 1")
            ax.set_ylabel(label)
    scales = [p[3] for p in panels(*series[0][1:])]
    for i, ax in enumerate(axes):  # after every series is in, so each scales to all of them
        if scales[i] == "log":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(tokens))
            ax.yaxis.set_minor_formatter(NullFormatter())
        elif scales[i]:
            ax.set_ylim(*scales[i])
        else:
            ax.set_ylim(bottom=0)
        ticks = sorted(xs_by_panel.get(i, ()))
        if i != 7 and ticks:
            ax.set_xlim(min(ticks) / 1.4, max(ticks) * 1.4)
            ax.set_xscale("log", base=2)
            ax.xaxis.set_minor_formatter(NullFormatter())
        ax.xaxis.set_major_locator(FixedLocator(ticks))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Door at the start room (listed order)" if i == 7 else "Swarm size N")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title, fontsize=11, color=INK, x=0.01, ha="left")
    gone = "".join(f"; {m} at N = {s['agents']}: {s['missing']} missing (every attempt set "
                   "aside), left out, not failures" for m, _, summary in series
                   for s in summary if s["missing"])
    fig.text(0.01, 0.012, f"One counted team per tree (32 trees{gone}). Success: share of "
             "teams in which any agent reached the exit, 95% Wilson intervals; dashed: best-of-N "
             "from the model's own N = 1 rate.\nMoves (all agents') and minutes (from the "
             "Trials' start) to the first exit: teams that got out; tokens per team (log scale): "
             "every team; medians and interquartile ranges. Overlap and posts: means, 95% "
             "intervals.", fontsize=8, color=INK_2, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def load(root: Path) -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines() if line]
    teams = sorted((team_row(r) for r in rows), key=lambda t: (t["agents"], t["tree"]))
    return teams, by_n(teams)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The sweep's figure, CSVs and table.")
    parser.add_argument("--series", action="append", required=True,
                        help="NAME=ROOT, repeatable; NAME=ROOT@light draws it lighter, in the "
                             "colour of the series before it, without its best-of-N line")
    parser.add_argument("--title", default="s3-maze with flock: success and cost against swarm "
                                           "size")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = []
    for s in args.series:
        name, root = s.split("=", 1)
        if root.endswith("@light"):
            root = root.removesuffix("@light")
            LIGHT.add(name)
        sources.append((name, Path(root)))
    args.out.mkdir(parents=True, exist_ok=True)
    series = [(model, *load(root)) for model, root in sources]
    write_csv(args.out / "teams.csv", [{"model": m} | t for m, teams, _ in series for t in teams])
    write_csv(args.out / "by_n.csv", [{"model": m} | s for m, _, summary in series
                                      for s in summary])
    (args.out / "by_n.md").write_text(table(series))
    figure(args.out / "scaling.png", series, args.title)
    print(table(series))
    return 0


if __name__ == "__main__":
    sys.exit(main())
