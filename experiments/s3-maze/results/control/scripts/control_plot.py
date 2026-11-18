# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8"]
# ///
"""The s3-maze positive control's figure and table: Qwen3.5-27B at N = 4 and 8 on every tree,
three conditions (the scaling sweep's runs as the baseline: board, no hint; board + protocol
hint; board off), from each grid's rows (and timing.py's per-agent lines):

    uv run control_plot.py --condition baseline=<root> --condition hint=<root> \
        --condition "board off=<root>" --timing timing.jsonl --dataset <s3-maze checkout> --out <dir>

Writes `control.png` (six panels), `control.csv` and `control.md` (per condition and N).
- success with 95% Wilson intervals, against best-of-N from the 27B's N = 1 rate (`--p1`);
- success by the start room's door (listed order) that leads to the exit, N = 4 and 8 pooled;
- room overlap and posts per agent (means, 95% intervals);
- which start-room door each agent's first move takes (N = 4 and 8 pooled): a split team spreads;
- agents that read or posted on the board before their first move (timing.py).
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
SIZES = (4, 8)


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


def first_doors(run: Path, layout: dict) -> list[int]:
    """The start-room door (1-3, listed order) of each agent's first accepted move."""
    doors = layout["doors"][layout["start"]]
    out = []
    for log in sorted(run.glob("results/*/agent-*/trials/*/verifier/moves.jsonl")):
        for line in log.read_text(errors="replace").splitlines():
            try:
                m = json.loads(line)
            except ValueError:
                continue
            if m.get("ok") and m.get("from") == layout["start"] and m.get("to") in doors:
                out.append(doors.index(m["to"]) + 1)
                break
    return out


def load(name: str, root: Path, dataset: Path) -> list[dict]:
    teams = []
    for r in map(json.loads, (root / "rows.jsonl").read_text().splitlines()):
        if r["agents"] not in SIZES:
            continue
        layout = json.loads((dataset / "tasks" / f"maze-t{r['tree']:02d}" / "environment"
                             / "maze" / "tree.json").read_text())
        teams.append({"condition": name, "tree": r["tree"], "agents": r["agents"],
                      "exit_branch": r["exit_branch"], "success": bool(r["success"]),
                      "room_overlap": r.get("room_overlap"),
                      "posts_per_agent": r["posts"] / r["agents"],
                      "first_doors": first_doors(root / "runs" / Path(r["run"]).name, layout),
                      "run": Path(r["run"]).name})
    return teams


def summary(name, teams, timing, p1, trees=32):
    out = []
    for n in SIZES:
        g = [t for t in teams if t["agents"] == n]
        k = sum(t["success"] for t in g)
        low, high = wilson(k, len(g))
        by_door = {b: [t["success"] for t in g if t["exit_branch"] == b] for b in (1, 2, 3)}
        first = [d for t in g for d in t["first_doors"]]
        lines = [x for x in timing if x["condition"] == name and x["agents"] == n
                 and x["first_move"] is not None]
        reads = [x["read_before_move"] for x in lines if x["read_before_move"] is not None]
        out.append({
            "condition": name, "agents": n, "teams": len(g), "missing": trees - len(g),
            "successes": k, "success_rate": k / len(g) if g else math.nan,
            "ci_low": low, "ci_high": high, "best_of_n_from_n1": 1 - (1 - p1) ** n,
            **{f"exit_door{b}": f"{sum(v)}/{len(v)}" for b, v in by_door.items()},
            "room_overlap": mean_ci([t["room_overlap"] for t in g
                                     if t["room_overlap"] is not None])[0],
            "posts_per_agent": mean_ci([t["posts_per_agent"] for t in g])[0],
            **{f"first_move_door{b}": first.count(b) / len(first) if first else math.nan
               for b in (1, 2, 3)},
            "posted_before_first_move": (sum(x["posted_before_move"] for x in lines) / len(lines)
                                         if lines else math.nan),
            "read_before_first_move": sum(reads) / len(reads) if reads else math.nan,
            "agents_with_sessions": len(reads)})
    return out


def pct(x):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.0f}%"


def table(rows) -> str:
    lines = ["| condition | N | success (95% CI) | best-of-N | success by exit door 1 / 2 / 3 "
             "| room overlap | posts per agent | first move through door 1 / 2 / 3 "
             "| posted / read before first move |",
             "|---|---|---|---|---|---|---|---|---|"]
    for s in rows:
        gone = f", {s['missing']} missing" if s["missing"] else ""
        lines.append(
            f"| {s['condition']} | {s['agents']} | {s['successes']}/{s['teams']} "
            f"({pct(s['success_rate'])}, {pct(s['ci_low'])}-{pct(s['ci_high'])}{gone}) "
            f"| {pct(s['best_of_n_from_n1'])} | {s['exit_door1']}, {s['exit_door2']}, "
            f"{s['exit_door3']} | {s['room_overlap']:.2f} | {s['posts_per_agent']:.1f} "
            f"| {pct(s['first_move_door1'])} / {pct(s['first_move_door2'])} / "
            f"{pct(s['first_move_door3'])} | {pct(s['posted_before_first_move'])} / "
            f"{pct(s['read_before_first_move'])} |")
    return "\n".join(lines) + "\n"


def figure(path: Path, conditions, timing, p1) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, grid = plt.subplots(2, 3, figsize=(15, 8.4), facecolor=SURFACE)
    ax = grid.flatten()
    width = 0.26
    for k, (name, teams) in enumerate(conditions):
        off = (k - (len(conditions) - 1) / 2) * width
        color = SERIES[k]
        # (a) success against N
        for j, n in enumerate(SIZES):
            g = [t for t in teams if t["agents"] == n]
            kk = sum(t["success"] for t in g)
            p, (lo, hi) = (kk / len(g) if g else math.nan), wilson(kk, len(g))
            ax[0].errorbar(j + off, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]], color=color,
                           marker="o", capsize=3, linestyle="none", label=name if j == 0 else None)
        # (b) success by the exit's door, N = 4 and 8
        for b in (1, 2, 3):
            v = [t["success"] for t in teams if t["exit_branch"] == b]
            p, (lo, hi) = (sum(v) / len(v) if v else math.nan), wilson(sum(v), len(v))
            ax[1].errorbar(b - 1 + off, 100 * p, yerr=[[100 * (p - lo)], [100 * (hi - p)]],
                           color=color, marker="o", capsize=3, linestyle="none")
        # (c) room overlap and (d) posts per agent, by N
        for i, key in ((2, "room_overlap"), (3, "posts_per_agent")):
            for j, n in enumerate(SIZES):
                m, lo, hi = mean_ci([t[key] for t in teams if t["agents"] == n
                                     and t[key] is not None])
                ax[i].errorbar(j + off, m, yerr=[[m - lo], [hi - m]], color=color, marker="o",
                               capsize=3, linestyle="none")
        # (e) the first move's door
        first = [d for t in teams for d in t["first_doors"]]
        for b in (1, 2, 3):
            ax[4].bar(b - 1 + off, 100 * first.count(b) / len(first) if first else 0, width * 0.9,
                      color=color)
        # (f) board use before the first move
        lines = [x for x in timing if x["condition"] == name and x["first_move"] is not None]
        reads = [x["read_before_move"] for x in lines if x["read_before_move"] is not None]
        for j, value in enumerate((
                sum(x["posted_before_move"] for x in lines) / len(lines) if lines else math.nan,
                sum(reads) / len(reads) if reads else math.nan)):
            if not math.isnan(value):
                ax[5].bar(j + off, 100 * value, width * 0.9, color=color)
            else:
                ax[5].text(j + off, 1, "n/a", ha="center", va="bottom", fontsize=7, color=INK_2,
                           rotation=90)
    for j, n in enumerate(SIZES):
        ax[0].hlines(100 * (1 - (1 - p1) ** n), j - 0.45, j + 0.45, colors=INK_2, linestyles="--",
                     label="best-of-N from N = 1 (thinking off)" if j == 0 else None)
    labels = [("Success (%)", ["N = 4", "N = 8"], (0, 105)),
              ("Success (%), N = 4 and 8", ["exit behind door 1", "door 2", "door 3"], (0, 105)),
              ("Room overlap", ["N = 4", "N = 8"], (0, 1)),
              ("Board posts per agent", ["N = 4", "N = 8"], None),
              ("Agents' first move (%), N = 4 and 8", ["through door 1", "door 2", "door 3"], (0, 105)),
              ("Agents (%) before their first move", ["posted", "read the board"], (0, 105))]
    for a, (label, ticks, ylim) in zip(ax, labels, strict=True):
        a.set_ylabel(label)
        a.set_xticks(range(len(ticks)), ticks)
        if ylim:
            a.set_ylim(*ylim)
        a.set_facecolor(SURFACE)
        a.grid(axis="y", color=GRID, linewidth=0.8)
        a.set_axisbelow(True)
        a.spines[["top", "right"]].set_visible(False)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("s3-maze positive control: Qwen3.5-27B, thinking off, at N = 4 and 8, all 32 "
                 "trees", fontsize=11, color=INK, x=0.01, ha="left")
    fig.text(0.01, 0.012, "Board, no hint: the thinking-off scaling sweep's runs. Board + hint: the "
             "same plus one protocol instruction in the team text. Board off: no board resource "
             "or tools.\nOne counted team per tree and N; 95% Wilson intervals (success), means "
             "with 95% intervals (overlap, posts). Doors are the start room's, in listed order; "
             "dashed: best-of-N from the\nthinking-off 27B's own N = 1 rate. Reads and posts "
             "before the first move come from the agents' Pi sessions.",
             fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.065, 1, 0.95))
    fig.savefig(path, dpi=200, facecolor=SURFACE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", action="append", required=True, help="NAME=GRID_ROOT")
    parser.add_argument("--timing", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--p1", type=float, default=0.25, help="the 27B's N = 1 success rate")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    timing = [json.loads(x) for x in args.timing.read_text().splitlines() if x]
    conditions = [(c.split("=", 1)[0], load(c.split("=", 1)[0], Path(c.split("=", 1)[1]),
                                            args.dataset)) for c in args.condition]
    rows = [s for name, teams in conditions for s in summary(name, teams, timing, args.p1)]
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "control.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    (args.out / "control.md").write_text(table(rows))
    figure(args.out / "control.png", conditions, timing, args.p1)
    print(table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
