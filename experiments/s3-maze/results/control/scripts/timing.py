"""When agents use the board relative to their first move, per counted team, from the run's own
logs: the first move (the maze service's move log), the first post (the board export) and, where
the run kept its agents' session logs (`<run dir>/sessions/`), the first board read and the
first post as tool calls. One JSON line per agent:

    python timing.py --label baseline=<grid root> --label hint=<grid root> ... --agents 4,8 --out timing.jsonl
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

# Pi runs every tool through its shell: `flock-tool message_board.<call> --json ...`, `maze move`
READ = re.compile(r"\bmessage_board\.(read|search|thread|history|activity|channels)\b")
POST = re.compile(r"\bmessage_board\.(post|reply|dm)\b")
MOVE = re.compile(r"\bmaze\s+move\b")


def epoch(text: str) -> float:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


def first_moves(run: Path) -> dict[str, float]:
    """Each agent's first move request (accepted or refused) by time."""
    out = {}
    for log in run.glob("results/*/agent-*/trials/*/verifier/moves.jsonl"):
        agent = log.parts[-5]
        for line in log.read_text(errors="replace").splitlines():
            try:
                t = epoch(json.loads(line)["time"])
            except (ValueError, KeyError):
                continue
            out[agent] = min(out.get(agent, t), t)
            break
    return out


def first_posts(run: Path) -> dict[str, float]:
    out = {}
    for export in run.glob("results/*/*/resources/message_board/board.jsonl"):
        for line in export.read_text(errors="replace").splitlines():
            m = json.loads(line)["message"]
            who = m["sender"]["external_id"]
            out[who] = min(out.get(who, m["created_at"]), m["created_at"])
    return out


def tool_calls(session: Path):
    """Each shell command in a Pi session log, in the order the model issued them (calls in one
    message share its time, so the order, not the time, says which came first)."""
    for line in session.read_text(errors="replace").splitlines():
        try:
            e = json.loads(line)
        except ValueError:  # e.g. an instructions line with an escape Python's json refuses
            continue
        msg = e.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        for part in msg.get("content") or []:
            if isinstance(part, dict) and part.get("type") == "toolCall":
                args = part.get("arguments") or {}
                yield f"{part.get('name')} {args.get('command') or json.dumps(args)}"


def from_sessions(run: Path) -> dict[str, dict]:
    """Per agent, the position of its first board read, board post and move among its tool
    calls (Pi's sessions, copied by the control driver to `<run dir>/sessions/`)."""
    out = {}
    for home in (run / "sessions").glob("*/agent-*/harness/sessions"):
        agent, first, i = home.parts[-3], {}, 0
        for session in sorted(home.glob("*.jsonl")):
            for command in tool_calls(session):
                for kind, pattern in (("read", READ), ("post", POST), ("move", MOVE)):
                    if pattern.search(command) and kind not in first:
                        first[kind] = i
                i += 1
        out[agent] = first
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", action="append", required=True, help="NAME=GRID_ROOT")
    parser.add_argument("--agents", default="4,8")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sizes = {int(x) for x in args.agents.split(",")}
    lines = []
    for label in args.label:
        name, root = label.split("=", 1)
        root = Path(root)
        for row in map(json.loads, (root / "rows.jsonl").read_text().splitlines()):
            if row["agents"] not in sizes:
                continue
            run = root / "runs" / Path(row["run"]).name
            moves, posts, calls = first_moves(run), first_posts(run), from_sessions(run)
            for k in range(1, row["agents"] + 1):
                agent = f"agent-{k:03d}"
                move, post, c = moves.get(agent), posts.get(agent), calls.get(agent, {})
                lines.append({
                    "condition": name, "agents": row["agents"], "tree": row["tree"],
                    "agent": agent, "first_move": move, "first_post": post,
                    "posted_before_move": None if move is None else (post is not None and post < move),
                    "sessions": bool(calls),
                    "read_before_move": (None if not calls or c.get("move") is None
                                         else c.get("read") is not None and c["read"] < c["move"]),
                    "post_call_before_move": (None if not calls or c.get("move") is None
                                              else c.get("post") is not None and c["post"] < c["move"]),
                })
    args.out.write_text("".join(json.dumps(x) + "\n" for x in lines))
    for name in dict.fromkeys(x["condition"] for x in lines):
        for n in sorted(sizes):
            xs = [x for x in lines if x["condition"] == name and x["agents"] == n]
            moved = [x for x in xs if x["first_move"] is not None]
            posted = sum(x["posted_before_move"] for x in moved)
            reads = [x for x in moved if x["read_before_move"] is not None]
            read = sum(x["read_before_move"] for x in reads)
            print(f"{name} N={n}: {len(xs)} agents, {len(moved)} moved; posted before the first "
                  f"move {posted}/{len(moved)}; read before it {read}/{len(reads)} (with sessions)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
