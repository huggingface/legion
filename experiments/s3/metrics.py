"""One row per team from s³ runs (huggingface/flock#115): `uv run experiments/s3/metrics.py
RUN_DIR [RUN_DIR ...] [--out rows.jsonl]`.

A run directory is one team: the bundle `flock launch` writes for a local run (docs/local.md),
with each agent's Trials (`trials/<swarm>-<agent>-tN/`, where Harbor puts the verifier's reward)
and the board export (`resources/message_board/{board.jsonl,channels.json}`) found anywhere
under it. Each row (one JSON line; experiments/s3-aggregation/README.md has every field):
- the task and team size, how the run ended, and `success`: any agent's reward is 1;
- per agent: its reward (its latest Trial with a verifier result), tokens (summed over models)
  and board posts;
- the message graph: an edge from a sender to each teammate it mentions (`@agent-002`; a group
  mention such as `@all` names every member) or messages directly (a DM or a private channel of
  the two), with in-degree centralization and reciprocity as in the s³ paper's Appendix M. A
  post that mentions no one adds no edge;
- `missing`: what the run directory lacks (an agent's reward, the board export), so a row never
  passes for complete when it is not.

The board may hold earlier runs' messages (a reused Space): only those posted between the run's
start and end count. `EXTRAS` takes per-environment metrics (s3-maze: moves per agent and room
overlap, from the move logs its verifier copies next to the reward).
"""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def trials(run_dir: Path, agent: str) -> list[Path]:
    """The agent's Trial directories, oldest first (`<swarm>-<agent>-t<N>`)."""
    pattern = re.compile(rf"(?:.+-)?{re.escape(agent)}-t(\d+)")
    found = [(int(m[1]), p) for p in run_dir.rglob(f"*{agent}-t*")
             if p.is_dir() and p.parent.name == "trials" and (m := pattern.fullmatch(p.name))]
    return [p for _, p in sorted(found)]


def reward(run_dir: Path, agent: str) -> tuple[float | None, Path | None]:
    """The reward of the agent's latest Trial that has one, and that Trial's directory."""
    for trial in reversed(trials(run_dir, agent)):
        result = trial / "result.json"
        rewards = ((load_json(result).get("verifier_result") or {}).get("rewards")
                   if result.is_file() else None)
        if rewards and "reward" in rewards:
            return float(rewards["reward"]), trial
        text = trial / "verifier" / "reward.txt"
        if text.is_file() and text.read_text().strip():
            return float(text.read_text().strip()), trial
    return None, None


def tokens(agent: dict) -> dict[str, int]:
    usage = dict.fromkeys(TOKEN_FIELDS, 0)
    for by_model in (agent.get("tokens") or {}).values():
        for key in TOKEN_FIELDS:
            usage[key] += int(by_model.get(key) or 0)
    return usage


def epoch(value) -> float:
    return float(value) if isinstance(value, int | float) else datetime.fromisoformat(
        value).timestamp()


def board(run_dir: Path, start: float, end: float) -> tuple[list[dict], dict[int, dict], bool]:
    """(the run's messages, oldest first, channels by id, whether an export was found)."""
    messages, channels, found = {}, {}, False
    for export in run_dir.rglob("resources/message_board/board.jsonl"):
        found = True
        listing = export.with_name("channels.json")
        for channel in load_json(listing) if listing.is_file() else []:
            channels[channel["id"]] = channel
        for line in export.read_text(encoding="utf-8").splitlines():
            if line.strip():
                record = json.loads(line)
                message = record["message"] | {"channel_id": record["channel_id"]}
                if start <= epoch(message["created_at"]) <= end:
                    messages[message["id"]] = message
    return sorted(messages.values(), key=lambda m: m["id"]), channels, found


def sender(message: dict) -> str | None:
    return (message.get("sender") or {}).get("external_id")


def edges(messages: list[dict], channels: dict[int, dict], team: list[str]) -> set[tuple]:
    """(sender, receiver) for each teammate a message mentions or is sent to directly."""
    members = set(team)
    out = set()
    for message in messages:
        source = sender(message)
        if source not in members:
            continue
        targets = {m.get("external_id") for m in message.get("mentions") or []}
        channel = channels.get(message["channel_id"]) or {}
        if channel.get("kind") in ("dm", "private"):
            inside = {m.get("external_id") for m in channel.get("members") or []} & members
            if len(inside) == 2 and source in inside:  # a DM, or a private channel of the two
                targets |= inside
        out |= {(source, target) for target in targets & members if target != source}
    return out


def centralization(graph: set[tuple], team: list[str]) -> float | None:
    """Appendix M: sum over agents of (k* - k_j) / (N - 1)^2, k_j agent j's in-degree and k*
    the largest: 0 when everyone receives from as many teammates, 1 when one agent receives
    from everyone and no one else from anyone. None for a team of one."""
    n = len(team)
    if n < 2:
        return None
    degree = Counter(target for _, target in graph)
    k = [degree[a] for a in team]
    return sum(max(k) - kj for kj in k) / (n - 1) ** 2


def reciprocity(graph: set[tuple]) -> float | None:
    """Appendix M: the fraction of edges whose reverse edge exists. None without edges."""
    return sum((b, a) in graph for a, b in graph) / len(graph) if graph else None


# Per-environment metrics: f(run_dir, run, team, trial dirs by agent) -> extra row fields.
# s3-maze adds moves per agent and room overlap, (V - U) / V, from its move logs here.
EXTRAS: list[Callable[[Path, dict, list[str], dict[str, Path | None]], dict]] = []


def row(run_dir: Path) -> dict:
    run = load_json(run_dir / "run.json")
    agents = sorted(run.get("agents") or [], key=lambda a: a["name"])
    team = [a["name"] for a in agents]
    harbor = next((t.get("harbor") or {} for t in run["spec"]["types"].values()), {})
    task = (harbor.get("config") or {}).get("task") or {}
    start, end = epoch(run["started_at"]), epoch(run["ended_at"])
    messages, channels, exported = board(run_dir, start, end)
    posts = Counter(sender(m) for m in messages)
    graph = edges(messages, channels, team)
    per_agent, trial_dirs, missing = {}, {}, []
    for agent in agents:
        name = agent["name"]
        value, trial_dirs[name] = reward(run_dir, name)
        if value is None:
            missing.append(f"reward: {name}")
        usage = tokens(agent)
        per_agent[name] = {"reward": value, "tokens": usage, "posts": posts[name],
                           "turns": agent.get("turns_completed")}
    if not exported:
        missing.append("board export")
    rewards = [a["reward"] for a in per_agent.values() if a["reward"] is not None]
    out = {
        "run": run_dir.name,
        "swarm": run.get("swarm"),
        "task": task.get("selector"),
        "dataset": task.get("repo"),
        "revision": task.get("revision"),
        "agents": len(team),
        "stop_reason": run.get("stop_reason"),
        "duration_s": run.get("duration_s"),
        "success": any(r == 1 for r in rewards) if rewards else None,
        "rewards": {name: a["reward"] for name, a in per_agent.items()},
        "tokens": {key: sum(a["tokens"][key] for a in per_agent.values())
                   for key in TOKEN_FIELDS},
        "cost_usd": (run.get("totals") or {}).get("cost_usd"),
        "posts": sum(posts[name] for name in team),
        "other_posts": sum(n for who, n in posts.items() if who not in team),
        "per_agent": per_agent,
        "edges": sorted([a, b] for a, b in graph),
        "centralization": centralization(graph, team),
        "reciprocity": reciprocity(graph),
        "missing": missing,
    }
    for extra in EXTRAS:
        out |= extra(run_dir, run, team, trial_dirs)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="One row per team (JSON lines) from s³ runs.")
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, help="append the rows here (default: stdout)")
    args = parser.parse_args(argv)
    rows = [json.dumps(row(d)) for d in args.run_dirs]
    if args.out:
        with args.out.open("a", encoding="utf-8") as f:
            f.write("".join(r + "\n" for r in rows))
    else:
        print("\n".join(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
