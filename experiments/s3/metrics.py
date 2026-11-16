"""One row per team from s³ runs (huggingface/flock#115): `uv run experiments/s3/metrics.py
RUN_DIR [RUN_DIR ...] [--out rows.jsonl]`.

A run directory is one team: a local run's bundle (`.flock/<swarm>-<time>/`, docs/local.md), or
a Slurm run's directory (`flock init`'s RUN_DIR, docs/slurm.md), where the workers copy what a
run is graded on into `results/`. Under either, each agent's Trials
(`<agent>/trials/<swarm>-<agent>-tN/`: Harbor's `result.json` and `verifier/`) and the board
export (`resources/message_board/{board.jsonl,channels.json}`) are found anywhere. Each row (one
JSON line; experiments/s3-aggregation/README.md has every field):
- the task and team size, how the run ended, and `success`: any agent's reward is 1;
- per agent: its reward (its latest Trial with a verifier result), tokens and board posts;
- the message graph: an edge from a sender to each teammate it mentions (`@agent-002`; a group
  mention such as `@all` names every member) or messages directly (a DM or a private channel of
  the two), with in-degree centralization and reciprocity as in the s³ paper's Appendix M. A
  post that mentions no one adds no edge;
- `missing`: what the run directory lacks (an agent's reward, the board export), so a row never
  passes for complete when it is not.

A local run's `run.json` gives the spec, the agents, their tokens (summed over models), the
cost and how the run ended. A Slurm run has no such bundle: the team and the task come from
its `experiment/spec.yaml`, each agent's tokens from its Trials' results (no reasoning split),
and the run's span from its Trials. The board may hold earlier runs' messages (a reused Space):
only those posted within the run's span count. `EXTRAS` takes per-environment metrics (s3-maze:
moves per agent and room overlap, from the move logs its verifier copies next to the reward).
"""

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import yaml

TOKEN_FIELDS = ("input", "output", "cache_read", "cache_write", "reasoning")


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def epoch(value) -> float:
    return float(value) if isinstance(value, int | float) else datetime.fromisoformat(
        value).timestamp()


def bundle(run_dir: Path) -> dict | None:
    """A local run's run.json; None for a Slurm run (its controller's run.json, in the bucket,
    holds neither the spec nor the agents)."""
    path = run_dir / "run.json"
    doc = load_json(path) if path.is_file() else {}
    return doc if "spec" in doc and "agents" in doc else None


def all_trials(run_dir: Path) -> dict[str, list[Path]]:
    """Each agent's Trial directories, oldest first: `<agent>/trials/<swarm>-<agent>-t<N>/`
    (a local run's agents/_harbor/, a Slurm run's results/<swarm>/), one per Trial name."""
    found: dict[str, dict[str, tuple[int, Path]]] = {}
    for trial in run_dir.rglob("trials/*"):
        agent = trial.parent.parent.name
        m = re.fullmatch(rf"(?:.+-)?{re.escape(agent)}-t(\d+)", trial.name)
        if m and trial.is_dir():
            found.setdefault(agent, {}).setdefault(trial.name, (int(m[1]), trial))
    return {a: [p for _, p in sorted(t.values())] for a, t in sorted(found.items())}


def results(trials: list[Path]) -> list[dict]:
    return [load_json(t / "result.json") for t in trials if (t / "result.json").is_file()]


def reward(trials: list[Path]) -> float | None:
    """The reward of the agent's latest Trial that has one."""
    for trial in reversed(trials):
        result = trial / "result.json"
        rewards = ((load_json(result).get("verifier_result") or {}).get("rewards")
                   if result.is_file() else None)
        if rewards and "reward" in rewards:
            return float(rewards["reward"])
        text = trial / "verifier" / "reward.txt"
        if text.is_file() and not text.is_symlink() and text.read_text().strip():
            return float(text.read_text().strip())
    return None


def bundle_tokens(agent: dict) -> dict[str, int]:
    usage = dict.fromkeys(TOKEN_FIELDS, 0)
    for by_model in (agent.get("tokens") or {}).values():
        for key in TOKEN_FIELDS:
            usage[key] += int(by_model.get(key) or 0)
    return usage


def trial_tokens(trials: list[Path]) -> dict[str, int]:
    """Harbor's counts: its input includes the cached tokens."""
    usage = dict.fromkeys(TOKEN_FIELDS, 0)
    for result in results(trials):
        agent = result.get("agent_result") or {}
        cached = int(agent.get("n_cache_tokens") or 0)
        usage["input"] += int(agent.get("n_input_tokens") or 0) - cached
        usage["cache_read"] += cached
        usage["output"] += int(agent.get("n_output_tokens") or 0)
    return usage


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


# Per-environment metrics: f(run_dir, team, Trial dirs by agent) -> extra row fields.
# s3-maze adds moves per agent and room overlap, (V - U) / V, from its move logs here.
EXTRAS: list[Callable[[Path, list[str], dict[str, list[Path]]], dict]] = []


def local_run(run: dict) -> dict:
    """What a local bundle says about the run."""
    harbor = next((t.get("harbor") or {} for t in run["spec"]["types"].values()), {})
    task = (harbor.get("config") or {}).get("task") or {}
    agents = sorted(run["agents"], key=lambda a: a["name"])
    return {"swarm": run.get("swarm"), "task": task, "team": [a["name"] for a in agents],
            "start": epoch(run["started_at"]), "end": epoch(run["ended_at"]),
            "stop_reason": run.get("stop_reason"), "duration_s": run.get("duration_s"),
            "cost_usd": (run.get("totals") or {}).get("cost_usd"),
            "tokens": {a["name"]: bundle_tokens(a) for a in agents},
            "turns": {a["name"]: a.get("turns_completed") for a in agents}}


def slurm_run(run_dir: Path, trials: dict[str, list[Path]]) -> dict:
    """What a Slurm run's directory says: its spec (`flock init` copies the experiment) and the
    Trials the workers kept."""
    path = run_dir / "experiment" / "spec.yaml"
    spec = yaml.safe_load(path.read_text()) if path.is_file() else {}
    agents = spec.get("agents") or {}
    if spec.get("naming") == "consecutive-id" and isinstance(agents.get("count"), int):
        team = [f"agent-{k:03d}" for k in range(1, agents["count"] + 1)]
    else:
        team = list(trials)
    times = [(epoch(r["started_at"]), epoch(r["finished_at"]))
             for ts in trials.values() for r in results(ts)
             if r.get("started_at") and r.get("finished_at")]
    start, end = (min(t[0] for t in times), max(t[1] for t in times)) if times else (0, 0)
    names = {r.get("task_name") for ts in trials.values() for r in results(ts)}
    task = spec.get("task") or {"selector": next((n.split("/")[-1] for n in names if n), None)}
    swarm = next((t.name.removesuffix(f"-{a}-t{t.name.rsplit('-t', 1)[1]}")
                  for a, ts in trials.items() for t in ts), spec.get("name"))
    return {"swarm": swarm, "task": task, "team": team, "start": start, "end": end,
            "stop_reason": None, "duration_s": round(end - start, 1) if times else None,
            "cost_usd": None, "tokens": {a: trial_tokens(trials.get(a, [])) for a in team},
            "turns": dict.fromkeys(team)}


def row(run_dir: Path) -> dict:
    trials = all_trials(run_dir)
    run = bundle(run_dir)
    info = local_run(run) if run is not None else slurm_run(run_dir, trials)
    team = info["team"]
    messages, channels, exported = board(run_dir, info["start"], info["end"])
    posts = Counter(sender(m) for m in messages)
    graph = edges(messages, channels, team)
    per_agent, missing = {}, []
    for name in team:
        value = reward(trials.get(name, []))
        if value is None:
            missing.append(f"reward: {name}")
        per_agent[name] = {"reward": value, "tokens": info["tokens"][name],
                           "posts": posts[name], "turns": info["turns"][name]}
    if not exported:
        missing.append("board export")
    rewards = [a["reward"] for a in per_agent.values() if a["reward"] is not None]
    task = info["task"]
    out = {
        "run": run_dir.name,
        "swarm": info["swarm"],
        "task": task.get("selector"),
        "dataset": task.get("repo"),
        "revision": task.get("revision"),
        "agents": len(team),
        "stop_reason": info["stop_reason"],
        "duration_s": info["duration_s"],
        "success": any(r == 1 for r in rewards) if rewards else None,
        "rewards": {name: a["reward"] for name, a in per_agent.items()},
        "tokens": {key: sum(a["tokens"][key] for a in per_agent.values())
                   for key in TOKEN_FIELDS},
        "cost_usd": info["cost_usd"],
        "posts": sum(posts[name] for name in team),
        "other_posts": sum(n for who, n in posts.items() if who not in team),
        "per_agent": per_agent,
        "edges": sorted([a, b] for a, b in graph),
        "centralization": centralization(graph, team),
        "reciprocity": reciprocity(graph),
        "missing": missing,
    }
    for extra in EXTRAS:
        out |= extra(run_dir, team, trials)
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
