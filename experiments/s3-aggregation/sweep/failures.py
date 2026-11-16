"""How the teams of a grid fail: `uv run experiments/s3-aggregation/sweep/failures.py <root>
[--sessions <dir>]`, writing `<root>/failures.md` and `<root>/agents.csv`.

For each agent of each team in `<root>/rows.jsonl`: the answer the verifier saw (none, malformed,
the wrong count, the wrong numbers, unsorted, or right), against the problem's numbers
(the dataset's problems.json at the template's pin), and, from its event stream (`--sessions`:
a copy of the state bucket, `<dir>/<run>/agent/<agent>/sessions/<agent>.jsonl`, e.g. from
`hf buckets sync hf://buckets/<bucket> <dir> --include "*/sessions/agent-*.jsonl"`), what it did: board posts
through `flock-tool`, tool calls that failed, and whether it ever reached `flock-tool` at all
(a small model may call util-linux's `flock` instead). A team fails when no agent's answer is
right; the summary counts the failed teams' agents by answer and behaviour.
"""

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import yaml
from huggingface_hub import hf_hub_download

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE.parent / "spec.template.yaml"


def problems() -> dict[str, list[int]]:
    revision = yaml.safe_load(TEMPLATE.read_text())["task"]["revision"]
    path = hf_hub_download("HuggingFaceSwarms/s3-aggregation", "problems.json",
                           repo_type="dataset", revision=revision)
    return json.loads(Path(path).read_text())["problems"]


def answer_kind(text: str | None, team: list[int]) -> str:
    if text is None:
        return "no answer"
    try:
        answer = json.loads(text)
    except ValueError:
        return "malformed"
    if not isinstance(answer, list) or not all(type(x) is int for x in answer):
        return "malformed"
    if answer == sorted(team):
        return "right"
    if len(answer) != len(team):
        return "wrong count"
    if sorted(answer) == sorted(team):
        return "unsorted"
    return "wrong numbers"


def latest_answer(run_dir: Path, agent: str) -> str | None:
    trials = sorted(run_dir.rglob(f"{agent}/trials/*-{agent}-t*"),
                    key=lambda p: int(re.search(r"-t(\d+)$", p.name)[1]))
    for trial in reversed(trials):
        path = trial / "verifier" / "answer.json"
        if path.is_file():
            return path.read_text(errors="replace")
        if (trial / "verifier" / "reward.txt").is_file():
            return None  # graded, and no answer file
    return None


def behaviour(path: Path | None) -> dict:
    """What an agent did, from its event stream: tool calls, failed ones, board posts."""
    out = {"tool_calls": 0, "tool_errors": 0, "posts": 0, "used_flock_tool": False,
           "called_util_flock": False, "session": path is not None and path.is_file()}
    if not out["session"]:
        return out
    calls = {}
    for line in path.read_text(errors="replace").splitlines():
        event = json.loads(line)
        message = event.get("message") or {}
        if event.get("type") != "message":
            continue
        if message.get("role") == "assistant":
            for part in message.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "toolCall":
                    out["tool_calls"] += 1
                    command = json.dumps(part.get("arguments") or {})
                    calls[part.get("id")] = command
                    out["used_flock_tool"] |= "flock-tool" in command
                    out["called_util_flock"] |= bool(re.search(r"(^|[^-\w])flock (?!-tool)",
                                                               command.replace("\\\\n", " ")))
        elif message.get("role") == "toolResult":
            text = " ".join(p.get("text", "") for p in message.get("content") or []
                            if isinstance(p, dict))
            failed = message.get("isError") or '"error"' in text
            out["tool_errors"] += bool(failed)
            command = calls.get(message.get("toolCallId"), "")
            if "message_board.post" in command and not failed:
                out["posts"] += 1
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--sessions", type=Path, help="a copy of the state bucket")
    args = parser.parse_args(argv)
    numbers = problems()
    rows = [json.loads(x) for x in (args.root / "rows.jsonl").read_text().splitlines()]
    agents = []
    for r in rows:
        team = numbers[f"p{r['problem']:02d}"][:r["agents"]]
        run_dir = args.root / "runs" / r["run"]
        for name in sorted(r["rewards"]):
            session = (args.sessions / r["run"] / "agent" / name / "sessions" / f"{name}.jsonl"
                       if args.sessions else None)
            agents.append({"run": r["run"], "task": r["task"], "agents": r["agents"],
                           "team_success": bool(r["success"]), "agent": name,
                           "reward": r["rewards"][name],
                           "answer": answer_kind(latest_answer(run_dir, name), team),
                           **behaviour(session)})
    with (args.root / "agents.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agents[0]))
        writer.writeheader()
        writer.writerows(agents)
    failed = [a for a in agents if not a["team_success"]]
    by_n = Counter(r["agents"] for r in rows if not r["success"])
    per_n = ", ".join(f"N={n}: {c}" for n, c in sorted(by_n.items()))
    lines = [f"Teams: {len(rows)}, failed: {sum(by_n.values())} ({per_n})", "",
             "Agents of failed teams, by the answer the verifier saw:", ""]
    lines += [f"- {k}: {c}" for k, c in Counter(a["answer"] for a in failed).most_common()]
    seen = [a for a in failed if a["session"]]
    if seen:
        errors, calls = sum(a["tool_errors"] for a in seen), sum(a["tool_calls"] for a in seen)
        erring = sum(a["tool_errors"] > 0 for a in seen)
        lines += ["", f"Their behaviour ({len(seen)} with an event stream):", "",
                  f"- never posted on the board: {sum(a['posts'] == 0 for a in seen)}",
                  f"- never reached flock-tool: {sum(not a['used_flock_tool'] for a in seen)}",
                  f"- called util-linux `flock` instead: {sum(a['called_util_flock'] for a in seen)}",
                  f"- had failed tool calls: {erring} ({errors} of {calls} calls)"]
    (args.root / "failures.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
