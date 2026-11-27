"""Score exact-integer SmallMath runs and stratify by difficulty."""
import argparse
import json
import re
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from math_eval_common import load_runs, majority, offline_reward, read_text, summary

REVISION = "57dbf3f7e83aa8c5c980387a2216d7bb87e4e7cc"


def canonical(answer):
    answer = answer.strip()
    return str(int(answer)) if re.fullmatch(r"[+-]?\d+", answer) else answer


def score(roots, manifest):
    if manifest is None:
        manifest = hf_hub_download("HuggingFaceSwarms/SmallMath", "tasks/manifest.json",
                                   repo_type="dataset", revision=REVISION)
    metadata = json.loads(Path(manifest).read_text())
    levels = {item["task"]: int(item["level"]) for item in metadata["tasks"]}
    teams = []
    for team in load_runs(roots):
        if team["task"] not in levels or levels[team["task"]] not in range(1, 7):
            raise ValueError(f"Missing valid difficulty for {team['task']}")
        rows = [{"agent": a["agent"], "answer": canonical(read_text(a["verifier"] / "answer.txt")),
                 "correct": offline_reward(a)} for a in team["agents"]]
        winner = majority([row["answer"] for row in rows])
        votes = [row["correct"] for row in rows if row["answer"] == winner] if winner else []
        if len(set(votes)) > 1:
            raise ValueError("Identical answers received inconsistent grades")
        teams.append({"run": team["run"], "swarm": team["swarm"], "task": team["task"],
                      "level": levels[team["task"]], "n": len(rows), "agents": rows,
                      "majority_answer": winner, "maj_at_n": votes[0] if votes else 0})
    def metrics(group):
        return {"solo": summary(a["correct"] for t in group for a in t["agents"]),
                "maj_at_n": summary(t["maj_at_n"] for t in group),
                "by_n": {str(n): summary(t["maj_at_n"] for t in group if t["n"] == n)
                         for n in sorted({t["n"] for t in group})}}
    return {"overall": metrics(teams), "levels": {str(n): metrics([t for t in teams
            if t["level"] == n]) for n in range(1, 7)}, "teams": teams}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--manifest", help="Pinned SmallMath tasks/manifest.json")
    args = parser.parse_args()
    print(json.dumps(score(args.runs, args.manifest), indent=2))


if __name__ == "__main__":
    main()
