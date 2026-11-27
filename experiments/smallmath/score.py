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
# The pinned verifier's MAX_LEN and MAX_INT_DIGITS are both 1000. This is a
# conservative subset of its grammar: an optional sign followed by ASCII digits.
MAX_INTEGER_LEN = 1000


def canonical(answer):
    answer = answer.strip()
    if len(answer) <= MAX_INTEGER_LEN and re.fullmatch(r"[+-]?[0-9]+", answer):
        negative = answer.startswith("-")
        digits = answer.lstrip("+-").lstrip("0") or "0"
        return "-" + digits if negative and digits != "0" else digits
    return answer


def score(roots, manifest):
    if manifest is None:
        manifest = hf_hub_download("HuggingFaceSwarms/SmallMath", "tasks/manifest.json",
                                   repo_type="dataset", revision=REVISION)
    metadata = json.loads(Path(manifest).read_text())
    levels = {item["task"]: int(item["level"]) for item in metadata["tasks"]}
    teams = []
    for team in load_runs(roots):
        if team["task_id"] not in levels or levels[team["task_id"]] not in range(1, 7):
            raise ValueError(f"Missing valid difficulty for {team['task']}")
        rows = []
        for agent in team["agents"]:
            answer = canonical(read_text(agent["verifier"] / "answer.txt"))
            correct = offline_reward(agent)
            rows.append({"agent": agent["agent"], "answer": answer, "correct": correct})
        winner = majority([row["answer"] for row in rows])
        votes = [row["correct"] for row in rows if row["answer"] == winner] if winner else []
        if len(set(votes)) > 1:
            raise ValueError("Identical answers received inconsistent grades")
        teams.append({"run": team["run"], "swarm": team["swarm"], "task": team["task"],
                      "task_id": team["task_id"], "level": levels[team["task_id"]], "n": len(rows), "agents": rows,
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
