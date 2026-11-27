"""Score verifier results, optionally adjudicating rejected structured answers."""
import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from math_eval_common import (
    compare_binary,
    find_last_boxed_content,
    judge_text,
    load_runs,
    majority,
    offline_reward,
    output_tokens,
    parse_answer_verdict,
    read_text,
    remove_self_evaluation,
    summary,
)

REPO = "HuggingFaceSwarms/IMO-AnswerBench"
REVISION = "40d9d407c8b5e4b62c6b2047e21c807214f2a068"


def dataset_file(task, directory, filename):
    if directory:
        root = Path(directory)
        suffix = ".json" if filename == "tests/reference.json" else ".md"
        paths = [root / "tasks" / task / filename, root / task / filename,
                 root / f"{task}{suffix}"]
        path = next((p for p in paths if p.is_file()), None)
        if path is None:
            raise ValueError(f"{filename} not found for {task} in {root}")
    else:
        path = hf_hub_download(REPO, f"tasks/{task}/{filename}",
                               repo_type="dataset", revision=REVISION)
    return Path(path).read_text(encoding="utf-8")


def reference(task, directory):
    return json.loads(dataset_file(task, directory, "tests/reference.json"))


def normalize(answer):
    # Preserve all internal text: QED extraction and the judge see line breaks,
    # spacing and Unicode exactly as cleaned. This is independent of reward.
    return answer.strip()


def equivalent(answer, gold, problem, args):
    # QED-Nano extracts a final box, falling back to the last five lines.
    # The caller cleans once; repeating cleanup could change an indented heading.
    student = find_last_boxed_content(answer)
    if not student:
        lines = answer.split("\n")
        student = "..." + "\n".join(lines[-5:]) if len(lines) >= 5 else answer
    template = (Path(__file__).resolve().parents[1] / "qed-nano-prompts/answerbench.txt").read_text()
    prompt = template.format(problem_statement=problem, student_answer=student, gold_answer=gold)
    return parse_answer_verdict(judge_text(prompt, args.judge_model, args.judge_base_url,
                                          args.judge_api_key_env,
                                          gemini=getattr(args, "provider", "openai") == "gemini"))


def score(roots, args):
    teams = []
    for team in load_runs(roots):
        rows = []
        ref = None
        problem = None
        for agent in team["agents"]:
            submission = remove_self_evaluation(read_text(agent["verifier"] / "answer.txt"))
            answer = normalize(submission)
            correct = offline_reward(agent)
            stage2 = False
            if args.judge and not correct and answer:
                grade_path = agent["verifier"] / "grade.json"
                if grade_path.is_file() and json.loads(grade_path.read_text()).get("why", "").startswith("grader error"):
                    raise ValueError(f"Verifier failed for {agent['trial']}")
                if ref is None:
                    ref = reference(team["task_id"], args.reference_dir)
                if ref["kind"] in {"expression", "set", "tuple", "interval", "literal"}:
                    if problem is None:
                        problem = dataset_file(team["task_id"], args.reference_dir, "instruction.md")
                    stage2 = equivalent(submission, ref["answer"], problem, args)
                    correct = int(stage2)
            rows.append({"agent": agent["agent"], "answer": answer, "correct": correct,
                         "stage2_accepted": stage2, "output_tokens": output_tokens(agent)})
        winner = majority([row["answer"] for row in rows])
        votes = [row["correct"] for row in rows if row["answer"] == winner] if winner else []
        if len(set(votes)) > 1:
            raise ValueError("Identical answers received inconsistent grades")
        teams.append({"run": team["run"], "swarm": team["swarm"], "task": team["task"], "task_id": team["task_id"],
                      "agents": rows, "n": len(rows), "majority_answer": winner,
                      "maj_at_n": votes[0] if votes else 0,
                      "best_of_n": max(row["correct"] for row in rows)})
    return {"comparison": compare_binary(teams, getattr(args, "k", 4)), "teams": teams, "solo": summary(row["correct"] for t in teams for row in t["agents"]),
            "maj_at_n": summary(t["maj_at_n"] for t in teams),
            "best_of_n": summary(t["best_of_n"] for t in teams),
            "by_n": {str(n): {"maj_at_n": summary(t["maj_at_n"] for t in teams if t["n"] == n),
                              "best_of_n": summary(t["best_of_n"] for t in teams if t["n"] == n)}
                     for n in sorted({t["n"] for t in teams})}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Legion results directories; one per run")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--judge-model")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env", help="Defaults to HF_INFERENCE_TOKEN or GEMINI_API_KEY")
    parser.add_argument("--reference-dir")
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be >= 2")
    if not args.judge_api_key_env:
        args.judge_api_key_env = "GEMINI_API_KEY" if args.provider == "gemini" else "HF_INFERENCE_TOKEN"
    if args.judge and not args.judge_model:
        parser.error("--judge requires --judge-model")
    print(json.dumps(score(args.runs, args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
