"""Judge ProofBench submissions on the 0–7 IMO scale and emit JSONL."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from math_eval_common import (
    DEFAULT_JUDGE_BASE_URL,
    DEFAULT_JUDGE_MODEL,
    compare_proof,
    judge_text,
    load_runs,
    majority,
    output_tokens,
    parse_proof_points,
    read_text,
    remove_self_evaluation,
    summary,
)


def grade(proof, ref, args):
    if ref.get("max_score") != 7:
        raise ValueError("Expected ProofBench reference with max_score=7")
    for field in ("problem", "solution", "grading_guidelines"):
        if not isinstance(ref.get(field), str):
            raise TypeError(f"Missing reference {field}")
    template = (Path(__file__).resolve().parents[1] / "qed-nano-prompts/proofbench.txt").read_text()
    prompt = template.format(problem_statement=ref["problem"], solution=ref["solution"],
                             guidelines=ref["grading_guidelines"],
                             student_answer=remove_self_evaluation(proof))
    response = judge_text(prompt, args.judge_model, args.judge_base_url, args.judge_api_key_env,
                          gemini=args.provider == "gemini")
    return parse_proof_points(response), response



def score(roots, args):
    records = []
    for team in load_runs(roots):
        rows = []
        for agent in team["agents"]:
            verifier = agent["verifier"]
            ref = json.loads((verifier / "reference.json").read_text())
            status = json.loads((verifier / "grade.json").read_text()).get("status")
            proof = read_text(verifier / "proof.md")
            if status == "submitted_for_judge" and proof:
                value, reason = grade(proof, ref, args)
            else:
                value, reason = 0, "No structurally valid proof submitted"
            rows.append({"type": "proof", "run": team["run"], "swarm": team["swarm"],
                         "task": team["task"], "task_id": team["task_id"], "agent": agent["agent"], "score": value,
                         "percent_of_7": 100 * value / 7, "reason": reason, "proof": proof,
                         "output_tokens": output_tokens(agent)})
        selected = [r for r in rows if len(rows) > 1 and r["agent"] == args.selected_agent]
        if len(rows) > 1 and args.selected_agent and not selected:
            raise ValueError(f"Selected agent {args.selected_agent} absent from {team['task']}")
        consensus = majority([r["proof"] for r in rows])
        # A consensus is an identical proof, not a vote over numerical grades.
        consensus_scores = [r["score"] for r in rows if consensus and r["proof"] == consensus]
        records.extend(rows)
        records.append({"type": "team", "run": team["run"], "swarm": team["swarm"],
                        "task": team["task"], "task_id": team["task_id"], "n": len(rows),
                        "mean": summary(r["score"] for r in rows)["mean"],
                        "max": max(r["score"] for r in rows),
                        "consensus": summary(consensus_scores)["mean"],
                        "selected": selected[0]["score"] if selected else None,
                        "selected_agent": args.selected_agent if len(rows) > 1 else None,
                        "scale_max": 7,
                        "output_tokens": (sum(r["output_tokens"] for r in rows)
                                          if all(r["output_tokens"] is not None for r in rows) else None),
                        "mean_percent_of_7": 100 * sum(r["score"] for r in rows) / (7 * len(rows)),
                        "max_percent_of_7": 100 * max(r["score"] for r in rows) / 7,
                        "consensus_percent_of_7": (100 * sum(consensus_scores) / (7 * len(consensus_scores))
                                                   if consensus_scores else None),
                        "selected_percent_of_7": 100 * selected[0]["score"] / 7 if selected else None})
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-api-key-env", help="Defaults to HF_INFERENCE_TOKEN or GEMINI_API_KEY")
    parser.add_argument("--selected-agent", help="Agent whose proof was selected before grading")
    parser.add_argument("--output", help="Write JSONL to this file instead of stdout")
    args = parser.parse_args()
    if args.k < 2:
        parser.error("--k must be >= 2")
    if not args.judge_base_url and args.provider == "openai":
        args.judge_base_url = DEFAULT_JUDGE_BASE_URL
    if not args.judge_api_key_env:
        args.judge_api_key_env = "GEMINI_API_KEY" if args.provider == "gemini" else "HF_INFERENCE_TOKEN"
    records = score(args.runs, args)
    records.extend(compare_proof(records, args.k))
    lines = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records)
    if args.output:
        Path(args.output).write_text(lines)
    else:
        print(lines, end="")


if __name__ == "__main__":
    main()
