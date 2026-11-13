"""Summarize downloaded run artifacts without treating missing verification as zero reward.

Usage: python report.py RUN_DIRECTORY
Writes summary.json beside the collected run artifacts. Never reads credential files.
"""
import json
import sys
from pathlib import Path


def summarize(root):
    trials = []
    for path in sorted(root.rglob("result.json")):
        data = json.loads(path.read_text())
        if "verifier_result" not in data:
            continue
        verifier = data.get("verifier_result") or {}
        agent = data.get("agent_result") or {}
        error = data.get("exception_info") or {}
        trials.append({
            "trial": path.parent.name,
            "result": str(path.relative_to(root)),
            "verified": bool(verifier.get("rewards")),
            "rewards": verifier.get("rewards"),
            "exception_type": error.get("exception_type"),
            "input_tokens": agent.get("n_input_tokens"),
            "output_tokens": agent.get("n_output_tokens"),
            "cached_tokens": agent.get("n_cache_tokens"),
        })
    return {
        "trials": trials,
        "verified_trials": sum(t["verified"] for t in trials),
        "missing_verifier_trials": sum(not t["verified"] for t in trials),
        "token_accounting_complete": bool(trials) and all(
            t["input_tokens"] is not None and t["output_tokens"] is not None for t in trials),
        "total_input_tokens": sum(t["input_tokens"] or 0 for t in trials),
        "total_output_tokens": sum(t["output_tokens"] or 0 for t in trials),
        "note": "Trials are not independent samples. Missing rewards remain null, not zero. "
                "Check final candidate/submission SHAs and native evaluator logs separately. "
                "Router hardware cost is not inferred from a zero API price.",
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    result = summarize(root)
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
