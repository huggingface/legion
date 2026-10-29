"""Summarize completed benchmark cases without extrapolating missing configurations."""

import argparse
import collections
import json
import statistics
from pathlib import Path


def analyze(directory):
    runtime_path = directory / "runtime.json"
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    required = runtime.get("required_repetitions", 3)
    expected = runtime.get("expected_cases", 168)
    summary = directory / "summary.jsonl"
    rows = [json.loads(line) for line in summary.read_text().splitlines()]
    groups = collections.defaultdict(list)
    for row in rows:
        groups[(row["dataset"], row["layout"], row["gpus"], row["concurrency"])].append(row)
    aggregate = []
    for (dataset, layout, gpus, concurrency), cases in sorted(groups.items()):
        elapsed = sum(c["seconds"] for c in cases)
        complete = sum(c["completed"] for c in cases)
        requests = sum(c["requests"] for c in cases)
        aggregate.append(
            {
                "dataset": dataset,
                "layout": layout,
                "gpus": gpus,
                "concurrency": concurrency,
                "repetitions": len(cases),
                "output_tokens_s": sum(c["output_tokens_s"] * c["seconds"] for c in cases)
                / elapsed,
                "output_tokens_s_stdev": statistics.stdev(c["output_tokens_s"] for c in cases)
                if len(cases) > 1
                else 0,
                "requests_s": complete / elapsed,
                "error_rate": 1 - complete / requests,
                "mean_p50_latency_s": statistics.mean(
                    c["latency_s_p50"] for c in cases if c["latency_s_p50"] is not None
                )
                if complete
                else None,
            }
        )
    eligible = [r for r in aggregate if r["repetitions"] == required and r["error_rate"] == 0]
    throughput = [r for r in eligible if r["dataset"] == "ultrachat"]
    latency = [r for r in eligible if r["dataset"] == "agent_escape" and r["concurrency"] == 4]
    report = {
        "completed_cases": len(rows),
        "expected_cases": expected,
        "coverage_complete": len(rows) == expected,
        "aggregates": aggregate,
        "best_measured_throughput": max(
            throughput, key=lambda r: r["output_tokens_s"], default=None
        ),
        "best_measured_four_agent_latency": min(
            latency, key=lambda r: r["mean_p50_latency_s"], default=None
        ),
        "note": f"Recommendations require {required} error-free repetition(s); rankings cover "
        "measured cases only. Means of per-case latency medians are labeled as such.",
    }
    return report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("measurements", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.measurements), indent=2))


if __name__ == "__main__":
    main()
