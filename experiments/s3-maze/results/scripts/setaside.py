"""Why each set-aside attempt of a grid didn't count: `python flock/sweep/setaside.py <root>`,
from `<root>/rows-discarded.jsonl` and the attempt's own logs (s3-aggregation's, for the maze).
A restarted driver scores an attempt again from what it left, so one attempt can have several
entries: its reason uses them all. Writes `<root>/setaside.md`; the reasons, first that applies:

- it ran while its model server was overloaded (`--overloaded`), or an agent's server thrashed
  for 5 minutes or more (`--health`);
- the worker was relaunched (more than one incarnation: flock#121's case), after running out of
  memory or after a node failure;
- the launch failed (or was cancelled), so its controller never came up;
- the worker never started, or started more than 2 min after the launch (the loop counts from
  the launch);
- the controller's node was reclaimed (spot) mid-run;
- the task images couldn't be pulled or built, so the Trials never started;
- the worker ran out of memory;
- the verifier timed out on a loaded worker;
- an agent's Trial failed before its verifier ran;
- the run was cancelled before its Trials were graded (e.g. the driver stopped).

What the model does (no move, no post, no exit) is never a reason: those runs count.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path


def reason(row: dict, run_dir: Path) -> str:
    logs = {p.name: p.read_text(errors="replace") for p in (run_dir / "logs").glob("*")
            if p.is_file()} if (run_dir / "logs").is_dir() else {}
    workers = " ".join(v for k, v in logs.items() if "-agent-w" in k)
    controllers = " ".join(v for k, v in logs.items() if k.startswith(("flock-ctl-",
                                                                         "controller-")))
    if "overloaded model server" in (row.get("set_aside") or ""):
        return "ran on an overloaded model server (one 27B GPU, KV-cache thrash)"
    if (row.get("thrashed_min") or 0) >= 5:  # sweep.py's THRASH_LIMIT
        return "an agent's model server thrashed (prefix-cache hits under 50%) for 5+ min"
    if (row.get("incarnations") or 0) > 1:
        if re.search(r"CANCELED: unresponsive", controllers):
            return "worker stopped answering (e.g. while grading its Trials) and was relaunched"
        if re.search(r"preempted \(memory\)|draining \(memory\)|rotation wanted", workers):
            return "worker relaunched after running out of memory (flock#121's case)"
        return "worker relaunched after a node failure (flock#121's case)"
    if row.get("outcome") == "launch":
        return "launch failed or was cancelled (the controller ended before it came up)"
    if row.get("worker_wait_s") is None:
        return "worker never started"
    if row["worker_wait_s"] > 120:  # sweep.py's WAIT_LIMIT
        return "worker started more than 2 min after the launch (the loop counts from it)"
    if "DUE TO NODE FAILURE" in controllers or row.get("outcome") == "controller lost":
        return "controller's node reclaimed (spot)"
    if re.search(r"toomanyrequests|Error: building at STEP|failed to build", workers):
        return "task images not pulled or built: Trials never started"
    if re.search(r"preempted \(memory\)|oom_kill", workers):
        return "worker ran out of memory (preempted or killed, not relaunched)"
    results = list(run_dir.glob("results/*/agent-*/trials/*/result.json"))
    if any("VerifierTimeoutError" in p.read_text(errors="replace") for p in results):
        return "verifier timed out on a loaded worker"
    failed = [(json.loads(p.read_text()).get("exception_info") or {}).get("exception_type")
              for p in results]
    if kinds := sorted({k for k in failed if k and k != "CancelledError"}):
        return f"an agent's Trial failed before its verifier ran ({', '.join(kinds)})"
    return "cancelled before grading (e.g. the driver stopped)"


def main(root: Path) -> int:
    attempts = {}
    path = root / "rows-discarded.jsonl"
    for line in path.read_text().splitlines() if path.exists() else []:
        row = json.loads(line)
        seen = attempts.setdefault(row["run"], row)  # the driver that watched it knew most
        seen["incarnations"] = max(seen.get("incarnations") or 0, row.get("incarnations") or 0)
        seen["set_aside"] = "; ".join(filter(None, {seen.get("set_aside"), row.get("set_aside")}))
        if row.get("outcome") == "controller lost":
            seen["outcome"] = row["outcome"]
    kept = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines() if x]
    reasons = Counter(reason(r, root / "runs" / run) for run, r in attempts.items())
    counted = {(r["tree"], r["agents"]) for r in kept}
    missing = sorted({(r["tree"], r["agents"]) for r in attempts.values()} - counted)
    rerun = sum(r.get("reruns", 0) > 0 for r in kept) + len(missing)
    lines = [(f"Attempts set aside: {len(attempts)} (rows-discarded.jsonl); teams that needed a "
              f"rerun: {rerun} of {len(counted) + len(missing)}"), ""]
    lines += [f"- {why}: {k}" for why, k in reasons.most_common()]
    if missing:
        lines += ["", f"Teams with no attempt that counted ({len(missing)}; missing from the "
                      "results, not failures): " + ", ".join(f"tree {t:02d} at N = {n}"
                                                             for t, n in missing)]
    (root / "setaside.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
