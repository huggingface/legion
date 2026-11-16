"""Why each set-aside attempt of a grid didn't count: `uv run experiments/s3-aggregation/sweep/
setaside.py <root>`, from `<root>/rows-discarded.jsonl` and the attempt's own logs. A restarted
driver scores an attempt again from what it left, so one attempt can have several entries: its
reason uses them all. Writes `<root>/setaside.md`; the reasons, first that applies:

- the worker was relaunched (more than one incarnation: #121's case) after it ran out of memory
  (the worker's memory ladder rotated or preempted it), or else after a node failure;
- the controller's node was reclaimed (spot) mid-run;
- the task image couldn't be pulled (Docker Hub's unauthenticated limit), so the Trials never
  started;
- the verifier timed out (Harbor's 60 s) on a loaded worker;
- the run was cancelled before its Trials were graded (the driver stopped and its in-flight
  teams cancelled, e.g. to switch the workers to the image mirror).
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
    controllers = " ".join(v for k, v in logs.items() if k.startswith("flock-ctl-"))
    if (row.get("incarnations") or 0) > 1:
        if re.search(r"preempted \(memory\)|draining \(memory\)|rotation wanted", workers):
            return "worker relaunched after running out of memory (#121's case)"
        return "worker relaunched after a node failure (#121's case)"
    if "DUE TO NODE FAILURE" in controllers or row.get("outcome") == "controller lost":
        return "controller's node reclaimed (spot)"
    if "toomanyrequests" in workers:
        return "task image not pulled (Docker Hub limit): Trials never started"
    if any("VerifierTimeoutError" in p.read_text(errors="replace")
           for p in run_dir.glob("results/*/agent-*/trials/*/result.json")):
        return "verifier timed out (60 s) on a loaded worker"
    return "cancelled before grading (driver stopped, in-flight teams cancelled)"


def main(root: Path) -> int:
    attempts = {}
    for line in (root / "rows-discarded.jsonl").read_text().splitlines():
        row = json.loads(line)
        seen = attempts.setdefault(row["run"], row)  # the driver that watched it knew most
        seen["incarnations"] = max(seen.get("incarnations") or 0, row.get("incarnations") or 0)
        if row.get("outcome") == "controller lost":
            seen["outcome"] = row["outcome"]
    kept = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
    reasons = Counter(reason(r, root / "runs" / run) for run, r in attempts.items())
    rerun = sum(r.get("reruns", 0) > 0 for r in kept)
    head = (f"Attempts set aside: {len(attempts)} (rows-discarded.jsonl); "
            f"teams that needed a rerun: {rerun} of {len(kept)}")
    lines = [head, ""]
    lines += [f"- {why}: {k}" for why, k in reasons.most_common()]
    (root / "setaside.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
