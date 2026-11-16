"""How long each kept team's worker waited in the Slurm queue: `uv run experiments/s3-aggregation/
sweep/waits.py <root>`. The loop's `until` counts from the launch, so a worker that starts late
leaves its agents less of the 15 minutes; a run whose loop ran out before its worker started has
no rewards and was run again (sweep.py). The wait is from the controller writing the worker's
launch file to the worker job's first log line.
"""

import json
import statistics
import sys
from datetime import datetime
from pathlib import Path


def wait_s(run_dir: Path) -> float | None:
    launches = sorted((run_dir / "launch").glob("*.env"), key=lambda p: p.stat().st_mtime)
    logs = sorted((run_dir / "logs").glob("*-agent-w*.out"))
    if not launches or not logs:
        return None
    first = logs[0].read_text(errors="replace").split("\n", 1)[0]
    try:
        started = datetime.fromisoformat(first.split(" ", 1)[0])
    except ValueError:
        return None
    return started.timestamp() - launches[0].stat().st_mtime


def main(root: Path) -> int:
    rows = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
    waits = [(r, wait_s(root / "runs" / r["run"])) for r in rows]
    known = [(r, w) for r, w in waits if w is not None]
    long = [(r, w) for r, w in known if w > 300]
    print(f"teams: {len(rows)}, worker wait known for {len(known)}: median "
          f"{statistics.median(w for _, w in known):.0f} s, max {max(w for _, w in known):.0f} s; "
          f"over 5 min: {len(long)} (succeeded: {sum(bool(r['success']) for r, _ in long)}; "
          f"by N: {sorted(r['agents'] for r, _ in long)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
