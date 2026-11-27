"""Run one CIB dimension through `legion launch`, case by case, for the parity check against
the dataset's published results (until `legion sweep` exists).

Every solo case runs first; then the three team seeds of each scenario whose solo passed, as
the benchmark's gate does. Each case is one local run of spec.yaml (CASE set) in its own folder,
`<output>/cases/<task id>/`, with at most `--concurrency` at once. A case with a verifier result
is not run again; one without (an infrastructure failure) is moved aside as `<id>.failed-N` and
run again on the next invocation. The end prints the dataset scorer's command over the results.

Only manifests without aliases and reader grading (Sharing: `manifest.json`) are supported.

    uv run python experiments/cib/parity.py --dataset ../cib-harbor --manifest manifest.json \
        --output results/cib/parity-sharing --concurrency 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from legion.current import read_current

SPEC = Path(__file__).resolve().with_name("spec.yaml")
POLL_S = 30


def result(case: Path) -> Path | None:
    """The case's verifier result (the dataset's run envelope), or None."""
    found = sorted(case.glob(".legion/*/agents/_harbor/*/trials/*/verifier/result.json"))
    return found[-1] if found else None


def run_case(output: Path, task_id: str) -> Path | None:
    case = output / "cases" / task_id
    if result(case) is not None:
        return result(case)
    if case.exists():  # an earlier attempt without a result: keep it for the record
        n = 1
        while case.with_name(f"{task_id}.failed-{n}").exists():
            n += 1
        case.rename(case.with_name(f"{task_id}.failed-{n}"))
    case.mkdir(parents=True)
    env = {**os.environ, "CASE": task_id,
           "RUN_NAME": "cib-" + hashlib.sha256(task_id.encode()).hexdigest()[:8]}
    with (case / "launch.out").open("w") as out:
        code = subprocess.run([sys.executable, "-m", "legion", "launch", str(SPEC)], cwd=case,
                              env=env, stdout=out, stderr=subprocess.STDOUT, check=False).returncode
    if code == 0:  # the run is up in the background: it ends after its one turn and grading
        while read_current(case / ".legion" / "current") is not None:
            time.sleep(POLL_S)
    found = result(case)
    print(f"{task_id}: {'graded' if found else f'no verifier result (see {case})'}", flush=True)
    return found


def solo_passed(path: Path | None) -> bool:
    """The benchmark's gate: a finished solo whose checker passed, not refused or judge-failed."""
    if path is None:
        return False
    run = json.loads(path.read_text())
    transcript = run.get("transcript") or {}
    return (run.get("status") == "finished" and (transcript.get("score") or {}).get("pass") is True
            and not transcript.get("refused") and not transcript.get("judge_failed"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset", type=Path, required=True,
                        help="a clone of the dataset at the spec's task revision")
    parser.add_argument("--manifest", default="manifest.json", help="relative to --dataset")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads((args.dataset / args.manifest).read_text())
    tasks = manifest["tasks"]
    if manifest.get("grading_protocol") or any("canonical_task_id" in t for t in tasks):
        parser.error(f"{args.manifest}: aliases and reader grading need the dataset's own sweep")
    output = args.output.resolve()
    solos = [t for t in tasks if t["identity"]["variant"] == "solo"]
    with ThreadPoolExecutor(max(1, args.concurrency)) as pool:
        done = dict(zip([t["task_id"] for t in solos],
                        pool.map(lambda t: run_case(output, t["task_id"]), solos), strict=True))
        passed = {t["identity"]["scenario_id"] for t in solos if solo_passed(done[t["task_id"]])}
        seeds = [t for t in tasks if t["identity"]["variant"] == "manipulation"
                 and t["identity"]["scenario_id"] in passed]
        done |= dict(zip([t["task_id"] for t in seeds],
                         pool.map(lambda t: run_case(output, t["task_id"]), seeds), strict=True))
    runs = sorted(str(p) for p in done.values() if p is not None)
    (output / "runs.txt").write_text("\n".join(runs) + "\n")
    print(f"{len(passed)}/{len(solos)} solos passed; {len(runs)}/{len(solos) + len(seeds)} "
          f"cases graded. Score them with the dataset's scorer:")
    tests = args.dataset.resolve() / solos[0]["path"] / "tests"
    print(f"  xargs -a {output}/runs.txt uv run --with '{args.dataset.resolve()}/adapter' "
          f"python -m collab_harbor.scoring --manifest {args.dataset.resolve() / args.manifest} "
          f"--upstream-root {tests} --output {output}/scores.json --runs")
    return 0 if len(runs) == len(solos) + len(seeds) else 1


if __name__ == "__main__":
    raise SystemExit(main())
