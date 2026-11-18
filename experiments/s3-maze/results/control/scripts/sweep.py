"""The s3-maze positive control: the scaling sweep's driver (the dataset's flock/sweep/sweep.py,
unchanged) with one condition applied to every team's spec, from flock's checkout:

    .venv/bin/python -B .flock/s3-maze/control/sweep.py --condition hint|noboard [sweep.py args]

- `hint`: the board, plus `team_file: hint.md` (HINT below): flock appends the team text to
  each agent's instructions. The dataset, the board rules, the loop and the limits are the
  sweep's.
- `noboard`: no `message_board` resource, so no board tools. The task's instruction.md still
  says to use the message board (the dataset is unchanged); the run has no board export, so that
  isn't counted as missing.

Each team's agent sessions (the worker's Pi session logs, synced to the run's bucket) are copied
to `<run dir>/sessions/` after the release, for when agents read and post on the board.
"""

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import HfApi

DS = Path(__file__).resolve().parents[1] / "ds"
sys.path.insert(0, str(DS / "flock" / "sweep"))
import sweep as base  # noqa: E402

HINT = ("Before you go through a door, read the board and post which door you are taking; pick "
        "one no teammate has claimed; post when a branch is a dead end.\n")
CONDITION: str | None = None
BUCKET: str | None = None
_write_experiment, _finish = base.write_experiment, base.finish


def write_experiment(src: Path, tree: int, n: int, name: str, urls: list[str], model: str,
                     revision: str) -> None:
    _write_experiment(src, tree, n, name, urls, model, revision)
    spec = yaml.safe_load((src / "spec.yaml").read_text())
    if CONDITION == "hint":
        (src / "hint.md").write_text(HINT)
        spec["team_file"] = "hint.md"
    elif CONDITION == "noboard":
        spec["resources"].pop("message_board")
        if not spec["resources"]:
            del spec["resources"]
    (src / "spec.yaml").write_text(
        f"# {spec['task']['selector']} with N = {n} for the s3-maze positive control "
        f"({CONDITION}): the scaling sweep's spec with {model}, plus the condition.\n"
        + yaml.safe_dump(spec, sort_keys=False))


def sessions(run_dir: Path, name: str) -> int:
    """The team's session logs from the bucket into `<run dir>/sessions/`, at their bucket paths
    under the swarm: `<type>/<agent>/sessions/<agent>.jsonl` (the worker's) and
    `<type>/<agent>/harness/sessions/*.jsonl` (Pi's, with every tool call); how many."""
    api, out, n = HfApi(), run_dir / "sessions", 0
    try:
        files = [f.path for f in api.list_bucket_tree(BUCKET, prefix=f"{name}/", recursive=True)
                 if "/sessions/" in f.path and f.path.endswith(".jsonl")]
        pairs = [(f, out / f.removeprefix(f"{name}/")) for f in files]
        for _, local in pairs:
            local.parent.mkdir(parents=True, exist_ok=True)
        api.download_bucket_files(BUCKET, [(f, str(local)) for f, local in pairs])
        n = len(files)
    except Exception as e:  # noqa: BLE001 - the row counts without them; say so
        base.say(name, "sessions not copied:", repr(e)[:200])
    return n


def finish(args, run_dir: Path, name: str, tree: int, n: int, started: float,
           why: str | None = None) -> dict:
    row = _finish(args, run_dir, name, tree, n, started, why)
    if CONDITION == "noboard":
        row["missing"] = [m for m in row["missing"] if m != "board export"]
    row["condition"] = CONDITION
    row["sessions"] = sessions(run_dir, name)
    return row


base.write_experiment, base.finish = write_experiment, finish

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--condition", choices=["hint", "noboard"], required=True)
    parser.add_argument("--bucket", default="lewtun/s3-maze-control")
    known, rest = parser.parse_known_args()
    CONDITION, BUCKET = known.condition, known.bucket
    raise SystemExit(base.main(rest + ["--bucket", BUCKET]))
