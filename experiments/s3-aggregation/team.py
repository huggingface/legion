"""Write the flock spec for one s3-aggregation team:

    uv run experiments/s3-aggregation/team.py --problem 7 --agents 4 [--where sandbox|cluster]

writes `out/agg-p07-n04/spec.yaml` (or `--out DIR`) from spec.template.yaml: the task
`agg-p07-n04`, 4 agents (agent-001 .. agent-004), each with its role file, the dataset's
`roles/agg-p07-n04/slot-NN.md` at the template's pinned revision, copied into `DIR/roles/`.
`--where` picks the template's `where:` block: HF Sandboxes with a Space board, or the HF
cluster with a local board. The dataset is private for now: HF_TOKEN must read it.
"""

import argparse
import copy
import shutil
import sys
import tempfile
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

from flock.models import load_spec_yaml

HERE = Path(__file__).resolve().parent
DATASET = "HuggingFaceSwarms/s3-aggregation"
PROBLEMS, SIZES = range(1, 33), (1, 2, 4, 8, 16)


def merge(base: dict, extra: dict) -> dict:
    out = dict(base)
    for key, value in extra.items():
        both = isinstance(value, dict) and isinstance(out.get(key), dict)
        out[key] = merge(out[key], value) if both else value
    return out


def fill(value, task: str):
    if isinstance(value, dict):
        return {k: fill(v, task) for k, v in value.items()}
    if isinstance(value, list):
        return [fill(v, task) for v in value]
    return value.replace("TEAM", task) if isinstance(value, str) else value


def team_spec(template: dict, problem: int, agents: int, where: str) -> tuple[str, dict]:
    """(the task's name, the team's spec) from the template."""
    template = copy.deepcopy(template)
    places = template.pop("where")
    task = f"agg-p{problem:02d}-n{agents:02d}"
    spec = fill(merge(template, places[where]), task)
    spec["agents"] = {"count": agents,
                      "role_files": [f"roles/slot-{k:02d}.md" for k in range(1, agents + 1)]}
    spec["workers"]["agents_per_worker"] = agents
    return task, spec


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Write one s3-aggregation team's flock spec.")
    parser.add_argument("--problem", type=int, required=True, choices=PROBLEMS)
    parser.add_argument("--agents", type=int, required=True, choices=SIZES)
    parser.add_argument("--where", choices=("sandbox", "cluster"), default="sandbox")
    parser.add_argument("--out", type=Path, help="default: out/<task> next to this script")
    args = parser.parse_args(argv)
    template = load_spec_yaml((HERE / "spec.template.yaml").read_text())
    task, spec = team_spec(template, args.problem, args.agents, args.where)
    out = args.out or HERE / "out" / task
    roles = out / "roles"
    shutil.rmtree(roles, ignore_errors=True)
    roles.mkdir(parents=True)
    with tempfile.TemporaryDirectory() as d:
        snapshot_download(DATASET, repo_type="dataset", revision=spec["task"]["revision"],
                          allow_patterns=[f"roles/{task}/*"], local_dir=d)
        files = sorted((Path(d) / "roles" / task).glob("slot-*.md"))
        if len(files) != args.agents:
            raise SystemExit(f"{DATASET}: {len(files)} role files for {task}, not {args.agents}")
        for path in files:
            shutil.copy(path, roles / path.name)
    header = (f"# {task} ({args.where}), written by experiments/s3-aggregation/team.py from "
              "spec.template.yaml: edit the template, not this file.\n")
    (out / "spec.yaml").write_text(header + yaml.safe_dump(spec, sort_keys=False, width=100))
    print(out / "spec.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(main())
