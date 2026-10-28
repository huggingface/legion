"""Snapshot code and prompts, then submit one sequential B300 benchmark allocation."""

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def submit(data, output_root, mode="benchmark", layout="tp", gpus=1):
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    data = data.resolve()
    manifest = json.loads((data / "manifest.json").read_text())
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    run = output_root.resolve() / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ")
    run.mkdir(parents=True, mode=0o700)
    shutil.copytree(data, run / "data")
    (run / "source").mkdir()
    for name in ("benchmark.py", "serve.sbatch", "prepare.py", "submit.py"):
        shutil.copy2(here / name, run / "source" / name)
    source = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run / "source").iterdir()
    }
    (run / "source-manifest.json").write_text(json.dumps(source, indent=2))
    (run / "git-revision.txt").write_text(
        subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True)
    )
    command = [
        "sbatch",
        "--parsable",
        "--output",
        str(run / "slurm.log"),
        str(run / "source/serve.sbatch"),
        str(repo),
        str(run / "data"),
        str(run),
        mode,
        layout,
        str(gpus),
    ]
    job = subprocess.check_output(command, text=True).strip().split(";")[0]
    result = {"job_id": job, "run": str(run), "mode": mode, "command": command}
    (run / "submission.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output-root", type=Path, default=Path(".flock/b300-qwen/runs"))
    p.add_argument("--mode", choices=("benchmark", "serve"), default="benchmark")
    p.add_argument("--layout", choices=("tp", "replicas"), default="tp")
    p.add_argument("--gpus", type=int, choices=(1, 2, 4, 8), default=1)
    args = p.parse_args()
    print(
        json.dumps(submit(args.data, args.output_root, args.mode, args.layout, args.gpus), indent=2)
    )


if __name__ == "__main__":
    main()
