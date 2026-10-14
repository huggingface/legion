"""Download a prompter run's slices from the bucket and concatenate them into one JSONL.

    python fetch_runs.py round1 out/sent1.jsonl
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import BucketFile, HfApi

BUCKET = "cmpatino/flock"
PREFIX = "load-10k-runs/"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("out")
    a = ap.parse_args()
    api = HfApi()
    paths = sorted(f.path for f in api.list_bucket_tree(BUCKET, prefix=f"{PREFIX}{a.run}/", recursive=True)
                   if isinstance(f, BucketFile) and f.path.endswith(".jsonl"))
    with tempfile.TemporaryDirectory() as tmp, open(a.out, "w") as out:
        plan = [(p, str(Path(tmp) / Path(p).name)) for p in paths]
        api.download_bucket_files(BUCKET, plan)
        n = 0
        for _, local in plan:
            for line in Path(local).read_text().splitlines():
                if line.strip():
                    out.write(line + "\n")
                    n += 1
    print(f"{len(paths)} slices, {n} records -> {a.out}")


if __name__ == "__main__":
    main()
