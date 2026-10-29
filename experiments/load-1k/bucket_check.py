"""Per-agent bucket completeness from the bucket alone (no controller calls):

    python bucket_check.py hf://buckets/cmpatino/flock/load-1k load-1k [--sample 20]

An agent prefix <swarm>/<type>/<agent>/ is complete if it holds workdir/ files, its session log
sessions/<agent>.jsonl and a harness transcript harness/projects/<enc-cwd>/<id>.jsonl. For a
random sample, also checks that the transcript id matches session_id in <agent>.state.json.
"""

import argparse
import json
import random
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from huggingface_hub import BucketFile, HfApi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("swarm")
    ap.add_argument("--sample", type=int, default=20)
    a = ap.parse_args()
    ns, name, *rest = a.url.removeprefix("hf://buckets/").strip("/").split("/")
    bucket, base = f"{ns}/{name}", "/".join([*rest, a.swarm]) + "/"
    api = HfApi()
    t0 = time.time()
    files: dict[str, list[str]] = defaultdict(list)
    newest: dict[str, str] = {}
    size = 0
    for f in api.list_bucket_tree(bucket, prefix=base, recursive=True):
        if isinstance(f, BucketFile):
            _type, agent, rel = f.path[len(base):].split("/", 2)
            files[agent].append(rel)
            size += f.size or 0
            mtime = str(getattr(f, "mtime", "") or getattr(f, "last_modified", "") or "")
            newest[agent] = max(newest.get(agent, ""), mtime)
    list_s = time.time() - t0
    status = {}
    for agent, got in files.items():
        status[agent] = {
            "workdir": any(p.startswith("workdir/") for p in got),
            "session_log": f"sessions/{agent}.jsonl" in got,
            "transcript": any(p.startswith("harness/projects/") and p.endswith(".jsonl") for p in got),
        }
    complete = [x for x, s in status.items() if all(s.values())]
    sample = random.sample(complete, min(a.sample, len(complete)))
    match = 0
    with tempfile.TemporaryDirectory() as tmp:
        plan = []
        for agent in sample:
            typ = agent.rsplit("-", 1)[0]
            plan.append((f"{base}{typ}/{agent}/sessions/{agent}.state.json", str(Path(tmp) / f"{agent}.json")))
        api.download_bucket_files(bucket, plan)
        for agent in sample:
            sid = json.loads((Path(tmp) / f"{agent}.json").read_text()).get("session_id")
            match += any(p.endswith(f"/{sid}.jsonl") for p in files[agent] if p.startswith("harness/"))
    print(json.dumps({
        "agent_prefixes": len(files), "complete": len(complete),
        "with_workdir": sum(s["workdir"] for s in status.values()),
        "with_session_log": sum(s["session_log"] for s in status.values()),
        "with_transcript": sum(s["transcript"] for s in status.values()),
        "files": sum(map(len, files.values())), "bytes": size, "list_s": round(list_s, 1),
        "sample_transcript_matches_state": f"{match}/{len(sample)}",
        "example": sorted(files[complete[0]]) if complete else None,
        "incomplete": [{"agent": x, **s} for x, s in status.items() if not all(s.values())][:5],
    }, indent=1))


if __name__ == "__main__":
    main()
