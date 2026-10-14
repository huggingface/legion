"""Check every agent's bucket prefix: workdir/, sessions/<name>.jsonl, harness/.../<session>.jsonl.

    python bucket_check.py <bucket-url-with-prefix> <swarm>     (SWARM_HARNESS_URL/TOKEN for ids)
"""

from __future__ import annotations

import json
import os
import sys
from collections import defaultdict

import httpx
from huggingface_hub import BucketFile, HfApi


def main() -> None:
    url, swarm = sys.argv[1], sys.argv[2]
    ns, name, *rest = url.removeprefix("hf://buckets/").strip("/").split("/")
    base = "/".join([*rest, swarm]) + "/"
    c = httpx.Client(base_url=os.environ["SWARM_HARNESS_URL"], timeout=60,
                     headers={"Authorization": f"Bearer {os.environ['SWARM_HARNESS_TOKEN']}"})
    agents = c.get(f"/swarms/{swarm}/agents", params={"limit": 5000}).json()["agents"]
    files: dict[str, list[str]] = defaultdict(list)
    sizes = 0
    for f in HfApi().list_bucket_tree(f"{ns}/{name}", prefix=base, recursive=True):
        if isinstance(f, BucketFile):
            _type, agent, rel = f.path[len(base):].split("/", 2)
            files[agent].append(rel)
            sizes += f.size or 0
    bad = []
    for a in agents:
        got = files.get(a["name"], [])
        ok = {
            "workdir": any(p.startswith("workdir/") for p in got),
            "session_log": f"sessions/{a['name']}.jsonl" in got,
            "harness_transcript": bool(a["session_id"]) and any(
                p.startswith("harness/projects/") and p.endswith(f"/{a['session_id']}.jsonl") for p in got),
        }
        if not all(ok.values()):
            bad.append({"agent": a["name"], **ok, "files": got})
    print(json.dumps({"agents": len(agents), "complete": len(agents) - len(bad),
                      "files": sum(map(len, files.values())), "bytes": sizes,
                      "example": {k: sorted(v) for k, v in list(files.items())[:1]},
                      "incomplete": bad[:5]}, indent=1))


if __name__ == "__main__":
    main()
