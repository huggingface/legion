#!/usr/bin/env python3
"""CPU-seconds per Claude turn: cold `claude -p` process vs a turn sent to an already-resident
streaming-input session. Sequential, N=1, so it is safe on a shared box.
  resident_cpu.py --turns 3 --base /tmp/x --out out/
"""

import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from packing import PROMPT, TCK, cmd_for, tree_stats

ap = argparse.ArgumentParser()
ap.add_argument("--turns", type=int, default=3)
ap.add_argument("--model", default="claude-haiku-5-5")
ap.add_argument("--base", required=True)
ap.add_argument("--out", default="out")
a = ap.parse_args()
base = Path(a.base)
base.mkdir(parents=True, exist_ok=True)
res = {"cold": [], "resident": []}

for i in range(a.turns):
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    out = subprocess.run(
        cmd_for("claude", a.model),
        cwd=base,
        capture_output=True,
        check=False,
        text=True,
        stdin=subprocess.DEVNULL,
    )
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    r = [
        json.loads(l)
        for l in out.stdout.splitlines()
        if l.startswith("{") and '"type":"result"' in l
    ][-1]
    res["cold"].append(
        {
            "cpu_s": round(ru1.ru_utime + ru1.ru_stime - ru0.ru_utime - ru0.ru_stime, 2),
            "wall_s": round(time.monotonic() - t0, 2),
            "api_ms": r.get("duration_api_ms"),
            "cost": r.get("total_cost_usd"),
            "result": r.get("result"),
        }
    )

p = subprocess.Popen(
    [
        "claude",
        "-p",
        "--model",
        a.model,
        "--max-budget-usd",
        "0.10",
        "--permission-mode",
        "bypassPermissions",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--verbose",
    ],
    cwd=base,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
me = os.getpid()
prev_cost = 0.0
for i in range(a.turns + 1):  # turn 0 includes process startup
    _, k0, _ = tree_stats(me)
    t0 = time.monotonic()
    p.stdin.write(
        json.dumps({"type": "user", "message": {"role": "user", "content": PROMPT}}) + "\n"
    )
    p.stdin.flush()
    for line in p.stdout:
        ev = json.loads(line)
        if ev.get("type") == "result":
            break
    _, k1, _ = tree_stats(me)
    res["resident"].append(
        {
            "turn": i,
            "cpu_s": round((k1 - k0) / TCK, 2),
            "wall_s": round(time.monotonic() - t0, 2),
            "api_ms": ev.get("duration_api_ms"),
            "cost_cum": ev.get("total_cost_usd"),
            "result": ev.get("result"),
        }
    )
p.stdin.close()
p.wait(timeout=30)
Path(a.out).mkdir(parents=True, exist_ok=True)
(Path(a.out) / "claude-resident-cpu.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res))
