"""Poll the controller every N seconds: swarm status, workers and /healthz -> one JSON line each.

    python monitor.py load-1k out/monitor.jsonl [--every 15]
"""

from __future__ import annotations

import argparse
import json
import os
import time

import httpx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("swarm")
    ap.add_argument("out")
    ap.add_argument("--every", type=float, default=15)
    a = ap.parse_args()
    c = httpx.Client(base_url=os.environ["FLOCK_URL"], timeout=30,
                     headers={"Authorization": f"Bearer {os.environ['FLOCK_TOKEN']}"})
    while True:
        rec: dict = {"t": time.time()}
        for key, path in (("health", "/healthz"), ("swarm", f"/swarms/{a.swarm}"),
                          ("workers", "/workers")):
            t0 = time.time()
            try:
                r = c.get(path, params={"swarm": a.swarm} if key == "workers" else None)
                rec[key] = r.json() if r.status_code == 200 else {"status": r.status_code}
            except Exception as e:  # noqa: BLE001
                rec[key] = {"error": repr(e)[:200]}
            rec[f"{key}_s"] = round(time.time() - t0, 3)
        if isinstance(rec.get("swarm"), dict) and "spec" in rec["swarm"]:
            rec["swarm"].pop("spec")
        if isinstance(rec.get("workers"), list):
            for w in rec["workers"]:
                w["agents"] = len(w["agents"])
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        time.sleep(a.every)


if __name__ == "__main__":
    main()
