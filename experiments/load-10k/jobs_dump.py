"""Dump every job labelled experiment=flock-10k: one JSON line per job (id, name, labels,
stage, message, created/started/ended, flavor) to <out>/jobs.jsonl, and the log of each job to
<out>/logs/<id>.log (terminal jobs once; running jobs re-fetched each time).

    python jobs_dump.py out [--no-logs] [--label experiment=flock-10k]
"""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi

TERMINAL = {"COMPLETED", "CANCELED", "ERROR", "DELETED"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--no-logs", action="store_true")
    ap.add_argument("--label", default="experiment=flock-10k")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    key, value = a.label.split("=", 1)
    api = HfApi()
    rows = []
    for j in api.list_jobs(labels={key: value}, namespace="cmpatino"):
        st = j.status
        rows.append({
            "id": j.id, "labels": dict(j.labels or {}), "stage": str(getattr(st.stage, "value", st.stage)),
            "message": st.message, "created_at": str(j.created_at), "flavor": str(j.flavor),
            "started_at": str(j.started_at), "finished_at": str(j.finished_at),
            "durations": None if j.durations is None else vars(j.durations),
        })
    rows.sort(key=lambda r: r["created_at"])
    with open(out / "jobs.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    by_stage: dict[str, int] = {}
    for r in rows:
        by_stage[r["stage"]] = by_stage.get(r["stage"], 0) + 1
    print(json.dumps({"jobs": len(rows), "by_stage": by_stage}))
    if a.no_logs:
        return
    for r in rows:
        path = out / "logs" / f"{r['id']}.log"
        if r["stage"] in TERMINAL and path.exists() and path.stat().st_size > 0:
            continue
        try:
            lines = list(api.fetch_job_logs(job_id=r["id"], namespace="cmpatino"))
        except Exception as e:  # noqa: BLE001
            print(f"{r['id']}: logs failed: {type(e).__name__}")
            continue
        path.write_text("\n".join(lines) + "\n")
    print("logs saved")


if __name__ == "__main__":
    main()
