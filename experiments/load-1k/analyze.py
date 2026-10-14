"""Summarise a prompt_load.py run: latencies overall and per worker, replies, tokens, cost.

    python analyze.py out/sent.jsonl [--swarm load-1k] [--out summary.json]

first event = first event in the agent's stream after the send (server ts) minus the client send
time; turn = turn_started -> turn_ended (server ts); e2e = turn_ended - send. Controller cost and
Claude's estimate come from the agents' status (cumulative, so run this right after a run).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import httpx

from swarm_harness.controller.prices import cost_usd
from swarm_harness.models import TokenUsage


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))], 2) if xs else None


def stats(rows):
    fe = [r["first_event_ts"] - r["sent"] for r in rows if "first_event_ts" in r]
    turn = [r["turn_ended_ts"] - r["turn_started_ts"] for r in rows if "turn_ended_ts" in r]
    e2e = [r["turn_ended_ts"] - r["sent"] for r in rows if "turn_ended_ts" in r]
    ack = [r["ack_s"] for r in rows if "ack_s" in r]
    return {"n": len(rows), "turn_ended": len(turn), "ok": sum(bool(r.get("ok")) for r in rows),
            "first_event_p50": pct(fe, 50), "first_event_p95": pct(fe, 95),
            "turn_p50": pct(turn, 50), "turn_p95": pct(turn, 95), "turn_max": pct(turn, 100),
            "e2e_p50": pct(e2e, 50), "e2e_p95": pct(e2e, 95), "e2e_max": pct(e2e, 100),
            "ack_p50": pct(ack, 50), "ack_p95": pct(ack, 95), "ack_max": pct(ack, 100)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sent")
    ap.add_argument("--swarm", default="load-1k")
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = [json.loads(x) for x in open(a.sent) if x.strip()]
    by_worker = defaultdict(list)
    for r in rows:
        by_worker[r.get("worker")].append(r)
    tokens: dict[str, TokenUsage] = {}
    for r in rows:
        for m, u in (r.get("tokens") or {}).items():
            tokens[m] = tokens.get(m, TokenUsage()) + TokenUsage(**u)
    sent = [r["sent"] for r in rows if "sent" in r]
    ended = [r["turn_ended_ts"] for r in rows if "turn_ended_ts" in r]
    out = {
        "overall": stats(rows),
        "per_worker": {w: stats(rs) for w, rs in sorted(by_worker.items(), key=lambda x: str(x[0]))},
        "acks": dict(Counter(r.get("ack", {}).get("status") for r in rows)),
        "session_events": dict(Counter(r.get("first_event_type") for r in rows)),
        "anomalies": dict(Counter(an["type"] for r in rows for an in r.get("anomalies", []))),
        "exceptions": [r["exception"] for r in rows if "exception" in r][:5],
        "bad_replies": [{k: r.get(k) for k in ("agent", "codeword", "reply", "status")}
                        for r in rows if "turn_ended_ts" in r and not r.get("ok")][:10],
        "send_window_s": round(max(sent) - min(sent), 1) if sent else None,
        "first_send_to_last_end_s": round(max(ended) - min(sent), 1) if ended else None,
        "tokens_this_run": {m: u.model_dump() for m, u in tokens.items()},
        "cost_this_run_usd": round(cost_usd(tokens), 6),
    }
    if os.environ.get("SWARM_HARNESS_URL"):
        c = httpx.Client(base_url=os.environ["SWARM_HARNESS_URL"], timeout=60,
                         headers={"Authorization": f"Bearer {os.environ['SWARM_HARNESS_TOKEN']}"})
        agents, cursor = [], None
        while True:
            page = c.get(f"/swarms/{a.swarm}/agents", params={"limit": 500, **({"cursor": cursor} if cursor else {})}).json()
            agents += page["agents"]
            cursor = page["cursor"]
            if not cursor:
                break
        out["controller_cost_usd_total"] = round(sum(x["cost_usd"] for x in agents), 6)
        out["claude_estimate_usd_total"] = round(sum(x["harness_estimate_usd"] or 0 for x in agents), 6)
        out["agents_by_state"] = dict(Counter(x["state"] for x in agents))
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
