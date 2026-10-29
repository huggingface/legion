"""Turn metrics for prompted agents, from the controller's per-agent event streams.

    python measure.py <sent.jsonl> [--since EPOCH] [--out summary.json]

For each agent in sent.jsonl (from prompt_all.sh, or any file of {"agent","sent","codeword"}),
reads /agents/<name>/events and /agents/<name>, takes the first turn whose user message contains
the prompt sent at `sent`, and reports: time to first event after the prompt, turn latency
(turn_started -> turn_ended), reply == codeword, tokens, controller cost vs Claude's estimate.
"""

import argparse
import json
import os
import statistics
from datetime import datetime

import httpx


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sent")
    ap.add_argument("--out")
    ap.add_argument("--expect", choices=["codeword", "none"], default="codeword")
    args = ap.parse_args()
    c = httpx.Client(base_url=os.environ["FLOCK_URL"], timeout=60,
                     headers={"Authorization": f"Bearer {os.environ['FLOCK_TOKEN']}"})
    with open(args.sent) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    per, acks = [], {}
    for r in rows:
        if "ack" in r:
            acks[r["ack"]["status"]] = acks.get(r["ack"]["status"], 0) + 1
        evs = c.get(f"/agents/{r['agent']}/events", params={"limit": 5000}).json()["events"]
        after = [e for e in evs if ts(e["ts"]) >= r["sent"] - 1]
        started = next((e for e in after if e["type"] == "turn_started"), None)
        ended = next((e for e in after if e["type"] == "turn_ended"
                      and started and e.get("turn_id") == started.get("turn_id")), None)
        session = next((e["type"] for e in after if e["type"].startswith("session_")), None)
        tools = [e for e in after if started and e.get("turn_id") == started.get("turn_id")
                 and e["type"] == "message" and any(b.get("type") == "toolCall" for b in
                 (e["data"]["message"].get("content") or []) if isinstance(b, dict))]
        d = ended["data"]["data"] if ended else {}
        per.append({
            "agent": r["agent"],
            "first_event_s": ts(after[0]["ts"]) - r["sent"] if after else None,
            "turn_s": ts(ended["ts"]) - ts(started["ts"]) if ended else None,
            "e2e_s": ts(ended["ts"]) - r["sent"] if ended else None,
            "ok": bool(ended) and (args.expect == "none" or (d.get("result") or "").strip() == r.get("codeword")),
            "reply": d.get("result"),
            "session_event": session,
            "tool_calls": len(tools),
            "tokens": d.get("tokens", {}),
        })
    replies = (c.get(f"/agents/{p['agent']}") for p in per)
    status = {a["name"]: a for a in (r.json() for r in replies if r.status_code == 200)}
    done = [p for p in per if p["turn_s"] is not None]
    tok = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for p in done:
        for u in p["tokens"].values():
            for k in tok:
                tok[k] += u.get(k, 0)
    summary = {
        "agents": len(per),
        "acks": acks,
        "turn_ended": len(done),
        "reply_ok": sum(p["ok"] for p in per),
        "session_events": {k: sum(p["session_event"] == k for p in per) for k in
                           {p["session_event"] for p in per}},
        "turns_with_tool_calls": sum(p["tool_calls"] > 0 for p in done),
        "first_event_s": {"p50": pct([p["first_event_s"] for p in per if p["first_event_s"] is not None], 50),
                          "p95": pct([p["first_event_s"] for p in per if p["first_event_s"] is not None], 95),
                          "min": min((p["first_event_s"] for p in per if p["first_event_s"] is not None), default=None)},
        "turn_s": {"p50": pct([p["turn_s"] for p in done], 50), "p95": pct([p["turn_s"] for p in done], 95),
                   "max": max((p["turn_s"] for p in done), default=None)},
        "prompt_to_end_s": {"p50": pct([p["e2e_s"] for p in done], 50), "p95": pct([p["e2e_s"] for p in done], 95),
                            "max": max((p["e2e_s"] for p in done), default=None)},
        "tokens_this_turn": tok,
        "controller_cost_usd_total": round(sum(s["cost_usd"] for s in status.values()), 6),
        "claude_estimate_usd_total": round(sum(s["harness_estimate_usd"] or 0 for s in status.values()), 6),
        "mean_turn_s": statistics.mean([p["turn_s"] for p in done]) if done else None,
        "failures": [p for p in per if not p["ok"]][:10],
    }
    print(json.dumps(summary, indent=1))
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"summary": summary, "agents": per}, f, indent=1)


if __name__ == "__main__":
    main()
