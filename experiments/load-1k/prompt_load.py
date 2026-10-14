"""Prompt every agent of a swarm with at most C agents in flight, then long-poll each agent's
event stream until its turn ends (so C is also the number of concurrent events readers).

    python prompt_load.py <swarm> --concurrency 50 --out out/sent.jsonl

Per agent: POST the prompt (the body `swarm-harness prompt` sends), wait --first-poll-delay, read the
stream, then GET /agents/<name>/events?after=<next>&wait=25 (with --poll-gap between polls) until
turn_ended. Agents are shuffled so the in-flight prompts spread over all workers. Round 1 of the
1k run (no delay, no gap, name order) cost about 8 requests per agent and tripped the HF Spaces
edge's per-IP rate limit (429) at about 20 requests/s. Writes one JSON line per agent:
ack status and latency, server timestamps of the first event / turn_started / turn_ended, the
reply, tokens, worker, and HTTP errors seen. Needs SWARM_HARNESS_URL and SWARM_HARNESS_TOKEN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import secrets
import time
from collections import Counter
from datetime import datetime

import httpx

PROMPT = "Your codeword is {cw}. Record it in your file and reply with only the codeword."


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


async def call(c: httpx.AsyncClient, errors: Counter, method: str, path: str, *, until: float, **kw):
    """Retry 5xx, 429 and transport errors until `until` (epoch s). The HF Spaces edge answers 429
    per client IP; back off for longer on those."""
    attempt = 0
    while True:
        try:
            r = await c.request(method, path, **kw)
            if r.status_code < 500 and r.status_code != 429:
                return r
            errors[f"{r.status_code} {path.split('/')[1]}"] += 1
            pause = min(60, 10 * (attempt + 1)) if r.status_code == 429 else min(10, 0.5 * 2**attempt)
        except httpx.HTTPError as e:
            errors[f"{type(e).__name__} {path.split('/')[1]}"] += 1
            pause = min(10, 0.5 * 2**attempt)
        attempt += 1
        if time.time() + pause > until:
            raise RuntimeError(f"{method} {path} failed after {attempt} attempts")
        await asyncio.sleep(pause)


async def one(c, name, worker, a, errors, out):
    """Send, wait `first_poll_delay`, read the whole stream once, then long-poll from its end
    (pausing `poll_gap` between polls) until this prompt's turn ends. Events are matched to the
    prompt by server timestamp (>= send - 1 s), so no cursor read is needed before sending."""
    rec = {"agent": name, "worker": worker}
    try:
        cw = a.expected.get(name) or f"{name}-{secrets.token_hex(3)}"
        rec["codeword"] = cw
        rec["sent"] = time.time()
        until = rec["sent"] + a.deadline
        r = await call(c, errors, "POST", f"/agents/{name}/prompt", until=until,
                       json={"text": a.text.format(cw=cw), "timeout": None, "reply_to": None})
        rec["ack_s"] = time.time() - rec["sent"]
        rec["ack"] = r.json()
        await asyncio.sleep(a.first_poll_delay)
        cursor, polls, started = 0, 0, None
        params = {"after": 0, "limit": 5000}
        while time.time() < until:
            r = await call(c, errors, "GET", f"/agents/{name}/events", params=params, timeout=60,
                           until=until)
            polls += 1
            page = r.json()
            for e in page["events"]:
                if ts(e["ts"]) < rec["sent"] - 1:
                    continue
                if "first_event_ts" not in rec:
                    rec["first_event_ts"] = ts(e["ts"])
                    rec["first_event_type"] = e["type"]
                if e["type"] == "turn_started" and started is None:
                    started = e
                    rec["turn_started_ts"] = ts(e["ts"])
                if e["type"] == "turn_ended" and started and e.get("turn_id") == started.get("turn_id"):
                    d = e["data"]["data"]
                    rec.update(turn_ended_ts=ts(e["ts"]), ended_client=time.time(),
                               status=d.get("status"), reply=(d.get("result") or "").strip(),
                               tokens=d.get("tokens", {}))
                if started and e.get("turn_id") == started.get("turn_id") and e["type"] == "message":
                    content = e["data"].get("message", {}).get("content") or []
                    rec["tool_calls"] = rec.get("tool_calls", 0) + sum(
                        1 for b in content if isinstance(b, dict) and b.get("type") == "toolCall")
                if e["type"] in ("prompt_deferred", "turn_interrupted", "error"):
                    rec.setdefault("anomalies", []).append({"type": e["type"], "data": e["data"]})
            cursor = page["next"]
            if "turn_ended_ts" in rec:
                break
            params = {"after": cursor, "wait": 25}
            await asyncio.sleep(a.poll_gap)
        rec["polls"] = polls
        rec["ok"] = rec.get("reply") == cw
    except Exception as e:  # noqa: BLE001
        rec["exception"] = repr(e)[:300]
    out.write(json.dumps(rec) + "\n")
    out.flush()
    return rec


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("swarm")
    ap.add_argument("--concurrency", type=int, default=50)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", help="agent names (default: every agent)")
    ap.add_argument("--deadline", type=float, default=600)
    ap.add_argument("--text", default=PROMPT)
    ap.add_argument("--first-poll-delay", type=float, default=0.0, help="s to wait after the ack")
    ap.add_argument("--poll-gap", type=float, default=0.0, help="s between polls of one agent")
    ap.add_argument("--expect-from", help="sent.jsonl of an earlier run: expect each agent's codeword "
                    "from it as the reply (use with a --text that asks for it)")
    ap.add_argument("--no-shuffle", action="store_true", help="prompt in name order (one worker at a time)")
    a = ap.parse_args()
    a.expected = {}
    if a.expect_from:
        with open(a.expect_from) as f:  # noqa: ASYNC230
            for line in f:
                r = json.loads(line)
                if r.get("ok"):
                    a.expected[r["agent"]] = r["codeword"]
    limits = httpx.Limits(max_connections=a.concurrency + 10, max_keepalive_connections=a.concurrency + 10)
    async with httpx.AsyncClient(base_url=os.environ["SWARM_HARNESS_URL"], limits=limits,
                                 timeout=httpx.Timeout(30, read=60),
                                 headers={"Authorization": f"Bearer {os.environ['SWARM_HARNESS_TOKEN']}"}) as c:
        errors: Counter = Counter()
        agents, cursor = [], None
        while True:
            r = await c.get(f"/swarms/{a.swarm}/agents", params={"limit": 500, **({"cursor": cursor} if cursor else {})})
            r.raise_for_status()
            page = r.json()
            agents += [(x["name"], x["worker"]) for x in page["agents"]]
            cursor = page["cursor"]
            if not cursor:
                break
        if a.only:
            agents = [x for x in agents if x[0] in set(a.only)]
        if not a.no_shuffle:  # spread the in-flight prompts over all workers
            random.shuffle(agents)
        sem = asyncio.Semaphore(a.concurrency)
        t0 = time.time()

        async def guarded(name, worker, out):
            async with sem:
                return await one(c, name, worker, a, errors, out)

        with open(a.out, "w") as out:  # noqa: ASYNC230 - one small local file
            recs = await asyncio.gather(*(guarded(n, w, out) for n, w in agents))
        done = [r for r in recs if "turn_ended_ts" in r]
        print(json.dumps({"agents": len(recs), "turn_ended": len(done), "ok": sum(bool(r.get("ok")) for r in recs),
                          "acks": dict(Counter(r.get("ack", {}).get("status") for r in recs)),
                          "exceptions": sum("exception" in r for r in recs),
                          "wall_s": round(time.time() - t0, 1), "http_errors": dict(errors)}))


if __name__ == "__main__":
    asyncio.run(main())
