"""Prompt every agent of a swarm with at most C agents in flight, then long-poll each agent's
event stream until its turn ends (so C is also the number of concurrent events readers).

    python prompt_load.py <swarm> --concurrency 50 --out out/sent.jsonl

Per agent: read the event cursor, POST the prompt (the body `swarm-harness prompt` sends), then
GET /agents/<name>/events?after=<cursor>&wait=25 until turn_ended. Writes one JSON line per agent:
ack status and latency, server timestamps of the first event / turn_started / turn_ended, the
reply, tokens, worker, and HTTP errors seen. Needs SWARM_HARNESS_URL and SWARM_HARNESS_TOKEN.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import time
from collections import Counter
from datetime import datetime

import httpx

PROMPT = "Your codeword is {cw}. Record it in your file and reply with only the codeword."


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


async def call(c: httpx.AsyncClient, errors: Counter, method: str, path: str, **kw):
    for attempt in range(6):
        try:
            r = await c.request(method, path, **kw)
            if r.status_code < 500 and r.status_code != 429:
                return r
            errors[f"{r.status_code} {path.split('/')[1]}"] += 1
        except httpx.HTTPError as e:
            errors[f"{type(e).__name__} {path.split('/')[1]}"] += 1
        await asyncio.sleep(min(10, 0.5 * 2**attempt))
    raise RuntimeError(f"{method} {path} failed after retries")


async def one(c, name, worker, text_fmt, deadline_s, errors, out):
    rec = {"agent": name, "worker": worker}
    try:
        cursor = 0  # the stream's length: page through it once
        while True:
            r = await call(c, errors, "GET", f"/agents/{name}/events", params={"after": cursor, "limit": 5000})
            evs = r.json()["events"]
            cursor += len(evs)
            if len(evs) < 5000:
                break
        cw = f"{name}-{secrets.token_hex(3)}"
        rec.update(codeword=cw, cursor=cursor)
        rec["sent"] = time.time()
        r = await call(c, errors, "POST", f"/agents/{name}/prompt",
                       json={"text": text_fmt.format(cw=cw), "timeout": None, "reply_to": None})
        rec["ack_s"] = time.time() - rec["sent"]
        rec["ack"] = r.json()
        polls = 0
        started = None
        while time.time() - rec["sent"] < deadline_s:
            r = await call(c, errors, "GET", f"/agents/{name}/events",
                           params={"after": cursor, "wait": 25}, timeout=60)
            polls += 1
            page = r.json()
            for e in page["events"]:
                if "first_event_ts" not in rec:
                    rec["first_event_ts"] = ts(e["ts"])
                    rec["first_event_client"] = time.time()
                    rec["first_event_type"] = e["type"]
                if e["type"] == "turn_started" and started is None:
                    started = e
                    rec["turn_started_ts"] = ts(e["ts"])
                if e["type"] == "turn_ended" and started and e.get("turn_id") == started.get("turn_id"):
                    d = e["data"]["data"]
                    rec.update(turn_ended_ts=ts(e["ts"]), ended_client=time.time(),
                               status=d.get("status"), reply=(d.get("result") or "").strip(),
                               tokens=d.get("tokens", {}))
                if e["type"] in ("prompt_deferred", "turn_interrupted", "error"):
                    rec.setdefault("anomalies", []).append({"type": e["type"], "data": e["data"]})
            cursor = page["next"]
            if "turn_ended_ts" in rec:
                break
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
    a = ap.parse_args()
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
        sem = asyncio.Semaphore(a.concurrency)
        t0 = time.time()

        async def guarded(name, worker, out):
            async with sem:
                return await one(c, name, worker, a.text, a.deadline, errors, out)

        with open(a.out, "w") as out:  # noqa: ASYNC230 - one small local file
            recs = await asyncio.gather(*(guarded(n, w, out) for n, w in agents))
        done = [r for r in recs if "turn_ended_ts" in r]
        print(json.dumps({"agents": len(recs), "turn_ended": len(done), "ok": sum(bool(r.get("ok")) for r in recs),
                          "acks": dict(Counter(r.get("ack", {}).get("status") for r in recs)),
                          "exceptions": sum("exception" in r for r in recs),
                          "wall_s": round(time.time() - t0, 1), "http_errors": dict(errors)}))


if __name__ == "__main__":
    asyncio.run(main())
