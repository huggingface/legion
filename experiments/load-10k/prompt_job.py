# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx>=0.27", "huggingface_hub>=2.2.0"]
# ///
"""One prompter slice of the 10k load test, run as an HF Job (`hf jobs uv run`, cpu-basic).

    prompt_job.py <swarm> --slice K/N --rate 10 --out hf://buckets/<ns>/<b>/<prefix>/<name>.jsonl

The HF Spaces edge rate-limits one client IP at about 20 requests/s (load-1k, check 2), so the
10k agents are prompted from N jobs, each with its own IP, each paced at `--rate` requests/s with
at most `--concurrency` requests in flight. Slice K takes every N-th agent of the sorted list, so
every slice touches every worker, and shuffles it.

Two phases, so reads are one request per agent rather than per-agent long-polling:
  A. send the codeword prompt to every agent of the slice (1 POST each);
  B. `--read-delay` s after the last send, read each agent's whole event stream once
     (1 GET each) and pick out this prompt's turn; agents whose turn has not ended are re-read
     every `--retry-gap` s until `--deadline` s after their send.
One JSON line per agent (ack, server timestamps of the first event / turn_started / turn_ended,
reply, tokens, worker, HTTP errors) is uploaded to `--out`; a summary is printed to the job log.
Needs FLOCK_URL, FLOCK_TOKEN and HF_TOKEN in the environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import secrets
import sys
import time
from collections import Counter
from datetime import datetime

import httpx

PROMPT = "Your codeword is {cw}. Record it in your file and reply with only the codeword."


def ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


class Pacer:
    """Spaces request starts at least 1/rate s apart (a token-interval limiter)."""

    def __init__(self, rate: float) -> None:
        self.gap = 1.0 / rate
        self.next = time.monotonic()
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            self.next = max(self.next, now) + self.gap
            delay = self.next - self.gap - now
        if delay > 0:
            await asyncio.sleep(delay)


async def call(c, pacer, errors, lat, method, path, *, until, **kw):
    """Paced request; retry 5xx, 429 and transport errors until `until` (epoch s)."""
    attempt = 0
    while True:
        await pacer.wait()
        t0 = time.monotonic()
        try:
            r = await c.request(method, path, **kw)
            lat[method].append(time.monotonic() - t0)
            if r.status_code < 500 and r.status_code != 429:
                return r
            errors[f"{r.status_code} {method} {path.split('/')[1]}"] += 1
            pause = min(60, 10 * (attempt + 1)) if r.status_code == 429 else min(10, 0.5 * 2**attempt)
        except httpx.HTTPError as e:
            errors[f"{type(e).__name__} {method} {path.split('/')[1]}"] += 1
            pause = min(10, 0.5 * 2**attempt)
        attempt += 1
        if time.time() + pause > until:
            raise RuntimeError(f"{method} {path} failed after {attempt} attempts")
        await asyncio.sleep(pause)


def scan(rec: dict, events: list[dict]) -> None:
    """Fold an agent's event stream into its record: this prompt's turn is the first turn_started
    at or after the send (server time, 1 s slack)."""
    started = rec.get("_started")
    for e in events:
        if ts(e["ts"]) < rec["sent"] - 1:
            continue
        if "first_event_ts" not in rec:
            rec["first_event_ts"] = ts(e["ts"])
            rec["first_event_type"] = e["type"]
        if e["type"] == "turn_started" and started is None:
            started = e
            rec["_started"] = e
            rec["turn_started_ts"] = ts(e["ts"])
        if started and e.get("turn_id") == started.get("turn_id"):
            if e["type"] == "turn_ended":
                d = e["data"]["data"]
                rec.update(turn_ended_ts=ts(e["ts"]), status=d.get("status"),
                           reply=(d.get("result") or "").strip(), tokens=d.get("tokens", {}))
            elif e["type"] == "message":
                content = e["data"].get("message", {}).get("content") or []
                rec["tool_calls"] = rec.get("tool_calls", 0) + sum(
                    1 for b in content if isinstance(b, dict) and b.get("type") == "toolCall")
        if e["type"] in ("prompt_deferred", "turn_interrupted", "error"):
            rec.setdefault("anomalies", []).append({"type": e["type"], "data": e["data"]})


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("swarm")
    ap.add_argument("--slice", default="0/1", help="K/N: every N-th agent starting at K")
    ap.add_argument("--rate", type=float, default=10.0, help="requests/s from this job")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--out", required=True, help="hf://buckets/... path for the JSONL")
    ap.add_argument("--text", default=PROMPT)
    ap.add_argument("--read-delay", type=float, default=20.0, help="s after the last send")
    ap.add_argument("--retry-gap", type=float, default=15.0)
    ap.add_argument("--deadline", type=float, default=600.0, help="s after send to give up")
    ap.add_argument("--limit", type=int, default=0, help="only the first n agents of the slice")
    return ap.parse_args()


async def run(a: argparse.Namespace) -> dict:
    """Phases A and B; returns the summary with the records under "recs" (in slice order)."""
    k, n = (int(x) for x in a.slice.split("/"))
    pacer, errors, lat = Pacer(a.rate), Counter(), {"GET": [], "POST": []}
    limits = httpx.Limits(max_connections=a.concurrency + 5, max_keepalive_connections=a.concurrency + 5)
    t_start = time.time()
    async with httpx.AsyncClient(base_url=os.environ["FLOCK_URL"], limits=limits,
                                 timeout=httpx.Timeout(30, read=60),
                                 headers={"Authorization": f"Bearer {os.environ['FLOCK_TOKEN']}"}) as c:
        agents, cursor = [], None
        while True:
            r = await call(c, pacer, errors, lat, "GET", f"/swarms/{a.swarm}/agents", until=time.time() + 300,
                           params={"limit": 500, **({"cursor": cursor} if cursor else {})})
            page = r.json()
            agents += [(x["name"], x["worker"]) for x in page["agents"]]
            cursor = page["cursor"]
            if not cursor:
                break
        agents.sort()
        mine = agents[k::n]
        if a.limit:
            mine = mine[: a.limit]
        random.shuffle(mine)
        print(json.dumps({"phase": "listed", "agents": len(agents), "slice": a.slice, "mine": len(mine),
                          "list_s": round(time.time() - t_start, 1)}), flush=True)
        recs = {name: {"agent": name, "worker": worker} for name, worker in mine}
        sem = asyncio.Semaphore(a.concurrency)

        async def send(name: str) -> None:
            rec = recs[name]
            async with sem:
                cw = f"{name}-{secrets.token_hex(3)}"
                rec["codeword"] = cw
                rec["sent"] = time.time()
                try:
                    r = await call(c, pacer, errors, lat, "POST", f"/agents/{name}/prompt",
                                   until=rec["sent"] + a.deadline,
                                   json={"text": a.text.format(cw=cw), "timeout": None, "reply_to": None})
                    rec["ack_s"] = round(time.time() - rec["sent"], 3)
                    rec["ack"] = r.json()
                except Exception as e:  # noqa: BLE001
                    rec["exception"] = repr(e)[:300]

        t_a = time.time()
        await asyncio.gather(*(send(name) for name, _ in mine))
        sent = [r for r in recs.values() if "ack" in r]
        print(json.dumps({"phase": "sent", "n": len(sent), "wall_s": round(time.time() - t_a, 1),
                          "acks": dict(Counter(r["ack"].get("status") for r in sent)),
                          "exceptions": sum("exception" in r for r in recs.values()),
                          "http_errors": dict(errors)}), flush=True)
        await asyncio.sleep(a.read_delay)

        async def read(name: str) -> None:
            rec = recs[name]
            async with sem:
                try:
                    r = await call(c, pacer, errors, lat, "GET", f"/agents/{name}/events",
                                   params={"after": 0, "limit": 5000}, until=rec["sent"] + a.deadline)
                    rec["polls"] = rec.get("polls", 0) + 1
                    scan(rec, r.json()["events"])
                except Exception as e:  # noqa: BLE001
                    rec["exception"] = repr(e)[:300]

        t_b = time.time()
        todo = [r["agent"] for r in sent]
        rounds = 0
        while todo:
            await asyncio.gather(*(read(name) for name in todo))
            rounds += 1
            todo = [x for x in todo if "turn_ended_ts" not in recs[x] and "exception" not in recs[x]
                    and time.time() < recs[x]["sent"] + a.deadline]
            print(json.dumps({"phase": "read", "round": rounds, "pending": len(todo),
                              "wall_s": round(time.time() - t_b, 1)}), flush=True)
            if todo:
                await asyncio.sleep(a.retry_gap)
    for rec in recs.values():
        rec.pop("_started", None)
        rec["ok"] = rec.get("reply") == rec.get("codeword")
    done = [r for r in recs.values() if "turn_ended_ts" in r]

    def pct(xs, p):
        xs = sorted(xs)
        return round(xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))], 3) if xs else None

    summary = {
        "phase": "done", "slice": a.slice, "agents": len(recs), "sent": len(sent), "turn_ended": len(done),
        "ok": sum(r["ok"] for r in recs.values()),
        "exceptions": sum("exception" in r for r in recs.values()),
        "http_errors": dict(errors), "wall_s": round(time.time() - t_start, 1),
        "post_lat_p50_p95_max": [pct(lat["POST"], 50), pct(lat["POST"], 95), pct(lat["POST"], 100)],
        "get_lat_p50_p95_max": [pct(lat["GET"], 50), pct(lat["GET"], 95), pct(lat["GET"], 100)],
        "requests": len(lat["POST"]) + len(lat["GET"]),
        "e2e_p50_p95": [pct([r["turn_ended_ts"] - r["sent"] for r in done], 50),
                        pct([r["turn_ended_ts"] - r["sent"] for r in done], 95)],
    }
    return {**summary, "recs": [recs[name] for name, _ in mine]}


def main() -> None:
    a = parse_args()
    summary = asyncio.run(run(a))
    recs = summary.pop("recs")
    local = f"/tmp/{os.path.basename(a.out)}" if a.out.startswith("hf://") else a.out
    with open(local, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    if a.out.startswith("hf://buckets/"):
        from huggingface_hub import HfApi

        ns, bucket, path = a.out.removeprefix("hf://buckets/").split("/", 2)
        for attempt in range(5):
            try:
                HfApi().batch_bucket_files(f"{ns}/{bucket}", add=[(local, path)])
                summary["uploaded"] = a.out
                break
            except Exception as e:  # noqa: BLE001
                summary["upload_error"] = repr(e)[:200]
                time.sleep(10 * (attempt + 1))
    print(json.dumps(summary), flush=True)
    if not summary.get("uploaded") and a.out.startswith("hf://"):
        sys.exit(1)


if __name__ == "__main__":
    main()
