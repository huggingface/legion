"""Controller load test: a real controller process (fake jobs) and simulated workers that send
what flock.worker sends (command long-poll, a heartbeat every 30 s, an event batch every 3 s, the
same timeouts), while a probe times `/healthz` and an operator call (`GET /swarms/<s>`). Agents
loop on their own (v8's `every: turn_end`) and get board notifications (`comms_*` events), the
traffic that dominates v8's event streams at 1,000 agents.

    uv run python scripts/ctrl_load.py --workers 108 --agents-per-worker 12 --duration 300

The spec is experiments/v8/spec.yaml (prompts inlined), without the ramp, with every URL pointed
at a closed local port, written to the store before the controller starts (an apply would join
the board). Prints latency per window and the controller's own route stats. Before /
after numbers: run it on two commits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import multiprocessing as mp
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

TOKEN = "load-token"
H = {"Authorization": f"Bearer {TOKEN}"}
ROOT = Path(__file__).resolve().parent.parent
COMMS = ("comms_channel_message", "comms_thread_reply", "comms_mention")
CLOSED = "127.0.0.1:9"  # nothing listens: world, board and model probes fail fast


def v8_spec(workers: int, per_worker: int):
    os.environ.update(WORLD_HOST=CLOSED.split(":")[0], CHATBOT_BUCKET="x", SLURM_PARTITION="p",
                      OPENAI_BASE_URLS=f"http://{CLOSED}/v1",
                      CHATBOT_BOARD_URL=f"http://{CLOSED}")
    from flock.ops import read_spec

    spec = read_spec(str(ROOT / "experiments/v8/spec.yaml")).model_dump(mode="json")
    names = sorted(spec["types"])
    for i, name in enumerate(names):
        ts = spec["types"][name]
        ts["replicas_ramp"] = None
        ts["agents_per_worker"] = per_worker
        ts["replicas"] = per_worker * (workers // len(names) + (i < workers % len(names)))
    spec["limits"] = {"alive": workers * per_worker}
    spec["comms"]["channel_id"] = 1  # as after a real apply: no board call
    from flock.models import SwarmSpec

    return SwarmSpec.model_validate(spec)


def now_iso(dt: float = 0.0) -> str:
    return (datetime.now(UTC) + timedelta(seconds=dt)).isoformat()


class SimWorker:
    """One worker: each agent runs turns of `turn_s` s back to back (a prompt starts one at
    once), with `rate` message events/s in a turn and `comms_rate` board notifications/s."""

    def __init__(self, wid: str, url: str, a: argparse.Namespace, stats: dict):
        self.wid, self.a, self.stats = wid, a, stats
        self.http = httpx.AsyncClient(base_url=url, headers=H, timeout=httpx.Timeout(20, read=60),
                                      limits=httpx.Limits(max_connections=3))
        self.agents: dict[str, dict] = {}
        self.outbox: list[dict] = []
        self.rev: str | None = None

    async def call(self, kind: str, method: str, path: str, **kw) -> httpx.Response | None:
        t0 = time.monotonic()
        try:
            r = await self.http.request(method, path, **kw)
            key = f"{kind}_{r.status_code // 100}xx"
            self.stats[key] = self.stats.get(key, 0) + 1
            return r if r.status_code < 300 else None
        except httpx.HTTPError as e:
            key = f"{kind}_{type(e).__name__}"
            self.stats[key] = self.stats.get(key, 0) + 1
            return None
        finally:
            if kind != "poll":
                self.stats.setdefault(f"{kind}_s", []).append(time.monotonic() - t0)

    def ev(self, name: str, typ: str, data: dict | None = None) -> None:
        ag = self.agents[name]
        self.outbox.append({"type": typ, "ts": now_iso(), "agent": name, "turn_id": ag["turn"],
                            "data": data or {}})

    async def poll(self) -> None:
        while True:
            t0 = time.monotonic()
            r = await self.call("poll", "GET", f"/workers/{self.wid}/commands",
                                params={"wait": 25}, timeout=httpx.Timeout(20, read=45))
            for c in (r.json().get("commands", []) if r is not None and r.content else []):
                if c["kind"] == "assign":  # v8's loop is `every: turn_end`, run by the worker
                    self.agents[c["agent"]] = {"state": "idle", "turn": None, "turns": 0,
                                               "next": time.monotonic() + random.uniform(0, 30),
                                               "until": 0.0, "tok": 0}
                elif c["kind"] == "unassign":
                    self.agents.pop(c["agent"], None)
                elif c["kind"] == "prompt" and c["agent"] in self.agents:
                    self.agents[c["agent"]]["next"] = 0.0
            await asyncio.sleep(max(0.0, 1.0 - (time.monotonic() - t0)) if r else 2.0)

    async def heartbeat(self) -> None:
        await asyncio.sleep(random.random() * 30)
        while True:
            hb = {"worker": self.wid, "deadline": now_iso(54 * 3600), "spec_rev": self.rev,
                  "agents": [{"name": n, "state": ag["state"], "turn_id": ag["turn"],
                              "turns": ag["turns"],
                              "tokens": {"m": {"input": ag["tok"], "output": ag["tok"] // 10}}}
                             for n, ag in self.agents.items()]}
            r = await self.call("hb", "POST", f"/workers/{self.wid}/heartbeat", json=hb)
            if r is not None:
                self.rev = r.json().get("spec_rev", self.rev)
            await asyncio.sleep(30 * random.uniform(0.9, 1.1))

    async def flush(self) -> None:
        pad = "x" * self.a.event_bytes
        while True:
            await asyncio.sleep(3.0)
            t = time.monotonic()
            for n, ag in list(self.agents.items()):
                for _ in range(sum(random.random() < self.a.comms_rate for _ in range(3))):
                    typ = random.choices(COMMS, (47, 50, 3))[0]  # v8's mix at 1,000 agents
                    self.ev(n, typ, {"msg_id": random.getrandbits(30),
                                     "thread": random.getrandbits(20),
                                     "from": "someone", "channel": "general", "channel_id": 1,
                                     "identity_type": "agent"})
                if ag["state"] != "working":
                    if t < ag["next"]:
                        continue
                    ag.update(state="working", turn=f"t{random.getrandbits(32):x}",
                              until=t + random.uniform(*self.a.turn_s))
                    self.ev(n, "turn_started")
                for _ in range(sum(random.random() < self.a.rate for _ in range(3))):
                    ag["tok"] += 500
                    self.ev(n, "message", {"role": "toolResult", "content": pad})
                if t >= ag["until"]:
                    self.ev(n, "turn_ended", {"turns": ag["turns"] + 1})
                    ag.update(state="idle", turn=None, turns=ag["turns"] + 1,
                              next=t + random.uniform(1, 10))
            if self.outbox:
                batch, self.outbox = self.outbox[:500], self.outbox[500:]
                r = await self.call("events", "POST", f"/workers/{self.wid}/events",
                                    json={"worker": self.wid, "events": batch})
                if r is None:  # as the worker: kept and sent again
                    self.outbox[:0] = batch


def run_workers(wids: list[str], url: str, a: argparse.Namespace, q) -> None:
    stats: dict = {}

    async def main() -> None:
        sims = [SimWorker(w, url, a, stats) for w in wids]
        tasks = [asyncio.create_task(f()) for s in sims for f in (s.poll, s.heartbeat, s.flush)]
        await asyncio.sleep(a.duration)
        for t in tasks:
            t.cancel()

    asyncio.run(main())
    q.put(stats)


def pct(xs: list[float], p: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=108)
    ap.add_argument("--agents-per-worker", type=int, default=12)
    ap.add_argument("--duration", type=float, default=300.0)
    # defaults: v8's hourly rollups at 1,000 agents (2026-11-10 03Z): ~0.03 message/s and
    # ~0.07 comms events/s per agent; 0.2 comms/s is the busiest hour seen (resends included)
    ap.add_argument("--rate", type=float, default=0.05, help="message events/s per agent in a turn")
    ap.add_argument("--comms-rate", type=float, default=0.1, help="comms events/s per agent")
    ap.add_argument("--event-bytes", type=int, default=1500)
    ap.add_argument("--turn-s", type=float, nargs=2, default=(60.0, 180.0))
    ap.add_argument("--procs", type=int, default=4, help="client processes")
    ap.add_argument("--port", type=int, default=7979)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--env", action="append", default=[], help="K=V for the controller")
    a = ap.parse_args()
    url = f"http://127.0.0.1:{a.port}"
    data = tempfile.mkdtemp(prefix="ctrl-load-")
    env = {**os.environ, "FLOCK_TOKEN": TOKEN, "FLOCK_FAKE_JOBS": "1", "FLOCK_DATA": data,
           "PORT": str(a.port), "FLOCK_STATE_URL": "", "FLOCK_URL": url,
           "OPENAI_API_KEY": "x", "CHATBOT_BOARD_INVITE": "x", "CHATBOT_WORLD_TOKEN": "x",
           "HF_HUB_OFFLINE": "1",
           **dict(kv.split("=", 1) for kv in a.env)}
    from flock.controller.store import Store

    st = Store(data)  # stored directly: an apply would join the (absent) board
    st.put_swarm(v8_spec(a.workers, a.agents_per_worker), time.time())
    st.close()
    log = open(Path(data) / "controller.log", "w")  # noqa: SIM115 - the process's whole life
    ctl = subprocess.Popen([sys.executable, "-m", "flock.controller.main"], env=env,
                           stdout=log, stderr=subprocess.STDOUT)
    try:
        c = httpx.Client(base_url=url, headers=H, timeout=120)
        for _ in range(100):
            try:
                c.get("/healthz")
                break
            except httpx.TransportError:
                time.sleep(0.2)
        while len(wids := [w["id"] for w in c.get("/workers").json()
                           if w.get("job_id")]) < a.workers:
            time.sleep(1)
        print(f"{len(wids)} workers, {a.workers * a.agents_per_worker} agents; data {data}",
              flush=True)
        q = mp.get_context("spawn").Queue()
        procs = [mp.get_context("spawn").Process(target=run_workers,
                                                 args=(wids[i::a.procs], url, a, q))
                 for i in range(a.procs)]
        for p in procs:
            p.start()
        probe = httpx.Client(base_url=url, headers=H, timeout=httpx.Timeout(120))
        hz: list[float] = []
        op: list[float] = []
        t_end = time.monotonic() + a.duration
        print("window_end_s  healthz_p50  healthz_max  op_p50  op_max  in_flight  rss_mb  cpu_%",
              flush=True)
        t0 = time.monotonic()
        last = probe.get("/healthz").json()["process"]
        while time.monotonic() < t_end:
            wh, wo = [], []
            w_end = min(t_end, time.monotonic() + a.window)
            while time.monotonic() < w_end:
                s = time.monotonic()
                try:
                    h = probe.get("/healthz").json()
                except httpx.HTTPError:
                    h = None
                wh.append(time.monotonic() - s)
                s = time.monotonic()
                try:
                    probe.get("/swarms/v8")
                except httpx.HTTPError:
                    pass
                wo.append(time.monotonic() - s)
                time.sleep(max(0.0, 2.0 - (time.monotonic() - s)))
            hz += wh
            op += wo
            proc = (h or {}).get("process", last)
            cpu = 100 * (proc["cpu_s"] - last["cpu_s"]) / max(1e-9, proc["time"] - last["time"])
            last = proc
            print(f"{time.monotonic() - t0:12.0f}  {pct(wh, .5):11.3f}  {max(wh):11.3f}  "
                  f"{pct(wo, .5):6.3f}  {max(wo):6.3f}  "
                  f"{(h or {}).get('requests', {}).get('in_flight', '?'):>9}  "
                  f"{proc['rss_mb']:6.0f}  {cpu:5.0f}", flush=True)
        stats: dict = {}
        for _ in procs:
            for k, v in q.get().items():
                stats[k] = stats.get(k, 0 if isinstance(v, int) else []) + v
        for p in procs:
            p.join()
        h = probe.get("/healthz").json()
        print(f"\nhealthz  p50 {pct(hz, .5):.3f}s  p95 {pct(hz, .95):.3f}s  max {max(hz):.3f}s")
        print(f"operator p50 {pct(op, .5):.3f}s  p95 {pct(op, .95):.3f}s  max {max(op):.3f}s")
        for k in ("hb_s", "events_s"):
            xs = stats.pop(k, [])
            if xs:
                print(f"worker {k[:-2]:6}  n {len(xs)}  p50 {pct(xs, .5):.3f}s  "
                      f"p95 {pct(xs, .95):.3f}s  max {max(xs):.3f}s")
        print("worker calls:", json.dumps(dict(sorted(stats.items()))))
        print("controller routes:")
        for route, v in h["requests"]["routes"].items():
            print(f"  {route:45} n {v['n']:6}  mean {v['mean_s']:.3f}s  max {v['max_s']:.3f}s")
        log.flush()
        text = (Path(data) / "controller.log").read_text()
        print(f"unresponsive cancels: {text.count('unresponsive')}, "
              f"crash lines: {text.count(' crash ')}")
        print(f"peak in_flight {h['requests']['peak_in_flight']}, "
              f"rss {h['process']['rss_mb']} MB")
    finally:
        ctl.terminate()
        ctl.wait(30)


if __name__ == "__main__":
    main()
