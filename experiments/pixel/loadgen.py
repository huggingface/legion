#!/usr/bin/env python3
"""Scripted painters against a running canvas: N workers, each at R calls/s.

Each worker loops claim -> paint (the claimed pixels, with the target colours the claim
reveals), pacing every HTTP call to R/s on an open-loop schedule. Prints achieved req/s and
latency percentiles per endpoint. Stdlib only (one keep-alive connection per worker).

    python loadgen.py --url http://127.0.0.1:8765 --workers 100 --rate 1 --duration 60
"""

import argparse
import http.client
import json
import os
import threading
import time
import urllib.parse


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


class Worker(threading.Thread):
    def __init__(self, i, args, stop_at, token):
        super().__init__(daemon=True)
        self.i, self.args, self.stop_at = i, args, stop_at
        u = urllib.parse.urlparse(args.url)
        self.host, self.port = u.hostname, u.port or 80
        self.h = {"Content-Type": "application/json", "X-Flock-Agent": f"load-{i:05d}"}
        if token:
            self.h["Authorization"] = f"Bearer {token}"
        self.lat: dict[str, list[float]] = {"claim": [], "paint": []}
        self.codes: dict[int, int] = {}
        self.errors = 0
        self.useful = 0

    def _post(self, conn, path, body):
        t0 = time.perf_counter()
        conn.request("POST", path, json.dumps(body), self.h)
        r = conn.getresponse()
        data = r.read()
        self.lat[path.strip("/")].append(time.perf_counter() - t0)
        self.codes[r.status] = self.codes.get(r.status, 0) + 1
        return r.status, data

    def run(self):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        period = 1.0 / self.args.rate
        nxt = time.monotonic() + (self.i % 100) / 100 * period  # spread start phases
        pending = []
        while True:
            now = time.monotonic()
            if now >= self.stop_at:
                break
            if nxt > now:
                time.sleep(nxt - now)
            nxt += period
            try:
                if not pending:
                    st, data = self._post(conn, "/claim", {"n": self.args.n})
                    if st == 200:
                        pending = json.loads(data)["pixels"]
                else:
                    st, data = self._post(conn, "/paint", {"pixels": pending})
                    pending = []
                    if st == 200:
                        self.useful += json.loads(data)["counts"].get("useful", 0)
            except (OSError, http.client.HTTPException):
                self.errors += 1
                conn.close()
                conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        conn.close()


def cpu_seconds(pid: int) -> float:
    """utime + stime of a process (Linux /proc), in seconds."""
    with open(f"/proc/{pid}/stat") as fh:
        f = fh.read().rsplit(")", 1)[1].split()
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


class Poller(threading.Thread):
    """Dashboard-like reads every --poll-every s (progress page / driver), timed separately."""

    def __init__(self, url, paths, every, stop_at):
        super().__init__(daemon=True)
        u = urllib.parse.urlparse(url)
        self.host, self.port, self.paths, self.every, self.stop_at = (
            u.hostname,
            u.port or 80,
            paths,
            every,
            stop_at,
        )
        self.lat: dict[str, list[float]] = {p: [] for p in paths}

    def run(self):
        conn = http.client.HTTPConnection(self.host, self.port, timeout=30)
        while time.monotonic() < self.stop_at:
            for p in self.paths:
                t0 = time.perf_counter()
                conn.request("GET", p)
                conn.getresponse().read()
                self.lat[p].append(time.perf_counter() - t0)
            time.sleep(self.every)
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--url", default=os.environ.get("CANVAS_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--workers", type=int, default=100)
    ap.add_argument("--rate", type=float, default=1.0, help="HTTP calls/s per worker")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("-n", type=int, default=None, help="pixels per claim (server default)")
    ap.add_argument(
        "--poll", default="", help="comma list of GET paths polled every --poll-every s"
    )
    ap.add_argument("--poll-every", type=float, default=10.0)
    ap.add_argument("--pid", type=int, default=0, help="server pid: report its CPU use")
    a = ap.parse_args()
    token = os.environ.get("CANVAS_TOKEN", "")
    stop_at = time.monotonic() + a.duration
    ws = [Worker(i, a, stop_at, token) for i in range(a.workers)]
    t0 = time.monotonic()
    cpu0 = cpu_seconds(a.pid) if a.pid else None
    poller = Poller(a.url, [p for p in a.poll.split(",") if p], a.poll_every, stop_at)
    if poller.paths:
        poller.start()
    for w in ws:
        w.start()
    for w in ws:
        w.join()
    el = time.monotonic() - t0
    if poller.paths:
        poller.join()
    lat = {k: [x for w in ws for x in w.lat[k]] for k in ("claim", "paint")}
    allx = lat["claim"] + lat["paint"]
    codes: dict[int, int] = {}
    for w in ws:
        for c, n in w.codes.items():
            codes[c] = codes.get(c, 0) + n
    print(
        f"workers={a.workers} rate={a.rate}/s each duration={el:.1f}s target={a.workers * a.rate:.0f} req/s"
    )
    print(
        f"requests={len(allx)} achieved={len(allx) / el:.1f} req/s  status={codes} "
        f"conn_errors={sum(w.errors for w in ws)}"
    )
    for k, xs in [("all", allx), *lat.items()]:
        print(
            f"  {k:5s} n={len(xs):6d} p50={pct(xs, 50) * 1e3:6.1f}ms "
            f"p95={pct(xs, 95) * 1e3:6.1f}ms p99={pct(xs, 99) * 1e3:6.1f}ms "
            f"max={max(xs, default=0) * 1e3:6.1f}ms"
        )
    for p, xs in poller.lat.items():
        print(
            f"  poll {p}: n={len(xs)} p50={pct(xs, 50) * 1e3:.1f}ms "
            f"max={max(xs, default=0) * 1e3:.1f}ms"
        )
    if cpu0 is not None:
        used = cpu_seconds(a.pid) - cpu0
        print(f"server cpu: {used:.1f} s over {el:.1f} s = {100 * used / el:.0f}% of one core")
    print(
        f"useful pixels painted={sum(w.useful for w in ws)} "
        f"({sum(w.useful for w in ws) / el:.0f} px/s)"
    )
    try:
        conn = http.client.HTTPConnection(ws[0].host, ws[0].port, timeout=30)
        conn.request("GET", "/score?agents=false")
        s = json.loads(conn.getresponse().read())
        print(
            f"score: exact_match={s['exact_match_pct']}% coverage={s['coverage_pct']}% "
            f"psnr={s['psnr']} paints={s['paints']}"
        )
    except Exception as e:  # noqa: BLE001
        print(f"score: unavailable ({e})")


if __name__ == "__main__":
    main()
