"""Load-sizing smoke for the self-hosted endpoint (load-smoke.md): run the real phase-1 task
(worlds/chatbot2, a local in-memory board, `flock.controller.localrun` in local mode) against the endpoint with
C agents for a few minutes per level, measure model calls/s, p50/p90 latency and errors through
a recording proxy, and recommend an agent count (the knee x 0.8).

    SELFHOSTED_BASE_URL=https://.../v1 SELFHOSTED_API_KEY=... MODEL_DS=... [MODEL_GLM=...] \\
      python experiments/chatbot-v2/selfhosted/load_smoke.py [--levels 8,16,32,64] [--minutes 5]

It sends real requests to the endpoint (no other network: the board is in memory, the world is
local). Run it from the repo root with flock installed. Output: a table on stdout and
<out>/load-smoke.md + results.json."""

from __future__ import annotations

import argparse
import itertools
import json
import os
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def proxy(upstream: str, log: Path) -> ThreadingHTTPServer:
    """A pass-through proxy on 127.0.0.1 that records every call: start, end, status, tokens."""
    import httpx

    client, lock = httpx.Client(timeout=600, limits=httpx.Limits(max_connections=512)), \
        threading.Lock()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a) -> None:  # quiet
            pass

        def _go(self, method: str) -> None:
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            head = {k: v for k, v in self.headers.items()
                    if k.lower() in ("authorization", "content-type", "accept")}
            t0, rec = time.time(), {"path": self.path, "method": method}
            try:
                r = client.request(method, upstream.rstrip("/") + self.path.removeprefix("/v1"),
                                   content=body, headers=head)
                status, data, ctype = r.status_code, r.content, r.headers.get("content-type", "")
                try:
                    usage = json.loads(data).get("usage") or {}
                except (ValueError, AttributeError):
                    usage = {}
                rec |= {"status": status, "in": usage.get("prompt_tokens"),
                        "out": usage.get("completion_tokens")}
            except httpx.HTTPError as e:
                status, ctype = 502, "application/json"
                data = json.dumps({"error": repr(e)}).encode()
                rec |= {"status": 0, "error": repr(e)[:200]}
            rec |= {"t0": t0, "t1": time.time()}
            with lock, open(log, "a") as f:
                f.write(json.dumps(rec) + "\n")
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:
            self._go("POST")

        def do_GET(self) -> None:
            self._go("GET")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def summarize(recs: list[dict], t0: float, t1: float) -> dict:
    """Model calls that started in [t0, t1]: calls/s, p50/p90 latency (ms), errors, out tok/s."""
    calls = sorted((r for r in recs if r.get("path", "").endswith("/chat/completions")
                    and t0 <= r["t0"] <= t1), key=lambda r: r["t1"] - r["t0"])
    lat = [1000 * (r["t1"] - r["t0"]) for r in calls]

    def pct(q: float) -> float:
        return round(lat[min(len(lat) - 1, int(q * len(lat)))]) if lat else 0

    span = max(t1 - t0, 1e-9)
    return {"calls": len(calls), "calls_s": round(len(calls) / span, 2), "p50_ms": pct(0.5),
            "p90_ms": pct(0.9), "errors": sum(r["status"] != 200 for r in calls),
            "out_tok_s": round(sum(r.get("out") or 0 for r in calls) / span)}


def knee(rows: list[dict]) -> int | None:
    """probe-gpu-knee.md's rule: the first doubling of C whose calls/s gain is under 15% while
    p50 rises at least 40% (or that brings errors); the knee is the level before it. None: no
    knee within the levels (the endpoint kept up)."""
    rows = sorted(rows, key=lambda r: r["agents"])
    for a, b in itertools.pairwise(rows):
        gain = b["calls_s"] / a["calls_s"] - 1 if a["calls_s"] else 0
        slower = b["p50_ms"] / a["p50_ms"] - 1 if a["p50_ms"] else 0
        if (gain < 0.15 and slower >= 0.40) or (b["errors"] > max(2, 0.01 * b["calls"])):
            return a["agents"]
    return None


def level(c: int, minutes: float, warmup: float, work: Path, per_team: int) -> dict:
    """One level: C agents (C / per_team teams) on the phase-1 task for `minutes`."""
    sys.path.insert(0, str(ROOT / "tests"))
    import fake_msgboard

    exp, log = work / f"c{c}", work / f"c{c}-calls.jsonl"
    shutil.copytree(ROOT / "experiments" / "chatbot-v2", exp,
                    ignore=shutil.ignore_patterns(".flock*", "*.local.yaml"))
    srv = proxy(os.environ["SELFHOSTED_BASE_URL"], log)
    teams = max(2, -(-c // per_team))
    env = {**os.environ, "SELFHOSTED_BASE_URL": f"http://127.0.0.1:{srv.server_port}/v1"}
    env.pop("BASE_URL_DS", None)
    env.pop("BASE_URL_GLM", None)
    subprocess.run([sys.executable, str(HERE / "render.py"), "--into", str(exp), "--teams",
                    str(teams), "--hours", str(minutes / 60)], env=env, check=True,
                   capture_output=True)
    board, bsrv, burl = fake_msgboard.serve(fake_msgboard.Board(channels={1: "load"}))
    for f in ("spec.yaml",):
        d = yaml.safe_load((exp / f).read_text())
        d["comms"].update(url=burl, channel="load", invite_env="LOAD_INVITE")
        d["swarm"] = f"load-c{c}"
        for t in d["types"].values():
            if t.get("loop"):
                t["loop"].pop("final", None)
        (exp / f).write_text(yaml.safe_dump(d, sort_keys=False))
    env |= {"LOAD_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": secrets.token_hex(16),
            "CHATBOT_WS": str(exp / "ws"), "CHATBOT_CONFIG": str(exp / "world.yaml")}
    env = {k: v for k, v in env.items() if not k.startswith("FLOCK_")}
    out = open(exp / "run.out", "w")  # noqa: SIM115 (closed below)
    t_start = time.time()
    proc = subprocess.Popen([sys.executable, "-m", "flock.controller.localrun", "spec.yaml", "--until",
                             "idle"], cwd=exp, env=env, stdout=out, stderr=subprocess.STDOUT)
    try:
        proc.wait(timeout=60 * minutes + 240)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=60)
    finally:
        out.close()
        srv.shutdown()
        bsrv.shutdown()
    recs = [json.loads(x) for x in log.read_text().splitlines()] if log.is_file() else []
    first = min((r["t0"] for r in recs), default=t_start)
    s = summarize(recs, first + warmup, first + 60 * minutes)
    return {"agents": c, "teams": teams, **s, "exit": proc.returncode}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--levels", default="8,16,32,64")
    ap.add_argument("--minutes", type=float, default=5)
    ap.add_argument("--warmup", type=float, default=60, help="seconds left out at the start")
    ap.add_argument("--out", type=Path, default=Path("load-smoke-out"))
    a = ap.parse_args()
    for v in ("SELFHOSTED_BASE_URL", "SELFHOSTED_API_KEY", "MODEL_DS"):
        if not os.environ.get(v):
            sys.exit(f"{v} is not set")
    per_team = sum(yaml.safe_load((HERE / "scale.yaml").read_text())["per_team"].values())
    a.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="load-smoke-") as tmp:
        for c in (int(x) for x in a.levels.split(",")):
            rows.append(level(c, a.minutes, a.warmup, Path(tmp), per_team))
            print(json.dumps(rows[-1]), flush=True)
            for p in Path(tmp).glob(f"c{c}-calls.jsonl"):
                shutil.copy(p, a.out / p.name)
    k = knee(rows)
    best = max(rows, key=lambda r: r["calls_s"])
    rec = int(0.8 * (k if k else best["agents"]))
    table = ["| agents | teams | calls/s | p50 ms | p90 ms | errors | out tok/s |",
             "|---|---|---|---|---|---|---|"] + [
        f"| {r['agents']} | {r['teams']} | {r['calls_s']} | {r['p50_ms']} | {r['p90_ms']} | "
        f"{r['errors']} | {r['out_tok_s']} |" for r in rows]
    note = (f"Knee: {k} agents. Recommended agent count: {rec} (knee x 0.8)." if k else
            f"No knee up to {best['agents']} agents: the endpoint kept up. Recommended: at least "
            f"{rec}; repeat with higher levels (--levels 64,128,256) to find the knee.")
    text = "\n".join(["# Load smoke", "", f"{a.minutes:g} min per level, the first "
                      f"{a.warmup:g} s left out; model: {os.environ['MODEL_DS']}"
                      + (f" + {os.environ['MODEL_GLM']}" if os.environ.get("MODEL_GLM") else ""),
                      "", *table, "", note, ""])
    (a.out / "load-smoke.md").write_text(text)
    (a.out / "results.json").write_text(json.dumps({"rows": rows, "knee": k,
                                                    "recommended_agents": rec}, indent=1))
    print(text)


if __name__ == "__main__":
    main()
