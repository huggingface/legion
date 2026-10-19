#!/usr/bin/env python3
"""Drive the pixel swarm: keep painters prompted until the canvas is done.

Every --interval seconds:
  1. GET <canvas>/score; stop when the work set is fully painted (coverage 100%).
  2. Prompt painters that have nothing queued. Bulk path: `flock prompt --swarm S --type T
     --state idle "<step>"` when that CLI exists (PR "prompt-type"; detected at start).
     Until then: list agents over the controller API (what `flock agents` reads) and POST one
     prompt per agent (what `flock prompt` sends).
  3. Print one progress line and append it to progress.jsonl; every prompt sent goes to
     prompts.jsonl (agent, prompt_id, ts) so flock turns join the canvas log by agent + time.

With `tool_choice: required` each prompt is one bash step (claim 10 | paint), so the driver is
the loop. --max-steps caps steps per agent (0 = no cap).

Env: FLOCK_URL, FLOCK_TOKEN (controller), CANVAS_URL (canvas, reads need no token).
Stdlib only. --dry-run prints what it would send.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

BOOT = (
    'python3 -c "import os,urllib.request as u;'
    "u.urlretrieve(os.environ['CANVAS_URL']+'/cli/canvas','canvas')\" "
    "&& chmod +x canvas && ./canvas status"
)
STEP = "./canvas claim 10 --json | ./canvas paint - --json"
# The bulk path cannot tell new agents from old ones, so it installs when missing.
STEP_SELF_INSTALL = f"test -x canvas || {BOOT}; {STEP}"


def prompt_text(cmd: str) -> str:
    return f"Next step. Run exactly this one command:\n{cmd}"


def http(
    method: str, url: str, token: str | None = None, body: dict | None = None, timeout: float = 30
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


class Controller:
    def __init__(self, url: str, token: str, swarm: str, type_: str):
        self.url, self.token, self.swarm, self.type = url.rstrip("/"), token, swarm, type_

    def agents(self) -> list[dict]:
        out, cursor = [], None
        while True:
            q = {"type": self.type, **({"cursor": cursor} if cursor else {})}
            page = http(
                "GET",
                f"{self.url}/swarms/{self.swarm}/agents?{urllib.parse.urlencode(q)}",
                self.token,
            )
            out += page["agents"]
            cursor = page.get("cursor")
            if not cursor:
                return out

    def prompt(self, agent: str, text: str) -> dict:
        return http(
            "POST",
            f"{self.url}/agents/{agent}/prompt",
            self.token,
            {"text": text, "timeout": None, "reply_to": None},
        )

    def status(self) -> dict:
        return http("GET", f"{self.url}/swarms/{self.swarm}", self.token)["status"]


def bulk_cli_available() -> bool:
    if not shutil.which("flock"):
        return False
    r = subprocess.run(["flock", "prompt", "--help"], capture_output=True, text=True, check=False)
    return "--swarm" in r.stdout and "--state" in r.stdout


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt idle painters until the canvas is done.")
    ap.add_argument("--swarm", default="pixel-100")
    ap.add_argument("--type", default="painter")
    ap.add_argument("--canvas-url", default=os.environ.get("CANVAS_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--flock-url", default=os.environ.get("FLOCK_URL", "http://localhost:7860"))
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=0, help="per agent; 0 = no cap")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this; 0 = no cap")
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.80, help="l4x1 = 0.80")
    ap.add_argument("--out", default="progress.jsonl")
    ap.add_argument("--prompts-log", default="prompts.jsonl")
    ap.add_argument("--no-bulk", action="store_true", help="always prompt per agent")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    ctl = Controller(a.flock_url, os.environ.get("FLOCK_TOKEN", ""), a.swarm, a.type)
    canvas = a.canvas_url.rstrip("/")
    bulk = not a.no_bulk and bulk_cli_available()
    print(
        f"driver: swarm={a.swarm} type={a.type} canvas={canvas} "
        f"mode={'bulk flock prompt' if bulk else 'per-agent prompts'}",
        flush=True,
    )
    steps: dict[str, int] = {}
    booted: set[str] = set()
    t0 = time.time()
    last_seq, last_t = None, None
    with open(a.out, "a") as prog, open(a.prompts_log, "a") as plog:
        while True:
            now = time.time()
            try:
                score = http("GET", f"{canvas}/score")
                status = http("GET", f"{canvas}/status")
            except (urllib.error.URLError, OSError) as e:
                print(f"canvas unreachable: {e}; retrying", flush=True)
                time.sleep(a.interval)
                continue
            done = score["work_done"] >= score["work_total"]
            sent = 0
            idle = working = 0
            cost = None
            try:
                agents = ctl.agents()
                cost = ctl.status().get("cost_usd")
            except (urllib.error.URLError, OSError) as e:
                print(f"controller error: {e}", flush=True)
                agents = []
            for ag in agents:
                idle += ag["state"] == "idle"
                working += ag["state"] == "working"
            if not done and agents:
                if bulk:
                    cmd = [
                        "flock",
                        "prompt",
                        "--swarm",
                        a.swarm,
                        "--type",
                        a.type,
                        "--state",
                        "idle",
                        prompt_text(STEP_SELF_INSTALL),
                    ]
                    if a.dry_run:
                        print("would run:", " ".join(cmd[:8]), "...")
                    else:
                        subprocess.run(cmd, check=False, capture_output=True)
                    sent = idle
                else:
                    for ag in agents:
                        name = ag["name"]
                        if ag["state"] not in ("idle", "working") or ag.get("pending", 0) > 0:
                            continue
                        if a.max_steps and steps.get(name, 0) >= a.max_steps:
                            continue
                        # Never booted, or stepped 3+ times without ever reaching the canvas
                        # (the install failed): send the install step.
                        seen = name in score.get("agents", {})
                        first = name not in booted or (not seen and steps.get(name, 0) >= 3)
                        if first:
                            steps[name] = 0
                        text = prompt_text(BOOT if first else STEP)
                        if a.dry_run:
                            ack = {"prompt_id": "dry-run"}
                        else:
                            try:
                                ack = ctl.prompt(name, text)
                            except (urllib.error.URLError, OSError) as e:
                                print(f"prompt {name} failed: {e}", flush=True)
                                continue
                        booted.add(name)
                        if not first:
                            steps[name] = steps.get(name, 0) + 1
                        sent += 1
                        plog.write(
                            json.dumps(
                                {
                                    "ts": round(time.time(), 3),
                                    "agent": name,
                                    "prompt_id": ack.get("prompt_id"),
                                    "kind": "boot" if first else "step",
                                }
                            )
                            + "\n"
                        )
                    plog.flush()
            seq = status["seq"]
            rate = None if last_seq is None else (seq - last_seq) / max(1e-9, now - last_t)
            last_seq, last_t = seq, now
            gpu = a.gpu_usd_per_hour * (now - t0) / 3600
            row = {
                "ts": round(now, 3),
                "elapsed_s": round(now - t0, 1),
                "coverage_pct": score["coverage_pct"],
                "exact_match_pct": score["exact_match_pct"],
                "work_done": score["work_done"],
                "work_total": score["work_total"],
                "paints": score["paints"],
                "seq": seq,
                "canvas_req_per_s": status["req_per_s_10s"],
                "paint_calls_per_s": None if rate is None else round(rate, 2),
                "agents": len(agents),
                "idle": idle,
                "working": working,
                "prompts_sent": sent,
                "flock_cost_usd": cost,
                "gpu_usd_est": round(gpu, 3),
            }
            prog.write(json.dumps(row) + "\n")
            prog.flush()
            print(
                f"[{row['elapsed_s']:7.0f}s] done {score['coverage_pct']:6.2f}%  "
                f"exact {score['exact_match_pct']:6.2f}%  canvas {status['req_per_s_10s']:.1f} req/s  "
                f"agents {len(agents)} (idle {idle}, working {working})  sent {sent}  "
                f"cost flock ${cost if cost is not None else 0:.2f} + gpu ~${gpu:.2f}",
                flush=True,
            )
            if done:
                print("canvas complete; stopping", flush=True)
                return 0
            if a.max_minutes and now - t0 > a.max_minutes * 60:
                print("max minutes reached; stopping", flush=True)
                return 0
            if (
                a.max_steps
                and agents
                and all(steps.get(x["name"], 0) >= a.max_steps for x in agents)
            ):
                print("every agent reached --max-steps; stopping", flush=True)
                return 0
            time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
