#!/usr/bin/env python3
"""Drive the pixel swarm: keep painters prompted until the canvas is done.

Every --interval seconds:
  1. GET <canvas>/score; stop when the work set is fully painted (coverage 100%).
  2. Prompt painters that have nothing queued (default: per agent). List agents over the
     controller API (what `flock agents` reads) and POST one prompt per agent (what `flock prompt`
     sends) to every idle or working agent with no pending prompt, so the next step is always
     queued. --bulk instead runs `flock prompt --swarm S --type T --state idle "<step>"` (PR
     "prompt-type"); it only reaches idle agents, so steps wait for the controller to see the
     agent idle (smoke 2026-10-19: ~0.3 vs 1.8 paint calls/s for 3 agents). Either way a new
     agent gets the install step once, per agent, and later prompts are the plain step.
  3. Print one progress line and append it to progress.jsonl; every prompt sent goes to
     prompts.jsonl (agent, prompt_id, ts) so flock turns join the canvas log by agent + time.

With `tool_choice: required` each prompt is one bash step (claim 10 | paint), so the driver is
the loop. --max-steps caps steps per agent (0 = no cap).

Env: FLOCK_URL, FLOCK_TOKEN (controller), CANVAS_URL (canvas, reads need no token).
Stdlib only. --dry-run prints what it would send.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# No single quotes: LFM2.5 over-escapes them inside its tool-call JSON (smoke 2026-10-19).
BOOT = (
    'python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" '
    "$CANVAS_URL/cli/canvas canvas && chmod +x canvas && ./canvas status"
)
STEP = "./canvas claim 10 --json | ./canvas paint - --json"


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


class Publisher:
    """Feeds the progress Space (experiments/pixel/progress/): `runs.json`,
    `runs/<id>/progress.jsonl` every --publish-every s, and `final/<id>/` at the end.
    Targets a Space (--space-id, needs HF_TOKEN and huggingface_hub) and/or a local dir."""

    def __init__(self, a, progress_path: str):
        self.space_id, self.local = a.space_id, a.publish_dir
        self.every, self.progress_path = a.publish_every, progress_path
        self.last = 0.0
        self.run = {
            "id": a.run_id,
            "name": a.run_name or a.run_id,
            "swarm": a.swarm,
            "started_at": round(time.time(), 3),
            "finished_at": None,
            "status": "running",
            "canvas_url": a.canvas_url.rstrip("/"),
            "controller_url": a.controller_url,
            "board_url": a.board_url,
        }
        self.runs = [r for r in self._read_runs() if r.get("id") != a.run_id] + [self.run]

    @property
    def on(self) -> bool:
        return bool(self.space_id or self.local)

    def _read_runs(self) -> list[dict]:
        try:
            if self.local and os.path.exists(os.path.join(self.local, "runs.json")):
                with open(os.path.join(self.local, "runs.json")) as fh:
                    return json.load(fh)
            if self.space_id:
                from huggingface_hub import hf_hub_download

                p = hf_hub_download(self.space_id, "runs.json", repo_type="space")
                with open(p) as fh:
                    return json.load(fh)
        except Exception as e:  # noqa: BLE001 - a fresh Space has no runs.json
            print(f"publish: no previous runs.json ({type(e).__name__})", flush=True)
        return []

    def put(self, files: dict[str, bytes], msg: str) -> None:
        if self.local:
            for path, data in files.items():
                dst = os.path.join(self.local, path)
                os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
                with open(dst, "wb") as fh:
                    fh.write(data)
        if self.space_id:
            try:
                from huggingface_hub import CommitOperationAdd, HfApi

                ops = [
                    CommitOperationAdd(path_in_repo=k, path_or_fileobj=v) for k, v in files.items()
                ]
                HfApi().create_commit(self.space_id, ops, commit_message=msg, repo_type="space")
            except Exception as e:  # noqa: BLE001 - never stop the run for the dashboard
                print(f"publish failed: {type(e).__name__}: {str(e)[:200]}", flush=True)

    def _state(self) -> dict[str, bytes]:
        with open(self.progress_path, "rb") as fh:
            prog = fh.read()
        return {
            "runs.json": json.dumps(self.runs, indent=1).encode(),
            f"runs/{self.run['id']}/progress.jsonl": prog,
        }

    def tick(self) -> None:
        if self.on and time.time() - self.last >= self.every:
            self.last = time.time()
            self.put(self._state(), f"progress {self.run['id']}")

    def finish(self, canvas: str, status: str) -> None:
        if not self.on:
            return
        self.run["status"], self.run["finished_at"] = status, round(time.time(), 3)
        base = f"final/{self.run['id']}"
        files = self._state()
        try:
            with urllib.request.urlopen(f"{canvas}/canvas.png", timeout=60) as r:
                files[f"{base}/canvas.png"] = r.read()
            score = http("GET", f"{canvas}/score")
            score["target"] = http("GET", f"{canvas}/target/info")
            files[f"{base}/score.json"] = json.dumps(score).encode()
            files[f"{base}/paint-log.jsonl.gz"] = gzip.compress(fetch_log(canvas), 6)
            self.run["final"] = base
        except (urllib.error.URLError, OSError) as e:
            print(f"final artefacts incomplete: {e}", flush=True)
        files["runs.json"] = json.dumps(self.runs, indent=1).encode()
        self.put(files, f"final {self.run['id']} ({status})")


def fetch_log(canvas: str) -> bytes:
    """The whole paint log, paged through GET /log (5000 entries per call)."""
    out, after = [], 0
    while True:
        with urllib.request.urlopen(f"{canvas}/log?after={after}&limit=5000", timeout=60) as r:
            chunk = r.read()
        if not chunk.strip():
            return b"".join(out)
        out.append(chunk)
        after = json.loads(chunk.rstrip(b"\n").rsplit(b"\n", 1)[-1])["seq"]


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
    ap.add_argument(
        "--bulk", action="store_true", help="steps via `flock prompt --type --state idle`"
    )
    ap.add_argument(
        "--no-bulk", action="store_true", help="per agent (the default; kept for old scripts)"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--space-id",
        default=os.environ.get("SPACE_ID"),
        help="progress Space, e.g. cmpatino/flock-pixel (needs HF_TOKEN)",
    )
    ap.add_argument("--publish-dir", default=None, help="also/instead write the Space files here")
    ap.add_argument("--publish-every", type=float, default=60.0)
    ap.add_argument("--run-id", default=None, help="default: <swarm>-<UTC time>")
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--controller-url",
        default=os.environ.get("FLOCK_PUBLIC_URL"),
        help="public controller status page for the Space's links",
    )
    ap.add_argument("--board-url", default=os.environ.get("BOARD_URL"))
    a = ap.parse_args()
    a.run_id = a.run_id or f"{a.swarm}-{time.strftime('%Y%m%d-%H%M', time.gmtime())}"
    a.controller_url = a.controller_url or a.flock_url
    pub = Publisher(a, a.out)
    try:
        reason = drive(a, pub)
    except KeyboardInterrupt:
        reason = "stopped"
    pub.finish(a.canvas_url.rstrip("/"), reason)
    print(f"run {a.run_id}: {reason}", flush=True)
    return 0


def drive(a, pub: Publisher) -> str:
    ctl = Controller(a.flock_url, os.environ.get("FLOCK_TOKEN", ""), a.swarm, a.type)
    canvas = a.canvas_url.rstrip("/")
    bulk = a.bulk and not a.no_bulk and bulk_cli_available()
    if a.bulk and not bulk:
        print("driver: --bulk needs `flock prompt --swarm/--type/--state`; prompting per agent")
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
            cost = pending = None
            try:
                agents = ctl.agents()
                fst = ctl.status()
                cost, pending = fst.get("cost_usd"), fst.get("pending_prompts")
            except (urllib.error.URLError, OSError) as e:
                print(f"controller error: {e}", flush=True)
                agents = []
            by_state: dict[str, int] = {}
            for ag in agents:
                by_state[ag["state"]] = by_state.get(ag["state"], 0) + 1
            idle, working = by_state.get("idle", 0), by_state.get("working", 0)
            turns = sum(ag.get("turns") or 0 for ag in agents)
            if not done and agents and bulk:
                # New agents (and ones that never reached the canvas after 3 steps) get the
                # install step individually; everyone idle then gets the plain step in bulk.
                for ag in agents:
                    name = ag["name"]
                    seen = name in score.get("agents", {})
                    if name in booted and (seen or steps.get(name, 0) < 3):
                        continue
                    if ag["state"] not in ("idle", "working"):
                        continue
                    steps[name] = 0
                    ack = {"prompt_id": "dry-run"}
                    if not a.dry_run:
                        try:
                            ack = ctl.prompt(name, prompt_text(BOOT))
                        except (urllib.error.URLError, OSError) as e:
                            print(f"prompt {name} failed: {e}", flush=True)
                            continue
                    booted.add(name)
                    sent += 1
                    plog.write(
                        json.dumps(
                            {
                                "ts": round(time.time(), 3),
                                "agent": name,
                                "prompt_id": ack.get("prompt_id"),
                                "kind": "boot",
                            }
                        )
                        + "\n"
                    )
                for ag in agents:
                    if ag["state"] == "idle" and ag["name"] in booted:
                        steps[ag["name"]] = steps.get(ag["name"], 0) + 1
                plog.flush()
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
                        prompt_text(STEP),
                    ]
                    if a.dry_run:
                        print("would run:", " ".join(cmd[:8]), "...")
                    else:
                        subprocess.run(cmd, check=False, capture_output=True)
                    sent += idle
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
                "done_pct": score["coverage_pct"],
                "exact_pct": score["exact_match_pct"],
                "mse": score["mse"],
                "psnr": score["psnr"],
                "work_done": score["work_done"],
                "work_total": score["work_total"],
                "paints": score["paints"],
                "seq": seq,
                "req_s": status["req_per_s_10s"],
                "claims_outstanding": status["claims_outstanding"],
                "paint_calls_per_s": None if rate is None else round(rate, 2),
                "agents": len(agents),
                "agents_by_state": by_state,
                "turns": turns,
                "pending_prompts": pending,
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
            pub.tick()
            if done:
                print("canvas complete; stopping", flush=True)
                return "finished"
            if a.max_minutes and now - t0 > a.max_minutes * 60:
                print("max minutes reached; stopping", flush=True)
                return "stopped: max minutes"
            if (
                a.max_steps
                and agents
                and all(steps.get(x["name"], 0) >= a.max_steps for x in agents)
            ):
                print("every agent reached --max-steps; stopping", flush=True)
                return "stopped: max steps"
            time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
