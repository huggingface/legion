#!/usr/bin/env python3
"""Drive the pixel swarm: keep painters prompted until the canvas is done.

Default (bulk, for 1k agents): every --interval seconds
  1. canvas: GET /score?agents=false and /status (2 calls); stop at coverage 100%.
  2. controller: GET /swarms/{s} (1 call): per-type agent counts by state, cost, pending.
  3. per painter type, ONE `POST /swarms/{s}/prompts {type, state: idle, text}` (the route
     behind `flock prompt --swarm S --type T --state idle`, PR "prompt-type"). The first prompt
     to a type is the install step; later ones are the paint step.
  4. every --install-check-every s (rare, per agent): list agents, read canvas paint calls per
     agent; an idle agent with no new paint call over 3 checks gets the install step alone.
  5. one progress row (progress.jsonl) + the progress Space upload every --publish-every s.
Every HTTP call backs off exponentially with jitter on 429 / 5xx / transport errors (run 1:
9 x HTTP 429 from the Space edge at 100 agents with per-agent prompts).
If the controller has no bulk route (404/405), the driver falls back to --per-agent.

--per-agent: the run-1 mode, one prompt per agent per tick over `POST /agents/{name}/prompt`
(lists agents every tick: ~N+2 requests per tick).

With `done_tool: true` use --repeat: one prompt asks for the step repeatedly within a turn.

Env: FLOCK_URL, FLOCK_TOKEN (controller), CANVAS_URL (canvas, reads need no token).
Stdlib only (huggingface_hub only for the Space upload). --dry-run prints what it would send.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

# No single quotes: LFM2.5 over-escapes them inside its tool-call JSON (smoke 2026-10-19).
BOOT = (
    'python3 -c "import sys,urllib.request as u;u.urlretrieve(*sys.argv[1:])" '
    "$CANVAS_URL/cli/canvas canvas && python3 canvas status"
)
# `python3 canvas`, not `./canvas`: a workdir restored from the bucket loses the exec bit (run 1:
# 10 agents looped on "Permission denied" after a re-apply).
STEP = "python3 canvas claim 10 --json | python3 canvas paint - --json"


REPEAT = False  # --repeat: multi-step turns (spec `done_tool: true`)


def prompt_text(cmd: str) -> str:
    if REPEAT and cmd == STEP:
        return (
            "Run exactly this command. After each result run it again, the same way, until the "
            f'output says "You are done"; then call done.\n{cmd}'
        )
    if REPEAT:
        return f"Run exactly this one command, then call done:\n{cmd}"
    return f"Next step. Run exactly this one command:\n{cmd}"


class Backoff:
    """Retry policy shared by every request: exponential, full jitter, Retry-After honoured."""

    def __init__(self, tries: int = 6, base: float = 0.5, cap: float = 30.0):
        self.tries, self.base, self.cap = tries, base, cap
        self.retries = 0  # cumulative, recorded in progress rows
        self.sleep = time.sleep

    def delay(self, attempt: int, retry_after: str | None) -> float:
        d = random.uniform(0, min(self.cap, self.base * 2**attempt))
        if retry_after:
            try:
                d = max(d, min(self.cap, float(retry_after)))
            except ValueError:
                pass
        return d


BACKOFF = Backoff()


def http(
    method: str, url: str, token: str | None = None, body: dict | None = None, timeout: float = 30
) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        host = urllib.parse.urlsplit(url).hostname or ""
        if host.endswith(".hf.jobs") and os.environ.get("HF_TOKEN"):
            # job-mode controller behind the jobs proxy: HF token for the proxy, flock token
            # in X-Flock-Token (the proxy forwards it; smoke 2026-10-20)
            headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
            headers["X-Flock-Token"] = token
        else:
            headers["Authorization"] = f"Bearer {token}"
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(BACKOFF.tries):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            if e.code != 429 and e.code < 500 or attempt == BACKOFF.tries - 1:
                raise
            wait = BACKOFF.delay(attempt, e.headers.get("Retry-After"))
        except (urllib.error.URLError, OSError):
            if attempt == BACKOFF.tries - 1:
                raise
            wait = BACKOFF.delay(attempt, None)
        BACKOFF.retries += 1
        BACKOFF.sleep(wait)
    raise RuntimeError("unreachable")


class Controller:
    def __init__(self, url: str, token: str, swarm: str):
        self.url, self.token, self.swarm = url.rstrip("/"), token, swarm
        self.bulk_ok = True

    def agents(self, type_: str) -> list[dict]:
        out, cursor = [], None
        while True:
            q = {"type": type_, **({"cursor": cursor} if cursor else {})}
            page = http(
                "GET",
                f"{self.url}/swarms/{self.swarm}/agents?{urllib.parse.urlencode(q)}",
                self.token,
            )
            out += page["agents"]
            cursor = page.get("cursor")
            if not cursor:
                return out

    def children(self, parent: str) -> list[dict]:
        out, cursor = [], None
        while True:
            q = {"parent": parent, **({"cursor": cursor} if cursor else {})}
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

    def bulk(self, type_: str, text: str, state: str = "idle") -> dict | None:
        """One prompt to every `state` agent of a type; None if the route does not exist."""
        try:
            return http(
                "POST",
                f"{self.url}/swarms/{self.swarm}/prompts",
                self.token,
                {"type": type_, "state": state, "text": text},
            )
        except urllib.error.HTTPError as e:
            if e.code in (404, 405) and "no type" not in (e.read() or b"").decode(errors="ignore"):
                self.bulk_ok = False
                return None
            raise

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
            "gpus": a.gpus,
            "types": a.types,
            "max_pixels_per_call": None,
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
            # fresh=1: the final agent map must not be a cached image from up to N paints ago.
            for name, q in (("canvas.png", ""), ("agents.png", "?fresh=1")):
                with urllib.request.urlopen(f"{canvas}/{name}{q}", timeout=60) as r:
                    files[f"{base}/{name}"] = r.read()
            score = http("GET", f"{canvas}/score")
            score["target"] = http("GET", f"{canvas}/target/info")
            files[f"{base}/score.json"] = json.dumps(score).encode()
            files[f"{base}/paint-log.jsonl.gz"] = gzip.compress(fetch_log(canvas), 6)
            self.run["final"] = base
            tr = board_transcript()
            if tr is not None:
                files[f"{base}/board-transcript.jsonl"] = tr
                self.run["board_transcript"] = f"{base}/board-transcript.jsonl"
        except (urllib.error.URLError, OSError) as e:
            print(f"final artefacts incomplete: {e}", flush=True)
        files["runs.json"] = json.dumps(self.runs, indent=1).encode()
        self.put(files, f"final {self.run['id']} ({status})")


def board_transcript() -> bytes | None:
    """Experiment 2: the leaders' msgboard channel as JSONL (run/board.py transcript), when
    BOARD_URL and BOARD_TOKEN (a principal that can read the channel; never printed) are set.
    The channel: BOARD_CHANNEL (id) or BOARD_CHANNEL_NAME (default flock-pixel-test)."""
    if not (os.environ.get("BOARD_URL") and os.environ.get("BOARD_TOKEN")):
        return None
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        import board  # run/board.py, stdlib only

        return (board.transcript() + "\n").encode()
    except Exception as e:  # noqa: BLE001 - the dashboard must not stop the run
        print(f"board transcript not archived: {type(e).__name__}: {e}", flush=True)
        return None


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


def main() -> int:
    ap = argparse.ArgumentParser(description="Prompt idle painters until the canvas is done.")
    ap.add_argument("--swarm", default="pixel-100")
    ap.add_argument("--type", default="painter", help="painter type (see --types)")
    ap.add_argument("--types", default=None, help="comma list of painter types (1k: 3 types)")
    ap.add_argument("--canvas-url", default=os.environ.get("CANVAS_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--flock-url", default=os.environ.get("FLOCK_URL", "http://localhost:7860"))
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--max-steps", type=int, default=0, help="per agent (--per-agent); 0 = no cap")
    ap.add_argument("--max-minutes", type=float, default=0, help="stop after this; 0 = no cap")
    ap.add_argument("--gpu-usd-per-hour", type=float, default=0.80, help="all GPUs together")
    ap.add_argument("--gpus", type=int, default=1, help="GPU jobs in this run (progress Space)")
    ap.add_argument("--out", default="progress.jsonl")
    ap.add_argument("--prompts-log", default="prompts.jsonl")
    ap.add_argument("--per-agent", action="store_true", help="run-1 mode: one prompt per agent")
    ap.add_argument("--no-bulk", action="store_true", help="alias of --per-agent")
    ap.add_argument("--bulk", action="store_true", help="(default now; kept for old scripts)")
    ap.add_argument("--install-check-every", type=float, default=60.0)
    ap.add_argument(
        "--repeat",
        action="store_true",
        help="for `done_tool: true`: ask for the step repeatedly within a turn and prompt only "
        "idle agents (a prompt sent mid-turn would be steered into the running turn)",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--observe",
        action="store_true",
        help="experiment 2: send no prompts (leaders drive); record leaders' spawn "
        "timings, children states and per-region progress",
    )
    ap.add_argument("--leader-type", default="leader")
    ap.add_argument("--children-per-leader", type=int, default=100)
    ap.add_argument("--leader-every", type=float, default=30.0)
    ap.add_argument(
        "--space-id",
        default=os.environ.get("SPACE_ID"),
        help="progress Space, e.g. cmpatino/flock-pixel (needs HF_TOKEN)",
    )
    ap.add_argument("--publish-dir", default=None, help="also/instead write the Space files here")
    ap.add_argument("--publish-every", type=float, default=30.0)
    ap.add_argument("--run-id", default=None, help="default: <swarm>-<UTC time>")
    ap.add_argument("--run-name", default=None)
    ap.add_argument(
        "--controller-url",
        default=os.environ.get("FLOCK_PUBLIC_URL"),
        help="public controller status page for the Space's links",
    )
    ap.add_argument("--board-url", default=os.environ.get("BOARD_URL"))
    a = ap.parse_args()
    global REPEAT
    REPEAT = a.repeat
    a.types = [t for t in (a.types or a.type).split(",") if t]
    a.per_agent = a.per_agent or a.no_bulk
    a.run_id = a.run_id or f"{a.swarm}-{time.strftime('%Y%m%d-%H%M', time.gmtime())}"
    a.controller_url = a.controller_url or a.flock_url
    pub = Publisher(a, a.out)
    try:
        reason = Driver(a, pub).run()
    except KeyboardInterrupt:
        reason = "stopped"
    pub.finish(a.canvas_url.rstrip("/"), reason)
    print(f"run {a.run_id}: {reason}", flush=True)
    return 0


class Driver:
    def __init__(self, a, pub: Publisher):
        self.a, self.pub = a, pub
        self.ctl = Controller(a.flock_url, os.environ.get("FLOCK_TOKEN", ""), a.swarm)
        self.canvas = a.canvas_url.rstrip("/")
        self.booted: set[str] = set()  # agents sent the install step (per-agent paths)
        self.type_booted: set[str] = set()  # types sent the bulk install step
        self.steps: dict[str, int] = {}
        # Per agent: canvas paint calls at the last check, and checks since they last grew.
        self.last_calls: dict[str, int] = {}
        self.stalled: dict[str, int] = {}
        self.last_check = 0.0
        self.plog = None
        self.t0 = time.time()
        self.leader_t: dict[str, dict] = {}  # leader -> first/all children seen (s since start)
        self.last_leaders = 0.0
        self.leader_row: dict = {}

    def observe(self, now: float) -> dict:
        """Experiment 2: per leader, children by state and spawn timings (controller token,
        one request per leader every --leader-every s)."""
        if now - self.last_leaders < self.a.leader_every:
            return self.leader_row
        self.last_leaders = now
        try:
            leaders = self.ctl.agents(self.a.leader_type)
            row = {}
            for ld in leaders:
                kids = self.ctl.children(ld["name"])
                st: dict[str, int] = {}
                for k in kids:
                    st[k["state"]] = st.get(k["state"], 0) + 1
                tm = self.leader_t.setdefault(
                    ld["name"], {"first_child_s": None, "all_children_s": None}
                )
                if kids and tm["first_child_s"] is None:
                    tm["first_child_s"] = round(now - self.t0, 1)
                if len(kids) >= self.a.children_per_leader and tm["all_children_s"] is None:
                    tm["all_children_s"] = round(now - self.t0, 1)
                row[ld["name"]] = {
                    "state": ld["state"],
                    "turns": ld.get("turns"),
                    "cost_usd": ld.get("cost_usd"),
                    "children": len(kids),
                    "children_by_state": st,
                    **tm,
                }
            regions = http("GET", f"{self.canvas}/regions").get("regions", [])
            self.leader_row = {
                "leaders": row,
                "regions": [
                    {
                        "owner": r["owner"],
                        "y": r["y"],
                        "h": r["h"],
                        "grants": len(r["grants"]),
                        "done_pct": r.get("done_pct"),
                    }
                    for r in regions
                ],
            }
        except (urllib.error.URLError, OSError) as e:
            print(f"observe failed: {e}", flush=True)
        return self.leader_row

    def log_prompt(self, name: str, ack: dict, kind: str) -> None:
        self.plog.write(
            json.dumps(
                {
                    "ts": round(time.time(), 3),
                    "agent": name,
                    "prompt_id": ack.get("prompt_id"),
                    "kind": kind,
                }
            )
            + "\n"
        )

    def send(self, name: str, cmd: str, kind: str) -> bool:
        ack = {"prompt_id": "dry-run"}
        if not self.a.dry_run:
            try:
                ack = self.ctl.prompt(name, prompt_text(cmd))
            except (urllib.error.URLError, OSError) as e:
                print(f"prompt {name} failed: {e}", flush=True)
                return False
        self.log_prompt(name, ack, kind)
        return True

    def stalled_now(self, name: str, calls_by_agent: dict) -> bool:
        """True when the agent made no new paint call since the last check, 3 checks running
        (`/score` keeps agent names across deployments, so 'never seen' alone is not enough)."""
        calls = (calls_by_agent.get(name) or {}).get("calls", 0)
        grew = calls > self.last_calls.get(name, -1)
        self.stalled[name] = 0 if grew else self.stalled.get(name, 0) + 1
        self.last_calls[name] = calls
        return self.stalled[name] >= 3

    # ----------------------------------------------------------------- bulk
    def tick_bulk(self, now: float) -> int:
        sent = 0
        for t in self.a.types:
            cmd, kind = (STEP, "bulk-step") if t in self.type_booted else (BOOT, "bulk-boot")
            if self.a.dry_run:
                ack = {"queued": 0, "pending": 0}
            else:
                ack = self.ctl.bulk(t, prompt_text(cmd))
                if ack is None:
                    print(
                        "driver: controller has no POST /swarms/{s}/prompts; per-agent mode",
                        flush=True,
                    )
                    self.a.per_agent = True
                    return sent
            self.type_booted.add(t)
            n = ack.get("queued", 0) + ack.get("pending", 0)
            sent += n
            self.plog.write(
                json.dumps(
                    {
                        "ts": round(now, 3),
                        "type": t,
                        "kind": kind,
                        "prompts": n,
                        "skipped_state": ack.get("skipped_state"),
                    }
                )
                + "\n"
            )
        if now - self.last_check >= self.a.install_check_every:
            self.last_check = now
            sent += self.reinstall_stalled()
        return sent

    def reinstall_stalled(self) -> int:
        """Per agent and rare: idle agents with no paint call over 3 checks get the install."""
        try:
            calls = http("GET", f"{self.canvas}/score").get("agents", {})
            agents = [ag for t in self.a.types for ag in self.ctl.agents(t)]
        except (urllib.error.URLError, OSError) as e:
            print(f"install check skipped: {e}", flush=True)
            return 0
        sent = 0
        for ag in agents:
            name = ag["name"]
            if (
                self.stalled_now(name, calls)
                and ag["state"] == "idle"
                and self.send(name, BOOT, "boot")
            ):
                self.stalled[name] = 0
                sent += 1
        return sent

    # ------------------------------------------------------------- per agent
    def tick_per_agent(self, score_agents: dict) -> int:
        a, sent = self.a, 0
        agents = [ag for t in a.types for ag in self.ctl.agents(t)]
        for ag in agents:
            name = ag["name"]
            if ag["state"] not in ("idle", "working") or ag.get("pending", 0) > 0:
                continue
            if a.repeat and ag["state"] != "idle":
                continue
            if a.max_steps and self.steps.get(name, 0) >= a.max_steps:
                continue
            first = name not in self.booted or self.stalled_now(name, score_agents)
            if first:
                self.steps[name] = self.stalled[name] = 0
            if not self.send(name, BOOT if first else STEP, "boot" if first else "step"):
                continue
            self.booted.add(name)
            if not first:
                self.steps[name] = self.steps.get(name, 0) + 1
            sent += 1
        self.agent_names = [ag["name"] for ag in agents]
        return sent

    # ------------------------------------------------------------------ loop
    def run(self) -> str:
        a = self.a
        print(
            f"driver: swarm={a.swarm} types={','.join(a.types)} canvas={self.canvas} "
            "mode="
            + (
                "observe (leaders drive)"
                if a.observe
                else "per-agent prompts"
                if a.per_agent
                else "bulk POST /swarms/{s}/prompts"
            ),
            flush=True,
        )
        t0 = time.time()
        last_seq = last_t = last_done = None
        self.agent_names: list[str] = []
        with open(a.out, "a") as prog, open(a.prompts_log, "a") as self.plog:
            while True:
                now = time.time()
                try:
                    score = http("GET", f"{self.canvas}/score?agents={str(a.per_agent).lower()}")
                    status = http("GET", f"{self.canvas}/status")
                except (urllib.error.URLError, OSError) as e:
                    print(f"canvas unreachable: {e}; retrying", flush=True)
                    time.sleep(a.interval)
                    continue
                done = score["work_done"] >= score["work_total"]
                by_state: dict[str, int] = {}
                cost = pending = None
                try:
                    fst = self.ctl.status()
                    cost, pending = fst.get("cost_usd"), fst.get("pending_prompts")
                    for t in a.types:
                        for k, v in (
                            (fst.get("types") or {}).get(t, {}).get("agents_by_state") or {}
                        ).items():
                            by_state[k] = by_state.get(k, 0) + v
                except (urllib.error.URLError, OSError) as e:
                    print(f"controller error: {e}", flush=True)
                n_agents = sum(by_state.values())
                sent = 0
                if a.observe:
                    extra = self.observe(now)
                elif not done and n_agents:
                    try:
                        if not a.per_agent:
                            sent = self.tick_bulk(now)
                        if a.per_agent:
                            sent += self.tick_per_agent(score.get("agents", {}))
                    except (urllib.error.URLError, OSError) as e:
                        print(f"prompting failed this tick: {e}", flush=True)
                    self.plog.flush()
                seq = status["seq"]
                dt = None if last_t is None else max(1e-9, now - last_t)
                rate = None if dt is None else (seq - last_seq) / dt
                px_s = None if dt is None else (score["work_done"] - last_done) / dt
                last_seq, last_t, last_done = seq, now, score["work_done"]
                gpu = a.gpu_usd_per_hour * (now - t0) / 3600
                idle = by_state.get("idle", 0)
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
                    "px_per_s": None if px_s is None else round(px_s, 1),
                    "active_agents": status.get("active_agents_10s"),
                    "step_gap_p50_s": status.get("step_gap_p50_s"),
                    "model_p50_s": None,  # not exposed by flock or vLLM-through-proxy cheaply
                    "agents": n_agents,
                    "agents_by_state": by_state,
                    "idle_pct": round(100 * idle / n_agents, 1) if n_agents else None,
                    "pending_prompts": pending,
                    "prompts_sent": sent,
                    "http_retries": BACKOFF.retries,
                    "flock_cost_usd": cost,
                    "gpu_usd_est": round(gpu, 3),
                }
                if a.observe:
                    row.update(extra)
                prog.write(json.dumps(row) + "\n")
                prog.flush()
                print(
                    f"[{row['elapsed_s']:7.0f}s] done {score['coverage_pct']:6.2f}%  "
                    f"px/s {row['px_per_s'] or 0:7.1f}  calls/s {row['paint_calls_per_s'] or 0:6.1f}  "
                    f"active {row['active_agents']}  agents {n_agents} (idle {row['idle_pct']}%)  "
                    f"sent {sent}  retries {BACKOFF.retries}  gpu ~${gpu:.2f}",
                    flush=True,
                )
                self.pub.tick()
                if done:
                    print("canvas complete; stopping", flush=True)
                    return "finished"
                if a.max_minutes and now - t0 > a.max_minutes * 60:
                    print("max minutes reached; stopping", flush=True)
                    return "stopped: max minutes"
                if (
                    a.per_agent
                    and a.max_steps
                    and self.agent_names
                    and all(self.steps.get(x, 0) >= a.max_steps for x in self.agent_names)
                ):
                    print("every agent reached --max-steps; stopping", flush=True)
                    return "stopped: max steps"
                time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
