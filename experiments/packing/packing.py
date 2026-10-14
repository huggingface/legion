#!/usr/bin/env python3
"""Packing measurement: N concurrent trivial harness turns on one box, plus one idle resident session.

Samples this process's descendant tree every 0.5 s from /proc (RSS sum, CPU cores used) and the
container cgroup (memory.current). Stdlib only. Writes <out>/<harness>-N<n>.json per level and
<out>/<harness>-summary.json; flushes after every level so partial results survive an OOM.

  packing.py --harness claude --levels 5,10,20,40 --idle 60 --out out/
  packing.py --harness codex  --levels 1,3 --out out/          (CODEX_HOME must be prepared)
"""

import argparse
import json
import os
import resource
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

PROMPT = "Reply with the single word OK."
PAGE = os.sysconf("SC_PAGE_SIZE")
TCK = os.sysconf("SC_CLK_TCK")


def cmd_for(harness, model):
    if harness == "claude":
        return [
            "claude",
            "-p",
            "--model",
            model,
            "--max-budget-usd",
            "0.05",
            "--permission-mode",
            "bypassPermissions",
            "--output-format",
            "stream-json",
            "--verbose",
            PROMPT,
        ]
    return [
        "codex",
        "exec",
        "--json",
        "-m",
        model,
        "-c",
        'model_reasoning_effort="low"',
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "-c",
        'approval_policy="never"',
        PROMPT,
    ]


def read_cgroup(name):
    try:
        return int(Path("/sys/fs/cgroup", name).read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None


def tree_stats(root):
    """(rss_bytes, cpu_ticks, nprocs) summed over all descendants of root (root excluded)."""
    kids, info = {}, {}
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            s = Path(f"/proc/{p}/stat").read_text()
            f = s[s.rfind(")") + 2 :].split()
            rss = int(Path(f"/proc/{p}/statm").read_text().split()[1]) * PAGE
        except (OSError, IndexError, ValueError):
            continue
        kids.setdefault(int(f[1]), []).append(int(p))
        info[int(p)] = (rss, int(f[11]) + int(f[12]))
    rss = ticks = n = 0
    stack = list(kids.get(root, []))
    while stack:
        p = stack.pop()
        r, t = info[p]
        rss += r
        ticks += t
        n += 1
        stack += kids.get(p, [])
    return rss, ticks, n


class Sampler(threading.Thread):
    def __init__(self, root, interval=0.5):
        super().__init__(daemon=True)
        self.root, self.interval = root, interval
        self.samples, self.stop_ev = [], threading.Event()

    def run(self):
        last = None
        while not self.stop_ev.is_set():
            t = time.monotonic()
            rss, ticks, n = tree_stats(self.root)
            cores = (ticks - last[1]) / TCK / (t - last[0]) if last and ticks >= last[1] else None
            last = (t, ticks)
            self.samples.append(
                {
                    "t": t,
                    "rss_mb": rss / 2**20,
                    "cores": cores,
                    "nprocs": n,
                    "cg_mem_mb": (read_cgroup("memory.current") or 0) / 2**20,
                }
            )
            self.stop_ev.wait(self.interval)

    def stop(self):
        self.stop_ev.set()
        self.join()
        return self.samples


def pct(xs, q):
    xs = sorted(x for x in xs if x is not None)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else None


def run_turn(harness, model, workdir, rec):
    workdir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    p = subprocess.Popen(
        cmd_for(harness, model),
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        text=True,
    )
    rec.update(pid=p.pid, events=0)
    for line in p.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        rec["events"] += 1
        now = time.monotonic() - t0
        if ev.get("type") in ("system", "thread.started") and "first_event_s" not in rec:
            rec["first_event_s"] = round(now, 2)
        if ev.get("type") == "result":
            rec.update(
                {
                    k: ev.get(k)
                    for k in (
                        "is_error",
                        "result",
                        "duration_api_ms",
                        "total_cost_usd",
                        "session_id",
                    )
                }
            )
            rec["usage"] = ev.get("usage")
        if ev.get("type") == "turn.completed":
            rec["usage"] = ev.get("usage")
        if ev.get("type") == "item.completed" and ev.get("item", {}).get("type") == "agent_message":
            rec["result"] = ev["item"].get("text")
    rec["stderr_tail"] = p.stderr.read()[-300:]
    rec["rc"] = p.wait()
    rec["wall_s"] = round(time.monotonic() - t0, 2)


def level(harness, model, n, base, out):
    sampler = Sampler(os.getpid())
    sampler.start()
    time.sleep(1.0)
    cg0 = read_cgroup("memory.current")
    recs = [{"i": i} for i in range(n)]
    ths = [
        threading.Thread(target=run_turn, args=(harness, model, base / f"N{n}-{i:03d}", recs[i]))
        for i in range(n)
    ]
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0 = time.monotonic()
    [t.start() for t in ths]
    [t.join() for t in ths]
    wall = time.monotonic() - t0
    ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu_s = (ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime)
    time.sleep(1.0)
    samples = sampler.stop()
    ok = [
        r
        for r in recs
        if r.get("rc") == 0 and not r.get("is_error") and "OK" in (r.get("result") or "")
    ]
    busy = [s for s in samples if s["nprocs"] > 0]
    res = {
        "harness": harness,
        "model": model,
        "n": n,
        "ok": len(ok),
        "batch_wall_s": round(wall, 2),
        "cpu_s_per_turn": round(cpu_s / n, 2),
        "peak_rss_mb": round(max(s["rss_mb"] for s in samples), 1),
        "peak_rss_per_turn_mb": round(max(s["rss_mb"] for s in samples) / n, 1),
        "peak_cgroup_mem_mb": round(max(s["cg_mem_mb"] for s in samples), 1),
        "cgroup_mem_before_mb": round((cg0 or 0) / 2**20, 1),
        "peak_nprocs": max(s["nprocs"] for s in samples),
        "cores_mean": round(
            statistics.mean([s["cores"] for s in busy if s["cores"] is not None] or [0]), 2
        ),
        "cores_p50": pct([s["cores"] for s in busy], 0.5),
        "cores_p95": pct([s["cores"] for s in busy], 0.95),
        "cores_max": pct([s["cores"] for s in busy], 1.0),
        "wall_p50": pct([r.get("wall_s") for r in recs], 0.5),
        "wall_p95": pct([r.get("wall_s") for r in recs], 0.95),
        "wall_max": pct([r.get("wall_s") for r in recs], 1.0),
        "first_event_p50": pct([r.get("first_event_s") for r in recs], 0.5),
        "first_event_max": pct([r.get("first_event_s") for r in recs], 1.0),
        "api_ms_p50": pct([r.get("duration_api_ms") for r in recs], 0.5),
        "api_ms_p95": pct([r.get("duration_api_ms") for r in recs], 0.95),
        "cost_usd": round(sum(r.get("total_cost_usd") or 0 for r in recs), 5),
        "turns": recs,
        "samples": samples,
    }
    (out / f"{harness}-N{n}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k not in ("turns", "samples")}), flush=True)
    return res


def idle_resident(model, base, out, seconds):
    """Streaming-input Claude session: one prompt, then left alive `seconds` with stdin open."""
    wd = base / "idle"
    wd.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(
        [
            "claude",
            "-p",
            "--model",
            model,
            "--max-budget-usd",
            "0.05",
            "--permission-mode",
            "bypassPermissions",
            "--input-format",
            "stream-json",
            "--output-format",
            "stream-json",
            "--verbose",
        ],
        cwd=wd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    sampler = Sampler(os.getpid())
    sampler.start()
    t0 = time.monotonic()
    rec = {}
    p.stdin.write(
        json.dumps({"type": "user", "message": {"role": "user", "content": PROMPT}}) + "\n"
    )
    p.stdin.flush()
    for line in p.stdout:
        ev = json.loads(line)
        if ev.get("type") == "system" and "init_s" not in rec:
            rec["init_s"] = round(time.monotonic() - t0, 2)
        if ev.get("type") == "result":
            rec.update(
                result_s=round(time.monotonic() - t0, 2),
                **{k: ev.get(k) for k in ("result", "duration_api_ms", "total_cost_usd")},
            )
            break
    t_res = time.monotonic()
    time.sleep(seconds)
    p.stdin.close()
    p.wait(timeout=30)
    t_exit = time.monotonic()
    samples = sampler.stop()
    settle = min(5, seconds / 4)
    idle = [s for s in samples if t_res + settle <= s["t"] <= t_res + seconds]
    rec.update(
        idle_seconds=seconds,
        exit_after_stdin_close_s=round(t_exit - t_res - seconds, 2),
        peak_rss_mb=round(max(s["rss_mb"] for s in samples), 1),
        idle_rss_mb_mean=round(statistics.mean(s["rss_mb"] for s in idle), 1),
        idle_rss_mb_end=round(idle[-1]["rss_mb"], 1),
        idle_cores_mean=round(statistics.mean(s["cores"] or 0 for s in idle), 3),
        nprocs=max(s["nprocs"] for s in idle),
        samples=samples,
    )
    (out / "claude-idle.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps({k: v for k, v in rec.items() if k != "samples"}), flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", default="claude", choices=["claude", "codex"])
    ap.add_argument("--model")
    ap.add_argument("--levels", default="5,10,20,40")
    ap.add_argument(
        "--idle",
        type=int,
        default=0,
        help="seconds for the idle-resident probe (claude only, 0=skip)",
    )
    ap.add_argument("--base", default="/work/pack")
    ap.add_argument("--out", default="out")
    ap.add_argument(
        "--after-level", help="shell command run after each level (e.g. upload results)"
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(a.base)
    model = a.model or ("claude-haiku-5-5" if a.harness == "claude" else "gpt-6-luna")
    env = {
        "nproc": os.cpu_count(),
        "cgroup_memory_max": Path("/sys/fs/cgroup/memory.max").read_text().strip()
        if Path("/sys/fs/cgroup/memory.max").exists()
        else None,
        "cgroup_cpu_max": Path("/sys/fs/cgroup/cpu.max").read_text().strip()
        if Path("/sys/fs/cgroup/cpu.max").exists()
        else None,
        "mem_total_mb": int(Path("/proc/meminfo").read_text().split()[1]) // 1024,
    }
    print(json.dumps({"env": env}), flush=True)
    summary = {"env": env, "levels": []}
    if a.idle and a.harness == "claude":
        r = idle_resident(model, base, out, a.idle)
        summary["idle"] = {k: v for k, v in r.items() if k != "samples"}
    for n in [int(x) for x in a.levels.split(",")]:
        r = level(a.harness, model, n, base, out)
        summary["levels"].append({k: v for k, v in r.items() if k not in ("turns", "samples")})
        (out / f"{a.harness}-summary.json").write_text(json.dumps(summary, indent=1))
        if a.after_level:
            subprocess.run(a.after_level, shell=True, check=False)
        time.sleep(3)
    (out / f"{a.harness}-summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    sys.exit(main())
