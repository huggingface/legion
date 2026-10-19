#!/usr/bin/env python3
"""``canvas``: the command agents call to paint. Stdlib only (no install needed).

Env: CANVAS_URL (default http://127.0.0.1:8765), CANVAS_TOKEN, FLOCK_AGENT, FLOCK_TURN,
CANVAS_PROXY_TOKEN (optional: sent as ``Authorization: Bearer`` for the HF jobs proxy; the
canvas token then goes in ``X-Canvas-Token``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

USAGE = """canvas <command> [args] [--json]
  claim [-n N]            lease up to N pixels that still need paint (prints target colours)
  get X Y [W H]           target colour (and current colour) of a pixel or small region
  paint X,Y,RRGGBB ...    paint pixels (max per call is enforced by the server; '-' reads stdin)
  release                 give back your leased pixels
  score                   progress: exact match %, MSE/PSNR, paint outcomes
  status                  service status: agents, claims, req/s
  info                    target size, palette, limits"""


class CliError(Exception):
    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.code = code


def _headers() -> dict:
    h = {"Content-Type": "application/json", "X-Request-Id": uuid.uuid4().hex[:16]}
    tok = os.environ.get("CANVAS_TOKEN", "")
    proxy = os.environ.get("CANVAS_PROXY_TOKEN", "")
    if tok:
        h["X-Canvas-Token"] = tok
    if proxy:
        h["Authorization"] = f"Bearer {proxy}"
    elif tok:
        h["Authorization"] = f"Bearer {tok}"
    if os.environ.get("FLOCK_AGENT"):
        h["X-Flock-Agent"] = os.environ["FLOCK_AGENT"]
    if os.environ.get("FLOCK_TURN"):
        h["X-Flock-Turn"] = os.environ["FLOCK_TURN"]
    return h


def call(method: str, path: str, body: dict | None = None, retries: int = 3) -> dict:
    url = os.environ.get("CANVAS_URL", "http://127.0.0.1:8765").rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    headers = _headers()
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                out = json.loads(r.read() or b"{}")
                if isinstance(out, dict):
                    out.setdefault("req", headers["X-Request-Id"])
                return out
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read()).get("error") or e.reason
            except Exception:  # noqa: BLE001
                msg = e.reason
            if e.code == 429 and attempt < retries:
                wait = float(e.headers.get("Retry-After", "1"))
                time.sleep(min(wait, 5.0))
                continue
            code = 2 if e.code in (401, 403) else 1
            raise CliError(f"{msg} (HTTP {e.code})", code) from None
        except urllib.error.URLError as e:
            raise CliError(
                f"cannot reach canvas at {url}: {e.reason}; check CANVAS_URL", 3
            ) from None
    raise CliError("rate limited: slow down and retry", 1)


def parse_pixel(tok: str) -> dict:
    parts = tok.replace(" ", "").split(",")
    if len(parts) != 3:
        raise CliError(f"bad pixel {tok!r}: write X,Y,RRGGBB e.g. 10,20,ff9d00")
    try:
        x, y = int(parts[0]), int(parts[1])
    except ValueError:
        raise CliError(f"bad pixel {tok!r}: X and Y must be integers") from None
    hx = parts[2].lstrip("#").lower()
    if len(hx) != 6 or any(c not in "0123456789abcdef" for c in hx):
        raise CliError(f"bad colour in {tok!r}: want 6 hex digits like ff9d00")
    return {"x": x, "y": y, "hex": hx}


def fmt_claim(out: dict) -> str:
    px = out["pixels"]
    if not px:
        if out.get("done"):
            return "nothing to claim: the canvas matches the target. You are done."
        return "nothing free to claim right now (all remaining pixels are leased); retry in ~1 min."
    lines = [f"claimed {len(px)} px, lease {out['lease_s']:g}s, {out['remaining']} px left overall"]
    if "hex" in px[0]:
        lines.append("paint them with:")
        lines.append("canvas paint " + " ".join(f"{p['x']},{p['y']},{p['hex']}" for p in px))
    else:
        lines.append("pixels: " + " ".join(f"{p['x']},{p['y']}" for p in px))
        lines.append("look up target colours with: canvas get X Y")
    return "\n".join(lines)


def fmt_paint(out: dict) -> str:
    c = out["counts"]
    parts = ", ".join(f"{v} {k}" for k, v in c.items())
    lines = [f"painted {sum(c.values())} px: {parts} (log #{out['seq']})"]
    for r in out["results"]:
        if r["outcome"] in ("wrong", "overwritten"):
            lines.append(f"  {r['x']},{r['y']} {r['outcome']}: target is {r['want']}")
        elif r["outcome"] == "refused":
            lines.append(f"  {r['x']},{r['y']} refused: leased by another agent, skip it")
    s = out["score"]
    lines.append(f"canvas: {s['exact_match_pct']}% exact match, {s['todo']} px to go")
    return "\n".join(lines)


def fmt_get(out: dict) -> str:
    if "rows" in out:
        lines = [f"target region x={out['x']} y={out['y']} {out['w']}x{out['h']} (row by row):"]
        lines += [f"y={out['y'] + i}: {row}" for i, row in enumerate(out["rows"])]
        return "\n".join(lines)
    ok = "already correct" if out["ok"] else f"currently {out['current']}"
    return f"{out['x']},{out['y']} target {out['hex']} ({ok})"


def fmt_score(out: dict) -> str:
    p = out["paints"]
    return (
        f"exact match {out['exact_match_pct']}% ({out['todo']} px to go), "
        f"foreground coverage {out['coverage_pct']}%, MSE {out['mse']}, PSNR {out['psnr']}\n"
        f"paints: {p['useful']} useful, {p['duplicate']} duplicate, {p['wrong']} wrong, "
        f"{p['overwritten']} overwritten, {p['refused']} refused"
    )


def fmt_status(out: dict) -> str:
    c = out["config"]
    return (
        f"up {out['uptime_s']}s, {out['seq']} paint calls, {out['agents_seen']} agents "
        f"({out['agents_active_60s']} active), {out['claims_outstanding']} px leased, "
        f"{out['req_per_s_10s']} req/s, {out['exact_match_pct']}% done\n"
        f"limits: {c['max_pixels_per_call']} px per paint, claim up to {c['max_claim']}, "
        f"lease {c['claim_ttl_s']:g}s, {c['rate_limit_per_s']:g} calls/s per agent"
    )


def run(argv: list[str]) -> tuple[dict, str]:
    ap = argparse.ArgumentParser(prog="canvas", usage=USAGE, add_help=True)
    ap.add_argument("cmd", choices=["claim", "get", "paint", "release", "score", "status", "info"])
    ap.add_argument("args", nargs="*")
    ap.add_argument("-n", type=int, default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "claim":
        out = call("POST", "/claim", {"n": a.n})
        return out, fmt_claim(out)
    if a.cmd == "get":
        nums = a.args
        if len(nums) not in (2, 4) or not all(n.lstrip("-").isdigit() for n in nums):
            raise CliError("usage: canvas get X Y [W H]  (integers)")
        x, y = int(nums[0]), int(nums[1])
        if len(nums) == 2:
            out = call("GET", f"/target/pixel?x={x}&y={y}")
        else:
            q = urllib.parse.urlencode({"x": x, "y": y, "w": nums[2], "h": nums[3]})
            out = call("GET", f"/target/region?{q}")
        return out, fmt_get(out)
    if a.cmd == "paint":
        toks = a.args
        if toks == ["-"]:
            toks = sys.stdin.read().split()
        if not toks:
            raise CliError("usage: canvas paint X,Y,RRGGBB [X,Y,RRGGBB ...]")
        body = {"pixels": [parse_pixel(t) for t in toks]}
        if os.environ.get("FLOCK_TURN"):
            body["turn"] = os.environ["FLOCK_TURN"]
        out = call("POST", "/paint", body)
        return out, fmt_paint(out)
    if a.cmd == "release":
        out = call("POST", "/release", {})
        return out, f"released {out['released']} px"
    if a.cmd == "score":
        out = call("GET", "/score?agents=false")
        return out, fmt_score(out)
    if a.cmd == "status":
        out = call("GET", "/status")
        return out, fmt_status(out)
    out = call("GET", "/target/info")
    return out, (
        f"target {out['width']}x{out['height']}, {out['colours']} colours, background "
        f"{out['background']}, {out['work_pixels']} px differ from blank; "
        f"max {out['max_pixels_per_call']} px per paint"
    )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    as_json = "--json" in argv
    try:
        out, text = run(argv)
    except CliError as e:
        if as_json:
            print(json.dumps({"error": str(e)}))
        else:
            print(f"error: {e}", file=sys.stderr)
        return e.code
    print(json.dumps(out) if as_json else text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
