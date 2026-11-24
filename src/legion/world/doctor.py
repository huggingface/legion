"""`python -m legion.world doctor <url>`: lint a world's manifest against the author checklist
(docs/world.md) and optionally load its read-only tools. Returns (lines, failed)."""

from __future__ import annotations

import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from legion.world.host import WorldError, fetch_manifest
from legion.world.protocol import TOKEN_HEADER, WorldManifest, WorldTool

READ_VERBS = ("get", "list", "read", "search", "show")
BOUND_ARG = re.compile(r"limit|max|page|cursor|offset|bytes|top_k|count|size", re.IGNORECASE)
MAX_CALLS = 64
BURST_BUDGET_S = 60.0
CAPS = ("metrics-v1", "admin-v1", "workspace-upload-v1", "workspace-upload-v2")


def _looks_read(t: WorldTool) -> bool:
    return t.name.split("_")[0] in READ_VERBS


def _noun(t: WorldTool) -> set[str]:
    return {w.rstrip("s") for w in t.name.split("_")[1:]}


def _pct(xs: list[float], q: float) -> float:
    return sorted(xs)[min(len(xs) - 1, int(q * len(xs)))]


def lint(m: WorldManifest) -> list[str]:
    out = []
    reads = [t for t in m.tools if t.read_only or _looks_read(t)]
    for t in m.tools:
        if _looks_read(t) and not t.read_only:
            out.append(f"warn: {t.name} looks like a read but has no readOnlyHint (fast pool, retries)")
        if not (t.read_only or _looks_read(t)) and not any(_noun(t) & _noun(r) for r in reads):
            out.append(f"warn: write tool {t.name} has no read tool sharing its noun")
        if t.read_only and not any(BOUND_ARG.search(p) for p in t.parameters.get("properties", {})):
            out.append(f"warn: read tool {t.name} has no limit/page/bytes argument")
    return out


def doctor(url: str, token: str | None = None, calls: int = 0,
           transport: httpx.BaseTransport | None = None) -> tuple[list[str], bool]:
    url = url.rstrip("/")
    if calls > MAX_CALLS:
        return [f"FAIL: --calls {calls} exceeds the cap of {MAX_CALLS}"], True
    try:
        m = fetch_manifest(url, token, transport=transport)
    except WorldError as e:
        return [f"FAIL: {e}"], True
    out = [f"manifest ok: {m.name} {m.version}, {len(m.tools)} tools"]
    out += lint(m)
    out.append("capabilities: " + (", ".join(c for c in CAPS if c in m.capabilities) or "none of " + ", ".join(CAPS)))
    headers = {"X-Legion-Agent": "doctor", **({TOKEN_HEADER: token} if token else {})}
    with httpx.Client(transport=transport, timeout=10, headers=headers) as http:
        for path in ("/healthz", "/metrics"):
            try:
                code = http.get(url + path).status_code
            except httpx.HTTPError as e:
                code = type(e).__name__
            out.append(f"{path}: {code}" if code == 200 else f"warn: {path}: {code}")
        failed = False
        if calls > 0:
            failed = _burst(http, url, m, calls, out)
    return out, failed


def _burst(http: httpx.Client, url: str, m: WorldManifest, n: int, out: list[str]) -> bool:
    failed, done, mlat = False, threading.Event(), []

    def poll() -> None:  # /manifest latency while the tools are loaded
        while not done.is_set():
            t0 = time.perf_counter()
            with_err = None
            try:
                http.get(f"{url}/manifest")
            except httpx.HTTPError as e:
                with_err = e
            mlat.append(float("inf") if with_err else time.perf_counter() - t0)
            time.sleep(0.02)

    def one(name: str) -> tuple[float, str | None]:
        t0 = time.perf_counter()
        try:
            r = http.post(f"{url}/tools/{name}", json={"args": {}})
            body = r.json()
            code = None if r.status_code == 200 and body.get("ok") else \
                (body.get("error") or {}).get("code") or str(r.status_code)
        except (httpx.HTTPError, ValueError) as e:
            code = type(e).__name__
        return time.perf_counter() - t0, code

    p = threading.Thread(target=poll, daemon=True)
    p.start()
    t_end = time.monotonic() + BURST_BUDGET_S
    try:
        for t in (t for t in m.tools if t.read_only):
            if time.monotonic() > t_end:
                out.append(f"warn: burst budget {BURST_BUDGET_S:.0f} s spent; {t.name} and later tools skipped")
                break
            if t.parameters.get("required"):
                out.append(f"warn: {t.name} has required args; not called")
                continue
            with ThreadPoolExecutor(n) as ex:
                res = list(ex.map(one, [t.name] * n))
            lat = [r[0] for r in res]
            errs = sorted({c for _, c in res if c})
            out.append(f"{t.name}: {n} calls p50 {statistics.median(lat) * 1000:.0f} ms "
                       f"p95 {_pct(lat, .95) * 1000:.0f} ms errors {errs or 'none'}")
            if errs:
                out[-1] = "FAIL: " + out[-1]
                failed = True
    finally:
        done.set()
        p.join()
    if mlat:
        out.append(f"/manifest during burst: p50 {_pct(mlat, .5) * 1000:.0f} ms "
                   f"p95 {_pct(mlat, .95) * 1000:.0f} ms ({len(mlat)} polls)")
    return failed
