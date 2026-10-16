"""FastAPI app: the routes in design.md "API (controller)" plus /healthz and GET /swarms."""

from __future__ import annotations

import asyncio
import html
import os
import resource
import secrets
import threading
import time
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

import httpx
from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import Field, TypeAdapter
from starlette.background import BackgroundTask

from flock.controller.preflight import run_preflight
from flock.controller.reconcile import agent_state, record_heartbeat, worker_state
from flock.controller.store import Row, Store, agent_tokens
from flock.models import (
    AgentState,
    AgentStatus,
    Command,
    CommandBatch,
    EventBatch,
    Harness,
    PreflightReport,
    PromptAck,
    PromptCommand,
    PromptRequest,
    ProviderStatus,
    SwarmSpec,
    SwarmStatus,
    TypeHealth,
    TypeStatus,
    WorkerHeartbeat,
    WorkerState,
    WorkerStatus,
)

MAX_WAIT = 30.0
_COMMAND = TypeAdapter(Annotated[Command, Field(discriminator="kind")])


def _now() -> datetime:
    return datetime.now(UTC)


def _dt(ts: float | None) -> datetime | None:
    return None if ts is None else datetime.fromtimestamp(ts, UTC)


class Notifier:
    """Wake-ups for long-polls. `notify` is safe to call from any thread."""

    def __init__(self) -> None:
        self._events: dict[str, asyncio.Event] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def _fire(self, key: str) -> None:
        ev = self._events.pop(key, None)
        if ev is not None:
            ev.set()

    def notify(self, key: str) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            same = asyncio.get_running_loop() is loop
        except RuntimeError:
            same = False
        if same:
            self._fire(key)
        else:
            loop.call_soon_threadsafe(self._fire, key)

    async def wait(self, key: str, timeout: float) -> None:
        self._loop = asyncio.get_running_loop()
        ev = self._events.setdefault(key, asyncio.Event())
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except TimeoutError:
            pass

    async def long_poll(self, key: str, check: Callable[[], Any], wait: float) -> Any:
        """Return check()'s first truthy result, or its last result after `wait` seconds."""
        self._loop = asyncio.get_running_loop()
        deadline = self._loop.time() + max(0.0, min(wait, MAX_WAIT))
        while True:
            ev = self._events.setdefault(key, asyncio.Event())  # grab before checking: no lost wake
            result = check()
            remaining = deadline - self._loop.time()
            if result or remaining <= 0:
                return result
            try:
                await asyncio.wait_for(ev.wait(), remaining)
            except TimeoutError:
                pass


class RequestStats:
    """Pure ASGI middleware counting requests per route template and status, with latency
    (sum and max) and in-flight peak, for /healthz. Long-polls count their parked time."""

    def __init__(self, app) -> None:
        self.app = app
        self.lock = threading.Lock()
        self.routes: dict[str, dict[str, float]] = {}
        self.statuses: Counter[str] = Counter()
        self.in_flight = 0
        self.peak_in_flight = 0
        self.started = time.time()

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        status = {"code": 500}

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                status["code"] = message["status"]
            await send(message)

        t0 = time.monotonic()
        with self.lock:
            self.in_flight += 1
            self.peak_in_flight = max(self.peak_in_flight, self.in_flight)
        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            dt = time.monotonic() - t0
            route = scope.get("route")
            key = f"{scope['method']} {getattr(route, 'path', 'unmatched')}"
            with self.lock:
                self.in_flight -= 1
                r = self.routes.setdefault(key, {"n": 0, "sum_s": 0.0, "max_s": 0.0, "errors": 0})
                r["n"] += 1
                r["sum_s"] += dt
                r["max_s"] = max(r["max_s"], dt)
                if status["code"] >= 500:
                    r["errors"] += 1
                self.statuses[f"{status['code'] // 100}xx"] += 1

    def snapshot(self) -> dict:
        with self.lock:
            routes = {
                k: {"n": int(v["n"]), "mean_s": round(v["sum_s"] / v["n"], 4),
                    "max_s": round(v["max_s"], 3), "5xx": int(v["errors"])}
                for k, v in sorted(self.routes.items())
            }
            return {"since": self.started, "total": sum(self.statuses.values()),
                    "by_status": dict(self.statuses), "in_flight": self.in_flight,
                    "peak_in_flight": self.peak_in_flight, "routes": routes}


def process_stats() -> dict:
    """RSS and CPU seconds of the controller process (Spaces expose no metrics API)."""
    try:
        with open("/proc/self/statm") as f:
            rss = int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        rss = 0
    t = os.times()
    return {"rss_mb": round(rss / 2**20, 1),
            "max_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "cpu_s": round(t.user + t.system, 2), "threads": threading.active_count(),
            "time": round(time.time(), 1)}


# --------------------------------------------------------------------------- status builders


def agent_status(a: Row, w: Row | None, spec: SwarmSpec | None, pending: int, now: float):
    ts = spec.types.get(a["type"]) if spec else None
    return AgentStatus(
        name=a["name"],
        swarm=a["swarm"],
        type=a["type"],
        harness=ts.harness if ts else Harness.claude,
        model=ts.model if ts else "unknown",
        provider=ts.provider_name if ts else None,
        labels=ts.labels if ts else {},
        state=agent_state(a, w, now),
        idle_since=_dt(a["idle_since"]),
        turn_id=a["turn_id"],
        worker=a["worker"],
        session_id=a["session_id"],
        resumed=None if a["resumed"] is None else bool(a["resumed"]),
        pending=pending,
        tokens=agent_tokens(a),
        cost_usd=a["cost"],
        harness_estimate_usd=a["harness_estimate_usd"],
        turns=a["turns"],
        created_at=_dt(a["created_at"]),
        last_heartbeat=_dt(a["last_heartbeat"]),
        last_sync=_dt(a["last_sync"]),
        msgboard_principal=a["msgboard_principal"],
    )


def worker_status(w: Row, agents: list[str], now: float) -> WorkerStatus:
    return WorkerStatus(
        id=w["id"],
        swarm=w["swarm"],
        type=w["type"],
        job_id=w["job_id"],
        state=worker_state(w, now),
        deadline=_dt(w["deadline"]),
        agents=agents,
        last_heartbeat=_dt(w["last_heartbeat"]),
        launched_at=_dt(w["launched_at"] or w["created_at"]),
        crashes=w["crashes"],
    )


def swarm_status(store: Store, spec: SwarmSpec, now: float) -> SwarmStatus:
    workers = {w["id"]: w for w in store.list_workers(spec.swarm)}
    agents = store.list_agents(spec.swarm)
    pending = store.pending_counts(spec.swarm)
    types: dict[str, TypeStatus] = {}
    for type_name in sorted(set(spec.types) | set(store.type_names(spec.swarm))):
        ts = spec.types.get(type_name)
        mine = [a for a in agents if a["type"] == type_name]
        by_state = Counter(agent_state(a, workers.get(a["worker"]), now).value for a in mine)
        w_state = Counter(
            worker_state(w, now).value for w in workers.values() if w["type"] == type_name
        )
        types[type_name] = TypeStatus(
            name=type_name,
            health=TypeHealth(store.type_health(spec.swarm, type_name)),
            replicas=ts.replicas if ts else 0,
            agents_by_state=dict(by_state),
            workers_by_state=dict(w_state),
            cost_usd=round(sum(a["cost"] for a in mine), 6),
        )
    providers: dict[str, ProviderStatus] = {}
    for name, prov in spec.providers.items():
        users = [t for t, ts in spec.types.items() if ts.provider_name == name]
        providers[name] = ProviderStatus(
            name=name,
            type=prov.type,
            self_hosted=prov.self_hosted,
            types=users,
            working=sum(types[t].agents_by_state.get("working", 0) for t in users if t in types),
            max_working=prov.max_working,
        )
    return SwarmStatus(
        swarm=spec.swarm,
        namespace=spec.namespace,
        bucket=spec.bucket,
        types=types,
        providers=providers,
        agents_total=len(agents),
        workers_total=len(workers),
        pending_prompts=sum(pending.values()),
        cost_usd=round(sum(t.cost_usd for t in types.values()), 6),
    )


_AGENT_STATE_TONE = {
    "working": "good",
    "idle": "neutral",
    "stuck": "warn",
    "unknown": "serious",
    "dead": "crit",
}
_WORKER_STATE_TONE = {
    "launching": "accent",
    "running": "good",
    "draining": "warn",
    "unknown": "serious",
    "dead": "crit",
}
_STATE_ORDER = ["working", "idle", "stuck", "unknown", "dead", "launching", "running", "draining"]

_PAGE_CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#1b1b1a;--ink2:#5c5b57;--ink3:#8a8984;--line:#e6e5e1;
--accent:#2f6fde;--good:#0ca30c;--warn:#fab219;--serious:#ec835a;--crit:#d03b3b;--neutral:#9a9994;
--mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
--sans:system-ui,-apple-system,"Segoe UI",Roboto,Inter,sans-serif}
@media(prefers-color-scheme:dark){:root{--bg:#1a1a19;--card:#232321;--ink:#f1f0ec;--ink2:#b3b2ac;
--ink3:#85847f;--line:#34342f;--accent:#7ea6ff}}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 var(--sans)}
a{color:var(--accent);text-decoration:none}a:hover{text-decoration:underline}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px 48px}
header{display:flex;flex-wrap:wrap;align-items:baseline;gap:8px 16px;margin-bottom:24px}
header h1{font-size:18px;font-weight:650;letter-spacing:-.01em;margin:0}
header .meta{color:var(--ink3);font-size:13px;margin-left:auto}
.swarm{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:20px;margin-bottom:20px}
.swarm h2{font-size:17px;font-weight:650;margin:0;letter-spacing:-.01em}
.kicker{color:var(--ink3);font-size:12px;text-transform:uppercase;letter-spacing:.06em}
.head{display:flex;flex-wrap:wrap;align-items:center;gap:8px 12px}
.sub{color:var(--ink2);font-size:13px;margin:4px 0 16px;word-break:break-all}
.sub code{font-family:var(--mono);font-size:12px;color:var(--ink2)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin-bottom:18px}
.stat{border:1px solid var(--line);border-radius:10px;padding:12px 14px;background:var(--bg)}
.stat .v{font-size:26px;font-weight:600;line-height:1.1;letter-spacing:-.02em}
.stat .l{color:var(--ink2);font-size:12.5px;margin-top:4px}
.tbl{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;color:var(--ink3);font-weight:500;font-size:12px;text-transform:uppercase;
letter-spacing:.04em;padding:6px 10px;border-bottom:1px solid var(--line)}
td{padding:9px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:0}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}th.num{text-align:right}
td.name{font-weight:600}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{display:inline-flex;align-items:center;gap:6px;border:1px solid var(--line);border-radius:999px;
padding:1px 9px 1px 7px;font-size:12.5px;color:var(--ink2);background:var(--bg);white-space:nowrap}
.chip b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
.chip i,.badge i{width:8px;height:8px;border-radius:50%;background:var(--neutral);display:inline-block}
.good i{background:var(--good)}.warn i{background:var(--warn)}.serious i{background:var(--serious)}
.crit i{background:var(--crit)}.accent i{background:var(--accent)}
.badge{display:inline-flex;align-items:center;gap:6px;font-size:12.5px;font-weight:600;
border-radius:999px;padding:1px 9px 1px 7px;border:1px solid var(--line)}
.badge.crit{color:var(--crit);border-color:var(--crit)}
.badge.warn{color:var(--ink);border-color:var(--warn)}
details{margin-top:12px;border-top:1px solid var(--line);padding-top:10px}
summary{cursor:pointer;color:var(--ink2);font-size:13px;user-select:none}
summary:hover{color:var(--ink)}
.agents{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}
.agents a{font-family:var(--mono);font-size:12px;border:1px solid var(--line);border-radius:6px;
padding:2px 7px;color:var(--ink2);background:var(--bg)}
.agents a:hover{color:var(--accent);border-color:var(--accent);text-decoration:none}
.more{color:var(--ink3);font-size:12px;align-self:center}
.empty{border:1px dashed var(--line);border-radius:12px;padding:40px 20px;text-align:center;color:var(--ink2)}
.empty code{font-family:var(--mono);font-size:13px;color:var(--ink)}
footer{color:var(--ink3);font-size:12px;margin-top:24px}
"""


def _fmt_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n / 1_000:.1f}K"
    return f"{n:,}"


def _chips(counts: dict[str, int], tones: dict[str, str]) -> str:
    e = html.escape
    keys = sorted(counts, key=lambda s: (_STATE_ORDER.index(s) if s in _STATE_ORDER else 99, s))
    chips = [
        f'<span class="chip {tones.get(s, "neutral")}"><i></i>{e(s)} <b>{counts[s]}</b></span>'
        for s in keys
        if counts[s]
    ]
    return (
        f'<div class="chips">{"".join(chips)}</div>' if chips else '<span class="more">none</span>'
    )


def _status_page(store: Store, now: float) -> str:
    e = html.escape
    parts = []
    for sw in store.list_swarms():
        spec: SwarmSpec = sw["spec"]
        st = swarm_status(store, spec, now)
        removing = '<span class="badge warn"><i></i>removing</span>' if sw["removing"] else ""
        stats = [
            (_fmt_count(st.agents_total), "agents"),
            (_fmt_count(st.workers_total), "workers"),
            (_fmt_count(st.pending_prompts), "pending prompts"),
            (f"${st.cost_usd:,.2f}", "cost"),
        ]
        tiles = "".join(
            f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>'
            for v, l in stats
        )
        rows = []
        for t in st.types.values():
            health = (
                '<span class="badge good"><i></i>ok</span>'
                if t.health.value == "ok"
                else f'<span class="badge crit"><i></i>{e(t.health.value)}</span>'
            )
            ts = spec.types.get(t.name)
            model = (
                f"<code>{e(ts.resolved_model.qualified)}</code>"
                if ts and ts.resolved_model
                else (f"<code>{e(ts.model)}</code>" if ts else "")
            )
            rows.append(
                f'<tr><td class="name">{e(t.name)}</td><td>{health}</td><td>{model}</td>'
                f'<td class="num">{t.replicas:,}</td>'
                f"<td>{_chips(t.agents_by_state, _AGENT_STATE_TONE)}</td>"
                f"<td>{_chips(t.workers_by_state, _WORKER_STATE_TONE)}</td>"
                f'<td class="num">${t.cost_usd:,.2f}</td></tr>'
            )
        table = (
            '<div class="tbl"><table><thead><tr><th>agent type</th><th>health</th><th>model</th>'
            '<th class="num">replicas</th><th>agents</th><th>workers</th><th class="num">cost</th>'
            f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
        )
        details = []
        for type_name in st.types:
            names = [a["name"] for a in store.list_agents(spec.swarm, type_name)]
            links = "".join(f'<a href="/agents/{e(n)}">{e(n)}</a>' for n in names[:200])
            more = (
                f'<span class="more">and {len(names) - 200:,} more</span>'
                if len(names) > 200
                else ""
            )
            details.append(
                f"<details><summary>{e(type_name)} · {len(names):,} "
                f"{'agent' if len(names) == 1 else 'agents'}</summary>"
                f'<div class="agents">{links}{more}</div></details>'
            )
        provs = " · ".join(
            f"<code>{e(pv.name)}</code> {e(pv.type)}{' self-hosted' if pv.self_hosted else ''}, "
            f"working {pv.working}" + (f" / max {pv.max_working}" if pv.max_working else "")
            for pv in st.providers.values()
        )
        provs = f'<p class="sub">providers: {provs}</p>' if provs else ""
        parts.append(
            f'<section class="swarm"><div class="head"><span class="kicker">swarm</span>'
            f"<h2>{e(spec.swarm)}</h2>{removing}</div>"
            f'<p class="sub">namespace <code>{e(spec.namespace)}</code> · bucket '
            f"<code>{e(spec.bucket)}</code></p>{provs}"
            f'<div class="stats">{tiles}</div>{table}{"".join(details)}</section>'
        )
    body = "".join(parts) or (
        '<div class="empty">No swarms yet.<br>Apply one with '
        "<code>flock apply spec.yaml</code></div>"
    )
    stamp = e(_dt(now).strftime("%Y-%m-%d %H:%M:%S UTC"))
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<meta name='color-scheme' content='light dark'><title>flock</title>"
        f"<meta http-equiv='refresh' content='30'><style>{_PAGE_CSS}</style></head>"
        '<body><div class="wrap"><header><h1>flock</h1>'
        f'<span class="meta">{stamp} · refreshes every 30 s</span></header>{body}'
        "<footer>Agent links return JSON and need the API bearer token.</footer>"
        "</div></body></html>"
    )


# --------------------------------------------------------------------------- app


def create_app(
    store: Store,
    token: str,
    background: Callable[[Notifier], Awaitable[None]] | None = None,
    clock: Callable[[], datetime] = _now,
    stats: Callable[[], dict] | None = None,
    secret_source: Mapping[str, str] | None = None,
    preflight_transport: httpx.AsyncBaseTransport | None = None,
) -> FastAPI:
    """`secret_source` holds provider keys for preflight (default: the process environment,
    where the Space puts its secrets); `preflight_transport` lets tests fake the providers."""
    secret_source = os.environ if secret_source is None else secret_source
    notifier = Notifier()
    store.listener = lambda kind, key: notifier.notify(f"{kind}:{key}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        notifier._loop = asyncio.get_running_loop()
        task = asyncio.create_task(background(notifier)) if background else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="flock controller", lifespan=lifespan)
    app.state.store, app.state.notifier = store, notifier
    request_stats: dict[str, RequestStats] = {}

    def make_stats(app):
        request_stats["mw"] = RequestStats(app)
        return request_stats["mw"]

    app.add_middleware(make_stats)

    def auth(authorization: str = Header(default="")) -> None:
        if not token or not secrets.compare_digest(authorization, f"Bearer {token}"):
            raise HTTPException(401, "bad or missing bearer token")

    api = APIRouter(dependencies=[Depends(auth)])

    def spec_or_404(swarm: str) -> Row:
        row = store.get_swarm(swarm)
        if row is None:
            raise HTTPException(404, f"no swarm {swarm!r}")
        return row

    def agent_or_404(name: str) -> Row:
        a = store.get_agent(name)
        if a is None:
            raise HTTPException(404, f"no agent {name!r}")
        return a

    @app.get("/healthz")
    def healthz() -> dict:
        mw = request_stats.get("mw")
        return {"ok": True, **(stats() if stats else {}), "process": process_stats(),
                "requests": mw.snapshot() if mw else {}}

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return _status_page(store, clock().timestamp())

    # ------------------------------------------------------------- swarms

    @api.get("/swarms")
    def list_swarms() -> list[SwarmStatus]:
        now = clock().timestamp()
        return [swarm_status(store, sw["spec"], now) for sw in store.list_swarms()]

    @api.put("/swarms/{swarm}")
    def apply(swarm: str, spec: SwarmSpec) -> SwarmStatus:
        if spec.swarm != swarm:
            raise HTTPException(400, "spec.swarm does not match the path")
        for type_name in spec.types:
            owner = store.type_owner(type_name)
            if owner not in (None, swarm):
                raise HTTPException(409, f"type {type_name!r} already belongs to swarm {owner!r}")
        store.put_swarm(spec, clock().timestamp())
        return swarm_status(store, spec, clock().timestamp())

    @api.post("/swarms/{swarm}/preflight")
    async def preflight(
        swarm: str, spec: Annotated[SwarmSpec | None, Body()] = None
    ) -> PreflightReport:
        """Probe every provider-backed model in `spec` (or the applied spec when no body is
        sent) with one small tool-calling request. Nothing is stored."""
        if spec is None:
            spec = spec_or_404(swarm)["spec"]
        elif spec.swarm != swarm:
            raise HTTPException(400, "spec.swarm does not match the path")
        async with httpx.AsyncClient(transport=preflight_transport) as http:
            return await run_preflight(spec, secret_source, http)

    @api.get("/swarms/{swarm}")
    def get_swarm(swarm: str) -> dict:
        row = spec_or_404(swarm)
        status = swarm_status(store, row["spec"], clock().timestamp())
        return {
            "spec": row["spec"].model_dump(mode="json"),
            "status": status.model_dump(mode="json"),
            "removing": bool(row["removing"]),
        }

    @api.delete("/swarms/{swarm}", status_code=202)
    def remove(swarm: str) -> dict:
        spec_or_404(swarm)
        store.set_removing(swarm)
        return {"swarm": swarm, "removing": True}

    @api.get("/swarms/{swarm}/agents")
    def list_agents(
        swarm: str,
        type: str | None = None,
        state: AgentState | None = None,
        cursor: str | None = None,
        limit: int = 500,
    ) -> dict:
        spec = spec_or_404(swarm)["spec"]
        now = clock().timestamp()
        workers = {w["id"]: w for w in store.list_workers(swarm)}
        pending = store.pending_counts(swarm)
        out: list[AgentStatus] = []
        rows = sorted(store.list_agents(swarm, type), key=lambda a: a["name"])
        for a in rows:
            if cursor is not None and a["name"] <= cursor:
                continue
            st = agent_status(a, workers.get(a["worker"]), spec, pending.get(a["name"], 0), now)
            if state is None or st.state == state:
                out.append(st)
            if len(out) >= max(1, limit):
                break
        more = len(out) >= max(1, limit) and out[-1].name != rows[-1]["name"]
        return {
            "agents": [x.model_dump(mode="json") for x in out],
            "cursor": out[-1].name if more else None,
        }

    # ------------------------------------------------------------- agents

    @api.get("/agents/{name}")
    def get_agent(name: str) -> AgentStatus:
        a = agent_or_404(name)
        row = store.get_swarm(a["swarm"])
        w = store.get_worker(a["worker"]) if a["worker"] else None
        pending = store.pending_counts(a["swarm"]).get(name, 0)
        return agent_status(a, w, row["spec"] if row else None, pending, clock().timestamp())

    @api.post("/agents/{name}/prompt")
    def prompt(name: str, req: PromptRequest) -> PromptAck:
        a = agent_or_404(name)
        cmd = PromptCommand(
            agent=name,
            prompt_id=uuid.uuid4().hex,
            text=req.text,
            timeout=req.timeout,
            reply_to=req.reply_to,
        )
        now = clock().timestamp()
        store.add_prompt(name, cmd.model_dump_json(), now)
        w = store.get_worker(a["worker"]) if a["worker"] else None
        live = w is not None and worker_state(w, now) == WorkerState.running
        return PromptAck(
            prompt_id=cmd.prompt_id, agent=name, status="queued" if live else "pending"
        )

    @api.get("/agents/{name}/events")
    async def events(name: str, after: int = 0, wait: float = 0, limit: int = 1000) -> dict:
        if store.get_agent(name) is None and store.event_count(name) == 0:
            raise HTTPException(404, f"no agent {name!r}")
        after = max(0, after)
        await notifier.long_poll(f"events:{name}", lambda: store.event_count(name) > after, wait)
        evs = store.read_events(name, after, limit=max(1, min(limit, 5000)))
        return {"events": evs, "next": after + len(evs)}

    # ------------------------------------------------------------- workers

    @api.get("/workers")
    def list_workers(swarm: str | None = None, type: str | None = None) -> list[WorkerStatus]:
        now = clock().timestamp()
        by_worker: dict[str, list[str]] = {}
        for a in store.list_agents(swarm, type):
            if a["worker"]:
                by_worker.setdefault(a["worker"], []).append(a["name"])
        return [
            worker_status(w, by_worker.get(w["id"], []), now)
            for w in store.list_workers(swarm, type)
        ]

    def worker_or_404(worker_id: str) -> Row:
        w = store.get_worker(worker_id)
        if w is None:
            raise HTTPException(404, f"no worker {worker_id!r}")
        return w

    @api.get("/workers/{worker_id}")
    def get_worker(worker_id: str) -> WorkerStatus:
        w = worker_or_404(worker_id)
        names = [a["name"] for a in store.list_agents(worker=worker_id)]
        return worker_status(w, names, clock().timestamp())

    def stale(w: Row, incarnation: str | None) -> bool:
        """The caller is not the slot's current job. Workers that send no incarnation (older
        versions) are trusted, as before."""
        return incarnation is not None and w["incarnation"] != incarnation

    @api.get("/workers/{worker_id}/commands")
    async def commands(worker_id: str, wait: float = 25, incarnation: str | None = None):
        if stale(worker_or_404(worker_id), incarnation):
            raise HTTPException(409, "worker incarnation is not the slot's current job")

        def check():
            w = store.get_worker(worker_id)
            if w is None or stale(w, incarnation):
                return "stale"  # replaced while this poll was parked: never hand it commands
            if w["job_id"] is None:
                return None  # slot has no live job (stale process): give it nothing
            cmds = store.commands_for(worker_id)
            # A draining worker starts no turns; its prompts wait for the replacement job.
            draining = w["draining"] or w["retire_at"]
            prompts = [] if draining else store.prompts_for_worker(worker_id)
            return (cmds, prompts) if cmds or prompts else None

        found = await notifier.long_poll(f"commands:{worker_id}", check, wait)
        if found == "stale":
            raise HTTPException(409, "worker incarnation is not the slot's current job")
        if not found:
            return CommandBatch(commands=[])
        cmds, prompts = found
        batch = CommandBatch(
            commands=[_COMMAND.validate_json(c["body"]) for c in cmds]
            + [PromptCommand.model_validate_json(p["body"]) for p in prompts]
        )

        def ack() -> None:  # at-least-once: forget them only after the response went out
            store.delete_commands([c["id"] for c in cmds])
            store.delete_prompts([p["id"] for p in prompts])

        return JSONResponse(batch.model_dump(mode="json"), background=BackgroundTask(ack))

    @api.post("/workers/{worker_id}/heartbeat")
    def heartbeat(worker_id: str, hb: WorkerHeartbeat, incarnation: str | None = None) -> dict:
        if stale(worker_or_404(worker_id), incarnation):
            raise HTTPException(409, "worker incarnation is not the slot's current job")
        if hb.worker != worker_id:
            raise HTTPException(400, "heartbeat worker does not match the path")
        if not record_heartbeat(store, worker_id, hb, clock()):
            raise HTTPException(409, "worker slot has no live job")
        return {"ok": True}

    @api.post("/workers/{worker_id}/events")
    def post_events(worker_id: str, batch: EventBatch) -> dict:
        w = worker_or_404(worker_id)
        grouped: dict[str, list] = {}
        for ev in batch.events:
            grouped.setdefault(ev.agent, []).append(ev)
        for agent, evs in grouped.items():
            a = store.get_agent(agent)
            store.append_events(a["swarm"] if a else w["swarm"], agent, evs)
        return {"ok": True, "accepted": len(batch.events)}

    app.include_router(api)
    return app
