"""FastAPI app: the routes in design.md "API (controller)" plus /healthz and GET /swarms."""

from __future__ import annotations

import asyncio
import html
import logging
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
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import Field, TypeAdapter
from starlette.background import BackgroundTask
from starlette.requests import Request

from flock.comms.managed import ManagedBoard, ManagedBoardError
from flock.controller.comms import CommsError
from flock.controller.comms import resolve as comms_resolve
from flock.controller.loop import LoopEngine, loop_status, swarm_stream
from flock.controller.preflight import run_preflight
from flock.controller.reconcile import agent_state, record_heartbeat, worker_state
from flock.controller.spawn import (
    Refused,
    SpawnAck,
    SpawnRequest,
    agent_depth,
    agent_for_token,
    agent_may,
    remove_subtree,
    set_token_key,
    worker_cap_conflicts,
)
from flock.controller.spawn import spawn as spawn_child
from flock.controller.store import Row, Store, agent_tokens
from flock.models import (
    COMMS_TOOLS,
    AgentState,
    AgentStatus,
    BulkPromptAck,
    BulkPromptRequest,
    ChildrenPromptRequest,
    Command,
    CommandBatch,
    Event,
    EventBatch,
    Grant,
    Harness,
    LoopRequest,
    LoopStatus,
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
    comms_tools,
    loop_refusals,
    merge_grants,
    type_spec_rev,
    worker_type_spec,
    world_tools,
)
from flock.tools.bundle import MAX_BUNDLE_BYTES, read_manifest, version_of
from flock.tools.loader import ToolsError
from flock.tools.spec import comms_clashes
from flock.world.host import WorldError, WorldHost
from flock.world.host import resolve as world_resolve

log = logging.getLogger(__name__)
# Worker events the comms wake hook receives, by reason (docs/comms.md, "Wake events").
COMMS_WAKE_EVENTS = {"comms_mention": "mention", "comms_thread_reply": "thread",
                    "comms_dm": "dm", "comms_channel_message": "channel"}

MAX_WAIT = 30.0
ADMIT_GRACE = 10.0  # s, a heartbeat this soon after an admission may predate the prompt
SEEN_EVERY = 5.0  # s, min gap between last_seen writes per worker
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


def agent_status(
    a: Row,
    w: Row | None,
    spec: SwarmSpec | None,
    pending: int,
    now: float,
    children: list[str] | None = None,
    grant: Grant | None = None,
):
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
        parent=a["parent"],
        children=children or [],
        depth=agent_depth(spec, a),
        grants=grant.compact() if grant is not None else None,
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
        last_error=w.get("last_error"),
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
            launch_error=store.launch_error(spec.swarm, type_name)[0],
            loop=loop_status(store, spec.swarm, type_name),
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
            zero_usage_agents=sum(
                1 for a in agents if a["type"] in users and a["turns"] > 0
                and not any(sum(u.model_dump().values()) for u in agent_tokens(a).values())),
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
        budget_stop=store.swarm_flags(spec.swarm).get("budget_stop"),
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
            if t.launch_error:
                health += f' <span class="meta">{e(t.launch_error[:200])}</span>'
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
            + (f", <strong>{pv.zero_usage_agents} agents report zero tokens (cost unknown)</strong>"
               if pv.zero_usage_agents else "")
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
    enforce_max_working: bool | None = None,
    namespace: str | None = None,
    compute: str | None = None,
    loop_interval: float | None = None,
    comms_transport: httpx.BaseTransport | None = None,
    on_comms: Callable[[str, str, dict], None] | None = None,
    worlds: WorldHost | None = None,
    world_transport: httpx.BaseTransport | None = None,
    managed_board: ManagedBoard | None = None,
) -> FastAPI:
    """`secret_source` holds provider keys for preflight (default: the process environment,
    where the Space puts its secrets); `preflight_transport` lets tests fake the providers.
    `enforce_max_working` defaults to FLOCK_MAX_WORKING_ENFORCE (on unless "0").
    `namespace`, when set (job mode), is the only namespace a spec may launch workers in: a
    network group spans one namespace, so workers elsewhere could never reach the controller.
    `compute`, when set ("hf-jobs", "local", "slurm" or "external"), is the only `compute.kind` a spec may ask for:
    this controller's jobs client launches workers one way (docs/local.md).
    `loop_interval`: seconds between loop ticks (controller/loop.py) while the app runs;
    default FLOCK_LOOP_INTERVAL or 1; 0 turns the ticker off (tests drive `app.state.loops`).
    `comms_transport` lets tests fake the message board (docs/comms.md). Every `comms_mention`
    (reason "mention") and `comms_thread_reply` (reason "thread") event a worker posts goes to
    the loop's wake queue as `wake(agent, "comms.<reason>")` (honoured when the type's
    `loop.on` lists it) and then to `on_comms(agent, reason, data)`, an optional observer.
    `worlds` starts the `command` worlds of `compute: local` specs (docs/world.md; default: a
    WorldHost logging to the controller's output); they stop with the app or `remove`.
    `world_transport` lets tests fake a `url` world. `managed_board` is the private board
    owned by flock run; applies and updates reuse it. Other launch paths refuse managed comms."""
    secret_source = os.environ if secret_source is None else secret_source
    worlds = WorldHost() if worlds is None else worlds
    if loop_interval is None:
        loop_interval = float(os.environ.get("FLOCK_LOOP_INTERVAL", "1"))
    if enforce_max_working is None:
        enforce_max_working = os.environ.get("FLOCK_MAX_WORKING_ENFORCE", "1") != "0"
    # Admission overlay (max_working): what the controller learned about an agent's turn since
    # its last heartbeat. agent -> (state, when): "admitted" (handed a prompt to start a turn),
    # "working" (turn_started seen) or "idle" (turn_ended / turn_interrupted / prompt_deferred
    # seen). A heartbeat that reports the agent replaces it (see `heartbeat_seen`).
    live: dict[str, tuple[str, float]] = {}
    notifier = Notifier()
    store.listener = lambda kind, key: notifier.notify(f"{kind}:{key}")
    set_token_key(store, token)

    def loop_send(swarm: str, type_name: str, agents: list[Row], build) -> list[str]:
        """The loop engine's prompts: the bulk-prompt path with state=idle, skip_pending."""
        req = BulkPromptRequest(type=type_name, state=AgentState.idle, text="", skip_pending=True)
        ack = BulkPromptAck(swarm=swarm, type=type_name, matched=0, queued=0, pending=0)
        sent: list[str] = []
        prompt_many(swarm, agents, req, ack, build=build, sent=sent)
        return sent

    loops = LoopEngine(store, send=loop_send, clock=clock,
                       capped=lambda spec, t: provider_cap(spec, t) is not None,
                       turn_state=lambda a, w, now: turn_state(a, w, now))

    async def loop_ticker() -> None:
        while True:
            try:
                await asyncio.to_thread(loops.tick)
            except Exception:
                logging.getLogger("flock.loop").exception("loop tick failed")
            await asyncio.sleep(loop_interval)

    def comms_event(agent: str, ev: Event) -> None:
        data = ev.data.get("data", ev.data) if isinstance(ev.data, dict) else {}
        reason = COMMS_WAKE_EVENTS[ev.type]
        try:  # `loop.notices`: named in the agent's next round prompt
            loops.notice(agent, reason, data)
        except Exception:
            log.exception("comms notice failed for %s", agent)
        try:
            woke = loops.wake(agent, f"comms.{reason}")
            log.debug("comms %s for %s (wake %s): %s", reason, agent, woke, data)
            if on_comms is not None:
                on_comms(agent, reason, {"agent": agent, **data})
        except Exception:  # a hook never breaks event ingestion
            log.exception("comms hook failed for %s", agent)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        notifier._loop = asyncio.get_running_loop()
        task = asyncio.create_task(background(notifier)) if background else None
        ticker = asyncio.create_task(loop_ticker()) if loop_interval and loop_interval > 0 else None
        yield
        for t in (task, ticker):
            if t:
                t.cancel()
        await asyncio.to_thread(worlds.stop_all)  # started worlds end with the controller

    app = FastAPI(title="flock controller", lifespan=lifespan)
    app.state.store, app.state.notifier, app.state.loops = store, notifier, loops
    request_stats: dict[str, RequestStats] = {}

    def make_stats(app):
        request_stats["mw"] = RequestStats(app)
        return request_stats["mw"]

    app.add_middleware(make_stats)

    def auth(
        request: Request,
        authorization: str = Header(default=""),
        x_flock_token: str = Header(default=""),
    ) -> None:
        """The controller token has full access. An agent token (docs/spawn.md) acts as that
        agent, scoped to itself, its subtree and prompting its parent. Either token comes in
        `X-Flock-Token` or as `Authorization: Bearer`; behind the HF Jobs proxy (job mode,
        docs/deploy-job.md) `Authorization` carries the caller's HF token for the proxy, so
        the flock token must use `X-Flock-Token` there. The header is checked first."""
        bearer = authorization.removeprefix("Bearer ")
        candidates = [t for t in (x_flock_token, bearer if bearer != authorization else "") if t]
        if token and any(secrets.compare_digest(c, token) for c in candidates):
            return
        agent = next((a for c in candidates if (a := agent_for_token(store, c))), None)
        if agent is None:
            raise HTTPException(401, "bad or missing flock token (X-Flock-Token or bearer)")
        route = getattr(request.scope.get("route"), "path", "")
        if not agent_may(
            store, agent, request.method, route, request.path_params, request.query_params
        ):
            raise HTTPException(403, f"agent {agent['name']} may not call {request.method} {route}")
        request.state.agent = agent

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
        if namespace and spec.namespace != namespace:
            raise HTTPException(
                400,
                f"this controller runs as a job in namespace {namespace!r} and its workers must "
                f"join its network group there; spec.namespace is {spec.namespace!r}",
            )
        if compute and spec.compute.kind != compute:
            raise HTTPException(
                400,
                f"this controller launches workers with compute {compute!r}; the spec asks for "
                f"compute {spec.compute.kind!r} (docs/local.md: `flock run` for local, a "
                "deployed controller and `flock apply` for hf-jobs)",
            )
        for type_name in spec.types:
            owner = store.type_owner(type_name)
            if owner not in (None, swarm):
                raise HTTPException(409, f"type {type_name!r} already belongs to swarm {owner!r}")
        if conflicts := worker_cap_conflicts(store, spec):
            raise HTTPException(409, "; ".join(conflicts))
        check_tools(swarm, spec)
        if refused := loop_refusals(spec):
            raise HTTPException(400, "; ".join(refused))
        prev = store.get_swarm(swarm)
        for type_name, ts in spec.types.items():  # docs/live.md: what a live change cannot do
            old = prev["spec"].types.get(type_name) if prev else None
            if old is not None and old.harness != ts.harness and store.list_agents(swarm, type_name):
                raise HTTPException(400, f"types.{type_name}.harness: a running type's harness "
                                         f"cannot change ({old.harness.value} -> "
                                         f"{ts.harness.value}); remove the type first or use a "
                                         "new type name")
        # Only flock run owns managed boards. Updates bind back to the same private service.
        try:
            if managed_board is not None:
                managed_board.resolve(spec)
            elif spec.comms is not None and spec.comms.managed:
                raise ManagedBoardError("comms.managed is supported by flock run only")
        except ManagedBoardError as e:
            raise HTTPException(400, str(e)) from None
        try:  # channel name -> id once, kickoff once (docs/comms.md)
            comms_resolve(spec, prev["spec"] if prev else None, secret_source,
                          transport=comms_transport)
        except CommsError as e:
            raise HTTPException(400, str(e)) from None
        try:  # start or reach the world, fetch and check its manifest (docs/world.md)
            world_resolve(spec, host=worlds, controller_token=token, secrets=secret_source,
                          folder=lambda t: folder_names(swarm, spec, t),
                          transport=world_transport)
        except WorldError as e:
            raise HTTPException(400, str(e)) from None
        store.put_swarm(spec, clock().timestamp())
        worlds.commit(swarm)  # a replaced world stops only now, after every check passed
        if spec.world is None or spec.world.endpoint != worlds.url(swarm):
            worlds.stop(swarm)  # the block went away or now names a url
        return swarm_status(store, spec, clock().timestamp())

    def folder_names(swarm: str, spec: SwarmSpec, type_name: str) -> list[str]:
        sel = spec.types[type_name].folder_tools
        if sel != "all":
            return list(sel or [])
        b = store.get_tool_bundle(swarm, spec.tools_version, body=False) if spec.tools_version else None
        return list(b["tools"]) if b else []

    def check_tools(swarm: str, spec: SwarmSpec) -> None:
        """The spec's tools bundle was uploaded and has every tool a type selects."""
        selecting = {n: ts.folder_tools for n, ts in spec.types.items() if ts.folder_tools is not None}
        if spec.tools_version is None:
            if selecting:
                raise HTTPException(400, f"types {', '.join(selecting)} select tools but the spec "
                                         "has no tools_version: apply it with `flock apply`, "
                                         "which uploads the tools folder (docs/tools.md)")
            return
        b = store.get_tool_bundle(swarm, spec.tools_version, body=False)
        if b is None:
            raise HTTPException(400, f"tools bundle {spec.tools_version} was not uploaded to "
                                     f"swarm {swarm!r} (POST /swarms/{swarm}/tools first)")
        for name, sel in selecting.items():
            unknown = [] if sel == "all" else [t for t in sel if t not in b["tools"]]
            if unknown:
                raise HTTPException(400, f"types.{name}.tools: unknown tool(s) {', '.join(unknown)}; "
                                         f"bundle {spec.tools_version} has: "
                                         f"{', '.join(b['tools']) or 'none'}")
            clash = comms_clashes(spec.types[name], spec.comms, b["tools"] if sel == "all" else sel)
            if clash:
                raise HTTPException(400, f"types.{name}.tools: {clash}")

    @api.post("/swarms/{swarm}/tools", status_code=201)
    async def upload_tools(swarm: str, request: Request) -> dict:
        """Store a tools bundle (the tar.gz `flock apply` packs) under its content hash. The
        swarm need not exist yet; the spec then names the version (`tools_version`)."""
        body = await request.body()
        if len(body) > MAX_BUNDLE_BYTES:
            raise HTTPException(413, f"tools bundle over {MAX_BUNDLE_BYTES} bytes")
        try:
            manifest = read_manifest(body)
        except ToolsError as e:
            raise HTTPException(400, str(e)) from None
        version = version_of(body)
        store.put_tool_bundle(swarm, version, body, manifest.names(), clock().timestamp())
        return {"swarm": swarm, "version": version, "size": len(body), "tools": manifest.names()}

    @api.get("/swarms/{swarm}/tools")
    def list_tools(swarm: str) -> dict:
        row = store.get_swarm(swarm)
        return {"swarm": swarm, "current": row["spec"].tools_version if row else None,
                "bundles": store.list_tool_bundles(swarm)}

    @api.get("/swarms/{swarm}/tools/{version}")
    def get_tools(swarm: str, version: str) -> Response:
        b = store.get_tool_bundle(swarm, version)
        if b is None:
            raise HTTPException(404, f"no tools bundle {version!r} for swarm {swarm!r}")
        return Response(b["body"], media_type="application/gzip")

    # ------------------------------------------------------------- grants (docs/live.md)

    def agent_grant(a: Row) -> Grant | None:
        """The grant in force for agent row `a`: its type's, then its own on top."""
        types, agents = store.agent_grants(a["swarm"])
        return merge_grants(types.get(a["type"]), agents.get(a["name"]))

    def over_budget(a: Row, g: Grant | None) -> bool:
        return g is not None and g.limits.budget_usd is not None and a["cost"] >= g.limits.budget_usd

    def grant_event(swarm: str, kind: str, target: str, data: dict) -> None:
        ev = Event(type=kind, ts=clock(), agent="", data={"target": target, **data})
        store.append_events(swarm, swarm_stream(swarm), [ev])
        if data.get("kind") == "agent":
            store.append_events(swarm, target, [ev.model_copy(update={"agent": target})])

    def grant_target(swarm: str, spec: SwarmSpec, target: str) -> tuple[str, str]:
        """(kind, type name) of a grant target: a type of the swarm, or one of its agents."""
        if target in spec.types:
            return "type", target
        a = store.get_agent(target)
        if a is None or a["swarm"] != swarm:
            raise HTTPException(404, f"no type or agent {target!r} in swarm {swarm!r}")
        if a["type"] not in spec.types:
            raise HTTPException(409, f"agent {target!r}: its type {a['type']!r} is no longer in "
                                     "the spec")
        return "agent", a["type"]

    def check_grant(swarm: str, spec: SwarmSpec, type_name: str, g: Grant) -> Grant:
        """Refuse what the workers cannot enforce; comms tools come back as `comms:<name>`."""
        ts = spec.types[type_name]
        bundle = (store.get_tool_bundle(swarm, spec.tools_version, body=False)
                  if spec.tools_version else None)
        folder = bundle["tools"] if bundle else []
        on_board = comms_tools(ts, spec.comms)
        refused: list[str] = []
        tools: dict[str, bool] = {}
        for name, on in g.tools.items():
            entry = name if name.startswith("comms:") else f"comms:{name}"
            if name in folder:
                tools[name] = on
            elif entry in COMMS_TOOLS:
                tools[entry] = on
                if not on:
                    continue
                if spec.comms is None:
                    refused.append(f"tools: +{name}: the swarm has no comms block")
                elif not on_board:
                    refused.append(f"tools: +{name}: type {type_name} has no comms tools, so its "
                                   "workers hold no board invite; give it `comms: true` or a "
                                   "comms tool in the spec")
                elif entry == "comms:search" and not spec.comms.search:
                    refused.append(f"tools: +{name}: comms.search is off in the spec")
            else:
                where = (f"bundle {spec.tools_version} has: {', '.join(folder) or 'none'}"
                         if bundle else "the swarm has no tools folder")
                refused.append(f"tools: unknown tool {name!r}; {where}; comms tools: post, read, "
                               "search")
        if g.bash is not None and ts.harness != Harness.llm:
            refused.append(f"bash: only the llm harness has a bash tool to switch (type "
                           f"{type_name} is {ts.harness.value})")
        if g.bash is False and ts.tool_choice == "bash":
            refused.append(f"bash: off contradicts type {type_name}'s tool_choice: bash")
        if refused:
            raise HTTPException(400, "; ".join(refused))
        return g.model_copy(update={"tools": tools})

    @api.get("/swarms/{swarm}/grants")
    def list_grants(swarm: str) -> dict:
        spec_or_404(swarm)
        return {"swarm": swarm, "grants": [
            {"target": r["target"], "kind": r["kind"], "grant": r["grant"].compact(),
             "updated_at": _dt(r["updated_at"])} for r in store.list_grants(swarm)]}

    @api.put("/swarms/{swarm}/grants/{target}")
    def put_grant(swarm: str, target: str, grant: Grant) -> dict:
        """Set (replace) the grant of a type or an agent; an empty body revokes it. It is an
        overlay on the spec: `flock update` / `flock apply` keep it. Workers merge it in at
        each agent's next turn boundary."""
        spec = spec_or_404(swarm)["spec"]
        kind, type_name = grant_target(swarm, spec, target)
        grant = check_grant(swarm, spec, type_name, grant)
        store.put_grant(swarm, target, kind, grant, clock().timestamp())
        if grant.empty():
            grant_event(swarm, "grant_revoked", target, {"kind": kind})
        else:
            grant_event(swarm, "grant", target, {"kind": kind, "grant": grant.compact()})
        return {"swarm": swarm, "target": target, "kind": kind, "grant": grant.compact()}

    @api.delete("/swarms/{swarm}/grants/{target}")
    def delete_grant(swarm: str, target: str) -> dict:
        spec_or_404(swarm)
        row = next((r for r in store.list_grants(swarm) if r["target"] == target), None)
        if row is None:
            raise HTTPException(404, f"no grant for {target!r} in swarm {swarm!r}")
        store.put_grant(swarm, target, row["kind"], Grant(), clock().timestamp())
        grant_event(swarm, "grant_revoked", target, {"kind": row["kind"]})
        return {"swarm": swarm, "target": target, "kind": row["kind"], "revoked": True}

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
        worlds.stop(swarm)  # a world flock started ends with its swarm (docs/world.md)
        return {"swarm": swarm, "removing": True}

    @api.get("/swarms/{swarm}/agents")
    def list_agents(
        swarm: str,
        type: str | None = None,
        state: AgentState | None = None,
        cursor: str | None = None,
        limit: int = 500,
        parent: str | None = None,
    ) -> dict:
        spec = spec_or_404(swarm)["spec"]
        now = clock().timestamp()
        workers = {w["id"]: w for w in store.list_workers(swarm)}
        pending = store.pending_counts(swarm)
        kids = store.children_by_parent(swarm)
        type_grants, agent_grants = store.agent_grants(swarm)
        out: list[AgentStatus] = []
        rows = sorted(store.list_agents(swarm, type, parent=parent), key=lambda a: a["name"])
        for a in rows:
            if cursor is not None and a["name"] <= cursor:
                continue
            st = agent_status(a, workers.get(a["worker"]), spec, pending.get(a["name"], 0), now,
                              kids.get(a["name"]), merge_grants(type_grants.get(a["type"]),
                                                                agent_grants.get(a["name"])))
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
        kids = [c["name"] for c in store.list_agents(parent=name)]
        return agent_status(a, w, row["spec"] if row else None, pending, clock().timestamp(), kids,
                            agent_grant(a))

    @api.post("/agents/{name}/spawn", status_code=201)
    def spawn(name: str, req: SpawnRequest) -> SpawnAck:
        try:
            ack, created = spawn_child(store, name, req, clock().timestamp())
        except KeyError:
            raise HTTPException(404, f"no agent {name!r}") from None
        except Refused as r:
            return JSONResponse({"reason": r.reason, "detail": r.message}, status_code=409)
        if not created:
            return JSONResponse(ack.model_dump(mode="json"), status_code=200)
        return ack

    @api.delete("/agents/{name}")
    def remove_agent(name: str) -> dict:
        a = agent_or_404(name)
        if a["parent"] is None:
            return JSONResponse(
                {"reason": "not_spawned", "detail": "replicas are removed by scaling their type"},
                status_code=409,
            )
        removed = remove_subtree(store, name, "removed", clock().timestamp())
        store.listener("reconcile", "")
        return {"removed": [r["name"] for r in removed]}

    @api.get("/whoami")
    def whoami(request: Request) -> dict:
        a = getattr(request.state, "agent", None)
        if a is None:
            return {"kind": "controller"}
        row = store.get_swarm(a["swarm"])
        return {
            "kind": "agent",
            "name": a["name"],
            "swarm": a["swarm"],
            "type": a["type"],
            "parent": a["parent"],
            "depth": agent_depth(row["spec"] if row else None, a),
            "children": [c["name"] for c in store.list_agents(parent=a["name"])],
        }

    def not_budget_stopped(swarm: str) -> None:
        if loops.hard_stopped(swarm):
            raise HTTPException(409, f"swarm {swarm!r} reached limits.budget_usd: every agent was "
                                     "stopped; raise the budget (flock apply) to prompt again")

    @api.post("/agents/{name}/prompt")
    def prompt(name: str, req: PromptRequest) -> PromptAck:
        a = agent_or_404(name)
        not_budget_stopped(a["swarm"])
        if over_budget(a, g := agent_grant(a)):
            raise HTTPException(409, f"agent {name!r} reached its granted limits.budget_usd "
                                     f"(${a['cost']:.4f} of ${g.limits.budget_usd:g}); grant more "
                                     f"or revoke it (docs/live.md)")
        cmd = PromptCommand(
            agent=name,
            prompt_id=uuid.uuid4().hex,
            text=req.text,
            timeout=req.timeout,
            reply_to=req.reply_to,
        )
        now = clock().timestamp()
        if req.idempotency_key:
            try:
                cmd.prompt_id = store.add_prompt_once(
                    name, req.idempotency_key,
                    req.model_dump_json(exclude={"idempotency_key"}), cmd.model_dump_json(), now)
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from None
        else:
            store.add_prompt(name, cmd.model_dump_json(), now)
        w = store.get_worker(a["worker"]) if a["worker"] else None
        live = w is not None and worker_state(w, now) == WorkerState.running
        return PromptAck(
            prompt_id=cmd.prompt_id, agent=name, status="queued" if live else "pending"
        )

    @api.post("/swarms/{swarm}/prompts")
    def bulk_prompt(swarm: str, req: BulkPromptRequest) -> BulkPromptAck:
        """The same prompt to every agent of a type (optionally only those in `state`), in one
        request: one pending prompt per agent, delivered and admitted (`max_working`) exactly
        as `POST /agents/{name}/prompt` would."""
        row = spec_or_404(swarm)
        if row["removing"]:
            raise HTTPException(409, f"swarm {swarm!r} is being removed")
        if req.type not in row["spec"].types:
            raise HTTPException(404, f"no type {req.type!r} in swarm {swarm!r}")
        not_budget_stopped(swarm)
        ack = BulkPromptAck(swarm=swarm, type=req.type, matched=0, queued=0, pending=0)
        return prompt_many(swarm, store.list_agents(swarm, req.type), req, ack)

    @api.post("/agents/{name}/children/prompts")
    def children_prompt(name: str, req: ChildrenPromptRequest) -> BulkPromptAck:
        """The same prompt to every direct child of `name` (optionally one type / state), in
        one request, with the rules of `POST /swarms/{swarm}/prompts`. The agent's own token
        may call it (docs/spawn.md), so a leader prompts 100 children in one request."""
        parent = agent_or_404(name)
        if parent["removing"]:
            raise HTTPException(409, f"agent {name!r} is being removed")
        not_budget_stopped(parent["swarm"])
        kids = store.list_agents(parent["swarm"], req.type, parent=name)
        ack = BulkPromptAck(swarm=parent["swarm"], type=req.type, parent=name, matched=0,
                            queued=0, pending=0)
        return prompt_many(parent["swarm"], kids, req, ack)

    def prompt_many(swarm: str, agents: list[Row], req: BulkPromptRequest | ChildrenPromptRequest,
                    ack: BulkPromptAck, build=None, sent: list[str] | None = None) -> BulkPromptAck:
        """One pending prompt per matching agent, delivered and admitted (`max_working`)
        exactly as `POST /agents/{name}/prompt` would. `skip_pending` (default on with
        state=idle) skips agents that already have an undelivered prompt. `build(agent)` makes
        each agent's command (the loop engine's per-agent loop prompts); `sent` collects the
        names that got one."""
        now = clock().timestamp()
        workers = {w["id"]: w for w in store.list_workers(swarm)}
        skip_pending = req.state == AgentState.idle if req.skip_pending is None else req.skip_pending
        pending = store.pending_counts(swarm) if skip_pending else {}
        type_grants, agent_grants = store.agent_grants(swarm)
        rows: list[tuple[str, str]] = []
        for a in sorted(agents, key=lambda a: a["name"]):
            ack.matched += 1
            w = workers.get(a["worker"])
            if a["removing"]:
                ack.skipped_removing += 1
                continue
            if req.state is not None and turn_state(a, w, now) != req.state:
                ack.skipped_state += 1
                continue
            if pending.get(a["name"]):
                ack.skipped_pending += 1  # run 1k: 645 calls stacked 73,394 prompts
                continue
            if over_budget(a, merge_grants(type_grants.get(a["type"]), agent_grants.get(a["name"]))):
                ack.skipped_budget += 1
                continue
            cmd = build(a) if build else PromptCommand(
                agent=a["name"], prompt_id=uuid.uuid4().hex, text=req.text, timeout=req.timeout,
                reply_to=req.reply_to)
            rows.append((a["name"], cmd.model_dump_json()))
            if sent is not None:
                sent.append(a["name"])
            if w is not None and worker_state(w, now) == WorkerState.running:
                ack.queued += 1
            else:
                ack.pending += 1
        store.add_prompts(rows, now)
        return ack

    # ------------------------------------------------------------- loops (docs/loop.md)

    @api.get("/swarms/{swarm}/loop")
    def get_loop(swarm: str) -> list[LoopStatus]:
        spec = spec_or_404(swarm)["spec"]
        return [st for t in sorted(spec.types)
                if (st := loops.status(swarm, t, detail=True)) is not None]

    @api.post("/swarms/{swarm}/loop")
    def control_loop(swarm: str, req: LoopRequest) -> LoopStatus:
        spec_or_404(swarm)
        try:
            return loops.control(swarm, req.type, req.action, final=req.final)
        except KeyError as e:
            raise HTTPException(404, str(e.args[0])) from None
        except ValueError as e:
            raise HTTPException(409, str(e)) from None

    @api.get("/swarms/{swarm}/events")
    async def swarm_events(swarm: str, after: int = 0, wait: float = 0, limit: int = 1000) -> dict:
        """Swarm-level events: loop_started, loop_tick, loop_stopped, final_round, budget_stop."""
        spec_or_404(swarm)
        key, after = swarm_stream(swarm), max(0, after)
        await notifier.long_poll(f"events:{key}", lambda: store.event_count(key) > after, wait)
        evs = store.read_events(key, after, limit=max(1, min(limit, 5000)))
        return {"events": evs, "next": after + len(evs)}

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

    def provider_cap(spec: SwarmSpec, type_name: str) -> tuple[str, int] | None:
        """(provider name, max_working) when admission is enforced for this type's provider."""
        ts = spec.types.get(type_name)
        r = ts.resolved_model if ts else None
        if not enforce_max_working or r is None or r.provider.max_working is None:
            return None
        return r.provider_name, r.provider.max_working

    def busy(a: Row, w: Row | None, now: float) -> bool:
        """In a turn, or handed a prompt to start one, as far as the controller knows."""
        if w is None or worker_state(w, now) == WorkerState.dead:
            live.pop(a["name"], None)
            return False
        seen = live.get(a["name"])
        if seen is not None:
            return seen[0] != "idle"
        return agent_state(a, w, now) == AgentState.working

    def turn_state(a: Row, w: Row | None, now: float) -> AgentState:
        """`agent_state`, with working/idle corrected by what the controller learned since the
        heartbeat (`live`): a prompt just handed out makes it working, a turn_ended event idle.
        Used by the bulk prompt's `state` filter so it neither re-prompts an agent whose turn
        is starting nor waits a heartbeat for one whose turn just ended."""
        st = agent_state(a, w, now)
        seen = live.get(a["name"])
        if seen is None or st not in (AgentState.idle, AgentState.working):
            return st
        return AgentState.idle if seen[0] == "idle" else AgentState.working

    def note_started(names: set[str], now: float) -> None:
        """Uncapped providers: remember prompts handed to agents not in a turn (`live`), as
        admission does for capped ones, so `turn_state` sees the turn start at once."""
        workers: dict[str, Row | None] = {}
        for name in names:
            seen = live.get(name)
            if seen is not None and seen[0] != "idle":
                continue
            a = store.get_agent(name)
            if a is None:
                continue
            if a["worker"] not in workers:
                workers[a["worker"]] = store.get_worker(a["worker"]) if a["worker"] else None
            if seen is None and agent_state(a, workers[a["worker"]], now) == AgentState.working:
                continue
            live[name] = ("admitted", now)

    def admission_view(spec: SwarmSpec, provider: str, cap: int, now: float
                       ) -> tuple[set[str], list[str], dict[str, str | None]]:
        """(busy agents, admission window, agent -> worker) for one provider. The window is
        the agents that get the provider's free slots: not busy, on a running worker, ordered
        by their oldest pending prompt across all workers (FIFO), at most `cap - busy` long."""
        users = {t for t, x in spec.types.items() if x.provider_name == provider}
        workers = {x["id"]: x for x in store.list_workers(spec.swarm)}
        taken: set[str] = set()
        owner: dict[str, str | None] = {}
        for a in store.agent_states(spec.swarm, users):
            owner[a["name"]] = a["worker"]
            if busy(a, workers.get(a["worker"]), now):
                taken.add(a["name"])
        free = cap - len(taken)
        window: list[str] = []
        if free > 0:
            for name in store.oldest_pending(spec.swarm, users):
                w = workers.get(owner.get(name))
                if name in taken or w is None or w["job_id"] is None or w["draining"] \
                        or w["retire_at"] or worker_state(w, now) != WorkerState.running:
                    continue  # a worker that cannot start the turn must not hold a slot
                window.append(name)
                if len(window) >= free:
                    break
        return taken, window, owner

    def admit(w: Row, prompts: list[Row]) -> list[Row]:
        """Provider `max_working` at prompt admission (design.md section 5). A prompt for an
        agent in a turn always goes out (it joins that turn). A prompt that would start a turn
        needs a free slot, and free slots go to pending prompts in FIFO order of creation
        across all workers of the provider (`admission_view`), not to whichever worker polls
        first: others stay queued here and go out on a later poll.

        Slots: agents whose fresh heartbeat says working, overridden by what the controller
        learned since that heartbeat (`live`): a prompt handed out takes the slot at once (so
        two polls between heartbeats cannot both take the last slot), and a turn_ended /
        turn_interrupted event releases it at once, without waiting for the next heartbeat.
        Still soft by about one heartbeat: a heartbeat built before the worker saw an event
        can briefly overrule it (see `heartbeat_seen`)."""
        if not prompts:
            return prompts
        sw = store.get_swarm(w["swarm"])
        cap = provider_cap(sw["spec"], w["type"]) if sw else None
        now = clock().timestamp()
        if cap is None:
            note_started({p["agent"] for p in prompts}, now)
            return prompts
        taken, window, _ = admission_view(sw["spec"], cap[0], cap[1], now)
        allowed = set(window)
        out = []
        for p in prompts:
            name = p["agent"]
            if name not in taken:
                if name not in allowed:
                    continue  # no slot for it yet: stays queued
                taken.add(name)
                live[name] = ("admitted", now)
            out.append(p)
        return out

    def wake_window(swarm: str, providers: set[str]) -> None:
        """A slot was released: wake the polls of the workers whose agents now get it."""
        sw = store.get_swarm(swarm)
        if sw is None:
            return
        spec, now = sw["spec"], clock().timestamp()
        for prov in providers:
            cap = spec.providers.get(prov)
            if cap is None or cap.max_working is None:
                continue
            _, window, owner = admission_view(spec, prov, cap.max_working, now)
            for wid in {owner[n] for n in window if owner.get(n)}:
                notifier.notify(f"commands:{wid}")

    def turn_seen(a: Row, evs: list, now: float) -> bool:
        """Apply an agent's turn events to `live`; True if that released its slot. Events
        are in the worker's order, and the batch is newer than any of them. A turn_ended with
        no turn_started after a fresh admission belongs to an earlier turn: ignored."""
        was = live.get(a["name"])
        state = was[0] if was else None
        for ev in evs:
            if ev.type == "turn_started":
                state = "working"
            elif ev.type == "prompt_deferred" or (
                ev.type in ("turn_ended", "turn_interrupted") and state != "admitted"
            ):
                state = "idle"
        if state is None or (was and state == was[0]):
            return False
        live[a["name"]] = (state, now)
        return state == "idle"

    def heartbeat_seen(hb: WorkerHeartbeat, now: float) -> bool:
        """A heartbeat is the source of truth again for the agents it reports, except a
        fresh admission (< ADMIT_GRACE) it reports idle: that heartbeat was most likely
        built before the worker received the prompt. True if it released a slot."""
        released = False
        for ah in hb.agents:
            seen = live.get(ah.name)
            if seen is None:
                continue
            if seen[0] == "admitted" and ah.state != "working" and now - seen[1] < ADMIT_GRACE:
                continue
            del live[ah.name]
            released |= seen[0] != "idle" and ah.state != "working"
        return released

    def reap_ephemeral(hb: WorkerHeartbeat) -> None:
        """docs/spawn.md "Ephemeral types": remove each spawned agent this heartbeat reports
        idle (or stuck) after at least one turn, with no prompt pending or just admitted. The
        heartbeat's tokens include its last turn, so the cost it leaves behind is complete."""
        now, pending = clock().timestamp(), None
        for ah in hb.agents:
            if ah.state == "working" or ah.turns < 1 or (live.get(ah.name) or ("",))[0] == "admitted":
                continue
            a = store.get_agent(ah.name)
            if a is None or a["parent"] is None or a["removing"]:
                continue
            pending = store.pending_counts(a["swarm"]) if pending is None else pending
            if not pending.get(ah.name):
                remove_subtree(store, ah.name, "ephemeral", now)
                store.listener("reconcile", "")

    def seen(w: Row) -> None:
        """Any request from the slot's current job proves it alive (reconcile.last_seen).
        Written at most every SEEN_EVERY seconds per worker."""
        now = clock().timestamp()
        if w["job_id"] is not None and now - (w.get("last_seen") or 0.0) >= SEEN_EVERY:
            store.update_worker(w["id"], last_seen=now)

    @api.get("/workers/{worker_id}/commands")
    async def commands(worker_id: str, wait: float = 25, incarnation: str | None = None):
        w0 = worker_or_404(worker_id)
        if stale(w0, incarnation):
            raise HTTPException(409, "worker incarnation is not the slot's current job")
        seen(w0)

        def check():
            w = store.get_worker(worker_id)
            if w is None or stale(w, incarnation):
                return "stale"  # replaced while this poll was parked: never hand it commands
            if w["job_id"] is None:
                return None  # slot has no live job (stale process): give it nothing
            cmds = store.commands_for(worker_id)
            # A draining worker starts no turns; its prompts wait for the replacement job. Under
            # the hard budget stop no prompt is delivered (a relaunch's resume prompt included).
            draining = w["draining"] or w["retire_at"] or loops.hard_stopped(w["swarm"])
            prompts = [] if draining else admit(w, store.prompts_for_worker(worker_id))
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
        w = store.get_worker(worker_id)
        sw = store.get_swarm(w["swarm"]) if w else None
        cap = provider_cap(sw["spec"], w["type"]) if sw else None
        if heartbeat_seen(hb, clock().timestamp()) and cap is not None:
            wake_window(w["swarm"], {cap[0]})
        if sw is not None and (ets := sw["spec"].types.get(w["type"])) and ets.ephemeral:
            reap_ephemeral(hb)
        out: dict[str, Any] = {"ok": True}
        spec = sw["spec"] if sw else None
        if spec is not None and spec.tools_version is not None and w is not None:
            # the swarm's current tools (docs/tools.md): a re-apply reaches running workers here
            ts = spec.types.get(w["type"])
            out["tools"] = {"version": spec.tools_version, "select": ts.folder_tools if ts else None}
        if spec is not None and spec.comms is not None and w is not None:
            # comms rules and protocols are live (docs/comms.md): a re-apply reaches workers here
            ts = spec.types.get(w["type"])
            out["comms"] = {**spec.comms.model_dump(mode="json", exclude={"kickoff"}),
                            "tools": comms_tools(ts, spec.comms) if ts else []}
        if spec is not None and spec.world is not None and w is not None:
            # the world's manifest, re-fetched at every apply (docs/world.md)
            ts = spec.types.get(w["type"])
            out["world"] = {**spec.world.model_dump(mode="json"),
                            "tools": world_tools(ts, spec.world) if ts else []}
        ts = spec.types.get(w["type"]) if spec is not None and w is not None else None
        if ts is not None:
            # docs/live.md: a changed type spec (prompts, limits, ...) and the grants in force
            # reach the worker here; it applies them at each agent's next turn boundary
            wts = worker_type_spec(ts)  # never `loop.notices` (controller only)
            rev = type_spec_rev(wts)
            if hb.spec_rev != rev:
                out["type_spec"], out["spec_rev"] = wts.model_dump(mode="json"), rev
            type_grants, agent_grants = store.agent_grants(w["swarm"])
            grants = {a.name: g.compact() for a in hb.agents
                      if (g := merge_grants(type_grants.get(w["type"]), agent_grants.get(a.name)))}
            if grants:  # absent: no grant in force for any of this worker's agents
                out["grants"] = grants
        return out

    @api.post("/workers/{worker_id}/events")
    def post_events(worker_id: str, batch: EventBatch, incarnation: str | None = None) -> dict:
        w = worker_or_404(worker_id)
        if not stale(w, incarnation):  # a replaced job's last events are still accepted
            seen(w)
        grouped: dict[str, list] = {}
        for ev in batch.events:
            grouped.setdefault(ev.agent, []).append(ev)
        now = clock().timestamp()
        sw = store.get_swarm(w["swarm"])
        cap = provider_cap(sw["spec"], w["type"]) if sw else None
        released = False
        for agent, evs in grouped.items():
            a = store.get_agent(agent)
            store.append_events(a["swarm"] if a else w["swarm"], agent, evs)
            for ev in evs:
                if ev.type in COMMS_WAKE_EVENTS:
                    comms_event(agent, ev)
            if a is not None and a["worker"] == worker_id:
                released |= turn_seen(a, evs, now)
            if a is not None:
                loops.on_events(a, evs, sw["spec"] if sw and sw["name"] == a["swarm"] else None)
        if released and cap is not None:  # slots freed now, not at the next heartbeat (run-1k finding 1)
            wake_window(w["swarm"], {cap[0]})
        return {"ok": True, "accepted": len(batch.events)}

    app.include_router(api)
    return app
