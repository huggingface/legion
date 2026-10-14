"""FastAPI app: the routes in design.md "API (controller)" plus /healthz and GET /swarms."""

from __future__ import annotations

import asyncio
import html
import secrets
import uuid
from collections import Counter
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import Field, TypeAdapter
from starlette.background import BackgroundTask

from swarm_harness.controller.reconcile import agent_state, record_heartbeat, worker_state
from swarm_harness.controller.store import Row, Store, agent_tokens
from swarm_harness.models import (
    AgentState,
    AgentStatus,
    Command,
    CommandBatch,
    EventBatch,
    Harness,
    PromptAck,
    PromptCommand,
    PromptRequest,
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


# --------------------------------------------------------------------------- status builders


def agent_status(a: Row, w: Row | None, spec: SwarmSpec | None, pending: int, now: float):
    ts = spec.types.get(a["type"]) if spec else None
    return AgentStatus(
        name=a["name"],
        swarm=a["swarm"],
        type=a["type"],
        harness=ts.harness if ts else Harness.claude,
        model=ts.model if ts else "unknown",
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
    return SwarmStatus(
        swarm=spec.swarm,
        namespace=spec.namespace,
        bucket=spec.bucket,
        types=types,
        agents_total=len(agents),
        workers_total=len(workers),
        pending_prompts=sum(pending.values()),
        cost_usd=round(sum(t.cost_usd for t in types.values()), 6),
    )


def _status_page(store: Store, now: float) -> str:
    e = html.escape
    parts = []
    for sw in store.list_swarms():
        spec: SwarmSpec = sw["spec"]
        st = swarm_status(store, spec, now)
        removing = " (removing)" if sw["removing"] else ""
        parts.append(
            f"<h2>{e(spec.swarm)}{removing}</h2><p>namespace {e(spec.namespace)} · "
            f"bucket {e(spec.bucket)} · {st.agents_total} agents · {st.workers_total} workers · "
            f"{st.pending_prompts} pending prompts · ${st.cost_usd:.2f}</p>"
        )
        rows = []
        for t in st.types.values():
            a_states = " ".join(f"{s}:{n}" for s, n in sorted(t.agents_by_state.items()))
            w_states = " ".join(f"{s}:{n}" for s, n in sorted(t.workers_by_state.items()))
            rows.append(
                f"<tr><td>{e(t.name)}</td><td>{e(t.health.value)}</td><td>{t.replicas}</td>"
                f"<td>{e(a_states)}</td><td>{e(w_states)}</td><td>${t.cost_usd:.2f}</td></tr>"
            )
        parts.append(
            "<table><tr><th>type</th><th>health</th><th>replicas</th><th>agents by state</th>"
            "<th>workers by state</th><th>cost</th></tr>" + "".join(rows) + "</table>"
        )
        for type_name in st.types:
            names = [a["name"] for a in store.list_agents(spec.swarm, type_name)]
            links = " ".join(f'<a href="/agents/{e(n)}">{e(n)}</a>' for n in names[:200])
            more = f" … and {len(names) - 200} more" if len(names) > 200 else ""
            parts.append(
                f"<details><summary>{e(type_name)} agents</summary>{links}{more}</details>"
            )
    body = "".join(parts) or "<p>No swarms.</p>"
    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>swarm-harness</title>"
        "<meta http-equiv='refresh' content='30'><style>body{font-family:sans-serif;margin:2em}"
        "table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:4px 8px}"
        "</style></head><body><h1>swarm-harness</h1>"
        f"<p>{e(_dt(now).isoformat(timespec='seconds'))}</p>{body}</body></html>"
    )


# --------------------------------------------------------------------------- app


def create_app(
    store: Store,
    token: str,
    background: Callable[[Notifier], Awaitable[None]] | None = None,
    clock: Callable[[], datetime] = _now,
) -> FastAPI:
    notifier = Notifier()
    store.listener = lambda kind, key: notifier.notify(f"{kind}:{key}")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        notifier._loop = asyncio.get_running_loop()
        task = asyncio.create_task(background(notifier)) if background else None
        yield
        if task:
            task.cancel()

    app = FastAPI(title="swarm-harness controller", lifespan=lifespan)
    app.state.store, app.state.notifier = store, notifier

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
        return {"ok": True}

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

    @api.get("/workers/{worker_id}/commands")
    async def commands(worker_id: str, wait: float = 25):
        worker_or_404(worker_id)

        def check():
            w = store.get_worker(worker_id)
            if w is None or w["job_id"] is None:
                return None  # slot has no live job (stale process): give it nothing
            cmds = store.commands_for(worker_id)
            # A draining worker starts no turns; its prompts wait for the replacement job.
            draining = w["draining"] or w["retire_at"]
            prompts = [] if draining else store.prompts_for_worker(worker_id)
            return (cmds, prompts) if cmds or prompts else None

        found = await notifier.long_poll(f"commands:{worker_id}", check, wait)
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
    def heartbeat(worker_id: str, hb: WorkerHeartbeat) -> dict:
        worker_or_404(worker_id)
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
