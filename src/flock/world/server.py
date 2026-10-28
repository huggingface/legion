"""Serve a `flock.world.World` over HTTP (World API v1, docs/world.md).

`flock world serve module:Class` imports `module` from the current directory, instantiates
`Class` (or takes a World instance named so) and serves it with uvicorn on
`$FLOCK_WORLD_HOST:$FLOCK_WORLD_PORT` (default 127.0.0.1:8765), requiring `X-World-Token` to
equal `$FLOCK_WORLD_TOKEN`. Synchronous tools and hooks run one at a time (a lock), so a world
needs no locking of its own, unless it is `concurrent`; they run in two thread pools, "fast"
and "slow" (`@tool(pool=)`), so slow tools never starve the fast ones; `async def` tools run on the event loop. An `async` tool the world marks `cancellable(name)` is
cancelled when its caller disconnects (checked every second), so a call nobody waits for any
more stops instead of running to the end.
`GET /metrics` (same token) reports calls, errors, latencies and bytes per tool over the last
5 minutes, the pools' queues, event-loop lag, RSS, CPU and the world's own `metrics()` (JSON, or
Prometheus text for `Accept: text/plain`).
Needs the `controller` extra (FastAPI, uvicorn).
"""

from __future__ import annotations

import asyncio
import contextlib
import hmac
import importlib
import inspect
import json
import logging
import math
import os
import re
import resource
import sys
import threading
import time
import uuid
from collections import Counter, deque
from typing import Any

from anyio import CapacityLimiter, to_thread
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response
from starlette.datastructures import Headers

from flock.tools.runner import check_args
from flock.world import Agent, ToolError, World
from flock.world.protocol import (
    AGENT_HEADER,
    HOOK_EVENTS,
    HOST_ENV,
    PORT_ENV,
    REQUEST_HEADER,
    SWARM_HEADER,
    TOKEN_ENV,
    TOKEN_HEADER,
    TURN_HEADER,
    TYPE_HEADER,
)
from flock.world.workspace import MAX_BYTES

log = logging.getLogger("flock.world")

DEFAULT_PORT = 8765
FAST_THREADS = 40  # tools that answer in well under a second, and hooks
SLOW_THREADS = 64  # tools that block for seconds or minutes (`@tool(pool="slow")`)
# Admission control (docs/world.md): a tool call is refused with 429 when the fast pool's oldest
# queued call waited longer than this, or the slow pool holds its threads plus this many queued.
FAST_MAX_WAIT_ENV, FAST_MAX_WAIT_S = "FLOCK_WORLD_FAST_MAX_WAIT_S", 2.0
SLOW_QUEUE_ENV, SLOW_QUEUE = "FLOCK_WORLD_SLOW_QUEUE", 8
RETRY_AFTER_ENV, RETRY_AFTER_S = "FLOCK_WORLD_RETRY_AFTER_S", 30.0


WINDOW_S = 300.0  # /metrics: calls, latencies, bytes and queue waits over the last 5 minutes


class _Metrics:
    """What GET /metrics reports. `call` and `lag` run on the event loop; `waited` runs in a
    pool thread (a deque append is atomic)."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self.clock = clock
        self.started = clock()
        self.calls: dict[str, deque] = {}  # tool -> (t, latency_s or None if refused, code, in, out)
        self.totals: dict[str, list[int]] = {}  # tool -> [calls, errors] since start
        self.waits: dict[str, deque] = {"fast": deque(), "slow": deque()}  # pool -> (t, wait_s)
        self.lags: deque = deque()  # (t, event-loop lag in s)
        self.overloaded = 0
        self.cpu_mark = (time.monotonic(), sum(os.times()[:2]))

    def _trim(self, q: deque, now: float) -> deque:
        while q and q[0][0] < now - WINDOW_S:
            q.popleft()
        return q

    def call(self, tool: str, latency: float | None, code: str | None, n_in: int, n_out: int) -> None:
        now = self.clock()
        self._trim(self.calls.setdefault(tool, deque()), now).append((now, latency, code, n_in, n_out))
        totals = self.totals.setdefault(tool, [0, 0])
        totals[0] += 1
        totals[1] += code is not None

    def waited(self, pool: str, wait: float) -> None:
        self.waits[pool].append((self.clock(), wait))

    def lag(self, lag: float) -> None:
        now = self.clock()
        self._trim(self.lags, now).append((now, lag))

    async def sample_lag(self) -> None:
        """Every second: how late a 1 s sleep wakes up (the event loop's lag)."""
        loop = asyncio.get_running_loop()
        while True:
            t = loop.time()
            await asyncio.sleep(1)
            self.lag(max(0.0, loop.time() - t - 1))

    def snapshot(self, pools: dict[str, dict[str, float]]) -> dict[str, Any]:
        now = self.clock()
        tools = {}
        for name, q in self.calls.items():
            rows = list(self._trim(q, now))
            lat = sorted(r[1] for r in rows if r[1] is not None)
            tools[name] = {
                "calls": len(rows), "errors_by_code": dict(Counter(r[2] for r in rows if r[2])),
                "latency_p50_s": _pct(lat, 0.5), "latency_p95_s": _pct(lat, 0.95),
                "latency_max_s": lat[-1] if lat else 0.0,
                "bytes_in": sum(r[3] for r in rows), "bytes_out": sum(r[4] for r in rows),
                "calls_total": self.totals[name][0], "errors_total": self.totals[name][1]}
        for pool, stats in pools.items():
            waits = sorted(w for _, w in list(self._trim(self.waits[pool], now)))
            stats.update(queue_wait_p95_s=_pct(waits, 0.95),
                         queue_wait_max_s=max(waits[-1] if waits else 0.0, stats["queue_wait_oldest_s"]))
        lags = [g for _, g in self._trim(self.lags, now)]
        wall, cpu = time.monotonic(), sum(os.times()[:2])
        cpu_pct = 100 * (cpu - self.cpu_mark[1]) / max(1e-9, wall - self.cpu_mark[0])
        self.cpu_mark = (wall, cpu)
        return {"uptime_s": now - self.started, "rss_mb": _rss_mb(), "cpu_pct": cpu_pct,
                "loop_lag_s": lags[-1] if lags else 0.0, "loop_lag_max_s": max(lags, default=0.0),
                "overloaded_429_total": self.overloaded, "window_s": WINDOW_S,
                "tools": tools, "pools": pools}


def _pct(values: list[float], q: float) -> float:
    """Nearest-rank percentile of sorted `values` (0.0 when empty)."""
    return values[max(0, math.ceil(q * len(values)) - 1)] if values else 0.0


def _rss_mb() -> float:
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**20
    except (OSError, ValueError, IndexError):  # not Linux: the peak instead
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _prometheus(m: dict[str, Any]) -> str:
    """The /metrics JSON as Prometheus text exposition (untyped samples)."""
    lines = []
    for key, value in m.items():
        if key in ("tools", "pools"):
            label = key[:-1]
            for name, stats in value.items():
                for k, v in stats.items():
                    if k == "errors_by_code":
                        lines += [f"flock_world_tool_errors{{tool={json.dumps(name)},"
                                  f"code={json.dumps(c)}}} {n}" for c, n in v.items()]
                    else:
                        lines.append(f"flock_world_{label}_{k}{{{label}={json.dumps(name)}}} {v}")
        elif key == "world":
            lines += [f"flock_world_world_{re.sub(r'[^A-Za-z0-9_]', '_', k)} {v}"
                      for k, v in value.items()]
        else:
            lines.append(f"flock_world_{key} {value}")
    return "\n".join(lines) + "\n"


_SECRET_NAME = re.compile(r"TOKEN|SECRET|PASSWORD|INVITE|API_KEY|_KEY$", re.IGNORECASE)


def _secret_values(env=None) -> list[str]:
    """Values of the world process's secret-looking variables (8+ chars), longest first."""
    env = os.environ if env is None else env
    return sorted({v for k, v in env.items() if _SECRET_NAME.search(k) and len(v) >= 8},
                  key=len, reverse=True)


def redact(value: Any, secrets: list[str] | None = None) -> Any:
    """`value` (JSON-able) with every secret value replaced by "[redacted]"."""
    secrets = _secret_values() if secrets is None else secrets
    if not secrets:
        return value
    text = json.dumps(value, default=str)
    if not any(s in text for s in secrets):
        return value
    for s in secrets:
        text = text.replace(json.dumps(s)[1:-1], "[redacted]")
    return json.loads(text)


def _ok_body(result: Any) -> bytes:
    """The success envelope for `result`, serialized and redacted (in a worker thread)."""
    return json.dumps({"ok": True, "result": redact(json.loads(json.dumps(result, default=str)))}
                      ).encode()


def _error(code: str, message: str, status: int = 200,
           retry_after_s: float | None = None) -> JSONResponse:
    error: dict[str, Any] = {"code": code, "message": message}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    return JSONResponse({"ok": False, "error": error}, status_code=status)


class _Auth:
    """Every route needs `X-World-Token: <token>`."""

    def __init__(self, app: Any, token: str) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] == "http" and not hmac.compare_digest(
                Headers(scope=scope).get(TOKEN_HEADER, ""), self.token):
            await _error("unauthorized", f"missing or wrong {TOKEN_HEADER}", 401)(
                scope, receive, send)
            return
        await self.app(scope, receive, send)


DISCONNECT_POLL_S = 1.0


class _Gone(Exception):
    """The caller disconnected before the tool finished."""


async def _unless_gone(aw: Any, request: Request) -> Any:
    """Await `aw`, cancelling it if the client disconnects first (raises _Gone)."""
    task = asyncio.ensure_future(aw)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_S)
            if done:
                return task.result()
            if await request.is_disconnected():
                task.cancel()
                await asyncio.wait({task})
                raise _Gone
    except asyncio.CancelledError:
        task.cancel()  # this handler cancelled: the tool's work goes too
        raise


def _agent(headers: Any) -> Agent:
    return Agent(name=headers.get(AGENT_HEADER, ""), type=headers.get(TYPE_HEADER, ""),
                 swarm=headers.get(SWARM_HEADER, ""), turn=headers.get(TURN_HEADER, ""),
                 request_id=headers.get(REQUEST_HEADER, ""))


def create_app(world: World, token: str | None) -> FastAPI:
    """The ASGI app for `world`. `token` None or "": no authentication (tests, a trusted
    network); otherwise every route needs `X-World-Token: <token>`."""
    metrics = _Metrics()

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> Any:
        task = asyncio.create_task(metrics.sample_lag())
        try:
            yield
        finally:
            task.cancel()

    app = FastAPI(title=f"world {world.name}", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    lock = threading.Lock()
    # Separate thread budgets: a slow tool (minutes) never takes a fast tool's or a hook's thread.
    limiters = {"fast": CapacityLimiter(FAST_THREADS), "slow": CapacityLimiter(SLOW_THREADS)}
    queued: dict[str, dict[object, float]] = {p: {} for p in limiters}  # pool -> call -> since
    queued_lock = threading.Lock()

    async def run(pool: str, fn, *args: Any) -> Any:
        pool = pool if pool in limiters else "fast"
        key, since = object(), metrics.clock()
        with queued_lock:
            queued[pool][key] = since

        def started(*a: Any) -> Any:  # in the pool thread: the call left the queue
            with queued_lock:
                queued[pool].pop(key, None)
            metrics.waited(pool, metrics.clock() - since)
            return locked(fn, *a)

        try:
            return await to_thread.run_sync(started, *args, limiter=limiters[pool])
        finally:  # cancelled while queued
            with queued_lock:
                queued[pool].pop(key, None)

    fast_max_wait = float(os.environ.get(FAST_MAX_WAIT_ENV) or FAST_MAX_WAIT_S)
    slow_queue = int(os.environ.get(SLOW_QUEUE_ENV) or SLOW_QUEUE)
    retry_after = float(os.environ.get(RETRY_AFTER_ENV) or RETRY_AFTER_S)

    def overloaded(pool: str) -> str | None:
        """Why a new call to `pool` is refused now, or None."""
        stats = pool_stats()[pool]
        if pool == "slow" and stats["in_flight"] >= stats["limit"] + slow_queue:
            return f"{stats['in_flight']:.0f} slow calls running or queued"
        if pool == "fast" and stats["queue_wait_oldest_s"] > fast_max_wait:
            return f"calls wait {stats['queue_wait_oldest_s']:.1f} s for a thread"
        return None

    def pool_stats() -> dict[str, dict[str, float]]:
        now = metrics.clock()
        with queued_lock:
            waiting = {p: (len(q), min(q.values(), default=now)) for p, q in queued.items()}
        return {p: {"in_flight": lim.borrowed_tokens + waiting[p][0], "queued": waiting[p][0],
                    "limit": lim.total_tokens, "queue_wait_oldest_s": now - waiting[p][1]}
                for p, lim in limiters.items()}

    cache: dict[str, Any] = {"version": None}

    def served() -> dict[str, Any]:
        """The manifest as served now, serialized once per manifest version (on the event
        loop: a health check never waits for a thread): what GET /manifest advertises is what
        is called. Building the dict is cheap; serializing it is not. Rebuilt only when
        `version` changes: a world that changes its description or capabilities at runtime
        must bump `version`."""
        m = world.manifest()
        if cache["version"] != m.get("version"):
            caps = list(dict.fromkeys([*(m.get("capabilities") or []), "metrics-v1"]))
            cache.update(version=m.get("version"), body=json.dumps({**m, "capabilities": caps}).encode(), schemas={
                t["name"]: t.get("parameters") or {"type": "object", "properties": {}}
                for t in m.get("tools") or []})
        return cache

    def locked(fn, *args: Any) -> Any:
        if getattr(world, "concurrent", False):  # the world guards its own state
            return fn(*args)
        with lock:
            return fn(*args)

    if token:  # plain ASGI (not @app.middleware): a client's disconnect still reaches the route
        app.add_middleware(_Auth, token=token)

    @app.get("/manifest")
    async def manifest() -> Response:
        return Response(served()["body"], media_type="application/json")

    @app.get("/metrics")
    async def get_metrics(request: Request) -> Response:
        try:
            custom = await asyncio.to_thread(world.metrics)
        except Exception:
            log.exception("world.metrics() failed")
            custom = {}
        m = {**metrics.snapshot(pool_stats()),
             "world": {k: v for k, v in (custom or {}).items() if isinstance(v, int | float)}}
        if "text/plain" in request.headers.get("accept", ""):
            return PlainTextResponse(_prometheus(m), media_type="text/plain; version=0.0.4")
        return JSONResponse(m)

    @app.post("/tools/{name}")
    async def call(name: str, request: Request) -> Response:
        t0 = metrics.clock()
        pool = getattr(world, "pool", lambda _: "fast")(name)
        # Decided from the path alone, before the body (an upload can be 100+ MB) is read.
        if why := overloaded(pool if pool in limiters else "fast"):
            metrics.overloaded += 1
            n = 0
            async for chunk in request.stream():  # drained unparsed: the client reads the 429
                n += len(chunk)
                if n > MAX_BYTES * 2:
                    break
            resp = _error("overloaded", f"world {world.name} is overloaded ({why}); retry in "
                          f"{retry_after:g} s", 429, retry_after_s=retry_after)
            resp.headers["Retry-After"] = str(math.ceil(retry_after))
            latency = None
        else:
            resp = await _call(name, request)
            latency = metrics.clock() - t0
        if name in served()["schemas"]:
            code = json.loads(resp.body)["error"]["code"] if isinstance(resp, JSONResponse) else None
            n_in = request.headers.get("content-length", "")
            metrics.call(name, latency, code, int(n_in) if n_in.isdigit() else 0, len(resp.body))
        return resp

    async def _call(name: str, request: Request) -> Response:
        try:
            limit = MAX_BYTES * 4 // 3 + (4 << 20) if "workspace-upload-v1" in world.capabilities else 1 << 20
            data = bytearray()
            async for chunk in request.stream():
                data.extend(chunk)
                if len(data) > limit:
                    return _error("bad_request", "tool request is too large", 413)
            body = await asyncio.to_thread(json.loads, data)  # MBs for an upload: off the loop
        except ValueError:
            return _error("bad_request", "the body must be JSON: {\"args\": {...}}", 400)
        args = body.get("args") if isinstance(body, dict) else None
        args = {} if args is None else args
        agent = _agent(request.headers)
        if not agent.name:
            return _error("no_agent", f"{AGENT_HEADER} is required", 400)
        tools = served()["schemas"]
        if name not in tools:
            return _error("unknown_tool", f"no tool {name!r}; tools: {', '.join(tools) or 'none'}",
                          404)
        if err := check_args(tools[name], args):
            return _error("bad_args", f"{name}: {err}")
        pool = getattr(world, "pool", lambda _: "fast")(name)
        try:
            if isinstance(body, dict) and "workspace" in body:
                handler = getattr(world, "call_workspace", None)
                if "workspace-upload-v1" not in world.capabilities or handler is None:
                    return _error("bad_request", "this world does not accept workspace uploads", 400)
                result = await run(pool, handler, name, args, agent, body["workspace"])
            else:
                result = await run(pool, world.call, name, args, agent)
            if inspect.isawaitable(result):
                if getattr(world, "cancellable", lambda _: False)(name):
                    try:
                        result = await _unless_gone(result, request)
                    except _Gone:
                        log.info("tool %s for %s: caller gone, cancelled", name, agent.name)
                        return _error("cancelled", "the caller disconnected", 499)
                else:
                    result = await result
            out = await asyncio.to_thread(_ok_body, result)
        except ToolError as e:
            return _error(e.code, redact(e.message), retry_after_s=e.retry_after_s)
        except Exception:
            # The exception text can hold paths, command output or credentials: it stays in
            # the world's log, under an id the agent can quote.
            ref = uuid.uuid4().hex[:8]
            log.exception("tool %s failed for %s (error id %s)", name, agent.name, ref)
            return _error("internal", f"the tool failed with an internal error (error id {ref})")
        return Response(out, media_type="application/json")

    @app.post("/hooks")
    async def hooks(request: Request) -> dict:
        try:
            body = await request.json()
        except ValueError:
            return {"ok": False}
        event = body.get("event") if isinstance(body, dict) else None
        if event not in HOOK_EVENTS or event not in world.hooks():
            return {"ok": True, "ignored": True}
        agent = Agent(name=str(body.get("agent") or ""), type=str(body.get("type") or ""),
                      swarm=str(body.get("swarm") or ""), turn=str(body.get("turn") or ""))
        fn = getattr(world, f"on_{event}")
        hook_args = (agent,) if event == "agent_start" else (agent, body.get("data") or {})
        try:
            result = await run("fast", fn, *hook_args)
            if inspect.isawaitable(result):  # an `async def` hook
                await result
        except Exception:
            log.exception("hook %s failed for %s", event, agent.name)
        return {"ok": True}

    return app


def load(target: str, path: str | None = None) -> World:
    """`module:Class` (or `module:instance`) from `path` (default: the current directory)."""
    module_name, _, attr = target.partition(":")
    if not module_name or not attr:
        raise ValueError(f"{target!r}: expected module:Class, e.g. taskboard:TaskBoard")
    sys.path.insert(0, path or os.getcwd())
    obj = getattr(importlib.import_module(module_name), attr)
    world = obj() if isinstance(obj, type) else obj
    if not isinstance(world, World):
        raise TypeError(f"{target!r} is not a flock.world.World (subclass or instance)")
    return world


def stop_on_usr1() -> None:
    """SIGUSR1 (a Spot notice on Slurm: the watcher signals every process of the job) shuts
    the server down like SIGTERM, once: later USR1s are ignored (docs/slurm.md)."""
    import signal

    sent = []

    def handler(signum, frame) -> None:
        if not sent:
            sent.append(signum)
            print("world: SIGUSR1: shutting down", flush=True)
            os.kill(os.getpid(), signal.SIGTERM)  # uvicorn's own graceful shutdown

    signal.signal(signal.SIGUSR1, handler)


def serve(target: str, *, host: str | None = None, port: int | None = None,
          token_env: str = TOKEN_ENV, insecure: bool = False) -> None:
    import uvicorn

    from flock.isolation import enabled, protect_supervisor

    if enabled():
        protect_supervisor()

    token = os.environ.get(token_env, "")
    if not token and not insecure:
        raise SystemExit(f"{token_env} is not set: a world needs its shared secret "
                         "(or pass --insecure for a world without one)")
    world = load(target)
    host = host or os.environ.get(HOST_ENV) or "127.0.0.1"
    port = port or int(os.environ.get(PORT_ENV) or DEFAULT_PORT)
    print(f"world {world.name} {world.version} on http://{host}:{port}", flush=True)
    stop_on_usr1()
    uvicorn.run(create_app(world, token or None), host=host, port=port, log_level="warning",
                access_log=False)
