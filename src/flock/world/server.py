"""Serve a `flock.world.World` over HTTP (World API v1, docs/world.md).

`flock world serve module:Class` imports `module` from the current directory, instantiates
`Class` (or takes a World instance named so) and serves it with uvicorn on
`$FLOCK_WORLD_HOST:$FLOCK_WORLD_PORT` (default 127.0.0.1:8765), requiring `X-World-Token` to
equal `$FLOCK_WORLD_TOKEN`. Synchronous tools and hooks run one at a time (a lock), so a world
needs no locking of its own, unless it is `concurrent`; they run in two thread pools, "fast"
and "slow" (`@tool(pool=)`), so slow tools never starve the fast ones; `async def` tools run on the event loop. An `async` tool the world marks `cancellable(name)` is
cancelled when its caller disconnects (checked every second), so a call nobody waits for any
more stops instead of running to the end.
Needs the `controller` extra (FastAPI, uvicorn).
"""

from __future__ import annotations

import asyncio
import hmac
import importlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import uuid
from typing import Any

from anyio import CapacityLimiter, to_thread
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
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


def _error(code: str, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"ok": False, "error": {"code": code, "message": message}},
                        status_code=status)


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
    app = FastAPI(title=f"world {world.name}", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()
    # Separate thread budgets: a slow tool (minutes) never takes a fast tool's or a hook's thread.
    limiters = {"fast": CapacityLimiter(FAST_THREADS), "slow": CapacityLimiter(SLOW_THREADS)}

    def run(pool: str, fn, *args: Any) -> Any:
        return to_thread.run_sync(locked, fn, *args, limiter=limiters.get(pool, limiters["fast"]))

    cache: dict[str, Any] = {"version": None}

    def served() -> dict[str, Any]:
        """The manifest as served now, serialized once per manifest version (on the event
        loop: a health check never waits for a thread): what GET /manifest advertises is what
        is called. Building the dict is cheap; serializing it is not. Rebuilt only when
        `version` changes: a world that changes its description or capabilities at runtime
        must bump `version`."""
        m = world.manifest()
        if cache["version"] != m.get("version"):
            cache.update(version=m.get("version"), body=json.dumps(m).encode(), schemas={
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

    @app.post("/tools/{name}")
    async def call(name: str, request: Request):
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
            return _error(e.code, redact(e.message))
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
