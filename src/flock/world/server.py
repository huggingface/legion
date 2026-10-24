"""Serve a `flock.world.World` over HTTP (World API v1, docs/world.md).

`flock world serve module:Class` imports `module` from the current directory, instantiates
`Class` (or takes a World instance named so) and serves it with uvicorn on
`$FLOCK_WORLD_HOST:$FLOCK_WORLD_PORT` (default 127.0.0.1:8765), requiring `X-World-Token` to
equal `$FLOCK_WORLD_TOKEN`. Synchronous tools and hooks run one at a time (a lock), so a world
needs no locking of its own; `async def` tools run on the event loop.
Needs the `controller` extra (FastAPI, uvicorn).
"""

from __future__ import annotations

import hmac
import importlib
import inspect
import json
import logging
import os
import sys
import threading
from typing import Any

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

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

log = logging.getLogger("flock.world")

DEFAULT_PORT = 8765


def _error(code: str, message: str, status: int = 200) -> JSONResponse:
    return JSONResponse({"ok": False, "error": {"code": code, "message": message}},
                        status_code=status)


def _agent(headers: Any) -> Agent:
    return Agent(name=headers.get(AGENT_HEADER, ""), type=headers.get(TYPE_HEADER, ""),
                 swarm=headers.get(SWARM_HEADER, ""), turn=headers.get(TURN_HEADER, ""),
                 request_id=headers.get(REQUEST_HEADER, ""))


def create_app(world: World, token: str | None) -> FastAPI:
    """The ASGI app for `world`. `token` None or "": no authentication (tests, a trusted
    network); otherwise every route needs `X-World-Token: <token>`."""
    app = FastAPI(title=f"world {world.name}", docs_url=None, redoc_url=None, openapi_url=None)
    lock = threading.Lock()

    def schemas() -> dict[str, dict]:
        """From the manifest as served now: what GET /manifest advertises is what is called."""
        return {t["name"]: t.get("parameters") or {"type": "object", "properties": {}}
                for t in world.manifest().get("tools") or []}

    def locked(fn, *args: Any) -> Any:
        with lock:
            return fn(*args)

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if token and not hmac.compare_digest(request.headers.get(TOKEN_HEADER, ""), token):
            return _error("unauthorized", f"missing or wrong {TOKEN_HEADER}", 401)
        return await call_next(request)

    @app.get("/manifest")
    async def manifest() -> dict:
        return await run_in_threadpool(locked, world.manifest)

    @app.post("/tools/{name}")
    async def call(name: str, request: Request):
        try:
            body = await request.json()
        except ValueError:
            return _error("bad_request", "the body must be JSON: {\"args\": {...}}", 400)
        args = body.get("args") if isinstance(body, dict) else None
        args = {} if args is None else args
        agent = _agent(request.headers)
        if not agent.name:
            return _error("no_agent", f"{AGENT_HEADER} is required", 400)
        tools = await run_in_threadpool(locked, schemas)
        if name not in tools:
            return _error("unknown_tool", f"no tool {name!r}; tools: {', '.join(tools) or 'none'}",
                          404)
        if err := check_args(tools[name], args):
            return _error("bad_args", f"{name}: {err}")
        try:
            result = await run_in_threadpool(locked, world.call, name, args, agent)
            if inspect.isawaitable(result):
                result = await result
            result = json.loads(json.dumps(result, default=str))
        except ToolError as e:
            return _error(e.code, e.message)
        except Exception as e:
            log.exception("tool %s failed for %s", name, agent.name)
            return _error("internal", f"{type(e).__name__}: {e}")
        return {"ok": True, "result": result}

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
            result = await run_in_threadpool(locked, fn, *hook_args)
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
