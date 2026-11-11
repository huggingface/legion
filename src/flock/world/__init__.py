"""`flock.world`: a small base for writing a world in Python (docs/world.md).

A world is any HTTP server that speaks World API v1 (`flock.world.protocol`); this base is a
convenience, not a requirement. Subclass `World`, mark tool methods with `@tool`, and serve it
with `python -m flock.world serve module:Class` (`flock.world.server`):

    from flock.world import Agent, ToolError, World, tool

    class Counter(World):
        name, version = "counter", "0.1.0"

        def __init__(self) -> None:
            self.n = 0

        @tool
        def bump(self, agent: Agent, by: int = 1) -> dict:
            \"\"\"Add `by` to the shared counter and return it.\"\"\"
            if by < 1:
                raise ToolError("by must be >= 1")
            self.n += by
            return {"n": self.n, "by": agent.name}

Tool schemas come from the type hints and docstrings with the tools-folder rules
(flock.tools.loader); a parameter named `agent` (or annotated `Agent`) receives the
caller and is never shown to the model. Override `manifest()` and `call()` to serve tools some
other way. The hooks `on_agent_start`, `on_agent_done` and `on_turn` are sent only when a
subclass overrides them (the manifest lists them in `hooks`).
"""

from __future__ import annotations

import inspect
import os
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flock.world.admin import TYPES, Admin, Knob
from flock.world.protocol import HOOK_EVENTS, STATE_DIR_ENV, WORLD_API

AGENT_PARAM = "agent"


@dataclass(frozen=True)
class Agent:
    """Who calls: the flock agent name, its type and swarm, the turn and the request id (as
    flock sent them; trust them when the token matched)."""

    name: str
    type: str = ""
    swarm: str = ""
    turn: str = ""
    request_id: str = ""


class ToolError(Exception):
    """Raise from a tool: the agent reads `message` as the tool's error result.
    `retry_after_s`: the call cannot succeed before then (a budget, a busy backend); the
    server sends it in the error envelope and the worker may hold the tool off until then."""

    def __init__(self, message: str, code: str = "tool_error",
                 retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s


POOLS = ("fast", "slow")


def tool(fn: Callable | None = None, *, pool: str = "fast", read_only: bool = False) -> Any:
    """Mark a method of a World subclass as a tool: `@tool`, or `@tool(pool="slow",
    read_only=True)`. `pool`: the server's thread pool for the call ("slow" for tools that
    block for seconds, so they never take the threads of the quick ones). `read_only`: the
    tool changes nothing (manifest `annotations.readOnlyHint`; the worker may retry it)."""
    if pool not in POOLS:
        raise ValueError(f"pool {pool!r} must be one of {POOLS}")

    def mark(f: Callable) -> Callable:
        f.__flock_world_tool__ = True  # type: ignore[attr-defined]
        f.__flock_world_pool__ = pool  # type: ignore[attr-defined]
        f.__flock_world_read_only__ = read_only  # type: ignore[attr-defined]
        return f

    return mark if fn is None else mark(fn)


def _is_agent(p: inspect.Parameter, hints: dict[str, Any]) -> bool:
    hint = hints.get(p.name)
    return p.name == AGENT_PARAM or hint is Agent


def tool_schema(fn: Callable) -> dict[str, Any]:
    """{name, description, parameters} of a tool method (raises flock.tools.loader.ToolsError)."""
    from flock.tools.loader import tool_def

    hints = typing.get_type_hints(fn)
    sig = inspect.signature(fn)
    keep = [p for p in sig.parameters.values() if not _is_agent(p, hints)]

    def shim(*_a: Any, **_k: Any) -> None: ...

    shim.__name__ = fn.__name__
    shim.__doc__ = fn.__doc__
    shim.__signature__ = sig.replace(parameters=keep)  # type: ignore[attr-defined]
    shim.__annotations__ = {k: v for k, v in hints.items() if k in {p.name for p in keep}}
    d = tool_def(shim, type(getattr(fn, "__self__", fn)).__name__)
    return {"name": d.name, "description": d.description, "parameters": d.parameters}


class World:
    """Base class. Class attributes: `name` (lowercase, `[a-z][a-z0-9-]*`), `version` (any
    string; bump it when the tools change, agents are told), `description` (shown to the
    agents), `capabilities` (extra names for the manifest)."""

    name: typing.ClassVar[str] = "world"
    version: typing.ClassVar[str] = "0.0.0"
    description: typing.ClassVar[str] = ""
    capabilities: typing.ClassVar[list[str]] = []
    # False (default): the server runs one tool call or hook at a time (a global lock), so a
    # world needs no locking of its own. True: calls and hooks run concurrently in the
    # server's thread pool; the world guards its own state (a slow tool then never holds up
    # the hooks or the other agents' calls).
    concurrent: typing.ClassVar[bool] = False

    # ------------------------------------------------------------------ tools
    def tools(self) -> dict[str, Callable]:
        """The bound `@tool` methods, by name (in definition order)."""
        found = self.__dict__.get("_flock_tools")
        if found is None:  # no __init__ needed in subclasses: built on first use
            found = {}
            for klass in reversed(type(self).__mro__):
                for attr, value in vars(klass).items():
                    if getattr(value, "__flock_world_tool__", False):
                        found[attr] = getattr(self, attr)
            self.__dict__["_flock_tools"] = found
        return found

    def hooks(self) -> list[str]:
        """The hooks a subclass overrides."""
        out = []
        for event in HOOK_EVENTS:
            meth = f"on_{event}"
            if getattr(type(self), meth) is not getattr(World, meth):
                out.append(event)
        return out

    def manifest(self) -> dict[str, Any]:
        """Built on every call (the server serves and dispatches from it), from the class
        attributes as they are now; the tool schemas are computed once."""
        hooks = self.hooks()
        caps = list(dict.fromkeys([*self.capabilities, *(["hooks"] if hooks else [])]))
        schemas = self.__dict__.get("_flock_schemas")
        if schemas is None:
            schemas = [tool_schema(fn) for fn in self.tools().values()]
            for t, fn in zip(schemas, self.tools().values(), strict=True):
                if getattr(fn, "__flock_world_read_only__", False):
                    t["annotations"] = {"readOnlyHint": True}
            self.__dict__["_flock_schemas"] = schemas
        return {"world_api": WORLD_API, "name": self.name, "version": self.version,
                "description": self.description, "tools": [dict(t) for t in schemas],
                "capabilities": caps, "hooks": hooks}

    def call(self, tool: str, args: dict[str, Any], agent: Agent) -> Any:
        """Run `tool` for `agent`; the return value is the result (JSON-serialisable, or a
        coroutine the server awaits). Raise ToolError for an error the agent should read;
        KeyError for an unknown tool."""
        fn = self.tools()[tool]
        hints = typing.get_type_hints(fn)
        extra = {p.name: agent for p in inspect.signature(fn).parameters.values()
                 if _is_agent(p, hints)}
        return fn(**args, **extra)

    def pool(self, tool: str) -> str:
        """The server's thread pool for `tool`: "fast" (default) or "slow" (`@tool(pool=)`)."""
        return getattr(self.tools().get(tool), "__flock_world_pool__", "fast")

    def cancellable(self, tool: str) -> bool:
        """May the server cancel this tool's `async` work when its caller disconnects (the
        HTTP client gave up, e.g. on its own timeout)? Default False: the work always runs to
        the end. A world returns True for tools whose work is wasted once nobody reads it."""
        return False

    def metrics(self) -> dict[str, float]:
        """World-specific numbers for `GET /metrics` (under `world`). Called in a thread,
        concurrently with tools, every scrape (about every 30 s): keep it cheap; take your
        own lock if you need one. Default: none."""
        return {}

    # ------------------------------------------------------------------ admin (docs/world.md)
    def state_dir(self) -> str | os.PathLike | None:
        """Where `knobs.json` lives (knob values and paused lanes set through the admin API,
        read back at the next start). Default: $FLOCK_WORLD_STATE_DIR, else None (kept in
        memory only). The server may call it before `__init__` ran (to serve /admin while the
        world cannot be built): do not rely on attributes `__init__` sets without a fallback."""
        return os.environ.get(STATE_DIR_ENV) or None

    @property
    def admin(self) -> Admin:
        """This world's knobs and paused lanes (the server may install one before __init__)."""
        if "_flock_admin" not in self.__dict__:
            self.__dict__["_flock_admin"] = Admin(self.state_dir())
        return self.__dict__["_flock_admin"]

    def knob(self, name: str, type: type, default: Any, *, min: float | None = None,
             max: float | None = None, doc: str = "",
             on_change: Callable[[Any], None] | None = None) -> Any:
        """Register a runtime knob (int, float, bool or str) and return its current value: the
        value set through `PATCH /admin/knobs` if any (it survives restarts), else `default`.
        A change calls `on_change(value)` (in a thread; raise to refuse it) and shows in
        `self.knobs[name]`."""
        if type not in TYPES:
            raise ValueError(f"knob {name}: type must be one of int, float, bool, str")
        return self.admin.register(Knob(name, type, default, min, max, doc, on_change))

    @property
    def knobs(self) -> dict[str, Any]:
        """The knobs' current values, by name (live: reflects admin changes)."""
        return self.admin.values

    def paused(self, lane: str) -> bool:
        """Did an operator pause `lane` (`POST /admin/pause`)? Lanes are names the world
        chooses and documents (e.g. "push", "open", "merge")."""
        return lane in self.admin.paused

    # ------------------------------------------------------------------ hooks
    def on_agent_start(self, agent: Agent) -> None:
        """An agent got its first prompt on its worker (again after a relaunch)."""

    def on_agent_done(self, agent: Agent, data: dict[str, Any]) -> None:
        """An agent was removed from the swarm (scale down, `flock stop --delete`, `flock-agent remove`)."""

    def on_turn(self, agent: Agent, data: dict[str, Any]) -> None:
        """An agent's turn ended: `data` has status, tool_calls, turn."""


__all__ = ["Admin", "Agent", "ToolError", "World", "tool", "tool_schema"]
