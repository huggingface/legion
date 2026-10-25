"""`flock.world`: a small base for writing a world in Python (docs/world.md).

A world is any HTTP server that speaks World API v1 (`flock.world.protocol`); this base is a
convenience, not a requirement. Subclass `World`, mark tool methods with `@tool`, and serve it
with `flock world serve module:Class` (`flock.world.server`):

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
(docs/tools.md, "Contract"); a parameter named `agent` (or annotated `Agent`) receives the
caller and is never shown to the model. Override `manifest()` and `call()` to serve tools some
other way. The hooks `on_agent_start`, `on_agent_done` and `on_turn` are sent only when a
subclass overrides them (the manifest lists them in `hooks`).
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from flock.world.protocol import HOOK_EVENTS, WORLD_API

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
    """Raise from a tool: the agent reads `message` as the tool's error result."""

    def __init__(self, message: str, code: str = "tool_error") -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def tool(fn: Callable) -> Callable:
    """Mark a method of a World subclass as a tool."""
    fn.__flock_world_tool__ = True  # type: ignore[attr-defined]
    return fn


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

    def cancellable(self, tool: str) -> bool:
        """May the server cancel this tool's `async` work when its caller disconnects (the
        HTTP client gave up, e.g. on its own timeout)? Default False: the work always runs to
        the end. A world returns True for tools whose work is wasted once nobody reads it."""
        return False

    # ------------------------------------------------------------------ hooks
    def on_agent_start(self, agent: Agent) -> None:
        """An agent got its first prompt on its worker (again after a relaunch)."""

    def on_agent_done(self, agent: Agent, data: dict[str, Any]) -> None:
        """An agent was removed from the swarm (scale down, `flock remove`)."""

    def on_turn(self, agent: Agent, data: dict[str, Any]) -> None:
        """An agent's turn ended: `data` has status, tool_calls, turn."""


__all__ = ["Agent", "ToolError", "World", "tool", "tool_schema"]
