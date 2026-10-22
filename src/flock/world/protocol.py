"""World API v1 (docs/world.md): the names both sides agree on and the manifest model.

A world is any HTTP server that answers two routes with a shared secret:

    GET  /manifest            -> {"world_api": "v1", "name", "version", "tools": [...], ...}
    POST /tools/<name>        {"args": {...}} -> {"ok": true, "result": ...}
                                              |  {"ok": false, "error": {"code", "message"}}
    POST /hooks               {"event", "agent", "type", "swarm", "data"}   (optional)

Every request carries `X-World-Token`; a tool call also names the calling agent
(`X-Flock-Agent`, `X-Flock-Type`, `X-Flock-Swarm`, `X-Flock-Turn`, `X-Request-Id`). This module
is pydantic-only, so the worker, the controller and a world written with `flock.world` share it.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

WORLD_API = "v1"

TOKEN_HEADER = "X-World-Token"
AGENT_HEADER = "X-Flock-Agent"
TYPE_HEADER = "X-Flock-Type"
SWARM_HEADER = "X-Flock-Swarm"
TURN_HEADER = "X-Flock-Turn"
REQUEST_HEADER = "X-Request-Id"

# Environment of a world flock starts (and of `flock world serve`), and of the workers.
TOKEN_ENV = "FLOCK_WORLD_TOKEN"
PORT_ENV = "FLOCK_WORLD_PORT"
HOST_ENV = "FLOCK_WORLD_HOST"
SWARM_ENV = "FLOCK_SWARM"

# Hooks flock sends to a world that lists them in its manifest's `hooks` (POST /hooks).
HOOK_EVENTS = ("agent_start", "agent_done", "turn")

WORLD_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
RESERVED = frozenset({"bash", "done"})  # the llm harness's own tools


class WorldTool(BaseModel):
    """One tool: a Chat Completions function (`parameters` is a JSON schema of type object)."""

    model_config = {"extra": "ignore"}

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not TOOL_NAME_RE.match(v):
            raise ValueError(f"tool name {v!r} must match {TOOL_NAME_RE.pattern}")
        if v in RESERVED:
            raise ValueError(f"tool name {v!r} is reserved (the llm harness's own tool)")
        return v

    @field_validator("parameters")
    @classmethod
    def _params(cls, v: dict[str, Any]) -> dict[str, Any]:
        if v.get("type", "object") != "object":
            raise ValueError("parameters must be a JSON schema of type object")
        props = v.get("properties", {})
        if not isinstance(props, dict):
            raise ValueError("parameters.properties must be an object")  # noqa: TRY004 (pydantic)
        return {**v, "type": "object", "properties": props}


class WorldManifest(BaseModel):
    """What `GET /manifest` returns. Unknown fields are ignored (changes within v1 are
    additive); unknown capabilities and hooks are kept but never used."""

    model_config = {"extra": "ignore"}

    world_api: str
    name: str
    version: str
    description: str = ""  # shown to the agents above the tool list
    tools: list[WorldTool] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)
    hooks: list[str] = Field(default_factory=list)  # which of HOOK_EVENTS to send

    @field_validator("world_api")
    @classmethod
    def _api(cls, v: str) -> str:
        if v != WORLD_API:
            raise ValueError(f"world_api {v!r} is not supported (this flock speaks {WORLD_API!r})")
        return v

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        if not WORLD_NAME_RE.match(v):
            raise ValueError(f"name {v!r} must match {WORLD_NAME_RE.pattern}")
        return v

    @field_validator("version")
    @classmethod
    def _version(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("version must not be empty")
        return v

    @field_validator("tools")
    @classmethod
    def _unique(cls, v: list[WorldTool]) -> list[WorldTool]:
        seen: set[str] = set()
        for t in v:
            if t.name in seen:
                raise ValueError(f"tool {t.name!r} is listed twice")
            seen.add(t.name)
        return v

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get(self, name: str) -> WorldTool | None:
        return next((t for t in self.tools if t.name == name), None)


def manifest_error(exc: Exception) -> str:
    """One line from a pydantic ValidationError (or anything else)."""
    errors = getattr(exc, "errors", None)
    if callable(errors):
        parts = []
        for e in errors()[:3]:
            where = ".".join(str(x) for x in e.get("loc", ()))
            parts.append(f"{where + ': ' if where else ''}{e.get('msg')}")
        return "; ".join(parts)
    return str(exc)
