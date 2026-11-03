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

import hashlib
import hmac
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

# Environment of a world flock starts (and of `python -m flock.world serve`), and of the workers.
TOKEN_ENV = "FLOCK_WORLD_TOKEN"
PORT_ENV = "FLOCK_WORLD_PORT"
HOST_ENV = "FLOCK_WORLD_HOST"
SWARM_ENV = "FLOCK_SWARM"
STATE_DIR_ENV = "FLOCK_WORLD_STATE_DIR"  # knobs.json (World.state_dir)
# The /admin secret of a world (docs/world.md, "Admin API"); never derived from the world
# secret, which every worker holds. Unset: /admin answers 403.
ADMIN_TOKEN_ENV = "FLOCK_WORLD_ADMIN_TOKEN"

# Hooks flock sends to a world that lists them in its manifest's `hooks` (POST /hooks).
HOOK_EVENTS = ("agent_start", "agent_done", "turn")

WORLD_NAME_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
TOOL_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
RESERVED = frozenset({"bash", "done"})  # the llm harness's own tools


class WorkspaceUpload(BaseModel):
    """`x-flock-workspace` on a tool (worlds with workspace-upload-v1): the worker uploads the
    local clone named by argument `arg` (`default` when absent), unless the value is in `skip`
    or starts with `skip_prefix`."""

    arg: str
    default: str | None = None
    skip: list[str] = Field(default_factory=list)
    skip_prefix: str | None = None


class WorldTool(BaseModel):
    """One tool: a Chat Completions function (`parameters` is a JSON schema of type object),
    plus optional flock metadata: `x-flock-workspace` and `x-flock-timeout_s` (the least HTTP
    timeout the worker gives this tool's calls), and `annotations.readOnlyHint` (the tool
    changes nothing: the worker may retry it once)."""

    model_config = {"extra": "ignore", "populate_by_name": True}

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    workspace: WorkspaceUpload | None = Field(None, alias="x-flock-workspace")
    timeout_s: float | None = Field(None, alias="x-flock-timeout_s", gt=0)
    annotations: dict[str, Any] = Field(default_factory=dict)  # MCP-style, e.g. readOnlyHint

    @property
    def read_only(self) -> bool:
        return self.annotations.get("readOnlyHint") is True

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


def credential_id(token: str, origin: str) -> str:
    """Names a (secret, origin) pair without revealing the secret: the controller writes it
    into the world block, a worker compares it with its own secret's before sending it."""
    return hmac.new(token.encode(), f"flock-world-credential\0{origin}".encode(),
                    hashlib.sha256).hexdigest()[:32]


WORLD_HOST_VAR = "${WORLD_HOST}"  # in world.url: the host is read from FLOCK_WORLD_HOST_FILE
WORLD_HOST_FILE_ENV = "FLOCK_WORLD_HOST_FILE"
DEFAULT_PORT = 7860  # what an old host-only host file means (the url's own port)


def fill_host(url: str, var: str, host: str) -> str:
    """`url` with `var` (`${WORLD_HOST}`, `${CTL_HOST}`) replaced by a host file's first word:
    `host:port` (the jobs bind a free port) also replaces the url's port after `var`; an old
    host-only file keeps the url's port (DEFAULT_PORT in every spec)."""
    return re.sub(re.escape(var) + (r"(:\d+)?" if ":" in host else ""), lambda _: host, url)


def origin_matches(origin: str, endpoint: str) -> bool:
    """A secret bound to `origin` may go to `endpoint`: the same URL, or, for a world.url with
    ${WORLD_HOST} (a world whose host moves, docs/slurm.md), that URL with any one host name."""
    if origin == endpoint:
        return True
    if WORLD_HOST_VAR not in origin:
        return False
    pre, _, post = origin.partition(WORLD_HOST_VAR)
    host = endpoint[len(pre):len(endpoint) - len(post)] if endpoint.startswith(pre) and \
        endpoint.endswith(post) and len(endpoint) > len(pre) + len(post) else ""
    return bool(re.fullmatch(r"[A-Za-z0-9.-]+(:[0-9]+)?", host))


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
