"""Runtime tools folder: plain Python functions next to the spec become agent tools.

What a tool module imports from here: `Context` (the optional run-context argument, never shown
to the model), `ToolError` (raise it to hand the model a clean error message) and `readonly`
(a decorator: the tool changes nothing; informational). The loader,
schema builder, runner and bundle code live in the submodules; this module stays stdlib-only so
the per-call subprocess starts fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

READONLY_ATTR = "__legion_readonly__"


@dataclass
class Context:
    """What a tool learns about the call. Declare a parameter named `ctx` (or annotated
    `Context`) to receive it; the model never sees it.

    - agent, type, swarm: who is calling (the swarm name is the run id).
    - workdir: the agent's working directory (also the call's cwd).
    - data_dir: a directory for tool state, shared by the agents of this worker, kept across
      tools versions, not synced to the bucket.
    - tools_dir: the extracted tools folder of this version; tools_version: its content hash.
    - env: the agent's environment (what its `bash` sees), also the call's process environment.
    """

    agent: str
    type: str
    swarm: str
    workdir: Path
    data_dir: Path
    tools_dir: Path
    tools_version: str
    env: dict[str, str] = field(default_factory=dict)


Ctx = Context


class ToolError(Exception):
    """Raise from a tool to return `{"error": message}` to the model (no exception type prefix)."""


def readonly[F](fn: F) -> F:
    """Mark a tool as read-only (`readonly: true` in the manifest): it changes no state.
    Informational: nothing in legion acts on it."""
    setattr(fn, READONLY_ATTR, True)
    return fn


__all__ = ["Context", "Ctx", "ToolError", "readonly"]
