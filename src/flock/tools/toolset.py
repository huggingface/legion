"""A Toolset: the tools one turn may call, from one tools version (docs/tools.md, "Versions").

The worker builds it (flock.worker.tools); the `llm` adapter takes one at the start of each turn
and keeps it for the whole turn, so a version switch never lands mid-turn. `flock-tool` reads the
same thing from the agent's pin file, which the worker rewrites at each turn start.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flock.tools import Context
from flock.tools.loader import ToolDef
from flock.tools.runner import (
    DEFAULT_TIMEOUT_S,
    RESULT_MAX_BYTES,
    Result,
    check_args,
    context_dict,
    run_tool,
)

PIN_ENV = "FLOCK_TOOLS_PIN"  # path of the agent's pin file, in every harness env


@dataclass(frozen=True)
class Toolset:
    version: str | None
    dir: Path | None
    swarm: str
    type: str
    data_dir: Path
    python: str | None = None  # the version's venv (tools/requirements.txt); None: the worker's
    tools: tuple[ToolDef, ...] = ()
    error: str | None = None  # why the wanted version is not (yet) available
    timeout_s: float = DEFAULT_TIMEOUT_S
    max_bytes: int = RESULT_MAX_BYTES
    # Built-in tools (ToolDef.builtin, e.g. comms:post; docs/comms.md) are served by the worker:
    # in process for the `llm` harness (`builtin_call(entry, args)`, never in the pin file), and
    # through the worker's local gateway (`gateway` host:port, this agent's `gateway_key`) for
    # `flock-tool`.
    gateway: str | None = None
    gateway_key: str | None = None
    builtin_call: Callable[[str, dict[str, Any]], Awaitable[Result]] | None = field(
        default=None, compare=False, repr=False)

    def get(self, name: str) -> ToolDef | None:
        return next((t for t in self.tools if t.name == name), None)

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def context(self, agent: str, workdir: Path, env: dict[str, str]) -> Context:
        assert self.dir is not None and self.version is not None
        return Context(agent=agent, type=self.type, swarm=self.swarm, workdir=Path(workdir),
                       data_dir=self.data_dir, tools_dir=self.dir, tools_version=self.version,
                       env=dict(env))

    async def call(self, name: str, args: Any, *, agent: str, workdir: Path,
                   env: dict[str, str]) -> Result:
        """Run tool `name`; an unknown name is an error result, like any other failure."""
        tool = self.get(name)
        if tool is not None and tool.builtin:
            return await self._call_builtin(tool, args, agent)
        if tool is None or self.dir is None:
            known = ", ".join(self.names()) or "none"
            return Result(json.dumps({"error": f"unknown tool {name!r}; tools: {known}"}), True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ctx = context_dict(self.context(agent, workdir, env)) if tool.ctx else None
        return await run_tool(tool, args, tools_dir=self.dir, workdir=workdir, ctx=ctx, env=env,
                              python=self.python, timeout=self.timeout_s, max_bytes=self.max_bytes)

    async def _call_builtin(self, tool: ToolDef, args: Any, agent: str) -> Result:
        assert tool.builtin is not None
        if err := check_args(tool.parameters, args):
            return Result(json.dumps({"error": f"{tool.name}: {err}"}), True)
        if self.builtin_call is not None:
            return await self.builtin_call(tool.builtin, args)
        if self.gateway is None:
            return Result(json.dumps({"error": f"{tool.name}: no worker gateway here"}), True)
        return await gateway_call(self.gateway, self.gateway_key or "", agent, tool.builtin, args,
                                  timeout=self.timeout_s)

    # ------------------------------------------------------------------ pin file
    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("builtin_call", None)
        for k in ("dir", "data_dir"):
            d[k] = None if d[k] is None else str(d[k])
        d["tools"] = [asdict(t) for t in self.tools]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Toolset:
        d = dict(d)
        d["dir"] = Path(d["dir"]) if d.get("dir") else None
        d["data_dir"] = Path(d["data_dir"])
        d["tools"] = tuple(ToolDef(**t) for t in d.get("tools") or [])
        d.pop("builtin_call", None)
        return cls(**d)

    def write_pin(self, path: Path, *, agent: str, workdir: Path,
                  env_deny: list[str] | tuple[str, ...] = ()) -> None:
        """The agent's pin file: this Toolset plus who calls, from where, and the environment
        variables a call must not inherit (the provider key; atomic)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"agent": agent, "workdir": str(workdir), "toolset": self.to_json(),
                                   "env_deny": sorted(set(env_deny))}))
        os.replace(tmp, path)


GATEWAY_LINE_MAX = 4 * 1024 * 1024


async def gateway_call(address: str, key: str, agent: str, entry: str, args: Any, *,
                       timeout: float = DEFAULT_TIMEOUT_S) -> Result:
    """One built-in call through the worker's gateway (flock.worker.comms): a JSON line out,
    a JSON line back, on 127.0.0.1."""
    host, _, port = address.rpartition(":")
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, int(port), limit=GATEWAY_LINE_MAX), 10)
    except (OSError, ValueError, TimeoutError) as e:
        return Result(json.dumps({"error": f"cannot reach the worker gateway: {e}"}), True)
    try:
        writer.write(json.dumps({"agent": agent, "key": key, "tool": entry, "args": args}).encode() + b"\n")
        await writer.drain()
        line = await asyncio.wait_for(reader.readline(), timeout)
        resp = json.loads(line)
        return Result(str(resp["output"]), bool(resp.get("error")), float(resp.get("duration_s") or 0))
    except (OSError, ValueError, KeyError, TimeoutError) as e:
        return Result(json.dumps({"error": f"worker gateway: {type(e).__name__}: {e}"}), True)
    finally:
        writer.close()


def read_pin(path: str | Path) -> tuple[Toolset, str, Path]:
    """(toolset, agent, workdir) from a pin file."""
    d = json.loads(Path(path).read_text())
    return Toolset.from_json(d["toolset"]), d["agent"], Path(d["workdir"])


# Never in a tool call's environment, whatever the harness env holds.
ENV_DENY = ("FLOCK_TOKEN",)


def pin_env_deny(path: str | Path) -> list[str]:
    """The variables a call from this pin must not inherit: the pin's list plus ENV_DENY."""
    d = json.loads(Path(path).read_text())
    return sorted({*ENV_DENY, *(d.get("env_deny") or [])})


def example_args(tool: ToolDef) -> dict[str, Any]:
    """Placeholder arguments for the instructions' example call (required parameters only)."""
    sample = {"string": "...", "integer": 1, "number": 1.0, "boolean": True, "array": [],
              "object": {}}
    out: dict[str, Any] = {}
    props = tool.parameters.get("properties") or {}
    def first(schema: dict[str, Any]) -> str | None:  # ["integer", "null"] -> "integer"
        t = schema.get("type")
        return next((x for x in t if x != "null"), None) if isinstance(t, list) else t

    for name in tool.parameters.get("required") or []:
        p = props.get(name) or {}
        if "enum" in p:
            out[name] = p["enum"][0]
        elif first(p) == "array" and first(p.get("items") or {}) == "string":
            out[name] = ["..."]
        else:
            out[name] = sample.get(first(p) or "", "...")
    return out


SECTION_START = "<!-- flock:tools -->"
SECTION_END = "<!-- /flock:tools -->"


def instructions_section(ts: Toolset) -> str:
    """The generated "Tools" section for CLI harnesses' CLAUDE.md / AGENTS.md ("" without tools)."""
    if not ts.tools:
        return ""
    how = ("Call these with the `flock-tool` command from your shell. It prints the result (JSON "
           "or text) and exits 1 on an error, whose message it prints as `{\"error\": ...}`. "
           "`flock-tool --list` shows each tool's parameters as JSON schema.")
    title = f"## Tools (version {ts.version})" if ts.version else "## Tools"
    lines = [SECTION_START, title, "", how, ""]
    for t in ts.tools:
        args = json.dumps(example_args(t), ensure_ascii=False)
        lines += [f"- `{t.name}`: {t.description or '(no description)'}",
                  f"  Example: `flock-tool {t.name} --json '{args}'`"]
    lines += ["", SECTION_END]
    return "\n".join(lines)


def replace_section(text: str, section: str, start_mark: str = SECTION_START,
                    end_mark: str = SECTION_END) -> str:
    """`text` with its generated section (between the markers) replaced by `section` (appended
    if absent; removed if `section` is empty)."""
    start, end = text.find(start_mark), text.find(end_mark)
    if start != -1 and end != -1 and end > start:
        text = (text[:start].rstrip("\n") + "\n" + text[end + len(end_mark):].lstrip("\n")).strip("\n")
        text = text + "\n" if text else ""
    if not section:
        return text
    return (text.rstrip("\n") + "\n\n" if text.strip() else "") + section + "\n"


__all__ = ["PIN_ENV", "Toolset", "instructions_section", "read_pin", "replace_section"]
