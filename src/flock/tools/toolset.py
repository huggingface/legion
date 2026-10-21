"""A Toolset: the tools one turn may call, from one tools version (docs/tools.md, "Versions").

The worker builds it (flock.worker.tools); the `llm` adapter takes one at the start of each turn
and keeps it for the whole turn, so a version switch never lands mid-turn. `flock-tool` reads the
same thing from the agent's pin file, which the worker rewrites at each turn start.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flock.tools import Context
from flock.tools.loader import ToolDef
from flock.tools.runner import DEFAULT_TIMEOUT_S, RESULT_MAX_BYTES, Result, context_dict, run_tool

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
        if tool is None or self.dir is None:
            known = ", ".join(self.names()) or "none"
            return Result(json.dumps({"error": f"unknown tool {name!r}; tools: {known}"}), True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        ctx = context_dict(self.context(agent, workdir, env)) if tool.ctx else None
        return await run_tool(tool, args, tools_dir=self.dir, workdir=workdir, ctx=ctx, env=env,
                              python=self.python, timeout=self.timeout_s, max_bytes=self.max_bytes)

    # ------------------------------------------------------------------ pin file
    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
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
        return cls(**d)

    def write_pin(self, path: Path, *, agent: str, workdir: Path) -> None:
        """The agent's pin file: this Toolset plus who calls and from where (atomic)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps({"agent": agent, "workdir": str(workdir), "toolset": self.to_json()}))
        os.replace(tmp, path)


def read_pin(path: str | Path) -> tuple[Toolset, str, Path]:
    """(toolset, agent, workdir) from a pin file."""
    d = json.loads(Path(path).read_text())
    return Toolset.from_json(d["toolset"]), d["agent"], Path(d["workdir"])


def example_args(tool: ToolDef) -> dict[str, Any]:
    """Placeholder arguments for the instructions' example call (required parameters only)."""
    sample = {"string": "...", "integer": 1, "number": 1.0, "boolean": True, "array": [],
              "object": {}}
    out: dict[str, Any] = {}
    props = tool.parameters.get("properties") or {}
    for name in tool.parameters.get("required") or []:
        p = props.get(name) or {}
        if "enum" in p:
            out[name] = p["enum"][0]
        elif p.get("type") == "array" and (p.get("items") or {}).get("type") == "string":
            out[name] = ["..."]
        else:
            out[name] = sample.get(p.get("type"), "...")
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
    lines = [SECTION_START, f"## Tools (version {ts.version})", "", how, ""]
    for t in ts.tools:
        args = json.dumps(example_args(t), ensure_ascii=False)
        lines += [f"- `{t.name}`: {t.description or '(no description)'}",
                  f"  Example: `flock-tool {t.name} --json '{args}'`"]
    lines += ["", SECTION_END]
    return "\n".join(lines)


def replace_section(text: str, section: str) -> str:
    """`text` with its generated section replaced by `section` (appended if absent; removed if
    `section` is empty)."""
    start, end = text.find(SECTION_START), text.find(SECTION_END)
    if start != -1 and end != -1 and end > start:
        text = (text[:start].rstrip("\n") + "\n" + text[end + len(SECTION_END):].lstrip("\n")).strip("\n")
        text = text + "\n" if text else ""
    if not section:
        return text
    return (text.rstrip("\n") + "\n\n" if text.strip() else "") + section + "\n"


__all__ = ["PIN_ENV", "Toolset", "instructions_section", "read_pin", "replace_section"]
