"""Tools folder -> tool definitions (docs/tools.md, "Contract").

Every `*.py` file at the top of the folder whose name does not start with `_` is a tool module.
Every function defined in it (not imported) whose name does not start with `_` and that has a
docstring is a tool: name = function name, description = the docstring's first paragraph,
parameters = a JSON schema built from the type hints (descriptions from an `Args:` section).
A parameter named `ctx`, or annotated `flock.tools.Context`, is injected at call time and
never shown to the model.

`load_dir` imports the modules in this process; `describe` runs it in a fresh `python -I`
subprocess (what `flock apply` uses), so a module's side effects, crashes and hangs stay
there. `python -m flock.tools.loader <dir>` prints the manifest as JSON.
"""

from __future__ import annotations

import importlib
import inspect
import json
import re
import subprocess
import sys
import types
import typing
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from flock.tools import Context

NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
RESERVED = frozenset({"bash", "done"})  # the llm harness's own tools
CTX_NAME = "ctx"
DESCRIBE_TIMEOUT_S = 60.0


class ToolsError(ValueError):
    """The tools folder cannot be loaded; the message says which file and why."""


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema, type object
    module: str  # module name inside the folder
    function: str
    ctx: str | None = None  # the parameter that receives the Context, if any
    is_async: bool = False

    def openai(self) -> dict[str, Any]:
        """The Chat Completions function tool."""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description, "parameters": self.parameters}}


@dataclass
class Manifest:
    tools: list[ToolDef] = field(default_factory=list)

    def names(self) -> list[str]:
        return [t.name for t in self.tools]

    def get(self, name: str) -> ToolDef | None:
        return next((t for t in self.tools if t.name == name), None)

    def to_json(self) -> dict[str, Any]:
        return {"tools": [asdict(t) for t in self.tools]}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Manifest:
        return cls([ToolDef(**t) for t in data.get("tools") or []])


# --------------------------------------------------------------------------- schema


def _first_paragraph(doc: str) -> str:
    para: list[str] = []
    for line in doc.strip().splitlines():
        if not line.strip():
            break
        para.append(line.strip())
    return " ".join(para)


_ARGS_HEAD = re.compile(r"^\s*(Args|Arguments|Parameters|Params)\s*:\s*$")
_ARG_LINE = re.compile(r"^\s*\**([A-Za-z_][A-Za-z0-9_]*)\**\s*(\([^)]*\))?\s*:\s*(.*)$")


def arg_docs(doc: str) -> dict[str, str]:
    """Per-parameter descriptions from a Google-style `Args:` section (continuation lines are
    joined; the section ends at a blank line or a less-indented line)."""
    out: dict[str, str] = {}
    lines = doc.splitlines()
    i = next((k for k, line in enumerate(lines) if _ARGS_HEAD.match(line)), None)
    if i is None:
        return out
    head_indent = len(lines[i]) - len(lines[i].lstrip())
    current: str | None = None
    item_indent = None
    for line in lines[i + 1:]:
        if not line.strip():
            break
        indent = len(line) - len(line.lstrip())
        if indent <= head_indent:
            break
        m = _ARG_LINE.match(line)
        if m and (item_indent is None or indent <= item_indent):
            item_indent = indent
            current = m.group(1)
            out[current] = m.group(3).strip()
        elif current is not None:
            out[current] = (out[current] + " " + line.strip()).strip()
    return out


def _is_context(hint: Any) -> bool:
    return hint is Context or (isinstance(hint, type) and issubclass(hint, Context))


def type_schema(hint: Any, where: str) -> tuple[dict[str, Any], bool]:
    """(JSON schema, optional) for a type hint; optional = `X | None` / `Optional[X]`."""
    if hint is inspect.Parameter.empty:
        raise ToolsError(f"{where}: needs a type annotation (str, int, float, bool, list[...], dict)")
    origin = typing.get_origin(hint)
    args = typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        rest = [a for a in args if a is not type(None)]
        if len(rest) != 1 or len(rest) == len(args):
            raise ToolsError(f"{where}: unions other than `X | None` are not supported")
        schema, _ = type_schema(rest[0], where)
        return schema, True
    if hint is typing.Any:
        return {}, False
    if hint is str:
        return {"type": "string"}, False
    if hint is bool:
        return {"type": "boolean"}, False
    if hint is int:
        return {"type": "integer"}, False
    if hint is float:
        return {"type": "number"}, False
    if hint in (list, typing.List) or origin is list:  # noqa: UP006
        items = type_schema(args[0], where)[0] if args else {}
        return {"type": "array", "items": items}, False
    if hint in (dict, typing.Dict) or origin is dict:  # noqa: UP006
        return {"type": "object"}, False
    if origin is typing.Literal:
        kinds = {type(a) for a in args}
        schema: dict[str, Any] = {"enum": list(args)}
        if kinds == {str}:
            schema["type"] = "string"
        elif kinds == {int}:
            schema["type"] = "integer"
        return schema, False
    raise ToolsError(f"{where}: unsupported type {hint!r}; use str, int, float, bool, "
                     "list[...], dict, Literal[...] or X | None")


def tool_def(fn: Any, module: str) -> ToolDef:
    """The ToolDef of one function (raises ToolsError naming module.function)."""
    name = fn.__name__
    where = f"{module}.{name}"
    if not NAME_RE.match(name):
        raise ToolsError(f"{where}: tool names must match {NAME_RE.pattern}")
    if name in RESERVED:
        raise ToolsError(f"{where}: {name!r} is reserved (the llm harness's own tool); rename it")
    doc = inspect.getdoc(fn) or ""
    try:
        hints = typing.get_type_hints(fn)
    except Exception as e:
        raise ToolsError(f"{where}: cannot resolve type hints: {type(e).__name__}: {e}") from e
    sig = inspect.signature(fn)
    docs = arg_docs(doc)
    props: dict[str, Any] = {}
    required: list[str] = []
    ctx = None
    for p in sig.parameters.values():
        hint = hints.get(p.name, p.annotation)
        if p.name == CTX_NAME or _is_context(hint):
            ctx = p.name
            continue
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise ToolsError(f"{where}: *args / **kwargs are not supported; name every parameter")
        if p.kind is p.POSITIONAL_ONLY:
            raise ToolsError(f"{where}: positional-only parameters are not supported")
        schema, optional = type_schema(hint, f"{where}({p.name})")
        schema = dict(schema)
        if docs.get(p.name):
            schema["description"] = docs[p.name]
        if p.default is not p.empty:
            try:
                json.dumps(p.default)
                if p.default is not None:
                    schema["default"] = p.default
            except (TypeError, ValueError):
                pass
        elif not optional:
            required.append(p.name)
        props[p.name] = schema
    params: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        params["required"] = required
    return ToolDef(name=name, description=_first_paragraph(doc), parameters=params, module=module,
                   function=name, ctx=ctx, is_async=inspect.iscoroutinefunction(fn))


def tool_modules(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.py") if not p.name.startswith("_"))


_LOADED: set[str] = set()


def load_dir(folder: str | Path) -> Manifest:
    """Import every tool module of `folder` (put on sys.path first, so modules may import their
    `_helpers`) and collect the tools. Raises ToolsError on a bad module or a duplicate name."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ToolsError(f"tools folder {folder} does not exist")
    sys.path.insert(0, str(folder))
    for name in _LOADED & set(sys.modules):  # an earlier folder's modules (same process)
        del sys.modules[name]
    importlib.invalidate_caches()
    tools: dict[str, ToolDef] = {}
    for path in tool_modules(folder):
        mod_name = path.stem
        _LOADED.add(mod_name)
        try:
            mod = importlib.import_module(mod_name)
        except SyntaxError as e:
            raise ToolsError(f"{path.name}: syntax error line {e.lineno}: {e.msg}") from e
        except Exception as e:
            raise ToolsError(f"{path.name}: import failed: {type(e).__name__}: {e}") from e
        if Path(getattr(mod, "__file__", "") or "").resolve() != path:
            raise ToolsError(f"{path.name}: module name {mod_name!r} is shadowed by "
                             f"{getattr(mod, '__file__', None) or 'a built-in module'}; rename the file")
        for attr, fn in vars(mod).items():
            if attr.startswith("_") or not inspect.isfunction(fn) or fn.__module__ != mod.__name__:
                continue
            if not inspect.getdoc(fn):
                continue  # no docstring: a helper, not a tool
            t = tool_def(fn, mod_name)
            if t.name in tools:
                raise ToolsError(f"{path.name}: tool {t.name!r} is also defined in "
                                 f"{tools[t.name].module}.py")
            tools[t.name] = t
    return Manifest(list(tools.values()))


def describe(folder: str | Path, *, python: str | None = None,
             timeout: float = DESCRIBE_TIMEOUT_S) -> Manifest:
    """`load_dir` in a fresh `python -I` process (cwd: the folder's parent). Raises ToolsError."""
    folder = Path(folder).resolve()
    if not folder.is_dir():
        raise ToolsError(f"tools folder {folder} does not exist")
    cmd = [python or sys.executable, "-I", "-m", "flock.tools.loader", str(folder)]
    try:
        p = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=timeout,
                           cwd=str(folder.parent), stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired as e:
        raise ToolsError(f"loading {folder} took over {timeout:g}s (a module blocks at import?)") from e
    try:
        out = json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout.strip() else None
    except ValueError:
        out = None
    if not isinstance(out, dict):
        tail = (p.stderr or p.stdout or "").strip()[-800:]
        raise ToolsError(f"loading {folder} failed (exit {p.returncode}): {tail}")
    if "error" in out:
        raise ToolsError(out["error"])
    return Manifest.from_json(out)


def select(manifest: Manifest, wanted: list[str] | str | None, *, where: str = "tools") -> list[ToolDef]:
    """The tools a type gets: `all`, a list of names (unknown names raise ToolsError), or none."""
    if wanted is None:
        return []
    if wanted == "all":
        return list(manifest.tools)
    if isinstance(wanted, str):
        raise ToolsError(f"{where}: use a list of tool names or `all`, not {wanted!r}")
    unknown = [n for n in wanted if manifest.get(n) is None]
    if unknown:
        known = ", ".join(manifest.names()) or "none"
        raise ToolsError(f"{where}: unknown tool(s) {', '.join(unknown)}; the tools folder has: {known}")
    return [t for n in dict.fromkeys(wanted) if (t := manifest.get(n)) is not None]


def _main(argv: list[str]) -> int:
    if len(argv) != 1:
        print(json.dumps({"error": "usage: python -m flock.tools.loader <tools dir>"}))
        return 2
    import contextlib
    import io

    sink = io.StringIO()  # prints at import time must not corrupt the JSON line
    try:
        with contextlib.redirect_stdout(sink):
            manifest = load_dir(argv[0])
        out: dict[str, Any] = manifest.to_json()
    except ToolsError as e:
        out = {"error": str(e)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
