"""Run one tool call in a subprocess (docs/tools.md, "Limits").

The parent (`run_tool`) validates the arguments against the tool's schema, starts
`<python> -I -m flock.tools.runner` in the agent's workdir with the agent's environment, sends
the call as JSON on stdin and reads one JSON line back. The call is killed (its whole process
group) after `timeout` seconds; a result over `max_bytes` is refused; prints go to stderr, whose
tail is kept for error messages. Whatever the tool does, the worker only sees a `Result`.

`-I` keeps the agent's workdir (the cwd) and PYTHON* variables off the import path, so a file the
agent wrote cannot shadow a module; the tools folder is put first on sys.path instead.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flock.tools.loader import ToolDef

DEFAULT_TIMEOUT_S = 60.0
RESULT_MAX_BYTES = 1_000_000
STDERR_TAIL = 2_000


@dataclass
class Result:
    output: str  # what the model sees: the tool's return value, or {"error": ...}
    error: bool
    duration_s: float = 0.0


def _err(msg: str, t0: float) -> Result:
    return Result(json.dumps({"error": msg}, ensure_ascii=False), True, time.monotonic() - t0)


_JSON_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,), "integer": (int,), "number": (int, float), "boolean": (bool,),
    "array": (list,), "object": (dict,),
}


def _check(schema: dict[str, Any], value: Any, where: str) -> str | None:
    if value is None:
        return None  # `X | None` parameters; a required one fails inside the call if it matters
    t = schema.get("type")
    if isinstance(t, str) and t in _JSON_TYPES:
        ok = isinstance(value, _JSON_TYPES[t]) and not (t in ("integer", "number") and isinstance(value, bool))
        if not ok:
            return f"{where} must be {t}, got {type(value).__name__} {json.dumps(value)[:60]}"
    if "enum" in schema and value not in schema["enum"]:
        return f"{where} must be one of {schema['enum']}, got {json.dumps(value)[:60]}"
    if t == "array" and isinstance(schema.get("items"), dict):
        for i, v in enumerate(value):
            if err := _check(schema["items"], v, f"{where}[{i}]"):
                return err
    return None


def check_args(params: dict[str, Any], args: Any) -> str | None:
    """An error message for arguments that do not fit the tool's schema, else None."""
    if not isinstance(args, dict):
        return f"arguments must be a JSON object, got {type(args).__name__}"
    props = params.get("properties") or {}
    unknown = [k for k in args if k not in props]
    if unknown:
        return f"unknown argument(s) {', '.join(unknown)}; expected: {', '.join(props) or 'none'}"
    missing = [k for k in params.get("required") or [] if k not in args]
    if missing:
        return f"missing required argument(s): {', '.join(missing)}"
    for k, v in args.items():
        if err := _check(props[k], v, k):
            return err
    return None


def _kill(proc: asyncio.subprocess.Process) -> None:
    """SIGKILL the call's process group (the tool and anything it started)."""
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(proc.pid, signal.SIGKILL)


async def _reap(proc: asyncio.subprocess.Process) -> None:
    """Kill, then drain the pipes: asyncio's wait() returns only once they are closed, and a
    pipe we stopped reading never reports its end."""
    _kill(proc)

    async def drain(stream: asyncio.StreamReader | None) -> None:
        while stream is not None and await stream.read(65536):
            pass

    with contextlib.suppress(TimeoutError, OSError):
        await asyncio.wait_for(asyncio.gather(drain(proc.stdout), drain(proc.stderr), proc.wait()), 5)


async def run_tool(tool: ToolDef, args: Any, *, tools_dir: str | Path, workdir: str | Path,
                   ctx: dict[str, Any] | None = None, env: dict[str, str] | None = None,
                   python: str | None = None, timeout: float = DEFAULT_TIMEOUT_S,
                   max_bytes: int = RESULT_MAX_BYTES) -> Result:
    """Run `tool(**args)` from the extracted `tools_dir`. `ctx` is the Context as a dict
    (paths as strings), passed when the tool asks for it. Never raises for the tool's sake."""
    t0 = time.monotonic()
    if err := check_args(tool.parameters, args):
        return _err(f"{tool.name}: {err}", t0)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    req = {"tools_dir": str(Path(tools_dir).resolve()), "module": tool.module, "function": tool.function,
           "args": args, "ctx_param": tool.ctx, "ctx": ctx, "is_async": tool.is_async}
    proc = await asyncio.create_subprocess_exec(
        python or sys.executable, "-I", "-m", "flock.tools.runner", cwd=str(workdir),
        env=dict(env) if env is not None else None, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True)
    assert proc.stdin and proc.stdout and proc.stderr
    tail = bytearray()

    async def read_stderr() -> None:
        while chunk := await proc.stderr.read(65536):
            tail.extend(chunk)
            del tail[:-STDERR_TAIL]

    async def read_stdout() -> bytes | None:
        buf = bytearray()
        while chunk := await proc.stdout.read(65536):
            buf.extend(chunk)
            if len(buf) > max_bytes + 256:  # the JSON envelope adds a little
                return None
        return bytes(buf)

    async def call() -> bytes | None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError):
            proc.stdin.write(json.dumps(req).encode())
            await proc.stdin.drain()
            proc.stdin.close()
        return await read_stdout()

    err_task = asyncio.create_task(read_stderr())
    timed_out, out = False, None
    try:
        out = await asyncio.wait_for(call(), timeout)
        if out is not None:
            # stdout closed: the call is over. A process the tool left behind may still hold
            # stderr open, so give the tail a moment, not more.
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(asyncio.shield(err_task), 0.2)
    except TimeoutError:
        timed_out = True
    except asyncio.CancelledError:
        _kill(proc)
        raise
    finally:
        err_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await err_task
    await _reap(proc)  # also ends anything the tool left running in its group
    if timed_out:
        return _err(f"{tool.name}: timed out after {timeout:g}s and was killed", t0)
    if out is None:
        return _err(f"{tool.name}: result is over the {max_bytes}-byte limit", t0)
    try:
        resp = json.loads(out.decode(errors="replace").strip().splitlines()[-1])
    except (ValueError, IndexError):
        why = tail.decode(errors="replace").strip()[-600:] or f"exit code {proc.returncode}"
        return _err(f"{tool.name}: the tool process died: {why}", t0)
    if not resp.get("ok"):
        return _err(str(resp.get("error")), t0)
    result = resp.get("result")
    if len(result.encode()) > max_bytes:
        return _err(f"{tool.name}: result is over the {max_bytes}-byte limit", t0)
    return Result(result, False, time.monotonic() - t0)


def run_tool_sync(tool: ToolDef, args: Any, **kw: Any) -> Result:
    return asyncio.run(run_tool(tool, args, **kw))


# --------------------------------------------------------------------------- child side


def render(value: Any) -> str:
    """A tool's return value as the model sees it: a str as is, anything else as JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)


def _child() -> int:
    out = os.fdopen(os.dup(1), "w", encoding="utf-8")
    os.dup2(2, 1)  # the tool's prints (and its children's) go to stderr
    sys.stdout = sys.stderr

    def reply(obj: dict[str, Any]) -> int:
        out.write(json.dumps(obj, ensure_ascii=False) + "\n")
        out.flush()
        return 0

    try:
        req = json.loads(sys.stdin.read())
        sys.path.insert(0, req["tools_dir"])
        import importlib

        from flock.tools import Context, ToolError

        fn = getattr(importlib.import_module(req["module"]), req["function"])
        args = dict(req.get("args") or {})
        if req.get("ctx_param") and req.get("ctx") is not None:
            c = dict(req["ctx"])
            for k in ("workdir", "data_dir", "tools_dir"):
                c[k] = Path(c[k])
            args[req["ctx_param"]] = Context(**c)
        try:
            value = fn(**args)
            if asyncio.iscoroutine(value):
                value = asyncio.run(value)
        except ToolError as e:
            return reply({"ok": False, "error": str(e)})
        return reply({"ok": True, "result": render(value)})
    except BaseException as e:
        if isinstance(e, KeyboardInterrupt):
            raise
        msg = f"{type(e).__name__}: {e}"
        return reply({"ok": False, "error": msg[:2000]})


def context_dict(ctx: Any) -> dict[str, Any]:
    """A Context (dataclass) as the JSON the child rebuilds it from."""
    d = asdict(ctx)
    for k in ("workdir", "data_dir", "tools_dir"):
        d[k] = str(d[k])
    return d


if __name__ == "__main__":
    sys.exit(_child())
