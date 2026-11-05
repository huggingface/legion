"""Tools folder: loader and schema builder, the subprocess runner, bundles, apply-time selection."""

from __future__ import annotations

import io
import json
import tarfile
import textwrap
from pathlib import Path

import pytest

from flock.models import SwarmSpec
from flock.tools import Context
from flock.tools.bundle import (
    MANIFEST_NAME,
    extract,
    load_extracted,
    pack,
    read_manifest,
    version_of,
)
from flock.tools.loader import ToolsError, describe, load_dir, select
from flock.tools.runner import check_args, context_dict, run_tool
from flock.tools.spec import prepare


def write(folder: Path, name: str, src: str) -> Path:
    p = folder / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(src))
    return p


TYPES_SRC = '''
    from __future__ import annotations

    from typing import Literal, Optional

    from flock.tools import Context


    def everything(s: str, i: int, f: float, b: bool, tags: list[str], d: dict,
                   maybe: Optional[int], m2: str | None = None, n: int = 3,
                   mode: Literal["a", "b"] = "a", ctx: Context = None) -> dict:
        """Exercise every annotation type.

        More text that is not the description.

        Args:
            s: a string
            i: an integer,
                continued on the next line
            tags: some tags
        """
        return {}


    def no_doc(x: int) -> int:
        return x


    def _private(x: int) -> int:
        """Hidden."""
        return x


    async def later(x: int) -> int:
        """An async tool."""
        return x * 2


    def typed_ctx(c: Context) -> str:
        """Context by annotation, any name."""
        return c.agent
'''


def test_schema_covers_every_annotation_type(tmp_path):
    write(tmp_path / "tools", "kinds.py", TYPES_SRC)
    m = load_dir(tmp_path / "tools")
    assert m.names() == ["everything", "later", "typed_ctx"]  # no docstring / private: skipped
    t = m.get("everything")
    assert t.description == "Exercise every annotation type."
    assert t.ctx == "ctx" and t.module == "kinds" and not t.is_async
    p = t.parameters
    assert p["type"] == "object" and p["additionalProperties"] is False
    assert p["required"] == ["s", "i", "f", "b", "tags", "d", "maybe"]  # nullable is not omittable
    props = p["properties"]
    assert "ctx" not in props
    assert props["s"] == {"type": "string", "description": "a string"}
    assert props["i"] == {"type": "integer", "description": "an integer, continued on the next line"}
    assert props["f"] == {"type": "number"} and props["b"] == {"type": "boolean"}
    assert props["tags"] == {"type": "array", "items": {"type": "string"}, "description": "some tags"}
    assert props["d"] == {"type": "object"}
    assert props["maybe"] == {"type": ["integer", "null"]}
    assert props["m2"] == {"type": ["string", "null"]}  # default None: optional, null allowed
    assert props["n"] == {"type": "integer", "default": 3}
    assert props["mode"] == {"enum": ["a", "b"], "type": "string", "default": "a"}
    assert m.get("later").is_async
    assert m.get("typed_ctx").ctx == "c" and m.get("typed_ctx").parameters["properties"] == {}
    assert m.get("everything").openai()["function"]["name"] == "everything"


@pytest.mark.parametrize("src, msg", [
    ("def f(x):\n    'Doc.'\n", "f(x): needs a type annotation"),
    ("def f(x: set) -> int:\n    'Doc.'\n", "unsupported type"),
    ("def f(x: int | str) -> int:\n    'Doc.'\n", "unions other than"),
    ("def f(*xs: int) -> int:\n    'Doc.'\n", "*args / **kwargs"),
    ("def bash(x: int) -> int:\n    'Doc.'\n", "'bash' is reserved"),
    ("def f(x: int -> int:\n    'Doc.'\n", "bad.py: syntax error line 1"),
    ("import nonexistent_pkg_xyz\n", "bad.py: import failed: ModuleNotFoundError"),
    ("raise RuntimeError('boom at import')\n", "bad.py: import failed: RuntimeError: boom at import"),
])
def test_bad_module_gives_a_clear_error(tmp_path, src, msg):
    write(tmp_path / "tools", "bad.py", src)
    with pytest.raises(ToolsError, match=msg.replace("(", r"\(").replace(")", r"\)").replace("*", r"\*")):
        describe(tmp_path / "tools")


def test_duplicate_names_and_shadowed_modules(tmp_path):
    write(tmp_path / "a" / "tools", "one.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n")
    write(tmp_path / "a" / "tools", "two.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n")
    with pytest.raises(ToolsError, match="'f' is also defined in one.py"):
        describe(tmp_path / "a" / "tools")
    write(tmp_path / "b" / "tools", "json.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n")
    with pytest.raises(ToolsError, match="json.py: module name 'json' is shadowed"):
        describe(tmp_path / "b" / "tools")


def test_helpers_and_imported_functions_are_not_tools(tmp_path):
    folder = tmp_path / "tools"
    write(folder, "_shared.py", "def helper(x: int) -> int:\n    'Doc.'\n    return x + 1\n")
    write(folder, "main.py", """
        from os.path import join
        from _shared import helper

        def use(x: int) -> int:
            '''Uses a helper module.'''
            return helper(x)
    """)
    print_src = "print('noise at import')\ndef loud() -> str:\n    'Doc.'\n    return 'ok'\n"
    write(folder, "loud.py", print_src)
    assert describe(folder).names() == ["loud", "use"]


def test_readonly_tools_are_flagged_in_the_manifest(tmp_path):
    """`@flock.tools.readonly` puts `readonly: true` in the manifest; other tools leave the key
    out, so a folder without the flag keeps its manifest and version."""
    folder = tmp_path / "tools"
    write(folder, "board.py", """
        from flock.tools import readonly

        @readonly
        def task_board() -> str:
            '''Show the task board.'''
            return "ok"

        def claim(task: str) -> str:
            '''Claim a task.'''
            return task
    """)
    m = describe(folder)
    assert {t.name: t.readonly for t in m.tools} == {"task_board": True, "claim": False}
    assert {t["name"]: t.get("readonly") for t in m.to_json()["tools"]} == {"task_board": True,
                                                                       "claim": None}
    from flock.comms.tools import tool_def as comms_tool
    assert [comms_tool(e).readonly for e in ("comms:post", "comms:read", "comms:search")] == [
        False, True, True]


def test_select_is_strict():
    from flock.tools.loader import Manifest, ToolDef

    m = Manifest([ToolDef("add", "", {}, "m", "add"), ToolDef("sub", "", {}, "m", "sub")])
    assert [t.name for t in select(m, "all")] == ["add", "sub"]
    assert [t.name for t in select(m, ["sub"])] == ["sub"]
    assert select(m, None) == []
    with pytest.raises(ToolsError, match="types.dev.tools: unknown tool.s. mul; the tools folder has: add, sub"):
        select(m, ["add", "mul"], where="types.dev.tools")


NULLABLE_SRC = """
from typing import Literal


def opt(x: int | None, tags: list[int | None], mode: Literal["a", "b"] | None,
        y: int | None = None) -> str:
    "Nullable parameters."
    return f"{x}/{tags}/{mode}/{y}"


def plain(x: int, xs: list[int]) -> str:
    "Not nullable."
    return "body ran"
"""


def test_nullable_means_null_accepted_not_omittable(tmp_path):
    m = load_dir(write(tmp_path / "tools", "n.py", NULLABLE_SRC).parent)
    opt, plain = m.get("opt").parameters, m.get("plain").parameters
    assert opt["required"] == ["x", "tags", "mode"]
    assert opt["properties"]["tags"]["items"] == {"type": ["integer", "null"]}
    assert opt["properties"]["mode"] == {"enum": ["a", "b", None], "type": ["string", "null"]}
    assert "missing required argument(s): x" in check_args(opt, {"tags": [], "mode": None})
    assert check_args(opt, {"x": None, "tags": [1, None], "mode": None}) is None
    assert check_args(opt, {"x": 1, "tags": [], "mode": "a", "y": None}) is None
    assert "mode must be one of" in check_args(opt, {"x": 1, "tags": [], "mode": "c"})
    assert check_args(plain, {"x": None, "xs": []}) == "x must be integer, got null"
    assert check_args(plain, {"x": 1, "xs": [1, None]}) == "xs[1] must be integer, got null"


async def test_nullable_calls_match_the_schema(tmp_path):
    folder = write(tmp_path / "tools", "n.py", NULLABLE_SRC).parent
    m = load_dir(folder)
    wd = tmp_path / "wd"
    r = await run_tool(m.get("opt"), {"x": None, "tags": [None], "mode": None}, tools_dir=folder,
                       workdir=wd)
    assert (r.output, r.error) == ("None/[None]/None/None", False)
    r = await run_tool(m.get("opt"), {"tags": [], "mode": "a"}, tools_dir=folder, workdir=wd)
    assert r.error and "missing required argument(s): x" in r.output
    r = await run_tool(m.get("plain"), {"x": None, "xs": []}, tools_dir=folder, workdir=wd)
    assert r.error and "x must be integer, got null" in r.output


def test_check_args():
    params = {"type": "object", "properties": {"a": {"type": "integer"}, "xs": {
        "type": "array", "items": {"type": "string"}}, "f": {"type": "number"}},
        "required": ["a"]}
    assert check_args(params, {"a": 1, "xs": ["x"], "f": 1}) is None
    assert "missing required argument(s): a" in check_args(params, {})
    assert "unknown argument(s) z" in check_args(params, {"a": 1, "z": 2})
    assert "a must be integer, got bool" in check_args(params, {"a": True})
    assert "xs[1] must be string" in check_args(params, {"a": 1, "xs": ["x", 2]})
    assert "must be a JSON object" in check_args(params, ["a"])


# --------------------------------------------------------------------------- runner


RUN_SRC = '''
    import os
    import sys
    import time

    from flock.tools import Context, ToolError


    def add(a: int, b: int) -> int:
        """Add."""
        print("this goes to stderr, not into the result")
        return a + b


    def text(s: str) -> str:
        """Strings come back as is."""
        return s.upper()


    def boom(x: int) -> int:
        """Raises."""
        raise ValueError(f"bad x {x}")


    def refuse() -> str:
        """Raises a ToolError."""
        raise ToolError("not allowed right now")


    def slow(seconds: float) -> str:
        """Sleeps."""
        time.sleep(seconds)
        return "slept"


    def big(n: int) -> str:
        """Returns n bytes."""
        return "x" * n


    def die() -> str:
        """Kills its own process."""
        os._exit(3)


    def leaves_child() -> str:
        """Starts a background process that keeps stderr open."""
        import subprocess
        subprocess.Popen(["sleep", "30"])
        return "left one"


    def who(ctx: Context) -> dict:
        """The context and the cwd."""
        (ctx.data_dir / "note").write_text("kept")
        return {"agent": ctx.agent, "type": ctx.type, "swarm": ctx.swarm, "cwd": os.getcwd(),
                "workdir": str(ctx.workdir), "version": ctx.tools_version,
                "env": os.environ.get("MY_VAR"), "ctx_env": ctx.env.get("MY_VAR"),
                "path0": sys.path[0]}


    async def later(x: int) -> dict:
        """Async."""
        return {"x": x * 2}
'''


@pytest.fixture
def run_env(tmp_path):
    folder = write(tmp_path / "tools", "t.py", RUN_SRC).parent
    m = load_dir(folder)
    wd = tmp_path / "work"
    return m, folder, wd


async def test_runner_results_and_errors(run_env, tmp_path):
    m, folder, wd = run_env
    r = await run_tool(m.get("add"), {"a": 2, "b": 3}, tools_dir=folder, workdir=wd)
    assert (r.output, r.error) == ("5", False)
    r = await run_tool(m.get("text"), {"s": "hi"}, tools_dir=folder, workdir=wd)
    assert r.output == "HI"
    r = await run_tool(m.get("later"), {"x": 4}, tools_dir=folder, workdir=wd)
    assert json.loads(r.output) == {"x": 8}
    r = await run_tool(m.get("boom"), {"x": 7}, tools_dir=folder, workdir=wd)
    assert r.error and json.loads(r.output) == {"error": "ValueError: bad x 7"}
    r = await run_tool(m.get("refuse"), {}, tools_dir=folder, workdir=wd)
    assert r.error and json.loads(r.output) == {"error": "not allowed right now"}
    r = await run_tool(m.get("add"), {"a": "2", "b": 3}, tools_dir=folder, workdir=wd)
    assert r.error and "a must be integer" in json.loads(r.output)["error"]
    r = await run_tool(m.get("die"), {}, tools_dir=folder, workdir=wd)
    assert r.error and "the tool process died" in json.loads(r.output)["error"]


async def test_runner_timeout_kills_the_call(run_env):
    m, folder, wd = run_env
    r = await run_tool(m.get("slow"), {"seconds": 30}, tools_dir=folder, workdir=wd, timeout=0.5)
    assert r.error and json.loads(r.output) == {"error": "slow: timed out after 0.5s and was killed"}
    assert r.duration_s < 5


async def test_runner_caps_the_result_size(run_env):
    m, folder, wd = run_env
    r = await run_tool(m.get("big"), {"n": 5000}, tools_dir=folder, workdir=wd, max_bytes=10_000)
    assert not r.error and len(r.output) == 5000
    r = await run_tool(m.get("big"), {"n": 3_000_000}, tools_dir=folder, workdir=wd, max_bytes=10_000)
    assert r.error and "over the 10000-byte limit" in r.output


async def test_runner_does_not_wait_for_leftover_processes(run_env):
    m, folder, wd = run_env
    r = await run_tool(m.get("leaves_child"), {}, tools_dir=folder, workdir=wd, timeout=10)
    assert (r.output, r.error) == ("left one", False) and r.duration_s < 5


async def test_runner_context_cwd_env_and_isolation(run_env, tmp_path):
    m, folder, wd = run_env
    wd.mkdir(parents=True)
    (wd / "t.py").write_text("raise SystemExit('the agent planted this')\n")  # never imported
    data = tmp_path / "data"
    data.mkdir()
    ctx = Context(agent="dev-0001", type="dev", swarm="s1", workdir=wd, data_dir=data,
                  tools_dir=folder, tools_version="0123456789abcdef", env={"MY_VAR": "v"})
    r = await run_tool(m.get("who"), {}, tools_dir=folder, workdir=wd, ctx=context_dict(ctx),
                       env={"MY_VAR": "v", "PATH": "/usr/bin:/bin"})
    assert not r.error, r.output
    out = json.loads(r.output)
    assert out == {"agent": "dev-0001", "type": "dev", "swarm": "s1", "cwd": str(wd),
                   "workdir": str(wd), "version": "0123456789abcdef", "env": "v", "ctx_env": "v",
                   "path0": str(folder.resolve())}
    assert (data / "note").read_text() == "kept"


# --------------------------------------------------------------------------- bundles


def test_pack_is_deterministic_and_excludes_caches(tmp_path):
    folder = write(tmp_path / "tools", "t.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n").parent
    describe(folder)  # runs with -B: writes no __pycache__ into the folder
    (folder / "__pycache__").mkdir(exist_ok=True)
    (folder / "__pycache__" / "t.cpython-312.pyc").write_bytes(b"junk")
    (folder / ".DS_Store").write_bytes(b"junk")
    write(folder, "data/rules.yaml", "a: 1\n")
    b1, b2 = pack(folder), pack(folder)
    assert b1.version == b2.version == version_of(b1.data) and len(b1.version) == 16
    with tarfile.open(fileobj=io.BytesIO(b1.data), mode="r:gz") as tar:
        names = sorted(tar.getnames())
    assert names == sorted([MANIFEST_NAME, "data/rules.yaml", "t.py"])
    assert read_manifest(b1.data).names() == ["f"]
    (folder / "t.py").write_text("def f(x: int) -> int:\n    'Doc 2.'\n    return x\n")
    assert pack(folder).version != b1.version


def test_pack_refuses_symlinks_and_big_folders(tmp_path, monkeypatch):
    folder = write(tmp_path / "tools", "t.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n").parent
    (folder / "link.py").symlink_to(folder / "t.py")
    with pytest.raises(ToolsError, match="symlinks are not allowed"):
        pack(folder)
    (folder / "link.py").unlink()
    import flock.tools.bundle as bundle_mod

    monkeypatch.setattr(bundle_mod, "MAX_BUNDLE_BYTES", 100)
    with pytest.raises(ToolsError, match="the limit is 100"):
        pack(folder)


def test_extract_checks_hash_and_members(tmp_path):
    folder = write(tmp_path / "tools", "t.py", "def f(x: int) -> int:\n    'Doc.'\n    return x\n").parent
    b = pack(folder)
    with pytest.raises(ToolsError, match="does not match"):
        extract(b.data, tmp_path / "x", "0" * 16)
    dest = extract(b.data, tmp_path / "v" / b.version, b.version)
    assert (dest / "t.py").is_file() and load_extracted(dest).names() == ["f"]
    assert extract(b.data, dest, b.version) == dest  # already there: kept
    evil = io.BytesIO()
    with tarfile.open(fileobj=evil, mode="w:gz") as tar:
        ti = tarfile.TarInfo("../escape.py")
        ti.size = 1
        tar.addfile(ti, io.BytesIO(b"x"))
    with pytest.raises(ToolsError, match="escapes the folder"):
        extract(evil.getvalue(), tmp_path / "evil")
    assert not (tmp_path / "escape.py").exists() and not (tmp_path / "evil").exists()
    with pytest.raises(ToolsError, match="not a tools bundle"):
        read_manifest(b"not a tarball")


# --------------------------------------------------------------------------- apply-time


def spec_with(tmp_path, tools=None, tools_dir=None, **extra) -> tuple[SwarmSpec, Path]:
    t = {"harness": "pi", "model": "m"}
    if tools is not None:
        t["tools"] = tools
    raw = {"swarm": "s", "namespace": "ns", "bucket": "b", "types": {"dev": t}, **extra}
    if tools_dir is not None:
        raw["tools_dir"] = tools_dir
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text("# unused\n")
    return SwarmSpec.model_validate(raw), spec_file


def test_prepare_selection_and_defaults(tmp_path):
    write(tmp_path / "tools", "t.py", "def add(a: int, b: int) -> int:\n    'Add.'\n    return a + b\n")
    spec, f = spec_with(tmp_path)
    assert prepare(spec, f) is None  # no type selects tools: an unrelated tools/ is ignored
    spec, f = spec_with(tmp_path, tools=["add"])
    b = prepare(spec, f)
    assert b is not None and b.manifest.names() == ["add"]
    spec, f = spec_with(tmp_path, tools="all")
    assert prepare(spec, f).version == b.version
    spec, f = spec_with(tmp_path, tools=["add", "mul"])
    with pytest.raises(ToolsError, match="types.dev.tools: unknown tool.s. mul"):
        prepare(spec, f)
    spec, f = spec_with(tmp_path, tools=["add"], tools_dir="elsewhere")
    with pytest.raises(ToolsError, match="types dev select tools, but the tools folder .*elsewhere does not exist"):
        prepare(spec, f)
    write(tmp_path / "other", "o.py", "def mul(a: int, b: int) -> int:\n    'Mul.'\n    return a * b\n")
    spec, f = spec_with(tmp_path, tools=["mul"], tools_dir="other")
    assert prepare(spec, f).manifest.names() == ["mul"]


def test_spec_fields_are_optional_and_checked():
    base = {"swarm": "s", "namespace": "ns", "bucket": "b",
            "types": {"dev": {"harness": "pi", "model": "m"}}}
    s = SwarmSpec.model_validate(base)
    assert s.tools_dir is None and s.tools_version is None and s.types["dev"].tools is None
    with pytest.raises(ValueError, match="tools_version is written by `flock launch`"):
        SwarmSpec.model_validate({**base, "tools_version": "nothex"})
    with pytest.raises(ValueError):
        SwarmSpec.model_validate({**base, "types": {"dev": {"harness": "pi", "model": "m",
                                                            "tools": "some"}}})


# --------------------------------------------------------------------------- the docs example


async def test_example_folder_works(tmp_path):
    """examples/tools-demo (the docs/tools.md example): its spec selects tools the folder has, and
    ws_write + run_bot do what the docs say."""
    import yaml

    from flock.tools.toolset import Toolset

    spec_file = Path(__file__).parent.parent / "examples" / "tools-demo" / "spec.yaml"
    spec = SwarmSpec.model_validate(yaml.safe_load(spec_file.read_text()))
    b = prepare(spec, spec_file)
    assert b is not None and b.manifest.names() == ["ws_write", "run_bot"]
    w = b.manifest.get("ws_write")
    assert w.parameters["required"] == ["filename", "content", "reason"] and w.ctx == "ctx"
    assert w.parameters["properties"]["filename"]["description"].startswith("a plain file name")
    dest = extract(b.data, tmp_path / "tools" / b.version, b.version)
    ts = Toolset(version=b.version, dir=dest, swarm="tools-demo", type="dev",
                 data_dir=tmp_path / "data", tools=tuple(b.manifest.tools))
    wd = tmp_path / "agents" / "dev-0000"
    wd.mkdir(parents=True)
    env = {"PATH": "/usr/bin:/bin"}
    bot = "import sys\nfor line in sys.stdin:\n    print('You said: ' + line.strip())\n"
    r = await ts.call("ws_write", {"filename": "bot.py", "content": bot, "reason": "first bot"},
                      agent="dev-0000", workdir=wd, env=env)
    assert not r.error and json.loads(r.output) == {"written": "ws/bot.py", "chars": len(bot)}
    r = await ts.call("ws_write", {"filename": "bad.py", "content": "def (", "reason": "x"},
                      agent="dev-0000", workdir=wd, env=env)
    assert r.error and "not written: syntax error line 1" in r.output
    r = await ts.call("run_bot", {"path": "ws/bot.py", "messages": ["hi", "bye"]},
                      agent="dev-0000", workdir=wd, env=env)
    assert json.loads(r.output) == {"conversation": [{"user": "hi", "bot": "You said: hi"},
                                                     {"user": "bye", "bot": "You said: bye"}]}
    r = await ts.call("run_bot", {"path": "../../etc/passwd", "messages": ["x"]},
                      agent="dev-0000", workdir=wd, env=env)
    assert r.error and "no such file in your workdir" in r.output
    log = [json.loads(x) for x in (tmp_path / "data" / "writes.jsonl").read_text().splitlines()]
    assert log[0]["agent"] == "dev-0000" and log[0]["tools_version"] == b.version
