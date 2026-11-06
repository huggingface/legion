"""Tools folder shipping: `flock launch` packs and uploads it, the controller stores and checks it,
assigns and heartbeats carry the version, workers fetch and switch versions between turns, CLI
harnesses get a Tools section and `flock-tool`, and the llm adapter keeps one version per turn."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from flock import cli
from flock.models import AssignCommand, SwarmSpec, type_spec_rev
from flock.tools.bundle import pack
from flock.tools.toolset import PIN_ENV, SECTION_START, Toolset, read_pin
from flock.worker.agents import AgentRuntime
from flock.worker.events import Outbox
from flock.worker.tools import ToolsManager

from .conftest import FAKE_PROVIDERS, wire
from .worker_fakes import ScriptedAdapter, until

runner = CliRunner()

ADD_V1 = '''
def add(a: int, b: int) -> int:
    """Add two integers (v1)."""
    return a + b
'''
ADD_V2 = ADD_V1 + '''

def mul(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b
'''


def folder(root: Path, src: str = ADD_V1, name: str = "tools") -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "calc.py").write_text(textwrap.dedent(src))
    return d


def raw_spec(tools=None, **extra) -> dict:
    t = {"harness": "llm", "model": "fake:m", "replicas": 1, "agents_per_worker": 2}
    if tools is not None:
        t["tools"] = tools
    return {"swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/sh", "providers": FAKE_PROVIDERS,
            "types": {"dev": t}, **extra}


def write_spec(tmp_path: Path, raw: dict) -> Path:
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    return f


# --------------------------------------------------------------------------- apply + controller


def test_apply_uploads_the_folder_and_pins_its_version(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec(tools=["add"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 0, r.output
    v1 = pack(tmp_path / "tools").version
    assert f"tools: version {v1}" in r.output and ": add" in r.output
    sw = store.get_swarm("demo")["spec"]
    assert sw.tools_version == v1 and sw.types["dev"].tools == ["add"]
    assert store.get_tool_bundle("demo", v1)["tools"] == ["add"]

    # same folder: a resume, same version; changed folder: `launch` refuses, `update` ships it
    assert runner.invoke(cli.app, ["launch", str(f)]).exit_code == 0
    assert store.get_swarm("demo")["spec"].tools_version == v1
    folder(tmp_path, ADD_V2)
    f = write_spec(tmp_path, raw_spec(tools="all"))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1 and "run `flock update` first" in r.output
    r = runner.invoke(cli.app, ["update", str(f)])
    assert r.exit_code == 0, r.output
    v2 = store.get_swarm("demo")["spec"].tools_version
    assert v2 != v1 and store.get_tool_bundle("demo", v2)["tools"] == ["add", "mul"]
    assert [b["version"] for b in store.list_tool_bundles("demo")] == [v2, v1]


def test_apply_refuses_unknown_tools_and_bad_modules(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec(tools=["add", "mul"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1
    assert "types.dev.tools: unknown tool(s) mul; the tools folder has: add" in r.output
    assert store.get_swarm("demo") is None  # nothing applied
    (tmp_path / "tools" / "broken.py").write_text("def f(:\n")
    f = write_spec(tmp_path, raw_spec(tools=["add"]))
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 1 and "broken.py: syntax error line 1" in r.output


def test_existing_specs_are_unchanged(tmp_path, monkeypatch, app, store):
    """No type selects tools: an unrelated tools/ next to the spec is not shipped."""
    wire(monkeypatch, app)
    folder(tmp_path)
    f = write_spec(tmp_path, raw_spec())
    r = runner.invoke(cli.app, ["launch", str(f)])
    assert r.exit_code == 0 and "tools:" not in r.output
    assert store.get_swarm("demo")["spec"].tools_version is None
    assert store.list_tool_bundles("demo") == []


async def test_controller_checks_bundles_and_tells_workers(tmp_path, client, store, tick):
    b = pack(folder(tmp_path))
    spec = SwarmSpec.model_validate(raw_spec(tools=["add"]))
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "no tools_version" in r.text
    spec.tools_version = b.version
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "was not uploaded" in r.text
    r = await client.post("/swarms/demo/tools", content=b"garbage")
    assert r.status_code == 400 and "not a tools bundle" in r.text
    r = await client.post("/swarms/demo/tools", content=b.data,
                          headers={"content-type": "application/gzip"})
    assert r.status_code == 201 and r.json() == {"swarm": "demo", "version": b.version,
                                                 "size": len(b.data), "tools": ["add"]}
    spec.types["dev"].tools = ["add", "nope"]
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 400 and "types.dev.tools: unknown tool(s) nope" in r.text
    spec.types["dev"].tools = ["add"]
    r = await client.put("/swarms/demo", json=spec.model_dump(mode="json"))
    assert r.status_code == 200, r.text
    r = await client.get(f"/swarms/demo/tools/{b.version}")
    assert r.status_code == 200 and r.content == b.data
    assert (await client.get("/swarms/demo/tools/0000000000000000")).status_code == 404
    assert (await client.get("/swarms/demo/tools")).json()["current"] == b.version

    # the assign carries the version; the heartbeat response carries version + selection
    tick()
    [w] = store.list_workers("demo")
    [cmd] = [AssignCommand.model_validate_json(c["body"]) for c in store.commands_for(w["id"])]
    assert cmd.tools_version == b.version and cmd.spec.tools == ["add"]
    store.update_worker(w["id"], job_id="job-1", incarnation="inc-1")
    hb = {"worker": w["id"], "deadline": "2026-10-15T09:00:00Z", "agents": [],
          "spec_rev": type_spec_rev(cmd.spec)}  # the worker has this type spec (docs/live.md)
    r = await client.post(f"/workers/{w['id']}/heartbeat", json=hb, params={"incarnation": "inc-1"})
    assert r.status_code == 200, r.text
    assert r.json() == {"ok": True, "tools": {"version": b.version, "select": ["add"]}}


async def test_bundle_retention_keeps_the_current_version(store):
    from flock.controller.store import Store

    spec = SwarmSpec.model_validate({**raw_spec(), "tools_version": f"{0:016x}"})
    store.put_swarm(spec, 0.0)
    for i in range(Store.TOOL_BUNDLES_KEPT + 3):
        store.put_tool_bundle("demo", f"{i:016x}", b"x", [], float(i))
    kept = {b["version"] for b in store.list_tool_bundles("demo")}
    assert len(kept) == Store.TOOL_BUNDLES_KEPT + 1 and f"{0:016x}" in kept
    store.delete_swarm("demo")
    assert store.list_tool_bundles("demo") == []


# --------------------------------------------------------------------------- worker versions


class Fetcher:
    def __init__(self) -> None:
        self.bundles: dict[str, bytes] = {}
        self.calls: list[str] = []
        self.gate: asyncio.Event | None = None
        self.fail = False

    async def __call__(self, version: str) -> bytes:
        self.calls.append(version)
        if self.gate is not None:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("controller unreachable")
        return self.bundles[version]


def manager(tmp_path: Path, fetch: Fetcher) -> ToolsManager:
    return ToolsManager(tmp_path / "agents" / "_tools", swarm="demo", type_name="dev", fetch=fetch)


async def test_manager_switches_only_when_the_new_version_is_ready(tmp_path):
    f = Fetcher()
    b1 = pack(folder(tmp_path / "v1"))
    b2 = pack(folder(tmp_path / "v2", ADD_V2))
    f.bundles = {b1.version: b1.data, b2.version: b2.data}
    m = manager(tmp_path, f)
    assert m.current().version is None and m.current().tools == ()
    await m.ensure(b1.version, ["add"])
    ts1 = m.current()
    assert ts1.version == b1.version and ts1.names() == ["add"]
    assert ts1.dir == tmp_path / "agents" / "_tools" / b1.version and (ts1.dir / "calc.py").is_file()
    assert ts1.data_dir == tmp_path / "agents" / "_tools_data"

    m.want(b1.version, "all")  # same version: the selection applies at once
    assert m.current().names() == ["add"]
    f.gate = asyncio.Event()
    task = m.want(b2.version, ["add", "mul"])
    await asyncio.sleep(0.05)
    assert m.current().version == b1.version  # still fetching: turns keep v1
    f.gate.set()
    await task
    ts2 = m.current()
    assert ts2.version == b2.version and ts2.names() == ["add", "mul"]
    assert ts1.version == b1.version and (ts1.dir / "calc.py").is_file()  # a turn's snapshot stays valid
    r = await ts2.call("mul", {"a": 6, "b": 7}, agent="dev-0000", workdir=tmp_path / "wd", env={})
    assert (r.output, r.error) == ("42", False)
    assert f.calls == [b1.version, b2.version]
    m.want(None, None)
    assert m.current().version is None


async def test_manager_keeps_the_old_version_when_a_fetch_fails(tmp_path):
    f = Fetcher()
    b1 = pack(folder(tmp_path / "v1"))
    f.bundles = {b1.version: b1.data}
    m = manager(tmp_path, f)
    await m.ensure(b1.version, ["add"])
    f.fail = True
    await m.ensure("f" * 16, ["add"])
    ts = m.current()
    assert ts.version == b1.version and "controller unreachable" in ts.error
    f.fail = False
    f.bundles["f" * 16] = b1.data  # wrong content for that hash
    await m.ensure("f" * 16, ["add"])
    assert m.current().version == b1.version and "does not match" in m.current().error


@pytest.mark.skipif(not __import__("shutil").which("uv"), reason="needs uv")
async def test_requirements_get_a_per_version_venv(tmp_path, monkeypatch):
    """A wheel shipped inside the folder and named by a relative path in requirements.txt,
    installed offline into the version's venv from the extracted bundle (the source folder is
    gone by then); the tool imports it and flock.tools at once."""
    import shutil
    import zipfile

    d = folder(tmp_path / "src", "")
    (d / "wheels").mkdir()
    whl = d / "wheels" / "flockdep-0.1-py3-none-any.whl"
    meta = "Metadata-Version: 2.1\nName: flockdep\nVersion: 0.1\n"
    wheel = "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("flockdep/__init__.py", "ANSWER = 41\n")
        z.writestr("flockdep-0.1.dist-info/METADATA", meta)
        z.writestr("flockdep-0.1.dist-info/WHEEL", wheel)
        z.writestr("flockdep-0.1.dist-info/RECORD", "")
    src = '''
        from flock.tools import Context


        def answer(ctx: Context) -> int:
            """Uses the dependency."""
            import flockdep
            return flockdep.ANSWER + 1
    '''
    (d / "calc.py").write_text(textwrap.dedent(src))
    (d / "requirements.txt").write_text("# a wheel shipped in the bundle\n./wheels/flockdep-0.1-py3-none-any.whl\n")
    monkeypatch.setenv("UV_OFFLINE", "1")
    b = pack(d)
    shutil.rmtree(d)  # only the bundle remains
    f = Fetcher()
    f.bundles = {b.version: b.data}
    m = manager(tmp_path, f)
    await m.ensure(b.version, "all")
    ts = m.current()
    assert ts.error is None and ts.python and ts.python.startswith(str(m.root / f"{b.version}.venv"))
    r = await ts.call("answer", {}, agent="dev-0000", workdir=tmp_path / "wd", env=dict(os.environ))
    assert (r.output, r.error) == ("42", False)


# --------------------------------------------------------------------------- CLI harnesses


async def run_tool_cli(tool: Path, args: list[str], env: dict) -> subprocess.CompletedProcess:
    return await asyncio.to_thread(subprocess.run, [str(tool), *args], capture_output=True,
                                   text=True, env=env, timeout=30, check=False)


async def test_cli_harness_gets_a_tools_section_a_pin_and_flock_tool(tmp_path):
    f = Fetcher()
    b1 = pack(folder(tmp_path / "v1"))
    b2 = pack(folder(tmp_path / "v2", ADD_V2))
    f.bundles = {b1.version: b1.data, b2.version: b2.data}
    m = manager(tmp_path, f)
    await m.ensure(b1.version, "all")
    from .worker_fakes import spec as type_spec

    work = tmp_path / "agents"
    adapter = ScriptedAdapter(tmp_path / "home", turn_time=0.05)
    outbox = Outbox()
    rt = AgentRuntime("dev-0000", "dev", type_spec(harness="pi", system_prompt="You are dev."), swarm="demo",
                      work_root=work, adapter=adapter, outbox=outbox, toolbox=m.current)
    await rt.setup(restore=False)
    await rt.prompt("first")
    await until(lambda: rt.state.turns == 1, timeout=5)
    md = (work / "dev-0000" / "AGENTS.md").read_text()
    assert md.startswith("You are dev.") and SECTION_START in md
    assert "- `add`: Add two integers (v1)." in md
    assert """flock-tool add --json '{"a": 1, "b": 1}'""" in md
    env = adapter.envs[0]
    assert env[PIN_ENV] == str(rt.pin_path)
    ts, agent, wd = read_pin(rt.pin_path)
    assert (ts.version, agent, wd) == (b1.version, "dev-0000", work / "dev-0000")
    events = [e for e in outbox.take() if e.type in ("turn_started", "turn_ended")]
    assert [e.data.get("data", e.data).get("tools_version") for e in events] == [b1.version] * 2

    # flock-tool, as the harness's shell would run it
    tool = Path(sys.executable).parent / "flock-tool"
    run_env = {**os.environ, PIN_ENV: str(rt.pin_path)}
    p = await run_tool_cli(tool, ["add", "--json", '{"a": 2, "b": 3}'], run_env)
    assert (p.returncode, p.stdout.strip()) == (0, "5")
    p = await run_tool_cli(tool, ["add", "--json", '{"a": "x", "b": 3}'], run_env)
    assert p.returncode == 1 and "a must be integer" in json.loads(p.stdout)["error"]
    p = await run_tool_cli(tool, ["--list"], run_env)
    assert [t["name"] for t in json.loads(p.stdout)["tools"]] == ["add"]

    # a new version: the next turn pins it, and the next burst rewrites the section
    await m.ensure(b2.version, "all")
    await rt.prompt("second")
    await until(lambda: rt.state.turns == 2, timeout=5)
    assert read_pin(rt.pin_path)[0].version == b2.version
    p = await run_tool_cli(tool, ["mul", "--json", '{"a": 2, "b": 3}'], run_env)
    assert (p.returncode, p.stdout.strip()) == (0, "6")
    await rt.stop()
    await rt.prompt("third")
    await until(lambda: rt.state.turns == 3, timeout=5)
    md = (work / "dev-0000" / "AGENTS.md").read_text()
    assert md.count(SECTION_START) == 1 and "- `mul`: Multiply two integers." in md
    await rt.stop()


async def _cli_runtime(tmp_path: Path, m: ToolsManager, tspec=None) -> tuple[AgentRuntime, ScriptedAdapter]:
    from .worker_fakes import spec as type_spec

    adapter = ScriptedAdapter(tmp_path / "home", turn_time=0.03)
    rt = AgentRuntime("dev-0000", "dev", tspec or type_spec(harness="pi", idle_grace="30s"), swarm="demo",
                      work_root=tmp_path / "agents", adapter=adapter, outbox=Outbox(),
                      toolbox=m.current, secret_source={"PROVIDER_KEY": "dummy-key"})
    await rt.setup(restore=False)
    return rt, adapter


async def test_cli_harness_tools_enabled_then_disabled_in_one_burst(tmp_path):
    f = Fetcher()
    b = pack(folder(tmp_path / "v1"))
    f.bundles = {b.version: b.data}
    m = manager(tmp_path, f)
    await m.ensure(b.version, "all")
    rt, adapter = await _cli_runtime(tmp_path, m)
    tool = Path(sys.executable).parent / "flock-tool"
    try:
        await rt.prompt("first")
        await until(lambda: rt.state.turns == 1, timeout=5)
        env = {**os.environ, **adapter.envs[0]}
        assert (await run_tool_cli(tool, ["add", "--json", '{"a": 2, "b": 3}'], env)).stdout.strip() == "5"
        await m.ensure(None, None)  # the spec drops the tools; the harness keeps running
        await rt.prompt("second")
        await until(lambda: rt.state.turns == 2, timeout=5)
        assert len(adapter.starts) == 1
        p = await run_tool_cli(tool, ["add", "--json", '{"a": 2, "b": 3}'], env)
        assert p.returncode == 1 and json.loads(p.stdout) == {"error": "unknown tool 'add'; tools: none"}
        assert read_pin(rt.pin_path)[0].version is None
        assert SECTION_START not in (tmp_path / "agents" / "dev-0000" / "AGENTS.md").read_text()
    finally:
        await rt.stop()


async def test_cli_harness_tools_enabled_mid_burst(tmp_path):
    f = Fetcher()
    b = pack(folder(tmp_path / "v1"))
    f.bundles = {b.version: b.data}
    m = manager(tmp_path, f)
    rt, adapter = await _cli_runtime(tmp_path, m)
    tool = Path(sys.executable).parent / "flock-tool"
    try:
        await rt.prompt("first")
        await until(lambda: rt.state.turns == 1, timeout=5)
        env = {**os.environ, **adapter.envs[0]}
        assert env[PIN_ENV] == str(rt.pin_path)  # set from the start, before any tools exist
        assert not rt.pin_path.exists()  # an agent that never had tools gets no pin file
        p = await run_tool_cli(tool, ["add", "--json", '{"a": 2, "b": 3}'], env)
        assert p.returncode == 2 and "no tools here" in p.stdout
        await m.ensure(b.version, "all")
        await rt.prompt("second")
        await until(lambda: rt.state.turns == 2, timeout=5)
        assert len(adapter.starts) == 1
        p = await run_tool_cli(tool, ["add", "--json", '{"a": 2, "b": 3}'], env)
        assert (p.returncode, p.stdout.strip()) == (0, "5")
        md = (tmp_path / "agents" / "dev-0000" / "AGENTS.md").read_text()
        assert "- `add`: Add two integers (v1)." in md  # the section follows at the turn boundary
    finally:
        await rt.stop()


async def test_flock_tool_never_passes_the_provider_key(tmp_path):
    """A provider-backed CLI harness gets its key in its env; a tool called through flock-tool
    does not (the same rule as the llm path), nor FLOCK_TOKEN."""
    src = ADD_V1 + '''

def secrets_seen() -> dict:
    """Which credentials the tool process can see (names only)."""
    import os
    return {k: k in os.environ for k in ("PROVIDER_KEY", "FLOCK_TOKEN", "KEEP_ME")}
'''
    f = Fetcher()
    b = pack(folder(tmp_path / "v1", src))
    f.bundles = {b.version: b.data}
    m = manager(tmp_path, f)
    await m.ensure(b.version, "all")
    tspec = SwarmSpec.model_validate({
        "swarm": "demo", "namespace": "ns", "bucket": "hf://buckets/ns/b",
        "providers": {"fake": {"type": "openai_compat", "base_url": "http://127.0.0.1:9/v1",
                               "api_key_env": "PROVIDER_KEY", "self_hosted": True}},
        "types": {"dev": {"harness": "llm", "model": "fake:dummy", "idle_grace": "30s",
                          "turn_timeout": "2s", "env": {"KEEP_ME": "1"}}},
    }).types["dev"]
    rt, adapter = await _cli_runtime(tmp_path, m, tspec)
    try:
        await rt.prompt("first")
        await until(lambda: rt.state.turns == 1, timeout=5)
        env = {**os.environ, **adapter.envs[0], "FLOCK_TOKEN": "worker-secret"}
        assert env["PROVIDER_KEY"] == "dummy-key"  # the harness itself needs it
        tool = Path(sys.executable).parent / "flock-tool"
        p = await run_tool_cli(tool, ["secrets_seen"], env)
        assert json.loads(p.stdout) == {"PROVIDER_KEY": False, "FLOCK_TOKEN": False, "KEEP_ME": True}
    finally:
        await rt.stop()


def test_credential_names_per_harness():
    from flock.worker.adapters.base import credential_env_names

    def t(harness: str, ptype: str, model: str = "p:m") -> object:
        return SwarmSpec.model_validate({
            "swarm": "s", "namespace": "n", "bucket": "b",
            "providers": {"p": {"type": ptype, "base_url": "http://x/v1", "api_key_env": "K",
                                "self_hosted": True}},
            "types": {"a": {"harness": harness, "model": model}}}).types["a"]

    for h in ("pi", "llm"):
        assert credential_env_names(t(h, "openai_compat")) == ["K"]
    assert credential_env_names(t("pi", "openai_compat", "native-model")) == []  # native


class DelayedStartAdapter(ScriptedAdapter):
    """The harness runs its turn at once, but turn_started reaches the worker only on release."""

    def __init__(self, home: Path) -> None:
        super().__init__(home, hang=True)
        self.release = asyncio.Event()

    async def events(self):
        async for ev in super().events():
            if ev.kind == "turn_started":
                await self.release.wait()
            yield ev


async def test_a_prompt_while_awaiting_turn_started_keeps_the_turns_pin(tmp_path):
    """Steering a turn whose turn_started has not arrived yet must not repin it to a version
    that landed meanwhile: the whole harness turn calls one version."""
    f = Fetcher()
    b1 = pack(folder(tmp_path / "v1"))
    b2 = pack(folder(tmp_path / "v2", ADD_V2))
    f.bundles = {b1.version: b1.data, b2.version: b2.data}
    m = manager(tmp_path, f)
    await m.ensure(b1.version, "all")
    from .worker_fakes import spec as type_spec

    adapter = DelayedStartAdapter(tmp_path / "home")
    outbox = Outbox()
    rt = AgentRuntime("dev-0000", "dev", type_spec(idle_grace="30s", turn_timeout="20s"),
                      swarm="demo", work_root=tmp_path / "agents", adapter=adapter, outbox=outbox,
                      toolbox=m.current)
    await rt.setup(restore=False)
    try:
        await rt.prompt("first")
        await until(lambda: len(adapter.sent) == 1, timeout=5)
        assert rt.awaiting and rt.turn_id is None
        assert read_pin(rt.pin_path)[0].version == b1.version
        await m.ensure(b2.version, "all")
        await rt.prompt("steer the same turn")
        assert len(adapter.sent) == 2 and read_pin(rt.pin_path)[0].version == b1.version
        adapter.release.set()
        await until(lambda: rt.turn_id is not None, timeout=5)
        started = [e for e in outbox.take() if e.type == "turn_started"]
        assert [e.data.get("data", e.data).get("tools_version") for e in started] == [b1.version]
        assert read_pin(rt.pin_path)[0].version == b1.version  # unchanged to the turn's end
    finally:
        adapter.release.set()
        await rt.stop()
    # the next turn boundary takes the new version
    await rt.prompt("next")
    await until(lambda: read_pin(rt.pin_path)[0].version == b2.version, timeout=5)
    adapter.release.set()
    await rt.stop()


async def test_restored_workdir_keeps_the_tools_section(tmp_path):
    """A workdir restored with an old AGENTS.md, a spec system_prompt: the base text is written
    first and the generated section is added to it."""
    f = Fetcher()
    b = pack(folder(tmp_path / "v1"))
    f.bundles = {b.version: b.data}
    m = manager(tmp_path, f)
    await m.ensure(b.version, "all")
    from .worker_fakes import spec as type_spec

    wd = tmp_path / "agents" / "dev-0000"
    wd.mkdir(parents=True)
    (wd / "AGENTS.md").write_text("Old instructions from the bucket.\n")
    adapter = ScriptedAdapter(tmp_path / "home", turn_time=0.03)
    rt = AgentRuntime("dev-0000", "dev", type_spec(harness="pi", system_prompt="System instructions.",
                                                   idle_grace="30s"),
                      swarm="demo", work_root=tmp_path / "agents", adapter=adapter,
                      outbox=Outbox(), toolbox=m.current)
    await rt.setup(restore=False)
    try:
        await rt.prompt("first")
        await until(lambda: rt.state.turns == 1, timeout=5)
        md = (wd / "AGENTS.md").read_text()
        assert md.startswith("System instructions.\n\n" + SECTION_START)
        assert "Old instructions" not in md and "- `add`: Add two integers (v1)." in md
    finally:
        await rt.stop()


def test_flock_tool_without_a_pin(tmp_path):
    tool = Path(sys.executable).parent / "flock-tool"
    env = {k: v for k, v in os.environ.items() if k != PIN_ENV}
    p = subprocess.run([str(tool), "add"], capture_output=True, text=True, env=env, timeout=30,
                       check=False)
    assert p.returncode == 2 and "no tools here" in p.stdout


def test_instructions_section_round_trip():
    from flock.tools.loader import ToolDef
    from flock.tools.toolset import example_args, instructions_section, replace_section

    ts = Toolset(version="v", dir=Path("/x"), swarm="s", type="t", data_dir=Path("/d"), tools=(
        ToolDef("run_bot", "Run a bot.", {"type": "object", "properties": {
            "path": {"type": "string"}, "messages": {"type": "array", "items": {"type": "string"}}},
            "required": ["path", "messages"]}, "m", "run_bot"),))
    sec = instructions_section(ts)
    assert """flock-tool run_bot --json '{"path": "...", "messages": ["..."]}'""" in sec
    nullable = ToolDef("f", "F.", {"type": "object", "properties": {
        "n": {"type": ["integer", "null"]}, "xs": {"type": ["array", "null"], "items": {
            "type": ["string", "null"]}}}, "required": ["n", "xs"]}, "m", "f")
    assert example_args(nullable) == {"n": 1, "xs": ["..."]}
    once = replace_section("Hello.\n", sec)
    assert once == f"Hello.\n\n{sec}\n" and replace_section(once, sec) == once
    assert replace_section(once, "") == "Hello.\n"
    assert replace_section("", sec) == sec + "\n"


# --------------------------------------------------------------------------- llm: one version per turn


async def test_llm_turn_keeps_its_version_and_the_next_turn_switches(tmp_path, monkeypatch):
    from flock.worker.adapters.llm import LLMAdapter

    from . import fake_openai
    from .test_worker_llm import KEY_ENV, llm_spec, turn

    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    try:
        slow_v1 = ADD_V1 + '''

def slow(seconds: float) -> str:
    """Sleep, then answer (v1)."""
    import time
    time.sleep(seconds)
    return "slept"
'''
        f = Fetcher()
        b1 = pack(folder(tmp_path / "v1", slow_v1))
        b2 = pack(folder(tmp_path / "v2", slow_v1.replace("(v1)", "(v2)")))
        f.bundles = {b1.version: b1.data, b2.version: b2.data}
        m = manager(tmp_path, f)
        await m.ensure(b1.version, "all")
        wd = tmp_path / "agents" / "dev-0000"
        a = LLMAdapter(wd, agent="dev-0000", home=tmp_path / "llm", backoff_s=0.01, toolbox=m.current)
        spec = llm_spec(f"http://127.0.0.1:{srv.server_port}/v1", tool_choice="required",
                        done_tool=True)
        await a.start(wd, None, spec, env={KEY_ENV: "k"})
        await a.send('call: slow {"seconds": 1.0}\ndone after 2')
        got: list = []

        async def collect():
            got.extend(await turn(a, timeout=30))

        task = asyncio.create_task(collect())
        await until(lambda: any(e.kind == "message" and e.message.get("role") == "assistant"
                                for e in got) or log.exists(), timeout=10)
        await m.ensure(b2.version, "all")  # lands while `slow` runs
        await task
        reqs = [json.loads(x)["body"] for x in log.read_text().splitlines()]
        descs = [{t["function"]["name"]: t["function"]["description"] for t in r["tools"]} for r in reqs]
        assert len(reqs) == 3 and all(d["slow"].endswith("(v1).") for d in descs)
        assert {"bash", "done", "add", "slow"} == set(descs[0])
        started = next(e for e in got if e.kind == "turn_started")
        ended = next(e for e in got if e.kind == "turn_ended")
        assert started.data["tools_version"] == ended.data["tools_version"] == b1.version
        res = next(e.message for e in got if e.kind == "message" and e.message["role"] == "toolResult")
        assert res["toolName"] == "slow" and res["content"][0]["text"] == "slept"
        assert res["toolsVersion"] == b1.version and not res["isError"]

        await a.send('call: add {"a": 1, "b": 2}\ndone after 1')
        evs = await turn(a, timeout=30)
        assert next(e for e in evs if e.kind == "turn_started").data["tools_version"] == b2.version
        last = json.loads(log.read_text().splitlines()[-1])["body"]
        assert {t["function"]["name"]: t["function"]["description"] for t in last["tools"]}[
            "slow"].endswith("(v2).")
        await a.stop()
    finally:
        srv.shutdown()


async def test_llm_tool_errors_and_text_calls(tmp_path, monkeypatch):
    from flock.worker.adapters.llm import LLMAdapter

    from . import fake_openai
    from .test_worker_llm import KEY_ENV, llm_spec, turn

    log = tmp_path / "requests.jsonl"
    srv = fake_openai.serve(0, str(log))
    try:
        boom = ADD_V1 + '''

def boom(x: int) -> int:
    """Always fails."""
    raise ValueError(f"no {x}")
'''
        f = Fetcher()
        b = pack(folder(tmp_path / "v1", boom))
        f.bundles = {b.version: b.data}
        m = manager(tmp_path, f)
        await m.ensure(b.version, ["boom", "add"])
        wd = tmp_path / "agents" / "dev-0000"
        a = LLMAdapter(wd, agent="dev-0000", home=tmp_path / "llm", backoff_s=0.01, toolbox=m.current)
        await a.start(wd, None, llm_spec(f"http://127.0.0.1:{srv.server_port}/v1"), env={KEY_ENV: "k"})
        assert "task tools" in a.messages[0]["content"]
        await a.send('call: boom {"x": 3}')
        evs = await turn(a)
        res = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult")
        assert res["isError"] and json.loads(res["content"][0]["text"]) == {"error": "ValueError: no 3"}
        assert evs[-1].status == "completed"  # a tool error is a result, not a failed turn
        # a small model writing the call as JSON text: run, and stored as a native call
        await a.send('astext: add {"a": 20, "b": 22}')
        evs = await turn(a)
        res = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "toolResult")
        assert res["toolName"] == "add" and res["content"][0]["text"] == "42"
        asst = next(e.message for e in evs if e.kind == "message" and e.message["role"] == "assistant")
        assert asst["toolCallsFromText"] is True
        await a.stop()
    finally:
        srv.shutdown()
