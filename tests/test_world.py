"""Worlds (docs/world.md): the `flock.world` base and server, the manifest and spec models, the
controller's apply (start, health check, refusals), the worker hub (calls, gateway, section,
hooks, secret). The example world (worlds/taskboard) and a recording world are served in
process with uvicorn on 127.0.0.1; no network beyond loopback."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import random
import shutil
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError

from flock import cli
from flock.controller.api import create_app as controller_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.models import (
    AssignCommand,
    SwarmSpec,
    TypeSpec,
    WorldSpec,
    world_tools,
)
from flock.tools.toolset import Toolset, read_pin
from flock.worker.agents import AgentRuntime
from flock.worker.comms import CommsHub
from flock.worker.events import Outbox
from flock.world import Agent, ToolError, World, tool
from flock.world.host import WorldError, WorldHost, check_tools, derived_token, fetch_manifest
from flock.world.hub import WorldHub
from flock.world.protocol import TOKEN_ENV, WorldManifest
from flock.world.server import create_app

from .conftest import TOKEN, Clock
from .worker_fakes import ScriptedAdapter, spec, until

ROOT = Path(__file__).resolve().parent.parent
TASKBOARD = ROOT / "worlds" / "taskboard"
SECRET = "world-secret"


def load_taskboard():
    s = importlib.util.spec_from_file_location("taskboard", TASKBOARD / "taskboard.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod.TaskBoard


class Recorder(World):
    """Records what flock sends; `version` can be bumped per instance."""

    name, version = "rec", "1.0"

    def __init__(self) -> None:
        self.calls: list[Agent] = []
        self.hooks_seen: list[tuple[str, str, dict]] = []

    @tool
    def echo(self, agent: Agent, text: str, times: int = 1) -> dict:
        """Echo `text` back.

        Args:
            text: what to echo
            times: how often
        """
        self.calls.append(agent)
        return {"text": text * times, "by": agent.name}

    @tool
    def refuse(self, agent: Agent) -> dict:
        """Always fails."""
        raise ToolError("not today", "nope")

    @tool
    def crash(self) -> dict:
        """Raises a bug."""
        return {"x": 1 / 0}

    @tool
    async def later(self, agent: Agent) -> str:
        """An async tool."""
        await asyncio.sleep(0)
        return f"later {agent.name}"

    def on_agent_start(self, agent: Agent) -> None:
        self.hooks_seen.append(("agent_start", agent.name, {}))

    def on_turn(self, agent: Agent, data: dict) -> None:
        self.hooks_seen.append(("turn", agent.name, {**data, "turn": agent.turn}))

    def on_agent_done(self, agent: Agent, data: dict) -> None:
        self.hooks_seen.append(("agent_done", agent.name, data))


class Served:
    """A World served by uvicorn on a thread (127.0.0.1, a free port)."""

    def __init__(self, world: World, token: str | None = SECRET) -> None:
        self.world = world
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        self.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
        self.server = uvicorn.Server(uvicorn.Config(create_app(world, token), log_level="warning"))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [sock]},
                                       daemon=True)
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            assert time.monotonic() < deadline, "world server did not start"
            time.sleep(0.01)

    def close(self) -> None:
        self.server.should_exit = True
        self.thread.join(5)


@pytest.fixture
def recorder():
    s = Served(Recorder())
    yield s
    s.close()


@pytest.fixture
def taskboard(tmp_path):
    s = Served(load_taskboard()(db=str(tmp_path / "tb.db"), seed=str(TASKBOARD / "tasks.txt")))
    yield s
    s.close()


H = {"X-World-Token": SECRET, "X-Flock-Agent": "dev-0001"}


# --------------------------------------------------------------------------- base and server


def test_manifest_from_a_world_subclass():
    m = Recorder().manifest()
    assert m["world_api"] == "v1" and m["name"] == "rec" and m["version"] == "1.0"
    assert [t["name"] for t in m["tools"]] == ["echo", "refuse", "crash", "later"]
    echo = m["tools"][0]
    assert echo["description"] == "Echo `text` back."
    assert echo["parameters"] == {
        "type": "object", "additionalProperties": False, "required": ["text"],
        "properties": {"text": {"type": "string", "description": "what to echo"},
                       "times": {"type": "integer", "description": "how often", "default": 1}}}
    assert m["hooks"] == ["agent_start", "agent_done", "turn"] and "hooks" in m["capabilities"]
    WorldManifest.model_validate(m)
    tb = load_taskboard()
    assert tb.hooks(tb.__new__(tb)) == ["agent_done"]  # only what the subclass overrides


def test_manifest_validation():
    ok = {"world_api": "v1", "name": "w", "version": "1", "tools": [{"name": "a"}]}
    assert WorldManifest.model_validate({**ok, "extra": 1}).tools[0].parameters == {
        "type": "object", "properties": {}}
    for bad, msg in [({"world_api": "v2"}, "not supported"), ({"name": "Bad Name"}, "must match"),
                     ({"tools": [{"name": "a"}, {"name": "a"}]}, "listed twice"),
                     ({"tools": [{"name": "done"}]}, "reserved"),
                     ({"tools": [{"name": "a", "parameters": {"type": "array"}}]}, "type object"),
                     ({"version": " "}, "version")]:
        with pytest.raises(ValidationError, match=msg):
            WorldManifest.model_validate({**ok, **bad})


def test_server_protocol():
    rec = Recorder()
    c = TestClient(create_app(rec, SECRET))
    for method, path in [("GET", "/manifest"), ("POST", "/tools/echo"), ("POST", "/hooks")]:
        r = c.request(method, path, json={}, headers={"X-World-Token": "wrong"})
        assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized"
    assert c.get("/manifest", headers=H).json()["name"] == "rec"
    headers = {**H, "X-Flock-Type": "dev", "X-Flock-Swarm": "sw", "X-Flock-Turn": "t-3",
               "X-Request-Id": "r1"}
    assert c.post("/tools/echo", json={"args": {"text": "ab", "times": 2}}, headers=headers).json() \
        == {"ok": True, "result": {"text": "abab", "by": "dev-0001"}}
    assert rec.calls == [Agent("dev-0001", "dev", "sw", "t-3", "r1")]
    r = c.post("/tools/echo", json={"args": {"text": "x"}}, headers={"X-World-Token": SECRET})
    assert r.status_code == 400 and r.json()["error"]["code"] == "no_agent"
    r = c.post("/tools/nope", json={"args": {}}, headers=H)
    assert r.status_code == 404 and "tools: echo, refuse, crash, later" in r.json()["error"]["message"]
    assert c.post("/tools/echo", json={"args": {"text": 3}}, headers=H).json()["error"] == {
        "code": "bad_args", "message": "echo: text must be string, got int 3"}
    assert c.post("/tools/refuse", json={}, headers=H).json() == {
        "ok": False, "error": {"code": "nope", "message": "not today"}}
    assert c.post("/tools/crash", json={"args": {}}, headers=H).json()["error"] == {
        "code": "internal", "message": "ZeroDivisionError: division by zero"}
    assert c.post("/tools/later", json={"args": {}}, headers=H).json()["result"] == "later dev-0001"
    assert c.post("/hooks", json={"event": "turn", "agent": "dev-0001", "turn": "t-1",
                                  "data": {"status": "completed"}}, headers=H).json() == {"ok": True}
    assert rec.hooks_seen == [("turn", "dev-0001", {"status": "completed", "turn": "t-1"})]
    assert c.post("/hooks", json={"event": "unknown"}, headers=H).json()["ignored"]


def test_taskboard_claims_are_exclusive_and_released_when_an_agent_is_done(tmp_path):
    c = TestClient(create_app(load_taskboard()(db=str(tmp_path / "tb.db"),
                                               seed=str(TASKBOARD / "tasks.txt")), SECRET))
    b = {**H, "X-Flock-Agent": "dev-0002"}

    def call(headers, tool_name, **args):
        return c.post(f"/tools/{tool_name}", json={"args": args}, headers=headers).json()

    assert [t["name"] for t in call(H, "list_tasks")["result"]["tasks"]] == [
        "greeting", "farewell", "small talk"]
    assert call(H, "claim_task", name="greeting")["result"]["owner"] == "dev-0001"
    assert call(b, "claim_task", name="greeting")["error"] == {
        "code": "held", "message": "'greeting' is claimed by dev-0001"}
    assert call(b, "finish_task", name="greeting", note="x")["error"]["code"] == "not_yours"
    assert call(H, "finish_task", name="greeting", note="replies in ws/bot.py")["ok"]
    assert call(b, "claim_task", name="greeting")["error"]["code"] == "done"
    assert call(b, "add_task", name="  jokes  ", description="tell one")["result"]["task"] == "jokes"
    assert call(b, "add_task", name="jokes")["error"]["code"] == "exists"
    assert call(b, "claim_task", name="jokes")["ok"]
    assert call(b, "claim_task", name="nope")["error"]["code"] == "unknown_task"
    c.post("/hooks", json={"event": "agent_done", "agent": "dev-0002", "data": {"reason": "x"}},
           headers=H)
    open_ = [t["name"] for t in call(H, "list_tasks", status="open")["result"]["tasks"]]
    assert "jokes" in open_ and "greeting" not in open_


# --------------------------------------------------------------------------- spec


def raw(world: dict | None = None, **types) -> dict:
    return {"swarm": "sw", "compute": "local",
            "world": world if world is not None else {"command": "flock world serve tb:TB"},
            "types": types or {"dev": {"harness": "claude", "model": "m"}}}


def test_world_spec_validates():
    s = SwarmSpec.model_validate(raw())
    assert s.world.command == ["flock", "world", "serve", "tb:TB"] and s.world.url is None
    for world, msg in [({}, "`command`"), ({"dir": "w", "url": "http://w"}, "give the command"),
                       ({"url": "ftp://w"}, "http"), ({"command": []}, "non-empty"),
                       ({"url": "http://w", "token_env": "no-dash"}, "env var name"),
                       ({"url": "http://w", "port": 1}, "Extra inputs")]:
        with pytest.raises(ValidationError, match=msg):
            SwarmSpec.model_validate(raw(world))
    jobs = {**raw(), "compute": "hf-jobs", "namespace": "ns", "bucket": "hf://buckets/ns/b"}
    with pytest.raises(ValidationError, match="world.url is required for compute hf-jobs"):
        SwarmSpec.model_validate(jobs)
    jobs["world"] = {"command": ["x"], "url": "http://world.invalid/"}  # one block, both scales
    assert SwarmSpec.model_validate(jobs).world.url == "http://world.invalid"
    no_world = {**raw(dev={"harness": "claude", "model": "m", "tools": ["world:claim"]})}
    del no_world["world"]
    with pytest.raises(ValidationError, match="swarm-level `world:` block"):
        SwarmSpec.model_validate(no_world)
    with pytest.raises(ValidationError, match="world: false"):
        SwarmSpec.model_validate(raw(dev={"harness": "claude", "model": "m", "world": False,
                                          "tools": ["world:claim"]}))
    with pytest.raises(ValidationError, match="bad world tool"):
        SwarmSpec.model_validate(raw(dev={"harness": "claude", "model": "m",
                                          "tools": ["world:no-dash"]}))


def test_world_tools_selection():
    w = WorldSpec(url="http://w", manifest=WorldManifest(
        world_api="v1", name="w", version="1", tools=[{"name": "a"}, {"name": "b"}]))
    ts = TypeSpec(harness="claude", model="m")
    assert world_tools(ts, w) == ["a", "b"] and world_tools(ts, None) == []
    ts = TypeSpec(harness="claude", model="m", tools=["ws_write", "world:b", "comms:post"])
    assert world_tools(ts, w) == ["b"] and ts.folder_tools == ["ws_write"]
    assert TypeSpec(harness="claude", model="m", tools=["world:a"]).folder_tools is None
    assert world_tools(TypeSpec(harness="claude", model="m", world=False), w) == []


def test_cli_resolves_world_dir_next_to_the_spec(tmp_path):
    (tmp_path / "world").mkdir()
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw({"dir": "world", "command": ["x"]})))
    assert cli.load_spec(str(f)).world.dir == str((tmp_path / "world").resolve())
    f.write_text(yaml.safe_dump(raw({"command": ["x"]})))
    assert cli.load_spec(str(f)).world.dir == str(tmp_path.resolve())
    f.write_text(yaml.safe_dump(raw({"dir": "missing", "command": ["x"]})))
    with pytest.raises(cli.typer.Exit):
        cli.load_spec(str(f))


# --------------------------------------------------------------------------- controller


def world_spec(url: str, token_env: str | None = "WORLD_TOKEN", **types) -> SwarmSpec:
    w = {"url": url, **({"token_env": token_env} if token_env else {})}
    types = types or {"dev": {"harness": "claude", "model": "m", "replicas": 2,
                              "agents_per_worker": 2}}
    return SwarmSpec.model_validate({"swarm": "sw", "compute": "local", "world": w,
                                     "types": types})


@pytest.fixture
def controller(store):
    clock = Clock()
    secrets = {"WORLD_TOKEN": SECRET}
    app = controller_app(store, TOKEN, clock=clock, secret_source=secrets, loop_interval=0)
    jobs = FakeJobsClient()
    settings = Settings(public_url="http://c", token=TOKEN, secret_source=secrets,
                        rng=random.Random(0))

    async def put(spec: SwarmSpec) -> httpx.Response:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            return await c.put(f"/swarms/{spec.swarm}", json=spec.model_dump(mode="json"))

    app.state.put, app.state.jobs = put, jobs
    app.state.tick = lambda: reconcile(store, jobs, clock(), settings)
    app.state.clock = clock
    return app


async def test_apply_fetches_the_manifest_and_hands_the_world_to_workers(controller, store,
                                                                         taskboard):
    r = await controller.state.put(world_spec(taskboard.url))
    assert r.status_code == 200, r.text
    stored = store.get_swarm("sw")["spec"]
    assert stored.world.endpoint == taskboard.url and stored.world.manifest.name == "taskboard"
    assert stored.world.manifest.names() == ["list_tasks", "add_task", "claim_task",
                                             "finish_task", "release_task"]
    assert SECRET not in json.dumps(stored.model_dump(mode="json"))
    controller.state.tick()
    (launch,) = controller.state.jobs.launches
    assert launch["secrets"][TOKEN_ENV] == SECRET  # a job secret, never in the spec
    wid = launch["worker_id"]
    transport = httpx.ASGITransport(app=controller)
    async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                 headers={"Authorization": f"Bearer {TOKEN}"}) as c:
        batch = (await c.get(f"/workers/{wid}/commands", params={"wait": 0})).json()["commands"]
        assigns = [AssignCommand.model_validate(x) for x in batch if x["kind"] == "assign"]
        assert assigns and all(a.world.manifest.name == "taskboard" for a in assigns)
        hb = {"worker": wid, "deadline": "2030-01-01T00:00:00Z", "draining": False, "agents": []}
        out = (await c.post(f"/workers/{wid}/heartbeat", json=hb)).json()
    assert out["world"]["endpoint"] == taskboard.url and out["world"]["tools"][0] == "list_tasks"


async def test_apply_refuses_a_world_it_cannot_use(controller, taskboard):
    put = controller.state.put
    sel = {"dev": {"harness": "claude", "model": "m", "tools": ["world:claim_task", "world:fly"]}}
    r = await put(world_spec(taskboard.url, **sel))
    assert r.status_code == 400 and "unknown world tool(s) fly; world taskboard has: list_tasks" \
        in r.json()["detail"]
    r = await put(world_spec(taskboard.url, token_env="MISSING"))
    assert r.status_code == 400 and "MISSING is not set on the controller" in r.json()["detail"]
    r = await put(world_spec(taskboard.url, token_env=None))  # no token sent: 401
    assert r.status_code == 400 and "refused the token (401)" in r.json()["detail"]
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        dead = f"http://127.0.0.1:{s.getsockname()[1]}"
    r = await put(world_spec(dead))
    assert r.status_code == 400 and "cannot reach" in r.json()["detail"]
    clash = world_spec(taskboard.url, dev={"harness": "claude", "model": "m"})
    clash.world.manifest = fetch_manifest(taskboard.url, SECRET)
    with pytest.raises(WorldError, match=r"world tool\(s\) list_tasks have the name of a folder"):
        check_tools(clash, lambda t: ["list_tasks"])
    check_tools(clash, lambda t: ["ws_write"])


def test_fetch_manifest_reports_an_invalid_manifest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"world_api": "v1", "name": "w", "version": "1",
                                         "tools": [{"name": "bash"}]})

    with pytest.raises(WorldError, match=r"invalid manifest .*reserved"):
        fetch_manifest("http://w", "t", transport=httpx.MockTransport(handler))


def _copy_taskboard(dst: Path) -> Path:
    shutil.copytree(TASKBOARD, dst, ignore=shutil.ignore_patterns("*.db*", "__pycache__"))
    return dst


def test_world_host_starts_keeps_restarts_and_stops_a_command_world(tmp_path):
    d = _copy_taskboard(tmp_path / "tb")
    host = WorldHost(log_dir=tmp_path / "logs")
    w = WorldSpec(dir=str(d), command=["{python}", "-m", "flock.cli", "world", "serve",
                                       "taskboard:TaskBoard"])
    try:
        url, started = host.ensure("sw", w, "tok")
        assert started and fetch_manifest(url, "tok").name == "taskboard"
        assert (d / "taskboard.db").is_file()  # state in the world's dir
        assert host.ensure("sw", w, "tok") == (url, False)  # unchanged: kept
        src = d / "taskboard.py"
        src.write_text(src.read_text().replace('version = "0.1.0"', 'version = "0.2.0"'))
        url2, started = host.ensure("sw", w, "tok")
        assert started and fetch_manifest(url2, "tok").version == "0.2.0"
        pid = host._procs["sw"].popen.pid
        host.stop("sw")
        assert host.url("sw") is None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        bad = WorldSpec(dir=str(d), command=["{python}", "-c",
                                                    "import sys; print('boom'); sys.exit(3)"])
        with pytest.raises(WorldError, match=r"(?s)exited with 3 .*boom"):
            host.ensure("bad", bad, None)
    finally:
        host.stop_all()


async def test_apply_starts_a_command_world_and_remove_stops_it(store, tmp_path):
    d = _copy_taskboard(tmp_path / "tb")
    host = WorldHost(log_dir=tmp_path / "logs")
    app = controller_app(store, TOKEN, secret_source={}, loop_interval=0, worlds=host)
    spec = SwarmSpec.model_validate({
        "swarm": "sw", "compute": "local",
        "world": {"dir": str(d), "command": [sys.executable, "-m", "flock.cli", "world", "serve",
                                             "taskboard:TaskBoard"]},
        "types": {"dev": {"harness": "claude", "model": "m"}}})
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://t",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            r = await c.put("/swarms/sw", json=spec.model_dump(mode="json"))
            assert r.status_code == 200, r.text
            url = store.get_swarm("sw")["spec"].world.endpoint
            assert url == host.url("sw")
            # the generated secret: derived from the controller's token, never stored
            assert fetch_manifest(url, derived_token(TOKEN, "sw")).name == "taskboard"
            with pytest.raises(WorldError, match="401"):
                fetch_manifest(url, "wrong")
            r = await c.put("/swarms/sw", json=spec.model_dump(mode="json"))  # re-apply: kept
            assert r.status_code == 200 and host.url("sw") == url
            assert (await c.delete("/swarms/sw")).status_code == 202
            assert host.url("sw") is None
    finally:
        host.stop_all()


# --------------------------------------------------------------------------- worker hub


def hub_for(served: Served, names=None, *, token=SECRET, runtime=None) -> WorldHub:
    hub = WorldHub(swarm="sw", type_name="dev", env={TOKEN_ENV: token} if token else {},
                   runtime=runtime)
    m = WorldManifest.model_validate(served.world.manifest())
    hub.configure(WorldSpec(url=served.url, endpoint=served.url, manifest=m),
                  names if names is not None else m.names())
    return hub


async def test_hub_forwards_calls_with_the_agent_and_the_token(recorder):
    hub = hub_for(recorder, ["echo", "refuse", "crash"])
    try:
        r = await hub.call("dev-0001", "world:echo", {"text": "hi"})
        assert not r.error and json.loads(r.output) == {"text": "hi", "by": "dev-0001"}
        a = recorder.world.calls[-1]
        assert (a.name, a.type, a.swarm) == ("dev-0001", "dev", "sw") and len(a.request_id) == 32
        r = await hub.call("dev-0001", "world:refuse", {})
        assert r.error and json.loads(r.output) == {"error": "not today"}
        r = await hub.call("dev-0001", "world:crash", {})
        assert r.error and "ZeroDivisionError" in r.output
        r = await hub.call("dev-0001", "world:echo", {"txt": "hi"})  # refused before the world
        assert r.error and "unknown argument(s) txt" in r.output and len(recorder.world.calls) == 1
        r = await hub.call("dev-0001", "world:later", {})
        assert r.error and "not enabled" in r.output
    finally:
        await hub.aclose()
    bad = hub_for(recorder, token="wrong")
    try:
        r = await bad.call("dev-0001", "world:echo", {"text": "hi"})
        assert r.error and "refused flock's token (401)" in r.output
    finally:
        await bad.aclose()
    gone = hub_for(recorder)
    gone.spec.endpoint = "http://127.0.0.1:9"
    r = await gone.call("dev-0001", "world:echo", {"text": "hi"})
    assert r.error and "unreachable" in r.output
    await gone.aclose()


async def test_hub_retries_429_then_gives_the_result():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.headers.get("x-world-token"))
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"ok": False})
        return httpx.Response(200, json={"ok": True, "text": "fine"})

    hub = WorldHub(swarm="sw", type_name="dev", env={TOKEN_ENV: "t"},
                   transport=httpx.MockTransport(handler))
    m = WorldManifest(world_api="v1", name="w", version="1", tools=[{"name": "a"}])
    hub.configure(WorldSpec(url="http://w", endpoint="http://w", manifest=m), ["a"])
    r = await hub.call("x", "world:a", {})
    assert (r.output, r.error, seen) == ("fine", False, ["t", "t"])
    await hub.aclose()


async def test_world_tools_sit_next_to_comms_tools_and_reach_flock_tool_via_the_gateway(
        tmp_path, recorder):
    comms = CommsHub(swarm="sw", env={})
    world = hub_for(recorder, ["echo"])
    comms.routes["world"] = world.call
    base = Toolset(version=None, dir=None, swarm="sw", type="dev", data_dir=tmp_path / "data")

    async def comms_call(entry, args):
        from flock.tools.runner import Result
        return Result(json.dumps({"comms": entry}), False)

    try:
        await comms.start_gateway()
        ts = world.toolset_for("dev-0001", base, comms)
        assert ts.names() == ["echo"] and ts.get("echo").builtin == "world:echo"
        assert ts.gateway == comms.address and ts.gateway_key == comms.key("dev-0001")
        r = await ts.call("echo", {"text": "in process"}, agent="dev-0001", workdir=tmp_path, env={})
        assert json.loads(r.output)["text"] == "in process"
        # the composed router hands non-world built-ins to the previous handler
        mixed = world.toolset_for("dev-0001", Toolset(**{**base.__dict__,
                                                         "builtin_call": comms_call}), comms)
        assert json.loads((await mixed.builtin_call("comms:post", {})).output) == {
            "comms": "comms:post"}
        # flock-tool: the pin has no in-process handler, the call goes through the gateway
        ts.write_pin(tmp_path / "pin.json", agent="dev-0001", workdir=tmp_path)
        pinned, agent, workdir = read_pin(tmp_path / "pin.json")
        assert pinned.builtin_call is None
        r = await pinned.call("echo", {"text": "via gateway"}, agent=agent, workdir=workdir, env={})
        assert not r.error and json.loads(r.output) == {"text": "via gateway", "by": "dev-0001"}
        forged = Toolset(**{**pinned.__dict__, "gateway_key": "forged"})
        r = await forged.call("echo", {"text": "x"}, agent=agent, workdir=workdir, env={})
        assert r.error and "unknown agent or key" in r.output
    finally:
        await comms.aclose()
        await world.aclose()


async def test_runtime_section_hooks_and_the_secret_stays_in_the_worker(tmp_path, recorder):
    agents: dict[str, AgentRuntime] = {}
    comms = CommsHub(swarm="sw", env={})
    world = hub_for(recorder, runtime=agents.get)
    comms.routes["world"] = world.call
    base = Toolset(version=None, dir=None, swarm="sw", type="dev", data_dir=tmp_path / "data")
    adapter = ScriptedAdapter(tmp_path / "home")
    rt = AgentRuntime("dev-0001", "dev", spec(secrets=[TOKEN_ENV, "OTHER"]), swarm="sw",
                      work_root=tmp_path / "agents", adapter=adapter, outbox=Outbox(),
                      secret_source={TOKEN_ENV: SECRET, "OTHER": "o"},
                      toolbox=lambda: world.toolset_for("dev-0001", base, comms), comms=comms,
                      world=world)
    agents["dev-0001"] = rt
    rec = recorder.world
    try:
        await rt.setup(restore=False)
        await rt.prompt("first")
        await until(lambda: any(h[0] == "turn" for h in rec.hooks_seen))
        env = adapter.envs[0]
        assert TOKEN_ENV not in env and env["OTHER"] == "o"  # listed in secrets, still dropped
        pin = json.loads((tmp_path / "agents" / "_tools_pins" / "dev-0001.json").read_text())
        assert TOKEN_ENV in pin["env_deny"] and pin["toolset"]["gateway"] == comms.address
        assert SECRET not in json.dumps(pin)
        md = (rt.workdir / "CLAUDE.md").read_text()
        assert "## World: rec (version 1.0)" in md and "`echo`, `refuse`, `crash`, `later`" in md
        assert "`flock-tool echo --json" in md and "changed" not in md
        assert rec.hooks_seen[0] == ("agent_start", "dev-0001", {})
        assert rec.hooks_seen[1] == ("turn", "dev-0001", {"status": "completed",
                                                          "tool_calls": 0, "turn": "t-1"})
        # a re-apply with a new manifest version: noted once, at the next turn boundary
        rec.version = "2.0"
        world.configure(WorldSpec(url=recorder.url, endpoint=recorder.url,
                                  manifest=WorldManifest.model_validate(rec.manifest())),
                        ["echo"])
        await until(lambda: rt.turn_id is None)
        await rt.prompt("second")
        await until(lambda: sum(h[0] == "turn" for h in rec.hooks_seen) == 2)
        md = (rt.workdir / "CLAUDE.md").read_text()
        assert "## World: rec (version 2.0)" in md and "(version 1.0 -> 2.0)" in md
        assert "`refuse`" not in md
        await rt.prompt("third")
        await until(lambda: sum(h[0] == "turn" for h in rec.hooks_seen) == 3)
        assert "changed" not in (rt.workdir / "CLAUDE.md").read_text()
        world.on_done("dev-0001", "unassigned")
        await until(lambda: rec.hooks_seen[-1][0] == "agent_done")
        assert rec.hooks_seen[-1] == ("agent_done", "dev-0001", {"reason": "unassigned"})
        assert [h[0] for h in rec.hooks_seen].count("agent_start") == 1
    finally:
        await rt.stop()
        await comms.aclose()
        await world.aclose()
