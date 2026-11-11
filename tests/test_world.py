"""Worlds (docs/world.md): the `flock.world` base and server, the manifest and spec models, the
controller's apply (start, health check, refusals), the worker hub (calls, gateway, section,
hooks, secret). A small task-board world (tests/fixtures/taskboard) and a recording world are served in
process with uvicorn on 127.0.0.1; no network beyond loopback."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import random
import re
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
from typer.testing import CliRunner

from flock import cli
from flock.controller.api import create_app as controller_app
from flock.controller.jobs import FakeJobsClient
from flock.controller.reconcile import Settings, reconcile
from flock.models import (
    SwarmSpec,
    TypeSpec,
    WorldSpec,
    world_tools,
)
from flock.world import Agent, ToolError, World, tool
from flock.world.host import WorldError, WorldHost, check_tools, derived_token, fetch_manifest
from flock.world.protocol import TOKEN_ENV, WorldManifest, agent_token
from flock.world.server import create_app

from .conftest import FAKE_PROVIDERS, HB, TOKEN, Clock

ROOT = Path(__file__).resolve().parent.parent
TASKBOARD = Path(__file__).resolve().parent / "fixtures" / "taskboard"
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
    crash = c.post("/tools/crash", json={"args": {}}, headers=H).json()["error"]
    assert crash["code"] == "internal" and "division" not in crash["message"]
    assert re.fullmatch(r"the tool failed with an internal error \(error id [0-9a-f]{8}\)",
                        crash["message"])
    assert c.post("/tools/later", json={"args": {}}, headers=H).json()["result"] == "later dev-0001"
    assert c.post("/hooks", json={"event": "turn", "agent": "dev-0001", "turn": "t-1",
                                  "data": {"status": "completed"}}, headers=H).json() == {"ok": True}
    assert rec.hooks_seen == [("turn", "dev-0001", {"status": "completed", "turn": "t-1"})]
    assert c.post("/hooks", json={"event": "unknown"}, headers=H).json()["ignored"]


def test_an_agent_token_acts_only_as_its_agent_on_manifest_and_tools():
    c = TestClient(create_app(Recorder(), SECRET))
    own = {"X-World-Token": agent_token(SECRET, "dev-0001"), "X-Flock-Agent": "dev-0001"}
    assert c.get("/manifest", headers=own).json()["name"] == "rec"
    assert c.post("/tools/echo", json={"args": {"text": "a"}}, headers=own).json()["result"] \
        == {"text": "a", "by": "dev-0001"}
    for method, path, headers in [("POST", "/tools/echo", {**own, "X-Flock-Agent": "dev-0002"}),
                                  ("GET", "/manifest", {"X-World-Token": own["X-World-Token"]}),
                                  ("GET", "/metrics", own), ("POST", "/hooks", own)]:
        r = c.request(method, path, json={"args": {"text": "a"}}, headers=headers)
        assert r.status_code == 401, (method, path)


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
            "providers": FAKE_PROVIDERS, "types": types or {"dev": {**HB, "model": "fake:m"}}}


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
    no_world = {**raw(dev={**HB, "model": "fake:m", "tools": ["world:claim"]})}
    del no_world["world"]
    with pytest.raises(ValidationError, match="swarm-level `world:` block"):
        SwarmSpec.model_validate(no_world)
    with pytest.raises(ValidationError, match="world: false"):
        SwarmSpec.model_validate(raw(dev={**HB, "model": "fake:m", "world": False,
                                          "tools": ["world:claim"]}))
    with pytest.raises(ValidationError, match="bad world tool"):
        SwarmSpec.model_validate(raw(dev={**HB, "model": "fake:m",
                                          "tools": ["world:no-dash"]}))


def test_world_tools_selection():
    w = WorldSpec(url="http://w", manifest=WorldManifest(
        world_api="v1", name="w", version="1", tools=[{"name": "a"}, {"name": "b"}]))
    ts = TypeSpec(**HB, model="m")
    assert world_tools(ts, w) == ["a", "b"] and world_tools(ts, None) == []
    ts = TypeSpec(**HB, model="m", tools=["ws_write", "world:b", "comms:post"])
    assert world_tools(ts, w) == ["b"] and ts.folder_tools == ["ws_write"]
    assert TypeSpec(**HB, model="m", tools=["world:a"]).folder_tools is None
    assert world_tools(TypeSpec(**HB, model="m", world=False), w) == []


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
    types = types or {"dev": {**HB, "model": "fake:m", "replicas": 2,
                              "agents_per_worker": 2}}
    return SwarmSpec.model_validate({"swarm": "sw", "compute": "local", "world": w,
                                     "providers": FAKE_PROVIDERS, "types": types})


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


async def test_apply_fetches_the_manifest(controller, store, taskboard):
    r = await controller.state.put(world_spec(taskboard.url))
    assert r.status_code == 200, r.text
    stored = store.get_swarm("sw")["spec"]
    assert stored.world.endpoint == taskboard.url and stored.world.manifest.name == "taskboard"
    assert stored.world.manifest.names() == ["list_tasks", "add_task", "claim_task",
                                             "finish_task", "release_task"]
    assert SECRET not in json.dumps(stored.model_dump(mode="json"))
    controller.state.tick()
    (launch,) = controller.state.jobs.launches
    assert TOKEN_ENV not in launch["secrets"]  # the world's secret never reaches a worker


async def test_apply_refuses_a_world_it_cannot_use(controller, taskboard):
    put = controller.state.put
    sel = {"dev": {**HB, "model": "fake:m", "tools": ["world:claim_task", "world:fly"]}}
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
    clash = world_spec(taskboard.url, dev={**HB, "model": "fake:m"})
    clash.world.manifest = fetch_manifest(taskboard.url, SECRET)
    with pytest.raises(WorldError, match=r"world tool\(s\) list_tasks have the name of a folder"):
        check_tools(clash, lambda t: ["list_tasks"])
    check_tools(clash, lambda t: ["ws_write"])


def test_fetch_manifest_reports_an_invalid_manifest():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"world_api": "v1", "name": "w", "version": "1",
                                         "tools": [{"name": "1x"}]})

    with pytest.raises(WorldError, match=r"invalid manifest .*must match"):
        fetch_manifest("http://w", "t", transport=httpx.MockTransport(handler))


def _copy_taskboard(dst: Path) -> Path:
    shutil.copytree(TASKBOARD, dst, ignore=shutil.ignore_patterns("*.db*", "__pycache__"))
    return dst


def test_world_host_starts_keeps_replaces_and_stops_a_command_world(tmp_path):
    d = _copy_taskboard(tmp_path / "tb")
    host = WorldHost(log_dir=tmp_path / "logs")
    w = WorldSpec(dir=str(d), command=["{python}", "-m", "flock.cli", "world", "serve",
                                       "taskboard:TaskBoard"])
    try:
        url, started = host.ensure("sw", w, "tok")
        assert started and host.url("sw") is None  # a candidate until committed
        host.commit("sw")
        assert host.url("sw") == url and fetch_manifest(url, "tok").name == "taskboard"
        assert (d / "taskboard.db").is_file()  # state in the world's dir
        assert host.ensure("sw", w, "tok") == (url, False)  # unchanged: kept
        src = d / "taskboard.py"
        src.write_text(src.read_text().replace('version = "0.1.0"', 'version = "0.2.0"'))
        url2, started = host.ensure("sw", w, "tok")
        # the candidate runs next to the incumbent; a discard leaves the incumbent as it was
        assert started and url2 != url and fetch_manifest(url2, "tok").version == "0.2.0"
        assert fetch_manifest(url, "tok").version == "0.1.0"
        host.discard("sw")
        with pytest.raises(WorldError, match="cannot reach"):
            fetch_manifest(url2, "tok")
        assert host.url("sw") == url and fetch_manifest(url, "tok").version == "0.1.0"
        url3, started = host.ensure("sw", w, "tok")
        old_pid = host._procs["sw"].popen.pid
        host.commit("sw")  # the replacement takes over, then the incumbent stops
        assert host.url("sw") == url3 and fetch_manifest(url3, "tok").version == "0.2.0"
        with pytest.raises(ProcessLookupError):
            os.kill(old_pid, 0)
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


TINY_WORLD = """import json, os
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        data = json.dumps({"world_api": "v1", "name": "tiny", "version": "1",
                           "tools": [{"name": "ping"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(data)))
        self.end_headers(); self.wfile.write(data)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(os.environ["FLOCK_WORLD_PORT"])), H).serve_forever()
"""


async def test_a_refused_reapply_keeps_the_running_world(store, tmp_path):
    """A changed command whose apply is refused (unknown selected tool) leaves the running
    world and the stored endpoint as they were."""
    (tmp_path / "world.py").write_text(TINY_WORLD)
    host = WorldHost(log_dir=tmp_path / "logs", start_timeout=10)
    app = controller_app(store, TOKEN, worlds=host, compute="local", loop_interval=0,
                         secret_source={})
    body = {"swarm": "sw", "compute": "local",
            "world": {"dir": str(tmp_path), "command": [sys.executable, "world.py"]},
            "providers": FAKE_PROVIDERS, "types": {"dev": {**HB, "model": "fake:m", "replicas": 0}}}
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t",
                                     headers={"Authorization": f"Bearer {TOKEN}"}) as c:
            assert (await c.put("/swarms/sw", json=body)).status_code == 200
            endpoint = store.get_swarm("sw")["spec"].world.endpoint
            incumbent = host._procs["sw"].popen
            bad = json.loads(json.dumps(body))
            bad["world"]["command"].append("changed")
            bad["types"]["dev"]["tools"] = ["world:missing"]
            r = await c.put("/swarms/sw", json=bad)
            assert r.status_code == 400 and "unknown world tool(s) missing" in r.text
            assert incumbent.poll() is None and host.url("sw") == endpoint
            assert store.get_swarm("sw")["spec"].world.endpoint == endpoint
            assert (await asyncio.to_thread(httpx.get, f"{endpoint}/manifest")).status_code == 200
            assert not host._pending  # the refused candidate is gone
    finally:
        host.stop_all()


HELPER_WORLD = """import os, subprocess, sys, time
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer
subprocess.Popen([sys.executable, "-c", "import os, signal, time; from pathlib import Path; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "Path('helper.pid').write_text(str(os.getpid())); time.sleep(60)"])
while not Path("helper.pid").exists():
    time.sleep(0.01)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200); self.send_header("Content-Length", "2"); self.end_headers()
        self.wfile.write(b"{}")
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(os.environ["FLOCK_WORLD_PORT"])), H).serve_forever()
"""


def _alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not Path("/proc").is_dir(), reason="needs procfs")
def test_stop_kills_a_helper_that_ignores_sigterm(tmp_path):
    """The world's leader exits on SIGTERM at once; a helper in its process group ignores it
    and is SIGKILLed before the world counts as stopped."""
    (tmp_path / "world.py").write_text(HELPER_WORLD)
    host = WorldHost(log_dir=tmp_path / "logs", start_timeout=10, stop_grace=1.0)
    helper = None
    try:
        host.ensure("sw", WorldSpec(dir=str(tmp_path), command=[sys.executable, "world.py"]), "t")
        host.commit("sw")
        leader = host._procs["sw"].popen
        helper = int((tmp_path / "helper.pid").read_text())
        assert _alive(helper)
        host.stop("sw")
        assert leader.poll() is not None and not _alive(helper)
    finally:
        host.stop_all()
        if helper is not None and _alive(helper):
            os.kill(helper, 9)


async def test_apply_starts_a_command_world_and_remove_stops_it(store, tmp_path):
    d = _copy_taskboard(tmp_path / "tb")
    host = WorldHost(log_dir=tmp_path / "logs")
    app = controller_app(store, TOKEN, secret_source={}, loop_interval=0, worlds=host)
    spec = SwarmSpec.model_validate({
        "swarm": "sw", "compute": "local",
        "world": {"dir": str(d), "command": [sys.executable, "-m", "flock.cli", "world", "serve",
                                             "taskboard:TaskBoard"]},
        "providers": FAKE_PROVIDERS, "types": {"dev": {**HB, "model": "fake:m"}}})
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


# --------------------------------------------------------------------------- changing worlds


class Changing(World):
    """A world whose manifest() and call() are overridden and change at run time."""

    name, version = "changing", "1"

    def manifest(self):
        names = ["first"] if self.version == "1" else ["second"]
        return {"world_api": "v1", "name": self.name, "version": self.version,
                "tools": [{"name": n, "description": "d",
                           "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}}}
                          for n in names]}

    def call(self, tool, args, agent):
        return {"called": tool, "version": self.version}


class AsyncHook(World):
    name = "async-hook"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def on_agent_done(self, agent: Agent, data: dict) -> None:
        await asyncio.sleep(0)
        self.seen.append(agent.name)


def test_server_dispatches_from_the_manifest_it_serves_and_awaits_async_hooks():
    w = Changing()
    c = TestClient(create_app(w, SECRET))
    assert c.post("/tools/first", json={"args": {"x": 1}}, headers=H).json()["ok"]
    w.version = "2"
    assert [t["name"] for t in c.get("/manifest", headers=H).json()["tools"]] == ["second"]
    assert c.post("/tools/second", json={"args": {"x": 2}}, headers=H).json() == {
        "ok": True, "result": {"called": "second", "version": "2"}}
    assert c.post("/tools/first", json={"args": {}}, headers=H).status_code == 404
    assert c.post("/tools/second", json={"args": {"x": "no"}}, headers=H).json()["error"][
        "code"] == "bad_args"
    rec = Recorder()  # the base class: a changed version is served and dispatched alike
    c = TestClient(create_app(rec, SECRET))
    rec.version = "9"
    assert c.get("/manifest", headers=H).json()["version"] == "9"
    hooked = AsyncHook()
    c = TestClient(create_app(hooked, SECRET))
    assert c.post("/hooks", json={"event": "agent_done", "agent": "dev-0001"},
                  headers=H).json() == {"ok": True}
    assert hooked.seen == ["dev-0001"]


def test_tool_results_and_errors_never_carry_secret_env_values(monkeypatch):
    from flock.world.server import redact

    monkeypatch.setenv("HF_TOKEN", "hf_supersecretvalue")
    monkeypatch.setenv("BOARD_INVITE", "invite-code-1234")
    monkeypatch.setenv("SHORT_TOKEN", "abc")  # too short to redact safely
    out = redact({"log": "push with hf_supersecretvalue failed", "n": 3,
                  "nested": ["invite-code-1234", "abc"]})
    assert out == {"log": "push with [redacted] failed", "n": 3, "nested": ["[redacted]", "abc"]}
    assert redact("nothing here") == "nothing here"


# --------------------------------------------------------------------------- doctor

class Doctored(World):
    name, version = "doc", "1.0"

    @tool(read_only=True)
    def list_items(self, agent: Agent, limit: int = 10) -> list:
        """List items.

        Args:
            limit: max items
        """
        return []

    @tool
    def create_item(self, agent: Agent, text: str = "") -> dict:
        """Create an item."""
        return {}

    @tool
    def get_thing(self, agent: Agent) -> dict:
        """Looks like a read, lacks the hint."""
        return {}


def test_doctor_warns_and_passes(capsys):
    s = Served(Doctored())
    try:
        r = CliRunner().invoke(cli.app, ["world", "doctor", s.url, "--token", SECRET, "--calls", "5"])
    finally:
        s.close()
    assert r.exit_code == 0, r.output
    assert "warn: get_thing looks like a read but has no readOnlyHint" in r.output
    assert "warn: write tool create_item has no read tool sharing its noun" not in r.output  # list_items
    assert "warn: write tool get_thing" not in r.output
    assert "list_items: 5 calls" in r.output and "errors none" in r.output
    assert "/manifest during burst" in r.output and "/healthz: 200" in r.output


def test_doctor_flags_a_missing_read_and_a_failing_read_only_tool():
    from flock.world.doctor import doctor

    def handler(request: httpx.Request) -> httpx.Response:
        p = request.url.path
        if p == "/manifest":
            tool_ = lambda n, ro: {"name": n, "annotations": {"readOnlyHint": ro}}
            return httpx.Response(200, json={"world_api": "v1", "name": "w", "version": "1",
                                             "tools": [tool_("open_pr", False), tool_("list_runs", True)]})
        if p.startswith("/tools/"):
            return httpx.Response(200, json={"ok": False, "error": {"code": "backend_down"}})
        return httpx.Response(200, json={})

    lines, failed = doctor("http://w", None, 3, transport=httpx.MockTransport(handler))
    text = "\n".join(lines)
    assert failed
    assert "warn: write tool open_pr has no read tool sharing its noun" in text
    assert "FAIL: list_runs" in text and "backend_down" in text


def test_doctor_manifest_unreachable():
    r = CliRunner().invoke(cli.app, ["world", "doctor", "http://127.0.0.1:1"])
    assert r.exit_code == 1 and "FAIL" in r.output
