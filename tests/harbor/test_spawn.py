"""resources.spawn and the spawn.* tools: a Harbor agent's calls reach the controller through a
real worker's controller_call, with that agent's own token, whatever the request names."""

import asyncio
import json
import shutil
import socket
import sys
from io import StringIO
from pathlib import Path

import httpx
import pytest

import flock.harbor
from flock.controller import localrun
from flock.harbor.spec import Spec, compile_spec
from flock.harbor.tools.bundle import build_tool_bundle
from flock.harbor.tools.runner import load_module
from flock.ops import check_spec, read_spec

from ..test_agent_controller_calls import AGENT, AgentRoutes, gateway_worker
from .test_adapter import TASK
from .test_cli_launch import pins  # noqa: F401 - the fixture

PLUGINS = Path(flock.harbor.__file__).with_name("builtin")
tools = load_module(PLUGINS / "tools" / "spawn.py")
resource_module = load_module(PLUGINS / "resources" / "spawn.py")
SpawnResource = resource_module.Resource


class Refusing(AgentRoutes):
    """Refuses a spawn of type `helper` as the controller's admission does."""

    def handler(self, request):
        if request.url.path.endswith("/spawn") and json.loads(request.content)["type"] == "helper":
            self.agent_calls.append((request.method, request.url.path,
                                     request.headers["authorization"], {}))
            return httpx.Response(409, json={"reason": "max_children", "detail": "3 of 3"})
        return super().handler(request)


@pytest.fixture
async def proxied(tmp_path, monkeypatch):
    """A worker with AGENT assigned, its spawn resource serving AGENT's socket; the tools'
    environment points at that socket (the host side of the Trial's mount)."""
    ctrl = Refusing("controller")
    worker = await gateway_worker(tmp_path, ctrl)
    resource = SpawnResource({}, state_dir=tmp_path, agents=[], secrets=[])
    resource.controller = worker.controller_call  # what HarborHost does
    await resource.start()
    await resource.add([AGENT])
    bound = resource.bind(AGENT)
    monkeypatch.setenv("SPAWN_SOCKET", bound["mounts"][0]["source"])
    monkeypatch.setenv("SPAWN_AGENT", bound["env"]["SPAWN_AGENT"])
    yield ctrl, resource
    await resource.stop()
    await worker.client.aclose()


async def test_spawn_goes_to_the_controller_as_the_agent(proxied):
    ctrl, _ = proxied
    ack = await asyncio.to_thread(tools.spawn, "analyst", prompt="go", key="a1")
    assert ack == {"name": f"{AGENT}-analyst-0001", "created": True}
    assert ctrl.agent_calls == [("POST", f"/agents/{AGENT}/spawn", "Bearer flock_at_leader", {})]
    status = await asyncio.to_thread(tools.status)
    assert status["name"] == AGENT


async def test_a_refusal_comes_back_with_its_reason(proxied):
    with pytest.raises(RuntimeError, match="max_children: 3 of 3"):
        await asyncio.to_thread(tools.spawn, "helper")
    with pytest.raises(RuntimeError, match="HTTP 403: agent leader-0000 may not call"):
        await asyncio.to_thread(tools.status, "someone-else")


async def test_the_socket_names_the_agent_not_the_request(proxied, monkeypatch):
    ctrl, _ = proxied
    monkeypatch.setenv("SPAWN_AGENT", "other-0001")  # claims another agent's spawn route
    await asyncio.to_thread(tools.spawn, "analyst")
    # still AGENT's token: the controller's scope check refuses another agent's subtree
    assert ctrl.agent_calls[-1] == ("POST", "/agents/other-0001/spawn",
                                    "Bearer flock_at_leader", {})
    with pytest.raises(RuntimeError, match="not allowed for an agent"):
        await asyncio.to_thread(tools._call, "GET", "/workers")


async def test_a_bad_line_is_an_error_and_a_removed_agent_has_no_socket(proxied, monkeypatch):
    _, resource = proxied
    path = resource.bind(AGENT)["mounts"][0]["source"]

    def raw(data):
        with socket.socket(socket.AF_UNIX) as s:
            s.connect(path)
            s.sendall(data)
            return json.loads(s.makefile("rb").readline())

    assert "bad request" in (await asyncio.to_thread(raw, b"[1]\n"))["error"]
    monkeypatch.setattr(resource_module, "READ_S", 0.1)  # a client that never sends its line
    assert "TimeoutError" in (await asyncio.to_thread(raw, b""))["error"]
    await resource.remove([AGENT])
    assert not Path(path).exists()


SPEC = """
name: crew
model: m
task: {repo: https://huggingface.co/datasets/x/y, selector: t, revision: main}
resources_dir: RES
resources: {spawn: {}}
tools_dir: TOOLS
tools: [spawn.*]
agents:
  roles:
    leader: {count: 1}
    helper: {count: 0, workers: {initial: 1, max_extra: 1}}
spawn: {leader: {types: [helper], max: 2}}
environment: {backend: BACKEND}
"""


def test_only_a_role_with_spawn_rules_sees_spawn(tmp_path):
    text = SPEC.replace("RES", str(PLUGINS / "resources")).replace("TOOLS", str(PLUGINS / "tools"))
    (tmp_path / "spec.yaml").write_text(text.replace("BACKEND", "docker"))
    spec = Spec.load(tmp_path / "spec.yaml")
    bundle = build_tool_bundle(spec.tools_dir, spec.tools, tmp_path / "tools.tar.gz")
    types = compile_spec(spec, TASK, bundle).types
    assert sorted(types["leader"].harbor.tools["tools"]) == [
        "spawn.prompt", "spawn.remove", "spawn.spawn", "spawn.status"]
    assert sorted(types["helper"].harbor.tools["tools"]) == ["spawn.prompt", "spawn.status"]
    (tmp_path / "spec.yaml").write_text(text.replace("BACKEND", "hf-sandbox"))
    with pytest.raises(ValueError, match="resources.spawn needs a mount"):
        Spec.load(tmp_path / "spec.yaml")


LOCAL = """
name: crew
model: huggingface/moonshotai/Kimi-K2.5
task:
  repo: https://huggingface.co/datasets/harborframework/terminal-bench-2.1
  selector: regex-log
  revision: main
resources_dir: resources
resources: {spawn: {}}
tools_dir: tools
tools: [spawn.*]
agents:
  roles:
    leader: {count: 1}
    helper: {count: 0, workers: {initial: 1, max_extra: 0}}
spawn: {leader: {types: [helper], max: 1}}
environment: {backend: docker}
limits: {run_timeout_seconds: 900}
"""


@pytest.mark.usefixtures("pins")
def test_a_leader_spawns_a_child_that_runs_a_turn(tmp_path, monkeypatch):
    """A local run (controller, worker processes, HarborAdapter, FlockPi on fakes): the
    leader's first turn calls spawn.spawn; the controller creates the child, places it on a
    helper worker, and the child answers its first prompt."""
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()  # a usable HOME: TMPDIR stays short enough for AF_UNIX
    monkeypatch.setenv("FLUSH_INTERVAL", "0.2")
    exp = tmp_path / "exp"
    for folder in ("resources", "tools"):
        (exp / folder).mkdir(parents=True)
        shutil.copy(PLUGINS / folder / "spawn.py", exp / folder / "spawn.py")
    (exp / "spec.yaml").write_text(LOCAL)
    spec = read_spec(str(exp / "spec.yaml"))
    folders, _ = check_spec(spec, str(exp / "spec.yaml"))
    out = StringIO()
    result = asyncio.run(localrun.run_local(localrun.RunOptions(
        spec=spec, data_dir=tmp_path / ".flock", until=60, until_idle=True, prompt="go",
        refresh=0.2, reconcile_interval=0.2, out=out, tools=folders.data,
        worker_command=[sys.executable, str(Path(__file__).with_name("fake_spawn_worker.py"))])))
    assert result.reason == "idle" and result.exit_code == 0, out.getvalue()
    child = "leader-0000-helper-0001"

    def events(agent):
        return [json.loads(x) for x in
                (result.run_dir / "events" / f"{agent}.jsonl").read_text().splitlines()]

    spawned = [e for e in events("leader-0000") if e["type"] == "spawned"]
    assert [e["data"]["child"] for e in spawned] == [child], out.getvalue()
    kid = events(child)
    assert "reply-" in json.dumps([e for e in kid if e["type"] == "message"]), kid
