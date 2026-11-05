"""resources.world and the world.* tools against a real flock.world server on 127.0.0.1."""

import json
import os
import socket
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import httpx
import pytest

pytest.importorskip("harbor")
uvicorn = pytest.importorskip("uvicorn")

import flock.harbor
from flock.harbor.tools.bundle import build_tool_bundle
from flock.harbor.tools.runner import load_module
from flock.world import Agent, ToolError, World, tool
from flock.world.server import create_app
from flock.world.workspace import apply_patch

TOOLS_DIR = Path(flock.harbor.__file__).with_name("plugins") / "tools"
world_tools = load_module(TOOLS_DIR / "world.py")
WorldResource = load_module(TOOLS_DIR.parent / "resources" / "world.py").Resource
SECRET = "world-secret"


class Shared(World):
    name, version = "shared", "1.0"
    capabilities = ["workspace-upload-v1", "workspace-upload-v2"]

    def __init__(self):
        self.uploads, self.need_full = [], False

    def manifest(self):
        m = super().manifest()
        m["tools"][0]["x-flock-workspace"] = {"arg": "path", "default": "clone", "skip": ["main"]}
        return m

    def call_workspace(self, name, args, agent, snapshot):
        self.uploads.append(snapshot)
        if self.need_full and "patch" in snapshot:
            raise ToolError("unknown base; send the full tree", "need_full")
        return self.call(name, args, agent)

    @tool
    def submit(self, agent: Agent, path: str = "clone") -> dict:
        """Submit a local clone."""
        return {"by": agent.name, "path": path}


@pytest.fixture
def served():
    world = Shared()
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    server = uvicorn.Server(uvicorn.Config(create_app(world, SECRET), log_level="warning"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.01)
    world.url = f"http://127.0.0.1:{sock.getsockname()[1]}"
    yield world
    server.should_exit = True
    thread.join(5)


def git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True,
                          text=True).stdout.strip()


@pytest.fixture
def clone(tmp_path):
    """A clone of a one-commit origin, with a committed change and an untracked file."""
    origin = tmp_path / "origin"
    origin.mkdir()
    git(origin, "init", "-q", "-b", "main")
    (origin / "README.md").write_text("seed\n")
    git(origin, "add", ".")
    git(origin, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "seed")
    git(tmp_path, "clone", "-q", str(origin), "clone")
    work = tmp_path / "clone"
    (work / "README.md").write_text("changed\n")
    git(work, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "change")
    (work / "new.py").write_text("print(1)\n")
    return work


def invoke(directory, name, arguments, env):
    result = subprocess.run(
        [sys.executable, "-S", str(directory / "bin/flock-tool"), name, "--json",
         json.dumps(arguments)],
        env={**os.environ, **env, "FLOCK_TOOLS_MANIFEST": str(directory / "manifest.json")},
        capture_output=True, text=True, timeout=30,
    )
    return result.returncode, json.loads(result.stdout)


async def test_unix_transport_proxies_each_agent_and_uploads_its_clone(
    served, clone, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLOCK_WORLD_TOKEN", SECRET)
    bundle = build_tool_bundle(TOOLS_DIR, ["world.*"], tmp_path / "tools.tar.gz")
    assert bundle.manifest["requires"] == ["world"]
    directory = tmp_path / "bundle"
    with tarfile.open(bundle.archive) as archive:
        archive.extractall(directory, filter="data")
    secrets = []
    resource = WorldResource({"url": served.url}, state_dir=tmp_path / "state",
                             agents=["agent-001", "agent-002"], secrets=secrets)
    try:
        await resource.start()
        assert resource.metadata["tools"] == ["submit"]
        binding = resource.bind("agent-001")
        assert binding["env"] == {"WORLD_SOCKET": "/run/flock-world.sock"}
        env = {"WORLD_SOCKET": binding["mounts"][0]["source"]}
        code, listed = invoke(directory, "world.tools", {}, env)
        assert code == 0 and listed["tools"][0]["name"] == "submit"

        code, result = invoke(directory, "world.call",
                              {"tool": "submit", "args": {"path": str(clone)}}, env)
        assert (code, result) == (0, {"by": "agent-001", "path": str(clone)})
        upload = served.uploads[-1]
        assert upload["base_sha"] == git(clone, "rev-parse", "HEAD~1")
        tree = tmp_path / "tree"
        git(tmp_path, "clone", "-q", str(tmp_path / "origin"), "tree")
        assert apply_patch(upload, tree) == ["README.md", "new.py"]
        assert (tree / "README.md").read_text() == "changed\n"

        served.need_full = True  # the world lacks the base: the whole tree follows
        env = {"WORLD_SOCKET": resource.bind("agent-002")["mounts"][0]["source"]}
        code, result = invoke(directory, "world.call",
                              {"tool": "submit", "args": {"path": str(clone)}}, env)
        assert result["by"] == "agent-002"
        assert sorted(served.uploads[-1]["files"]) == ["README.md", "new.py"]
        assert await resource.export(tmp_path) == {"metrics": "metrics.json"}
        assert all(s not in json.dumps(resource.metadata) for s in secrets)
    finally:
        await resource.stop()
    assert not Path(binding["mounts"][0]["source"]).exists()


async def test_http_transport_token_acts_only_as_its_agent(served, tmp_path, monkeypatch):
    monkeypatch.setenv("FLOCK_WORLD_TOKEN", SECRET)
    resource = WorldResource({"url": served.url, "transport": "http"},
                             state_dir=tmp_path, agents=["agent-001"], secrets=[])
    try:
        await resource.start()
        binding = resource.bind("agent-001")
        assert "mounts" not in binding and binding["allowed_hosts"] == ["127.0.0.1"]
        for key, value in binding["env"].items():
            monkeypatch.setenv(key, value)
        assert world_tools.call("submit", {"path": "main"}) == {"by": "agent-001", "path": "main"}
        assert served.uploads == []  # "main" is in the tool's skip list: nothing to upload
        monkeypatch.setenv("WORLD_AGENT", "agent-002")
        with pytest.raises(RuntimeError, match="unauthorized"):
            world_tools.call("submit", {"path": "main"})
    finally:
        await resource.stop()


async def test_socket_proxy_names_the_agent_and_opens_only_its_routes(
    served, tmp_path, monkeypatch
):
    monkeypatch.setenv("FLOCK_WORLD_TOKEN", SECRET)
    resource = WorldResource({"url": served.url}, state_dir=tmp_path, agents=["agent-001"],
                             secrets=[])
    try:
        await resource.start()
        path = resource.bind("agent-001")["mounts"][0]["source"]
        spoofed = {"X-Flock-Agent": "agent-002", "X-Flock-Type": "interviewer",
                   "X-World-Token": "guess"}
        transport = httpx.HTTPTransport(uds=path)
        with httpx.Client(base_url="http://world", transport=transport) as client:
            r = client.post("/tools/submit", json={"args": {"path": "main"}}, headers=spoofed)
            assert r.json()["result"]["by"] == "agent-001"
            hook = {"event": "agent_done", "agent": "agent-002"}
            assert client.post("/hooks", json=hook).status_code == 403
            assert client.get("/metrics").status_code == 403

        def status(request):
            with socket.socket(socket.AF_UNIX) as sock:
                sock.connect(path)
                sock.sendall(request)
                return int(sock.recv(64).split()[1])

        assert status(b"POST /tools/../hooks HTTP/1.1\r\nContent-Length: 2\r\n\r\n{}") == 403
        too_big = b"POST /tools/submit HTTP/1.1\r\nContent-Length: 999999999999\r\n\r\n"
        assert status(too_big) == 413
    finally:
        await resource.stop()
