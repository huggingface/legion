"""Two real Pi processes, fake model and local git/Hub: no external calls or jobs."""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

from flock.tools.toolset import Toolset
from flock.worker.adapters.pi import PiAdapter, PiModelConfig
from flock.worker.agents import AgentRuntime
from flock.worker.comms import CommsHub
from flock.worker.events import Outbox

from . import fake_openai
from .test_chatbot4 import git, setup
from .test_worker_pi import PI_BIN, pi_bin, pi_spec
from .test_world import Served, hub_for
from .worker_fakes import until

__all__ = ["setup"]


@pi_bin
async def test_two_pi_agents_submit_review_and_merge_offline(setup, tmp_path, monkeypatch):
    world, _, remote = setup
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ["PATH"])
    monkeypatch.setenv("FLOCK_TEST_PI_KEY", "offline-dummy")
    server = fake_openai.serve(0, str(tmp_path / "model.jsonl"))
    served = Served(world)
    agents = {}
    hub = hub_for(served, runtime=agents.get)
    gateway = CommsHub(swarm="sw", runtime=agents.get)
    await gateway.start_gateway()
    gateway.routes["world"] = hub.call
    base = Toolset(version=None, dir=None, swarm="sw", type="agent", data_dir=tmp_path / "tools")
    try:
        for name in ("cedar", "birch"):
            wd = tmp_path / "agents" / name
            wd.mkdir(parents=True)
            adapter = PiAdapter(wd, agent=name, home=tmp_path / "pi" / name, pi_bin=PI_BIN,
                model_config=PiModelConfig(base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    api_key_env="FLOCK_TEST_PI_KEY", model_id="fake-model", provider="flock"))
            rt = AgentRuntime(name, "agent", pi_spec(secrets=["FLOCK_TEST_PI_KEY"], idle_grace="1h"),
                swarm="sw", work_root=tmp_path / "agents", adapter=adapter, outbox=Outbox(),
                toolbox=lambda name=name: hub.toolset_for(name, base, gateway), world=hub)
            agents[name] = rt
            await rt.setup(restore=False)
            git(tmp_path, "clone", str(remote), str(wd / "chatbot-swarm"))
            (wd / "chatbot-swarm" / "bot/modules" / f"{name}.py").write_text(
                f'def respond(text, memory):\n    return "Hello from {name}" if text == "{name}" else None\n')
        async def call(name, tool, args):
            rt = agents[name]
            count = rt.state.turns
            command = f"flock-tool {tool} --json {shlex.quote(json.dumps(args))}"
            await rt.prompt("bash: " + command)
            await until(lambda: rt.state.turns > count, timeout=30)
        for name in agents:
            await call(name, "open_pr", {"path": "chatbot-swarm", "branch": "hello",
                "title": f"Hello from {name}", "description": "Offline smoke: tested the interface"})
        assert len(world.list_prs()) == 2
        first, second = world.list_prs()
        await call("cedar", "review_pr", {"pr": first["pr"], "head_sha": first["head_sha"],
                                          "verdict": "approve", "comment": "self-review must fail"})
        assert world.list_prs()[0]["reviews"] == {}
        await call("birch", "review_pr", {"pr": first["pr"], "head_sha": first["head_sha"],
                                          "verdict": "approve", "comment": "checked the module"})
        await call("cedar", "merge_pr", {"pr": first["pr"]})
        assert world.list_prs("merged")[0]["pr"] == first["pr"]
        await call("cedar", "review_pr", {"pr": second["pr"], "head_sha": second["head_sha"],
                                          "verdict": "approve", "comment": "checked the module"})
        await call("birch", "merge_pr", {"pr": second["pr"]})
        assert world.list_prs()[0]["pr"] == second["pr"]  # stale base refused
    finally:
        for rt in agents.values():
            await rt.stop()
        await gateway.aclose()
        await hub.aclose()
        served.close()
        server.shutdown()
