"""fake_worker.py whose leader-0000 calls the spawn.spawn tool in its first turn, through
the host side of its Trial's spawn socket (test_spawn.py's local run)."""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `tests` importable

import flock.harbor
from flock.harbor import sources
from flock.harbor.tools.runner import load_module
from flock.worker.main import main
from tests.harbor import test_adapter

tool = load_module(Path(flock.harbor.__file__).with_name("builtin") / "tools" / "spawn.py")


async def download(task, directory):  # as fake_worker.py's (importing it would run it)
    directory.mkdir(parents=True, exist_ok=True)
    task.local_path = directory
    return directory


init, run_pi = test_adapter.Trial.__init__, test_adapter.FakeEnv.exec


def trial(self, config, fakes):
    init(self, config, fakes)
    env = self.agent_environment  # the Trial's env and mounts, as Harbor wires them in
    env.agent_env = config.agent.env
    env.mounts = {m["target"]: m["source"] for m in config.environment.mounts or []}


async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
    agent = self.agent_env.get("SPAWN_AGENT", "")
    if " pi --print " in command and agent == "leader-0000" and not self.fakes.pi:
        os.environ.update(SPAWN_SOCKET=self.mounts["/run/flock-spawn.sock"], SPAWN_AGENT=agent)
        await asyncio.to_thread(tool.spawn, "helper", prompt="help", key="h1")
    return await run_pi(self, command, cwd, env, timeout_sec, user)


test_adapter.install(setattr)
test_adapter.Trial.__init__, test_adapter.FakeEnv.exec = trial, exec
sources.download_task = download
main()
