"""ProgramBench run-only setup: installer access and the native verifier time budget."""
import asyncio

from harbor.models.task.config import NetworkMode, NetworkPolicy
from pydantic import BaseModel, ConfigDict

from flock.harbor import launch
from flock.worker import agents as worker_agents

INSTALL_HOSTS = ["raw.githubusercontent.com", "github.com", "objects.githubusercontent.com",
                 "nodejs.org", "registry.npmjs.org"]


class Options(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Resource:
    def __init__(self, options, *, state_dir, agents, secrets):
        Options.model_validate(options)
        self.metadata = {"verifier_wait_seconds": 7500, "installer_hosts": INSTALL_HOSTS}

    async def start(self):
        # ProgramBench declares a7200-second verifier; allow cleanup on top.
        worker_agents.VERIFY_WAIT = 7500.0
        original = launch.attach_pi_setup
        if getattr(original, "programbench_setup", False):
            return

        def attach(trial, inference):
            original(trial, inference)
            install = trial.agent.install

            async def install_with_network(environment):
                baseline = trial._network_plan(None).agent_env_baseline
                if baseline.network_mode == NetworkMode.PUBLIC:
                    return await install(environment)
                if not environment.capabilities.dynamic_network_policy:
                    raise RuntimeError("Installer needs dynamic network policy")
                policy = NetworkPolicy(network_mode=NetworkMode.ALLOWLIST,
                                       allowed_hosts=list(dict.fromkeys(
                                           baseline.allowed_hosts + INSTALL_HOSTS)))
                try:
                    await environment.set_network_policy(policy)
                    return await install(environment)
                finally:
                    restore = asyncio.create_task(environment.set_network_policy(baseline))
                    try:
                        await asyncio.shield(restore)
                    except asyncio.CancelledError:
                        await restore
                        raise
            trial.agent.install = install_with_network

        attach.programbench_setup = True
        launch.attach_pi_setup = attach

    def bind(self, name):
        return {}

    async def wait(self):
        await asyncio.Event().wait()

    async def export(self, directory):
        return {}

    async def stop(self):
        pass
