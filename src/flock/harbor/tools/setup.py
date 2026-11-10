# vendored from eabb46f:src/flock/tools/setup.py
"""Install the same selected tool bundle for any native installed Harbor agent."""

import asyncio

from harbor.trial.hooks import TrialEvent

TOOLS_ROOT = "/opt/flock-tools"
TOOLS_MANIFEST = f"{TOOLS_ROOT}/manifest.json"


def attach_tool_setup(trial, bundle):
    prepared = False

    async def prepare_tools(event):
        nonlocal prepared
        if prepared:
            return
        async with asyncio.timeout(180):
            environment = trial.agent_environment
            agent = trial.agent
            await agent.ensure_system_dependencies(environment, ("python3",))
            await environment.upload_file(bundle.archive, "/opt/flock-tools.tar.gz")
            await agent.exec_as_root(
                environment,
                f"mkdir -p {TOOLS_ROOT} && python3 -m tarfile -e /opt/flock-tools.tar.gz {TOOLS_ROOT} "
                f"&& chmod -R a+rX {TOOLS_ROOT} "
                f"&& ln -sf {TOOLS_ROOT}/bin/flock-tool /usr/local/bin/flock-tool",
                timeout_sec=60,
            )
            await agent.exec_as_agent(
                environment, "flock-tool --list", env=agent.extra_env, timeout_sec=30
            )
        prepared = True

    trial.add_hook(TrialEvent.AGENT_START, prepare_tools)
