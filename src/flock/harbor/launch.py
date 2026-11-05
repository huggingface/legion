# from shepherd f3970bc:src/flock/launch.py (lines 32-71, build_trial_config only)
"""Build one agent's Harbor TrialConfig exactly as shepherd's launcher does."""

import os
from urllib.parse import urlsplit

from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)

from .config import PI_VERSION
from .tools.setup import TOOLS_MANIFEST


def build_trial_config(config, resolved, run_dir, name, agent_instructions):
    inference = config.inference
    agent_kwargs = {"version": PI_VERSION, "max_turns": config.limits.agent_max_turns}
    if inference.reasoning_effort is not None:
        agent_kwargs["reasoning_effort"] = inference.reasoning_effort
    return TrialConfig(
        task=(
            TaskConfig(path=resolved.local_path)
            if resolved.local_path is not None
            else TaskConfig(
                **resolved.task_id.model_dump(),
                download_dir=run_dir / "state" / "sources",
            )
        ),
        trial_name=f"{run_dir.name}-{name}",
        trials_dir=run_dir / "trials",
        agent=AgentConfig(
            name="pi",
            override_timeout_sec=config.limits.agent_timeout_seconds,
            model_name=config.model,
            base_url=inference.base_url,
            model_api_key_name=inference.api_key_env,
            model_api_format=inference.api_format,
            env={
                inference.api_key_env: os.environ[inference.api_key_env],
                "FLOCK_TOOLS_MANIFEST": TOOLS_MANIFEST,
            },
            extra_allowed_hosts=[
                urlsplit(inference.base_url).hostname,
            ],
            kwargs=agent_kwargs,
        ),
        # Podman Compose force-removes shared images with `down --rmi local`.
        # delete=False still removes trial containers; job cleanup owns images.
        environment=EnvironmentConfig(
            type=config.environment.backend,
            delete=config.environment.backend != "podman",
        ),
        extra_instructions=[agent_instructions],
    )
