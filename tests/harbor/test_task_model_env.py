"""Model bindings use the actual Trial connection and stay outside the agent environment."""
import json
from pathlib import Path

import pytest
from harbor.models.task.id import GitTaskId

from flock.harbor.config import Config
from flock.harbor.launch import build_trial_config
from flock.harbor.sources import ResolvedTask


@pytest.mark.parametrize("api", ["openai_chat_completions", "openai_responses", "anthropic_messages"])
def test_task_services_receive_the_selected_connection_without_agent_context_or_saved_keys(tmp_path, api):
    config = Config.model_validate({
        "model": "model-id", "inference": {"urls": ["http://proxy:8003/v1"],
            "api_key_env": "MODEL_KEY", "api_format": api, "reasoning_effort": "low"},
        "task": {"repo": "https://huggingface.co/datasets/org/tasks", "selector": "test",
                 "model_env": {"model": "TASK_MODEL", "api_key": "TASK_MODEL_API_KEY", "profile": "TASK_PROFILE"}},
        "limits": {"agent_max_turns": 256},
    })
    task = ResolvedTask(GitTaskId(git_url=config.task.repo, git_commit_id="a" * 40, path=Path("tasks/test")),
                        "a" * 40, "test")
    trial = build_trial_config(config, task, tmp_path, "agent-001", "", "private-model-key",
                               context_env={"PROVIDER_OPTION": "value", "FLOCK_RESULTS_DIR": "/run/path",
                                            "PARTNER_API_KEY": "private-partner-key"})
    assert trial.agent.model_api_format.value == api
    assert "TASK_PROFILE" not in trial.agent.env
    assert trial.environment.env == {}
    assert trial.environment.kwargs["compose_secret_env"] == {"TASK_MODEL_API_KEY": "MODEL_KEY"}
    profile = json.loads(trial.environment.kwargs["compose_env"]["TASK_PROFILE"])
    assert profile["model"] == trial.agent.model_name
    assert profile["parameters"] == {"api_format": api, "reasoning_effort": "low", "max_turns": 256}
    assert profile["task_environment"] == {"PROVIDER_OPTION": "value"}
    assert "private-model-key" not in trial.model_dump_json()
    assert "private-partner-key" not in trial.model_dump_json()
