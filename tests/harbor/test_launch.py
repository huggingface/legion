# from shepherd f3970bc:tests/test_launch.py (the build_trial_config tests: lines 16-42, 632-643)
from pathlib import Path

import pytest
from harbor.models.task.id import GitTaskId

from flock.harbor import launch
from flock.harbor.config import Config
from flock.harbor.sources import ResolvedTask

EXAMPLES = Path(__file__).parent / "examples"


@pytest.mark.parametrize("effort", [None, "off", "low"])
def test_reasoning_effort_reaches_native_pi(monkeypatch, tmp_path, effort):
    from harbor.agents.factory import AgentFactory

    spec = Config.load(EXAMPLES / "pi-hf-agentescape-20.yaml")
    spec.inference.reasoning_effort = effort
    spec.limits.agent_timeout_seconds = 3480
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    resolved = ResolvedTask(
        GitTaskId(
            git_url=spec.task.repo,
            git_commit_id=spec.task.revision,
            path=Path("tasks") / spec.task.selector,
        ),
        spec.task.revision,
        "test",
    )
    config = launch.build_trial_config(
        spec, resolved, tmp_path, "agent-001", "instructions"
    )
    assert ("reasoning_effort" in config.agent.kwargs) == (effort is not None)
    assert config.agent.override_timeout_sec == 3480
    agent = AgentFactory.create_agent_from_config(
        config.agent, logs_dir=tmp_path / "logs"
    )
    assert agent.options.reasoning_effort == effort
    assert agent.options.max_turns == 200


# Lines 45-49 of shepherd's `prepared` fixture; the rest fakes run_experiment, not brought in.
@pytest.fixture
def prepared(monkeypatch, tmp_path):
    spec = Config.load(EXAMPLES / "pi-hf.yaml")
    spec.output_dir = tmp_path / "results"
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    return spec


@pytest.mark.parametrize("backend,delete", [("docker", True), ("podman", False)])
def test_podman_retains_images_shared_by_trials(prepared, tmp_path, backend, delete):
    prepared.environment.backend = backend
    task = ResolvedTask(
        GitTaskId(
            git_url=prepared.task.repo, git_commit_id="c" * 40, path=Path("tasks/test")
        ),
        "c" * 40,
        "test",
    )
    trial = launch.build_trial_config(prepared, task, tmp_path, "agent-001", "")
    assert trial.environment.delete is delete
