from pathlib import Path

import pytest
from harbor.models.registry import DatasetMetadata
from harbor.models.task.id import GitTaskId
from pydantic import ValidationError

from shepherd.config import Config, TaskSettings
from shepherd.sources import resolve_task


@pytest.mark.parametrize("dataset", ["unversioned", "name@@version"])
def test_explicit_dataset_still_requires_a_version(dataset):
    data = Config.load(Path("examples/pi-hf-agentescape.yaml")).task.model_dump()
    data["dataset"] = dataset
    with pytest.raises(ValidationError, match="name@version"):
        TaskSettings.model_validate(data)


@pytest.mark.parametrize(
    "update",
    [
        {"agents": {"count": 0}},
        {"limits": {"run_timeout_seconds": float("inf")}},
        {"limits": {"agent_timeout_seconds": 0}},
        {"inference": {"base_url": "https://secret@router.huggingface.co/v1"}},
        {"board": {"mode": "shared"}},
    ],
)
def test_unsupported_or_unsafe_configuration_is_rejected(update):
    data = Config.load(Path("examples/pi-hf.yaml")).model_dump()
    data.update(update)
    with pytest.raises(ValidationError):
        Config.model_validate(data)


@pytest.mark.parametrize("count", [0, 2])
@pytest.mark.parametrize("dataset", ["terminal-bench-2.1@2.1.0", None])
async def test_selector_must_match_once(monkeypatch, count, dataset):
    from shepherd import sources

    task = Config.load(Path("examples/pi-hf.yaml")).task
    task.dataset = dataset
    task.selector = "*"

    async def metadata(self, name):
        assert name == (dataset or "")
        return DatasetMetadata(
            name=name,
            task_ids=[
                GitTaskId(
                    git_url=task.repo,
                    git_commit_id="a" * 40,
                    path=Path(f"tasks/task{i}"),
                )
                for i in range(count)
            ],
        )

    monkeypatch.setattr(sources.GitRepoRegistryClient, "get_dataset_metadata", metadata)
    with pytest.raises(ValueError, match="exactly one"):
        await resolve_task(task)


async def test_preserves_registry_and_separate_task_commit(monkeypatch):
    from shepherd import sources

    task = Config.load(Path("examples/pi-hf.yaml")).task

    async def metadata(self, name):
        assert name == task.dataset
        return DatasetMetadata(
            name=name,
            task_ids=[
                GitTaskId(
                    git_url=task.repo,
                    git_commit_id="b" * 40,
                    path=Path("tasks/regex-log"),
                )
            ],
        )

    monkeypatch.setattr(sources.GitRepoRegistryClient, "get_dataset_metadata", metadata)
    result = await resolve_task(task)
    assert result.registry_commit == task.revision
    assert result.task_id.git_commit_id == "b" * 40
    assert result.manifest()["dataset"] == task.dataset


async def test_repository_discovery_preserves_identity_and_pinned_task(monkeypatch):
    from shepherd import sources

    task = Config.load(Path("examples/pi-hf-agentescape.yaml")).task
    assert task.dataset is None
    identity = f"huggingface.co/HuggingFaceSwarms/AgentEscapeBench-harbor/tree/{task.revision}/tasks"

    async def metadata(self, name):
        assert name == ""
        return DatasetMetadata(
            name=identity,
            version=task.revision[:12],
            task_ids=[
                GitTaskId(
                    git_url=task.repo,
                    git_commit_id=task.revision,
                    path=Path("tasks") / name,
                )
                for name in ("another-task", task.selector)
            ],
        )

    monkeypatch.setattr(sources.GitRepoRegistryClient, "get_dataset_metadata", metadata)
    result = await resolve_task(task)
    assert result.registry_commit == task.revision
    assert result.task_id.git_commit_id == task.revision
    assert result.task_id.path == Path("tasks") / task.selector
    assert result.manifest()["dataset"] == f"{identity}@{task.revision[:12]}"
