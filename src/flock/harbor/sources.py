# from shepherd f3970bc:src/flock/sources.py
"""Resolve one Hub task through Harbor's registry or repository discovery."""

import re
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path

from harbor.models.task.id import GitTaskId
from harbor.registry.client.git_repo import GitRepoRegistryClient, resolve_repo_source
from harbor.tasks.client import TaskClient

from .config import TaskSettings


@dataclass
class ResolvedTask:
    task_id: GitTaskId
    registry_commit: str
    dataset: str
    local_path: Path | None = None

    def manifest(self):
        return {
            "registry_commit": self.registry_commit,
            "dataset": self.dataset,
            "task": self.task_id.model_dump(mode="json"),
        }


async def resolve_task(settings: TaskSettings) -> ResolvedTask:
    repo = resolve_repo_source(settings.repo)
    repo.ref = settings.revision
    repo.resolved_sha = (
        settings.revision
        if re.fullmatch(r"[0-9a-fA-F]{40}", settings.revision)
        else None
    )
    client = GitRepoRegistryClient(repo)
    # Pin the repository before discovery; registry entries may pin another revision.
    commit = await client._get_resolved_sha()
    metadata = await client.get_dataset_metadata(settings.dataset or "")
    matches = [
        task
        for task in metadata.task_ids
        if fnmatchcase(task.get_name(), settings.selector)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Task selector must match exactly one task; matched {len(matches)}"
        )
    task = matches[0]
    if not isinstance(task, GitTaskId):
        raise ValueError("The selected Hub entry must reference a Git task")
    # Resolve any entry-level branch/tag independently of the registry revision.
    task_repo = resolve_repo_source(task.git_url)
    task_repo.ref = task.git_commit_id or "HEAD"
    task_repo.resolved_sha = (
        task.git_commit_id
        if re.fullmatch(r"[0-9a-fA-F]{40}", task.git_commit_id or "")
        else None
    )
    task_commit = await GitRepoRegistryClient(task_repo)._get_resolved_sha()
    return ResolvedTask(
        task.model_copy(update={"git_commit_id": task_commit}),
        commit,
        settings.dataset or f"{metadata.name}@{metadata.version}",
    )


async def download_task(task: ResolvedTask, directory: Path) -> Path:
    downloaded = await TaskClient().download_tasks([task.task_id], output_dir=directory)
    task.local_path = downloaded.results[0].path
    return task.local_path
