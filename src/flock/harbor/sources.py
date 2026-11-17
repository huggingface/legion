"""Resolve Hub tasks through Harbor's registry or repository discovery: one for every agent,
or one per agent (pin_tasks)."""

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


class ManyTasks(ValueError):
    """The selector matched more than one task (pin_tasks may give them one to each agent)."""

    def __init__(self, matched: int):
        super().__init__(f"Task selector must match exactly one task; matched {matched}")
        self.matched = matched


async def _matches(settings: TaskSettings):
    """(the pinned registry commit, the dataset, the matching tasks sorted by name)."""
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
    matches = sorted(
        (task for task in metadata.task_ids if fnmatchcase(task.get_name(), settings.selector)),
        key=lambda task: task.get_name(),
    )
    return commit, settings.dataset or f"{metadata.name}@{metadata.version}", matches


async def _pin(task, commit: str, dataset: str) -> ResolvedTask:
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
    return ResolvedTask(task.model_copy(update={"git_commit_id": task_commit}), commit, dataset)


async def resolve_task(settings: TaskSettings) -> ResolvedTask:
    commit, dataset, matches = await _matches(settings)
    if len(matches) > 1:
        raise ManyTasks(len(matches))
    if len(matches) != 1:
        raise ValueError(
            f"Task selector must match exactly one task; matched {len(matches)}"
        )
    return await _pin(matches[0], commit, dataset)


async def resolve_tasks(settings: TaskSettings) -> list[ResolvedTask]:
    """Every task the selector matches, sorted by name, each pinned."""
    commit, dataset, matches = await _matches(settings)
    return [await _pin(task, commit, dataset) for task in matches]


async def pin_tasks(settings: TaskSettings, names: list[str]) -> dict:
    """The task manifest a spec runs: its one task's, or, when the selector matches one task per
    agent (`names`: `naming: consecutive-id` and one type, in order), the first's with `tasks`,
    each agent's own: agent k runs the k-th match by name, and no other agent gets it."""
    try:
        return (await resolve_task(settings)).manifest()
    except ManyTasks as e:
        if e.matched != len(names):
            raise ValueError(f"{e}: one task per agent needs naming: consecutive-id, one type "
                             f"and agents.count: {e.matched}") from None
    tasks = await resolve_tasks(settings)
    manifest = tasks[0].manifest()
    manifest["tasks"] = {name: task.manifest()["task"]
                         for name, task in zip(names, tasks, strict=True)}
    return manifest


async def download_task(task: ResolvedTask, directory: Path) -> Path:
    downloaded = await TaskClient().download_tasks([task.task_id], output_dir=directory)
    task.local_path = downloaded.results[0].path
    return task.local_path


async def download_tasks(tasks: list[ResolvedTask], directory: Path) -> list[Path]:
    """Several tasks at once: Harbor clones each repository once for all of its tasks."""
    downloaded = await TaskClient().download_tasks([t.task_id for t in tasks],
                                                    output_dir=directory)
    for task, result in zip(tasks, downloaded.results, strict=True):
        task.local_path = result.path
    return [task.local_path for task in tasks]
