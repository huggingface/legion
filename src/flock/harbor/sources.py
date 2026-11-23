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


async def _dataset(settings: TaskSettings):
    """(the pinned registry commit, the dataset, its tasks)."""
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
    return commit, settings.dataset or f"{metadata.name}@{metadata.version}", metadata.task_ids


def _one(tasks, selector: str, hint: str = ""):
    matches = [task for task in tasks if fnmatchcase(task.get_name(), selector)]
    if len(matches) != 1:
        raise ValueError(f"Task selector {selector!r} must match exactly one task; matched "
                         f"{len(matches)}{hint if len(matches) > 1 else ''}")
    return matches[0]


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
    """A string selector's one task, for every agent."""
    if not isinstance(settings.selector, str):
        raise ValueError("a list of selectors (one task per agent) needs `flock launch`")
    commit, dataset, tasks = await _dataset(settings)
    hint = " (one task per agent: list one selector per agent)"
    return await _pin(_one(tasks, settings.selector, hint), commit, dataset)


async def pin_tasks(settings: TaskSettings, names: list[str]) -> dict:
    """The task manifest a spec runs: a string selector's one task, or, for a list, the first's
    with `tasks`, each agent's own by position: names[k] (`naming: consecutive-id` and one type,
    else empty) runs the task the k-th entry matches."""
    if isinstance(settings.selector, str):
        return (await resolve_task(settings)).manifest()
    if len(settings.selector) != len(names):
        raise ValueError(f"a list of {len(settings.selector)} selectors gives one task per agent: "
                         f"it needs naming: consecutive-id, no roles and agents.count: "
                         f"{len(settings.selector)}")
    commit, dataset, tasks = await _dataset(settings)
    picked = [_one(tasks, selector) for selector in settings.selector]
    if len({task.get_name() for task in picked}) < len(picked):
        raise ValueError("two selectors match the same task: each agent needs its own")
    pinned = [await _pin(task, commit, dataset) for task in picked]
    manifest = pinned[0].manifest()
    manifest["tasks"] = {name: task.manifest()["task"]
                         for name, task in zip(names, pinned, strict=True)}
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
