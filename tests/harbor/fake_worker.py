"""Harbor faked for a `flock-worker` process: its Trials are test_adapter.py's (the real FlockPi
on a fake environment), container networks are faked at their command, the task download writes
an empty folder, and the controller's model check asks tests/fake_models.py. `fake_harbor()`
(also the task pin) is what tests/harbor/site/sitecustomize.py installs in every tiny run
process."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `tests` importable

from harbor.models.task.id import GitTaskId

from flock.harbor import sources
from tests.harbor.test_adapter import install


async def download(task, directory):
    directory.mkdir(parents=True, exist_ok=True)
    task.local_path = directory
    return directory


async def resolve(settings):
    """The task pinned without the Hub: commit a*40 of `selector`."""
    task = GitTaskId(git_url="https://example.com/tasks.git", git_commit_id="a" * 40,
                     path=settings.selector)
    return sources.ResolvedTask(task, "d" * 40, "tasks@1")


def fake_harbor():
    from flock.controller import inference
    from tests import fake_models

    install(setattr)
    sources.download_task, sources.resolve_task = download, resolve
    inference.TRANSPORT = fake_models.transport()  # the controller's GET /models
