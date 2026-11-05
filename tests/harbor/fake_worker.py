"""`flock-worker` with Harbor faked, the worker command of test_cli_launch.py's local run: its
Trials are test_adapter.py's (the real FlockPi on a fake environment), container networks are
faked at their command, and the task download writes an empty folder."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # `tests` importable

from flock.harbor import sources
from flock.worker.main import main
from tests.harbor.test_adapter import install


async def download(task, directory):
    directory.mkdir(parents=True, exist_ok=True)
    task.local_path = directory
    return directory


install(setattr)
sources.download_task = download
main()
