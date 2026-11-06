"""On PYTHONPATH in the tiny runs (tests/tinyrun.py): a process whose env sets TINY_FAKE_HARBOR
(every process of a tiny run: the local run, a controller, its workers) runs Harbor faked as
tests/harbor/fake_worker.py does, the task pin too."""

import os

if os.environ.get("TINY_FAKE_HARBOR"):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # `tests` importable
    from tests.harbor.fake_worker import fake_harbor

    fake_harbor()
