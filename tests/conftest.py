import os
from pathlib import Path

import pytest

from flock import resources

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture
def board_provider(monkeypatch):
    original = resources.load_module
    module = original(Path(__file__).resolve().parents[1] / "resources" / "board.py")
    monkeypatch.setattr(
        resources,
        "load_module",
        lambda path: module if path.name == "board.py" else original(path),
    )
    return module.Resource
