import os
from pathlib import Path

import pytest

from flock import resources

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture
def message_board_provider(monkeypatch):
    original = resources.load_module
    module = original(
        Path(__file__).resolve().parents[1] / "resources" / "message_board.py"
    )
    monkeypatch.setattr(
        resources,
        "load_module",
        lambda path: module if path.name == "message_board.py" else original(path),
    )
    return module.Resource
