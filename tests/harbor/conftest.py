# from shepherd eabb46f:tests/conftest.py
import os

import pytest

from flock.harbor import resources
from flock.harbor.config import BUILTIN

os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


@pytest.fixture
def message_board_provider(monkeypatch):
    original = resources.load_module
    module = original(BUILTIN / "resources" / "message_board.py")
    monkeypatch.setattr(
        resources,
        "load_module",
        lambda path: module if path.name == "message_board.py" else original(path),
    )
    return module.Resource
