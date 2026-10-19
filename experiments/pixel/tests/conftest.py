from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from canvas.app import create_app
from canvas.config import Settings

BG = (0, 0, 0)


def make_target(path, w=8, h=8, seed=0):
    """RGBA target: a few colours, a transparent border (-> background black)."""
    rng = np.random.default_rng(seed)
    pal = np.array([[255, 210, 30, 255], [255, 157, 0, 255], [255, 255, 255, 255]], np.uint8)
    a = pal[rng.integers(0, len(pal), size=(h, w))]
    a[0, :, 3] = 0  # transparent top row
    a[:, 0, 3] = 0  # transparent left column
    Image.fromarray(a, "RGBA").save(path)
    rgb = a[..., :3].copy()
    rgb[a[..., 3] == 0] = BG
    return rgb


@pytest.fixture
def target(tmp_path):
    p = tmp_path / "target.png"
    rgb = make_target(p)
    return p, rgb


@pytest.fixture
def mk(tmp_path, target):
    """Factory: mk(**settings) -> (TestClient, World, target_rgb)."""
    clients = []

    def _mk(**kw):
        kw.setdefault("rate_limit", 0)
        kw.setdefault("target", target[0])
        cfg = Settings(**kw)
        app = create_app(cfg)
        c = TestClient(app)
        c.__enter__()
        clients.append(c)
        return c, app.state.world, target[1]

    yield _mk
    for c in clients:
        c.__exit__(None, None, None)


def H(name="a1", token=None, **extra):
    h = {"X-Flock-Agent": name}
    if token:
        h["Authorization"] = f"Bearer {token}"
    h.update(extra)
    return h


def hx(rgb):
    return "{:02x}{:02x}{:02x}".format(*tuple(int(c) for c in rgb))
