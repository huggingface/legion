"""Background / alpha handling and the read-open, write-token policy."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import H
from fastapi.testclient import TestClient
from PIL import Image

from canvas.app import create_app
from canvas.config import DEFAULT_TARGET, Settings
from canvas.world import World


def _rgba(tmp_path):
    a = np.zeros((4, 4, 4), np.uint8)  # all transparent background
    a[1, 1] = (255, 157, 0, 255)  # opaque orange
    a[1, 2] = (0, 0, 0, 255)  # opaque black
    a[2, 1] = (255, 255, 255, 255)  # opaque white: same colour as the blank canvas, still work
    a[2, 2] = (200, 100, 50, 128)  # partial alpha edge pixel
    p = tmp_path / "rgba.png"
    Image.fromarray(a, "RGBA").save(p)
    return p


def test_defaults_white_blank_alpha_zero_background_partial_over_blank(tmp_path):
    w = World(Settings(target=_rgba(tmp_path)))
    assert w.blank == (255, 255, 255) and w.bg == (255, 255, 255)
    assert w.work_total == 4
    assert tuple(w.target[0, 0]) == (255, 255, 255)
    # Partial alpha composited over the white blank: round(c * a + 255 * (1 - a)), a = 128/255.
    assert tuple(w.target[2, 2]) == (227, 177, 152)
    assert w.matched == 12  # the 12 transparent pixels are done; the 4 work pixels are not
    # Opaque white on the white canvas still needs painting, and painting it is useful.
    assert not w.match[2 * 4 + 1]
    assert w.paint("a", [(1, 2, 255, 255, 255)])["results"][0]["outcome"] == "useful"
    assert w.paint("a", [(0, 0, 255, 255, 255)])["results"][0]["outcome"] == "duplicate"
    claimable = w.claim("b", 10)["pixels"]
    assert {(p["x"], p["y"]) for p in claimable} == {(1, 1), (2, 1), (2, 2)}


def test_background_and_blank_overrides(tmp_path):
    w = World(Settings(target=_rgba(tmp_path), blank="000000", background="00ff00"))
    assert tuple(w.target[0, 0]) == (0, 255, 0)  # alpha 0 -> CANVAS_BACKGROUND
    assert tuple(w.target[2, 2]) == (100, 50, 25)  # partial alpha over the black blank
    # Background differs from the blank canvas: not work, but claimable so it gets fixed.
    assert w.work_total == 4 and len(w.order) == 16


def test_background_is_work_switch(tmp_path):
    w = World(Settings(target=_rgba(tmp_path), background_is_work=True))
    assert w.work_total == 16 and w.matched == 0
    assert w.paint("a", [(0, 0, 255, 255, 255)])["results"][0]["outcome"] == "useful"
    assert len(w.claim("b", 50)["pixels"]) == 10  # capped, background included
    s = w.score(agents=False)
    assert s["work_total"] == 16 and s["coverage_pct"] == round(100 / 16, 4)


@pytest.mark.parametrize("bg_work, total", [(False, 622_780), (True, 1024 * 1024)])
def test_shipped_target(bg_work, total):
    w = World(Settings(target=DEFAULT_TARGET, background_is_work=bg_work))
    assert w.alpha is not None and w.target.shape == (1024, 1024, 3)
    assert int((w.alpha == 0).sum()) == 425_796
    assert int(((w.alpha > 0) & (w.alpha < 255)).sum()) == 121_114
    assert w.work_total == total
    assert w.matched == 1024 * 1024 - total


def test_reads_open_writes_need_token(tmp_path):
    app = create_app(Settings(target=_rgba(tmp_path), token="sec", admin_token="adm", rate_limit=0))
    with TestClient(app) as c:
        for path in [
            "/canvas.png",
            "/score",
            "/status",
            "/replay",
            "/replay/frame?t=0",
            "/log",
            "/target/info",
            "/target/region?x=0&y=0&w=2&h=2",
            "/target/pixel?x=1&y=1",
            "/target.png",
            "/region?x=0&y=0&w=2&h=2",
            "/regions",
        ]:
            assert c.get(path).status_code == 200, path
        writes = [
            ("/paint", {"pixels": [{"x": 1, "y": 1, "hex": "ff9d00"}]}),
            ("/claim", {}),
            ("/release", {}),
            ("/claim/region", {"x": 0, "y": 0, "w": 1, "h": 1}),
            ("/claim/region/grant", {"agent": "k"}),
            ("/release/region", {}),
            ("/agents/register", {"name": "p"}),
        ]
        for path, body in writes:
            assert c.post(path, json=body, headers=H("a")).status_code == 401, path
        assert c.post("/claim", json={}, headers=H("a", token="sec")).status_code == 200


def test_cors_on_public_reads_only(tmp_path):
    app = create_app(Settings(target=_rgba(tmp_path), token="sec", rate_limit=0))
    with TestClient(app) as c:
        origin = {"Origin": "https://cmpatino-flock-pixel.static.hf.space"}
        for path in [
            "/canvas.png",
            "/score",
            "/status",
            "/log",
            "/replay/frame?t=0",
            "/target/info",
        ]:
            r = c.get(path, headers=origin)
            assert r.headers.get("access-control-allow-origin") == "*", path
        pre = c.options("/score", headers={**origin, "Access-Control-Request-Method": "GET"})
        assert pre.status_code == 204 and "GET" in pre.headers["access-control-allow-methods"]
        r = c.post("/claim", json={}, headers={**origin, **H("a", token="sec")})
        assert r.status_code == 200 and "access-control-allow-origin" not in r.headers
