"""Agent map (/agents.png, /replay/agents), activity (/agents/active) and /status additions."""

import io
import time

import numpy as np
from conftest import H, hx
from PIL import Image

from canvas.world import agent_colour


def _img(r):
    return np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"))


def test_agent_map_live_and_replay(mk):
    c, _w, t = mk(mask_every=1)
    c.post("/paint", json={"pixels": [{"x": 1, "y": 1, "hex": hx(t[1, 1])}]}, headers=H("alice"))
    c.post("/paint", json={"pixels": [{"x": 2, "y": 2, "hex": "010203"}]}, headers=H("bob"))
    c.post("/paint", json={"pixels": [{"x": 1, "y": 1, "hex": hx(t[1, 1])}]}, headers=H("bob"))
    m = _img(c.get("/agents.png"))
    assert tuple(m[1, 1]) == agent_colour("bob")  # last painter wins
    assert tuple(m[2, 2]) == agent_colour("bob")
    assert tuple(m[3, 3]) == (255, 255, 255)  # unpainted = blank
    assert agent_colour("alice") != agent_colour("bob")
    r1 = _img(c.get("/replay/agents?t=1"))
    assert tuple(r1[1, 1]) == agent_colour("alice") and tuple(r1[2, 2]) == (255, 255, 255)
    assert np.array_equal(_img(c.get("/replay/agents?t=3")), m)


def test_agent_map_is_cached_per_n_paints(mk):
    c, _w, t = mk(mask_every=5)
    first = c.get("/agents.png").content
    c.post("/paint", json={"pixels": [{"x": 1, "y": 1, "hex": hx(t[1, 1])}]}, headers=H("a"))
    assert c.get("/agents.png").content == first  # within 5 paints: cached
    assert c.get("/agents.png?fresh=1").content != first  # the driver's final upload
    for _ in range(5):
        c.post("/paint", json={"pixels": [{"x": 2, "y": 2, "hex": "000000"}]}, headers=H("a"))
    assert c.get("/agents.png").content != first


def test_distinct_hues_for_1k_agents():
    cols = {agent_colour(f"painter-a-{i:04d}") for i in range(1000)}
    assert len(cols) > 900  # stable and well spread


def test_active_agents_and_status(mk):
    c, w, _t = mk()
    now = time.time()
    w.paint("old", [(1, 1, 0, 0, 0)], now=now - 30)
    w.paint("x", [(2, 2, 0, 0, 0)], now=now - 2)
    w.paint("y", [(3, 3, 0, 0, 0)], now=now - 1)
    w.paint("x", [(4, 4, 0, 0, 0)], now=now)
    r = c.get("/agents/active", params={"window": 10}).json()
    assert r["count"] == 2 and r["agents"] == ["x", "y"]
    assert c.get("/agents/active", params={"window": 60}).json()["count"] == 3
    st = c.get("/status").json()
    assert st["active_agents_10s"] == 2
    assert st["step_gap_p50_s"] == 2.0  # x painted at now-2 and now
