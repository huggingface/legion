"""Claim order (scattered / rows), the O(1) free list, and the painted-mask endpoints."""

import io

import numpy as np
from conftest import H, hx, make_target
from PIL import Image

from canvas.config import Settings
from canvas.world import World


def _world(tmp_path, **kw):
    p = tmp_path / "t.png"
    make_target(p, w=32, h=32, seed=5)
    return World(Settings(target=p, rate_limit=0, **kw))


def _paint_claim(w, agent, claim, wrong=()):
    px = []
    for q in claim["pixels"]:
        x, y = q["x"], q["y"]
        c = (1, 2, 3) if (x, y) in wrong else w.target[y, x]
        px.append((x, y, *map(int, c)))
    return w.paint(agent, px) if px else None


def test_scattered_is_seeded_and_not_contiguous(tmp_path):
    a = _world(tmp_path, claim_seed=7)
    b = _world(tmp_path, claim_seed=7)
    c = _world(tmp_path, claim_seed=8)
    ca, cb, cc = (w.claim("x", 10)["pixels"] for w in (a, b, c))
    assert ca == cb and ca != cc
    # Not a row-major run: the 10 pixels span many rows.
    assert len({p["y"] for p in ca}) > 3
    rows = _world(tmp_path, claim_order="rows").claim("x", 10)["pixels"]
    flat = [p["y"] * 32 + p["x"] for p in rows]
    assert flat == sorted(flat) and rows[0] == {"x": 1, "y": 1, "hex": rows[0]["hex"]}


def test_every_work_pixel_handed_out_once_then_done(tmp_path):
    w = _world(tmp_path)
    seen = []
    while True:
        cl = w.claim("a", 10)
        if not cl["pixels"]:
            break
        seen += [(p["x"], p["y"]) for p in cl["pixels"]]
        _paint_claim(w, "a", cl)
    assert len(seen) == len(set(seen)) == w.work_total
    assert w.score(agents=False)["coverage_pct"] == 100.0 and cl["done"]


def test_free_list_requeues_wrong_released_and_expired(tmp_path):
    w = _world(tmp_path, claim_ttl=10)
    t0 = 1000.0
    first = w.claim("a", 3, now=t0)
    bad = (first["pixels"][0]["x"], first["pixels"][0]["y"])
    _paint_claim(w, "a", first, wrong={bad})  # one pixel painted wrong -> back in the queue
    nxt = w.claim("b", 1, now=t0)
    assert (nxt["pixels"][0]["x"], nxt["pixels"][0]["y"]) == bad
    # Released leases come back first.
    held = w.claim("c", 4, now=t0)["pixels"]
    w.release("c")
    again = w.claim("d", 4, now=t0)["pixels"]
    assert [(p["x"], p["y"]) for p in again] == [(p["x"], p["y"]) for p in held]
    # Expired leases come back after the TTL; before it they are not handed out.
    before = w.claim("e", 10, now=t0 + 5)["pixels"]
    assert not {(p["x"], p["y"]) for p in before} & {(p["x"], p["y"]) for p in again}
    after = w.claim("f", 4, now=t0 + 11)["pixels"]
    assert {(p["x"], p["y"]) for p in after} & {(p["x"], p["y"]) for p in again}
    # Overwriting a done pixel puts it back in the queue too.
    x, y = again[0]["x"], again[0]["y"]
    w.paint("g", [(x, y, *map(int, w.target[y, x]))])
    w.paint("h", [(x, y, 1, 2, 3)])
    assert w.requeue and w.requeue[-1] == y * 32 + x


def test_claim_cost_is_independent_of_canvas_progress(tmp_path):
    """O(1) amortised: a claim touches the queue head, not a scan of the whole canvas."""
    w = _world(tmp_path)
    for _ in range(50):
        _paint_claim(w, "a", w.claim("a", 10))
    head = w.head
    w.claim("a", 10)
    assert w.head - head <= 10 + 4096  # one chunk at most (here: exactly the claimed pixels)


def test_restart_loses_no_pixel(tmp_path):
    data = tmp_path / "d"
    w = _world(tmp_path, data_dir=data)
    for _ in range(20):
        _paint_claim(w, "a", w.claim("a", 10), wrong=set())
    w.claim("a", 10)  # outstanding lease
    w.close()
    w2 = _world(tmp_path, data_dir=data)
    seen = set()
    while True:
        cl = w2.claim("b", 10, now=w2.lease_until.max() + 1)
        if not cl["pixels"]:
            break
        seen |= {(p["x"], p["y"]) for p in cl["pixels"]}
        _paint_claim(w2, "b", cl)
    assert w2.score(agents=False)["coverage_pct"] == 100.0


def test_mask_endpoints(mk):
    c, _w, t = mk(mask_every=3)
    m0 = np.asarray(Image.open(io.BytesIO(c.get("/mask.png").content)))
    assert m0.shape == (8, 8) and not m0.any()
    c.post("/paint", json={"pixels": [{"x": 2, "y": 2, "hex": hx(t[2, 2])}]}, headers=H())
    # Cached: within mask_every paint calls the old mask is served.
    assert not np.asarray(Image.open(io.BytesIO(c.get("/mask.png").content))).any()
    for _ in range(2):
        c.post("/paint", json={"pixels": [{"x": 3, "y": 4, "hex": "010203"}]}, headers=H())
    m = np.asarray(Image.open(io.BytesIO(c.get("/mask.png").content)))
    assert m[2, 2] and m[4, 3] and m.sum() == 2
    done = np.asarray(Image.open(io.BytesIO(c.get("/mask.png?kind=done").content)))
    assert done[2, 2] and not done[4, 3]
    ov = np.asarray(Image.open(io.BytesIO(c.get("/mask.png?style=overlay").content)))
    assert ov.shape == (8, 8, 4) and tuple(ov[2, 2]) == (255, 0, 255, 255) and ov[0, 0, 3] == 0
    r1 = np.asarray(Image.open(io.BytesIO(c.get("/replay/mask?t=1&style=grey").content)))
    assert r1[2, 2] and r1.sum() == 1
    assert c.get("/mask.png?kind=nope").status_code == 400
