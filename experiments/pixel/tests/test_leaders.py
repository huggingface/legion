"""Experiment 2 support: region queues for leaders + children, bulk grant, renew, per-region
progress, /score?region=, leader-territory agent map, CLI region verbs."""

import io

import numpy as np
from conftest import H, make_target
from PIL import Image

from canvas.config import Settings
from canvas.world import World, agent_colour, root_of


def _w(tmp_path, **kw):
    p = tmp_path / "t.png"
    make_target(p, w=32, h=32, seed=4)
    return World(Settings(target=p, rate_limit=0, region_enforce=True, **kw))


def _paint(w, agent, cl):
    return w.paint(agent, [(q["x"], q["y"], *bytes.fromhex(q["hex"])) for q in cl["pixels"]])


def test_root_of():
    assert root_of("leader-0003-painter-a-0042") == "leader-0003"
    assert root_of("leader-0003") == "leader-0003"
    assert root_of("painter-a-0001") == "painter-a-0001"
    assert root_of("leader-0001-painter-b-0007-helper-0012") == "leader-0001"


def test_children_paint_only_their_band_until_region_done(tmp_path):
    w = _w(tmp_path)
    w.claim_region("leader-0000", 0, 0, 32, 16)
    w.claim_region("leader-0001", 0, 16, 32, 16)
    kids0 = [f"leader-0000-painter-a-{i:04d}" for i in range(3)]
    w.grant_region("leader-0000", kids0)  # bulk grant
    assert w.claim("stranger", 10)["pixels"] == []  # the whole canvas is leased
    seen = set()
    while True:
        progressed = False
        for k in kids0:
            cl = w.claim(k, 10)
            assert all(q["y"] < 16 for q in cl["pixels"])
            if cl["pixels"]:
                progressed = True
                seen |= {(q["x"], q["y"]) for q in cl["pixels"]}
                _paint(w, k, cl)
        if not progressed:
            break
    last = w.claim(kids0[0], 10)
    assert last["pixels"] == [] and last["region_done"] and not last["done"]
    regs = {r["owner"]: r for r in w.list_regions()}
    assert regs["leader-0000"]["done_pct"] == 100.0 and regs["leader-0001"]["done_pct"] == 0.0
    assert len(seen) == regs["leader-0000"]["work"]
    # Releasing band 1 hands its pixels back to everyone.
    w.release_region("leader-0001")
    assert w.claim("stranger", 10)["pixels"]


def test_wrong_and_expired_pixels_go_back_to_the_region_queue(tmp_path):
    w = _w(tmp_path, claim_ttl=10)
    w.claim_region("L", 0, 0, 32, 32, now=1000.0)
    w.grant_region("L", ["k1", "k2"], now=1000.0)
    cl = w.claim("k1", 2, now=1000.0)
    a, b = cl["pixels"]
    w.paint("k1", [(a["x"], a["y"], 1, 2, 3)], now=1000.0)  # wrong -> region requeue
    nxt = w.claim("k2", 1, now=1000.0)["pixels"][0]
    assert (nxt["x"], nxt["y"]) == (a["x"], a["y"])
    # b stays leased by k1 until expiry, then returns to k2 through the region queue
    later = w.claim("k2", 10, now=1011.0)["pixels"]
    assert (b["x"], b["y"]) in {(q["x"], q["y"]) for q in later}


def test_renew_and_region_score(mk):
    c, _w, t = mk()
    r1 = c.post("/claim/region", json={"x": 1, "y": 1, "w": 4, "h": 4, "ttl": 60}, headers=H("L"))
    r2 = c.post("/claim/region", json={"x": 1, "y": 1, "w": 4, "h": 4, "ttl": 600}, headers=H("L"))
    assert r2.status_code == 200 and r2.json()["renewed"] and r2.json()["id"] == r1.json()["id"]
    assert r2.json()["until"] > r1.json()["until"]
    assert (
        c.post(
            "/claim/region", json={"x": 2, "y": 2, "w": 2, "h": 2}, headers=H("other")
        ).status_code
        == 409
    )
    g = c.post("/claim/region/grant", json={"agents": ["k1", "k2"]}, headers=H("L")).json()
    assert g["granted"] == 2 and g["regions"][0]["grants"] == ["k1", "k2"]
    s = c.get("/score", params={"region": "1,1,4,4"}).json()
    assert s["work_total"] == 16 and s["work_done"] == 0
    c.post(
        "/paint",
        json={"pixels": [{"x": 1, "y": 1, "hex": "{:02x}{:02x}{:02x}".format(*tuple(t[1, 1]))}]},
        headers=H("k1"),
    )
    assert c.get("/score", params={"region": "1,1,4,4"}).json()["work_done"] == 1
    assert c.get("/regions").json()["regions"][0]["done"] == 1
    assert c.get("/score", params={"region": "nope"}).status_code == 400


def test_leader_territory_map(mk):
    c, _w2, _t = mk(mask_every=1)
    for who, xy in (
        ("leader-0001-painter-a-0002", (1, 1)),
        ("leader-0001-painter-b-0003", (2, 2)),
        ("leader-0002-painter-a-0004", (3, 3)),
    ):
        c.post(
            "/paint", json={"pixels": [{"x": xy[0], "y": xy[1], "hex": "000000"}]}, headers=H(who)
        )
    m = np.asarray(Image.open(io.BytesIO(c.get("/agents.png?by=parent").content)).convert("RGB"))
    assert tuple(m[1, 1]) == tuple(m[2, 2]) == agent_colour("leader-0001")
    assert tuple(m[3, 3]) == agent_colour("leader-0002")
    r = np.asarray(
        Image.open(io.BytesIO(c.get("/replay/agents?t=3&by=parent").content)).convert("RGB")
    )
    assert np.array_equal(r, m)
    per = np.asarray(Image.open(io.BytesIO(c.get("/agents.png").content)).convert("RGB"))
    assert tuple(per[1, 1]) != tuple(per[2, 2])  # per-agent colours differ


def test_equal_work_bands_on_shipped_target():
    from canvas.config import DEFAULT_TARGET

    w = World(Settings(target=DEFAULT_TARGET))
    bands = w.bands(10)
    assert bands[0]["y"] == 0 and sum(b["h"] for b in bands) == 1024
    assert sum(b["work"] for b in bands) == w.work_total == 622_780
    works = [b["work"] for b in bands]
    assert max(works) - min(works) < 0.02 * w.work_total  # ~equal work (rows are discrete)


def test_pattern_grant_lends_a_band_to_another_leaders_team(tmp_path):
    w = _w(tmp_path)
    w.claim_region("leader-0003", 0, 0, 32, 16)
    w.claim_region("leader-0007", 0, 16, 32, 16)
    helper = "leader-0007-painter-b-0042"
    w.grant_region("leader-0007", ["leader-0007-*"])  # own team in one call
    assert all(q["y"] >= 16 for q in w.claim(helper, 10)["pixels"])
    w.release(helper)
    w.grant_region("leader-0003", ["leader-0007-*"])  # 0003 asks 0007's team for help
    ys = set()
    for _ in range(80):
        cl = w.claim(helper, 10)
        ys |= {q["y"] for q in cl["pixels"]}
        if cl["pixels"]:
            _paint(w, helper, cl)
    assert min(ys) < 16 <= max(ys)  # it painted in both bands
    assert w.claim("leader-0009-painter-a-0001", 10)["pixels"] == []  # others still blocked
