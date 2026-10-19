from __future__ import annotations

import io
import json
import math

import numpy as np
from conftest import BG, H, hx
from PIL import Image


def px(x, y, rgb):
    return {"x": x, "y": y, "hex": hx(rgb)}


def test_target_alpha_becomes_background(mk):
    c, w, t = mk()
    info = c.get("/target/info").json()
    assert info["width"] == 8 and info["height"] == 8
    assert info["background"] == "000000" and info["blank"] == "000000"
    assert info["max_pixels_per_call"] == 10 and info["max_claim"] == 10
    assert c.get("/target/pixel", params={"x": 0, "y": 3}).json()["hex"] == "000000"
    assert c.get("/target/pixel", params={"x": 0, "y": 3}).json()["ok"] is True  # blank == bg
    assert c.get("/target/pixel", params={"x": 3, "y": 3}).json()["hex"] == hx(t[3, 3])
    reg = c.get("/target/region", params={"x": 1, "y": 1, "w": 3, "h": 2}).json()
    assert reg["rows"][0].split() == [hx(t[1, i]) for i in (1, 2, 3)]
    raw = c.get("/target/region", params={"x": 0, "y": 0, "w": 8, "h": 8, "format": "raw"})
    assert np.array_equal(np.frombuffer(raw.content, np.uint8).reshape(8, 8, 3), t)
    png = c.get("/target/region", params={"x": 0, "y": 0, "w": 8, "h": 8, "format": "png"})
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(png.content))), t)
    assert c.get("/target/region", params={"x": 6, "y": 0, "w": 4, "h": 1}).status_code == 400
    assert w.work_total == 49  # 7x7 non-transparent


def test_pixel_cap_default_and_one(mk):
    c, _, t = mk()
    many = [px(i % 8, 1 + i // 8, t[1 + i // 8, i % 8]) for i in range(11)]
    r = c.post("/paint", json={"pixels": many}, headers=H())
    assert r.status_code == 400 and "11 > 10" in r.json()["error"]
    assert c.post("/paint", json={"pixels": many[:10]}, headers=H()).status_code == 200

    c1, _, t = mk(max_pixels_per_call=1, max_claim=1)
    assert c1.post("/paint", json={"pixels": [px(1, 1, t[1, 1])]}, headers=H()).status_code == 200
    r = c1.post("/paint", json={"pixels": [px(1, 1, t[1, 1]), px(2, 1, t[1, 2])]}, headers=H())
    assert r.status_code == 400 and "split into calls of 1" in r.json()["error"]
    assert len(c1.post("/claim", json={"n": 5}, headers=H()).json()["pixels"]) == 1


def test_bad_pixels_are_rejected_with_short_errors(mk):
    c, _, _ = mk()
    r = c.post("/paint", json={"pixels": [{"x": 99, "y": 0, "hex": "ffffff"}]}, headers=H())
    assert r.status_code == 400 and "out of bounds" in r.json()["error"]
    r = c.post("/paint", json={"pixels": [{"x": 1, "y": 1, "hex": "zz"}]}, headers=H())
    assert r.status_code == 400 and "bad hex" in r.json()["error"]
    r = c.post("/paint", json={"pixels": [{"x": 1, "y": 1, "r": 300, "g": 0, "b": 0}]}, headers=H())
    assert r.status_code == 400
    r = c.post("/paint", json={"pixels": []}, headers=H())
    assert r.status_code == 400


def test_claims_ttl_and_exclusivity(mk):
    c, w, t = mk(claim_ttl=60, max_claim=10)
    a = c.post("/claim", json={"n": 10}, headers=H("a")).json()
    b = c.post("/claim", json={"n": 10}, headers=H("b")).json()
    sa = {(p["x"], p["y"]) for p in a["pixels"]}
    sb = {(p["x"], p["y"]) for p in b["pixels"]}
    assert len(sa) == 10 and len(sb) == 10 and not sa & sb
    # Background (already matching) pixels are never handed out; target colour is revealed.
    for p in a["pixels"]:
        assert p["x"] > 0 and p["y"] > 0 and p["hex"] == hx(t[p["y"], p["x"]])
    assert sum(r["len"] for r in a["runs"]) == 10
    assert w.claims_outstanding() == 20
    # A new claim by the same agent drops its previous lease set.
    a2 = c.post("/claim", json={"n": 3}, headers=H("a")).json()
    assert len(a2["pixels"]) == 3 and w.claims_outstanding() == 13
    # After the TTL, leases expire and pixels become claimable by others.
    later = w.claim("c", 49, now=w.lease_until.max() + 1)
    assert len(later["pixels"]) == 10  # capped by max_claim
    # Release frees an agent's leases.
    assert c.post("/release", headers=H("b")).json()["released"] == 10
    # Painting correct colour removes the pixel from the work queue for good.
    for p in a2["pixels"]:
        c.post("/paint", json={"pixels": [p]}, headers=H("a"))
    rest = w.claim("z", 49, now=w.lease_until.max() + 1)
    assert rest["remaining"] == 49 - 3


def test_claim_until_done(mk):
    c, _w, _ = mk()
    painted = 0
    for _ in range(20):
        cl = c.post("/claim", json={}, headers=H("solo")).json()
        if not cl["pixels"]:
            assert cl["done"]
            break
        r = c.post("/paint", json={"pixels": cl["pixels"]}, headers=H("solo")).json()
        assert all(x["outcome"] == "useful" for x in r["results"])
        painted += len(cl["pixels"])
    assert painted == 49
    s = c.get("/score").json()
    assert s["exact_match_pct"] == 100.0 and s["coverage_pct"] == 100.0 and s["psnr"] is None


def test_accounting_and_score(mk):
    c, _w, t = mk()
    good = t[2, 2]
    bad = (1, 2, 3)

    def paint(x, y, rgb, who="a"):
        return c.post("/paint", json={"pixels": [px(x, y, rgb)]}, headers=H(who)).json()

    assert paint(2, 2, good)["results"][0]["outcome"] == "useful"
    assert paint(2, 2, good)["results"][0]["outcome"] == "duplicate"
    r = paint(2, 2, bad, "b")["results"][0]
    assert r["outcome"] == "overwritten" and r["want"] == hx(good)
    assert paint(2, 2, bad, "b")["results"][0]["outcome"] == "wrong"
    assert paint(3, 3, bad, "b")["results"][0]["outcome"] == "wrong"
    assert paint(0, 0, BG)["results"][0]["outcome"] == "duplicate"  # background already right
    s = c.get("/score").json()
    assert s["paints"] == {"useful": 1, "duplicate": 2, "wrong": 2, "overwritten": 1, "refused": 0}
    assert s["agents"]["a"]["useful"] == 1 and s["agents"]["b"]["overwritten"] == 1
    # Score agrees with a from-scratch numpy computation.
    canvas = np.zeros_like(t)
    canvas[2, 2] = bad
    canvas[3, 3] = bad
    match = (canvas == t).all(-1)
    mse = ((canvas.astype(float) - t) ** 2).mean()
    assert s["matched"] == int(match.sum())
    assert math.isclose(s["mse"], round(mse, 4))
    assert math.isclose(s["psnr"], round(10 * math.log10(255**2 / mse), 3))
    assert s["painted_pct"] == round(100 * 3 / 64, 4)
    live = np.asarray(Image.open(io.BytesIO(c.get("/canvas.png").content)))
    assert np.array_equal(live, canvas)
    reg = c.get("/region", params={"x": 2, "y": 2, "w": 2, "h": 2}).json()
    assert reg["rows"][0].split()[0] == hx(bad)


def test_enforced_claims_refuse_others(mk):
    c, _w, _t = mk(claim_mode="enforced")
    a = c.post("/claim", json={"n": 1}, headers=H("a")).json()["pixels"][0]
    r = c.post("/paint", json={"pixels": [a]}, headers=H("b")).json()
    assert r["results"][0]["outcome"] == "refused"
    assert c.get("/target/pixel", params=a).json()["ok"] is False
    r = c.post("/paint", json={"pixels": [a]}, headers=H("a")).json()
    assert r["results"][0]["outcome"] == "useful"


def test_rate_limit(mk):
    c, _, _t = mk(rate_limit=1, rate_burst=2)
    codes = [c.post("/claim", json={"n": 1}, headers=H("r")).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = c.post("/claim", json={"n": 1}, headers=H("r"))
    assert "retry in" in r.json()["error"] and "Retry-After" in r.headers
    assert c.post("/claim", json={"n": 1}, headers=H("other")).status_code == 200


def _random_paints(t, n_calls, seed=1, cap=10):
    rng = np.random.default_rng(seed)
    h, w = t.shape[:2]
    calls = []
    for i in range(n_calls):
        k = int(rng.integers(1, cap + 1))
        pts = []
        for _ in range(k):
            x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
            rgb = t[y, x] if rng.random() < 0.6 else rng.integers(0, 256, 3)
            pts.append(px(x, y, rgb))
        calls.append((f"ag{i % 5}", pts))
    return calls


def test_log_and_replay_frames_match_sequence(mk):
    c, w, t = mk(keyframe_every=3, keyframe_cache=2)
    calls = _random_paints(t, 40)
    expected = [np.zeros_like(t)]
    for i, (agent, pts) in enumerate(calls):
        r = c.post(
            "/paint",
            json={"pixels": pts, "turn": f"turn-{i}"},
            headers=H(agent, **{"X-Request-Id": f"req-{i}"}),
        )
        assert r.status_code == 200, r.text
        cur = expected[-1].copy()
        for p in pts:
            cur[p["y"], p["x"]] = [int(p["hex"][j : j + 2], 16) for j in (0, 2, 4)]
        expected.append(cur)
    # Frames in scrambled order exercise keyframe cache hits, misses and eviction.
    for tt in [40, 0, 7, 3, 39, 1, 22, 21, 6, 40, 13]:
        assert np.array_equal(w.frame(tt), expected[tt]), tt
    png = c.get("/replay/frame", params={"t": 17}).content
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(png))), expected[17])
    assert np.array_equal(w.frame(999), expected[-1])  # clamped to live
    # The log joins to flock events by agent / turn / req.
    lines = c.get("/log", params={"after": 5, "limit": 3}).text.splitlines()
    es = [json.loads(ln) for ln in lines]
    assert [e["seq"] for e in es] == [6, 7, 8]
    assert es[0]["turn"] == "turn-5" and es[0]["req"] == "req-5" and es[0]["agent"] == "ag0"
    assert len(es[0]["px"]) == len(calls[5][1]) and es[0]["ts"] > 0
    assert c.get("/replay").status_code == 200 and "Canvas replay" in c.get("/replay").text


def test_turn_from_header(mk):
    c, _, t = mk()
    c.post("/paint", json={"pixels": [px(1, 1, t[1, 1])]}, headers=H("a", **{"X-Flock-Turn": "T9"}))
    e = json.loads(c.get("/log").text.splitlines()[0])
    assert e["turn"] == "T9" and len(e["req"]) == 16


def test_persistence_round_trip(mk, tmp_path):
    data = tmp_path / "data"
    c, w, t = mk(data_dir=data, keyframe_every=4)
    calls = _random_paints(t, 25, seed=7)
    for agent, pts in calls:
        assert c.post("/paint", json={"pixels": pts}, headers=H(agent)).status_code == 200
    held = c.post("/claim", json={"n": 4}, headers=H("holder")).json()["pixels"]
    score1 = c.get("/score").json()
    canvas1 = w.canvas.copy()
    frame9 = w.frame(9)
    c.__exit__(None, None, None)  # shutdown -> snapshot

    c2, w2, _ = mk(data_dir=data, keyframe_every=4)
    assert np.array_equal(w2.canvas, canvas1)
    assert np.array_equal(w2.frame(9), frame9)
    s2 = c2.get("/score").json()
    for k in ("seq", "matched", "mse", "paints"):
        assert s2[k] == score1[k], k
    # Paint counters rebuild exactly from the log; claim counters are soft state.
    strip = lambda ag: {n: {**v, "claims": 0} for n, v in ag.items() if v["calls"]}
    assert strip(s2["agents"]) == strip(score1["agents"])
    # Claims survived the restart.
    assert w2.claims_outstanding() == len(held)
    other = w2.claim("someone", 10)
    assert not {(p["x"], p["y"]) for p in held} & {(p["x"], p["y"]) for p in other["pixels"]}
    # Log continues at the next seq; /log reads from disk.
    r = c2.post("/paint", json={"pixels": calls[0][1]}, headers=H("again")).json()
    assert r["seq"] == 26
    assert json.loads(c2.get("/log", params={"after": 25}).text)["agent"] == "again"
    assert len(c2.get("/log", params={"after": 0, "limit": 5000}).text.splitlines()) == 26
    c2.__exit__(None, None, None)

    # A torn last line (crash mid-write) is dropped; the log stays the source of truth.
    log = data / "paint.log.jsonl"
    with open(log, "a") as fh:
        fh.write('{"seq":27,"ts":1,"agent":"x","px":[[1,1,')
    (data / "claims.npz").unlink()
    c3, w3, _ = mk(data_dir=data)
    assert w3.seq == 26 and w3.claims_outstanding() == 0
    assert log.read_text().endswith("\n")
    assert c3.post("/paint", json={"pixels": calls[1][1]}, headers=H("z")).json()["seq"] == 27
