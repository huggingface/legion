from conftest import H, hx


def _paint(c, t, x, y, who):
    body = {"pixels": [{"x": x, "y": y, "hex": hx(t[y, x])}]}
    return c.post("/paint", json=body, headers=H(who)).json()["results"][0]["outcome"]


def test_overlap_rejected_and_listing(mk):
    c, _, _ = mk()
    r = c.post("/claim/region", json={"x": 1, "y": 1, "w": 3, "h": 3}, headers=H("lead"))
    assert r.status_code == 200 and r.json()["owner"] == "lead"
    rid = r.json()["id"]
    r = c.post("/claim/region", json={"x": 3, "y": 3, "w": 2, "h": 2}, headers=H("other"))
    assert r.status_code == 409 and "overlaps lease" in r.json()["error"]
    # Touching but not overlapping is fine.
    assert (
        c.post(
            "/claim/region", json={"x": 4, "y": 1, "w": 2, "h": 2}, headers=H("other")
        ).status_code
        == 200
    )
    assert (
        c.post("/claim/region", json={"x": 7, "y": 7, "w": 2, "h": 2}, headers=H("o")).status_code
        == 400
    )  # out of bounds
    regs = c.get("/regions").json()["regions"]
    assert {r["owner"] for r in regs} == {"lead", "other"}
    # Only the holder can release a lease by id.
    assert c.post("/release/region", json={"id": rid}, headers=H("other")).status_code == 409
    assert c.post("/release/region", json={"id": rid}, headers=H("lead")).json()["released"] == 1
    assert len(c.get("/regions").json()["regions"]) == 1


def test_ttl_expiry(mk):
    c, w, _ = mk()
    r = w.claim_region("lead", 1, 1, 4, 4, ttl=10, now=1000.0)
    assert r["until"] == 1010.0
    try:
        w.claim_region("b", 2, 2, 2, 2, now=1005.0)
        raise AssertionError("overlap should be rejected")
    except Exception as e:  # noqa: BLE001
        assert "overlaps" in str(e)
    assert w.claim_region("b", 2, 2, 2, 2, now=1011.0)["owner"] == "b"  # expired
    assert (
        c.post(
            "/claim/region", json={"x": 1, "y": 1, "w": 1, "h": 1, "ttl": 1e9}, headers=H("x")
        ).status_code
        == 400
    )


def test_advisory_by_default(mk):
    c, _, t = mk()
    c.post("/claim/region", json={"x": 1, "y": 1, "w": 7, "h": 7}, headers=H("lead"))
    assert _paint(c, t, 2, 2, "stranger") == "useful"


def test_enforced_regions_and_grants(mk):
    c, w, t = mk(region_enforce=True)
    rid = c.post("/claim/region", json={"x": 1, "y": 1, "w": 3, "h": 7}, headers=H("lead")).json()[
        "id"
    ]
    # A stranger can neither paint nor pixel-claim inside the region.
    assert _paint(c, t, 2, 2, "stranger") == "refused"
    got = c.post("/claim", json={"n": 10}, headers=H("stranger")).json()["pixels"]
    assert got and all(p["x"] >= 4 for p in got)
    c.post("/release", headers=H("stranger"))
    # The holder can; grants hand the region to children.
    assert _paint(c, t, 1, 1, "lead") == "useful"
    assert _paint(c, t, 2, 1, "kid") == "refused"
    assert (
        c.post("/claim/region/grant", json={"agent": "kid"}, headers=H("stranger")).status_code
        == 400
    )
    g = c.post("/claim/region/grant", json={"agent": "kid", "id": rid}, headers=H("lead")).json()
    assert g["regions"][0]["grants"] == ["kid"]
    assert _paint(c, t, 2, 1, "kid") == "useful"
    kid = c.post("/claim", json={"n": 10}, headers=H("kid")).json()["pixels"]
    assert any(p["x"] < 4 for p in kid)
    # After release, strangers may paint there again.
    c.post("/release/region", json={}, headers=H("lead"))
    assert _paint(c, t, 3, 3, "stranger") == "useful"
    # Refusals are logged, so a rebuild from the log matches.
    assert w.score(agents=False)["paints"]["refused"] == 2


def test_regions_survive_restart(mk, tmp_path):
    data = tmp_path / "d"
    c, _, t = mk(data_dir=data, region_enforce=True)
    c.post("/claim/region", json={"x": 1, "y": 1, "w": 2, "h": 2}, headers=H("lead"))
    c.post("/claim/region/grant", json={"agent": "kid"}, headers=H("lead"))
    c.__exit__(None, None, None)
    c2, _, _ = mk(data_dir=data, region_enforce=True)
    regs = c2.get("/regions").json()["regions"]
    assert regs[0]["owner"] == "lead" and regs[0]["grants"] == ["kid"]
    assert _paint(c2, t, 1, 1, "stranger") == "refused"
