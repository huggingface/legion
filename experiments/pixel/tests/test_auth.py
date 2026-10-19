from __future__ import annotations

from conftest import H, hx


def _paint(c, t, headers):
    return c.post(
        "/paint", json={"pixels": [{"x": 1, "y": 1, "hex": hx(t[1, 1])}]}, headers=headers
    )


def test_open_mode_needs_name(mk):
    c, _, t = mk()
    assert _paint(c, t, {}).json()["error"].startswith("missing agent name")
    assert _paint(c, t, H("dev")).status_code == 200


def test_shared_secret_mode(mk):
    c, _, t = mk(token="s3cret")
    assert _paint(c, t, H("a")).status_code == 401
    assert _paint(c, t, H("a", token="nope")).status_code == 401
    r = _paint(c, t, H("a", token="s3cret"))
    assert r.status_code == 200 and r.json()["agent"] == "a"
    # Behind the HF proxy: Authorization carries the HF token, X-Canvas-Token ours.
    r = _paint(
        c, t, {"X-Flock-Agent": "b", "Authorization": "Bearer hf_x", "X-Canvas-Token": "s3cret"}
    )
    assert r.status_code == 200 and r.json()["agent"] == "b"
    assert _paint(c, t, {"Authorization": "Bearer s3cret"}).status_code == 400  # no name
    # Reads stay open unless CANVAS_READ_AUTH=1.
    assert c.get("/score").status_code == 200


def test_per_agent_tokens(mk, tmp_path):
    data = tmp_path / "d"
    c, _, t = mk(admin_token="adm", data_dir=data)
    assert c.post("/agents/register", json={"name": "p1"}).status_code == 401
    assert (
        c.post("/agents/register", json={"name": "bad name!"}, headers=H(token="adm")).status_code
        == 400
    )
    tok = c.post("/agents/register", json={"name": "p1"}, headers=H(token="adm")).json()["token"]
    # Identity is the token: the X-Flock-Agent header cannot spoof it.
    r = _paint(c, t, H("someone-else", token=tok))
    assert r.status_code == 200 and r.json()["agent"] == "p1"
    assert _paint(c, t, H("p1", token="cv_wrong")).status_code == 401
    assert _paint(c, t, H("p1")).status_code == 401  # not open: admin token configured
    # Token file stores hashes only, and survives restarts.
    assert tok not in (data / "agents.json").read_text()
    c.__exit__(None, None, None)
    c2, _, _ = mk(admin_token="adm", data_dir=data)
    assert _paint(c2, t, {"Authorization": f"Bearer {tok}"}).json()["agent"] == "p1"
    # Re-registering rotates.
    tok2 = c2.post("/agents/register", json={"name": "p1"}, headers=H(token="adm")).json()["token"]
    assert _paint(c2, t, {"Authorization": f"Bearer {tok}"}).status_code == 401
    assert _paint(c2, t, {"Authorization": f"Bearer {tok2}"}).status_code == 200


def test_both_modes_and_read_auth(mk):
    c, _, t = mk(token="shared", admin_token="adm", read_auth=True)
    tok = c.post("/agents/register", json={"name": "p"}, headers=H(token="adm")).json()["token"]
    assert _paint(c, t, H("x", token="shared")).json()["agent"] == "x"
    assert _paint(c, t, {"Authorization": f"Bearer {tok}"}).json()["agent"] == "p"
    assert c.get("/score").status_code == 401
    assert c.get("/score", headers={"X-Canvas-Token": tok}).status_code == 200
    assert c.get("/status", headers=H(token="shared")).json()["auth"]["read_auth"] is True


def test_registration_disabled_without_admin(mk):
    c, _, _ = mk(token="s")
    assert c.post("/agents/register", json={"name": "p"}, headers=H(token="s")).status_code == 403
