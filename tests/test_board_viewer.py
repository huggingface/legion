"""Real SQLite exports; fake Hub/OAuth boundaries, never remote jobs or buckets."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import httpx
import pytest
from fastapi.testclient import TestClient

from flock.board_viewer.app import create_app
from flock.board_viewer.auth import COOKIE, Access
from flock.board_viewer.cache import BoardCache
from flock.board_viewer.export import capture, publish
from flock.board_viewer.import_board import import_board

RUN = "a" * 32
PREFIX = f"flock/runs/demo/{RUN}/board"
BUCKET = "team/boards"


class Hub:
    def __init__(self):
        self.files = {}
        self.members = [SimpleNamespace(username="alice")]
        self.fail = False
        self.calls = []

    def batch_bucket_files(self, bucket, *, add):
        self.calls.append([dest for _, dest in add])
        if self.fail:
            raise RuntimeError("fake credential-bearing error")
        for source, dest in add:
            self.files[dest] = source if isinstance(source, bytes) else Path(source).read_bytes()

    def list_bucket_tree(self, *args, **kwargs):
        if self.fail:
            raise RuntimeError("unavailable")
        return [SimpleNamespace(path=p) for p in self.files]

    def download_bucket_files(self, bucket, files, *, raise_on_missing_files):
        for remote, local in files:
            Path(local).write_bytes(self.files[remote])

    def list_organization_members(self, org):
        if self.fail:
            raise RuntimeError("token must never escape")
        return self.members

    def get_bucket_paths_info(self, bucket, paths):
        return [SimpleNamespace(size=len(self.files[p])) for p in paths if p in self.files]


@pytest.fixture
def source(tmp_path):
    db = pytest.importorskip("msgboard.db")
    data = tmp_path / "board"
    data.mkdir()
    conn = db.connect(data / "board.db")
    db.migrate(conn, data / "board.db")
    conn.execute("INSERT INTO principals (id,issuer,external_id,identity_type,role,display_name,"
                 "created_at) VALUES (1,'local','secret-identity','agent','member','Agent One',1)")
    conn.execute("INSERT INTO credentials (id,principal_id,secret_hash,created_at) "
                 "VALUES ('credential-id',1,'secret-hash',1)")
    for cid, kind in [(1, "public"), (2, "private"), (3, "dm")]:
        conn.execute("INSERT INTO channels(id,name,kind,created_by,created_at) VALUES (?,?,?,?,1)",
                     (cid, f"channel-{cid}", kind, 1))
        conn.execute("INSERT INTO channel_members VALUES (?,1,1,1)", (cid,))
    for mid, cid, parent in [(1, 1, None), (2, 1, 1), (3, 2, None), (4, 3, None)]:
        conn.execute("INSERT INTO messages(id,channel_id,seq,sender_id,body,request_id,payload_hash,"
                     "created_at,reply_to,thread_id) VALUES (?,?,?,1,?,?,?,1,?,?)",
                     (mid, cid, mid, f"hello {mid} <script>alert(1)</script>", str(mid), "hash", parent, parent))
    blob = b"<html>attachment content</html>"
    sha = hashlib.sha256(blob).hexdigest()
    (data / "blobs/objects/ab").mkdir(parents=True)
    (data / "blobs/objects/ab/file").write_bytes(blob)
    conn.execute("INSERT INTO attachments(id,channel_id,uploader_id,upload_id,storage_key,filename,"
                 "content_type,size,sha256,state,published,created_at) "
                 "VALUES ('att1',1,1,'up1','ab/file','test.html','text/html',?,?, 'ready',1,1)",
                 (len(blob), sha))
    conn.execute("UPDATE messages SET attachments='[\"att1\"]' WHERE id=1")
    conn.execute("INSERT INTO message_reactions VALUES (1,1,'👍',1)")
    yield data, conn
    conn.close()


@pytest.fixture
def published(tmp_path, source):
    hub = Hub()
    bundle = tmp_path / "bundle"
    capture(source[0], bundle)
    manifest = publish(hub, BUCKET, PREFIX, bundle)
    cache = BoardCache(hub, BUCKET, tmp_path / "cache")
    cache.refresh()
    return hub, cache, manifest, bundle


def test_export_allowlist_consistent_wal_and_private_conversations(source, tmp_path):
    data, conn = source
    conn.execute("BEGIN IMMEDIATE")
    conn.execute("UPDATE messages SET body='uncommitted' WHERE id=1")
    target = tmp_path / "export"
    doc = capture(data, target)
    conn.rollback()
    assert len(doc["messages"]) == 4
    assert "uncommitted" not in json.dumps(doc)
    assert {c["kind"] for c in doc["channels"]} == {"public", "private", "dm"}
    text = (target / "board.json").read_text()
    assert "secret-hash" not in text and "secret-identity" not in text
    assert "credentials" not in doc and not (target / "private.db").exists()
    assert doc["attachments"][0]["filename"] == "test.html"
    assert len(doc["message_reactions"]) == 1


def test_publication_manifest_last_and_failed_upload_preserves_previous(published):
    hub, cache, manifest, bundle = published
    path = PREFIX.removesuffix("board") + "viewer/manifest.json"
    assert hub.calls[-1] == [path]
    old = hub.files[path]
    hub.fail = True
    with pytest.raises(RuntimeError):
        publish(hub, BUCKET, PREFIX, bundle)
    assert hub.files[path] == old
    assert cache.boards[RUN]["manifest"] == manifest


@pytest.mark.parametrize("failure", ["checksum", "missing", "schema", "traversal", "listing"])
def test_cache_retains_previous_generation(published, failure):
    hub, cache, manifest, bundle = published
    new = publish(hub, BUCKET, PREFIX, bundle)
    base = PREFIX.removesuffix("board") + "viewer/"
    if failure == "checksum":
        hub.files[f"{base}{new['generation']}/board.json"] = b"corrupt"
    elif failure == "missing":
        del hub.files[f"{base}{new['generation']}/board.json"]
    elif failure == "listing":
        hub.fail = True
    else:
        if failure == "schema":
            new["schema_version"] = 999
        else:
            new["files"]["../escape"] = "a" * 64
        hub.files[base + "manifest.json"] = json.dumps(new).encode()
    cache.refresh()
    assert cache.error
    assert cache.boards[RUN]["manifest"] == manifest


@pytest.fixture
def viewer(published):
    hub, cache, _, _ = published
    access = Access(hub, "team", "client", "secret", "https://testserver")
    app = create_app(cache, access, refresh=False)
    with TestClient(app, base_url="https://testserver") as client:
        yield client, access, hub, cache


def sign_in(client, access, username="alice"):
    access.sessions["session"] = username, time.time() + 60
    client.cookies.set(COOKIE, "session")


@pytest.mark.parametrize("path", ["/api/boards", f"/api/boards/{RUN}",
    f"/api/boards/{RUN}/messages", f"/api/boards/{RUN}/attachments/att1", "/viewer.js"])
def test_all_data_requires_login(viewer, path):
    client, _, _, _ = viewer
    r = client.get(path)
    assert r.status_code == 401
    assert "hello" not in r.text


def test_member_search_threads_pagination_attachments_and_isolation(viewer):
    client, access, _, _ = viewer
    sign_in(client, access)
    assert len(client.get("/api/boards").json()["boards"]) == 1
    root = f"/api/boards/{RUN}"
    assert client.get(root + "/messages?thread=1").json()["total"] == 2
    assert client.get(root + "/messages?channel=3").json()["messages"][0]["id"] == 4
    assert client.get(root + "/messages?q=HELLO&author=1&limit=2&offset=2").json()["messages"][0]["id"] == 3
    focus = client.get(root + "/messages?focus=4&limit=2").json()
    assert focus["offset"] == 2 and focus["messages"][-1]["id"] == 4
    assert client.get(root + "/messages?limit=201").status_code == 422
    assert client.get(root + "/messages/2").json()["thread_id"] == 1
    assert client.get("/api/boards/unknown/messages").status_code == 404
    attachment = client.get(root + "/attachments/att1")
    assert attachment.headers["content-type"] == "application/octet-stream"
    assert attachment.headers["content-disposition"].startswith("attachment;")
    assert attachment.headers["cache-control"] == "no-store"
    assert client.post(root + "/messages", json={"body": "write"}).status_code == 405
    assert "innerHTML" not in client.get("/viewer.js").text


def test_revocation_and_expired_membership_fail_closed(viewer):
    client, access, hub, _ = viewer
    sign_in(client, access)
    assert client.get("/api/boards").status_code == 200
    hub.members = []
    access.checked_at -= 61
    assert client.get("/api/boards").status_code == 403
    hub.fail = True
    access.checked_at -= 61
    r = client.get("/api/boards")
    assert r.status_code == 503 and "token" not in r.text


def test_oauth_state_cookie_binding_replay_and_logout(viewer, monkeypatch):
    client, access, _, _ = viewer
    login = client.get("/auth/login", follow_redirects=False)
    state = parse_qs(urlparse(login.headers["location"]).query)["state"][0]
    assert client.get("/auth/callback?code=x&state=wrong").status_code == 400
    class OAuth:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def post(self, url, **kw):
            return httpx.Response(200, json={"access_token": "never-stored"}, request=httpx.Request("POST", url))
        def get(self, url, **kw):
            return httpx.Response(200, json={"preferred_username": "alice"}, request=httpx.Request("GET", url))
    monkeypatch.setattr("flock.board_viewer.auth.httpx.Client", OAuth)
    r = client.get(f"/auth/callback?code=x&state={state}", follow_redirects=False)
    assert r.status_code == 303
    assert client.get("/api/boards").status_code == 200
    assert "never-stored" not in repr(access.sessions)
    assert client.get(f"/auth/callback?code=x&state={state}").status_code == 400
    client.post("/auth/logout")
    assert client.get("/api/boards").status_code == 401


def test_verified_archive_import_and_rejections(source, tmp_path):
    data, conn = source
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    hub = Hub()
    for path in data.rglob("*"):
        if path.is_file():
            hub.files[f"{PREFIX}/{path.relative_to(data)}"] = path.read_bytes()
    run = tmp_path / "old"
    (run / "board").mkdir(parents=True)
    meta = {"running": False, "storage": {"kind": "hf-bucket", "verified_at": "now",
                                           "bucket": BUCKET, "prefix": PREFIX}}
    path = run / "board/metadata.json"
    path.write_text(json.dumps(meta))
    assert import_board(run, hub)["completed"]
    meta["running"] = True
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="stopped"):
        import_board(run, hub)
    meta["running"] = False
    path.write_text(json.dumps(meta))
    hub.files[f"{PREFIX}/board.db-wal"] = b"uncheckpointed"
    with pytest.raises(ValueError, match="WAL"):
        import_board(run, hub)


def test_cache_cold_restart(published, tmp_path):
    hub, _, manifest, _ = published
    second = BoardCache(hub, BUCKET, published[1].root)
    second.refresh()
    assert second.boards[RUN]["manifest"] == manifest


def test_embedded_sign_in_uses_an_allowed_popup(viewer):
    # HF's iframe allows escaping popups but forbids navigating the top window.
    from html.parser import HTMLParser

    class Links(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []

        def handle_starttag(self, tag, attrs):
            if tag == "a":
                self.links.append(dict(attrs))

    client, _, _, _ = viewer
    response = client.get("/")
    assert response.status_code == 200
    parser = Links()
    parser.feed(response.text)
    login = next(link for link in parser.links if link.get("href") == "/auth/login")
    assert login["target"] == "_blank"
    assert "noopener" in login["rel"].split()
    # The new-tab entrypoint remains available before an authenticated session exists.
    response = client.get(login["href"], follow_redirects=False)
    assert response.status_code == 307
    assert response.headers["location"].startswith("https://huggingface.co/oauth/authorize?")
