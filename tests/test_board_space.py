"""`comms.board: space` (flock init duplicates a board Space, fakes only) and `dashboard:` (the
run's flock-overview row written to a registry bucket)."""

from __future__ import annotations

import json
import stat
from datetime import datetime, timedelta

import pytest
import yaml

from flock import rundir
from flock.models import SwarmSpec
from flock.ops import SpecError, read_spec
from flock.specrender import read_env_file
from tests.fake_msgboard import Board, serve

from .conftest import FAKE_PROVIDERS

SPEC = {
    "swarm": "demo", "namespace": "acme",
    "bucket": "hf://buckets/acme/runs",
    "comms": {"board": "space", "template": "acme/msgboard-template", "invite_env": "BOARD_INVITE",
              "channel": "general", "space_secrets": {"HF_TOKEN": "BOARD_BUCKET_TOKEN"}},
    "providers": FAKE_PROVIDERS,
    "types": {"a": {"harness": "llm", "model": "fake:m", "comms": True}},
}


class FakeApi:
    """HfApi's Space calls; a duplicated Space's operator token is its board's admin token, as in
    msgboard's Space app."""

    def __init__(self, board: Board):
        self.board, self.spaces, self.calls = board, set(), []

    def repo_exists(self, repo_id, repo_type=None):
        return repo_id in self.spaces

    def duplicate_repo(self, from_id, to_id, **kw):
        self.calls.append((from_id, to_id, kw))
        self.spaces.add(to_id)
        master = next(s["value"] for s in kw["space_secrets"] if s["key"] == "MSGBOARD_OPERATOR_TOKEN")
        self.board.tokens[master] = self.board.principal("operator", "user")["id"]


@pytest.fixture
def space(monkeypatch):
    board, srv, url = serve(Board(channels={1: "random"}))  # one board at the Space's root
    api = FakeApi(board)
    monkeypatch.setattr(rundir, "hub_api", lambda token: api)
    monkeypatch.setattr(rundir, "space_url", lambda sid: url)
    monkeypatch.setattr(rundir, "check_venv", lambda env: rundir.Check("venv", True))
    yield board, api, url
    srv.shutdown()


def _folder(tmp_path, spec=SPEC):
    f = tmp_path / "exp"
    f.mkdir(parents=True)
    (f / "spec.yaml").write_text(yaml.safe_dump(spec))
    return f


ENV = {"HF_TOKEN": "hf_x", "BOARD_BUCKET_TOKEN": "hf_scoped"}


def test_init_creates_the_board_once_then_reuses_it(tmp_path, space):
    board, api, url = space
    run = tmp_path / "run"
    rep = rundir.init(run, _folder(tmp_path), env=ENV)
    checks = {c.name: c for c in rep.checks}
    assert checks["board space"].ok, checks["board space"].detail
    assert checks["spec"].ok and checks["board"].ok
    (src, to, kw), = api.calls
    assert (src, to) == ("acme/msgboard-template", "acme/board-demo")
    secrets = {s["key"]: s["value"] for s in kw["space_secrets"]}
    assert kw["private"] and set(secrets) == {"MSGBOARD_OPERATOR_TOKEN", "HF_TOKEN"}
    assert secrets["HF_TOKEN"] == "hf_scoped"  # the mapped, bucket-scoped token, not the operator's
    variables = {v["key"]: v["value"] for v in kw["space_variables"]}
    assert kw["space_hardware"] == "cpu-basic"
    # flock's only: none of the template's
    assert variables == {"MSGBOARD_REPLICA": "hf://buckets/acme/runs/board/demo"}
    assert "general" in board.channels.values()
    values = read_env_file(run / "run.env")
    assert values["BOARD_URL"] == url and values["BOARD_INVITE"] in board.invites
    # a single-use admin invite link for the operator, next to the operator token
    admin = read_env_file(run / "board.env")["BOARD_ADMIN_INVITE"]
    assert admin.startswith(url + "/?invite=") and admin.split("=", 1)[1] in board.invites
    assert admin not in (run / "run.env").read_text()
    invites = len(board.invites)
    # the operator token: its own file, mode 600, never in run.env (every job sources run.env)
    # nor in the env files commands and local runs read
    op = read_env_file(run / "board.env")["BOARD_OPERATOR_TOKEN"]
    assert stat.S_IMODE((run / "board.env").stat().st_mode) == 0o600
    assert stat.S_IMODE((run / "run.env").stat().st_mode) == 0o600
    assert op not in (run / "run.env").read_text() and "BOARD_OPERATOR_TOKEN" not in values
    assert all("board.env" not in f for f in rundir.run_context(run)[1])
    assert op not in json.dumps(rep.created + rep.notes + [c.detail for c in rep.checks])
    spec = read_spec(str(run / "experiment" / "spec.yaml"), rundir.run_context(run)[1])
    assert spec.comms.url == url and op not in spec.model_dump_json()
    # again: the doctor; nothing new on the Hub or the board, same invite
    rep = rundir.init(run, env=ENV)
    assert {c.name: c for c in rep.checks}["board space"].ok and len(api.calls) == 1
    assert any("not re-applied" in n for n in rep.notes)
    assert read_env_file(run / "run.env")["BOARD_INVITE"] == values["BOARD_INVITE"]
    assert list(board.channels.values()).count("general") == 1
    assert len(board.invites) == invites  # no second admin invite
    assert read_env_file(run / "board.env")["BOARD_ADMIN_INVITE"] == admin
    # an invite that no longer works (the board lost it, or it was revoked) is replaced
    del board.invites[values["BOARD_INVITE"]]
    rep = rundir.init(run, env=ENV)
    new = read_env_file(run / "run.env")["BOARD_INVITE"]
    assert new != values["BOARD_INVITE"] and new in board.invites
    assert any("no longer works" in n for n in rep.notes)
    assert stat.S_IMODE((run / "run.env").stat().st_mode) == 0o600


def test_spec_variables_override_without_dropping_the_defaults(tmp_path, space, monkeypatch):
    _, api, _ = space
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: FakeBucket())
    spec = {**SPEC, "dashboard": DASH, "comms": {**SPEC["comms"], "space_variables": {
        "MSGBOARD_SNAPSHOT_INTERVAL": "30", "MSGBOARD_REPLICA": ""}}}
    rundir.init(tmp_path / "run", _folder(tmp_path, spec), env=ENV)
    variables = {v["key"]: v["value"] for v in api.calls[0][2]["space_variables"]}
    assert variables == {"MSGBOARD_SNAPSHOT_INTERVAL": "30",  # only because the spec says so
                         "MSGBOARD_REPLICA": ""}  # blanked: no replica


def test_swarms_sharing_a_bucket_get_their_own_replica(tmp_path, space):
    _, api, _ = space
    for name in ("demo", "demo-2"):
        rep = rundir.init(tmp_path / name, _folder(tmp_path / name, {**SPEC, "swarm": name}), env=ENV)
        assert {c.name: c for c in rep.checks}["board space"].ok
    replicas = [{v["key"]: v["value"] for v in kw["space_variables"]}["MSGBOARD_REPLICA"]
                for _, _, kw in api.calls]
    assert replicas == ["hf://buckets/acme/runs/board/demo", "hf://buckets/acme/runs/board/demo-2"]


def test_board_space_needs_a_template(tmp_path, space):
    from flock.models import CommsSpec

    comms = {k: v for k, v in SPEC["comms"].items() if k != "template"}
    with pytest.raises(ValueError, match="template"):
        CommsSpec.model_validate({**comms, "url": "https://b.invalid"})
    _, api, _ = space
    rep = rundir.init(tmp_path / "run", _folder(tmp_path, {**SPEC, "comms": comms}), env=ENV)
    c = {c.name: c for c in rep.checks}["board space"]
    assert c.ok is False and "template" in c.detail and not api.calls


def test_replica_needs_a_mapped_hf_token(tmp_path, space):
    _, api, _ = space
    spec = {**SPEC, "comms": {k: v for k, v in SPEC["comms"].items() if k != "space_secrets"}}
    rep = rundir.init(tmp_path / "run", _folder(tmp_path, spec), env=ENV)
    c = {c.name: c for c in rep.checks}["board space"]
    assert c.ok is False and "space_secrets" in c.detail and not api.calls


def test_health_wait_survives_connection_errors(tmp_path, space, monkeypatch):
    import httpx

    real, fails = httpx.get, []

    def flaky(url, **kw):
        if url.endswith("/v1/health") and not fails:
            fails.append(url)
            raise httpx.ConnectError("building")
        return real(url, **kw)

    monkeypatch.setattr(httpx, "get", flaky)
    monkeypatch.setattr(rundir, "_sleep", lambda s: None)
    rep = rundir.init(tmp_path / "run", _folder(tmp_path), env=ENV)
    assert fails and {c.name: c for c in rep.checks}["board space"].ok


def test_health_wait_holds_while_the_board_is_starting(tmp_path, space, monkeypatch):
    import httpx

    board, _, _ = space
    real, starting = httpx.get, []

    def slow(url, **kw):  # msgboard's Space app while it restores: 200, but not ready
        if url.endswith("/v1/health") and len(starting) < 2:
            starting.append(url)
            return httpx.Response(200, json={"status": "starting", "error": None})
        return real(url, **kw)

    monkeypatch.setattr(httpx, "get", slow)
    monkeypatch.setattr(rundir, "_sleep", lambda s: None)
    rep = rundir.init(tmp_path / "run", _folder(tmp_path), env=ENV)
    assert len(starting) == 2 and {c.name: c for c in rep.checks}["board space"].ok
    assert "general" in board.channels.values()


def test_health_wait_reports_a_board_that_never_starts(tmp_path, space, monkeypatch):
    import httpx

    board, _, _ = space
    monkeypatch.setattr(httpx, "get", lambda url, **kw: httpx.Response(
        200, json={"status": "starting", "error": "ValueError: bad operator token"}))
    monkeypatch.setattr(rundir, "_sleep", lambda s: None)
    monkeypatch.setattr(rundir, "BOARD_WAIT_S", 0)
    rep = rundir.init(tmp_path / "run", _folder(tmp_path), env=ENV)
    c = {c.name: c for c in rep.checks}["board space"]
    assert c.ok is False and "starting: ValueError: bad operator token" in c.detail
    assert "general" not in board.channels.values()  # nothing posted to a board that is not up


def test_board_auth_uses_the_hf_token_on_hf_spaces(monkeypatch):
    from flock.comms.board import _auth

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    space = "https://acme-board-demo.hf.space"
    assert _auth("fbt_b", space) == {"X-Msgboard-Token": "fbt_b", "Authorization": "Bearer hf_x"}
    assert _auth(None, space) == {"Authorization": "Bearer hf_x"}  # join on a private Space
    # elsewhere the HF token is never sent; old public boards still get the Bearer fallback
    assert _auth("fbt_b", "https://board.invalid") == {"X-Msgboard-Token": "fbt_b",
                                                        "Authorization": "Bearer fbt_b"}
    monkeypatch.delenv("HF_TOKEN")
    assert _auth("fbt_b", space)["Authorization"] == "Bearer fbt_b"


def test_an_existing_space_without_its_operator_token_is_refused(tmp_path, space):
    _, api, _ = space
    api.spaces.add("acme/board-demo")
    rep = rundir.init(tmp_path / "run", _folder(tmp_path), env=ENV)
    c = {c.name: c for c in rep.checks}["board space"]
    assert c.ok is False and "BOARD_OPERATOR_TOKEN" in c.detail and not api.calls


def test_board_url_form_and_space_without_init(tmp_path):
    f = _folder(tmp_path, {**SPEC, "comms": {**SPEC["comms"], "board": "https://b.invalid/boards/x/"}})
    assert read_spec(str(f / "spec.yaml")).comms.url == "https://b.invalid/boards/x"
    with pytest.raises(SpecError, match="BOARD_URL"):
        read_spec(str(_folder(tmp_path / "s") / "spec.yaml"))


ORG = "https://acme-board-demo.hf.space"  # a `board: space` board: the Space's root
SHARED = "https://acme-boards.hf.space/boards/demo"  # a board under a path, given by url
DASH = {"registry": "hf://buckets/acme/overview", "name": "Demo", "setup": "One line.",
        "viewer": "https://acme-viewer.hf.space"}


def _spec(**over):
    return SwarmSpec.model_validate({**SPEC, "comms": {**SPEC["comms"], "url": ORG},
                                     "dashboard": DASH, **over})


class FakeBucket:
    def __init__(self, fail=False, files=None):
        self.fail, self.files = fail, dict(files or {})

    def list(self, prefix):
        return [r for r in self.files if r.startswith(prefix)]

    def download(self, files):
        for r, p in files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self.files[r]))

    def upload(self, files):
        if self.fail:
            raise PermissionError("403")
        self.files.update({r: json.loads(p.read_text()) for p, r in files})


def test_dashboard_row_written_and_rewritten(monkeypatch):
    b = FakeBucket()
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: b)
    c = rundir.register_dashboard(_spec(), {"HF_TOKEN": "t"})
    assert c.ok, c.detail
    created = b.files["runs/demo.json"].pop("created_at")
    assert datetime.fromisoformat(created).utcoffset() == timedelta(0)  # UTC ISO 8601
    assert b.files["runs/demo.json"] == {
        "id": "demo", "name": "Demo", "title": None, "setup": "One line.", "board_id": None,
        "board": ORG + "/", "board_public": "https://huggingface.co/spaces/acme/board-demo",
        "viewer": "https://acme-viewer.hf.space",
        "viewer_public": "https://huggingface.co/spaces/acme/viewer",  # from the registry's namespace
        "viewer_run": "acme/runs/demo", "dataset": None, "code_repo": None, "spec_url": None}
    b.files["runs/demo.json"]["created_at"] = "2026-01-01T00:00:00+00:00"
    assert rundir.register_dashboard(_spec(dashboard={**DASH, "title": "T"}), {"HF_TOKEN": "t"}).ok
    assert b.files["runs/demo.json"]["title"] == "T"  # rewritten, first write's time kept
    assert b.files["runs/demo.json"]["created_at"] == "2026-01-01T00:00:00+00:00"
    # a board given by url: its page derived the same way
    url_board = {"invite_env": "BOARD_INVITE", "channel": "general", "url": SHARED}
    assert rundir.register_dashboard(_spec(comms=url_board), {"HF_TOKEN": "t"}).ok
    assert b.files["runs/demo.json"]["board_public"] == "https://huggingface.co/spaces/acme/boards"
    assert b.files["runs/demo.json"]["board_id"] == "demo"
    # dashboard.id: a second row next to an existing experiment's (e.g. a resumed copy of it)
    other = {**url_board, "url": SHARED.replace("/boards/demo", "/boards/demo-2")}  # its own board
    c = rundir.register_dashboard(_spec(comms=other, dashboard={**DASH, "id": "demo-2"}), {"HF_TOKEN": "t"})
    assert c.ok, c.detail
    assert b.files["runs/demo-2.json"]["id"] == "demo-2" and b.files["runs/demo.json"]["id"] == "demo"


@pytest.mark.parametrize("over, why", [
    ({"comms": {**SPEC["comms"], "url": "https://elsewhere.invalid/boards/demo"}}, "board"),
    # nothing hard-coded: another org's Spaces are refused for an acme registry
    ({"dashboard": {**DASH, "viewer": "https://huggingfaceswarms-flock-viewer-v8.hf.space"}}, "viewer"),
    ({"bucket": "hf://buckets/someone/runs"}, "namespace"),
])
def test_dashboard_refuses_rows_the_overview_drops(monkeypatch, over, why):
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: pytest.fail("written"))
    c = rundir.register_dashboard(_spec(**over), {"HF_TOKEN": "t"})
    assert c.ok is False and why in c.detail


def test_dashboard_refuses_a_board_id_another_run_uses(monkeypatch):
    b = FakeBucket(files={"runs/other.json": {"id": "other", "board_id": "demo"}})
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: b)
    shared = {"invite_env": "BOARD_INVITE", "channel": "general", "url": SHARED}
    c = rundir.register_dashboard(_spec(comms=shared), {"HF_TOKEN": "t"})
    assert c.ok is False and "board_id demo is already used" in c.detail and "runs/other.json" in c.detail
    assert "runs/demo.json" not in b.files
    del b.files["runs/other.json"]  # the same run's own row is no conflict
    b.files["runs/demo.json"] = {"id": "demo", "board_id": "demo"}
    assert rundir.register_dashboard(_spec(comms=shared), {"HF_TOKEN": "t"}).ok


def test_dashboard_write_failure_fails_the_check(monkeypatch):
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: FakeBucket(fail=True))
    c = rundir.register_dashboard(_spec(), {"HF_TOKEN": "t"})
    assert c.ok is False and "403" in c.detail


def test_dashboard_refused_with_compute_local_and_needs_a_viewer():
    with pytest.raises(ValueError, match="compute: local"):
        _spec(compute="local")
    with pytest.raises(ValueError, match="viewer"):
        _spec(dashboard={k: v for k, v in DASH.items() if k != "viewer"})


async def test_worker_join_sends_the_hf_token_only_to_hf_spaces(monkeypatch):
    import httpx

    from flock.worker.identity import join

    monkeypatch.setenv("HF_TOKEN", "hf_x")
    seen = []

    def handler(req):
        seen.append((req.url.host, req.headers.get("authorization")))
        return httpx.Response(200, json={"token": "t", "principal": {"id": 3, "external_id": "a"}})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as c:
        await join("https://x.hf.space/boards/b", "code", "a", client=c)
        await join("https://board.invalid/boards/b", "code", "a", client=c)
    assert seen == [("x.hf.space", "Bearer hf_x"), ("board.invalid", None)]
