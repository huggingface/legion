"""`comms.board: space` (flock init duplicates a board Space, fakes only) and `dashboard:` (the
run's flock-overview row written to a registry bucket)."""

from __future__ import annotations

import json
import stat

import pytest
import yaml

from flock import rundir
from flock.models import SwarmSpec
from flock.ops import SpecError, read_spec
from flock.specrender import read_env_file
from tests.fake_msgboard import Board, serve

SPEC = {
    "swarm": "demo", "namespace": "HuggingFaceSwarms",
    "bucket": "hf://buckets/HuggingFaceSwarms/runs",
    "comms": {"board": "space", "invite_env": "BOARD_INVITE", "channel": "general"},
    "types": {"a": {"harness": "claude", "model": "claude-opus-5-5", "comms": True}},
}


class FakeApi:
    """HfApi's Space calls; a duplicated Space derives its board's admin token, as msgboard does."""

    def __init__(self, board: Board):
        self.board, self.spaces, self.calls = board, set(), []

    def repo_exists(self, repo_id, repo_type=None):
        return repo_id in self.spaces

    def duplicate_repo(self, from_id, to_id, **kw):
        self.calls.append((from_id, to_id, kw))
        self.spaces.add(to_id)
        master = next(s["value"] for s in kw["space_secrets"] if s["key"] == "MSGBOARD_OPERATOR_TOKEN")
        self.board.tokens[rundir.board_token(master, "demo")] = self.board.principal("operator", "user")["id"]


@pytest.fixture
def space(monkeypatch):
    board, srv, url = serve(Board(channels={1: "random"}), prefix="/boards/demo")
    api = FakeApi(board)
    monkeypatch.setattr(rundir, "hub_api", lambda token: api)
    monkeypatch.setattr(rundir, "space_url", lambda sid: url.removesuffix("/boards/demo"))
    monkeypatch.setattr(rundir, "check_venv", lambda env: rundir.Check("venv", True))
    yield board, api, url
    srv.shutdown()


def _folder(tmp_path, spec=SPEC):
    f = tmp_path / "exp"
    f.mkdir(parents=True)
    (f / "spec.yaml").write_text(yaml.safe_dump(spec))
    return f


def test_init_creates_the_board_once_then_reuses_it(tmp_path, space):
    board, api, url = space
    run = tmp_path / "run"
    rep = rundir.init(run, _folder(tmp_path), env={"HF_TOKEN": "hf_x"})
    checks = {c.name: c for c in rep.checks}
    assert checks["board space"].ok, checks["board space"].detail
    assert checks["spec"].ok and "board" not in checks  # the provisioning replaces the health check
    (src, to, kw), = api.calls
    assert (src, to) == ("HuggingFaceSwarms/chatbot-v10-msgboard", "HuggingFaceSwarms/board-demo")
    assert kw["private"] and {s["key"] for s in kw["space_secrets"]} == {"MSGBOARD_OPERATOR_TOKEN", "HF_TOKEN"}
    variables = {v["key"]: v["value"] for v in kw["space_variables"]}
    assert json.loads(variables["MSGBOARD_BOARDS"]) == [
        {"id": "demo", "name": "demo", "storage_id": "demo", "require_existing": False}]
    assert variables["BOARD_READ_TOKEN"] == "public"
    assert "general" in board.channels.values()
    values = read_env_file(run / "run.env")
    assert values["BOARD_URL"] == url and values["BOARD_INVITE"] in board.invites
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
    rep = rundir.init(run, env={"HF_TOKEN": "hf_x"})
    assert {c.name: c for c in rep.checks}["board space"].ok and len(api.calls) == 1
    assert read_env_file(run / "run.env")["BOARD_INVITE"] == values["BOARD_INVITE"]
    assert list(board.channels.values()).count("general") == 1


def test_an_existing_space_without_its_operator_token_is_refused(tmp_path, space):
    _, api, _ = space
    api.spaces.add("HuggingFaceSwarms/board-demo")
    rep = rundir.init(tmp_path / "run", _folder(tmp_path), env={"HF_TOKEN": "hf_x"})
    c = {c.name: c for c in rep.checks}["board space"]
    assert c.ok is False and "BOARD_OPERATOR_TOKEN" in c.detail and not api.calls


def test_board_url_form_and_space_without_init(tmp_path):
    f = _folder(tmp_path, {**SPEC, "comms": {**SPEC["comms"], "board": "https://b.invalid/boards/x/"}})
    assert read_spec(str(f / "spec.yaml")).comms.url == "https://b.invalid/boards/x"
    with pytest.raises(SpecError, match="BOARD_URL"):
        read_spec(str(_folder(tmp_path / "s") / "spec.yaml"))


ORG = "https://huggingfaceswarms-board-demo.hf.space/boards/demo"
DASH = {"registry": "hf://buckets/HuggingFaceSwarms/overview", "name": "Demo", "setup": "One line."}


def _spec(**over):
    return SwarmSpec.model_validate({**SPEC, "comms": {**SPEC["comms"], "url": ORG},
                                     "dashboard": DASH, **over})


class FakeBucket:
    def __init__(self, fail=False):
        self.fail, self.files = fail, {}

    def upload(self, files):
        if self.fail:
            raise PermissionError("403")
        self.files.update({r: json.loads(p.read_text()) for p, r in files})


def test_dashboard_row_written_and_rewritten(monkeypatch):
    b = FakeBucket()
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: b)
    c = rundir.register_dashboard(_spec(), {"HF_TOKEN": "t"})
    assert c.ok, c.detail
    assert b.files["runs/demo.json"] == {
        "id": "demo", "name": "Demo", "title": None, "setup": "One line.", "board_id": "demo",
        "board": ORG + "/", "board_public": "https://huggingface.co/spaces/HuggingFaceSwarms/board-demo",
        "viewer": "https://huggingfaceswarms-flock-viewer-v8.hf.space", "viewer_public": None,
        "viewer_run": "HuggingFaceSwarms/runs/demo", "dataset": None, "code_repo": None,
        "spec_url": None}
    assert rundir.register_dashboard(_spec(dashboard={**DASH, "title": "T"}), {"HF_TOKEN": "t"}).ok
    assert b.files["runs/demo.json"]["title"] == "T"


@pytest.mark.parametrize("over, why", [
    ({"comms": {**SPEC["comms"], "url": "https://elsewhere.invalid/boards/demo"}}, "board"),
    ({"dashboard": {**DASH, "viewer": "https://someone-viewer.hf.space"}}, "viewer"),
    ({"bucket": "hf://buckets/cmpatino/runs"}, "namespace"),
])
def test_dashboard_refuses_rows_the_overview_drops(monkeypatch, over, why):
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: pytest.fail("written"))
    c = rundir.register_dashboard(_spec(**over), {"HF_TOKEN": "t"})
    assert c.ok is False and why in c.detail


def test_dashboard_write_failure_fails_the_check(monkeypatch):
    monkeypatch.setattr(rundir, "bucket_store", lambda url, token: FakeBucket(fail=True))
    c = rundir.register_dashboard(_spec(), {"HF_TOKEN": "t"})
    assert c.ok is False and "403" in c.detail


def test_dashboard_refused_with_compute_local():
    with pytest.raises(ValueError, match="compute: local"):
        _spec(compute="local")
