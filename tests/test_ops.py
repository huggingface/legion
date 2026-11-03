"""flock.ops: `flock launch` / `flock update` as library calls (no printing, no typer)."""

from __future__ import annotations

import os

import pytest
import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli, ops
from flock.cli import update_lines
from flock.client import Client, ClientError
from flock.current import write_current
from flock.ops import SpecError, launch, read_spec, update

from .conftest import TOKEN, make_spec


@pytest.fixture
def api(monkeypatch, app) -> Client:
    monkeypatch.setattr("flock.client.connect",
                        lambda *_: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    return Client()


def _folder(tmp_path, replicas: int = 3) -> str:
    f = tmp_path / "spec.yaml"
    raw = make_spec(researcher={"replicas": replicas}).model_dump(mode="json", exclude_none=True)
    f.write_text(yaml.safe_dump(raw))
    return str(f)


def test_launch_creates_then_resumes_then_refuses_a_changed_spec(api, tmp_path, store, tick):
    f = _folder(tmp_path)
    done = launch(api, read_spec(f), f)
    assert done["status"]["swarm"] == "demo" and "resumed" not in done and done["warnings"] == []
    tick()
    api.stop("demo", reason="night")
    done = launch(api, read_spec(f), f)
    assert done["resumed"]["held"] == {"types": {}, "agents": {}} and "status" not in done

    spec = read_spec(f)
    spec.types["researcher"].agents_per_worker = 3
    with pytest.raises(SpecError) as e:
        launch(api, spec, f)
    assert "run `flock update` first" in e.value.lines[-1] and e.value.done["warnings"] == []


def test_launch_adopt_without_a_checkpoint(api, tmp_path):
    f = _folder(tmp_path)
    with pytest.raises(SpecError) as e:
        launch(api, read_spec(f), f, adopt=True)
    assert e.value.lines[0].startswith("refused: ") and "--no-checkpoint" in e.value.lines[1]
    done = launch(api, read_spec(f), f, adopt=True, no_checkpoint=True)
    assert done["adopt_skipped"] and done["status"]["swarm"] == "demo"


def test_update_check_then_apply(api, tmp_path):
    f = _folder(tmp_path)
    launch(api, read_spec(f), f)
    f = _folder(tmp_path, replicas=5)
    sources: dict[str, str] = {}
    done = update(api, read_spec(f, sources=sources), f, sources=sources, replicas="spec",
                  check=True)
    assert not done["applied"] and any("spec.yaml" in x for x in done["diff"])
    assert update_lines(done, check=True)[-1] == "check ok: nothing was applied"
    assert api.swarm("demo")["spec"]["types"]["researcher"]["replicas"] == 3
    done = update(api, read_spec(f), f, replicas="spec")
    assert done["applied"] and any("replicas" in x for x in done["diff"])
    assert api.swarm("demo")["spec"]["types"]["researcher"]["replicas"] == 5
    done = update(api, read_spec(f), f)
    assert not done["applied"] and done["diff"] == [] and update_lines(done) == ["no changes"]
    with pytest.raises(SpecError, match="use keep or spec"):
        update(api, read_spec(f), f, replicas="all")


def test_a_failed_call_after_the_preflight_keeps_what_was_done(api, tmp_path, monkeypatch):
    f = _folder(tmp_path)

    def conflict(*a, **k):
        raise ClientError("error 409: exists", 409, '{"detail": "exists"}')

    monkeypatch.setattr(Client, "create", conflict)
    with pytest.raises(SpecError) as e:
        launch(api, read_spec(f), f, preflight=True)
    assert e.value.lines == ["error 409: exists"] and e.value.done["preflight"]["ok"]
    r = CliRunner().invoke(cli.app, ["launch", f, "--preflight"])  # the CLI still shows it
    assert r.exit_code == 1 and "preflight: no provider-backed models" in r.output
    assert r.output.rstrip().endswith("error 409: exists")


def test_start_local_refuses_another_swarms_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLOCK_URL", raising=False)
    monkeypatch.delenv("FLOCK_CURRENT", raising=False)
    write_current(tmp_path / ".flock", {"url": "http://127.0.0.1:1", "token": "t",
                                        "pid": os.getpid(), "swarm": "other"})
    with pytest.raises(SpecError, match="a local run of other is up"):
        ops.start_local(make_spec(), "spec.yaml")
    assert ops.start_local(make_spec("other"), "spec.yaml") is None  # already serves it
    monkeypatch.setenv("FLOCK_URL", "http://elsewhere:1")
    assert ops.start_local(make_spec(), "spec.yaml") is None  # an explicit controller


def test_release_wait_polls_until_no_worker_runs(monkeypatch):
    class Fake:
        def __init__(self):
            self.live = [[{"job_id": "j1"}, {"job_id": "j2"}], [{"job_id": "j2"}],
                         [{"job_id": None}]]

        def workers(self, swarm):
            return self.live.pop(0)

        def swarm(self, swarm):
            return {"status": {"released": {"unsynced": {"a-0001": "503"}}}}

    seen: list[int] = []
    monkeypatch.setattr(ops, "_sleep", lambda s: None)
    assert ops.release_wait(Fake(), "demo", progress=seen.append) == {"a-0001": "503"}
    assert seen == [2, 1]
