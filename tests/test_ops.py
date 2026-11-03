"""flock.ops: `flock launch` / `flock update` as library calls (no printing, no typer)."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient

from flock.client import Client
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
    lines = update(api, read_spec(f, sources=sources), f, sources=sources, replicas="spec",
                   check=True)
    assert lines[-1] == "check ok: nothing was applied"
    assert api.swarm("demo")["spec"]["types"]["researcher"]["replicas"] == 3
    assert any("replicas" in x for x in update(api, read_spec(f), f, replicas="spec"))
    assert api.swarm("demo")["spec"]["types"]["researcher"]["replicas"] == 5
    assert update(api, read_spec(f), f) == ["no changes"]
    with pytest.raises(SpecError, match="use keep or spec"):
        update(api, read_spec(f), f, replicas="all")
