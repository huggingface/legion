"""Resume gate: specs of runs we already have must still load after spec fields are deleted.
Only runs from the run5 branch onwards must resume (RESUMABLE; v8.json was generated on the v8
branch). Existing fixtures are frozen (they stand for swarms already stored); `uv run python
tests/test_resume_gate.py` only writes missing ones (hermetic: shell ${VAR}s are ignored, only
spec defaults apply; the repo path is written as `<repo>`)."""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path

import pytest
import yaml

from flock import models
from flock.cli import SpecError, read_spec
from flock.controller.store import Store
from flock.models import SwarmSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures" / "stored_specs"
EXPERIMENTS = sorted(p.parent for p in (ROOT / "experiments").glob("*/spec.yaml"))
RESUMABLE = {"hierarchy", "voting", "teams", "interviewer", "levels", "v8"}


def load_experiment(d: Path) -> SwarmSpec:
    return read_spec(str(d / "spec.yaml"))


def stored_json(spec: SwarmSpec) -> str:
    """What the controller persists: the applied JSON re-validated, then put_swarm's dump."""
    return SwarmSpec.model_validate(spec.model_dump(mode="json")).model_dump_json(indent=1)


@pytest.mark.parametrize("d", EXPERIMENTS, ids=lambda d: d.name)
def test_experiment_spec_loads_without_removed_keys(d: Path, monkeypatch, caplog) -> None:
    monkeypatch.setattr(models, "_warned", set())
    with caplog.at_level(logging.WARNING, logger="flock.models"):
        load_experiment(d)
    assert not [r for r in caplog.records if "was removed from flock" in r.getMessage()]


@pytest.mark.parametrize("f", sorted(FIXTURES.glob("*.json")), ids=lambda f: f.stem)
def test_stored_spec_resumes(f: Path, tmp_path: Path) -> None:
    spec = SwarmSpec.model_validate_json(f.read_text())
    store = Store(tmp_path)
    store.put_swarm(spec, 0.0)
    assert store.get_swarm(spec.swarm)["spec"] == spec


def test_fixtures_cover_every_resumable_run() -> None:
    assert {f.stem for f in FIXTURES.glob("*.json")} == RESUMABLE


def test_removed_key_dropped_typo_refused(monkeypatch, caplog, tmp_path: Path) -> None:
    monkeypatch.setitem(models.REMOVED, "LoopSpec", models.REMOVED["LoopSpec"] | {"gone_opt"})
    monkeypatch.setattr(models, "_warned", set())
    raw = json.loads((FIXTURES / "hierarchy.json").read_text())
    name = next(n for n, t in raw["types"].items() if t.get("loop"))
    raw["types"][name]["loop"]["gone_opt"] = "5s"
    with caplog.at_level(logging.WARNING, logger="flock.models"):  # a stored spec: dropped
        assert "gone_opt" not in SwarmSpec.model_validate(raw).model_dump_json()
        SwarmSpec.model_validate(raw)
    assert [r.getMessage() for r in caplog.records] == [
        "spec: LoopSpec.gone_opt was removed from flock; ignoring it"]
    monkeypatch.setenv("WORLD_DIR", str(ROOT / "worlds" / "chatbot4"))  # as `flock init` does
    spec = shutil.copytree(ROOT / "experiments" / "hierarchy", tmp_path / "h") / "spec.yaml"
    data = models.load_spec_yaml(spec.read_text())
    loop = data["types"][name]["loop"]
    loop["gone_opt"] = "5s"
    spec.write_text(yaml.safe_dump(data))
    read_spec(str(spec))  # a spec file: dropped too
    loop["gone_optt"] = loop.pop("gone_opt")
    spec.write_text(yaml.safe_dump(data))
    with pytest.raises(SpecError, match="gone_optt"):  # a typo: still refused
        read_spec(str(spec))

if __name__ == "__main__":
    os.environ.clear()  # identical fixtures for everyone: no shell ${VAR}s
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for d in EXPERIMENTS:
        # frozen: regenerating would hide a deletion
        if d.name not in RESUMABLE or (FIXTURES / f"{d.name}.json").exists():
            continue
        text = stored_json(load_experiment(d)).replace(str(ROOT), "<repo>")
        (FIXTURES / f"{d.name}.json").write_text(text + "\n")
        print(FIXTURES / f"{d.name}.json")
