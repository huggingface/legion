"""Resume gate: specs of runs we already have must still load after spec fields are deleted.
Only runs from the run5 branch onwards must resume (RESUMABLE; v8.json was generated on the v8
branch). The frozen stored specs live in the private HF dataset DATASET (stored_specs/<name>.json),
pinned to REVISION and fetched through the normal HF cache and token; without access the tests
skip. Rule of the dataset: never regenerate an existing file (they stand for swarms already
stored; regenerating would hide a deletion). To add a run: `uv run python tests/test_resume_gate.py
<dir>` writes the missing specs (hermetic: shell ${VAR}s are ignored, the repo path is `<repo>`),
upload the new file to the dataset, bump REVISION to the new commit and add the name to RESUMABLE."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

from flock import models
from flock.cli import SpecError, read_spec
from flock.controller.store import Store
from flock.models import SwarmSpec

ROOT = Path(__file__).resolve().parents[1]
DATASET = "HuggingFaceSwarms/flock-swarmchat-resume-spec"
REVISION = "51fa0dcd29698936f51ac6cf6750d945b47fa58f"
EXPERIMENTS = sorted(p.parent for p in (ROOT / "experiments").glob("*/spec.yaml"))
RESUMABLE = {"hierarchy", "voting", "teams", "interviewer", "levels", "v8"}


def stored_spec(name: str) -> Path:
    try:
        return Path(hf_hub_download(DATASET, f"stored_specs/{name}.json", repo_type="dataset",
                                    revision=REVISION))
    except (OSError, HfHubHTTPError) as e:  # offline (OSError) or 401 / 404
        pytest.skip(f"cannot fetch {DATASET}@{REVISION[:7]} stored_specs/{name}.json: {e!r}"[:300])


def load_experiment(d: Path) -> SwarmSpec:
    return read_spec(str(d / "spec.yaml"))


def stored_json(spec: SwarmSpec) -> str:
    """What the controller persists: the applied JSON re-validated, then put_swarm's dump."""
    return SwarmSpec.model_validate(spec.model_dump(mode="json")).model_dump_json(indent=1)


@pytest.mark.parametrize("name", sorted(RESUMABLE))
def test_stored_spec_resumes(name: str, tmp_path: Path) -> None:
    spec = SwarmSpec.model_validate_json(stored_spec(name).read_text())
    store = Store(tmp_path)
    store.put_swarm(spec, 0.0)
    assert store.get_swarm(spec.swarm)["spec"] == spec


def test_fixtures_cover_every_resumable_run() -> None:
    try:
        files = HfApi().list_repo_files(DATASET, repo_type="dataset", revision=REVISION)
    except (OSError, HfHubHTTPError) as e:  # offline (OSError) or 401 / 404
        pytest.skip(f"cannot list {DATASET}@{REVISION[:7]}: {e!r}"[:300])
    assert {Path(f).stem for f in files if f.startswith("stored_specs/")} == RESUMABLE


def test_removed_key_dropped_typo_refused(monkeypatch, caplog, tmp_path: Path) -> None:
    monkeypatch.setitem(models.REMOVED, "LoopSpec", models.REMOVED["LoopSpec"] | {"gone_opt"})
    monkeypatch.setattr(models, "_warned", set())
    raw = json.loads(stored_spec("hierarchy").read_text())
    name = next(n for n, t in raw["types"].items() if t.get("loop"))
    raw["types"][name]["loop"]["gone_opt"] = "5s"
    with caplog.at_level(logging.WARNING, logger="flock.models"):  # a stored spec: dropped
        assert "gone_opt" not in SwarmSpec.model_validate(raw).model_dump_json()
        SwarmSpec.model_validate(raw)
    assert [r.getMessage() for r in caplog.records] == [
        "spec: LoopSpec.gone_opt was removed from flock; ignoring it"]
    spec = tmp_path / "spec.yaml"
    loop = {"prompt": "next round", "gone_opt": "5s"}
    data = {"swarm": "h", "compute": {"kind": "local"},
            "types": {name: {"harness": "pi", "model": "claude-opus-5-5", "loop": loop}}}
    spec.write_text(yaml.safe_dump(data))
    read_spec(str(spec))  # a spec file: dropped too
    loop["gone_optt"] = loop.pop("gone_opt")
    spec.write_text(yaml.safe_dump(data))
    with pytest.raises(SpecError, match="gone_optt"):  # a typo: still refused
        read_spec(str(spec))

if __name__ == "__main__":
    out = Path(sys.argv[1])
    os.environ.clear()  # identical specs for everyone: no shell ${VAR}s
    out.mkdir(parents=True, exist_ok=True)
    for d in EXPERIMENTS:
        if d.name in RESUMABLE:  # compare with the dataset: never overwrite an existing file there
            text = stored_json(load_experiment(d)).replace(str(ROOT), "<repo>")
            (out / f"{d.name}.json").write_text(text + "\n")
            print(out / f"{d.name}.json")
