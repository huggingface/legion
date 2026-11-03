"""No experiment spec in the repo still sets a key flock removed (models.REMOVED)."""

from __future__ import annotations

from pathlib import Path

from flock.models import REMOVED, load_spec_yaml

ROOT = Path(__file__).resolve().parents[1]


def test_no_experiment_sets_a_removed_loop_key():
    found = []
    for f in sorted((ROOT / "experiments").rglob("*.yaml")):
        doc = load_spec_yaml(f.read_text())
        types = doc.get("types") if isinstance(doc, dict) else None
        for name, t in (types or {}).items():
            lp = (t or {}).get("loop") or {}
            for cls, part in (("LoopSpec", lp), ("LoopWhile", lp.get("while") or {}),
                              ("LoopFinal", lp.get("final") or {})):
                found += [f"{f.relative_to(ROOT)}: {name} {k}" for k in REMOVED[cls] & set(part)]
    assert found == []
