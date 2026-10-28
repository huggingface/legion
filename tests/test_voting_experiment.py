"""The voting experiment: hierarchy's setup plus a vote before PRs and worker spawns."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec
from flock.models import load_spec_yaml

from .test_hierarchy_experiment import check_types, module

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP, HIER = ROOT / "voting", ROOT / "hierarchy"


def test_same_spec_as_hierarchy_but_the_name():
    voting, hier = (load_spec_yaml((d / "spec.yaml").read_text()) for d in (EXP, HIER))
    assert voting.pop("swarm") == "voting" and hier.pop("swarm") == "hierarchy"
    assert voting == hier
    data = module(EXP, "render").render(placeholders=True)
    assert data["swarm"] == "voting" and set(data["types"]) == {"coordinator", "worker"}
    assert module(EXP, "waves").load(EXP / "waves.yaml").type == "coordinator"


def test_role_prompts_are_hierarchy_plus_voting():
    spec, coord, worker = check_types(EXP)
    for role in ("coordinator", "worker"):
        text = (EXP / f"prompts/{role}.md").read_text()
        base = (HIER / f"prompts/{role}.md").read_text()
        assert text.startswith(base) and "## Voting" in text[len(base):], role
    for text in ("PROPOSAL (PR):", "at least 2 👍 and more 👍 than 👎", "Vote every round",
                 "`votes`", "@mention the proposer", "Act only when it says passed"):
        assert text in coord and text in worker, text
    for text in ("own proposal", "own vote", "yourself", "your own 👍"):  # say nothing on self-votes
        assert text not in coord and text not in worker, text
    assert "PROPOSAL (workers):" in coord and "PROPOSAL (workers):" not in worker
    assert "vote on new proposals" in spec.types["worker"].loop.prompt
    assert spec.types["coordinator"].loop.prompt == spec.types["worker"].loop.prompt
    assert load_spec(str(HIER / "spec.yaml")).types["worker"].system_prompt in worker
