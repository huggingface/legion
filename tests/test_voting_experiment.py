"""The voting experiment: hierarchy's setup plus a vote before PRs and worker spawns."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec
from flock.models import load_spec_yaml

from .test_hierarchy_experiment import REPOS, check_types

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP, HIER = ROOT / "voting", ROOT / "hierarchy"


def test_same_spec_as_hierarchy_but_the_name():
    voting, hier = (load_spec_yaml((d / "spec.yaml").read_text()) for d in (EXP, HIER))
    assert voting.pop("swarm") == "${CHATBOT_RUN:-voting}" and hier.pop("swarm") == "${CHATBOT_RUN:-hierarchy}"
    assert voting == hier


def test_role_prompts_are_hierarchy_plus_voting():
    spec, coord, worker = check_types(EXP)
    for role in ("coordinator", "worker"):
        text = (EXP / f"prompts/{role}.md").read_text()
        base = (HIER / f"prompts/{role}.md").read_text().replace(REPOS["hierarchy"], REPOS["voting"])
        assert text.startswith(base) and "## Voting" in text[len(base):], role
    for text in ("PROPOSAL (PR):", "in the channel of the PR's topic (your team channel)",
                 "at least 2 👍 and more 👍 than 👎", "Each round, if you have time", "You may abstain",
                 "A proposal is a direction", "`votes`", "@mention the proposer",
                 "Act only when it says passed", REPOS["voting"]):
        assert text in coord and text in worker, text
    assert "A changed plan" not in coord + worker and "cmpatino" not in coord + worker
    for text in ("own proposal", "own vote", "yourself", "your own 👍"):  # say nothing on self-votes
        assert text not in coord and text not in worker, text
    assert "PROPOSAL (workers):" in coord and "PROPOSAL (workers):" not in worker
    assert "vote on open proposals in your channels and in #general" in spec.types["worker"].loop.prompt
    assert "Post a new-team proposal as a new thread in #general" in coord
    assert "new-team proposal" not in worker
    for f in (EXP / "prompts").iterdir():  # no separate proposals channel
        assert "#proposals" not in f.read_text(), f.name
    assert spec.types["coordinator"].loop.prompt == spec.types["worker"].loop.prompt
    hier = load_spec(str(HIER / "spec.yaml")).types["worker"].system_prompt
    assert hier.replace(REPOS["hierarchy"], REPOS["voting"]) in worker and REPOS["hierarchy"] not in worker

