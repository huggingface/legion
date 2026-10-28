"""The voting experiment: hierarchy's setup plus a vote before PRs and worker spawns."""
from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from flock.cli import load_spec
from flock.models import ALL_COMMS_TOOLS, load_spec_yaml

from . import fake_msgboard
from .test_comms import Team, custom
from .test_hierarchy_experiment import check_types, module
from .worker_fakes import until

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
    for text in ("PROPOSAL (PR):", "#proposals", "Do not join #proposals",
                 "at least 2 👍 and more 👍 than 👎", "Each round, if you have time", "You may abstain",
                 "A proposal is a direction", "`votes`", "@mention the proposer",
                 "Act only when it says passed", "[PROJECT: to be provided]"):
        assert text in coord and text in worker, text
    assert "A changed plan" not in coord + worker and "cmpatino" not in coord + worker
    for text in ("own proposal", "own vote", "yourself", "your own 👍"):  # say nothing on self-votes
        assert text not in coord and text not in worker, text
    assert "PROPOSAL (workers):" in coord and "PROPOSAL (workers):" not in worker
    assert "read #proposals and vote" in spec.types["worker"].loop.prompt
    assert spec.types["coordinator"].loop.prompt == spec.types["worker"].loop.prompt
    assert load_spec(str(HIER / "spec.yaml")).types["worker"].system_prompt in worker


@pytest.fixture
def board():
    b, server, url = fake_msgboard.serve()
    b.url = url
    yield b
    server.shutdown()


async def test_proposals_channel_works_without_joining_and_wakes_only_mentions(tmp_path, board):
    """#proposals is public but not joined: agents read and post there, and a proposal wakes only
    the agents it @mentions."""
    team = Team(tmp_path, board, ["cedar", "birch", "pine", "oak"])
    team.hub.poll_min_interval = 0.01
    team.hub.poll_cycle_s = 0.03
    team.spec.extended = True
    team.spec.wake = ["mention", "thread", "dm", "channel"]  # as spec.yaml
    team.hub.configure(team.spec, list(ALL_COMMS_TOOLS))
    try:
        await team.start()
        # seeded after the agents joined (the operator's step), so only cedar is a member
        out, error = await team.call("cedar", "create_channel", name="proposals", purpose="Proposals")
        assert not error, out
        out, error = await team.call("pine", "post", channel="proposals", thread="new",
                                     text="PROPOSAL (PR): memory module. @birch")
        assert not error, out
        page, error = await team.call("oak", "read", channel="proposals")
        assert not error and page["messages"][-1]["id"] == out["posted"]
        await until(lambda: custom(team.agents["birch"], "comms_mention"))
        await asyncio.sleep(0.3)  # a few poll cycles
        for kind in ("comms_mention", "comms_channel_message", "comms_thread_reply"):
            assert not custom(team.agents["oak"], kind), kind
        assert custom(team.agents["cedar"], "comms_channel_message")  # a member is woken
    finally:
        await team.close()
