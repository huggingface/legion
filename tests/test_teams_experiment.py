"""The teams experiment: one agent type, no spawning; agents form teams on the board."""
from __future__ import annotations

from pathlib import Path

from flock.cli import load_spec
from flock.models import ALL_COMMS_TOOLS, comms_tools

from .test_hierarchy_experiment import module

ROOT = Path(__file__).resolve().parents[1] / "experiments"
EXP = ROOT / "teams"
REPO = "HuggingFaceSwarms/chatbot-v7"
RULES = (  # Carlos's six rules, verbatim, with the tool insertions
    "You are one of several agents improving " + REPO + ". Help the team deliver a working result.",
    "1. Establish the goal. Read the brief, recent discussion, existing plan, and open changes: check `channels`, `search` the board and `list_prs` first.",
    "2. Build the shared plan.",
    "Put this in one shared planning document and update it as decisions change. Keep it in the main channel's shared doc: read it with `doc`, rewrite it with `edit_doc`.",
    "3. Form smaller teams. For each active workstream, start with 2–4 existing agents and create or reuse a dedicated channel (`create_channel` or `join_channel`).",
    "Coordinate each round's work in that channel;",
    "4. Claim and execute.",
    "5. Reassess.",
    "6. Verify and hand off. Test the combined behavior, follow the project's review process,",
    "When your PR merges, say so in the team channel.",
)


def test_one_type_no_spawn_and_the_rules():
    spec = load_spec(str(EXP / "spec.yaml"))
    run4 = load_spec(str(ROOT / "run4/spec.yaml")).types["agent"]
    assert set(spec.types) == {"agent"} and not spec.spawn
    agent = spec.types["agent"]
    skip = {"system_prompt": True}
    assert agent.replicas == 0 and agent.model_dump(exclude=skip) == run4.model_dump(exclude=skip)
    assert agent.system_prompt_append_file is None
    assert sorted(p.name for p in (EXP / "prompts").iterdir()) == ["final.md", "round.md", "system.md"]
    assert spec.limits.alive == 250 and spec.limits.spawned_total is None
    assert spec.comms.rules.max_channels == 100
    board = {"comms:" + n for n in ("channels", "create_channel", "join_channel", "search", "react",
                                    "post", "read", "doc", "edit_doc")}
    assert board <= set(ALL_COMMS_TOOLS) and board <= set(comms_tools(agent, spec.comms))
    text = agent.system_prompt
    for rule in RULES:
        assert rule in text, rule
    assert "Coordinate daily work" not in text and "All agents are equal" in text
    lower = text.lower()
    for banned in ("spawn", "flock remove", "$flock_parent", "vote", "voting", "#proposals",
                   "chatbot-v5", "chatbot-v6", "[project", "enforce", "flock prompt"):
        assert banned not in lower, banned
    assert f"`{REPO}` on Hugging Face" in text
    assert f"`https://huggingface.co/datasets/{REPO}` into `chatbot-swarm`" in text
    assert spec.comms.url == "https://huggingfaceswarms-chatbot-v7-msgboard.hf.space"


def test_render_and_waves_read_this_folder():
    data = module(EXP, "render").render()  # jobs: the Slurm overlay
    assert data["swarm"] == "teams" and set(data["types"]) == {"agent"} and "spawn" not in data
    smoke = module(EXP, "render").render("smoke", "slurm")  # the Slurm smoke
    assert smoke["types"]["agent"]["loop"]["while"]["until"] == "30m"
    waves = module(EXP, "waves").load(EXP / "waves.yaml")
    assert waves.type == "agent" and waves.cap == 250
    assert waves.events == [(600 * i, 20 * (i + 1)) for i in range(12)] + [(7200, 250)]
