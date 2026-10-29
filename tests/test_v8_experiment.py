"""The v8 experiment: six lanes as agent types, ramped by the controller, on the v8 seed."""
from pathlib import Path

import pytest

from flock.cli import load_spec
from flock.models import ALL_COMMS_TOOLS, comms_tools

from .test_hierarchy_experiment import module

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/v8"
SEED = ROOT / "worlds/chatbot4/seed-v8"
REPO = "HuggingFaceSwarms/SwarmChat-v8"
LANES = {"answers": (12, 12, "10m", 96), "understanding": (6, 6, "10m", 24),
         "conversation": (6, 6, "10m", 48),
         "routing": (6, 6, "20m", 24), "quality": (6, 6, "15m", 36), "safety": (3, 3, "30m", 6)}
# The swarm is steered by STYLE.md and the lanes, never told how it is scored.
HIDDEN = ("smoltalk", "benchmark", "leaderboard", "win rate", "evaluation", "interviewer", "turing")


@pytest.fixture(autouse=True)
def _spec_defaults(monkeypatch):  # the spec's ${VAR:-default}s, not the caller's run.env
    for var in ("CHATBOT_BOARD_URL", "CHATBOT_RUN"):
        monkeypatch.delenv(var, raising=False)


def test_six_lanes_are_run4_agents_with_their_own_ramp_and_prompt():
    spec = load_spec(str(EXP / "spec.yaml"))
    run4 = load_spec(str(ROOT / "experiments/run4/spec.yaml")).types["agent"]
    assert set(spec.types) == set(LANES) and not spec.spawn
    skip = {"system_prompt": True, "system_prompt_append_file": True, "replicas_ramp": True,
            "tools": True, "loop": {"on"}}
    board = [t for t in ALL_COMMS_TOOLS if t not in ("comms:dm", "comms:read_dms", "comms:votes")]
    for name, (start, add, every, top) in LANES.items():
        lane = spec.types[name]
        assert lane.model_dump(exclude=skip) == run4.model_dump(exclude=skip), name
        # no DMs: everything is public, and nothing wakes on one
        assert sorted(comms_tools(lane, spec.comms)) == sorted(board), name
        assert lane.tools[:len(run4.tools)] == run4.tools
        assert lane.loop.on == ["comms.mention", "comms.thread", "comms.channel"], name
        ramp = lane.replicas_ramp
        assert (ramp.start, ramp.add, ramp.every, ramp.max) == (start, add, every, top)
        assert lane.system_prompt_append_file == f"prompts/lanes/{name}.md"
        assert f"## Your lane: {name} (#{name})" in lane.system_prompt
    assert sum(r[3] for r in LANES.values()) <= spec.limits.alive == 250
    assert not (EXP / "waves.yaml").exists()  # controller.sbatch: replicas_ramp, no waves runner
    rules = spec.comms.rules
    assert rules.no_group_mentions and rules.max_len == 1200 and rules.max_channels == 40
    assert spec.comms.wake == ["mention", "thread", "channel"]
    assert spec.comms.url == "https://huggingfaceswarms-chatbot-v8-msgboard.hf.space"


def test_the_prompts_name_the_repo_and_lanes_and_hide_the_eval():
    text = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in text
    assert f"`https://huggingface.co/datasets/{REPO}` into `chatbot-swarm`" in text
    for name in LANES:
        assert f"| {name} | #{name} |" in text
    for part in ("an understanding engine", "a memory store", "a knowledge base", "a response composer"):
        assert part in text, part
    assert "prefer general mechanisms" in text and "millions of lines" not in text
    assert "`<lane>-<area>`" in text and "40 channels" in text and "propose it on #general" in text
    for path in [EXP / "prompts/system.md", *sorted((EXP / "prompts/lanes").glob("*.md"))]:
        lower = path.read_text().lower()
        for word in (*HIDDEN, "chatbot-v", "spawn", "vote", "@all", "read_dms", " dm,", "dms promptly"):
            assert word not in lower, (path.name, word)


def test_the_seed_documents_the_interface_and_hides_the_eval():
    for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "STYLE.md"):
        lower = (SEED / name).read_text().lower()
        for word in HIDDEN:
            assert word not in lower, (name, word)
    assert f"datasets/{REPO} chatbot-swarm" in (SEED / "CONTRIBUTING.md").read_text()
    readme = (SEED / "README.md").read_text()
    for hook in ("observe(text, memory)", "respond(text, memory)", "after(text, reply, memory)",
                 "choose(text, memory, candidates)", 'memory["turns"]', 'memory["understood"]',
                 "bot/understanding/", "bot/memory/", "bot/knowledge/", "bot/compose/"):
        assert hook in readme, hook
    for package in ("understanding", "memory", "knowledge", "compose"):
        assert (SEED / "bot" / package / "__init__.py").is_file(), package


def test_render_reads_this_folder():
    data = module(EXP, "render").render()  # jobs: the Slurm overlay
    assert data["swarm"] == "v8" and list(data["types"]) == list(LANES) and "spawn" not in data
    smoke = module(EXP, "render").render("smoke", "slurm")
    for name, lane in smoke["types"].items():
        assert lane["loop"]["while"]["until"] == "30m", name
        assert lane["replicas_ramp"]["start"] == lane["replicas_ramp"]["max"], name
    assert sum(lane["replicas_ramp"]["max"] for lane in smoke["types"].values()) == 16
