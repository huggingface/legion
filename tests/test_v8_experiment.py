"""The v8 experiment: six lanes as agent types, ramped by the controller, on the v8 seed; one
spec that resumes the stored v8 swarm with only the intended differences (README)."""
from pathlib import Path

import pytest

from flock.models import ALL_COMMS_TOOLS, SwarmSpec, comms_tools
from flock.ops import read_spec, resume_diff

from .test_resume_gate import stored_spec

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
def _env(monkeypatch):  # the spec's defaults, not the caller's run.env; the profile's partition
    for var in ("CHATBOT_BOARD_URL", "CHATBOT_REPO", "OPENAI_BASE_URLS", "MODEL_WORKER",
                "WORKER_CPUS", "WORKER_MEM", "WORLD_DIR", "CHATBOT_CHANNEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SLURM_PARTITION", "hopper-cpu")
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/HuggingFaceSwarms/chatbot-agent-workspaces/v8")


def test_six_lanes_with_their_own_ramp_and_prompt_and_no_dms():
    spec = read_spec(str(EXP / "spec.yaml"))
    assert set(spec.types) == set(LANES) and not spec.spawn
    board = [t for t in ALL_COMMS_TOOLS if t not in ("comms:dm", "comms:read_dms", "comms:votes")]
    for name, (start, add, every, top) in LANES.items():
        lane = spec.types[name]
        assert sorted(comms_tools(lane, spec.comms)) == sorted(board), name
        assert lane.loop.on == ["comms.mention", "comms.thread", "comms.channel"], name
        ramp = lane.replicas_ramp
        assert (ramp.start, ramp.add, ramp.every, ramp.max, ramp.hold_if) == (start, add, every, top, None)
        assert f"## Your lane: {name} (#{name})" in lane.system_prompt
    assert sum(r[3] for r in LANES.values()) <= spec.limits.alive == 250
    assert spec.comms.wake == ["mention", "thread", "channel"]
    assert spec.world.command[:6] == ["env", f"CHATBOT_REPO={REPO}", "CHATBOT_SEED_DIR=seed-v8",
                                      "CHATBOT_MAX_REVIEWERS=2",
                                      "CHATBOT_PR_OPEN_BUDGET_SCOPE=new_modules",
                                      "CHATBOT_BUCKET=hf://buckets/HuggingFaceSwarms/chatbot-agent-workspaces/v8"]


def test_differs_from_the_stored_v8_only_where_intended():
    """`flock launch` refuses a resume while these differ; `flock update` applies them. The
    fixture was rendered with the old overlay's placeholder bucket and namespace and no prompt
    files inlined (the files are PR #74's, unchanged)."""
    stored = SwarmSpec.model_validate_json(stored_spec("v8").read_text())
    lines = resume_diff(stored, read_spec(str(EXP / "spec.yaml")))
    assert [x.split(":")[0] for x in lines] == ["prompts changed", "bucket", "namespace",
                                                "world.command"]
    assert "base_urls" not in str(lines)  # same placeholder: the new URL comes from run.env


def test_the_prompts_name_the_repo_and_lanes_and_hide_the_eval():
    text = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in text
    for name in LANES:
        assert f"| {name} | #{name} |" in text
    for path in [EXP / "prompts/system.md", *sorted((EXP / "prompts/lanes").glob("*.md"))]:
        lower = path.read_text().lower()
        for word in (*HIDDEN, "chatbot-v", "spawn", "vote", "@all", "read_dms", " dm,", "dms promptly"):
            assert word not in lower, (path.name, word)


def test_the_seed_hides_the_eval():
    for name in ("README.md", "CONTRIBUTING.md", "AGENTS.md", "STYLE.md"):
        lower = (SEED / name).read_text().lower()
        for word in HIDDEN:
            assert word not in lower, (name, word)
    assert f"datasets/{REPO} chatbot-swarm" in (SEED / "CONTRIBUTING.md").read_text()
