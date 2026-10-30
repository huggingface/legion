"""The v12 experiment: levels' three-level hierarchy (lead -> managers -> workers) with v8's
comparison-based PR setup (seed-v8, world settings, work loop). Leads ramp, the rest spawn."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from flock.cli import load_spec

from .test_hierarchy_experiment import module
from .test_levels_experiment import SUPERVISION
from .test_levels_experiment import test_three_levels_spawn_with_caps as _levels_spawn
from .test_v8_experiment import HIDDEN

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/v12"
REPO = "HuggingFaceSwarms/SwarmChat-v12"
BOARD = "https://huggingfaceswarms-chatbot-v12-msgboard.hf.space/boards/chatbot-v12"
WORLD_SETTINGS = {"CHATBOT_SEED_DIR": "seed-v8", "CHATBOT_MAX_REVIEWERS": "2",
                  "CHATBOT_PR_OPEN_BUDGET_SCOPE": "new_modules"}


@pytest.fixture(autouse=True)
def _spec_defaults(monkeypatch):  # the spec's ${VAR:-default}s, not the caller's run.env
    for var in ("CHATBOT_BOARD_URL", "CHATBOT_RUN"):
        monkeypatch.delenv(var, raising=False)


def rules(role):
    return (EXP / f"prompts/{role}.md").read_text().split("## Team rules")[1].split("\n", 1)[1]


def run_env():
    block = (EXP / "README.md").read_text().split("```sh\n   RUN3_EXPERIMENT")[1].split("```")[0]
    return dict(re.match(r"\s*(\w+)=(\S+)", "RUN3_EXPERIMENT" + line if i == 0 else line).groups()
                for i, line in enumerate(block.splitlines()) if "=" in line)


def test_types_spawn_policy_caps_and_lead_ramp():
    spec = load_spec(str(EXP / "spec.yaml"))
    levels = load_spec(str(ROOT / "experiments/levels/spec.yaml"))
    assert set(spec.types) == {"lead", "manager", "worker"}
    skip = {"system_prompt": True, "replicas_ramp": True}
    for name, t in spec.types.items():  # levels' agents, apart from the lead's ramp
        assert t.model_dump(exclude=skip) == levels.types[name].model_dump(exclude=skip), name
    assert spec.spawn == levels.spawn and spec.limits == levels.limits
    assert spec.spawn["lead"].types == {"manager": 5} and spec.spawn["lead"].depth == 2
    assert spec.spawn["manager"].types == {"worker": 5} and spec.spawn["manager"].depth == 1
    ramp = spec.types["lead"].replicas_ramp  # v8's mechanism: the controller ramps, no waves runner
    assert (ramp.start, ramp.add, ramp.every, ramp.max) == (4, 2, "10m", 20)
    assert spec.types["manager"].replicas_ramp is None and spec.types["worker"].replicas_ramp is None
    assert not (EXP / "waves.yaml").exists() and not (EXP / "waves.py").exists()
    slots = {n: (t.workers.initial + t.workers.max_extra) * t.agents_per_worker
             for n, t in spec.types.items()}
    assert slots["lead"] >= 20 and slots["manager"] >= 100 and slots["worker"] >= 500
    assert spec.limits.alive >= 620 and spec.limits.spawned_total > 600
    assert spec.comms.url == BOARD and "dm" in spec.comms.wake  # DMs allowed (v10)
    assert spec.comms.rules.max_channels == 100


def test_prompts_keep_levels_structure_and_take_v8s_comparisons():
    spec = load_spec(str(EXP / "spec.yaml"))
    system = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in system and f"datasets/{REPO}` into `chatbot-swarm`" in system
    assert "three levels" in system and "Whoever starts agents supervises them" in system
    for part in ("an understanding engine", "a memory store", "a knowledge base", "a response composer"):
        assert part in system, part
    assert "prefer general mechanisms" in system and "STYLE.md" in system and "NOT a therapist" in system
    assert system.count("## Task") == 1 and system.count("prefer general mechanisms") == 1
    assert "dm, read_dms" in system and "DMs are allowed" in system
    loop = system.split("## The work loop")[1].split("## Setting")[0]
    assert "`tools/compare.py`" in loop and "multi-turn conversation got worse" in loop
    pr = next(line for line in system.splitlines() if line.startswith("- Every PR description"))
    review = next(line for line in system.splitlines() if line.startswith("- Every review"))
    assert "comparison" in pr and "score line" in pr and "comparison" in review
    assert "compare.py" in system.split("before submitting")[0].rsplit("\n", 1)[1]
    lead, manager, worker = (spec.types[n].system_prompt for n in ("lead", "manager", "worker"))
    everything = lead + manager + worker
    for gone in ("## Interviews", "flock spawn interviewer", "verdict", "vote", "flock prompt",
                 "SwarmChat-v10", "chatbot-v10", "## Your lane", "| lane |", "40 channels",
                 "There are no DMs"):
        assert gone not in everything, gone
    for word in HIDDEN:
        assert word not in everything.lower(), word
    assert rules("lead") == rules("manager") == rules("worker")
    assert f"improving {REPO}." in rules("lead")
    for text in ("Establish the goal", "Build the shared plan", "Claim and execute", "Reassess",
                 "Verify and hand off", "say so in the team channel"):
        assert text in rules("lead"), text
    assert SUPERVISION + "#general. Remove an agent that is done with `flock remove`." in lead
    assert SUPERVISION + "your lead in the team channel." in manager
    assert "flock spawn manager --prompt" in lead and "flock spawn worker --prompt" in manager
    assert "flock spawn" not in worker and "You cannot start other agents" in worker


def test_run_env_sets_the_v8_world_settings_and_names():
    env = run_env()
    assert {k: env[k] for k in WORLD_SETTINGS} == WORLD_SETTINGS
    assert env["CHATBOT_REPO"] == REPO and env["CHATBOT_BOARD_URL"] == BOARD
    assert env["RUN3_EXPERIMENT"] == env["CHATBOT_RUN"] == "v12"
    assert (ROOT / "worlds/chatbot4" / env["CHATBOT_SEED_DIR"] / "tools/compare.py").is_file()
    readme = (EXP / "README.md").read_text()
    assert "publish_seed.py --seed worlds/chatbot4/seed-v8 \\\n     --repo HuggingFaceSwarms/SwarmChat-v12" in readme
    assert f"{BOARD}/v1/health" in readme


def test_render_with_run_env(tmp_path):
    env_file = tmp_path / "run.env"
    env_file.write_text("".join(f"{k}={v}\n" for k, v in run_env().items()))
    data = module(EXP, "render").render(env_files=[str(env_file)])
    assert data["swarm"] == "v12" and data["comms"]["url"] == BOARD
    assert data["types"]["lead"]["replicas_ramp"]["max"] == 20 and data["compute"]["kind"] == "slurm"
    smoke = module(EXP, "render").render("smoke", "slurm")
    assert smoke["types"]["lead"]["replicas_ramp"]["start"] == smoke["types"]["lead"]["replicas_ramp"]["max"] == 2
    assert all(t["loop"]["while"]["until"] == "30m" for t in smoke["types"].values())


async def test_three_levels_spawn_with_caps(client, app, store, tick, monkeypatch):
    def no_ramp(path):  # the in-process controller checks one lead's children; the ramp would start 4
        spec = load_spec(path)
        spec.types["lead"].replicas_ramp = None
        return spec

    monkeypatch.setattr("tests.test_levels_experiment.EXP", EXP)
    monkeypatch.setattr("tests.test_levels_experiment.load_spec", no_ramp)
    await _levels_spawn(client, app, store, tick)
