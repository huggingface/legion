"""The v11 experiment: the interviewer hierarchy (v9) on v8's seed and comparison-based PR setup,
with coordinators ramped by the controller as v8's lanes."""
from __future__ import annotations

import re
from pathlib import Path

from flock.cli import load_spec
from flock.models import comms_tools

from .test_hierarchy_experiment import module

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/v11"
REPO = "HuggingFaceSwarms/SwarmChat-v11"
SETTINGS = {"CHATBOT_SEED_DIR": "seed-v8", "CHATBOT_MAX_REVIEWERS": "2",
            "CHATBOT_PR_OPEN_BUDGET_SCOPE": "new_modules"}


def run_env():
    """The README's run.env block, as a shell reads it (inline comments dropped)."""
    block = re.search(r"```sh\n\s*(RUN3_EXPERIMENT=v11.*?)```", (EXP / "README.md").read_text(), re.DOTALL)[1]
    return dict(re.sub(r"\s+#.*", "", line.strip()).split("=", 1) for line in block.strip().splitlines())


def test_types_caps_ramp_and_interviewers(monkeypatch):
    monkeypatch.delenv("CHATBOT_BOARD_URL", raising=False)
    spec = load_spec(str(EXP / "spec.yaml"))
    hierarchy = load_spec(str(ROOT / "experiments/hierarchy/spec.yaml"))
    assert set(spec.types) == {"coordinator", "worker", "interviewer"}
    skip = {"system_prompt": True, "tools": True, "replicas_ramp": True}
    for name in ("coordinator", "worker"):
        t = spec.types[name]
        assert t.model_dump(exclude=skip) == hierarchy.types[name].model_dump(exclude=skip)
        assert t.tools == [*hierarchy.types[name].tools, "world:verdicts"]
    ramp = spec.types["coordinator"].replicas_ramp  # v8's mechanism: the controller ramps
    assert (ramp.start, ramp.add, ramp.every, ramp.max) == (8, 6, "10m", 50)
    assert spec.types["worker"].replicas_ramp is None and spec.types["interviewer"].replicas_ramp is None
    assert not (EXP / "waves.yaml").exists() and not (EXP / "waves.py").exists()
    iv = spec.types["interviewer"]
    assert iv.ephemeral and iv.replicas == 0 and iv.harness == "llm"
    assert spec.models[iv.model].extra["chat_template_kwargs"] == {"thinking": False}
    assert {"world:run_bot", "world:record_verdict"} <= set(iv.tools)
    assert {"comms:post", "comms:dm", "comms:join_channel"} <= set(comms_tools(iv, spec.comms))
    assert "interviewer" not in spec.spawn
    assert spec.spawn["coordinator"].types == {"worker": 4, "interviewer": 2}
    assert spec.spawn["worker"].types == {"interviewer": 2}
    assert spec.limits.alive >= ramp.max + 200 + iv.workers.max_extra * iv.agents_per_worker
    assert spec.limits.spawned_total >= 400 + 150 * 48
    assert iv.workers.max_extra * iv.agents_per_worker <= 26
    assert spec.comms.url == "https://huggingfaceswarms-chatbot-v11-msgboard.hf.space/boards/chatbot-v11"


def test_prompts_carry_v8s_comparison_workflow_and_v9s_interviews():
    spec = load_spec(str(EXP / "spec.yaml"))
    system = (EXP / "prompts/system.md").read_text()
    assert f"`{REPO}` on Hugging Face" in system and f"datasets/{REPO}` into `chatbot-swarm`" in system
    assert "chatbot-v" not in system and "SwarmChat-v9" not in system
    for part in ("an understanding engine", "a memory store", "a knowledge base", "a response composer"):
        assert system.count(part) == 1, part  # the two Task sections merged without duplication
    assert "The real test: a strong interviewer" in system and "prefer general mechanisms" in system
    loop = system.split("## The work loop")[1].split("## Interviews")[0]
    assert "Compare with main with `tools/compare.py`" in loop  # the PR rule
    assert "Review others' PRs with your own comparison" in loop  # the review rule
    assert "no multi-turn conversation got worse" in loop and "Revert a merged change" in loop
    assert "review_full" in loop and "open_budget" in loop
    assert "## Interviews" in system and "flock spawn interviewer" in system
    assert "Treat a `script` verdict as a bug to fix" in system and "new key per interview" in system
    for name in ("coordinator", "worker"):
        prompt = spec.types[name].system_prompt
        assert "## Interviews" in prompt and REPO in prompt and "SwarmChat-v9" not in prompt
        assert "`tools/compare.py` comparison with main in every PR and every review" in prompt
    assert "record_verdict" in spec.types["interviewer"].system_prompt


def test_run_env_has_the_world_settings_and_seed_v8(tmp_path):
    env = run_env()
    assert {k: env[k] for k in SETTINGS} == SETTINGS
    assert env["CHATBOT_REPO"] == REPO and "MODEL_INTERVIEWER" in env and env["CHATBOT_RUN"] == "v11"
    world = (ROOT / "worlds/chatbot4/chatbot4.py").read_text()
    assert all(k in world for k in SETTINGS)  # the world reads them
    seed = ROOT / "worlds/chatbot4" / env["CHATBOT_SEED_DIR"]
    assert (seed / "tools/compare.py").is_file() and (seed / "STYLE.md").is_file()
    readme = (EXP / "README.md").read_text()
    assert f"--repo {REPO} \\\n     --seed worlds/chatbot4/seed-v8" in readme
    (tmp_path / "run.env").write_text("".join(f"{k}={v}\n" for k, v in env.items()))
    data = module(EXP, "render").render(env_files=[tmp_path / "run.env"])
    assert data["swarm"] == "v11" and data["comms"]["url"] == env["CHATBOT_BOARD_URL"]
    assert data["models"]["interviewer"]["model"] == f"vllm:{env['MODEL_INTERVIEWER']}"


def test_render_and_smoke_read_this_folder():
    data = module(EXP, "render").render()
    assert set(data["types"]) == {"coordinator", "worker", "interviewer"}
    smoke = module(EXP, "render").render("smoke")
    for name in ("coordinator", "worker"):
        assert smoke["types"][name]["loop"]["while"]["until"] == "30m", name
    assert smoke["types"]["coordinator"]["replicas_ramp"]["start"] == smoke["types"]["coordinator"]["replicas_ramp"]["max"]
