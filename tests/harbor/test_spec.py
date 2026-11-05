from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from flock.harbor.config import Config
from flock.harbor.spec import Spec

EXAMPLES = Path(__file__).parent / "examples"


def within(small, big):
    if isinstance(small, dict):
        return all(k in big and within(v, big[k]) for k, v in small.items())
    return small == big


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_shepherd_examples_load_unchanged(example):
    shepherd, spec = Config.load(example), Spec.load(example, env={})
    assert within(shepherd.model_dump(), spec.model_dump())
    assert spec.loop is None and spec.agents.roster() == shepherd.agents.roster()


V8 = """
name: v8
model: huggingface/${MODEL:-Qwen/Qwen3.8-27B}
task:
  repo: https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor
  selector: v8
team_file: prompts/system.md
resources:
  board: {url: '${BOARD_URL:-https://board.example}', channels: [general]}
tools: [board.*]
agents:
  roles:
    answers: &lane
      count: ${ANSWERS:-96}
      role_file: prompts/lanes/answers.md
      ramp: {start: 12, add: 12, every: 10m}
    safety:
      <<: *lane
      count: 6
      role_file: prompts/lanes/safety.md
      ramp: {start: 3, add: 3, every: 30m}
environment:
  backend: hf-sandbox
  kwargs: {flavor: cpu-basic}
limits:
  run_timeout_seconds: 180000
  agent_max_turns: 150
  budget_usd: 400
pricing: [0.3, 1.2, 0.03]
loop:
  prompt_file: prompts/round.md
  every: turn_end
  while: {until: 48h, max_rounds: 10000, agent_usd: 5}
  final: {prompt_file: prompts/final.md, reserve_usd: 20}
  resume_prompt: Your previous turn was interrupted. Continue.
"""


def v8_spec(tmp_path, text=V8, env=None):
    (tmp_path / "prompts").mkdir(exist_ok=True)
    (tmp_path / "prompts" / "round.md").write_text("Continue your work. Round {round}, ${spend}.")
    (tmp_path / "prompts" / "final.md").write_text("The run is ending.")
    path = tmp_path / "spec.yaml"
    path.write_text(text)
    return Spec.load(path, env={"ANSWERS": "24"} if env is None else env)


def test_v8_shaped_spec_loads(tmp_path):
    spec = v8_spec(tmp_path)
    assert spec.model == "huggingface/Qwen/Qwen3.8-27B"
    answers, safety = spec.agents.roles["answers"], spec.agents.roles["safety"]
    assert (answers.count, answers.ramp.every, safety.count, safety.ramp.start) == (24, 600, 6, 3)
    assert safety.role_file == tmp_path / "prompts" / "lanes" / "safety.md"
    assert spec.environment.model_dump() == {
        "backend": "hf-sandbox",
        "kwargs": {"flavor": "cpu-basic"},
    }
    assert (spec.limits.trial_hours, spec.limits.budget_usd) == (23, 400)
    loop = spec.loop
    assert (
        loop.prompt == "Continue your work. Round {round}, ${spend}." and loop.prompt_file is None
    )
    assert (loop.every, loop.while_.until, loop.final.prompt) == (
        0,
        48 * 3600,
        "The run is ending.",
    )
    # run.json holds this dump; it validates again as written
    dumped = spec.model_dump(mode="json")
    assert dumped["loop"]["while"]["max_rounds"] == 10000
    assert Spec.model_validate(dumped) == spec


@pytest.mark.parametrize(
    "edit, error",
    [
        (lambda d: d["limits"].update(run_timeout_seconds=3600), "longer than loop.while.until"),
        (lambda d: d.pop("loop"), "ramp needs loop"),
        (lambda d: d["resources"]["board"].pop("url"), "set resources.board.url"),
        (lambda d: d.pop("pricing"), "budgets need pricing"),
        (
            lambda d: d["agents"]["roles"]["safety"]["ramp"].update(start=7),
            "ramp.start must be <= count",
        ),
        (lambda d: d["loop"].update(prompt="both"), "not both"),
        (lambda d: d["loop"].update(every="soon"), "bad duration"),
        (lambda d: d["environment"].update(backend="slurm"), "backend"),
    ],
)
def test_spec_checks(tmp_path, edit, error):
    data = yaml.safe_load(V8)
    edit(data)
    with pytest.raises(ValidationError, match=error):
        v8_spec(tmp_path, yaml.safe_dump(data))


def test_unset_variable_without_default(tmp_path):
    with pytest.raises(ValueError, match="unset variables without a default: ANSWERS"):
        v8_spec(tmp_path, V8.replace("${ANSWERS:-96}", "${ANSWERS}"), env={})
