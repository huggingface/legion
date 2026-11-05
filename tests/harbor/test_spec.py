from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

from flock.harbor.config import Config
from flock.harbor.spec import Spec, compile_spec, substitute
from flock.models import SwarmSpec, load_spec_yaml, loop_refusals

EXAMPLES = Path(__file__).parent / "examples"
V8_FIXTURE = Path(__file__).parent / "fixtures" / "v8-harbor-spec.yaml"


def within(small, big):
    if isinstance(small, dict):
        return all(k in big and within(v, big[k]) for k, v in small.items())
    return small == big


@pytest.mark.parametrize(
    "example", [*sorted(EXAMPLES.glob("*.yaml")), V8_FIXTURE], ids=lambda p: p.name
)
def test_shepherd_configs_load_unchanged(example):
    shepherd, spec = Config.load(example), Spec.load(example, env={})
    assert within(shepherd.model_dump(), spec.model_dump())
    assert spec.loop is None and spec.agents.roster() == shepherd.agents.roster()


def test_v8_harbor_spec_loads():
    spec = Spec.load(V8_FIXTURE, env={})
    assert list(spec.roles()) == [
        "understanding", "answers", "conversation", "routing", "quality", "safety",
    ]
    assert "world.*" in spec.tools and set(spec.resources) == {"board", "world"}
    assert spec.compute.kind == "local" and spec.environment.backend == "docker"


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
  on: [comms.mention, board.thread]
  while: {until: 48h, max_rounds: 10000, agent_usd: 5}
  final: {prompt_file: prompts/final.md, reserve_usd: 20}
  resume_prompt: Your previous turn was interrupted. Continue.
"""


def v8_spec(tmp_path, text=V8, env=None):
    (tmp_path / "prompts" / "lanes").mkdir(parents=True, exist_ok=True)
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
    assert loop.prompt == "Continue your work. Round {round}, ${spend}." and not loop.prompt_file
    assert (loop.every, loop.while_.until, loop.final.prompt) == (
        "turn_end",
        "48h",
        "The run is ending.",
    )
    assert loop.on == ["board.mention", "board.thread"]  # comms.* is read as board.*
    # run.json holds this dump; it validates again as written
    dumped = spec.model_dump(mode="json")
    assert dumped["loop"]["while"]["max_rounds"] == 10000
    assert Spec.model_validate(dumped) == spec


@pytest.mark.parametrize(
    "edit, error",
    [
        (lambda d: d["limits"].update(run_timeout_seconds=3600), "longer than loop.while.until"),
        (lambda d: d["resources"]["board"].pop("url"), "set resources.board.url"),
        (lambda d: d.pop("pricing"), "budgets need pricing"),
        (
            lambda d: d["agents"]["roles"]["safety"]["ramp"].update(start=7),
            "ramp.start must be <= count",
        ),
        (lambda d: d["loop"].update(every="soon"), "bad duration"),
        (lambda d: d["environment"].update(backend="slurm"), "backend"),
        # review findings on the spec as first written
        (lambda d: d["limits"].update(budget_usd=10), r"reserve_usd \(20\) is more than"),
        (lambda d: d["limits"].update(trial_hours=24), "set limits.trial_hours below 24"),
        (
            lambda d: (d["loop"]["while"].pop("until"), d["limits"].pop("run_timeout_seconds")),
            "a loop without while.until needs limits.run_timeout_seconds",
        ),
        (lambda d: d["environment"].update(backend="docker") or d.update(compute="hf-jobs"),
         "compute hf-jobs cannot run containers"),
        (lambda d: d["environment"].update(backend="docker") or d.update(compute="slurm"),
         "compute slurm runs rootless"),
        (lambda d: d["loop"].update(on=["slack.mention"]), "expected board.<name>"),
    ],
)
def test_spec_checks(tmp_path, edit, error):
    data = load_spec_yaml(V8)
    edit(data)
    with pytest.raises(ValidationError, match=error):
        v8_spec(tmp_path, yaml.safe_dump(data))


def test_unset_variable_without_default(tmp_path):
    with pytest.raises(ValueError, match="unset variables without a default: ANSWERS"):
        v8_spec(tmp_path, V8.replace("${ANSWERS:-96}", "${ANSWERS}"), env={})


@pytest.mark.parametrize(
    "value, typed",
    [("20", 20), ("0.5", 0.5), ("true", True), ("0123", "0123"), ("yes", "yes"), ("on", "on"),
     ("null", "null"), ("1e3", 1000.0), ("x", "x")],
)
def test_placeholders_take_json_types(value, typed):
    out = substitute({"a": "${V}", "b": "n=${V}"}, {"V": value})
    assert out == {"a": typed, "b": f"n={value}"} and type(out["a"]) is type(typed)


COMPILE = """
name: v8
model: huggingface/Qwen/Qwen3.8-27B
inference: {max_working: 40, reasoning_effort: high}
task: {repo: https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor, selector: v8}
team_file: team.md
resources: {board: {url: https://board.example, channels: [general]}}
tools: [board.*]
agents:
  roles:
    answers:
      count: 3
      role_file: answers.md
      ramp: {start: 1, add: 1, every: 10m}
    safety:
      model: huggingface/other/Model
      workers: {agents_per_worker: 1, max_extra: 2}
      loop: {prompt: 'Safety round {round}.', every: 5m, on: [comms.mention], while: {until: 2h}}
environment: {backend: hf-sandbox, kwargs: {flavor: cpu-basic}}
workers: {agents_per_worker: 10, turn_timeout: 20m, worker: {sync_interval_s: 60}}
limits: {run_timeout_seconds: 180000, agent_max_turns: 150, budget_usd: 400, alive: 10}
pricing: [0.3, 1.2, 0.03]
spawn: {answers: {types: {safety: 2}}}
loop:
  prompt_file: round.md
  while: {until: 48h, agent_usd: 5}
  final: {prompt: The run is ending., reserve_usd: 20}
  resume_prompt: Continue.
"""
TASK = {"registry_commit": "d" * 40, "dataset": "v8@1", "task": {"git_url": "x", "path": "t"}}


def test_compile_gives_one_harbor_type_per_role(tmp_path):
    for name, text in [("team.md", "Team."), ("answers.md", "Answer."), ("round.md", "R{round}")]:
        (tmp_path / name).write_text(text)
    (tmp_path / "spec.yaml").write_text(COMPILE)
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    bundle = SimpleNamespace(manifest={"tools": {"board.post": {}}, "requires": ["board"]},
                             sha256="f" * 64)
    swarm = compile_spec(spec, TASK, bundle)

    assert (swarm.swarm, swarm.namespace, swarm.bucket, swarm.compute.kind) == (
        "v8", "", "", "local",
    )
    provider = swarm.providers["inference"]
    assert (provider.type, provider.base_url, provider.api_key_env, provider.max_working) == (
        "openai_compat", "https://router.huggingface.co/v1", "HF_TOKEN", 40,
    )
    assert provider.pricing == {"*": [0.3, 1.2, 0.03]}
    assert swarm.limits.model_dump() == {"alive": 10, "spawned_total": None, "budget_usd": 400}
    assert swarm.spawn["answers"].types == {"safety": 2}
    assert list(swarm.types) == ["answers", "safety"]

    answers, safety = swarm.types["answers"], swarm.types["safety"]
    assert answers.harness.value == safety.harness.value == "harbor"
    assert answers.model == "inference:huggingface/Qwen/Qwen3.8-27B"
    assert answers.harness_model == "huggingface/Qwen/Qwen3.8-27B"
    assert answers.resolved_model.pricing == [0.3, 1.2, 0.03]
    assert answers.replicas == 0  # the ramp raises them to max
    assert answers.replicas_ramp.model_dump(include={"start", "add", "every", "max"}) == {
        "start": 1, "add": 1, "every": "600s", "max": 3,
    }
    assert (answers.agents_per_worker, answers.turn_timeout, answers.worker.sync_interval_s) == (
        10, "20m", 60,
    )
    assert (answers.max_turns, answers.workers, answers.resume_prompt) == (150, None, "Continue.")
    assert answers.flavor == "cpu-basic" and answers.idle_grace == "120s"  # main's defaults
    loop = answers.loop
    assert (loop.prompt, loop.every, loop.while_.until, loop.while_.agent_usd) == (
        "R{round}", "turn_end", "48h", 5,
    )
    assert loop.final.prompt == "The run is ending." and loop.final.reserve_usd == 20

    h = answers.harbor
    assert h.task == TASK and h.tools == bundle.manifest and h.tools_sha256 == "f" * 64
    assert (h.team, h.role, h.role_text) == ("Team.", "answers", "Answer.")
    assert h.roster == {
        "answers-0000": "answers", "answers-0001": "answers", "answers-0002": "answers",
        "safety-0000": "safety",
    }
    assert h.resources == {"board": {"url": "https://board.example", "channels": ["general"]}}
    assert h.config == {
        "model": "huggingface/Qwen/Qwen3.8-27B",
        "inference": {
            "base_url": "https://router.huggingface.co/v1", "api_key_env": "HF_TOKEN",
            "api_format": "openai_chat_completions", "reasoning_effort": "high",
            "max_working": 40,
        },
        "task": {"repo": "https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor",
                 "dataset": None, "selector": "v8", "revision": "main"},
        "environment": {"backend": "hf-sandbox", "kwargs": {"flavor": "cpu-basic"}},
        "limits": {"agent_max_turns": 150, "agent_timeout_seconds": None, "trial_hours": 23},
        "tools_dir": str(spec.tools_dir), "tools": ["board.*"],
        "resources_dir": str(spec.resources_dir),
    }
    assert Spec.model_validate(h.config).model == h.config["model"]  # the worker rebuilds it

    assert safety.model == "inference:huggingface/other/Model"
    assert safety.harbor.config["model"] == "huggingface/other/Model"
    assert (safety.replicas, safety.replicas_ramp, safety.agents_per_worker) == (1, None, 1)
    assert safety.workers.model_dump() == {"initial": None, "max_extra": 2}
    assert (safety.harbor.role, safety.harbor.role_text) == ("safety", "")
    assert (safety.loop.prompt, safety.loop.every, safety.loop.on) == (
        "Safety round {round}.", "5m", ["board.mention"],
    )
    assert safety.loop.final is None and safety.resume_prompt.startswith("Your session")

    assert loop_refusals(swarm) == []
    assert SwarmSpec.model_validate_json(swarm.model_dump_json()) == swarm


def test_roles_sharing_the_run_loop_split_its_reserve(tmp_path):
    data = load_spec_yaml(COMPILE)
    data["agents"]["roles"]["safety"].pop("loop")
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    reserves = [ts.loop.final.reserve_usd for ts in swarm.types.values()]
    assert reserves == [10, 10]  # flock holds back their sum: the spec's 20
    harbor = swarm.types["answers"].harbor
    assert harbor.tools == {} and harbor.tools_sha256 is None  # no bundle: no tools


def test_a_shepherd_config_compiles():
    swarm = compile_spec(Spec.load(EXAMPLES / "pi-hf.yaml", env={}), TASK)
    (name, ts), = swarm.types.items()
    assert (swarm.swarm, name, ts.replicas, ts.loop) == ("pi-hf-team", "agent", 2, None)
    assert ts.harbor.role is None and ts.harbor.roster == {"agent-0000": None, "agent-0001": None}
    assert ts.harbor.team == (EXAMPLES.parent / "prompts" / "team.md").read_text()
    assert swarm.providers["inference"].pricing == {"*": [0.0, 0.0, 0.0]}
