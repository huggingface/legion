from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import flock.harbor
from flock.controller.jobs import HubJobsClient
from flock.controller.reconcile import reconcile
from flock.harbor.config import Config
from flock.harbor.spec import Spec, compile_spec, substitute
from flock.models import SwarmSpec, load_spec_yaml, loop_refusals

EXAMPLES = Path(__file__).parent / "examples"
SWARMCHAT_FIXTURE = Path(__file__).parent / "fixtures" / "swarmchat-harbor-spec.yaml"
RESOURCES = Path(flock.harbor.__file__).parent / "builtin" / "resources"


def within(small, big):
    if isinstance(small, dict):
        return all(k in big and within(v, big[k]) for k, v in small.items())
    return small == big


@pytest.mark.parametrize(
    "example", [*sorted(EXAMPLES.glob("*.yaml")), SWARMCHAT_FIXTURE], ids=lambda p: p.name
)
def test_shepherd_configs_load_unchanged(example):
    shepherd, spec = Config.load(example), Spec.load(example, env={})
    assert within(shepherd.model_dump(), spec.model_dump())
    assert spec.loop is None and spec.agents.roster() == shepherd.agents.roster()


def test_swarmchat_harbor_spec_loads():
    spec = Spec.load(SWARMCHAT_FIXTURE, env={})
    assert list(spec.roles()) == [
        "understanding", "answers", "conversation", "routing", "quality", "safety",
    ]
    assert "world.*" in spec.tools and set(spec.resources) == {"message_board", "world"}
    assert spec.compute.kind == "local" and spec.environment.backend == "docker"


V8 = """
name: v8
model: huggingface/${MODEL:-Qwen/Qwen3.8-27B}
task:
  repo: https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor
  selector: v8
team_file: prompts/system.md
resources:
  message_board: {url: '${BOARD_URL:-https://board.example}', channels: [general]}
tools: [message_board.*]
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
  flavor: cpu-basic
limits:
  run_timeout_seconds: 180000
  agent_max_turns: 150
  budget_usd: 400
pricing: [0.3, 1.2, 0.03]
loop:
  prompt_file: prompts/round.md
  every: turn_end
  on: [comms.mention, message_board.thread]
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
        "flavor": "cpu-basic",
    }
    assert (spec.limits.trial_hours, spec.limits.budget_usd) == (23, 400)
    loop = spec.loop
    assert loop.prompt == "Continue your work. Round {round}, ${spend}." and not loop.prompt_file
    assert (loop.every, loop.while_.until, loop.final.prompt) == (
        "turn_end",
        "48h",
        "The run is ending.",
    )
    assert loop.on == ["message_board.mention", "message_board.thread"]  # comms.* is read as message_board.*
    assert "transport" not in spec.resources["message_board"]
    ruled = v8_spec(tmp_path, V8.replace("[general]}", "[general], rules: {max_len: 2000}}"))
    assert ruled.resources["message_board"]["transport"] == "http"  # no mounts: no rule proxy
    # run.json holds this dump; it validates again as written
    dumped = spec.model_dump(mode="json")
    assert dumped["loop"]["while"]["max_rounds"] == 10000
    assert Spec.model_validate(dumped) == spec


@pytest.mark.parametrize(
    "edit, error",
    [
        (lambda d: d["limits"].update(run_timeout_seconds=3600), "longer than loop.while.until"),
        (lambda d: d["resources"]["message_board"].pop("url"), "set message_board.space_id"),
        (lambda d: d.pop("pricing"), "budgets need pricing"),
        (
            lambda d: d["agents"]["roles"]["safety"]["ramp"].update(start=7),
            "ramp.start must be <= count",
        ),
        (lambda d: d["loop"].update(every="soon"), "bad duration"),
        (lambda d: d["environment"].update(backend="slurm"), "backend"),
        # review findings on the spec as first written
        (lambda d: d["limits"].update(budget_usd=10), r"reserve_usd \(20\) is more than"),
        (lambda d: d["limits"].update(trial_hours=24), "hf-sandbox caps a sandbox at 24 h"),
        # 23.6 h plus the default turn_timeout of 30 min is over 24 h
        (lambda d: d["limits"].update(trial_hours=23.6), r"plus answers's turn_timeout \(30m"),
        (
            lambda d: (d["loop"]["while"].pop("until"), d["limits"].pop("run_timeout_seconds")),
            "a loop without while.until runs for limits.run_timeout_seconds",
        ),
        (lambda d: d["loop"]["while"].update(budget_usd=50),
         "would apply to each of the 2 roles sharing the run's loop"),
        (lambda d: d["agents"]["roles"]["safety"].update(model="huggingface/other/M"),
         "agents.roles.safety has its own model: budgets need its pricing"),
        (lambda d: d.update(environment={"backend": "docker"}) or d.update(compute="hf-jobs"),
         "compute hf-jobs cannot run containers"),
        (lambda d: d.update(environment={"backend": "docker"}) or d.update(compute="slurm"),
         "compute slurm runs rootless"),
        (lambda d: d["loop"].update(on=["slack.mention"]), "expected message_board.<name>"),
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
inference: {max_working: 40, reasoning_effort: high, api_key_env: HF_INFERENCE_TOKEN}
task: {repo: https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor, selector: v8}
team_file: team.md
resources: {message_board: {url: https://board.example, channels: [general]}}
tools: [message_board.*]
agents:
  roles:
    answers:
      count: 3
      role_file: answers.md
      ramp: {start: 1, add: 1, every: 10m}
    safety:
      model: huggingface/other/Model
      pricing: [1, 2, 0.1]
      workers: {agents_per_worker: 1, max_extra: 2}
      loop: {prompt: 'Safety round {round}.', every: 5m, on: [comms.mention], while: {until: 2h}}
environment: {backend: hf-sandbox, flavor: cpu-basic}
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
    (tmp_path / "spec.yaml").write_text(COMPILE + f"resources_dir: {RESOURCES}\n")
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    bundle = SimpleNamespace(manifest={"tools": {"message_board.post": {}}, "requires": ["message_board"]},
                             sha256="f" * 64)
    swarm = compile_spec(spec, TASK, bundle)

    assert (swarm.swarm, swarm.namespace, swarm.bucket, swarm.compute.kind) == (
        "v8", "", "", "local",
    )
    provider = swarm.providers["inference"]
    assert (provider.type, provider.base_url, provider.api_key_env, provider.max_working) == (
        "openai_compat", "https://router.huggingface.co/v1", "HF_INFERENCE_TOKEN", 40,
    )
    assert provider.pricing == {"*": [0.3, 1.2, 0.03], "huggingface/other/Model": [1, 2, 0.1]}
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
    # the board's token (its plugin's default token_env, as it has a url) and the sandboxes' token
    assert answers.secrets == safety.secrets == ["MSGBOARD_OPERATOR_TOKEN", "HF_TOKEN"]
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
    assert h.resources == {"message_board": {"url": "https://board.example", "channels": ["general"]}}
    assert h.config == {
        "model": "huggingface/Qwen/Qwen3.8-27B",
        "inference": {
            "base_url": "https://router.huggingface.co/v1",
            "api_key_env": "HF_INFERENCE_TOKEN",
            "api_format": "openai_chat_completions", "reasoning_effort": "high",
            "max_working": 40,
        },
        "task": {"repo": "https://huggingface.co/datasets/HuggingFaceSwarms/v8-harbor",
                 "dataset": None, "selector": "v8", "revision": "main"},
        "environment": {"backend": "hf-sandbox", "flavor": "cpu-basic"},
        "limits": {"agent_max_turns": 150, "agent_timeout_seconds": None, "trial_hours": 23},
        "tools_dir": str(spec.tools_dir), "tools": ["message_board.*"],
        "resources_dir": str(spec.resources_dir), "namespace": "",
    }
    assert Spec.model_validate(h.config).model == h.config["model"]  # the worker rebuilds it

    assert safety.model == "inference:huggingface/other/Model"
    assert safety.resolved_model.pricing == [1, 2, 0.1]
    assert safety.harbor.config["model"] == "huggingface/other/Model"
    assert (safety.replicas, safety.replicas_ramp, safety.agents_per_worker) == (1, None, 1)
    assert safety.workers.model_dump() == {"initial": None, "max_extra": 2}
    assert (safety.harbor.role, safety.harbor.role_text) == ("safety", "")
    assert (safety.loop.prompt, safety.loop.every, safety.loop.on) == (
        "Safety round {round}.", "5m", ["message_board.mention"],
    )
    assert safety.loop.final is None and safety.resume_prompt.startswith("Your session")

    assert loop_refusals(swarm) == []
    assert SwarmSpec.model_validate_json(swarm.model_dump_json()) == swarm


@pytest.mark.parametrize("edit, error", [
    (lambda d: d["inference"].pop("api_key_env"), "HF_TOKEN is the workers' Hub token"),
    (lambda d: d["inference"].pop("api_key_env") or d.update(environment={"backend": "podman"})
     or d.update(compute="slurm"), "HF_TOKEN is the workers' Hub token"),
    (lambda d: d["resources"]["message_board"].update(token_env="FLOCK_WORLD_ADMIN_TOKEN"),
     "FLOCK_WORLD_ADMIN_TOKEN is the world's admin secret"),
])
def test_compile_keeps_hub_and_admin_tokens_from_agents(tmp_path, edit, error):
    data = load_spec_yaml(COMPILE)
    edit(data)
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=error):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    data["environment"] = {"backend": "docker"}  # local docker: the operator's own token
    data["resources"]["message_board"].pop("token_env", None)
    data.pop("compute", None)
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)


def test_a_space_board_is_public_and_on_one_worker(tmp_path):
    """A worker deploys a space_id board (and rotates its operator token): one worker per run,
    and a private Space's gate would need the workers' Hub token in every container."""
    data = load_spec_yaml(COMPILE)
    data["resources"]["message_board"] = {"space_id": "ns/board", "channels": ["general"]}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")

    def compiled():
        (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
        return compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)

    with pytest.raises(ValueError, match="private: set it to false"):
        compiled()
    data["resources"]["message_board"]["private"] = False
    with pytest.raises(ValueError, match="more than one worker"):
        compiled()
    data["agents"]["roles"].pop("safety")
    data.pop("spawn")
    data["agents"]["roles"]["answers"].update(count=3, workers={"agents_per_worker": 3})
    data["agents"]["roles"]["answers"].pop("ramp", None)
    h = compiled().types["answers"]
    assert "HF_TOKEN" in h.secrets  # the worker deploys the Space with it


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


def test_a_loop_without_until_ends_at_the_run_timeout(tmp_path):
    data = load_spec_yaml(COMPILE)
    data["loop"]["while"].pop("until")
    data["limits"]["run_timeout_seconds"] = 9000.5  # over safety's own 2h
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert swarm.types["answers"].loop.while_.until == "9000500ms"
    assert swarm.types["safety"].loop.while_.until == "2h"  # its own


@pytest.mark.parametrize("path, where", [
    (lambda d: d["workers"], "workers"),
    (lambda d: d["agents"]["roles"]["safety"].setdefault("workers", {}),
     "agents.roles.safety.workers"),
])
def test_workers_env_refuses_secret_names(tmp_path, path, where):
    data = load_spec_yaml(COMPILE)
    path(data)["env"] = {"CHATBOT_REPO": "ns/chat", "OPENAI_API_KEY": "sk-x"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=f"{where}.env: OPENAI_API_KEY cannot be set here"):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)


def test_workers_env_reaches_the_hf_jobs_workers(tmp_path, monkeypatch, store, settings, clock):
    """A task's `[environment.env]` `${VAR}` is filled from the worker's own env (Harbor's
    resolve_env_vars): `workers.env` puts it there, in every type and every launched job."""
    data = load_spec_yaml(COMPILE)
    data.update(compute="hf-jobs", namespace="ns", bucket="hf://buckets/ns/sh")
    data["workers"]["env"] = {"CHATBOT_REPO": "${CHATBOT_REPO}"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    spec = Spec.load(tmp_path / "spec.yaml", env={"CHATBOT_REPO": "ns/chat"})
    swarm = compile_spec(spec, TASK)
    assert {name: ts.env for name, ts in swarm.types.items()} == {
        "answers": {"CHATBOT_REPO": "ns/chat"}, "safety": {"CHATBOT_REPO": "ns/chat"},
    }
    monkeypatch.setenv("FLOCK_WHEEL_URL", "hf://buckets/ns/flock/dist/x.whl")
    runs = []

    class Api:  # HfApi's run_job, recorded
        def run_job(self, **kw):
            runs.append(kw)
            return SimpleNamespace(id=f"job-{len(runs)}")

    jobs = HubJobsClient(token="t", image=None)
    jobs.api = Api()
    store.put_swarm(swarm, 0)
    reconcile(store, jobs, clock(), settings)
    assert {kw["labels"]["type"] for kw in runs} == {"answers", "safety"}
    assert all(kw["env"]["CHATBOT_REPO"] == "ns/chat" for kw in runs)


def test_a_shepherd_config_compiles():
    swarm = compile_spec(Spec.load(EXAMPLES / "pi-hf.yaml", env={}), TASK)
    (name, ts), = swarm.types.items()
    assert (swarm.swarm, name, ts.replicas, ts.loop) == ("pi-hf-team", "agent", 2, None)
    assert ts.harbor.role is None and ts.harbor.roster == {"agent-0000": None, "agent-0001": None}
    assert ts.secrets == []  # a managed board, docker: the worker reads no token
    assert ts.harbor.team == (EXAMPLES.parent / "prompts" / "team.md").read_text()
    assert swarm.providers["inference"].pricing == {"*": [0.0, 0.0, 0.0]}


def test_a_missing_resource_module_is_refused_at_compile(tmp_path):
    data = load_spec_yaml(COMPILE)
    data["resources"] = {"board": data["resources"]["message_board"]}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="no resource module .*board.py .*now message_board"):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
