from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from pydantic import ValidationError

import legion.harbor
from legion.controller.jobs import HubJobsClient
from legion.controller.reconcile import reconcile
from legion.harbor.config import Config
from legion.harbor.spec import Spec, compile_spec, substitute
from legion.harbor.tools.runner import load_module
from legion.models import SwarmSpec, load_spec_yaml, loop_refusals

EXAMPLES = Path(__file__).parent / "examples"
RESOURCES = Path(legion.harbor.__file__).parent / "builtin" / "resources"


def within(small, big):
    if isinstance(small, dict):
        return all(k in big and within(v, big[k]) for k, v in small.items())
    return small == big


@pytest.mark.parametrize("example", sorted(EXAMPLES.glob("*.yaml")), ids=lambda p: p.name)
def test_upstream_configs_load_unchanged(example):
    upstream, spec = Config.load(example), Spec.load(example, env={})
    assert within(upstream.model_dump(), spec.model_dump())
    assert spec.loop is None and spec.agents.roster() == upstream.agents.roster()


LANES = """
name: lanes
model: ${MODEL:-Qwen/Qwen3.8-27B}
task:
  repo: https://huggingface.co/datasets/a/b
  selector: lanes
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
  on: [message_board.mention, message_board.thread]
  while: {until: 48h, max_rounds: 10000, agent_usd: 5}
  final: {prompt_file: prompts/final.md, reserve_usd: 20}
  resume_prompt: Your previous turn was interrupted. Continue.
"""


def six_lane_spec(tmp_path, text=LANES, env=None):
    (tmp_path / "prompts" / "lanes").mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompts" / "round.md").write_text("Continue your work. Round {round}, ${spend}.")
    (tmp_path / "prompts" / "final.md").write_text("The run is ending.")
    path = tmp_path / "spec.yaml"
    path.write_text(text)
    return Spec.load(path, env={"ANSWERS": "24"} if env is None else env)


def test_six_lane_spec_loads(tmp_path):
    spec = six_lane_spec(tmp_path)
    assert spec.model == "Qwen/Qwen3.8-27B"
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
    assert loop.on == ["message_board.mention", "message_board.thread"]
    assert "transport" not in spec.resources["message_board"]
    ruled = six_lane_spec(tmp_path, LANES.replace("[general]}", "[general], rules: {max_len: 2000}}"))
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
        (lambda d: d["agents"]["roles"]["safety"].update(model="other/M"),
         "agents.roles.safety has its own model: budgets need its pricing"),
        (lambda d: d.update(environment={"backend": "docker"}) or d.update(compute="hf-jobs"),
         "compute hf-jobs cannot run containers"),
        (lambda d: d.update(environment={"backend": "docker"}) or d.update(compute="slurm"),
         "compute slurm runs rootless"),
        (lambda d: d["loop"].update(on=["slack.mention"]), "expected message_board.<name>"),
    ],
)
def test_spec_checks(tmp_path, edit, error):
    data = load_spec_yaml(LANES)
    edit(data)
    with pytest.raises(ValidationError, match=error):
        six_lane_spec(tmp_path, yaml.safe_dump(data))


def test_unset_variable_without_default(tmp_path):
    with pytest.raises(ValueError, match="unset variables without a default: ANSWERS"):
        six_lane_spec(tmp_path, LANES.replace("${ANSWERS:-96}", "${ANSWERS}"), env={})


@pytest.mark.parametrize(
    "value, typed",
    [("20", 20), ("0.5", 0.5), ("true", True), ("0123", "0123"), ("yes", "yes"), ("on", "on"),
     ("null", "null"), ("1e3", 1000.0), ("x", "x")],
)
def test_placeholders_take_json_types(value, typed):
    out = substitute({"a": "${V}", "b": "n=${V}"}, {"V": value})
    assert out == {"a": typed, "b": f"n={value}"} and type(out["a"]) is type(typed)


COMPILE = """
name: lanes
model: Qwen/Qwen3.8-27B
inference:
  hf_router: {provider: deepinfra}
  max_working: 40
  reasoning_effort: high
  api_key_env: HF_INFERENCE_TOKEN
task: {repo: https://huggingface.co/datasets/a/b, selector: lanes}
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
      model: other/Model
      inference: {urls: ['https://vllm.example/v1'], api_key_env: VLLM_KEY}
      pricing: [1, 2, 0.1]
      workers: {agents_per_worker: 1, max_extra: 2}
      loop: {prompt: 'Safety round {round}.', every: 5m, on: [message_board.mention], while: {until: 2h}}
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
TASK = {"registry_commit": "d" * 40, "dataset": "lanes@1", "task": {"git_url": "x", "path": "t"}}


def test_compile_gives_one_harbor_type_per_role(tmp_path):
    for name, text in [("team.md", "Team."), ("answers.md", "Answer."), ("round.md", "R{round}")]:
        (tmp_path / name).write_text(text)
    (tmp_path / "spec.yaml").write_text(COMPILE + f"resources_dir: {RESOURCES}\n")
    spec = Spec.load(tmp_path / "spec.yaml", env={})
    bundle = SimpleNamespace(manifest={"tools": {"message_board.post": {}}, "requires": ["message_board"]},
                             sha256="f" * 64)
    swarm = compile_spec(spec, TASK, bundle)

    assert (swarm.swarm, swarm.namespace, swarm.bucket_id, swarm.compute.kind) == (
        "lanes", "", "", "local",
    )
    blocks = swarm.inference_blocks()  # the run's, and the role's own
    assert blocks["inference"].model_dump(exclude_defaults=True) == {
        "hf_router": {"provider": "deepinfra"}, "api_key_env": "HF_INFERENCE_TOKEN",
        "max_working": 40, "reasoning_effort": "high"}
    assert blocks["types.safety.inference"].model_dump(exclude_defaults=True) == {
        "urls": ["https://vllm.example/v1"], "api_key_env": "VLLM_KEY"}
    assert swarm.limits.model_dump() == {"alive": 10, "spawned_total": None, "budget_usd": 400}
    assert swarm.spawn["answers"].types == {"safety": 2}
    assert list(swarm.types) == ["answers", "safety"]

    answers, safety = swarm.types["answers"], swarm.types["safety"]
    assert answers.harness.value == safety.harness.value == "harbor"
    assert answers.model == answers.harness_model == "Qwen/Qwen3.8-27B"
    assert answers.resolved_model.request_model == "Qwen/Qwen3.8-27B:deepinfra"
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
    assert h.roster == {} and swarm.naming == "words"  # the controller names them, and sends
    # each assign the roster: the run's names and their roles (reconcile.harbor_roster)
    assert h.resources == {"message_board": {"url": "https://board.example", "channels": ["general"]}}
    assert h.config == {  # inference is the type's (resolved_model): `legion update` may move it
        "model": "Qwen/Qwen3.8-27B",
        "task": {"repo": "https://huggingface.co/datasets/a/b",
                 "dataset": None, "selector": "lanes", "revision": "main"},
        "environment": {"backend": "hf-sandbox", "flavor": "cpu-basic"},
        "limits": {"agent_max_turns": 150, "agent_timeout_seconds": None, "trial_hours": 23},
        "tools_dir": str(spec.tools_dir), "tools": ["message_board.*"],
        "resources_dir": str(spec.resources_dir), "namespace": "",
    }
    assert Spec.model_validate(h.config).model == h.config["model"]  # the worker rebuilds it

    assert safety.model == safety.harbor.config["model"] == "other/Model"
    assert safety.resolved_model.pricing == [1, 2, 0.1]
    assert safety.resolved_model.request_model == "other/Model"
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
    (lambda d: d["resources"]["message_board"].update(token_env="LEGION_WORLD_ADMIN_TOKEN"),
     "LEGION_WORLD_ADMIN_TOKEN is the world's admin secret"),
    (lambda d: d["agents"]["roles"]["safety"].update(inference={"hf_router": {}}),
     r"agents.roles.safety.inference.api_key_env: HF_TOKEN is the workers' Hub token"),
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


def test_only_the_blocks_the_roles_run_on_are_checked(tmp_path):
    """The run's block (the router, HF_TOKEN by default) is unused when every role has its own:
    no agent gets its key, so it is not refused."""
    data = load_spec_yaml(COMPILE)
    data["inference"].pop("api_key_env")
    data["agents"]["roles"]["answers"]["inference"] = {"urls": ["https://vllm.example/v1"]}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert set(swarm.inference_blocks()) == {"types.answers.inference", "types.safety.inference"}


@pytest.mark.parametrize(("edit", "error"), [
    (lambda d: None, "cannot reach"),  # hf-sandbox
    (lambda d: d.update(environment={"backend": "hf-sandbox"}, compute="hf-jobs",
                        namespace="ns", bucket_id="ns/b"), "cannot reach"),
    (lambda d: d.update(environment={"backend": "podman"}, compute="slurm", bucket_id="ns/b"),
     "each worker's own node"),
])
def test_servers_on_this_machine_need_agents_on_it(tmp_path, edit, error):
    """`localhost` is the machine legion runs on: refused where the agents run elsewhere."""
    data = load_spec_yaml(COMPILE)
    data["inference"] = {"urls": ["http://localhost:8000/v1"]}
    edit(data)
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=f"inference.urls: http://localhost:8000/v1 .*{error}"):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    data.update(environment={"backend": "podman"}, compute="local")  # agents on this machine
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert swarm.inference.urls == ["http://localhost:8000/v1"]


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
    assert reserves == [10, 10]  # legion holds back their sum: the spec's 20
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
    path(data)["env"] = {"PYBOT_REPO": "ns/chat", "OPENAI_API_KEY": "sk-x"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=f"{where}.env: OPENAI_API_KEY cannot be set here"):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)


def test_workers_env_reaches_the_hf_jobs_workers(tmp_path, monkeypatch, store, settings, clock):
    """A task's `[environment.env]` `${VAR}` is filled from the worker's own env (Harbor's
    resolve_env_vars): `workers.env` puts it there, in every type and every launched job."""
    data = load_spec_yaml(COMPILE)
    data.update(compute="hf-jobs", namespace="ns", bucket_id="ns/sh")
    data["workers"]["env"] = {"PYBOT_REPO": "${PYBOT_REPO}"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    spec = Spec.load(tmp_path / "spec.yaml", env={"PYBOT_REPO": "ns/chat"})
    swarm = compile_spec(spec, TASK)
    assert {name: ts.env for name, ts in swarm.types.items()} == {
        "answers": {"PYBOT_REPO": "ns/chat"}, "safety": {"PYBOT_REPO": "ns/chat"},
    }
    monkeypatch.setenv("LEGION_WHEEL_URL", "hf://buckets/ns/legion/dist/x.whl")
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
    assert all(kw["env"]["PYBOT_REPO"] == "ns/chat" for kw in runs)


def test_an_upstream_config_compiles():
    swarm = compile_spec(Spec.load(EXAMPLES / "pi-hf.yaml", env={}), TASK)
    (name, ts), = swarm.types.items()
    assert (swarm.swarm, name, ts.replicas, ts.loop) == ("pi-hf-team", "agent", 2, None)
    assert ts.harbor.role is None and ts.harbor.roster == {} and swarm.naming == "words"
    assert ts.secrets == []  # a managed board, docker: the worker reads no token
    assert ts.harbor.team == (EXAMPLES.parent / "prompts" / "team.md").read_text()
    assert ts.pricing is None and ts.resolved_model.pricing == [0.0, 0.0, 0.0]


def test_a_missing_resource_module_is_refused_at_compile(tmp_path):
    data = load_spec_yaml(COMPILE)
    data["resources"] = {"board": data["resources"]["message_board"]}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="no resource module .*board.py"):
        compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)


def test_harbor_agents_get_legion_names_and_their_roles_reach_each_assign(
        tmp_path, monkeypatch, store, settings, clock):
    """Words names (`<adjective>-<noun>`, reserved on the controller), whatever the roles. The
    first pass creates every type's agents before any assign, and each assign carries the
    run's roster (names -> roles), a ramp's names up to its max included; a raised ramp's
    agents take the names the rosters listed."""
    import json

    from legion.controller.naming import HANDLE_RE

    data = load_spec_yaml(COMPILE)
    data.update(compute="hf-jobs", namespace="ns", bucket_id="ns/sh")
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    monkeypatch.setenv("LEGION_WHEEL_URL", "hf://buckets/ns/legion/dist/x.whl")
    jobs = HubJobsClient(token="t", image=None)
    jobs.api = SimpleNamespace(run_job=lambda **kw: SimpleNamespace(id="job"))
    store.put_swarm(swarm, 0)
    reconcile(store, jobs, clock(), settings)
    agents = {a["name"]: a["type"] for a in store.list_agents(swarm.swarm)}
    assert sorted(agents.values()) == ["answers", "safety"]  # the ramp's start: 1 answers
    assigns = [json.loads(c["body"]) for w in store.list_workers(swarm.swarm)
               for c in store.commands_for(w["id"]) if '"assign"' in c["body"]]
    assert sorted(a["agent"] for a in assigns) == sorted(agents)
    (roster,) = {json.dumps(a["spec"]["harbor"]["roster"]) for a in assigns}  # one, for all
    roster = json.loads(roster)
    assert sorted(roster.values()) == ["answers"] * 3 + ["safety"]  # the ramp's max listed
    assert agents.items() <= roster.items()
    assert all(HANDLE_RE.match(n) and len(n) <= 40 for n in roster)

    raised = swarm.model_copy(deep=True)
    raised.types["answers"].replicas_ramp = None
    raised.types["answers"].replicas = 3
    store.put_swarm(raised, 1)
    reconcile(store, jobs, clock(), settings)
    assert {a["name"]: a["type"] for a in store.list_agents(swarm.swarm)} == roster


CONSECUTIVE = """name: crew
model: m
task: {repo: https://huggingface.co/datasets/a/b, selector: lanes}
naming: consecutive-id
agents:
  roles:
    explorer: {count: 2}
    reviewer: {count: REVIEWERS}
"""


def consecutive_swarm(tmp_path, reviewers=1, ramp=None):
    text = CONSECUTIVE.replace("REVIEWERS", str(reviewers))
    if ramp:
        text = text.replace(f"{{count: {reviewers}}}", f"{{count: {reviewers}, ramp: {ramp}}}")
    (tmp_path / "spec.yaml").write_text(text)
    return compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)


def test_naming_consecutive_id_gives_consecutive_names(tmp_path):
    """`naming: consecutive-id`: the names the upstream launcher gives the same config (its roster()),
    numbered across the roles in order."""
    swarm = consecutive_swarm(tmp_path)
    want = Config.model_validate(
        {k: v for k, v in yaml.safe_load((tmp_path / "spec.yaml").read_text()).items()
         if k != "naming"}).agents.roster()
    assert swarm.naming == "consecutive-id"
    assert {a: ts.harbor.role for ts in swarm.types.values() for a in ts.harbor.names()} == (
        want) == {"agent-001": "explorer", "agent-002": "explorer", "agent-003": "reviewer"}


def test_a_role_name_prefix_reaches_its_type_and_needs_words_naming(tmp_path):
    text = CONSECUTIVE.replace("REVIEWERS", "1").replace(
        "{count: 1}", '{count: 1, name_prefix: "rev-"}')
    (tmp_path / "spec.yaml").write_text(text.replace("naming: consecutive-id\n", ""))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert (swarm.types["reviewer"].name_prefix, swarm.types["explorer"].name_prefix) == ("rev-", None)
    (tmp_path / "spec.yaml").write_text(text)
    with pytest.raises(ValueError, match="agents.roles.reviewer.name_prefix needs naming: words"):
        Spec.load(tmp_path / "spec.yaml", env={})
    (tmp_path / "spec.yaml").write_text(text.replace('"rev-"', '"Rev"'))
    with pytest.raises(ValueError, match="lowercase letters and hyphens"):
        Spec.load(tmp_path / "spec.yaml", env={})


def test_naming_consecutive_id_updates_never_rename_running_agents(tmp_path):
    from legion.ops import SpecError, keep_pins

    old = consecutive_swarm(tmp_path)
    grown = consecutive_swarm(tmp_path, reviewers=3)  # the last role may grow
    keep_pins(old, grown)
    assert grown.types["reviewer"].harbor.names() == ["agent-003", "agent-004", "agent-005"]
    data = yaml.safe_load((tmp_path / "spec.yaml").read_text())
    data["agents"]["roles"]["explorer"]["count"] = 3
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(SpecError, match="a count change before role reviewer renumbers"):
        keep_pins(old, compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK))
    words = old.model_copy(update={"naming": "words"})
    with pytest.raises(SpecError, match="names never change in a run"):
        keep_pins(words, consecutive_swarm(tmp_path))


def test_one_task_per_agent_refuses_a_resize_in_a_run(tmp_path):
    """A type whose agents each run their own task (sources.pin_tasks): an update keeps the
    pinned map, so a count that changes the agents is refused, never an agent without a task."""
    from legion.ops import SpecError, keep_pins

    def swarm(count):
        (tmp_path / "spec.yaml").write_text(
            "name: crew\nmodel: m\nnaming: consecutive-id\ntask:\n"
            "  repo: https://huggingface.co/datasets/a/b\n  selector: [team-s01, team-s02]\n"
            f"agents:\n  count: {count}\n")
        tasks = {f"agent-{k:03d}": {**TASK["task"], "path": f"tasks/team-s{k:02d}"}
                 for k in range(1, count + 1)}
        return compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), {**TASK, "tasks": tasks})

    old, same = swarm(2), swarm(2)
    keep_pins(old, same)
    assert same.types["agent"].harbor.task["tasks"] == old.types["agent"].harbor.task["tasks"]
    for count in (3, 1):
        with pytest.raises(SpecError, match="one task per agent can't be resized in a run"):
            keep_pins(old, swarm(count))


def test_naming_consecutive_id_a_raised_ramp_gets_consecutive_names(tmp_path, store, clock):
    """The last role's ramp raised (count 1 -> 3): its new agents get agent-NNN names, never
    `<type>-NNNN` (the ramp is compared too, not only the replicas, which a ramp keeps at 0)."""
    from legion.controller.naming import replica_names
    from legion.ops import keep_pins

    old = consecutive_swarm(tmp_path, ramp="{start: 1, add: 1, every: 10m}")
    raised = consecutive_swarm(tmp_path, reviewers=3, ramp="{start: 1, add: 1, every: 10m}")
    assert raised.types["reviewer"].replicas == old.types["reviewer"].replicas == 0
    keep_pins(old, raised)
    store.put_swarm(raised, 0)
    assert replica_names(store, raised, "reviewer", 3, clock().timestamp()) == [
        "agent-003", "agent-004", "agent-005"]
    # past the roster (live replicas above the count): the next free agent-NNN names
    assert replica_names(store, raised, "explorer", 3, clock().timestamp()) == [
        "agent-001", "agent-002", "agent-006"]


async def test_the_controller_refuses_a_naming_change_in_a_running_swarm(tmp_path, client, store,
                                                                          settings, clock):
    swarm = consecutive_swarm(tmp_path)  # naming: consecutive-id
    assert (await client.put("/swarms/crew", json=swarm.model_dump(mode="json"))).status_code == 200
    reconcile(store, SimpleNamespace(), clock(), settings)  # local: its agents exist now
    assert store.list_agents("crew")
    words = swarm.model_copy(update={"naming": "words"})
    r = await client.put("/swarms/crew", json=words.model_dump(mode="json"))
    assert r.status_code == 400 and "names never change in a run" in r.text, r.text


def test_e2b_workers_get_the_e2b_key_not_the_hub_token(tmp_path):
    data = load_spec_yaml(COMPILE)
    data["environment"] = {"backend": "e2b"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert swarm.types["answers"].secrets == ["MSGBOARD_OPERATOR_TOKEN", "E2B_API_KEY"]
    data.update(compute="hf-jobs", namespace="ns", bucket_id="ns/b")  # its bucket
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    swarm = compile_spec(Spec.load(tmp_path / "spec.yaml", env={}), TASK)
    assert swarm.types["answers"].secrets == [
        "MSGBOARD_OPERATOR_TOKEN", "HF_TOKEN", "E2B_API_KEY"]
    data.update(compute="slurm", bucket_id="ns/b")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="compute slurm keeps E2B_API_KEY"):
        Spec.load(tmp_path / "spec.yaml", env={})


def test_a_gateway_runs_on_e2b_hf_jobs_workers(tmp_path):
    data = load_spec_yaml(COMPILE)
    data.update(environment={"backend": "e2b"}, compute="hf-jobs", namespace="ns", bucket_id="ns/b")
    data["resources"]["world"] = {"url": "https://world.example", "transport": "gateway"}
    for name in ("team.md", "answers.md", "round.md"):
        (tmp_path / name).write_text("x")
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="world transport gateway needs resources.gateway"):
        Spec.load(tmp_path / "spec.yaml", env={})
    data["resources"]["gateway"] = {}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    Spec.load(tmp_path / "spec.yaml", env={})
    env = {"name": "ants", "upstream": "http://u", "secret_env": "ANT_ENV_SECRET",
           "credential_env": "ANT_TOKEN_{seat}", "headers": {"Authorization": "{credential}"},
           "tools": {"observe": {"method": "GET", "path": "/v1/observe"}}}
    data["resources"]["world"] = {"transport": "gateway", "env": env}
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="world.env needs naming: consecutive-id"):
        Spec.load(tmp_path / "spec.yaml", env={})
    data["naming"] = "consecutive-id"
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    Spec.load(tmp_path / "spec.yaml", env={})
    data["compute"] = "local"  # sandboxes cannot reach this machine
    (tmp_path / "spec.yaml").write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="resources.gateway needs backend e2b on compute hf-jobs"):
        Spec.load(tmp_path / "spec.yaml", env={})
    options = load_module(RESOURCES / "world.py").Options
    with pytest.raises(ValueError, match="world needs one of url and env"):
        options(url="https://world.example", env=env, transport="gateway")
    with pytest.raises(ValueError, match="world.env needs transport: gateway"):
        options(env=env)
