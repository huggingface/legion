"""A Harbor spec through legion's commands: `legion launch` and `legion update` compile it (the
task pinned once, the tool bundle's sha recorded), a resume keeps the run's pin and folder paths
(a moved folder, `--ref`), and a running type's Trial inputs are fixed; links inside the shipped
folders are refused, and `legion init` warns of them and writes the compiled secrets."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from harbor.models.task.id import GitTaskId
from typer.testing import CliRunner

import legion.harbor
from legion import cli, rundir
from legion.harbor import env as env_world
from legion.harbor import sources
from legion.ops import SpecError, check_spec, read_spec

EXPERIMENTS = Path(__file__).parents[2] / "experiments"
FIXTURES = Path(__file__).parents[1] / "fixtures"

from ..conftest import wire
from .test_worker_host import RESOURCE, TOOL

runner = CliRunner()
SPEC = """
name: crew
model: moonshotai/Kimi-K2.5
task:
  repo: https://huggingface.co/datasets/harborframework/terminal-bench-2.1
  selector: regex-log
  revision: main
team_file: team.md
resources_dir: resources
resources: RESOURCES
tools_dir: tools
tools: [sample.*]
agents: {count: COUNT}
environment: {backend: docker}
workers: {agents_per_worker: 2}
limits: {run_timeout_seconds: 900}
"""


def experiment(root: Path, count=2, team="Work together.", resources="{fake: {mount: true}}"
               ) -> Path:
    for folder, name, text in [("resources", "fake", RESOURCE), ("tools", "sample", TOOL)]:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / f"{name}.py").write_text(text)
    (root / "team.md").write_text(team)
    (root / "spec.yaml").write_text(SPEC.replace("COUNT", str(count)).replace("RESOURCES",
                                                                             resources))
    return root


@pytest.fixture
def pins(monkeypatch):
    """The Hub's answer per revision; a test moves `main` by changing it."""
    pins = {"main": "a" * 40}

    async def resolve(settings):
        task = GitTaskId(git_url="https://example.com/tasks.git",
                         git_commit_id=pins[settings.revision], path="regex-log")
        return sources.ResolvedTask(task, "d" * 40, "terminal-bench-2.1@2.1.0")

    monkeypatch.setattr(sources, "resolve_task", resolve)
    return pins


def test_launch_resume_and_update_on_a_harbor_spec(tmp_path, monkeypatch, app, store, tick,
                                                    pins):
    wire(monkeypatch, app)
    monkeypatch.setenv("LEGION_URL", "http://controller.invalid")  # no local controller started
    exp = experiment(tmp_path / "exp")
    git = ["git", "-C", str(exp), "-c", "user.name=t", "-c", "user.email=t@t"]
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-qm", "exp"]):  # for --ref
        subprocess.run(git + args, check=True, capture_output=True)
    r = runner.invoke(cli.app, ["launch", str(exp), "--skip-preflight"])
    assert r.exit_code == 0, r.output
    tick()
    stored = store.get_swarm("crew")["spec"].types["agent"]
    assert stored.harness.value == "harbor" and len(store.list_agents("crew", "agent")) == 2
    h = stored.harbor
    assert h.task["task"]["git_commit_id"] == "a" * 40 and h.team == "Work together."
    assert len(h.tools_sha256) == 64 and h.config["tools_dir"] == str(exp / "tools")
    # the folders reach workers through the controller, as the swarm's tools version
    assert store.get_swarm("crew")["spec"].tools_version == h.folders
    assert store.get_tool_bundle("crew", h.folders) is not None

    pins["main"] = "b" * 40  # the branch moved: the run keeps its pin
    r = runner.invoke(cli.app, ["launch", str(exp), "--skip-preflight"])
    assert r.exit_code == 0 and "held: nothing" in r.output, r.output
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 0 and r.output.strip() == "no changes", r.output
    moved = tmp_path / "moved"  # the run folder copied elsewhere, or --ref's copy: the same run
    shutil.copytree(exp, moved)
    r = runner.invoke(cli.app, ["launch", str(moved), "--skip-preflight"])
    assert r.exit_code == 0 and "held: nothing" in r.output, r.output
    for args in ([str(moved)], [str(exp), "--ref", "HEAD"]):
        r = runner.invoke(cli.app, ["update", *args])
        assert r.exit_code == 0 and r.output.strip() == "no changes", r.output
    assert store.get_swarm("crew")["spec"].types["agent"].harbor.config == h.config

    (exp / "tools" / "sample.py").write_text(TOOL + "# v2\n")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "harbor (tools_sha256, folders): a running" in r.output, r.output
    (exp / "tools" / "sample.py").write_text(TOOL)
    experiment(exp, team="Work alone.")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "types.agent.harbor (team): a running Harbor type" in r.output
    experiment(exp, count=3)  # the roster may grow
    r = runner.invoke(cli.app, ["update", str(exp), "--replicas", "spec"])
    assert r.exit_code == 0 and "types.agent.replicas: 2 -> 3" in r.output, r.output
    assert store.get_swarm("crew")["spec"].types["agent"].harbor.task == h.task


def experiment_spec(monkeypatch, pins, name, rev, env=None):
    """An experiments/ folder's spec (or an absolute folder's, or a spec file in one) as `legion
    launch` reads it, the task pin faked."""
    pins[rev] = "a" * 40
    for key in ("RUN_NAME", "BUCKET_ID"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    path = EXPERIMENTS / name
    path = path / "spec.yaml" if path.is_dir() else path
    swarm = read_spec(str(path))
    assert check_spec(swarm, str(path))[1] == []  # no refusals, no warnings
    assert all(ts.harbor.task["task"]["git_commit_id"] == "a" * 40 for ts in swarm.types.values())
    return swarm


def test_a_slurm_podman_spec_with_board_rules_compiles(monkeypatch, pins):
    swarm = experiment_spec(monkeypatch, pins, FIXTURES / "slurm_podman",
                            "3e235dff6880252a587fa479c09bcb1e16edf2eb")
    assert (swarm.swarm, swarm.compute.kind, list(swarm.types)) == (
        "slurm-podman", "slurm", ["agent"])
    h = swarm.types["agent"].harbor
    assert h.config["environment"]["backend"] == "podman"
    assert h.resources["message_board"]["rules"]["max_len"] == 1200
    assert "transport" not in h.resources["message_board"]  # managed: the rule proxy's socket
    assert h.config["limits"]["trial_hours"] == 0.15


def test_the_pybot_spec_compiles(monkeypatch, pins):
    monkeypatch.delenv("PYBOT_TASK_REVISION", raising=False)  # the spec's default: the pinned commit
    env = {"BUCKET_ID": "ns/runs", "BOARD_URL": "https://board.example",
           "WORLD_URL": "https://world.example", "PYBOT_REPO": "ns/bot"}
    swarm = experiment_spec(monkeypatch, pins, "pybot",
                            "8b9329d60446fc9212ba32b24ee80353474ef1f9", env)
    assert (swarm.types["steward"].name_prefix, swarm.types["answers"].name_prefix) == (
        "steward-", None)


def test_the_antbench_spec_compiles(monkeypatch, pins):
    """The launcher compiles it; the world job's env launcher reads its block and roster."""
    env = {"BUCKET_ID": "ns/runs", "ANTBENCH_TASK_REPO": "ns/tasks",
           "ANTBENCH_TASK_REVISION": "c" * 40}
    swarm = experiment_spec(monkeypatch, pins, "antbench", "c" * 40, env)
    ts = swarm.types["agent"]
    assert (swarm.compute.kind, ts.replicas, ts.agents_per_worker) == ("hf-jobs", 10, 10)
    assert ts.harbor.config["environment"]["backend"] == "e2b"
    assert {"E2B_API_KEY", "ANTBENCH_ENV_SECRET"} <= set(ts.secrets)
    config, names = env_world.load(EXPERIMENTS / "antbench" / "spec.yaml", {})
    # the worker expands the prefix: HF Jobs sets it only inside the network group
    assert config.upstream == "http://${HF_NETWORK_GROUP_PREFIX}antbench:8000"
    assert (names[0], names[9], config.credential_env) == ("agent-001", "agent-010",
                                                           "ANT_TOKEN_{seat}")


def test_links_in_the_shipped_folders_and_init(tmp_path, pins):
    """A linked tools folder ships; a link inside it is refused at launch and by `legion init`
    (which would copy what it points to). Init's run.env names the board's default token,
    known once the spec is compiled."""
    exp = experiment(tmp_path / "exp",
                     resources="{fake: {mount: true}, message_board: {url: https://board.example}}")
    board = Path(legion.harbor.__file__).with_name("builtin") / "resources" / "message_board.py"
    shutil.copy(board, exp / "resources" / "message_board.py")
    shutil.move(exp / "tools", tmp_path / "shared")
    (exp / "tools").symlink_to(tmp_path / "shared")
    assert read_spec(str(exp / "spec.yaml")).types["agent"].harbor.folders
    (tmp_path / "token").write_text("s3cret")
    (tmp_path / "shared" / "creds").symlink_to(tmp_path / "token")
    with pytest.raises(SpecError, match="symlinks are not allowed"):
        read_spec(str(exp / "spec.yaml"))
    env = {"PATH": os.environ["PATH"], "HOME": str(tmp_path)}
    with pytest.raises(ValueError, match=f"tools/creds -> {tmp_path / 'token'}; copy"):
        rundir.init(tmp_path / "run", exp, env=env)
    (tmp_path / "shared" / "creds").unlink()
    rep = rundir.init(tmp_path / "run", exp, env=env)
    assert not [c for c in rep.checks if c.name == "links" or c.ok is False], rep.checks
    assert "\n#MSGBOARD_OPERATOR_TOKEN=\n" in (tmp_path / "run" / "run.env").read_text()


def test_the_mini_swarm_spec_compiles(monkeypatch, pins):
    monkeypatch.delenv("BOARD_SPACE", raising=False)
    with pytest.raises(SpecError, match="BOARD_SPACE"):
        read_spec(str(EXPERIMENTS / "mini-swarm" / "spec.yaml"))
    swarm = experiment_spec(monkeypatch, pins, "mini-swarm",
                            "d0e1fee0f1036a80d5400d67f7fad9143668ba94",
                            {"BOARD_SPACE": "mini-swarm-test"})
    assert (swarm.swarm, swarm.compute.kind, list(swarm.types)) == ("mini-swarm-1", "local", ["agent"])
    assert swarm.namespace == ""
    ts = swarm.types["agent"]
    assert (ts.replicas, ts.agents_per_worker) == (3, 3)  # one worker: it deploys the board
    assert ts.harbor.resources == {"message_board": {
        "space_id": "mini-swarm-test", "private": False}}
    assert ts.resolved_model.headers == {}
    assert swarm.naming == "consecutive-id" and ts.harbor.names() == ["agent-001", "agent-002", "agent-003"]
    assert ts.harbor.config["tools"] == ["message_board.*"]
    assert ts.harbor.config["environment"] == {"backend": "hf-sandbox", "flavor": "cpu-basic"}
    assert ts.secrets == ["HF_TOKEN"]  # the worker deploys the Space and starts the sandboxes
    # hf-jobs.yaml: the same run (task pin, roster, loop, limits) with the same model on a vLLM
    # job legion starts
    for key in ("MODEL", "FLAVOR", "MAX_MODEL_LEN"):
        monkeypatch.delenv(key, raising=False)
    hf = experiment_spec(monkeypatch, pins, "mini-swarm/hf-jobs.yaml",
                         "d0e1fee0f1036a80d5400d67f7fad9143668ba94")
    other = hf.types["agent"]
    assert (hf.swarm, other.model, other.harbor, other.loop, other.secrets) == (
        "mini-swarm-hf-jobs-1", ts.model, ts.harbor, ts.loop, ts.secrets)
    hj = hf.inference.hf_jobs
    assert (hj.flavor, hj.vllm_args, hf.inference.api_key_env) == (
        "h200", ["max_model_len=131072"], "HF_INFERENCE_TOKEN")
    assert hj.timeout == "3300s"  # startup_timeout 30m + run_timeout_seconds 600 + 15m


def test_the_escape_room_spec_compiles(monkeypatch, pins):
    swarm = experiment_spec(monkeypatch, pins, "escape-room",
                            "eeccfb09a986db1670b56673b22ebea4d473d2a8")
    assert (swarm.swarm, swarm.compute.kind, list(swarm.types)) == ("escape-room-1", "local", ["agent"])
    ts = swarm.types["agent"]
    assert (ts.replicas, ts.agents_per_worker) == (4, 4)  # one worker, one board
    assert (ts.loop.prompt, ts.loop.while_.max_rounds) == ("Work on the task.", 1)  # one start
    assert swarm.naming == "words" and ts.harbor.config["environment"]["backend"] == "docker"
    assert ts.harbor.team.startswith("Solve the escape-room puzzle")
    assert ts.harbor.config["tools"] == ["message_board.*"] and ts.secrets == []
    assert ts.resolved_model.inference.key_env == "HF_INFERENCE_TOKEN"


def test_serve_takes_the_run_dirs_environment(tmp_path, monkeypatch, pins):
    """`legion --run RUN serve SPEC` fills the spec from the run's run.env and starts the servers
    with its environment (the Hub faked)."""
    from legion.controller.hf_jobs_servers import HFJobsServers

    from ..fake_hub_jobs import FakeHubJobs

    pins["d0e1fee0f1036a80d5400d67f7fad9143668ba94"] = "a" * 40
    monkeypatch.delenv("RUN_NAME", raising=False)
    run = tmp_path / "run"
    run.mkdir()
    (run / "run.env").write_text("RUN_NAME=from-run-env\nHF_TOKEN=run-env-token\n"
                                "BOARD_SPACE=from-run-env\n")
    tokens = []

    def starter(env):
        tokens.append(env.get("HF_TOKEN"))
        return HFJobsServers(FakeHubJobs(), env.get("HF_TOKEN"))

    monkeypatch.setattr("legion.controller.hf_jobs_servers.hf_jobs_servers", starter)
    r = runner.invoke(cli.app, ["--run", str(run), "serve",
                                str(EXPERIMENTS / "mini-swarm" / "hf-jobs.yaml"), "--stop"])
    assert r.exit_code == 0, r.output
    assert "no running servers of from-run-env" in r.output and tokens == ["run-env-token"]
