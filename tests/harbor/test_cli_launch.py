"""A Harbor spec through flock's commands: `flock launch` and `flock update` compile it (the
task pinned once, the tool bundle's sha recorded), a resume keeps the run's pin and folder paths
(a moved folder, `--ref`), and a running type's Trial inputs are fixed; links inside the shipped
folders are refused, and `flock init` warns of them and writes the compiled secrets; a local run drives it end to end (controller, a worker process
that unpacks the shipped tools and resources folders, HarborAdapter, FlockPi) on
test_adapter.py's fakes."""

import asyncio
import json
import os
import shutil
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("harbor")

from harbor.models.task.id import GitTaskId

import flock.harbor
from flock import cli, rundir
from flock.controller import localrun
from flock.harbor import sources
from flock.ops import SpecError, check_spec, read_spec

EXPERIMENTS = Path(__file__).parents[2] / "experiments"

from ..conftest import wire
from .test_worker_host import RESOURCE, TOOL

runner = CliRunner()
SPEC = """
name: crew
model: huggingface/moonshotai/Kimi-K2.5
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
    monkeypatch.setenv("FLOCK_URL", "http://controller.invalid")  # no local controller started
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
    """One of the first runs' specs as `flock launch` reads it, the task pin faked."""
    pins[rev] = "a" * 40
    for key in ("RUN_NAME", "BUCKET", "SLURM_PARTITION"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    path = EXPERIMENTS / name / "spec.yaml"
    swarm = read_spec(str(path))
    assert check_spec(swarm, str(path))[1] == []  # no refusals, no warnings
    assert all(ts.harbor.task["task"]["git_commit_id"] == "a" * 40 for ts in swarm.types.values())
    return swarm


def test_the_smoke_spec_compiles(monkeypatch, pins):
    swarm = experiment_spec(monkeypatch, pins, "harbor-smoke",
                            "3e235dff6880252a587fa479c09bcb1e16edf2eb")
    assert (swarm.swarm, swarm.compute.kind, list(swarm.types)) == ("hsmoke-1", "slurm", ["agent"])
    h = swarm.types["agent"].harbor
    assert h.config["environment"]["backend"] == "podman"
    assert h.resources["board"]["rules"]["max_len"] == 1200
    assert "transport" not in h.resources["board"]  # managed: the rule proxy's socket
    assert h.config["limits"]["trial_hours"] == 0.15


def test_the_v8_harbor_spec_compiles(monkeypatch, pins):
    env = {"TASK_REV": "c" * 40, "BOARD_URL": "https://board.example",
           "WORLD_URL": "https://world.example", "CHATBOT_REPO": "ns/chat"}
    swarm = experiment_spec(monkeypatch, pins, "v8-harbor", "c" * 40, env)
    assert (swarm.swarm, swarm.compute.kind, len(swarm.types)) == ("v8h-1", "hf-jobs", 6)
    h = swarm.types["answers"].harbor
    assert h.config["environment"]["kwargs"]["namespace"] == "HuggingFaceSwarms"
    assert h.resources["board"]["transport"] == h.resources["world"]["transport"] == "http"
    assert len(h.resources["board"]["channels"]) == 7  # the poller watches only these
    for ts in swarm.types.values():
        assert ts.env == {"CHATBOT_REPO": "ns/chat"} and ts.agents_per_worker == 1
        assert ts.secrets == ["MSGBOARD_OPERATOR_TOKEN", "CHATBOT_WORLD_TOKEN", "HF_TOKEN"]
        assert ts.loop.final.reserve_usd == 0.5 and ts.harbor.config["limits"]["trial_hours"] == 20


def test_links_in_the_shipped_folders_and_init(tmp_path, pins):
    """A linked tools folder ships; a link inside it is refused at launch and by `flock init`
    (which would copy what it points to). Init's run.env names the board's default token,
    known once the spec is compiled."""
    exp = experiment(tmp_path / "exp",
                     resources="{fake: {mount: true}, board: {url: https://board.example}}")
    board = Path(flock.harbor.__file__).with_name("plugins") / "resources" / "board.py"
    shutil.copy(board, exp / "resources" / "board.py")
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


def test_a_local_run_drives_harbor_trials(tmp_path, monkeypatch, pins):
    monkeypatch.setenv("HF_TOKEN", "test-inference-secret")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("FLUSH_INTERVAL", "0.2")
    spec_file = str(experiment(tmp_path / "exp") / "spec.yaml")
    spec = read_spec(spec_file)
    folders, _ = check_spec(spec, spec_file)
    shutil.rmtree(tmp_path / "exp")  # the worker unpacks the shipped folders, not these
    out = StringIO()
    result = asyncio.run(localrun.run_local(localrun.RunOptions(
        spec=spec, data_dir=tmp_path / ".flock", until=60, until_idle=True, prompt="go",
        refresh=0.2, reconcile_interval=0.2, out=out, tools=folders.data,
        worker_command=[sys.executable, str(Path(__file__).with_name("fake_worker.py"))])))
    assert result.reason == "idle" and result.exit_code == 0, out.getvalue()
    assert result.totals["turns_completed"] == 2, out.getvalue()
    for agent in ("agent-0000", "agent-0001"):
        events = [json.loads(x) for x in
                  (result.run_dir / "events" / f"{agent}.jsonl").read_text().splitlines()]
        assert "reply-" in json.dumps([e for e in events if e["type"] == "message"])
        assert not [e for e in events if e["type"] == "error"], events
