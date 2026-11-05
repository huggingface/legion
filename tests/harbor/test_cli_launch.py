"""A Harbor spec through flock's commands: `flock launch` and `flock update` compile it (the
task pinned once, the tool bundle's sha recorded), a resume keeps the run's pin, and a running
type's Trial inputs are fixed; a local run drives it end to end (controller, a worker process
that unpacks the shipped tools and resources folders, HarborAdapter, FlockPi) on
test_adapter.py's fakes."""

import asyncio
import json
import shutil
import sys
from io import StringIO
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("harbor")

from harbor.models.task.id import GitTaskId

from flock import cli
from flock.controller import localrun
from flock.harbor import sources
from flock.ops import check_spec, read_spec

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
resources: {fake: {mount: true}}
tools_dir: tools
tools: [sample.*]
agents: {count: COUNT}
environment: {backend: docker}
workers: {agents_per_worker: 2}
limits: {run_timeout_seconds: 900}
"""


def experiment(root: Path, count=2, team="Work together.") -> Path:
    for folder, name, text in [("resources", "fake", RESOURCE), ("tools", "sample", TOOL)]:
        (root / folder).mkdir(parents=True, exist_ok=True)
        (root / folder / f"{name}.py").write_text(text)
    (root / "team.md").write_text(team)
    (root / "spec.yaml").write_text(SPEC.replace("COUNT", str(count)))
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

    experiment(exp, team="Work alone.")
    r = runner.invoke(cli.app, ["update", str(exp)])
    assert r.exit_code == 1 and "types.agent.harbor (team): a running Harbor type" in r.output
    experiment(exp, count=3)  # the roster may grow
    r = runner.invoke(cli.app, ["update", str(exp), "--replicas", "spec"])
    assert r.exit_code == 0 and "types.agent.replicas: 2 -> 3" in r.output, r.output
    assert store.get_swarm("crew")["spec"].types["agent"].harbor.task == h.task


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
