"""A Harbor spec through flock's commands: `flock launch` and `flock update` compile it (the
task pinned once, the tool bundle's sha recorded), a resume keeps the run's pin, and a running
type's Trial inputs are fixed."""

from pathlib import Path

import pytest
from typer.testing import CliRunner

pytest.importorskip("harbor")

from harbor.models.task.id import GitTaskId

from flock import cli
from flock.harbor import sources

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

