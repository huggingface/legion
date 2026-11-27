import json
import os
import shutil
import sys
from pathlib import Path
from subprocess import CompletedProcess, check_output

import httpx
import pytest
from typer.testing import CliRunner

import legion
from legion import client, ops, rundir
from legion.cli import app
from legion.harbor.results import write_json
from legion.sweep import Attempt, make_plan


@pytest.fixture
def matrix(tmp_path):
    experiment = tmp_path / "experiment"
    experiment.mkdir()
    (experiment / "spec.yaml").write_text("""\
name: one-run
compute: slurm
bucket_id: example/sweep
task:
  repo: https://huggingface.co/datasets/example/tasks
  selector: original
  revision: aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
model: test-model
agents:
  count: 1
inference:
  urls: [http://model.example/v1]
loop:
  prompt: solve
  while:
    until: 30s
    max_rounds: 1
""")
    path = experiment / "sweep.yaml"
    path.write_text("""\
run_timeout: 10s
max_runs: 2
matrix:
  trial:
    a:
      task:
        selector: task-a
    b:
      task:
        selector: task-b
""")
    return path


def _record(root, matrix, phase):
    plan = make_plan(matrix)
    root.mkdir()
    write_json(root / "sweep.json", {"id": "test", "fingerprint": plan.fingerprint})
    cell = plan.cells[0]
    run = root / cell.key / f"sweep-test-{cell.key}-r0"
    run.mkdir(parents=True)
    write_json(run / "attempt.json", Attempt(phase=phase).model_dump())
    return plan, run


def test_plan_resolves_variants_without_creating_run_directories(matrix, tmp_path):
    root = tmp_path / "runs"
    command = ["sweep", str(matrix), "--root", str(root), "--plan"]
    runner = CliRunner()
    first = runner.invoke(app, command)
    second = runner.invoke(app, command)
    assert first.exit_code == second.exit_code == 0, first.output
    assert first.output == second.output
    cells = json.loads(first.output)
    assert [c["selector"] for c in cells] == ["task-a", "task-b"]
    assert all(c["agents"] == 1 and c["revision"] == "a" * 40 for c in cells)
    assert not root.exists()


def test_staging_excludes_designated_secrets_in_linked_directories(matrix, tmp_path):
    tools = tmp_path / "shared-tools"
    tools.mkdir()
    source = tools / "private.env"
    source.write_text("TEST_TOKEN=dummy\n")
    source.chmod(0o600)
    (tools / "lookup.py").write_text("# ordinary tool input\n")
    (matrix.parent / "tools").symlink_to(tools, target_is_directory=True)
    target = tmp_path / "run" / "experiment"
    rundir.copy_experiment(matrix.parent, target, rundir.InitReport(), exclude=source)
    assert (target / "tools" / "lookup.py").is_file()
    assert not (target / "tools" / source.name).exists()


@pytest.mark.parametrize("kind", ["linked_folder", "absolute_prompt"])
def test_resume_refuses_changed_file_dependencies(matrix, tmp_path, kind):
    spec = matrix.parent / "spec.yaml"
    if kind == "linked_folder":
        external = tmp_path / "shared-tools"
        external.mkdir()
        (matrix.parent / "tools").symlink_to(external, target_is_directory=True)
        spec.write_text(f"{spec.read_text()}tools_dir: tools\n")
        dependency = external / "lookup.py"
    else:
        dependency = tmp_path / "instructions.md"
        spec.write_text(spec.read_text().replace("prompt: solve", f"prompt_file: {dependency}"))
    dependency.write_text("initial\n")
    root = tmp_path / "runs"
    old, _ = _record(root, matrix, "completed")
    dependency.write_text("edited\n")
    new = make_plan(matrix)
    assert [cell.key for cell in old.cells] == [cell.key for cell in new.cells]
    assert new.fingerprint != old.fingerprint
    result = CliRunner().invoke(app, ["sweep", str(matrix), "--root", str(root)])
    assert result.exit_code == 1
    assert "the sweep inputs changed" in result.output


def test_default_tools_and_resources_affect_fingerprint(matrix, tmp_path):
    package = tmp_path / "installed" / "legion"
    shutil.copytree(Path(legion.__file__).parent, package, ignore=shutil.ignore_patterns("__pycache__"))
    env = {**os.environ, "PYTHONPATH": str(package.parent), "PYTHONDONTWRITEBYTECODE": "1"}
    command = [sys.executable, "-c", "from legion.cli import app; app()",
               "sweep", str(matrix), "--plan"]
    original = json.loads(check_output(command, env=env))
    for filename in ("tools/message_board.py", "resources/message_board.py"):
        module = package / "harbor" / "builtin" / filename
        with module.open("a") as file:
            file.write("\n# locally changed built-in\n")
        changed = json.loads(check_output(command, env=env))
        assert [cell["cell"] for cell in original] == [cell["cell"] for cell in changed]
        assert original != changed
        original = changed


@pytest.mark.parametrize("dummy", ["example-file-credential", """escaped-quote-'\"-\\-suffix""",
                                   "prefix-file-secret-" + "x" * 120 + "-suffix-file-secret"])
def test_plan_redacts_values_loaded_from_the_secrets_file(matrix, tmp_path, dummy):
    spec = matrix.parent / "spec.yaml"
    spec.write_text(spec.read_text().replace("inference:", "inference:\n  api_key_env: ${FILE_KEY}"))
    secrets = tmp_path / "secrets.env"
    secrets.write_text(f"FILE_KEY={dummy}\n")
    secrets.chmod(0o600)
    result = CliRunner().invoke(app, ["sweep", str(matrix), "--secrets", str(secrets), "--plan"])
    assert result.exit_code == 1
    assert "api_key_env" in result.output
    assert "input_value" not in result.output
    assert dummy[:15] not in result.output
    assert dummy[-14:] not in result.output


def test_unconfirmed_submission_blocks_new_cells_without_submitting(matrix, tmp_path, monkeypatch):
    root = tmp_path / "runs"
    _, run = _record(root, matrix, "launching")

    def slurm(*argv):
        assert argv[0] == "squeue", argv
        return CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(ops, "_slurm", slurm)
    result = CliRunner().invoke(app, ["sweep", str(matrix), "--root", str(root)])
    assert result.exit_code == 1
    assert "launch outcome unknown" in result.output
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert [(row["status"], row["selected_attempt"]) for row in rows] == [
        ("launching", str(run.relative_to(root))), ("pending", None),
    ]


@pytest.mark.parametrize("scenario", [None, "replaced", "execution", "failed_step", "rollover"])
def test_resume_adopts_existing_swarm_and_keeps_run_evidence(matrix, tmp_path, monkeypatch, scenario):
    root = tmp_path / "runs"
    plan, run = _record(root, matrix, "launching")
    # The other matrix cell is done already and must not be submitted or resumed.
    other = root / plan.cells[1].key / f"sweep-test-{plan.cells[1].key}-r0"
    other.mkdir(parents=True)
    write_json(other / "attempt.json", Attempt(phase="completed").model_dump())
    host = run / "hosts" / "controller.host"
    host.parent.mkdir()
    host.write_text("https://controller.example")
    artifact = run / "results" / run.name / "agent-001" / "trials" / "t1" / "result.json"
    artifact.parent.mkdir(parents=True)
    trial = {"exception_info": None, "verifier_result": {"rewards": {"reward": 0.0}}}
    if scenario == "rollover":
        write_json(artifact, {"exception_info": {"exception_type": "AgentTimeoutError"}})
        artifact = artifact.parents[1] / "t10" / "result.json"
        artifact.parent.mkdir()
    if scenario == "failed_step":
        trial["step_results"] = [{"exception_info": {"exception_type": "AgentTimeoutError"}}]
    write_json(artifact, trial)
    released, ended = False, False

    def http(request):
        nonlocal released
        path = request.url.path
        if path == "/healthz":
            return httpx.Response(200, json={"ok": True, "process": 123})
        if path == f"/swarms/{run.name}":
            return httpx.Response(200, json={"spec": {"compute": {"kind": "slurm"}}, "status": {
                "agents_total": 1, "pending_prompts": 0,
                "released": {"unsynced": {}} if released else None,
                "types": {"role": {"loop": {"state": "done"}, "replicas": 1, "health": "ok",
                                   "agents_by_state": {"idle": 1}}},
            }})
        if path == "/workers":
            return httpx.Response(200, json=[{
                "id": "worker-1", "job_id": None if released else "555", "agents": ["agent-001"],
                "crashes": 0,
            }])
        if path == f"/swarms/{run.name}/loop":
            reason = "errors" if scenario == "execution" else "max_rounds"
            return httpx.Response(200, json=[{"per_agent": {"agent-001": {"done_reason": reason}}}])
        if path == f"/swarms/{run.name}/pause":
            assert json.loads(request.content)["release"] is True
            released = True
            return httpx.Response(200, json={})
        if path == "/swarms":
            return httpx.Response(200, json=[{"swarm": run.name, "released": released}])
        if path == "/checkpoint":
            return httpx.Response(200, json={"files": 1})
        pytest.fail(f"unexpected controller call {request.method} {path}")

    def slurm(*argv):
        nonlocal ended
        controller = ops._job_name("controller", run)
        stdout = ""
        if argv[0] == "squeue":
            if controller in argv[argv.index("-n") + 1].split(",") and not ended:
                stdout = "RUNNING\n"
        elif argv[0] == "scancel":
            assert controller in argv
            ended = True
        elif argv[0] == "sacct":
            stdout = "worker-1|555|COMPLETED|0:0\n"
            if scenario == "replaced":
                stdout += "worker-1|554|PREEMPTED|0:75\n"
        else:
            pytest.fail(f"unexpected scheduler call {argv}")
        return CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(client, "connect", lambda _: httpx.Client(
        base_url="https://controller.example", transport=httpx.MockTransport(http),
    ))
    monkeypatch.setattr(ops, "_slurm", slurm)
    result = CliRunner().invoke(app, ["sweep", str(matrix), "--root", str(root)])
    failed = scenario in {"replaced", "execution", "failed_step"}
    assert result.exit_code == int(failed), result.output
    assert released and ended
    row, previous = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert len(row["attempts"]) == len(previous["attempts"]) == 1
    assert previous["status"] == "completed"
    assert row["status"] == ("infrastructure_failure" if failed else "completed")
    if failed:
        assert row["artifacts"] is None
    else:
        assert (root / row["artifacts"] / "agent-001" / "trials" / artifact.parent.name
                / "result.json").samefile(artifact)
