import json
from pathlib import Path
from subprocess import CompletedProcess

import httpx
import pytest
from typer.testing import CliRunner

from legion import client, ops
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


@pytest.mark.parametrize("failure", [None, "replaced", "execution"])
def test_resume_adopts_existing_swarm_and_keeps_run_evidence(matrix, tmp_path, monkeypatch, failure):
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
    write_json(artifact, {"exception_info": None, "verifier_result": {"rewards": {"reward": 0.0}}})
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
            reason = "errors" if failure == "execution" else "max_rounds"
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
            if failure == "replaced":
                stdout += "worker-1|554|PREEMPTED|0:75\n"
        else:
            pytest.fail(f"unexpected scheduler call {argv}")
        return CompletedProcess(argv, 0, stdout, "")

    monkeypatch.setattr(client, "connect", lambda _: httpx.Client(
        base_url="https://controller.example", transport=httpx.MockTransport(http),
    ))
    monkeypatch.setattr(ops, "_slurm", slurm)
    result = CliRunner().invoke(app, ["sweep", str(matrix), "--root", str(root)])
    assert result.exit_code == int(failure is not None), result.output
    assert released and ended
    row, previous = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    assert len(row["attempts"]) == len(previous["attempts"]) == 1
    assert previous["status"] == "completed"
    assert row["status"] == ("infrastructure_failure" if failure else "completed")
    if failure:
        assert row["artifacts"] is None
    else:
        assert (root / row["artifacts"] / Path("agent-001/trials/t1/result.json")).samefile(artifact)
