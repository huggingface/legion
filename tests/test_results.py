"""legion.results on run directories written by hand (docs/results.md): the evidence edge cases
a tiny run cannot produce on demand. The tiny runs in test_runs.py cover a real local and Slurm
run end to end."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from legion import cli, results
from legion.results import Outcome

SPEC = """\
name: r
model: m
task:
  repo: https://huggingface.co/datasets/ns/t
  selector: x
compute: slurm
environment:
  backend: podman
naming: consecutive-id
agents:
  count: 2
inference:
  api_key_env: ${KEY_ENV:-INFERENCE_TOKEN}
"""
DONE = {"verifier_result": {"rewards": {"reward": 1.0}}}


def write(path: Path, doc) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc if isinstance(doc, str) else json.dumps(doc))


def slurm_run(root: Path, trials: dict[str, dict] | None = None) -> Path:
    """A Slurm run dir of swarm `r`: its spec, and one Trial per agent (`DONE` by default)."""
    write(root / "experiment" / "spec.yaml", SPEC)
    for agent, doc in (trials or {"agent-001": DONE, "agent-002": DONE}).items():
        write(root / "results/r" / agent / "trials" / f"r-{agent}-t1" / "result.json", doc)
    return root


def record(folder: Path, worker: str, n: int, agents: list[str], exit: str | None) -> None:
    write(folder / worker / f"{n:032x}.json", {
        "worker": worker, "job_id": str(n), "started_at": f"2026-11-27T10:0{n}:00+00:00",
        "exit": exit, "agents": agents})


def test_spec_errors_never_quote_secrets(tmp_path):
    """A spec check whose message quotes its input (a resolved ${VAR}) is reported by its place
    and kind only, through the API and the CLI."""
    run = slurm_run(tmp_path / "run")
    write(run / "run.env", "KEY_ENV=hf-not-a-name-but-a-token-1234\n")
    r = results.load(run)
    assert r.reasons[0] == "spec: inference.api_key_env: value_error", r.reasons
    out = CliRunner().invoke(cli.app, ["results", str(run)])
    assert out.exit_code == 0 and "api_key_env: value_error" in out.output, out.output
    assert "token-1234" not in r.model_dump_json() + out.output


@pytest.mark.parametrize("step_error, outcome", [
    ("VerifierTimeoutError", Outcome.TIMEOUT), ("RuntimeError", Outcome.HARNESS_ERROR)])
def test_a_failed_step_is_not_completed(tmp_path, step_error, outcome):
    """A multi-step Trial whose later step failed keeps the first step's finite reward and no
    top-level exception: still not a completed result."""
    steps = {**DONE, "step_results": [{"exception_info": None},
                                      {"exception_info": {"exception_type": step_error}}]}
    r = results.load(slurm_run(tmp_path, {"agent-001": DONE, "agent-002": steps}))
    trial = r.agent("agent-002").trials[0]
    assert (trial.outcome, trial.exception_type, trial.reward) == (outcome, step_error, 1.0)
    assert (r.agent("agent-002").outcome, r.agent("agent-002").reward) == (outcome, None)
    assert r.outcome is outcome and r.agent("agent-001").outcome is Outcome.COMPLETED


def test_a_failed_submission_is_not_a_replacement(tmp_path):
    """The controller writes a launch env file before sbatch, also for a submission sbatch
    refused; the records say which launches ran."""
    run = slurm_run(tmp_path)
    for n in (1, 2):  # a refused sbatch, then the job that ran
        write(run / "launch" / f"w0-{n:032x}.env", "WORKER_ID=w0\n")
    record(run / "results/r/_workers", "w0", 2, ["agent-001", "agent-002"], "drained")
    r = results.load(run)
    assert (r.outcome, r.reasons) == (Outcome.COMPLETED, [])
    assert [i.job_id for i in r.workers[0].incarnations] == ["2"]


def test_a_crash_before_an_agent_arrived_is_not_its_own(tmp_path):
    """A local worker that crashed, was relaunched and then took a new agent: the relaunched
    agent is worker_replaced, the new one completed (the records say where each ran)."""
    bundle = tmp_path / ".legion" / "r-20261127-100000"
    write(bundle / "run.json", {
        "swarm": "r", "started_at": "2026-11-27T10:00:00+00:00",
        "ended_at": "2026-11-27T11:00:00+00:00", "stop_reason": "idle",
        "spec": {"types": {"agent": {"replicas": 2, "harbor": {"config": {"task": {}}}}}},
        "agents": [{"name": "a", "worker": "w0"}, {"name": "b", "worker": "w0"}],
        "workers": [{"id": "w0", "crashes": 1}]})
    harbor = bundle / "agents" / "_harbor"
    record(harbor / "_workers", "w0", 1, ["a"], None)  # killed
    record(harbor / "_workers", "w0", 2, ["a", "b"], "drained")
    for agent in ("a", "b"):
        write(harbor / agent / "trials" / f"r-{agent}-t1" / "result.json", DONE)
    r = results.load(tmp_path)  # the folder holding the bundle
    assert {a.name: (a.outcome, a.reward) for a in r.agents} == {
        "a": (Outcome.WORKER_REPLACED, 1.0), "b": (Outcome.COMPLETED, 1.0)}
    assert r.reasons == ["a: worker_replaced, worker w0"]


def test_one_local_bundle_wins_over_a_run_dir_layout(tmp_path):
    """A folder with one local bundle is that run, also when it holds experiment/ (a local
    `legion init` dir) or an empty results/."""
    bundle = tmp_path / ".legion" / "r-20261127-100000"
    write(bundle / "run.json", {"swarm": "r", "spec": {"types": {}}, "agents": []})
    write(tmp_path / "experiment" / "spec.yaml", SPEC)
    (tmp_path / "results").mkdir()
    r = results.load(tmp_path)
    assert (r.compute, r.root, r.attempt_id) == ("local", bundle, bundle.name)


def test_the_results_command_needs_no_controller(tmp_path, monkeypatch):
    """An archived run has no controller host file: `legion results` reads it under --run and
    LEGION_RUN_DIR too."""
    run = slurm_run(tmp_path / "archive")
    for argv in (["--run", str(run), "results", str(run)], ["results", str(run)]):
        monkeypatch.setenv("LEGION_RUN_DIR", str(run))
        out = CliRunner().invoke(cli.app, argv)
        assert out.exit_code == 0, out.output
        assert json.loads(out.output)["outcome"] == "completed"
