"""CIB admits only Legion-completed Trials to the dataset scorer."""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from legion.results import Outcome, load

spec = importlib.util.spec_from_file_location(
    "cib_parity", Path(__file__).resolve().parents[1] / "experiments/cib/parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


@pytest.mark.parametrize(("reward", "crashes", "status", "reason", "outcome", "admitted"), [
    (1, 0, "finished", None, Outcome.COMPLETED, True),
    (0, 0, "finished", None, Outcome.COMPLETED, True),
    (0, 0, "abandoned", "other", Outcome.COMPLETED, True),
    (0, 0, "abandoned", "unanswered_request_timeout", Outcome.COMPLETED, False),
    (0, 0, "abandoned", "episode_timeout", Outcome.COMPLETED, False),
    (0, 0, "infrastructure_error", None, Outcome.COMPLETED, False),
    (0, 1, "finished", None, Outcome.WORKER_REPLACED, False),
])
def test_verifier_admission(tmp_path, reward, crashes, status, reason, outcome, admitted):
    root = tmp_path / ".legion" / "cib-run"
    trial = root / "agents/_harbor/agent-001/trials/agent-001-t1"
    verifier = trial / "verifier/result.json"
    verifier.parent.mkdir(parents=True)
    identity = {"scenario_id": "one", "variant": "solo", "seed": 0}
    gateway = {"upstream": "example"}
    envelope = {"identity": identity, "fixture": False, "reward": reward, "status": status, "terminal_reason": reason,
                "configuration": {"agent": {"runner": "legion-launch", "model_gateway": gateway}}}
    verifier.write_text(json.dumps(envelope))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))
    (root / "run.json").write_text(json.dumps({
        "spec": {}, "ended_at": "2026-01-01T00:00:00Z", "stop_reason": "idle",
        "agents": [{"name": "agent-001", "worker": "worker"}],
        "workers": [{"id": "worker", "crashes": crashes}],
    }))
    assert load(tmp_path).outcome is outcome
    found = parity.result(tmp_path, identity, gateway)
    assert found == (verifier if admitted else None)
    if found:
        assert parity.read_json(found) == envelope  # the scorer receives the envelope unchanged


@pytest.mark.parametrize("reason", ["unanswered_request_timeout", "episode_timeout"])
def test_timeout_is_retried_on_resume(tmp_path, reason, monkeypatch):
    output = tmp_path / "output"
    task = {"task_id": "solo", "identity": {"scenario_id": "one", "variant": "solo", "seed": 0}}
    case = output / "cases" / task["task_id"]
    root = case / ".legion/run"
    trial = root / "agents/_harbor/agent-001/trials/agent-001-t1"
    verifier = trial / "verifier/result.json"
    verifier.parent.mkdir(parents=True)
    envelope = {"identity": task["identity"], "fixture": False, "status": "abandoned",
                "terminal_reason": reason, "reward": 0,
                "configuration": {"agent": {"runner": "legion-launch", "model_gateway": {}}}}
    verifier.write_text(json.dumps(envelope))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0}}}))
    (root / "run.json").write_text(json.dumps({
        "spec": {}, "ended_at": "2026-01-01T00:00:00Z", "stop_reason": "idle",
        "agents": [{"name": "agent-001", "worker": "worker"}],
        "workers": [{"id": "worker"}],
    }))
    # A real subprocess stands in for an unavailable external launcher.
    launcher = tmp_path / "launcher"
    launcher.write_text("#!/bin/sh\nexit 1\n")
    launcher.chmod(0o755)
    monkeypatch.setattr(parity.sys, "executable", str(launcher))
    with (tmp_path / "lock").open("w") as lock:
        assert parity.run_case(output, task, dict(os.environ), {}, lock.fileno()) is None
    assert not verifier.exists()
    assert (output / "cases/solo.failed-1/.legion/run/agents/_harbor/agent-001/trials"
            / "agent-001-t1/verifier/result.json").is_file()
    assert (case / "launch.out").is_file()


@pytest.mark.parametrize("reason", ["unanswered_request_timeout", "episode_timeout"])
def test_single_case_timeout_exit_with_dataset(tmp_path, monkeypatch, reason):
    """Exercise the CLI with the real pinned dataset adapter when its clone is available."""
    dataset_path = os.environ.get("CIB_TEST_DATASET")
    if not dataset_path:
        pytest.skip("set CIB_TEST_DATASET to the pinned dataset clone")
    dataset = Path(dataset_path).resolve()
    monkeypatch.syspath_prepend(str(dataset / "adapter/src"))
    pytest.importorskip("collab_harbor.scoring")
    manifest_path = dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    task = next(t for t in manifest["tasks"] if t["task_id"] ==
                "ci-sharing-h2-r-choosing-the-safe-field-station-solo-s0")
    output = tmp_path / "output"
    output.mkdir()
    _, gateway, fingerprint = parity.configuration(dataset, manifest_path, task["task_id"])
    (output / "configuration.sha256").write_text(fingerprint + "\n")
    root = output / "cases" / task["task_id"] / ".legion/run"
    trial = root / "agents/_harbor/agent-001/trials/agent-001-t1"
    verifier = trial / "verifier/result.json"
    verifier.parent.mkdir(parents=True)
    verifier.write_text(json.dumps({
        "identity": task["identity"], "fixture": False, "status": "abandoned",
        "terminal_reason": reason, "reward": 0,
        "configuration": {"agent": {"runner": "legion-launch", "model_gateway": gateway}},
    }))
    (trial / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": 0}}}))
    (root / "run.json").write_text(json.dumps({
        "spec": {}, "ended_at": "2026-01-01T00:00:00Z", "stop_reason": "idle",
        "agents": [{"name": "agent-001", "worker": "worker"}],
        "workers": [{"id": "worker"}],
    }))
    launcher = tmp_path / "launcher"
    launcher.write_text("#!/bin/sh\nexit 1\n")
    launcher.chmod(0o755)
    monkeypatch.setattr(parity.sys, "executable", str(launcher))
    monkeypatch.setattr(parity.sys, "argv", ["parity.py", "--dataset", str(dataset), "--output",
                                           str(output), "--case", task["task_id"]])
    assert parity.main() == 1
    assert (output / "cases" / (task["task_id"] + ".failed-1")).is_dir()
    assert (output / "runs.txt").read_text() == ""
