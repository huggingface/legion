"""CIB admits only Legion-completed Trials to the dataset scorer."""

import importlib.util
import json
from pathlib import Path

import pytest

from legion.results import Outcome, load

spec = importlib.util.spec_from_file_location(
    "cib_parity", Path(__file__).resolve().parents[1] / "experiments/cib/parity.py")
parity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parity)


@pytest.mark.parametrize(("reward", "crashes", "outcome", "admitted"), [
    (1, 0, Outcome.COMPLETED, True),
    (0, 0, Outcome.COMPLETED, True),
    (0, 1, Outcome.WORKER_REPLACED, False),
])
def test_verifier_admission(tmp_path, reward, crashes, outcome, admitted):
    root = tmp_path / ".legion" / "cib-run"
    trial = root / "agents/_harbor/agent-001/trials/agent-001-t1"
    verifier = trial / "verifier/result.json"
    verifier.parent.mkdir(parents=True)
    identity = {"scenario_id": "one", "variant": "solo", "seed": 0}
    gateway = {"upstream": "example"}
    envelope = {"identity": identity, "fixture": False, "reward": reward,
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
