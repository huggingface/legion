"""Offline regression tests for the experiment's identity-bound benchmark adapter."""

import importlib.util
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from flock.world import Agent, ToolError
from flock.world.server import create_app

MODULE = Path(__file__).resolve().parents[1] / "experiments/agent-escape/world.py"
spec = importlib.util.spec_from_file_location("escape_experiment", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def world(tmp_path):
    state = {}

    def backend(request):
        if request.url.path == "/env/reset":
            session = request.url.params["session_id"]
            state[session] = 0
            return httpx.Response(
                200,
                json={
                    "ok": True,
                    "case_id": session,
                    "narrative": "Puzzle",
                    "available_entities": [{"id": "visible"}],
                    "available_information": [],
                    "win_node_id": "goal",
                    "message": "/private/solution.json",
                    "golden_answer": "secret",
                },
            )
        _, _, session, action = request.url.path.split("/")
        args = json.loads(request.content)
        if action == "investigate":
            if args["node_id"] == "missing":
                return httpx.Response(404, json={"detail": "Node not found"})
            state[session] += 1
            return httpx.Response(200, json={"progress": state[session]})
        if action == "submit_answer":
            return httpx.Response(200, json={"ok": True, "is_win": args["answer"] == "correct"})
        return httpx.Response(200, json={"ok": True})

    client = httpx.Client(base_url="http://benchmark", transport=httpx.MockTransport(backend))
    return module.EscapeWorld(client=client, case_file="case", output=tmp_path, run_id="test")


def test_sessions_are_independent_and_observation_does_not_reset(world):
    a, b = Agent("worker-0000"), Agent("worker-0001")
    observation = world.get_observation(a)
    assert set(observation) == {
        "narrative",
        "available_entities",
        "available_information",
        "win_node_id",
    }
    assert world.investigate(a, "visible") == {"progress": 1}
    assert world.investigate(a, "visible") == {"progress": 2}
    assert world.investigate(b, "visible") == {"progress": 1}
    assert world.get_observation(a) == observation
    third = world.investigate(a, "visible")
    assert third["progress"] == 3
    assert "collaboration_reminder" in third
    assert "collaboration_reminder" not in world.investigate(b, "visible")
    assert world.sessions[a.name]["case_id"] != world.sessions[b.name]["case_id"]


def test_only_benchmark_win_records_success(world):
    a = Agent("worker-0000")
    assert world.submit_answer(a, "wrong")["is_win"] is False
    assert not (world.output / "success.json").exists()
    assert world.submit_answer(a, "correct")["is_win"] is True
    assert json.loads((world.output / "success.json").read_text())["agent"] == a.name
    assert world.investigate(a, "visible")["swarm_solved"] is True


def test_unknown_identity_and_invalid_action(world):
    with pytest.raises(ToolError, match="Unknown"):
        world.get_observation(Agent("outsider"))
    assert world.investigate(Agent("worker-0000"), "missing") == {
        "ok": False,
        "error": "Node not found",
    }


def test_http_auth_and_no_session_override(world):
    client = TestClient(create_app(world, "secret"))
    assert client.get("/manifest").status_code == 401
    headers = {"X-World-Token": "secret", "X-Flock-Agent": "worker-0000"}
    manifest = client.get("/manifest", headers=headers).json()
    assert {t["name"] for t in manifest["tools"]} == {
        "get_observation",
        "investigate",
        "tool_call",
        "use_item",
        "submit_answer",
    }
    response = client.post(
        "/tools/investigate",
        headers=headers,
        json={"args": {"node_id": "visible", "case_id": "other"}},
    )
    assert response.status_code == 200
    assert response.json()["ok"] is False
    assert not world.sessions
    result = client.post("/tools/get_observation", headers=headers, json={"args": {}})
    assert result.json()["ok"] is True
    assert list(world.sessions) == ["worker-0000"]


def test_backend_failure_never_reports_success(world):
    world.client = httpx.Client(
        base_url="http://benchmark",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(500, text="private error with solution")
        ),
    )
    with pytest.raises(ToolError, match="Benchmark server failed"):
        world.get_observation(Agent("worker-0000"))
    assert not (world.output / "success.json").exists()


def test_stop_drains_process_before_returning(monkeypatch):
    import signal
    from unittest.mock import Mock

    source = MODULE.with_name("run.py")
    spec = importlib.util.spec_from_file_location("escape_run", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    process = Mock()
    process.poll.return_value = None
    runner.stop(process, grace=7)
    process.send_signal.assert_called_once_with(signal.SIGINT)
    process.wait.assert_called_once_with(timeout=7)


def test_stop_reports_forced_shutdown(monkeypatch):
    import signal
    import subprocess
    from unittest.mock import Mock

    source = MODULE.with_name("run.py")
    spec = importlib.util.spec_from_file_location("escape_run", source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    process = Mock(pid=12345)
    process.poll.return_value = None
    process.wait.side_effect = [subprocess.TimeoutExpired("flock", 1), 0]
    kill = Mock()
    monkeypatch.setattr(runner.os, "killpg", kill)
    with pytest.raises(RuntimeError, match="did not shut down gracefully"):
        runner.stop(process, grace=1)
    kill.assert_called_once_with(12345, signal.SIGKILL)
