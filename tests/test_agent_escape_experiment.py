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


@pytest.fixture
def runner():
    source = MODULE.with_name("run.py")
    spec = importlib.util.spec_from_file_location("escape_selection_run", source)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.mark.parametrize("difficulty", [5, 10, 15, 20, 25])
def test_selection_sorts_files_within_difficulty(runner, tmp_path, difficulty):
    folder = tmp_path / "questions" / str(difficulty) / "narrative"
    folder.mkdir(parents=True)
    for name in ("z.json", "a.json", "m.json"):
        (folder / name).write_text("{}")
    (folder / "ignore.txt").write_text("ignored")
    (folder / "directory.json").mkdir()
    assert runner.select_case(tmp_path, difficulty, 0).name == "a.json"
    assert runner.select_case(tmp_path, difficulty, 1).name == "m.json"
    assert runner.select_case(tmp_path, difficulty, 2).name == "z.json"


def test_selection_errors(runner, tmp_path):
    with pytest.raises(ValueError, match="Unsupported difficulty"):
        runner.select_case(tmp_path, 11, 0)
    with pytest.raises(ValueError, match="directory is missing"):
        runner.select_case(tmp_path, 10, 0)
    folder = tmp_path / "questions/10/narrative"
    folder.mkdir(parents=True)
    with pytest.raises(ValueError, match="No scenario JSON"):
        runner.select_case(tmp_path, 10, 0)
    (folder / "a.json").write_text("{}")
    for index in (-1, 1, 60):
        with pytest.raises(ValueError, match="valid range is 0–0"):
            runner.select_case(tmp_path, 10, index)


@pytest.mark.parametrize("flags", [[], ["--difficulty", "20", "--scenario-index", "1"],
    ["--difficulty", "20", "--scenario-index", "1", "--communication", "off"]])
def test_launcher_passes_selection_and_records_metadata(runner, tmp_path, monkeypatch, flags):
    import hashlib
    from unittest.mock import Mock

    source = tmp_path / ".flock/deps/agent-escape"
    default = "skeleton_091c541b_narrative.json"
    for difficulty in (10, 20):
        folder = source / "questions" / str(difficulty) / "narrative"
        folder.mkdir(parents=True)
        (folder / default).write_text('{"puzzle": "first"}')
        (folder / "z.json").write_text('{"puzzle": "second"}')
    python = tmp_path / ".flock/deps/agent-escape-venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.sys, "argv", ["run.py", *flags])
    monkeypatch.setattr(runner.signal, "signal", Mock())
    monkeypatch.setattr(runner.subprocess, "check_output", Mock(side_effect=[runner.REVISION, ""]))
    backend = Mock()
    backend.poll.return_value = None
    flock = Mock(returncode=0)
    flock.poll.return_value = 0
    launch = Mock(side_effect=[backend, flock])
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.get.return_value.is_success = True
    monkeypatch.setattr(runner.httpx, "Client", Mock(return_value=client))
    assert runner.main() == 1  # Idle without a verified solution is not success.
    difficulty, index = (20, 1) if flags else (10, 0)
    selected = source / "questions" / str(difficulty) / "narrative" / (
        "z.json" if flags else default
    )
    assert launch.call_args_list[1].kwargs["env"]["ESCAPE_CASE_FILE"] == str(selected)
    output = next((tmp_path / ".flock/agent-escape-attempts").iterdir())
    for name in ("metadata.json", "result.json"):
        record = json.loads((output / name).read_text())
        assert record["difficulty"] == difficulty
        assert record["scenario_index"] == index
        assert record["case"] == selected.relative_to(source).as_posix()
        assert record["case_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
        assert record["benchmark_revision"] == runner.REVISION
        assert record['communication'] == ('off' if '--communication' in flags else 'on')
    if '--communication' in flags:
        assert record['persistence_status'] == 'not_applicable'
        metadata = json.loads((output / 'metadata.json').read_text())
        assert metadata['spec'].get('comms') is None
        assert '-comm-off-' in metadata['spec']['swarm']
        assert launch.call_args_list[1].kwargs['env']['ESCAPE_COMMUNICATION'] == 'off'


@pytest.mark.parametrize("flags, message", [
    (["--difficulty", "12"], "invalid choice"),
    (["--scenario-index", "-1"], "valid range"),
    (["--scenario-index", "1"], "valid range"),
])
def test_invalid_selection_exits_before_artifacts_or_processes(
    runner, tmp_path, monkeypatch, capsys, flags, message
):
    from unittest.mock import Mock

    folder = tmp_path / ".flock/deps/agent-escape/questions/10/narrative"
    folder.mkdir(parents=True)
    (folder / "a.json").write_text("{}")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.sys, "argv", ["run.py", *flags])
    monkeypatch.setattr(runner.signal, "signal", Mock())
    launch = Mock()
    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / ".flock/agent-escape-attempts").exists()
    launch.assert_not_called()


@pytest.mark.parametrize("agents", [2, 4, 8, 16, 32])
def test_configured_agent_count_bounds_identities(world, agents):
    world.agents = agents
    assert world.get_observation(Agent(f"worker-{agents-1:04d}"))["narrative"] == "Puzzle"
    with pytest.raises(ToolError, match="Unknown experiment worker"):
        world.get_observation(Agent(f"worker-{agents:04d}"))


def test_transport_failure_detected_even_when_flock_exits_idle(runner, tmp_path):
    event_file = tmp_path / "flock/run/events/worker-0000.jsonl"
    event_file.parent.mkdir(parents=True)
    rows = [
        {"type": "turn_ended", "agent": "worker-0000", "data": {"data": {
            "status": "error", "reason": "model_transport", "error": "private details"}}},
        {"type": "turn_ended", "agent": "worker-0001", "data": {"data": {
            "status": "completed"}}},
    ]
    event_file.write_text("\n".join(map(json.dumps, rows)))
    assert runner.agent_transport_failures(tmp_path) == [
        {"agent": "worker-0000", "reason": "model_transport"}]


def test_no_communication_world_preserves_isolation_and_stop(world, monkeypatch, tmp_path):
    monkeypatch.setenv('ESCAPE_COMMUNICATION', 'off')
    off = module.EscapeWorld(client=world.client, case_file='case', output=tmp_path / 'off',
                            run_id='off', agents=2)
    a, b = Agent('worker-0000'), Agent('worker-0001')
    assert off.get_observation(a) == world.get_observation(a)
    assert 'message board' not in off.description
    for n in range(1, 7):
        assert off.investigate(a, 'visible') == {'progress': n}
    assert off.investigate(b, 'visible') == {'progress': 1}
    assert off.sessions[a.name]['case_id'] != off.sessions[b.name]['case_id']
    assert off.submit_answer(a, 'correct')['is_win']
    assert off.investigate(b, 'visible')['swarm_solved']


def test_no_communication_spec_has_only_puzzle_capabilities(runner):
    import copy

    import yaml

    from flock.models import SwarmSpec, comms_tools

    original = yaml.safe_load(MODULE.with_name('spec-router-pi.yaml').read_text())
    unchanged = copy.deepcopy(original)
    runner.configure_communication(unchanged, 'on')
    assert unchanged == original
    runner.configure_communication(original, 'off')
    parsed = SwarmSpec.model_validate(original)
    assert parsed.comms is None
    assert comms_tools(parsed.types['worker'], parsed.comms) == []
    assert parsed.types['worker'].pi_tools_only
    prompt = Path(original['types']['worker']['system_prompt_file']).read_text()
    for text in [prompt, runner.initial_prompt('off')]:
        for forbidden in ['read(n=', 'post(', 'teammate', 'message board', 'thread 1']:
            assert forbidden not in text
    assert 'FLOCK_COMMS_RATE' not in original['types']['worker']['env']
