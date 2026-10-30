"""Slurm experiment orchestration with fake scheduler/processes; no real workloads."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("escape_slurm", ROOT / "experiments/agent-escape/slurm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.signal, "signal", lambda *a: None)
    return module


@pytest.fixture
def submission(launcher, tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "private-hf-token")
    server = tmp_path / "server"
    server.mkdir()
    (server / "endpoint.json").write_text(json.dumps({"base_url": "http://router/v1",
                                                   "model": "Qwen/Qwen3.8-27B", "instances": []}))
    (server / "api-key").write_text("private-server-token")
    monkeypatch.setattr(launcher, "load_launcher", lambda: SimpleNamespace(select_case=lambda *a: None))
    args = SimpleNamespace(server_run=server, spec=ROOT / "experiments/agent-escape/spec-router-qwen27b.yaml",
        difficulties=[5, 10], scenario_index=0, timeout=900, output=tmp_path / "out", partition="cpu",
        cpus=8, mem="16G", time="01:00:00", dry_run=False)
    calls = []
    def check(command, **kwargs):
        if command[0] == "git":
            return "revision\n"
        calls.append((command, kwargs))
        return "12345;cluster\n"
    monkeypatch.setattr(launcher.subprocess, "check_output", check)
    return args, calls


def test_submission_keeps_credentials_out_of_scheduler(launcher, submission):
    args, calls = submission
    assert launcher.submit(args) == 0
    command, kwargs = calls[0]
    assert "--export=NONE" in command
    assert "HF_TOKEN" not in kwargs["env"] and "LOCAL_KEY" not in kwargs["env"]
    assert "private-hf-token" not in str(command)
    config = (args.output / "config.json").read_text()
    assert "private-hf-token" not in config and "private-server-token" not in config
    assert (args.output / "secrets.json").stat().st_mode & 0o777 == 0o600
    assert json.loads((args.output / "submission.json").read_text())["job_id"] == "12345"


def test_dry_run_submits_nothing_and_removes_credentials(launcher, submission):
    args, calls = submission
    args.dry_run = True
    assert launcher.submit(args) == 0
    assert not calls and not (args.output / "secrets.json").exists()


def test_submission_failure_removes_credentials(launcher, submission, monkeypatch):
    args, _ = submission
    original = launcher.subprocess.check_output
    def fail(command, **kw):
        if command[0] == "sbatch":
            raise OSError("scheduler unavailable")
        return original(command, **kw)
    monkeypatch.setattr(launcher.subprocess, "check_output", fail)
    with pytest.raises(OSError):
        launcher.submit(args)
    assert not (args.output / "secrets.json").exists()


def test_secret_permissions_and_symlinks_rejected(launcher, tmp_path):
    p = tmp_path / "secrets"
    p.write_text('{}')
    p.chmod(0o644)
    with pytest.raises(ValueError):
        launcher.read_secrets(p)
    p.chmod(0o600)
    link = tmp_path / "link"
    link.symlink_to(p)
    with pytest.raises(ValueError):
        launcher.read_secrets(link)


@pytest.mark.parametrize("codes", [[0, 0], [1, 0]])
def test_attempts_are_sequential_and_results_preserved(launcher, submission, monkeypatch, codes):
    args, _ = submission
    output = launcher.prepare(args)
    observed = []
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {"data": [{"id": "Qwen/Qwen3.8-27B"}]})
    monkeypatch.setattr(launcher.httpx, "Client", Client)
    monkeypatch.setattr(launcher, "sample_metrics", lambda *a: [])
    class Process:
        def __init__(self, command, **kw):
            index = len(observed)
            assert index == 0 or observed[-1].polled
            self.polled = False
            self.returncode = codes[index]
            observed.append(self)
            assert kw["env"]["LOCAL_KEY"] == "private-server-token"
            attempt = output / f"attempt-{index}"
            attempt.mkdir()
            launcher.write_json(attempt / "result.json", {"reason": "verified_success" if not self.returncode else "idle"})
            kw["stdout"].write(f"Attempt artifacts: {attempt}\n")
            kw["stdout"].flush()
        def poll(self):
            self.polled = True
            return self.returncode
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    assert launcher.run(output / "config.json") == int(any(codes))
    results = launcher.read_json(output / "summary.json")
    assert [r["difficulty"] for r in results] == [5, 10]
    assert [r["exit_code"] for r in results] == codes
    assert not (output / "secrets.json").exists()


def test_changed_source_fails_before_launch(launcher, submission, monkeypatch):
    args, _ = submission
    output = launcher.prepare(args)
    config = launcher.read_json(output / "config.json")
    config["source_sha256"][str(ROOT / "experiments/agent-escape/run.py")] = "wrong"
    launcher.write_json(output / "config.json", config)
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *a, **kw: pytest.fail("must not launch"))
    assert launcher.run(output / "config.json") == 1
    assert launcher.read_json(output / "error.json")["error_type"] == "ValueError"
    assert not (output / "secrets.json").exists()


def test_stop_signal_interrupts_active_attempt_and_skips_next(launcher, submission, monkeypatch):
    args, _ = submission
    output = launcher.prepare(args)
    handlers = {}
    monkeypatch.setattr(launcher.signal, "signal", lambda sig, fn: handlers.setdefault(sig, fn))
    class Client:
        def __init__(self, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def get(self, url):
            return SimpleNamespace(raise_for_status=lambda: None,
                json=lambda: {"data": [{"id": "Qwen/Qwen3.8-27B"}]})
    monkeypatch.setattr(launcher.httpx, "Client", Client)
    monkeypatch.setattr(launcher, "sample_metrics", lambda *a: [])
    processes = []
    class Process:
        def __init__(self, command, **kw):
            processes.append(self)
            self.returncode = None
            self.signals = []
            attempt = output / "interrupted"
            attempt.mkdir()
            launcher.write_json(attempt / "result.json", {"reason": "interrupted"})
            kw["stdout"].write(f"Attempt artifacts: {attempt}\n")
            kw["stdout"].flush()
        def poll(self):
            return self.returncode
        def send_signal(self, sig):
            self.signals.append(sig)
            self.returncode = 1
    # Trigger the signal after Popen returns but before its first poll.
    def sample(*args):
        handlers[launcher.signal.SIGTERM]()
        return []
    monkeypatch.setattr(launcher, "sample_metrics", sample)
    monkeypatch.setattr(launcher.subprocess, "Popen", Process)
    assert launcher.run(output / "config.json") == 1
    assert len(processes) == 1
    assert processes[0].signals == [launcher.signal.SIGINT]
    assert not (output / "secrets.json").exists()


def test_invalid_selection_fails_before_writing_artifacts(launcher, submission, monkeypatch):
    args, _ = submission
    def invalid(*args):
        raise ValueError("Invalid scenario")
    monkeypatch.setattr(launcher, "load_launcher", lambda: SimpleNamespace(select_case=invalid))
    with pytest.raises(ValueError, match="scenario"):
        launcher.prepare(args)
    assert not args.output.exists()


def test_no_communication_preparation_needs_no_bucket_token(launcher, submission, monkeypatch):
    args, _ = submission
    args.communication = 'off'
    args.spec = ROOT / 'experiments/agent-escape/spec-router-pi.yaml'
    loader = importlib.util.spec_from_file_location('ablation_run', args.spec.with_name('run.py'))
    runner = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(runner)
    monkeypatch.setattr(runner, 'select_case', lambda *a: None)
    monkeypatch.setattr(launcher, 'load_launcher', lambda: runner)
    monkeypatch.delenv('HF_TOKEN')
    output = launcher.prepare(args)
    import yaml
    spec = yaml.safe_load((output / 'spec.yaml').read_text())
    assert spec.get('comms') is None
    assert spec['types']['worker']['comms'] is False
    assert launcher.read_secrets(output / 'secrets.json') == {'LOCAL_KEY': 'private-server-token'}
    assert launcher.read_json(output / 'config.json')['communication'] == 'off'
