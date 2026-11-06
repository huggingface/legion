# from shepherd f3970bc:tests/test_tools.py
import json
import os
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("harbor")

from harbor.trial.hooks import TrialEvent

from flock.harbor.instructions import build_agent_instructions
from flock.harbor.tools.bundle import build_tool_bundle
from flock.harbor.tools.runner import load_selected, validate_value
from flock.harbor.tools.setup import attach_tool_setup


@pytest.fixture
def tools(tmp_path):
    (tmp_path / "sample.py").write_text('''
def add(x: int, y: int = 2) -> dict:
    """Add two numbers."""
    return {"sum": x + y}

async def echo(values: list[str], label: str | None = None) -> dict:
    """Echo values and an optional label."""
    return {"values": values, "label": label}

def helper() -> dict:
    """This helper is not an exported tool."""
    return {}

TOOLS = {"add": add, "echo": echo}
''')
    (tmp_path / "empty.py").write_text("TOOLS = {}\n")
    (tmp_path / "unused.py").write_text(
        'raise RuntimeError("must not import unselected module")'
    )
    return tmp_path


@pytest.mark.parametrize(
    "selection",
    [
        ["sample.missing"],
        ["../sample.add"],
        ["sample.add", "sample.add"],
        ["sample.a*"],
        ["*.add"],
        ["missing.*"],
        ["empty.*"],
        ["sample.*", "sample.add"],
        ["sample.*", "sample.*"],
    ],
)
def test_invalid_selection_is_rejected(tools, selection):
    with pytest.raises(ValueError):
        build_tool_bundle(tools, selection, tools / "tools.tar.gz")


@pytest.mark.parametrize(
    "args", [{}, {"x": True}, {"x": "2"}, {"x": 2, "ctx": {}}, {"x": 2, "y": None}]
)
def test_arguments_are_validated_strictly(tools, args):
    _, definitions, _ = load_selected(tools, ["sample.add"])
    with pytest.raises(ValueError):
        validate_value(args, definitions["sample.add"]["parameters"])


def test_bundle_is_reproducible_and_runs_generic_async_tools(tools):
    first = build_tool_bundle(tools, ["sample.*"], tools / "first.tar.gz")
    second = build_tool_bundle(
        tools, ["sample.add", "sample.echo"], tools / "second.tar.gz"
    )
    assert first.sha256 == second.sha256
    assert list(first.manifest["tools"]) == ["sample.add", "sample.echo"]
    assert first.manifest["dependencies"] == {}
    destination = tools / "extracted"
    with tarfile.open(first.archive) as archive:
        assert "tools/unused.py" not in archive.getnames()
        archive.extractall(destination, filter="data")
    env = {**os.environ, "FLOCK_TOOLS_MANIFEST": str(destination / "manifest.json")}
    for name, args, expected in [
        ("sample.add", {"x": 3}, {"sum": 5}),
        ("sample.echo", {"values": ["hello"]}, {"values": ["hello"], "label": None}),
    ]:
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                str(destination / "bin/flock-tool"),
                name,
                "--json",
                json.dumps(args),
            ],
            env=env,
            text=True,
            capture_output=True,
            timeout=10,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(result.stdout) == expected
    agent_instructions = build_agent_instructions(
        "agent-001", {"agent-001": None}, first.manifest
    )
    assert "sample.add" in agent_instructions and "board.post" not in agent_instructions


async def test_setup_hook_is_generic_and_runs_once(tools):
    bundle = build_tool_bundle(tools, ["sample.add"], tools / "tools.tar.gz")
    callbacks = {}
    agent = SimpleNamespace(
        ensure_system_dependencies=AsyncMock(),
        exec_as_root=AsyncMock(),
        exec_as_agent=AsyncMock(),
        extra_env={"FLOCK_TOOLS_MANIFEST": "/opt/flock-tools/manifest.json"},
    )
    environment = SimpleNamespace(upload_file=AsyncMock())
    trial = SimpleNamespace(
        agent=agent,
        agent_environment=environment,
        add_hook=lambda event, callback: callbacks.update({event: callback}),
    )
    attach_tool_setup(trial, bundle)
    await callbacks[TrialEvent.AGENT_START](None)
    await callbacks[TrialEvent.AGENT_START](None)
    environment.upload_file.assert_awaited_once()
    agent.exec_as_agent.assert_awaited_once_with(
        environment, "flock-tool --list", env=agent.extra_env, timeout_sec=30
    )


async def test_setup_failure_propagates(tools):
    bundle = build_tool_bundle(tools, ["sample.add"], tools / "tools.tar.gz")
    callbacks = {}
    agent = SimpleNamespace(
        ensure_system_dependencies=AsyncMock(),
        exec_as_root=AsyncMock(),
        exec_as_agent=AsyncMock(side_effect=RuntimeError("missing dependency")),
        extra_env={},
    )
    trial = SimpleNamespace(
        agent=agent,
        agent_environment=SimpleNamespace(upload_file=AsyncMock()),
        add_hook=lambda event, callback: callbacks.update({event: callback}),
    )
    attach_tool_setup(trial, bundle)
    with pytest.raises(RuntimeError, match="missing dependency"):
        await callbacks[TrialEvent.AGENT_START](None)
    agent.exec_as_agent.assert_awaited_once()


def test_bundle_hash_ignores_dist_info_record(tools, tmp_path, monkeypatch):
    # RECORD hashes bin/ scripts whose shebang is the venv path: it differs per install path
    (tools / "dep.py").write_text('DEPENDENCIES = ["flockfakedep"]\ndef f() -> dict:\n    """f."""\n    return {}\nTOOLS = {"f": f}\n')
    hashes = []
    for i in range(2):
        site = tmp_path / f"venv{i}"
        info = site / "flockfakedep-1.0.dist-info"
        info.mkdir(parents=True)
        (site / "flockfakedep.py").write_text("X = 1\n")
        (info / "METADATA").write_text("Metadata-Version: 2.1\nName: flockfakedep\nVersion: 1.0\n")
        (info / "INSTALLER").write_text(["pip", "uv"][i] + "\n")
        (info / "RECORD").write_text(f"flockfakedep.py,,\n../../../bin/x,sha256={i},\n"
                                     "flockfakedep-1.0.dist-info/METADATA,,\n"
                                     "flockfakedep-1.0.dist-info/RECORD,,\n")
        monkeypatch.syspath_prepend(str(site))
        b = build_tool_bundle(tools, ["dep.*"], tools / f"b{i}.tar.gz")
        hashes.append(b.sha256)
    assert hashes[0] == hashes[1]
    with tarfile.open(b.archive) as archive:
        names = archive.getnames()
    assert "flockfakedep.py" in names and "flockfakedep-1.0.dist-info/METADATA" in names
    assert not {"flockfakedep-1.0.dist-info/RECORD", "flockfakedep-1.0.dist-info/INSTALLER"} & set(names)
