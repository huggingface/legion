"""legion.ops: `legion launch` / `legion update` as library calls (no printing, no typer)."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from legion import cli, ops
from legion.client import Client, ClientError
from legion.current import write_current
from legion.ops import SpecError, launch, read_spec
from legion.worker.agents import VERIFY_WAIT

from .conftest import make_spec, write_spec


def _folder(tmp_path) -> str:
    return write_spec(tmp_path / "spec.yaml", make_spec())


def test_a_failed_call_after_the_preflight_keeps_what_was_done(api, tmp_path, monkeypatch):
    f = _folder(tmp_path)

    def conflict(*a, **k):
        raise ClientError("error 409: exists", 409, '{"detail": "exists"}')

    monkeypatch.setattr(Client, "create", conflict)
    with pytest.raises(SpecError) as e:
        launch(api, read_spec(f), f, preflight=True)
    assert e.value.lines == ["error 409: exists"] and e.value.done["preflight"]["ok"]
    r = CliRunner().invoke(cli.app, ["launch", f, "--preflight"])  # the CLI still shows it
    assert r.exit_code == 1 and "1 model(s) unverified" in r.output
    assert r.output.rstrip().endswith("error 409: exists")


def test_start_local_refuses_another_swarms_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEGION_URL", raising=False)
    monkeypatch.delenv("LEGION_CURRENT", raising=False)
    write_current(tmp_path / ".legion", {"url": "http://127.0.0.1:1", "token": "t",
                                        "pid": os.getpid(), "swarm": "other"})
    with pytest.raises(SpecError, match="a local run of other is up"):
        ops.start_local(make_spec(), "spec.yaml")
    assert ops.start_local(make_spec("other"), "spec.yaml") is None  # already serves it
    monkeypatch.setenv("LEGION_URL", "http://elsewhere:1")
    assert ops.start_local(make_spec(), "spec.yaml") is None  # an explicit controller


@pytest.mark.parametrize("loop", [None, {"prompt": "go"}])
def test_start_local_ends_a_looped_run_when_idle(tmp_path, monkeypatch, loop):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LEGION_URL", raising=False)
    monkeypatch.delenv("LEGION_CURRENT", raising=False)
    cmds = []

    def popen(cmd, **_):
        cmds.append(cmd)
        raise RuntimeError("stop here")

    monkeypatch.setattr(ops.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError):
        ops.start_local(make_spec(agent={"replicas": 1, "loop": loop}), "spec.yaml")
    assert (cmds[0][-4:-2] == ["--until", "idle"]) == (loop is not None)
    if loop is not None:  # the grace covers the Trials' verifier wait
        assert float(cmds[0][cmds[0].index("--grace") + 1]) > VERIFY_WAIT
