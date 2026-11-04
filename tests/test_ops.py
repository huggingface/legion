"""flock.ops: `flock launch` / `flock update` as library calls (no printing, no typer)."""

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

from flock import cli, ops
from flock.client import Client, ClientError
from flock.current import write_current
from flock.ops import SpecError, launch, read_spec, update

from .conftest import make_spec, write_spec


def _folder(tmp_path) -> str:
    return write_spec(tmp_path / "spec.yaml", make_spec())


def test_update_refuses_an_unknown_replicas_mode(api, tmp_path):
    f = _folder(tmp_path)
    with pytest.raises(SpecError, match="use keep or spec"):
        update(api, read_spec(f), f, replicas="all")


def test_a_failed_call_after_the_preflight_keeps_what_was_done(api, tmp_path, monkeypatch):
    f = _folder(tmp_path)

    def conflict(*a, **k):
        raise ClientError("error 409: exists", 409, '{"detail": "exists"}')

    monkeypatch.setattr(Client, "create", conflict)
    with pytest.raises(SpecError) as e:
        launch(api, read_spec(f), f, preflight=True)
    assert e.value.lines == ["error 409: exists"] and e.value.done["preflight"]["ok"]
    r = CliRunner().invoke(cli.app, ["launch", f, "--preflight"])  # the CLI still shows it
    assert r.exit_code == 1 and "preflight: no provider-backed models" in r.output
    assert r.output.rstrip().endswith("error 409: exists")


def test_start_local_refuses_another_swarms_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("FLOCK_URL", raising=False)
    monkeypatch.delenv("FLOCK_CURRENT", raising=False)
    write_current(tmp_path / ".flock", {"url": "http://127.0.0.1:1", "token": "t",
                                        "pid": os.getpid(), "swarm": "other"})
    with pytest.raises(SpecError, match="a local run of other is up"):
        ops.start_local(make_spec(), "spec.yaml")
    assert ops.start_local(make_spec("other"), "spec.yaml") is None  # already serves it
    monkeypatch.setenv("FLOCK_URL", "http://elsewhere:1")
    assert ops.start_local(make_spec(), "spec.yaml") is None  # an explicit controller
