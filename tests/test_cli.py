from __future__ import annotations

import json

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from swarm_harness import cli

from .conftest import TOKEN, make_spec

runner = CliRunner()


def _wire(monkeypatch, app):
    monkeypatch.setattr(
        cli, "client", lambda: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    )


def test_apply_status_prompt(tmp_path, monkeypatch, app, store, tick):
    _wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(make_spec().model_dump(mode="json")))

    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert "demo" in r.output and "researcher" in r.output
    tick()

    r = runner.invoke(cli.app, ["status"])
    assert r.exit_code == 0, r.output
    assert "agents=3 workers=2" in r.output and "unknown:3" in r.output
    r = runner.invoke(cli.app, ["status", "demo"])
    assert r.exit_code == 0 and "researcher" in r.output

    r = runner.invoke(cli.app, ["prompt", "researcher-0001", "hello there"])
    assert r.exit_code == 0, r.output
    ack = json.loads(r.output)
    assert ack["agent"] == "researcher-0001" and ack["status"] == "pending"

    r = runner.invoke(cli.app, ["prompt", "nobody-0001", "x"])
    assert r.exit_code == 1


def test_scale_agents_workers_remove(tmp_path, monkeypatch, app, store, tick):
    _wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(make_spec().model_dump(mode="json")))
    assert runner.invoke(cli.app, ["apply", str(spec_file)]).exit_code == 0
    r = runner.invoke(cli.app, ["scale", "demo", "researcher", "5"])
    assert r.exit_code == 0, r.output
    tick()
    r = runner.invoke(cli.app, ["agents", "demo", "--type", "researcher"])
    assert r.exit_code == 0 and r.output.count("researcher-000") == 5
    r = runner.invoke(cli.app, ["workers", "demo"])
    assert r.exit_code == 0 and "demo-researcher-w0002" in r.output
    r = runner.invoke(cli.app, ["agent", "researcher-0004"])
    assert json.loads(r.output)["worker"] == "demo-researcher-w0002"
    r = runner.invoke(cli.app, ["events", "researcher-0004"])
    assert r.exit_code == 0 and r.output == ""
    r = runner.invoke(cli.app, ["remove", "demo"])
    assert r.exit_code == 0 and json.loads(r.output)["removing"] is True


def test_apply_inlines_system_prompt_file(tmp_path, monkeypatch, app, store):
    _wire(monkeypatch, app)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "r.md").write_text("You keep notes.md.")
    spec = make_spec(researcher={"replicas": 1, "system_prompt_file": "prompts/r.md"})
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec.model_dump(mode="json", exclude_none=True)))
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 0, r.output
    stored = store.get_swarm("demo")["spec"].types["researcher"]
    assert stored.system_prompt == "You keep notes.md."
    assert stored.system_prompt_file == "prompts/r.md"

    bad = make_spec(researcher={"replicas": 1, "system_prompt_file": "prompts/nope.md"})
    spec_file.write_text(yaml.safe_dump(bad.model_dump(mode="json", exclude_none=True)))
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 1 and "not found" in r.output
