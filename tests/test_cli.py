import json

import yaml
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from flock import cli

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


# --------------------------------------------------------------------------- providers


def _provider_app(store, clock, tool_call: bool):
    import httpx

    from flock.controller.api import create_app

    from .test_providers_controller import fake_provider

    fake, seen = fake_provider(tool_call=tool_call)
    app = create_app(store, TOKEN, clock=clock, secret_source={"VLLM_API_KEY": "k"},
                     preflight_transport=httpx.ASGITransport(app=fake))
    return app, seen


PROVIDER_SPEC = {
    "swarm": "op", "namespace": "ns", "bucket": "hf://buckets/ns/b",
    "providers": {"gpu1": {"type": "openai_compat", "base_url": "http://fake/v1",
                           "api_key_env": "VLLM_API_KEY", "self_hosted": True}},
    "models": {"q": {"model": "gpu1:Qwen/Q", "extra": {"chat_template_kwargs": {"enable_thinking": False}}}},
    "types": {"worker": {"harness": "codex", "model": "q", "replicas": 1}},
}


def test_apply_runs_preflight_and_refuses_on_failure(tmp_path, monkeypatch, store, clock):
    app, seen = _provider_app(store, clock, tool_call=False)
    _wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 1 and "preflight failed" in r.output and "FAIL" in r.output
    assert "`extra` is kept in the spec but not sent" in r.output
    assert store.get_swarm("op") is None and len(seen) == 1
    r = runner.invoke(cli.app, ["apply", "--skip-preflight", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert store.get_swarm("op") is not None and len(seen) == 1
    assert "provider gpu1 (openai_compat self-hosted): working 0, types worker" in r.output
    r = runner.invoke(cli.app, ["preflight", str(spec_file)])
    assert r.exit_code == 1 and "gpu1:Qwen/Q" in r.output


def test_status_prints_zero_usage_warning(monkeypatch):
    st = {"swarm": "op", "agents_total": 1, "workers_total": 1, "pending_prompts": 0, "cost_usd": 0,
          "types": {}, "providers": {"hf": {"name": "hf", "type": "openai_compat", "self_hosted": False,
                                            "types": ["w"], "working": 0, "max_working": None,
                                            "zero_usage_agents": 2}}}
    monkeypatch.setattr(cli, "_call", lambda *a, **k: {"status": st})
    r = runner.invoke(cli.app, ["status", "op"])
    assert r.exit_code == 0 and "2 agents finished turns with zero reported tokens" in r.output


def test_status_prints_launch_errors(monkeypatch):
    def t(name, health, err):
        return {"name": name, "health": health, "replicas": 1, "agents_by_state": {},
                "workers_by_state": {}, "cost_usd": 0, "launch_error": err}
    st = {"swarm": "op", "agents_total": 2, "workers_total": 2, "pending_prompts": 0, "cost_usd": 0,
          "types": {"a": t("a", "launch_refused", "BadRequestError: bad flavor"),
                    "b": t("b", "ok", "HfHubHTTPError: 503")}}
    monkeypatch.setattr(cli, "_call", lambda *a, **k: {"status": st})
    r = runner.invoke(cli.app, ["status", "op"])
    assert "type a: launch refused, not retrying until the next apply: BadRequestError" in r.output
    assert "type b: last launch error, retrying: HfHubHTTPError: 503" in r.output


def test_apply_with_passing_preflight(tmp_path, monkeypatch, store, clock):
    app, seen = _provider_app(store, clock, tool_call=True)
    _wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["preflight", str(spec_file)])
    assert r.exit_code == 0 and "ok" in r.output and "responses" in r.output
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert store.get_swarm("op")["spec"].types["worker"].provider_name == "gpu1"
    assert len(seen) == 4  # two probes, each a tool call plus a follow-up turn
    # native-only specs skip preflight unless asked
    native = tmp_path / "native.yaml"
    native.write_text(yaml.safe_dump(make_spec().model_dump(mode="json")))
    r = runner.invoke(cli.app, ["apply", str(native)])
    assert r.exit_code == 0 and len(seen) == 4
    r = runner.invoke(cli.app, ["apply", "--preflight", str(native)])
    assert r.exit_code == 0 and "no provider-backed models" in r.output


def test_unverified_models_warn_and_apply_unless_strict(tmp_path, monkeypatch, store, clock):
    import httpx

    from flock.controller.api import create_app

    from .test_providers_controller import fake_provider

    fake, seen = fake_provider()
    app = create_app(store, TOKEN, clock=clock, secret_source={},  # no provider key on the controller
                     preflight_transport=httpx.ASGITransport(app=fake))
    _wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["preflight", str(spec_file)])
    assert r.exit_code == 0 and "unverified" in r.output and "VLLM_API_KEY is not set" in r.output
    assert runner.invoke(cli.app, ["preflight", "--strict", str(spec_file)]).exit_code == 1
    r = runner.invoke(cli.app, ["apply", "--strict", str(spec_file)])
    assert r.exit_code == 1 and "preflight failed" in r.output and store.get_swarm("op") is None
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert "warning: 1 model(s) unverified" in r.output and store.get_swarm("op") is not None
    assert not seen  # nothing was sent without a key
    # skipping preflight is never silent either
    r = runner.invoke(cli.app, ["apply", "--skip-preflight", str(spec_file)])
    assert r.exit_code == 0 and "gpu1:Qwen/Q: unverified (preflight skipped)" in r.output
    r = runner.invoke(cli.app, ["apply", "--skip-preflight", "--strict", str(spec_file)])
    assert r.exit_code == 1 and "--strict: not applying unverified models" in r.output


def test_apply_reports_spec_errors_by_field(tmp_path, monkeypatch, app):
    _wire(monkeypatch, app)
    bad = {**PROVIDER_SPEC, "types": {"writer": {"harness": "claude", "model": "q"}}}
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(bad))
    r = runner.invoke(cli.app, ["apply", str(spec_file)])
    assert r.exit_code == 1
    assert "types.writer.model: harness 'claude' cannot use provider 'gpu1'" in r.output


def test_warn_extra_only_for_harnesses_that_cannot_send_it(monkeypatch, capsys):
    from flock.models import SwarmSpec

    spec = SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b",
        "providers": {"p": {"type": "openai_compat", "base_url": "http://x/v1", "api_key_env": "K",
                            "self_hosted": True},
                      "a": {"type": "anthropic", "api_key_env": "A", "self_hosted": True}},
        "models": {"m": {"model": "p:m", "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
                   "plain": {"model": "p:m"},
                   "am": {"model": "a:m", "extra": {"x": 1}}},
        "types": {"cx": {"harness": "codex", "model": "m"}, "pi": {"harness": "pi", "model": "m"},
                  "cl": {"harness": "claude", "model": "am"}, "nox": {"harness": "codex", "model": "plain"}}})
    cli._warn_extra(spec)
    err = capsys.readouterr().err
    assert all(f"type {t}: `extra`" in err for t in ("cx", "pi", "cl")) and "nox" not in err
    # the llm harness sends `extra` as body fields: never warned (PR #7 adds the harness)
    assert "llm" not in cli.EXTRA_NOT_SENT
    monkeypatch.setattr(cli, "EXTRA_NOT_SENT", frozenset({"claude"}))
    cli._warn_extra(spec)
    err = capsys.readouterr().err
    assert "type cl:" in err and "type cx:" not in err
