from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from flock import cli, ops
from flock.client import Client

from .conftest import HB, TOKEN, make_spec, wire, write_spec

runner = CliRunner()


def test_update_replicas_agents_workers_remove(tmp_path, monkeypatch, app, store, tick):
    wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    write_spec(spec_file, make_spec())
    assert runner.invoke(cli.app, ["launch", str(spec_file)]).exit_code == 0
    spec_file.write_text(spec_file.read_text().replace("replicas: 3", "replicas: 5"))
    r = runner.invoke(cli.app, ["update", str(spec_file), "--replicas", "spec"])
    assert r.exit_code == 0, r.output
    tick()
    r = runner.invoke(cli.app, ["status", "demo", "--type", "researcher"])
    assert r.exit_code == 0 and r.output.count("researcher-000") == 5, r.output
    r = runner.invoke(cli.app, ["status", "demo", "--json"])
    assert "demo-researcher-w0002" in [w["id"] for w in json.loads(r.output)["workers"]]
    r = runner.invoke(cli.app, ["status", "--agent", "researcher-0004"])
    assert json.loads(r.output)["worker"] == "demo-researcher-w0002"
    r = runner.invoke(cli.app, ["status", "researcher-0004", "--events"])
    assert r.exit_code == 0 and r.output == ""
    r = runner.invoke(cli.app, ["stop", "demo", "--delete", "--yes"])
    assert r.exit_code == 0 and json.loads(r.output)["removing"] is True


def test_apply_inlines_system_prompt_file(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "r.md").write_text("You keep notes.md.")
    spec = make_spec(researcher={"replicas": 1, "system_prompt_file": "prompts/r.md"})
    spec_file = tmp_path / "spec.yaml"
    write_spec(spec_file, spec)
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
    assert r.exit_code == 0, r.output
    stored = store.get_swarm("demo")["spec"].types["researcher"]
    assert stored.system_prompt == "You keep notes.md."
    assert stored.system_prompt_file == "prompts/r.md"

    bad = make_spec(researcher={"replicas": 1, "system_prompt_file": "prompts/nope.md"})
    write_spec(spec_file, bad)
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
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
    "types": {"worker": {**HB, "model": "q", "replicas": 1}},
}


def test_apply_runs_preflight_and_refuses_on_failure(tmp_path, monkeypatch, store, clock):
    app, seen = _provider_app(store, clock, tool_call=False)
    wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
    assert r.exit_code == 1 and "preflight failed" in r.output and "FAIL" in r.output
    assert store.get_swarm("op") is None and len(seen) == 1
    r = runner.invoke(cli.app, ["launch", "--skip-preflight", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert store.get_swarm("op") is not None and len(seen) == 1
    assert "provider gpu1 (openai_compat self-hosted): working 0, types worker" in r.output
    r = runner.invoke(cli.app, ["update", "--check", str(spec_file)])
    assert r.exit_code == 1 and "preflight gpu1:Qwen/Q: FAIL" in r.output


def _fake_status(monkeypatch, st):
    monkeypatch.setattr(Client, "_call", lambda _, m, path, **k: [] if path == "/workers"
                        else {"status": st})
    monkeypatch.setattr(Client, "health", lambda *_: None)


def test_status_prints_zero_usage_warning(monkeypatch):
    st = {"swarm": "op", "agents_total": 1, "workers_total": 1, "pending_prompts": 0, "cost_usd": 0,
          "types": {}, "providers": {"hf": {"name": "hf", "type": "openai_compat", "self_hosted": False,
                                            "types": ["w"], "working": 0, "max_working": None,
                                            "zero_usage_agents": 2}}}
    _fake_status(monkeypatch, st)
    r = runner.invoke(cli.app, ["status", "op"])
    assert r.exit_code == 0 and "2 agents finished turns with zero reported tokens" in r.output


def test_status_prints_launch_errors(monkeypatch):
    def t(name, health, err):
        return {"name": name, "health": health, "replicas": 1, "agents_by_state": {},
                "workers_by_state": {}, "cost_usd": 0, "launch_error": err}
    st = {"swarm": "op", "agents_total": 2, "workers_total": 2, "pending_prompts": 0, "cost_usd": 0,
          "types": {"a": t("a", "launch_refused", "BadRequestError: bad flavor"),
                    "b": t("b", "ok", "HfHubHTTPError: 503")}}
    _fake_status(monkeypatch, st)
    r = runner.invoke(cli.app, ["status", "op"])
    assert "type a: launch refused, not retrying until the next apply: BadRequestError" in r.output
    assert "type b: last launch error, retrying: HfHubHTTPError: 503" in r.output


def test_apply_with_passing_preflight(tmp_path, monkeypatch, store, clock):
    app, seen = _provider_app(store, clock, tool_call=True)
    wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
    assert r.exit_code == 0 and "ok" in r.output and "chat_completions" in r.output, r.output
    assert store.get_swarm("op")["spec"].types["worker"].provider_name == "gpu1"
    assert len(seen) == 2  # one probe: a tool call plus a follow-up turn
    r = runner.invoke(cli.app, ["update", "--check", str(spec_file)])  # was `flock preflight`
    assert r.exit_code == 0 and "preflight gpu1:Qwen/Q: ok" in r.output and len(seen) == 4
    # a spec with no types has no model to probe
    empty = tmp_path / "empty.yaml"
    write_spec(empty, make_spec("demo2").model_copy(update={"types": {}}))
    r = runner.invoke(cli.app, ["launch", "--preflight", str(empty)])
    assert r.exit_code == 0 and "no provider-backed models" in r.output and len(seen) == 4, r.output


def test_unverified_models_warn_and_apply_unless_strict(tmp_path, monkeypatch, store, clock):
    import httpx

    from flock.controller.api import create_app

    from .test_providers_controller import fake_provider

    fake, seen = fake_provider()
    app = create_app(store, TOKEN, clock=clock, secret_source={},  # no provider key on the controller
                     preflight_transport=httpx.ASGITransport(app=fake))
    wire(monkeypatch, app)
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(PROVIDER_SPEC))
    r = runner.invoke(cli.app, ["launch", "--strict", str(spec_file)])
    assert r.exit_code == 1 and "preflight failed" in r.output and store.get_swarm("op") is None
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
    assert r.exit_code == 0, r.output
    assert "warning: 1 model(s) unverified" in r.output and "VLLM_API_KEY is not set" in r.output
    assert store.get_swarm("op") is not None
    assert not seen  # nothing was sent without a key
    # skipping preflight is never silent either
    store.delete_swarm("op")
    r = runner.invoke(cli.app, ["launch", "--skip-preflight", str(spec_file)])
    assert r.exit_code == 0 and "gpu1:Qwen/Q: unverified (preflight skipped)" in r.output
    store.delete_swarm("op")
    r = runner.invoke(cli.app, ["launch", "--skip-preflight", "--strict", str(spec_file)])
    assert r.exit_code == 1 and "--strict: not launching unverified models" in r.output


def test_apply_reports_spec_errors_by_field(tmp_path, monkeypatch, app):
    wire(monkeypatch, app)
    bad = {**PROVIDER_SPEC, "types": {"writer": {**HB, "model": "native"}}}
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(bad))
    r = runner.invoke(cli.app, ["launch", str(spec_file)])
    assert r.exit_code == 1
    assert "types.writer.model: harness 'harbor' needs a provider-backed model" in r.output


def test_warn_extra_that_is_not_sent():
    from flock.models import SwarmSpec

    spec = SwarmSpec.model_validate({
        "swarm": "s", "namespace": "n", "bucket": "b",
        "providers": {"p": {"type": "openai_compat", "base_url": "http://x/v1", "api_key_env": "K",
                            "self_hosted": True}},
        "models": {"m": {"model": "p:m", "extra": {"chat_template_kwargs": {"enable_thinking": False}}},
                   "plain": {"model": "p:m"}},
        "types": {"hb": {"harness": "harbor", "model": "m", "harbor": {"task": {}, "config": {}}},
                  "nox": {"harness": "harbor", "model": "plain", "harbor": {"task": {}, "config": {}}}}})
    err = "\n".join(ops.extra_warnings(spec))
    assert "type hb: `extra`" in err and "nox" not in err


def test_apply_sends_its_git_source(tmp_path, monkeypatch, app, store):
    import subprocess

    wire(monkeypatch, app)
    monkeypatch.setenv("USER", "carlos")
    git = lambda *a: subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t",
                                     "-c", "user.email=t@t", *a], check=True, capture_output=True)
    git("init", "-q", "-b", "exp")
    spec_file = tmp_path / "spec.yaml"
    write_spec(spec_file, make_spec())
    git("add", ".")
    git("commit", "-qm", "spec")
    sha = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    assert runner.invoke(cli.app, ["launch", str(spec_file)]).exit_code == 0
    (gen1,) = store.spec_history("demo")
    assert gen1["source"] == {"by": "carlos", "git_sha": sha, "ref": "exp", "dirty": False}
    assert json.loads(ops.source_header(spec_file.parent / "x" / "spec.yaml")["X-Flock-Source"]) == {
        "by": "carlos"}  # not a repo (the folder does not exist)
    spec_file.write_text(spec_file.read_text().replace("replicas: 3", "replicas: 4"))
    assert json.loads(ops.source_header(spec_file)["X-Flock-Source"])["dirty"] is True


def test_status_shows_the_generation(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    monkeypatch.setenv("USER", "carlos")
    spec_file = tmp_path / "spec.yaml"
    write_spec(spec_file, make_spec())
    assert runner.invoke(cli.app, ["launch", str(spec_file)]).exit_code == 0
    spec_file.write_text(spec_file.read_text().replace("replicas: 3", "replicas: 5"))
    assert runner.invoke(cli.app, ["update", str(spec_file), "--replicas", "spec"]).exit_code == 0
    r = runner.invoke(cli.app, ["status", "demo"])
    assert r.exit_code == 0 and " gen 2" in r.output.splitlines()[1], r.output
    assert store.spec_history("demo")[1]["source"]["by"] == "carlos"


def test_hidden_world_app_and_plain_help():
    """run5-era Slurm world jobs call `flock world serve`; help is not re-wrapped by rich."""
    r = runner.invoke(cli.app, ["world", "serve", "--help"])
    assert r.exit_code == 0 and "module:Class" in r.output
    out = runner.invoke(cli.app, ["--help"]).output
    assert "  world " not in out and "╭" not in out
