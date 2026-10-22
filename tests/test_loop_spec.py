"""`loop:` in the spec (docs/loop.md): validation, the `while` alias on the wire, prompt files
inlined by `flock apply`, and world-backed settings accepted by the model but refused at apply."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from flock import cli
from flock.models import LoopSpec, SwarmSpec, agent_cap, load_spec_yaml, loop_refusals

from .conftest import TOKEN, make_spec


def spec_with(loop: dict, **type_kw) -> SwarmSpec:
    return make_spec(researcher={"replicas": 1, "loop": loop, **type_kw})


def test_loop_fields_and_the_while_alias_round_trip():
    s = spec_with({"prompt": "r{round}", "first": "boot", "every": "120s", "stagger": "4s",
                   "on": ["comms.mention", "comms.thread"],
                   "while": {"until": "2h", "max_rounds": 200, "budget_usd": 12, "agent_usd": 3.6,
                             "idle_rounds": 3},
                   "final": {"prompt": "fin", "reserve_usd": 0.6, "add_tools": ["submit_final"]}})
    lp = s.types["researcher"].loop
    assert lp.every_s == 120 and lp.while_.max_rounds == 200 and lp.final.reserve_usd == 0.6
    wire = s.model_dump(mode="json")["types"]["researcher"]["loop"]
    assert "while" in wire and "while_" not in wire and wire["while"]["until"] == "2h"
    assert SwarmSpec.model_validate_json(s.model_dump_json()) == s
    assert LoopSpec.model_validate({"prompt": "x"}).every == "turn_end"  # the default
    assert loop_refusals(s) == []


@pytest.mark.parametrize("loop, msg", [
    ({}, "loop needs prompt or prompt_file"),
    ({"prompt": "x", "every": "often"}, "bad duration"),
    ({"prompt": "x", "every": "0s"}, "duration > 0"),
    ({"prompt": "x", "on": ["board.mention"]}, "expected comms.<name>"),
    ({"prompt": "x", "while": {"max_rounds": 0}}, ">= 1"),
    ({"prompt": "x", "while": {"until": "soon"}}, "bad duration"),
    ({"prompt": "x", "while": {"agent_usd": -1}}, ">= 0"),
    ({"prompt": "x", "while": {"rounds": 3}}, "Extra inputs"),
    ({"prompt": "x", "final": {"reserve_usd": 1}}, "final needs prompt"),
    ({"prompt": "x", "final": {"prompt": "f", "reserve_usd": -1}}, ">= 0"),
    ({"prompt": "x", "while": {"agent_usd": {"glm": 2.1}}}, "no entry for model 'claude-opus-5-5'"),
    ({"prompt": "x", "wake": {"min_gap": "soon"}}, "bad duration"),
    ({"prompt": "x", "wake": {"window": "0s"}}, "window must be a duration > 0"),
    ({"prompt": "x", "wake": {"max": -1}}, ">= 0"),
    ({"prompt": "x", "wake": {"gap": "5s"}}, "Extra inputs"),
    ({"prompt": "x", "stop_when": {"all_done": True}}, "Extra inputs"),
])
def test_bad_loops_are_rejected(loop, msg):
    with pytest.raises(ValidationError, match=msg):
        spec_with(loop)


def test_agent_usd_by_model():
    s = spec_with({"prompt": "x", "while": {"agent_usd": {"claude-opus-5-5": 3.6, "glm": 2.1}}})
    assert agent_cap(s.types["researcher"]) == 3.6
    assert agent_cap(spec_with({"prompt": "x"}).types["researcher"]) is None


def test_world_backed_settings_are_accepted_but_refused():
    s = spec_with({"prompt": "x", "every": "rounds", "on": ["world.event:claim", "comms.mention"],
                   "context": ["world.state", "comms.recent:20"],
                   "while": {"score_lt": {"coverage_pct": 100}}})
    assert s.types["researcher"].loop.while_.score == {"lt": {"coverage_pct": 100}}
    assert loop_refusals(s) == [
        "types.researcher.loop.every: rounds needs a world (not in this version)",
        "types.researcher.loop.while.score_lt needs a world (not in this version)",
        "types.researcher.loop.on: world.event:claim needs a world (not in this version)",
        "types.researcher.loop.context: world.state needs a world (not in this version)",
        "types.researcher.loop.context: comms.recent:20 is not in this version",
    ]


def test_controller_refuses_what_it_cannot_run(app):
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    s = spec_with({"prompt": "x", "while": {"score_lt": {"coverage_pct": 100}}})
    r = c.put("/swarms/demo", json=s.model_dump(mode="json"))
    assert r.status_code == 400 and "needs a world (not in this version)" in r.text
    s = spec_with({"prompt_file": "round.md"})  # sent without `flock apply`'s inlining
    r = c.put("/swarms/demo", json=s.model_dump(mode="json"))
    assert r.status_code == 400 and "not inlined" in r.text


def test_flock_apply_inlines_prompt_files_and_refuses_world_settings(tmp_path, monkeypatch, app, store):
    monkeypatch.setattr(cli, "client", lambda: TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"}))
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "round.md").write_text("Round {round} ({why}).\n")
    (tmp_path / "prompts" / "final.md").write_text("Final: submit.\n")
    raw = make_spec(researcher={"replicas": 1}).model_dump(mode="json", exclude_none=True)
    raw["types"]["researcher"]["loop"] = {
        "prompt_file": "prompts/round.md", "every": "30s", "on": ["comms.mention"],
        "final": {"prompt_file": "prompts/final.md", "reserve_usd": 0.5}}
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["apply", str(f), "--skip-preflight"])
    assert r.exit_code == 0, r.output
    assert "loop.on comms.mention: no comms feeder: the type is not on the board" in r.output
    lp = store.get_swarm("demo")["spec"].types["researcher"].loop
    assert lp.prompt == "Round {round} ({why}).\n" and lp.final.prompt == "Final: submit.\n"

    raw["types"]["researcher"]["loop"]["while"] = {"score_ge": {"coverage_pct": 100}}
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["apply", str(f), "--skip-preflight"])
    assert r.exit_code == 1
    assert "refused: types.researcher.loop.while.score_ge needs a world (not in this version)" in r.output
    raw["types"]["researcher"]["loop"]["prompt_file"] = "prompts/missing.md"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["apply", str(f), "--skip-preflight"])
    assert r.exit_code == 1 and "missing.md not found" in r.output


def test_flock_run_checks_loops_like_apply(tmp_path, monkeypatch):
    """`flock run` runs the same CLI loop checks as `flock apply` (`_check_loops`): the `on:`
    warning, and refusals on the CLI before anything starts."""
    from flock.controller import localrun

    started: list = []

    async def fake_run_local(opts):
        started.append(opts)
        return localrun.RunResult(tmp_path, "idle", 0, {})

    monkeypatch.setattr(localrun, "run_local", fake_run_local)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "round.md").write_text("Round {round}.\n")
    raw = make_spec(researcher={"replicas": 1}).model_dump(mode="json", exclude_none=True)
    raw["compute"] = {"kind": "local"}
    raw["types"]["researcher"]["loop"] = {"prompt_file": "prompts/round.md", "every": "30s",
                                          "on": ["comms.mention"]}
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["run", str(f), "--skip-preflight"])
    assert r.exit_code == 0, r.output
    assert "loop.on comms.mention: no comms feeder" in r.output and len(started) == 1

    raw["types"]["researcher"]["loop"]["while"] = {"score_ge": {"coverage_pct": 100}}
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["run", str(f), "--skip-preflight"])
    assert r.exit_code == 1 and len(started) == 1  # refused on the CLI: no run started
    assert "refused: types.researcher.loop.while.score_ge needs a world (not in this version)" in r.output


DOCS = Path(__file__).parent.parent / "docs" / "loop.md"
HEADER = """\
swarm: demo
namespace: ns
bucket: hf://buckets/ns/sh
providers:
  hf: {type: openai_compat, base_url: https://router.huggingface.co/v1, api_key_env: HF_TOKEN,
       pricing: {"m/lfm": [0, 0, 0], "m/ds": [0, 0, 0], "m/glm": [0, 0, 0]}}
models: {lfm: {model: "hf:m/lfm"}, deepseek: {model: "hf:m/ds"}, glm: {model: "hf:m/glm"}}
"""


def _doc_example(heading: str) -> str:
    """The first ```yaml block after `heading` in docs/loop.md, verbatim."""
    rest = DOCS.read_text(encoding="utf-8").split(heading, 1)[1]
    return re.search(r"```yaml\n(.*?)```", rest, re.DOTALL).group(1)


@pytest.mark.parametrize("heading", ["## Spec", "### Eliza"])
def test_documented_examples_load_with_an_unquoted_on_key(heading, tmp_path):
    """`on:` is a YAML 1.1 boolean: `flock apply` must read it as the key "on" (defect 2)."""
    (tmp_path / "prompts").mkdir()
    for f in ("system.md", "round.md", "final.md"):
        (tmp_path / "prompts" / f).write_text(f)
    example = _doc_example(heading)
    assert re.search(r"^\s+on: \[comms\.", example, re.MULTILINE)  # unquoted, as documented
    (tmp_path / "spec.yaml").write_text(HEADER + example)
    spec = cli.load_spec(str(tmp_path / "spec.yaml"))
    for ts in spec.types.values():
        assert ts.loop.on and all(x.startswith("comms.") for x in ts.loop.on)


def test_spec_yaml_keeps_yes_no_on_off_as_strings():
    raw = load_spec_yaml("on: [a]\noff: 1\nyes: no\nflag: true\nOther: False\n")
    assert raw == {"on": ["a"], "off": 1, "yes": "no", "flag": True, "Other": False}


def test_wake_and_stop_when_defaults_and_refusal():
    lp = LoopSpec.model_validate({"prompt": "x"})
    assert lp.wake.model_dump() == {"min_gap": "30s", "max": 10, "window": None,
                                    "counts_as_round": False}
    assert lp.stop_when.all_agents_done is False
    lp = LoopSpec.model_validate({"prompt": "x", "wake": {"min_gap": "0s", "max": None, "window": "10m",
                                                          "counts_as_round": True},
                                  "stop_when": {"all_agents_done": True}})
    assert (lp.wake.max, lp.wake.window, lp.stop_when.all_agents_done) == (None, "10m", True)
    # `finished` is an argument of the llm harness's `done` tool
    s = spec_with({"prompt": "x", "stop_when": {"all_agents_done": True}})
    assert loop_refusals(s) == [
        "types.researcher.loop.stop_when.all_agents_done needs harness llm with the done tool"]
