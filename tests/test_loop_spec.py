"""`loop:` in the spec (docs/loop.md): validation, the `while` alias on the wire, prompt files
inlined by `legion launch`, and world-backed settings accepted by the model but refused at apply."""

from __future__ import annotations

import pytest
import yaml
from fastapi.testclient import TestClient
from pydantic import ValidationError
from typer.testing import CliRunner

from legion import cli
from legion.models import (
    LoopSpec,
    SwarmSpec,
    agent_cap,
    load_spec_yaml,
    loop_refusals,
    worker_type_spec,
)

from .conftest import TOKEN, make_spec, wire


def spec_with(loop: dict, **type_kw) -> SwarmSpec:
    return make_spec(researcher={"replicas": 1, "loop": loop, **type_kw})


def test_loop_fields_and_the_while_alias_round_trip():
    s = spec_with({"prompt": "r{round}", "every": "120s",
                   "on": ["message_board.mention", "message_board.thread"],
                   "while": {"until": "2h", "max_rounds": 200, "budget_usd": 12, "agent_usd": 3.6},
                   "final": {"prompt": "fin", "reserve_usd": 0.6}})
    lp = s.types["researcher"].loop
    assert lp.every_s == 120 and lp.while_.max_rounds == 200 and lp.final.reserve_usd == 0.6
    wire = s.model_dump(mode="json")["types"]["researcher"]["loop"]
    assert "while" in wire and "while_" not in wire and wire["while"]["until"] == "2h"
    assert SwarmSpec.model_validate_json(s.model_dump_json()) == s
    assert LoopSpec.model_validate({"prompt": "x"}).every == "turn_end"  # the default
    assert loop_refusals(s) == []


def test_f3_keys_stay_off_the_wire_unless_set_and_never_reach_workers():
    plain = spec_with({"prompt": "x", "while": {"until": "1h"}}).model_dump(mode="json")
    lp = plain["types"]["researcher"]["loop"]
    assert "start_prompt" not in lp and set(lp["while"]) == {
        "until", "max_rounds", "budget_usd", "agent_usd", "agent_output_tokens"}
    s = spec_with({"prompt": "x", "start_prompt": "go",
                   "while": {"until": "1h", "until_from": "first_turn", "quiet": "45s"}})
    assert SwarmSpec.model_validate_json(s.model_dump_json()) == s
    w = worker_type_spec(s.types["researcher"]).model_dump(mode="json")["loop"]
    assert "start_prompt" not in w and "quiet" not in w["while"] and "until_from" not in w["while"]


@pytest.mark.parametrize("loop, msg", [
    ({}, "loop needs prompt or prompt_file"),
    ({"prompt": "x", "every": "often"}, "bad duration"),
    ({"prompt": "x", "every": "0s"}, "duration > 0"),
    ({"prompt": "x", "on": ["slack.mention"]}, "expected message_board.<name>"),
    ({"prompt": "x", "while": {"max_rounds": 0}}, ">= 1"),
    ({"prompt": "x", "while": {"until": "soon"}}, "bad until"),
    ({"prompt": "x", "while": {"agent_usd": -1}}, ">= 0"),
    ({"prompt": "x", "while": {"agent_output_tokens": 0}}, ">= 1"),
    ({"prompt": "x", "while": {"rounds": 3}}, "Extra inputs"),
    ({"prompt": "x", "while": {"quiet": "0s"}}, "quiet must be a duration > 0"),
    ({"prompt": "x", "while": {"until_from": "first_turn"}}, "needs until as a duration"),
    ({"prompt": "x", "while": {"until": "2026-10-28T20:00Z", "until_from": "first_turn"}},
     "needs until as a duration"),
    ({"prompt": "x", "while": {"until": "1h", "until_from": "queue"}}, "'launch' or 'first_turn'"),
    ({"prompt": "x", "final": {"reserve_usd": 1}}, "final needs prompt"),
    ({"prompt": "x", "final": {"prompt": "f", "reserve_usd": -1}}, ">= 0"),
    ({"prompt": "x", "while": {"agent_usd": {"glm": 2.1}}}, "no entry for model 'm'"),
    ({"prompt": "x", "wake": {"min_gap": "soon"}}, "bad duration"),
    ({"prompt": "x", "wake": {"window": "0s"}}, "window must be a duration > 0"),
    ({"prompt": "x", "wake": {"max": -1}}, ">= 0"),
    ({"prompt": "x", "wake": {"gap": "5s"}}, "Extra inputs"),
    ({"prompt": "x", "every": "rounds"}, "bad duration"),
    ({"prompt": "x", "while": {"score_lt": {"c": 1}}}, "Extra inputs"),
])
def test_bad_loops_are_rejected(loop, msg):
    with pytest.raises(ValidationError, match=msg):
        spec_with(loop)


def test_agent_usd_by_model():
    s = spec_with({"prompt": "x", "while": {"agent_usd": {"m": 3.6, "glm": 2.1}}})
    assert agent_cap(s.types["researcher"]) == 3.6
    assert agent_cap(spec_with({"prompt": "x"}).types["researcher"]) is None


def test_controller_refuses_what_it_cannot_run(app):
    c = TestClient(app, headers={"Authorization": f"Bearer {TOKEN}"})
    s = spec_with({"prompt": "x", "on": ["world.event:claim"]})
    r = c.put("/swarms/demo", json=s.model_dump(mode="json"))
    assert r.status_code == 400 and "needs a world (not in this version)" in r.text
    s = spec_with({"prompt_file": "round.md"})  # sent without `legion launch`'s inlining
    r = c.put("/swarms/demo", json=s.model_dump(mode="json"))
    assert r.status_code == 400 and "not inlined" in r.text


def test_legion_apply_inlines_prompt_files_and_refuses_world_settings(tmp_path, monkeypatch, app, store):
    wire(monkeypatch, app)
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "round.md").write_text("Round {round} ({why}).\n")
    (tmp_path / "prompts" / "final.md").write_text("Final: submit.\n")
    raw = make_spec(researcher={"replicas": 1}).model_dump(mode="json", exclude_none=True)
    raw["types"]["researcher"]["loop"] = {
        "prompt_file": "prompts/round.md", "every": "30s", "on": ["message_board.mention", "message_board.channel"],
        "final": {"prompt_file": "prompts/final.md", "reserve_usd": 0.5}}
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["launch", str(f), "--skip-preflight"])
    assert r.exit_code == 0, r.output
    assert "loop.on message_board.mention: no feeder: the run has no message_board resource" in r.output
    lp = store.get_swarm("demo")["spec"].types["researcher"].loop
    assert lp.prompt == "Round {round} ({why}).\n" and lp.final.prompt == "Final: submit.\n"

    raw["types"]["researcher"]["loop"]["on"] = ["world.event:claim"]
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["launch", str(f), "--skip-preflight"])
    assert r.exit_code == 1
    assert "refused: types.researcher.loop.on: world.event:claim needs a world" in r.output
    raw["types"]["researcher"]["loop"]["on"] = ["message_board.mention"]
    raw["types"]["researcher"]["loop"]["prompt_file"] = "prompts/missing.md"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(cli.app, ["launch", str(f), "--skip-preflight"])
    assert r.exit_code == 1 and "missing.md not found" in r.output


def test_local_run_checks_loops_like_apply(tmp_path, monkeypatch):
    """The local run (legion.controller.localrun) runs the same CLI loop checks as `legion launch` (`_check_loops`): the `on:`
    warning, and refusals on the CLI before anything starts."""
    from legion.controller import localrun

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
                                          "on": ["message_board.mention"]}
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(localrun.app, [str(f), "--skip-preflight"])
    assert r.exit_code == 0, r.output
    assert "loop.on message_board.mention: no feeder" in r.output and len(started) == 1

    raw["types"]["researcher"]["loop"]["on"] = ["world.event:claim"]
    f.write_text(yaml.safe_dump(raw))
    r = CliRunner().invoke(localrun.app, [str(f), "--skip-preflight"])
    assert r.exit_code == 1 and len(started) == 1  # refused on the CLI: no run started
    assert "refused: types.researcher.loop.on: world.event:claim needs a world" in r.output


def test_spec_yaml_keeps_yes_no_on_off_as_strings():
    raw = load_spec_yaml("on: [a]\noff: 1\nyes: no\nflag: true\nOther: False\n")
    assert raw == {"on": ["a"], "off": 1, "yes": "no", "flag": True, "Other": False}


def test_wake_defaults():
    lp = LoopSpec.model_validate({"prompt": "x"})
    assert lp.wake.model_dump() == {"min_gap": "30s", "max": 10, "window": None,
                                    "counts_as_round": False}
    lp = LoopSpec.model_validate({"prompt": "x", "wake": {"min_gap": "0s", "max": None, "window": "10m",
                                                          "counts_as_round": True}})
    assert (lp.wake.max, lp.wake.window) == (None, "10m")


def test_until_takes_a_duration_or_a_clock_time():
    from datetime import UTC, datetime

    from legion.models import until_deadline
    t = datetime(2026, 10, 28, 20, tzinfo=UTC).timestamp()
    assert until_deadline("2h", 1000.0) == 8200.0
    for v in ("2026-10-28T20:00Z", "2026-10-28T20:00", "2026-10-28T22:00+02:00"):
        assert until_deadline(v, 5.0) == t  # naive = UTC; started_at is ignored
        assert LoopSpec.model_validate({"prompt": "x", "while": {"until": v}}).while_.until == v
    for bad in ("soon", "2026-13-40", "25"):
        with pytest.raises(ValidationError):
            LoopSpec.model_validate({"prompt": "x", "while": {"until": bad}})


def test_until_accepts_unquoted_yaml_times():
    from datetime import UTC, datetime

    import yaml

    from legion.models import until_deadline
    for text, t in (("2026-10-28T20:00:00Z", datetime(2026, 10, 28, 20, tzinfo=UTC)),
                    ("2026-10-28", datetime(2026, 10, 28, tzinfo=UTC))):
        w = yaml.safe_load(f"prompt: x\nwhile: {{until: {text}}}\n")
        assert not isinstance(w["while"]["until"], str)  # YAML parsed it
        until = LoopSpec.model_validate(w).while_.until
        assert until_deadline(until, 5.0) == t.timestamp()
