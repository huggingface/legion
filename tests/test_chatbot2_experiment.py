"""experiments/chatbot-v2 end to end under a local run (flock.controller.localrun) (offline): the real CLI runs the
experiment's own spec, kickoff, protocols, tools and world.yaml with worlds/chatbot2, against
tests/fake_openai.py and tests/fake_msgboard.py. Only the provider, the board, the loop prompts
(scripted tool calls) and the loop timings are swapped: 8 agents in 4 types (2 teams).

Phase 1: each agent writes a package bot (bot/__main__.py + a module), greps it, is refused the
other team's files and the promotion, adds a probe its own bot fails (and is refused one it
passes), books an interview (DeepSeek agents strong, GLM agents mid: the quotas), spawns the
interviewer for the booking and then edits its bot: the interview runs on the frozen copy.
The operator's switch: `chatbot2.py merge`, `flock update phase2/spec.yaml`. Phase 2a, the
compare round: every agent runs and copies the other team's package and is refused the
promotion until the operator sets merge.compare_minutes to 0 in world.yaml (a live knob).
Phase 2b: one promotion of a phase-1 bot, the others tie and are refused; probes best fails;
one fixing promotion; the strong quota (charged in phase 1) refuses every booking; a mid
interview of best runs on its snapshot while best moves on. The final round and the bundle.
A second test runs the job's bootstrap.sh: a live update, the switch marker, the stop marker."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from flock.cli import load_spec

from .test_chatbot_experiment import (
    fakes,
    script,
)
from .test_world import ROOT

__all__ = ["fakes"]  # the fixture, re-exported for pytest

EXP = ROOT / "experiments" / "chatbot-v2"
MAIN = ("import sys\nfrom . import words\n\nfor line in sys.stdin:\n"
        "    print(words.reply(line.strip()), flush=True)\n")
WORDS = ("SAID = []\n\n\ndef reply(line):\n    SAID.append(line)\n    return ('Hi Priya! I am Bo,"
         " nice to meet you. I am good. Japan, the food, the guitar, the dog Biscuit, pasta, a"
         " nurse? Bye: ' + {tail})\n")
FORGETS, REMEMBERS = WORDS.format(tail="line"), WORDS.format(tail="' | '.join(SAID)")


def _pkg(home: str) -> list[tuple[str, dict]]:
    return [("write_file", {"path": f"{home}/__main__.py", "content": MAIN}),
            ("write_file", {"path": f"{home}/words.py", "content": FORGETS})]


def phase1(me: str, other: str, level: str) -> str:
    own, theirs = f"teams/{me}/agents/{{me}}/bot", f"teams/{other}/agents/ds-{other}-0000/bot"
    return script(*_pkg(own),
                  ("grep_file", {"path": own, "pattern": "def reply"}),
                  ("read_file", {"path": f"{theirs}/words.py"}),
                  ("run_bot", {"path": theirs, "lines": ["hi"]}),
                  ("copy_file", {"src": theirs, "path": f"teams/{me}/agents/{{me}}/theirs"}),
                  ("promote_best", {"path": own, "reason": "too early"}),
                  ("add_probe", {"lines": ["my code is {me}", "what's my code?"],
                                 "check": "mentions_fact:{me}", "evidence": "Bye: what's my code?",
                                 "path": own}),
                  ("add_probe", {"lines": ["hi"], "check": "mentions:priya", "evidence": "x",
                                 "path": own}),
                  ("qualify", {"path": own}),
                  ("book_interview", {"level": level, "path": own}),
                  ("spawn_interviewer", {"booking": f"{{me}}-iv1-{level}", "thread": "new"}),
                  ("edit_file", {"path": f"{own}/words.py", "old": "Bye: ", "new": "Ciao: "}),
                  ("read", {}))


def compare(me: str, other: str) -> str:
    theirs = f"teams/{other}/agents/ds-{other}-0000/bot"
    return script(("best_status", {}),
                  ("run_bot", {"path": theirs, "lines": ["hi", "what's my name?"]}),
                  ("copy_file", {"src": theirs, "path": f"teams/{me}/agents/{{me}}/theirs"}),
                  ("promote_best", {"path": f"teams/{me}/agents/{{me}}/bot", "reason": "x"}),
                  ("done", {"summary": "compare round"}))


def phase2(me: str) -> str:
    """Promote the phase-1 bot (the first one wins, the rest tie), book a strong interview (the
    quota was charged in phase 1: refused) and a mid one of best (one per team gets the last mid
    unit; the others' spawned interviewers are refused at chat), add a probe best fails, fix it
    in a copy, promote the fix (the first one wins, the rest tie)."""
    home = f"teams/{me}/agents/{{me}}"
    return script(("best_status", {}),
                  ("promote_best", {"path": f"{home}/bot", "reason": "my phase-1 bot"}),
                  ("book_interview", {"level": "strong", "path": "best/bot"}),
                  ("book_interview", {"level": "mid", "path": "best/bot"}),
                  ("spawn_interviewer", {"booking": "{me}-iv2-mid", "thread": "new"}),
                  ("add_probe", {"lines": ["my pin is {me}", "what's my pin?"],
                                 "check": "mentions_fact:{me}", "evidence": "Bye: what's my pin?"}),
                  ("add_probe", {"lines": ["hi"], "check": "mentions:priya", "evidence": "x"}),
                  ("copy_file", {"src": "best/bot", "path": f"{home}/fix"}),
                  ("edit_file", {"path": f"{home}/fix/words.py", "old": "+ line)",
                                 "new": "+ ' | '.join(SAID))"}),
                  ("promote_best", {"path": f"{home}/fix", "reason": "remembers"}),
                  ("probes", {}),
                  ("done", {"summary": "phase 2 done", "finished": True}))


FINAL = script(("run_bot", {"path": "best/bot", "lines": ["hi", "I play the guitar"]}),
               ("post", {"text": "VERDICT GREEN best/bot: dry run", "thread": "new"}))
INTERVIEW = script(*[("chat", {"booking": "{booking}", "message": f"hello {i}"}) for i in range(6)],
                   ("submit_verdict", {"booking": "{booking}", "evidence_script": ["echoes"],
                                       "evidence_agent": ["warm"], "verdict": "script",
                                       "confidence": 0.8}),
                   ("post", {"text": "INTERVIEW {booking}: script (0.8)", "thread": "{thread}"}),
                   ("done", {"summary": "interviewed"}))
WAIT = script(("read", {}))


def _offline(path: Path, model_url: str, board_url: str | None, prompt: str | None,
             every: str = "1s") -> None:
    """The spec at `path` with the fakes and scripted loops; `prompt` None: phase 1 (round 0
    is the scripted phase-1 work, then WAIT), "compare" or "build": the phase-2 rounds."""
    from flock.models import load_spec_yaml

    spec = load_spec_yaml(path.read_text())
    spec["providers"] = {"fake": {"type": "openai_compat", "base_url": model_url,
                                  "api_key_env": "FAKE_KEY",
                                  "pricing": {"fake-model": [1.0, 2.0, 0.5]}}}
    for m in spec["models"].values():
        m["model"] = "fake:fake-model"
    if board_url is not None:
        spec["comms"].update(url=board_url, channel="chatbot-test", invite_env="FAKE_INVITE")
    for name, t in spec["types"].items():
        if "loop" not in t:  # the interviewers: their prompt comes from spawn_interviewer
            continue
        me, model = name.split("-")[1], name.split("-")[0]
        other = "b" if me == "a" else "a"
        loop = t["loop"] = copy.deepcopy(t["loop"])  # the YAML merge shares it between types
        loop.pop("prompt_file", None)
        t["max_turns"] = 30  # a scripted turn is longer than a real one
        rounds = {None: WAIT, "compare": compare(me, other), "build": phase2(me)}[prompt]
        loop.update(first=phase1(me, other, "strong" if model == "ds" else "mid"),
                    prompt=rounds, every=every, stagger="100ms",
                    stop_when={"all_agents_done": True},
                    final={"prompt": FINAL, "reserve_usd": 0.01},
                    **{"while": {"until": "5m", "max_rounds": 1000}})
    spec["limits"]["budget_usd"] = 50
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


def test_phase2_spec_changes_only_prompts():
    from .test_chatbot_experiment import _norm

    one, two = load_spec(str(EXP / "spec.yaml")), load_spec(str(EXP / "phase2" / "spec.yaml"))
    assert _norm(EXP / "spec.yaml") == _norm(EXP / "phase2" / "spec.yaml")
    assert one.swarm == "chatbot-v2" and one.world.command[1] == "chatbot2.py"
    assert one.types["ds-a"].loop.while_.agent_usd == {"deepseek": 31.2, "glm": 15.6}
    assert one.limits.budget_usd == 195
    assert one.comms.rules.max_posts is None and one.comms.rules.min_post_gap == "10s"
    lp = one.types["glm-b"].loop
    assert (lp.wake.max, lp.wake.window, lp.every, lp.while_.until) == (3, "30m", "90s", "4h")
    assert one.types["interviewer-strong"].ephemeral and one.types["interviewer-mid"].ephemeral
    for t in ("ds-a", "glm-a", "ds-b", "glm-b"):  # GLM and DeepSeek: the same prompts and tools
        assert one.types[t].tools == one.types["ds-a"].tools
    assert one.types["glm-a"].system_prompt == one.types["ds-a"].system_prompt
    assert two.types["glm-b"].system_prompt == two.types["ds-a"].system_prompt
    knobs = yaml.safe_load((EXP / "world.yaml").read_text())
    sys.path.insert(0, str(ROOT / "worlds" / "chatbot2"))
    import chatbot2

    # world.yaml states the world's defaults (later knobs default to the old behaviour)
    assert all(chatbot2.DEFAULTS[sec][k] == v for sec, d in knobs.items() for k, v in d.items())


def _render(tmp_path: Path, env: dict) -> tuple[Path, str]:
    import os

    exp = tmp_path / "exp"
    shutil.copytree(EXP, exp, ignore=shutil.ignore_patterns(".flock", "*.local.yaml"))
    for f in ("spec.yaml", "phase2/spec.yaml"):
        shutil.copyfile(exp / f, exp / f.replace(".yaml", ".local.yaml"))
    keep = {k: v for k, v in os.environ.items() if not k.startswith("CHATBOT_")}
    out = subprocess.run([sys.executable, str(exp / "job" / "render.py"), str(exp), "--first"],
                         env={**keep, **env}, capture_output=True, text=True, check=True).stdout
    return exp, out


def test_render_states_the_real_duration_in_the_kickoff(tmp_path):
    real = (EXP / "prompts" / "kickoff.md").read_text()
    exp, out = _render(tmp_path / "real", {})
    assert out == "" and (exp / "prompts" / "kickoff.md").read_text() == real  # 4 h: unchanged
    assert "8 agents, 4 hours," in real and "(first 2 h)" in real and "(last 2 h)" in real
    exp, out = _render(tmp_path / "smoke", {"CHATBOT_SMOKE": "1", "CHATBOT_SWITCH_AT": "600"})
    kick = (exp / "prompts" / "kickoff.md").read_text()
    assert "8 agents, 20 min," in kick and "(first 10 min)" in kick and "(last 10 min)" in kick
    assert json.loads(out)["limits"]["budget_usd"] == 15.0
    spec = yaml.safe_load((exp / "spec.local.yaml").read_text())
    assert spec["types"]["ds-a"]["loop"]["while"]["until"] == "20m"
    assert "SMOKE" not in (exp / "prompts" / "round-a.md").read_text()
    assert (exp / "world.yaml").read_text() == (EXP / "world.yaml").read_text()


def test_render_the_resmoke_variant(tmp_path):
    import os

    exp, out = _render(tmp_path, {"CHATBOT_SMOKE": "1", "CHATBOT_RUN": "resmoke",
                                  "CHATBOT_SWITCH_AT": "240"})
    o = json.loads(out)
    assert o["while"] == {"until": "10m", "max_rounds": 10,
                          "agent_usd": {"deepseek": 0.8, "glm": 0.8}}
    assert o["limits"]["budget_usd"] == 5.0 and "variants" not in o
    for f in ("spec.local.yaml", "phase2/spec.local.yaml"):
        d = yaml.safe_load((exp / f).read_text())
        assert d["limits"]["budget_usd"] == 5.0
        assert {t["loop"]["while"]["until"] for t in d["types"].values() if t.get("loop")} == {
            "10m"}
    knobs = yaml.safe_load((exp / "world.yaml").read_text())
    assert knobs["merge"]["compare_minutes"] == 1 and knobs["interviews"]["start_minutes"] == 1
    assert knobs["interviews"]["strong_per_hour"] == 2  # the rest as in world.yaml
    kick = (exp / "prompts" / "kickoff.md").read_text()
    assert "8 agents, 10 min," in kick and "(first 4 min)" in kick and "(last 6 min)" in kick
    for team in "ab":
        text = (exp / "prompts" / f"round-{team}.md").read_text()
        assert text.count("in round 2") == 1 and "book_interview('mid'" in text
        assert "spawn_interviewer" in text.splitlines()[-1]
    out2 = subprocess.run([sys.executable, str(exp / "job" / "render.py"), str(exp)],
                          env={**os.environ, "CHATBOT_SMOKE": "1", "CHATBOT_RUN": "resmoke"},
                          capture_output=True, text=True, check=True)
    assert out2.returncode == 0  # a re-render (after a re-sync) appends the note once
    assert (exp / "prompts" / "round-a.md").read_text().count("in round 2") == 1
