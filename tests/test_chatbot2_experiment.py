"""experiments/chatbot-v2 end to end under `flock run` (offline): the real CLI runs the
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

import copy
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

from flock.cli import load_spec

from .test_chatbot_experiment import (
    AGENTS,
    TEAMS,
    _kill,
    _no_secrets,
    _results,
    _until,
    _wait,
    _wlog,
    fakes,
    script,
)
from .test_local_run_e2e import _bundle_events, _env, _flock, _run_dir
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


def _setup(tmp_path: Path, model_url: str, board_url: str | None, prompt2: str) -> Path:
    exp = tmp_path / "exp"
    shutil.copytree(EXP, exp, ignore=shutil.ignore_patterns(".flock", "*.local.yaml"))
    tool = exp / "tools" / "interview.py"  # the interviewer's task: a script for the fake model
    src = tool.read_text()
    a = src.index("PROMPT = (")
    tool.write_text(src[:a] + f"PROMPT = {INTERVIEW!r}" + src[src.index("\n\n\n", a):])
    for w in ("chatbot2", "chatbot", "eliza"):  # chatbot2 imports ../chatbot and ../eliza
        shutil.copytree(ROOT / "worlds" / w, tmp_path / "worlds" / w,
                        ignore=shutil.ignore_patterns("ws", "__py*"))
    for f, prompt, wdir in (("spec.yaml", None, "../worlds/chatbot2"),
                            ("phase2/spec.yaml", prompt2, "../../worlds/chatbot2")):
        _offline(exp / f, model_url, board_url, prompt)
        doc = yaml.safe_load((exp / f).read_text())
        doc["world"]["dir"] = wdir  # the copied worlds, at the same relative place
        (exp / f).write_text(yaml.safe_dump(doc, sort_keys=False))
    return exp


def _wait_for(proc, out: Path, ws: Path, what: str, ok, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while not ok(_wlog(ws)):
        assert proc.poll() is None and time.monotonic() < deadline, (
            what, [(x["agent"], x["action"], x.get("tool"), x.get("why")) for x in _wlog(ws)
                   if x["action"] not in ("write_file", "edit_file")][-60:],
            out.read_text()[-3000:])
        time.sleep(0.5)


def test_chatbot2_experiment_dry_run(tmp_path, fakes):
    model_url, board, board_url = fakes
    exp = _setup(tmp_path, model_url, board_url, "compare")
    ws, bucket, cfg = tmp_path / "ws", tmp_path / "bucket", exp / "world.yaml"
    cfg.write_text(cfg.read_text().replace("compare_minutes: 15", "compare_minutes: 10"))
    env = {**_env(tmp_path), "FAKE_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_WS": str(ws), "CHATBOT_BUCKET": str(bucket), "CHATBOT_CONFIG": str(cfg)}
    out = tmp_path / "run.out"
    run_out = open(out, "w")  # noqa: SIM115 (closed below)
    proc = subprocess.Popen(_flock("run", "spec.yaml", "--until", "idle", "--skip-preflight"),
                            cwd=exp, env=env, stdout=run_out, stderr=subprocess.STDOUT, text=True)

    def operator(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(argv, cwd=exp, env=env, capture_output=True, text=True,
                              timeout=60, check=False)

    world_py = str(tmp_path / "worlds" / "chatbot2" / "chatbot2.py")
    try:  # phase 1: every agent's round 0 ran, and the 8 interviews filed their verdicts
        _wait_for(proc, out, ws, "phase 1", lambda log: {
            x["agent"] for x in log if x["action"] == "edit_file"} == set(AGENTS)
            and sum(x["action"] == "verdict" for x in log) == 8)
        assert not (ws / "best").exists() or not list((ws / "best").glob("bot*"))
        m = operator(sys.executable, world_py, "merge")
        assert m.returncode == 0 and json.loads(m.stdout)["phase"] == "merged", m.stdout + m.stderr
        u = operator(*_flock("update", "phase2/spec.yaml"))
        assert u.returncode == 0 and "prompts changed" in u.stdout, u.stdout + u.stderr
        # phase 2a: the compare round locks promotion for every agent, until the operator's 0
        _wait_for(proc, out, ws, "compare round", lambda log: {
            x["agent"] for x in log if x.get("why") == "compare round"} == set(AGENTS))
        cfg.write_text(cfg.read_text().replace("compare_minutes: 10", "compare_minutes: 0"))
        _offline(exp / "phase2" / "spec.yaml", model_url, board_url, "build")  # phase 2b
        doc = yaml.safe_load((exp / "phase2" / "spec.yaml").read_text())
        doc["world"]["dir"] = "../../worlds/chatbot2"
        (exp / "phase2" / "spec.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
        u = operator(*_flock("update", "phase2/spec.yaml"))
        assert u.returncode == 0 and "prompts changed" in u.stdout, u.stdout + u.stderr
        proc.wait(timeout=240)
    finally:
        if proc.poll() is None:
            proc.kill()
        run_out.close()
    text = out.read_text()
    assert proc.returncode == 0 and "stopping (idle)" in text, text[-3000:]
    run, log = _run_dir(exp), _wlog(ws)

    for team, other in (("a", "b"), ("b", "a")):
        for a in TEAMS[team]:
            res = _results(run, a)
            # phase 1: the package, grep, the other team refused, the promotion refused
            assert any(f"teams/{team}/agents/{a}/bot/words.py:4:def reply(line):" in r for r in res)
            assert sum(f"teams/{other}/ belongs to team {other.upper()} until the merge" in r
                       for r in res) == 3, res
            assert any("promotion opens after the merge" in r for r in res)
            assert any("already passes this probe" in r for r in res)
            # phase 2: the other team's package copied (recursively), the compare round
            assert (ws / f"teams/{team}/agents/{a}/theirs/words.py").is_file()
            assert any("compare round, promotion opens in 10 min" in r for r in res)
            assert any("no strong interview left this hour" in r for r in res)
    # probes: 8 added in phase 1 (each agent's own bot fails it), phase 2's against best
    live = [json.loads(x) for x in (ws / "probes" / "live.jsonl").read_text().splitlines()]
    p1 = [p for p in live if p["lines"][0].startswith("my code is")]
    assert sorted(p["author"] for p in p1) == sorted(AGENTS)
    assert all(p["evidence_matches"] and p["path"].endswith("/bot") for p in p1)
    p2 = [p for p in live if p["lines"][0].startswith("my pin is")]
    assert p2 and all(p["path"] == "best/bot" for p in p2)
    refused = [x for x in log if x["action"] == "probe_refused"]
    assert len(refused) >= 16  # the `mentions:priya` probes, twice per agent, at least
    # promotions: one phase-1 bot (the rest tie), one fix; every other try refused as a tie
    hist = [json.loads(x) for x in (ws / "best" / "history.jsonl").read_text().splitlines()]
    assert len(hist) == 2 and [h["kind"] for h in hist] == ["package", "package"], hist
    assert hist[0]["path"].endswith("/bot") and hist[1]["path"].endswith("/fix")
    ties = [x for x in log if x["action"] == "promote_refused" and "a tie" in x.get("why", "")]
    assert len(ties) >= 7, [x.get("why") for x in log if x["action"] == "promote_refused"]
    best = (ws / "best" / "bot" / "words.py").read_text()
    assert "' | '.join(SAID)" in best and best in (REMEMBERS, REMEMBERS.replace("Bye", "Ciao"))
    assert json.loads((ws / "best" / "current.json").read_text())["entry"] == "best/bot"
    assert (bucket / "best" / "bot" / "words.py").read_text() == best  # the mirror
    assert (bucket / "best" / "v1" / "__main__.py").is_file()
    assert not any(x["action"] == "mirror_error" for x in log)
    # interviews: phase 1, 2 strong (DeepSeek) + 2 mid (GLM) per team, on frozen copies
    verdicts = [json.loads(x) for x in (ws / "verdicts.jsonl").read_text().splitlines()]
    v1 = [v for v in verdicts if v["best_n"] is None]
    assert sorted((v["team"], v["level"]) for v in v1) == sorted(
        [(t, lv) for t in "ab" for lv in ("strong", "strong", "mid", "mid")])
    for v in v1:
        assert v["snapshot"] == f"teams/{v['team']}/interviews/{v['booking']}/bot"
        assert all("Bye: " in t["bot"] and "Ciao" not in t["bot"] for t in v["transcript"])
        assert v["counts"] == (v["level"] == "strong") and v["turns"] == 6
    starts = {x["booking"]: x["ts"] for x in log if x["action"] == "interview_start"}
    edits = {x["agent"]: x["ts"] for x in log if x["action"] == "edit_file"
             and x["path"].endswith("/bot/words.py")}
    assert any(starts[v["booking"]] > edits[v["booking"].split("-iv")[0]] for v in v1)
    # phase 2: the strong quota (charged) refuses every booking; one mid per team on best n1
    assert sum(x.get("why") == "quota" and x.get("level") == "strong" for x in log) >= 8
    v2 = [v for v in verdicts if v["best_n"] is not None]  # one mid unit left per team
    assert sorted(v["team"] for v in v2) == ["a", "b"] and {v["level"] for v in v2} == {"mid"}
    for v in v2:  # the frozen copy of best/v<n>, whatever best became afterwards
        snap = ws / v["snapshot"]
        assert (snap / "words.py").read_text() == (
            ws / "best" / f"v{v['best_n']}" / "words.py").read_text()
    gone = [e["data"] for a in AGENTS for e in _bundle_events(run, a)
            if e["type"] == "child_removed"]
    assert {g["reason"] for g in gone} == {"ephemeral"} and len(gone) >= 10, gone
    # what the agents saw: the v2 world and prompts, the tools
    reqs = [json.loads(x)["body"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    systems = [r["messages"][0]["content"] for r in reqs]
    assert any("## World: chatbot (version 0.2.0-teams)" in s and "ONE MODULE PER FEATURE" in s
               for s in systems)
    assert any("## World: chatbot (version 0.2.0-merged)" in s and "PHASE 2" in s for s in systems)
    offered = {frozenset(t["function"]["name"] for t in r["tools"]) for r in reqs}
    assert offered == {frozenset({
        "read_file", "grep_file", "write_file", "copy_file", "edit_file", "insert_after",
        "list_files", "run_bot", "promote_best", "best_status", "qualify", "add_probe", "probes",
        "book_interview", "verdicts", "spawn_interviewer", "post", "read", "search", "done"}),
        frozenset({"chat", "submit_verdict", "post", "done"})}
    assert len([m for m in board.msgs if "VERDICT GREEN" in m["body"]]) == len(AGENTS)
    doc = json.loads((run / "run.json").read_text())
    assert all(doc["loops"][t]["final_done"] == 2 for t in ("ds-a", "glm-a", "ds-b", "glm-b"))
    for secret in ("w0rld-t0ken", board.invite, "fake-key"):
        assert not any(secret in p.read_text(errors="ignore") for p in run.rglob("*")
                       if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".md", ""))


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


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
@pytest.mark.parametrize("variant", ["run3", "resmoke"])
def test_job_bootstrap_update_switch_and_stop(tmp_path, fakes, variant):
    """bootstrap.sh without a job (a directory as the bucket): control/update re-syncs
    <bucket>/experiment/ (world.yaml too: the world re-reads it) and runs `flock update`, then the
    marker is deleted; control/switch runs job/switch.sh (merge + phase-2 prompts); control/stop
    ends with a bundle and the final sync."""
    import os
    import tarfile

    model_url, board, board_url = fakes
    for f in (EXP / "job").glob("*.sh"):
        assert subprocess.run(["bash", "-n", str(f)], check=False).returncode == 0
        assert os.access(f, os.X_OK), f
    src, bucket = tmp_path / "src", tmp_path / "bucket"
    exp = _setup(tmp_path / "s", model_url, None, "compare")
    shutil.copytree(exp, src / "experiments" / "chatbot-v2")
    shutil.copytree(tmp_path / "s" / "worlds", src / "worlds")
    for f in ("spec.yaml", "phase2/spec.yaml"):  # the repo layout again
        p = src / "experiments" / "chatbot-v2" / f
        doc = yaml.safe_load(p.read_text())
        doc["world"]["dir"] = ("../" * f.count("/")) + "../../worlds/chatbot2"
        p.write_text(yaml.safe_dump(doc, sort_keys=False))
    (bucket / "payload").mkdir(parents=True)
    with tarfile.open(bucket / "payload" / "repo.tar.gz", "w:gz") as tar:
        for d in ("experiments", "worlds"):
            tar.add(src / d, arcname=d)
    env = {**_env(tmp_path), "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "PYTHON": sys.executable, "CHATBOT_PREINSTALLED": "1",
           "CHATBOT_HOME": str(tmp_path / "job"), "CHATBOT_BUCKET": str(bucket),
           "CHATBOT_BOARD_URL": board_url, "CHATBOT_CHANNEL": "chatbot-test",
           "CHATBOT_BOARD_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_SYNC_S": "3", "CHATBOT_CHECK_S": "1", "CHATBOT_TICK_S": "1",
           "CHATBOT_FLOCK_ARGS": "--skip-preflight", "CHATBOT_SMOKE": "1", "CHATBOT_RUN": variant}
    out = open(tmp_path / "job.out", "w")  # noqa: SIM115 (the process holds it)
    proc = subprocess.Popen(["bash", "-c", (EXP / "job" / "bootstrap.sh").read_text()], env=env,
                            stdout=out, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    try:
        ws = tmp_path / "job" / "ws"
        _wait_for(proc, tmp_path / "job.out", ws, "round 0", lambda log: {
            x["agent"] for x in log if x["action"] == "edit_file"} == set(AGENTS))
        job_exp = tmp_path / "job" / "repo" / "experiments" / "chatbot-v2"
        if variant == "resmoke":  # the variant's world knobs and round note, from the start
            knobs = yaml.safe_load((job_exp / "world.yaml").read_text())
            assert knobs["interviews"]["start_minutes"] == 1, knobs
            assert "in round 2" in (job_exp / "prompts" / "round-b.md").read_text()
        edited = bucket / "experiment"
        shutil.copytree(src / "experiments" / "chatbot-v2", edited)
        for f in ("spec.yaml", "phase2/spec.yaml"):
            (edited / f).write_text((edited / f).read_text().replace("min_post_gap: 10s",
                                                                     "min_post_gap: 5s"))
        (edited / "world.yaml").write_text("merge: {compare_minutes: 0}\n")
        (bucket / "control").mkdir()
        (bucket / "control" / "update").write_text("go\n")
        _until(proc, tmp_path, "update 1 ok")
        log1 = (bucket / "control" / "update-1.log").read_text()
        assert "comms.rules.min_post_gap: 10s -> 5s" in log1, log1
        assert not (bucket / "control" / "update").exists()  # the marker is consumed (-y)
        (bucket / "control" / "switch").write_text("go\n")
        _until(proc, tmp_path, "phase switch ok")
        switch = (bucket / "control" / "switch.log").read_text()
        assert '"phase":"merged"' in switch and "prompts changed" in switch, switch
        knobs = tmp_path / "job" / "repo" / "experiments" / "chatbot-v2" / "world.yaml"
        assert knobs.read_text() == "merge: {compare_minutes: 0}\n"  # what the world re-reads
        (bucket / "control" / "stop").write_text("stop\n")
        _wait(proc, 90)
    except BaseException:
        _kill(proc)
        raise
    text = (tmp_path / "job.out").read_text()
    assert "smoke settings" in text and "stop marker found" in text and "done" in text, text
    (run,) = (bucket / "flock" / ".flock").iterdir()
    doc = json.loads((run / "run.json").read_text())
    assert doc["swarm"] == f"chatbot-v2-{variant}" and doc["stop_reason"] == "interrupted"
    kick = board.msgs[0]["body"]  # smoke.json's duration, not the real run's 4 hours
    mins = {"run3": (20, 10, 10), "resmoke": (10, 5, 5)}[variant]
    assert kick.startswith(f"Kickoff (chatbot-v2 {variant}). 8 agents, {mins[0]} min, ONE"), kick
    assert f"(first {mins[1]} min)" in kick and f"(last {mins[2]} min)" in kick, kick
    if variant == "resmoke":
        assert "smoke settings: {" in text and '"budget_usd":5.0' in text, text
    assert (bucket / "ws" / "interviews.json").is_file() and (bucket / "ws" / "probes").is_dir()
    _no_secrets(bucket, board)
