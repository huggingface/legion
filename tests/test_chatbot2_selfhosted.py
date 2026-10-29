"""experiments/chatbot-v2/selfhosted: the render (N teams, price 0, no $ caps, continuous
cadence), the load smoke's analysis, and an offline run of a rendered 4-team swarm under
`flock run` with the fakes: phase 1 isolates every team from the 3 others, the rounds come
back to back (`every: turn_end`), price 0 never stops anything; phase 2 (a second run on the
same, merged workspace) reads across teams and promotes."""

import datetime as dt
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from flock.cli import load_spec

from .test_chatbot_experiment import _norm, _results, _wlog, fakes, script
from .test_local_run_e2e import _bundle_events, _env, _flock
from .test_world import ROOT

__all__ = ["fakes"]

EXP = ROOT / "experiments" / "chatbot-v2"
SH = EXP / "selfhosted"
MAIN = ("import sys\nfrom . import words\n\nfor line in sys.stdin:\n"
        "    print(words.reply(line.strip()), flush=True)\n")
WORDS = ("def reply(line):\n    return ('Hi Priya! I am Bo, nice to meet you. I am good. Japan, the"
         " food, the guitar, the dog Biscuit, pasta, a nurse? Bye: ' + line)\n")


def _render(tmp: Path, monkeypatch, *args: str, **envs: str) -> tuple[Path, str]:
    for k in ("SELFHOSTED_BASE_URL", "BASE_URL_DS", "BASE_URL_GLM", "MODEL_DS", "MODEL_GLM",
              "MODEL_INTERVIEWER_STRONG", "MODEL_INTERVIEWER_MID"):
        monkeypatch.delenv(k, raising=False)
    exp = tmp / "exp"
    shutil.copytree(EXP, exp, ignore=shutil.ignore_patterns(".flock*", "*.local.yaml"))
    r = subprocess.run([sys.executable, str(SH / "render.py"), "--into", str(exp), *args],
                       env={**__import__("os").environ, **envs}, capture_output=True, text=True,
                       timeout=60, check=False)
    assert r.returncode == 0, r.stderr
    return exp, r.stdout


def test_render_defaults_and_the_two_model_mix(tmp_path, monkeypatch):
    exp, out = _render(tmp_path, monkeypatch)  # no env: one model, placeholders
    assert yaml.safe_load(out) == {"teams": 4, "agents": 16, "per_team": {"ds": 4},
                                   "workers": 4, "models": ["ds:MODEL_DS"]}
    s = load_spec(str(exp / "spec.yaml"))
    assert sorted(s.types) == ["ds-a", "ds-b", "ds-c", "ds-d", "interviewer-mid",
                               "interviewer-strong"]
    assert s.providers["ds"].base_url == "https://your-endpoint.example.invalid/v1"
    exp, out = _render(tmp_path / "2", monkeypatch, MODEL_DS="m-ds", MODEL_GLM="m-glm",
                       SELFHOSTED_BASE_URL="http://gpu.internal:8000/v1")
    assert yaml.safe_load(out)["workers"] == 8 and yaml.safe_load(out)["agents"] == 16
    one, two = load_spec(str(exp / "spec.yaml")), load_spec(str(exp / "phase2" / "spec.yaml"))
    assert _norm(exp / "spec.yaml") == _norm(exp / "phase2" / "spec.yaml")  # only prompts differ
    assert sorted(t for t in one.types if one.types[t].loop) == [
        f"{m}-{t}" for m in ("ds", "glm") for t in "abcd"]
    for name, t in one.types.items():
        assert t.tool_choice == "required" and t.max_turns == 12 and t.max_budget_usd is None
        if t.loop:
            lp = t.loop
            assert (t.replicas, t.agents_per_worker, lp.every) == (2, 8, "turn_end")
            assert (lp.wake.min_gap, lp.wake.max, lp.wake.window) == ("10s", 20, "30m")
            assert (lp.while_.until, lp.while_.max_rounds, lp.while_.agent_usd) == (
                "240m", 600, None)
            assert lp.final.reserve_usd in (0, None)
            assert f"system-{name[-1]}" in t.system_prompt_file
    assert one.limits.budget_usd is None and one.comms.rules.min_post_gap == "10s"
    for name, p in one.providers.items():
        assert p.base_url == "http://gpu.internal:8000/v1" and p.self_hosted
        assert p.api_key_env == "SELFHOSTED_API_KEY" and p.pricing
        assert all(list(v) == [0, 0, 0] for v in p.pricing.values()), p.pricing
    assert one.models["interviewer-strong"].model == "ds:m-ds"
    assert one.models["interviewer-mid"].model == "glm:m-glm"
    assert two.types["glm-c"].loop.prompt_file == "round-c.md"
    sysc = (exp / "prompts" / "system-c.md").read_text()
    assert "TEAM C: ds-c-0000, ds-c-0001, glm-c-0000, glm-c-0001" in sysc
    assert "Teams A, B and D work on their own" in sysc and "16 agents" in sysc
    assert "[team-d]" in (exp / "prompts" / "kickoff.md").read_text()
    assert "team a" in (exp / "phase2" / "round-d.md").read_text()  # d compares a's bot
    assert yaml.safe_load((exp / "world.yaml").read_text())["probes"]["max_total"] == 320


def test_load_smoke_summary_and_knee():
    s = importlib.util.spec_from_file_location("load_smoke", SH / "load_smoke.py")
    ls = importlib.util.module_from_spec(s)
    s.loader.exec_module(ls)
    recs = [{"path": "/v1/chat/completions", "t0": 100 + i, "t1": 100.5 + i, "status": 200,
             "out": 20} for i in range(10)] + [
        {"path": "/v1/chat/completions", "t0": 105, "t1": 107, "status": 500},
        {"path": "/v1/models", "t0": 105, "t1": 105.1, "status": 200}]
    assert ls.summarize(recs, 100, 110) == {"calls": 11, "calls_s": 1.1, "p50_ms": 500,
                                            "p90_ms": 500, "errors": 1, "out_tok_s": 20}
    row = lambda c, cs, p50, err=0: {"agents": c, "calls_s": cs, "p50_ms": p50,
                                     "errors": err, "calls": 1000}
    assert ls.knee([row(8, 10, 500), row(16, 19, 520), row(32, 36, 600),
                    row(64, 40, 1100)]) == 32
    assert ls.knee([row(8, 10, 500), row(16, 19, 520), row(32, 30, 900, err=50)]) == 16
    assert ls.knee([row(8, 10, 500), row(16, 20, 500)]) is None


def _scripted(exp: Path, f: str, model_url: str, board_url: str, invite_env: str,
              first: dict[str, str] | None, rounds: dict[str, str]) -> None:
    d = yaml.safe_load((exp / f).read_text())
    d["comms"].update(url=board_url, channel="chatbot-test", invite_env=invite_env)
    for name, t in d["types"].items():
        if not t.get("loop"):
            continue
        lp, team = t["loop"], name.split("-")[1]
        lp.pop("prompt_file", None)
        if first:
            lp["first"] = first[team]
        else:
            lp.pop("first", None)
        lp.update(prompt=rounds[team], stagger="100ms",
                  final={"prompt": script(("done", {"summary": "final"}))})
        lp["while"]["max_rounds"] = 5 if first else 3
        t["max_turns"] = 30
    (exp / f).write_text(yaml.safe_dump(d, sort_keys=False))


def _ts(e: dict) -> float:
    return dt.datetime.fromisoformat(e["ts"]).timestamp()


def test_four_teams_offline_continuous_price_zero(tmp_path, fakes, monkeypatch):
    model_url, board, board_url = fakes
    exp, _ = _render(tmp_path, monkeypatch, "--teams", "4", "--per-team", "ds=1,glm=1",
                     MODEL_DS="fake-model", MODEL_GLM="fake-model",
                     SELFHOSTED_BASE_URL=model_url)
    teams = "abcd"
    agents = [f"{m}-{t}-0000" for t in teams for m in ("ds", "glm")]
    p1 = {t: script(("write_file", {"path": f"teams/{t}/agents/{{me}}/bot/__main__.py",
                                    "content": MAIN}),
                    ("write_file", {"path": f"teams/{t}/agents/{{me}}/bot/words.py",
                                    "content": WORDS}),
                    *[("read_file", {"path": f"teams/{u}/agents/ds-{u}-0000/bot/words.py"})
                      for u in teams if u != t],
                    ("list_files", {"prefix": "teams/"}),
                    ("promote_best", {"path": f"teams/{t}/agents/{{me}}/bot", "reason": "x"}),
                    ("done", {"summary": "phase 1"})) for t in teams}
    wait = {t: script(("done", {"summary": "round"})) for t in teams}
    _scripted(exp, "spec.yaml", model_url, board_url, "FAKE_INVITE", p1, wait)
    ws = tmp_path / "ws"
    env = {**_env(tmp_path), "FAKE_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_WS": str(ws), "CHATBOT_CONFIG": str(exp / "world.yaml"),
           "SELFHOSTED_API_KEY": "fake-key"}
    r = subprocess.run(_flock("run", "spec.yaml", "--until", "idle", "--skip-preflight"),
                       cwd=exp, env=env, capture_output=True, text=True, timeout=240, check=False)
    assert r.returncode == 0 and "stopping (idle)" in r.stdout, r.stdout[-3000:] + r.stderr
    (run,) = [p for p in (exp / ".flock").iterdir() if p.is_dir()]
    for a in agents:  # phase 1: each of the 3 other teams refused, the listing only its own
        t, res = a.split("-")[1], _results(run, a)
        assert sum(" belongs to team " in x and "until the merge" in x for x in res) == 3, res
        listing = next(x for x in res if '\\"total\\"' in x)
        assert all((f"teams/{u}/" in listing) == (u == t) for u in teams), listing
        assert any("promotion opens after the merge" in x for x in res)
        # continuous: each round goes out right after the previous round's turn ended
        ev = _bundle_events(run, a)
        ends = [_ts(e) for e in ev if e["type"] == "turn_ended"]
        sent = [(_ts(e), e["data"]["data"]["round"]) for e in ev if e["type"] == "loop_prompt"
                and e["data"]["data"]["kind"] == "loop"]
        assert [n for _, n in sent] == [1, 2, 3, 4, 5], sent
        gaps = [ts - max(x for x in ends if x <= ts) for ts, n in sent if n > 1]
        assert gaps and max(gaps) < 1.5, gaps
        assert {e["data"]["data"]["why"] for e in ev if e["type"] == "loop_prompt"
                and e["data"]["data"]["kind"] == "loop" and e["data"]["data"]["round"] > 1} == {
            "turn_end"}
    doc = json.loads((run / "run.json").read_text())
    assert doc["budget"]["stop"] is None and doc["budget"]["cost_usd"] == 0, doc["budget"]
    assert doc["budget"]["budget_usd"] is None
    assert all(doc["loops"][f"{m}-{t}"]["state"] == "done" for t in teams for m in ("ds", "glm"))

    # phase 2: the operator's merge (here: the world restarted on a merged workspace)
    (ws / "phase.json").write_text(json.dumps({"phase": "merged", "before": "teams", "ts": 0}))
    p2 = {t: script(*[("read_file", {"path": f"teams/{u}/agents/ds-{u}-0000/bot/words.py",
                                     "start": 1, "end": 1}) for u in teams if u != t],
                    ("copy_file", {"src": f"teams/{teams[(teams.index(t) + 1) % 4]}/agents/"
                                          f"ds-{teams[(teams.index(t) + 1) % 4]}-0000/bot",
                                   "path": f"teams/{t}/agents/{{me}}/next"}),
                    ("promote_best", {"path": f"teams/{t}/agents/{{me}}/bot", "reason": "p2"}),
                    ("done", {"summary": "phase 2", "finished": True})) for t in teams}
    _scripted(exp, "phase2/spec.yaml", model_url, board_url, "FAKE_INVITE", None, p2)
    d = yaml.safe_load((exp / "phase2" / "spec.yaml").read_text())
    for t in d["types"].values():
        if t.get("loop"):
            t["loop"]["stop_when"] = {"all_agents_done": True}
    (exp / "phase2" / "spec.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
    r = subprocess.run(_flock("run", "phase2/spec.yaml", "--until", "idle", "--skip-preflight",
                              "--data-dir", ".flock2"), cwd=exp, env=env, capture_output=True,
                       text=True, timeout=240, check=False)
    assert r.returncode == 0 and "stopping (idle)" in r.stdout, r.stdout[-3000:] + r.stderr
    (run2,) = [p for p in (exp / ".flock2").iterdir() if p.is_dir()]
    for a in agents:
        res = _results(run2, a)
        assert not any("belongs to team" in x for x in res), res
        assert sum("lines 1-1 of" in x for x in res) == 3
        assert (ws / f"teams/{a.split('-')[1]}/agents/{a}/next/words.py").read_text() == WORDS
    log = _wlog(ws)
    assert sum(x["action"] == "promote" for x in log) == 1  # identical bots: the rest tie
    assert sum(x["action"] == "promote_refused" and "a tie" in x.get("why", "") for x in log) == 7


def test_job_bootstrap_renders_the_variant(tmp_path, fakes):
    """CHATBOT_VARIANT=selfhosted: the bootstrap renders the variant into its copy of the
    experiment before anything else (the endpoint and models from the job's env), runs it, and
    the stop marker ends it with a bundle."""
    import os
    import tarfile

    from .test_chatbot_experiment import _no_secrets, _until, _wait

    model_url, board, board_url = fakes
    bucket = tmp_path / "bucket"
    (bucket / "payload").mkdir(parents=True)
    with tarfile.open(bucket / "payload" / "repo.tar.gz", "w:gz") as tar:
        for d in ("experiments/chatbot-v2", "worlds/chatbot2", "worlds/chatbot", "worlds/eliza"):
            tar.add(ROOT / d, arcname=d, filter=lambda i: None if "__pycache__" in i.name
                    or "/ws" in i.name or ".flock" in i.name else i)
    env = {**_env(tmp_path), "PATH": f"{Path(sys.executable).parent}:{os.environ['PATH']}",
           "PYTHON": sys.executable, "CHATBOT_PREINSTALLED": "1",
           "CHATBOT_HOME": str(tmp_path / "job"), "CHATBOT_BUCKET": str(bucket),
           "CHATBOT_BOARD_URL": board_url, "CHATBOT_CHANNEL": "chatbot-test",
           "CHATBOT_BOARD_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_SYNC_S": "3", "CHATBOT_CHECK_S": "1", "CHATBOT_TICK_S": "1",
           "CHATBOT_FLOCK_ARGS": "--skip-preflight", "CHATBOT_SMOKE": "1",
           "CHATBOT_VARIANT": "selfhosted", "SELFHOSTED_BASE_URL": model_url,
           "SELFHOSTED_API_KEY": "fake-key", "MODEL_DS": "fake-model"}
    out = open(tmp_path / "job.out", "w")  # noqa: SIM115 (the process holds it)
    proc = subprocess.Popen(["bash", "-c", (EXP / "job" / "bootstrap.sh").read_text()], env=env,
                            stdout=out, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    _until(proc, tmp_path, "flock run started")
    spec = yaml.safe_load((tmp_path / "job/repo/experiments/chatbot-v2/spec.local.yaml")
                          .read_text())
    assert sorted(t for t in spec["types"] if spec["types"][t].get("loop")) == [
        "ds-a", "ds-b", "ds-c", "ds-d"]
    assert spec["providers"]["ds"]["base_url"] == model_url
    assert spec["comms"]["url"] == board_url
    assert spec["limits"]["budget_usd"] is not None  # the smoke's caps still apply
    assert "agents: 16" in (tmp_path / "job" / "logs" / "variant.log").read_text()
    (bucket / "control").mkdir()
    (bucket / "control" / "stop").write_text("stop\n")
    _wait(proc, 120)
    text = (tmp_path / "job.out").read_text()
    assert "stop marker found" in text and "final sync" in text, text[-3000:]
    _no_secrets(bucket, board)
