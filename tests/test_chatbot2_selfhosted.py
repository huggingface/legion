"""experiments/chatbot-v2/selfhosted: the render (N teams, price 0, no $ caps, continuous
cadence), the load smoke's analysis, and an offline run of a rendered 4-team swarm under
a local run with the fakes: phase 1 isolates every team from the 3 others, the rounds come
back to back (`every: turn_end`), price 0 never stops anything; phase 2 (a second run on the
same, merged workspace) reads across teams and promotes."""

from __future__ import annotations

import datetime as dt
import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from flock.cli import load_spec

from .test_chatbot_experiment import _norm, fakes, script
from .test_local_run_e2e import _env
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
