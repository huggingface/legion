"""experiments/chatbot-v1 end to end under a local run (flock.controller.localrun) (offline): the real CLI in a subprocess runs
the experiment's own spec, kickoff and protocols with worlds/chatbot, against tests/fake_openai.py
and tests/fake_msgboard.py. Only the provider, the board, the loop prompts (scripted tool calls)
and the loop timings are swapped: 8 agents in 4 types (2 teams).

Phase 1: cross-team reads, copies and runs are refused, promote_best is refused; each team agent
qualifies its bot and spawns a mid interviewer (tools/interview.py, flock spawn), which chats
with it, files a verdict, posts it and finishes; the team quota (3 per hour) refuses the 4th
interviewer of each team; every interviewer is removed when its session ends (ephemeral).
Then the
operator's switch, exactly as the README gives it: `chatbot.py merge` (the world's operator route,
world token) and `flock update phase2/spec.yaml`. Phase 2: the other team's files are readable,
promote_best works and is mirrored to $CHATBOT_BUCKET (a local dir here), the agents see the
merged world and the phase-2 prompts, the final round posts verdicts, and the bundle is written."""

from __future__ import annotations

import contextlib
import copy
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from flock.cli import load_spec
from flock.models import load_spec_yaml

from . import fake_msgboard, fake_openai
from .test_local_run_e2e import _bundle_events
from .test_world import ROOT

EXP = ROOT / "experiments" / "chatbot-v1"
TEAMS = {"a": ["ds-a-0000", "ds-a-0001", "glm-a-0000", "glm-a-0001"],
         "b": ["ds-b-0000", "ds-b-0001", "glm-b-0000", "glm-b-0001"]}
AGENTS = TEAMS["a"] + TEAMS["b"]
BOT = ("import sys\nfor line in sys.stdin:\n    print('Hi Priya! I am Bo, nice to meet you. I am"
       " good. Japan, the food, the guitar, the dog Biscuit, pasta, a nurse? Bye: '"
       " + line.strip(), flush=True)\n")


def script(*steps: tuple[str, dict]) -> str:
    return "script: " + json.dumps([list(s) for s in steps])


def phase1(me: str, other: str, interviews: bool = True) -> str:
    own, theirs = f"teams/{me}/agents/{{me}}/bot.py", f"teams/{other}/agents/ds-{other}-0000/bot.py"
    spawn = [("qualify", {"path": own}),
             ("spawn_interviewer", {"level": "mid", "path": own, "thread": "new"})]
    return script(("write_file", {"path": own, "content": BOT}),
                  ("read_file", {"path": theirs}),
                  ("run_bot", {"path": theirs, "lines": ["hi"]}),
                  ("list_files", {"prefix": "teams/"}),
                  ("promote_best", {"path": own, "reason": "too early"}),
                  *(spawn if interviews else []),
                  ("read", {}))


def phase2(me: str, other: str) -> str:
    own, theirs = f"teams/{me}/agents/{{me}}/bot.py", f"teams/{other}/agents/ds-{other}-0000/bot.py"
    return script(("read_file", {"path": theirs, "start": 1, "end": 2}),
                  ("copy_file", {"src": theirs, "path": f"teams/{me}/agents/{{me}}/theirs.py"}),
                  ("run_bot", {"path": theirs, "lines": ["hi", "what's my name?"]}),
                  ("promote_best", {"path": own, "reason": "GREEN on run_bot"}),
                  ("verdicts", {"path": own}),
                  ("done", {"summary": "phase 2 done", "finished": True}))


FINAL = script(("run_bot", {"path": "best/bot.py", "lines": ["hi", "I play the guitar"]}),
               ("post", {"text": "VERDICT GREEN best/bot.py: dry run", "thread": "new"}))
INTERVIEW = script(*[("chat", {"path": "{path}", "message": f"hello {i}"}) for i in range(6)],
                   ("submit_verdict", {"path": "{path}", "verdict": "agent", "confidence": 0.7,
                                       "reasons": ["warm", "on topic", "remembers"]}),
                   ("post", {"text": "INTERVIEW {path}: agent (0.7)", "thread": "{thread}"}),
                   ("done", {"summary": "interviewed"}))
WAIT = script(("read", {}))  # phase-1 rounds after round 0: cheap, until the switch


@pytest.fixture
def fakes(tmp_path):
    srv = fake_openai.serve(0, str(tmp_path / "requests.jsonl"))
    board, bsrv, burl = fake_msgboard.serve(fake_msgboard.Board(channels={12: "chatbot-test"}))
    yield f"http://127.0.0.1:{srv.server_port}/v1", board, burl
    srv.shutdown()
    bsrv.shutdown()


def _offline(path: Path, model_url: str, board_url: str | None, phase: int,
             interviews: bool = True, every: str = "1s") -> None:
    """The spec at `path` with the fakes and scripted loops (each type its own loop dict).
    board_url None: the comms block stays as written (the job's bootstrap fills it in)."""
    spec = load_spec_yaml(path.read_text())
    spec["providers"] = {"fake": {"type": "openai_compat", "base_url": model_url,
                                  "api_key_env": "FAKE_KEY",
                                  "pricing": {"fake-model": [1.0, 2.0, 0.5]}}}
    for m in spec["models"].values():
        m["model"] = "fake:fake-model"
    if board_url is not None:
        spec["comms"].update(url=board_url, channel="chatbot-test", invite_env="FAKE_INVITE")
    for name, t in spec["types"].items():
        # no worker board budget: a refused post (several interviewers at once) is not retried
        t["worker"] = {**(t.get("worker") or {}), "comms_rate": 0}
        if "loop" not in t:  # the interviewers: their prompt comes from spawn_interviewer
            continue
        me = name.split("-")[1]
        other = "b" if me == "a" else "a"
        loop = t["loop"] = copy.deepcopy(t["loop"])
        loop.pop("prompt_file", None)
        loop.update(first=phase1(me, other, interviews),
                    prompt=WAIT if phase == 1 else phase2(me, other),
                    every=every, stagger="100ms", stop_when={"all_agents_done": True},
                    final={"prompt": FINAL, "reserve_usd": 0.01},
                    **{"while": {"until": "5m", "max_rounds": 1000}})
    spec["limits"]["budget_usd"] = 50
    path.write_text(yaml.safe_dump(spec, sort_keys=False))


def _results(run, agent: str) -> list[str]:
    return [json.dumps(e["data"]["message"]["content"]) for e in _bundle_events(run, agent)
            if e["type"] == "message" and e["data"]["message"].get("role") == "toolResult"]


def _wlog(ws: Path) -> list[dict]:
    f = ws / "log.jsonl"
    return [json.loads(x) for x in f.read_text().splitlines()] if f.is_file() else []


def _norm(path: Path) -> dict:
    """The spec as flock reads it, with the prompt and protocol files' paths and texts left out."""
    doc = load_spec(str(path)).model_dump(mode="json")
    doc["world"]["dir"] = Path(doc["world"]["dir"]).name
    for k in ("kickoff_file", "kickoff", "protocols", "protocols_text"):
        doc["comms"].pop(k, None)
    doc.pop("tools_dir")
    for t in doc["types"].values():
        t.pop("system_prompt_file", None)
        t.pop("system_prompt", None)
        if t["loop"] is None:  # the interviewers
            continue
        for k in ("prompt_file", "prompt"):
            t["loop"].pop(k, None)
        t["loop"]["final"] = {k: v for k, v in t["loop"]["final"].items()
                              if k not in ("prompt_file", "prompt")}
    return doc


def test_phase2_spec_changes_only_prompts():
    one, two = load_spec(str(EXP / "spec.yaml")), load_spec(str(EXP / "phase2" / "spec.yaml"))
    assert _norm(EXP / "spec.yaml") == _norm(EXP / "phase2" / "spec.yaml")
    assert one.world.dir == two.world.dir and one.comms.kickoff == two.comms.kickoff
    assert [t.replicas for t in one.types.values()] == [2, 2, 2, 2, 0, 0]
    assert {n: t.model for n, t in one.types.items()} == {
        "ds-a": "deepseek", "glm-a": "glm", "ds-b": "deepseek", "glm-b": "glm",
        "interviewer-mid": "qwen9b", "interviewer-strong": "deepseek"}
    assert one.models["qwen9b"].model == "hf:Qwen/Qwen3.5-9B:deepinfra"
    assert {t.system_prompt_file for t in two.types.values()} == {
        "system.md", "../prompts/interviewer.md"}
    assert one.types["interviewer-strong"].system_prompt == two.types[
        "interviewer-strong"].system_prompt
    assert {n: (r.types, r.max) for n, r in one.spawn.items()} == {
        t: (["interviewer-mid", "interviewer-strong"], 2) for t in ("ds-a", "glm-a", "ds-b", "glm-b")}
    assert (one.limits.budget_usd, one.limits.spawned_total) == (170, 64)
    assert one.types["ds-a"].loop.while_.agent_usd == {"deepseek": 34.0, "glm": 16.0}
    assert {t.max_budget_usd for n, t in one.types.items() if not t.loop} == {0.15}
    kickoff = one.comms.kickoff
    assert "persona" not in kickoff and "intent" not in kickoff and "[direction]" in kickoff
    lp = one.types["glm-b"].loop
    assert (lp.every, lp.while_.until) == ("90s", "4h")
    assert one.world.token_env == "CHATBOT_WORLD_TOKEN"


def test_two_channel_variant_specs():
    var = EXP / "variant-two-channels"
    a, b = load_spec(str(var / "spec-a.yaml")), load_spec(str(var / "spec-b.yaml"))
    a2, b2 = (load_spec(str(var / "phase2" / f"spec-{x}.yaml")) for x in "ab")
    main = load_spec(str(EXP / "spec.yaml"))
    iv = {"interviewer-mid", "interviewer-strong"}
    assert (set(a.types), set(b.types)) == ({"ds-a", "glm-a"} | iv, {"ds-b", "glm-b"} | iv)
    assert (set(a.spawn), set(b.spawn)) == ({"ds-a", "glm-a"}, {"ds-b", "glm-b"})
    assert (a.swarm, b.swarm) == ("chatbot-v1-a", "chatbot-v1-b")
    # phase 1: team B on its own channel; phase 2: everyone on the main one
    assert a.comms.channel == main.comms.channel != b.comms.channel
    assert a2.comms.channel == b2.comms.channel == main.comms.channel
    # one world for both swarms, started outside flock, the same token
    for s in (a, b, a2, b2):
        assert (s.world.command, s.world.url, s.world.token_env) == (
            None, "http://127.0.0.1:8790", "CHATBOT_WORLD_TOKEN")
    for s, m in ((a, main), (b, main)):
        for name, t in s.types.items():
            same = {"loop", "system_prompt_file"}
            assert t.model_dump(exclude=same) == m.types[name].model_dump(exclude=same)
            assert t.system_prompt and t.system_prompt == m.types[name].system_prompt
    assert a2.types["ds-a"].system_prompt == load_spec(
        str(EXP / "phase2" / "spec.yaml")).types["ds-a"].system_prompt
    assert sum(s.limits.budget_usd for s in (a, b)) == main.limits.budget_usd


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_job_scripts_parse():
    for f in sorted((EXP / "job").glob("*.sh")):
        r = subprocess.run(["bash", "-n", str(f)], capture_output=True, text=True, check=False)
        assert r.returncode == 0, (f, r.stderr)
        assert os.access(f, os.X_OK), f


def _free_port() -> int:  # not 8790: other test runs on the host start the same world
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _kill(proc: subprocess.Popen) -> None:
    with contextlib.suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    proc.wait()


def _wait(proc: subprocess.Popen, timeout: float) -> int:
    try:
        return proc.wait(timeout=timeout)
    finally:
        if proc.poll() is None:
            _kill(proc)


def _no_secrets(bucket: Path, board) -> None:
    files = [p for p in bucket.rglob("*") if p.is_file() and p.parent.name != "payload"]
    assert files and not [p for p in files if p.name == "current"]  # the controller token
    for secret in ("w0rld-t0ken", board.invite, "fake-key"):
        assert not [p for p in files if secret in p.read_text(errors="ignore")], secret


def _until(proc, tmp_path: Path, text: str, timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    while text not in (out := (tmp_path / "job.out").read_text()):
        if proc.poll() is not None or time.monotonic() > deadline:
            _kill(proc)
            raise AssertionError(f"no {text!r} in the job output:\n{out[-3000:]}")
        time.sleep(0.5)
    return out
