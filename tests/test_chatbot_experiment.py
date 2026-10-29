"""experiments/chatbot-v1 end to end under `flock run` (offline): the real CLI in a subprocess runs
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
import sys
import time
from pathlib import Path

import httpx
import pytest
import yaml

from flock.cli import load_spec
from flock.models import load_spec_yaml

from . import fake_msgboard, fake_openai
from .test_local_run_e2e import _bundle_events, _env, _flock, _run_dir
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


def test_chatbot_experiment_dry_run(tmp_path, fakes):
    model_url, board, board_url = fakes
    exp, ws, bucket = tmp_path / "exp", tmp_path / "ws", tmp_path / "bucket"
    shutil.copytree(EXP, exp, ignore=shutil.ignore_patterns(".flock", "*.local.yaml"))
    tool = exp / "tools" / "interview.py"  # the interviewer's task: a script for the fake model
    src = tool.read_text()
    a = src.index("PROMPT = (")
    tool.write_text(src[:a] + f"PROMPT = {INTERVIEW!r}" + src[src.index("\n\n\n", a):])
    for w in ("chatbot", "eliza"):  # chatbot.py imports the Eliza world from ../eliza
        shutil.copytree(ROOT / "worlds" / w, tmp_path / "worlds" / w,
                        ignore=shutil.ignore_patterns("ws", "__py*"))
    for spec_file, phase, wdir in ((exp / "spec.yaml", 1, "../worlds/chatbot"),
                                   (exp / "phase2" / "spec.yaml", 2, "../../worlds/chatbot")):
        _offline(spec_file, model_url, board_url, phase)
        doc = yaml.safe_load(spec_file.read_text())
        doc["world"]["dir"] = wdir  # the copied worlds, at the same relative place
        spec_file.write_text(yaml.safe_dump(doc, sort_keys=False))
    env = {**_env(tmp_path), "FAKE_INVITE": board.invite, "CHATBOT_WORLD_TOKEN": "w0rld-t0ken",
           "CHATBOT_WS": str(ws), "CHATBOT_BUCKET": str(bucket)}
    run_out = open(tmp_path / "run.out", "w")  # noqa: SIM115 (closed below)
    proc = subprocess.Popen(_flock("run", "spec.yaml", "--until", "idle", "--skip-preflight"),
                            cwd=exp, env=env, stdout=run_out, stderr=subprocess.STDOUT, text=True)
    try:
        # phase 1: every agent's round 0 ran into the refused promotion
        deadline = time.monotonic() + 120
        while ({x["agent"] for x in _wlog(ws) if x.get("tool") == "promote_best"} != set(AGENTS)
               or sum(x["action"] == "verdict" for x in _wlog(ws)) < 6
               or len({x["agent"] for x in _wlog(ws) if x.get("why") == "quota"}) < 2):
            assert proc.poll() is None and time.monotonic() < deadline, (
                [(x["agent"], x["action"], x.get("tool"), x.get("why")) for x in _wlog(ws)
                 if x["action"] in ("verdict", "refused", "qualify")],
                (tmp_path / "run.out").read_text()[-3000:])
            time.sleep(0.5)
        assert not (ws / "best" / "bot.py").exists()

        # the operator's switch: the world first, then the phase-2 prompts
        world_py = str(tmp_path / "worlds" / "chatbot" / "chatbot.py")
        m = subprocess.run([sys.executable, world_py, "merge"], env=env, capture_output=True,
                           text=True, timeout=30, check=False)
        assert m.returncode == 0 and json.loads(m.stdout)["phase"] == "merged", m.stdout + m.stderr
        bad = subprocess.run([sys.executable, world_py, "status"], capture_output=True, text=True,
                             env={**env, "CHATBOT_WORLD_TOKEN": "wrong"}, timeout=30, check=False)
        assert bad.returncode == 1 and "unauthorized" in bad.stdout
        u = subprocess.run(_flock("update", "phase2/spec.yaml"), cwd=exp, env=env,
                           capture_output=True, text=True, timeout=60, check=False)
        assert u.returncode == 0, u.stdout + u.stderr
        assert "prompts changed" in u.stdout and "loop" in u.stdout, u.stdout
        proc.wait(timeout=180)
    finally:
        if proc.poll() is None:
            proc.kill()
        run_out.close()
    out = (tmp_path / "run.out").read_text()
    assert proc.returncode == 0 and "stopping (idle)" in out, out[-3000:]
    run = _run_dir(exp)

    # the world ran as a child process and stopped with the run
    url = json.loads((ws / "world.json").read_text())["url"]
    with pytest.raises(httpx.ConnectError):
        httpx.get(f"{url}/manifest", timeout=2)

    wlog = _wlog(ws)
    assert [(x["action"], x["phase"]) for x in wlog if x["agent"] == "operator"] == [
        ("phase", "merged")]
    for team, other in (("a", "b"), ("b", "a")):
        for a in TEAMS[team]:
            res = _results(run, a)
            theirs = f"teams/{other}/agents/ds-{other}-0000/bot.py"
            # phase 1: the other team's files and the promotion were refused
            assert sum(f"teams/{other}/ belongs to team {other.upper()} until the merge" in r
                       for r in res) == 2, res
            assert any("promotion opens after the merge" in r for r in res), res
            listing = next(r for r in res if '\\"total\\"' in r)
            assert f"teams/{team}/agents/" in listing and f"teams/{other}/" not in listing
            # phase 2: read, copied and ran the other team's bot, promoted its own
            assert any(f"# {theirs}: lines 1-2 of 3" in r for r in res), res
            assert (ws / f"teams/{team}/agents/{a}/theirs.py").read_text() == BOT
            assert any('\\"promoted\\": true' in r for r in res), res
            assert {(x["action"], x.get("tool")) for x in wlog if x["agent"] == a} >= {
                ("refused", "read_file"), ("refused", "run_bot"), ("refused", "promote_best"),
                ("write_file", None), ("copy_file", None), ("promote", None)}
    # interviews: 3 per team (the quota), each 6 chat turns, a verdict, a post; the 4th refused
    verdicts = [json.loads(x) for x in (ws / "verdicts.jsonl").read_text().splitlines()]
    assert sorted(v["team"] for v in verdicts) == ["a"] * 3 + ["b"] * 3
    for v in verdicts:
        assert (v["verdict"], v["level"], v["turns"]) == ("agent", "mid", 6)
        assert v["interviewer"].startswith(v["path"].split("/")[3] + "-interviewer-mid-")
        assert v["transcript"][0]["bot"].endswith("hello 0")
    assert len({x["agent"] for x in wlog if x.get("why") == "quota"}) == 2  # 1 per team
    assert len([m for m in board.msgs if m["body"].startswith("INTERVIEW teams/")]) >= 6
    # the interviewers are ephemeral: each removed when its session ended (the quota-refused
    # ones too), its files gone from the worker, its events kept in the bundle
    gone = [e["data"] for a in AGENTS for e in _bundle_events(run, a) if e["type"] == "child_removed"]
    assert len(gone) == 8 and {g["reason"] for g in gone} == {"ephemeral"}, gone
    assert all((run / "events" / f"{g['child']}.jsonl").is_file() for g in gone)
    removed = json.loads((run / "run.json").read_text())["removed_agents"]
    assert sorted(r["name"] for r in removed) == sorted(g["child"] for g in gone)
    assert not [p for p in (run / "agents").iterdir() if "-interviewer-" in p.name]
    assert not list((run / "state").rglob("*-interviewer-*"))
    for a in AGENTS:  # phase 2: the team reads its verdicts
        res, mine = _results(run, a), [v for v in verdicts if f"/{a}/" in v["path"]]
        seen = next(r for r in res if '\\"quota\\"' in r and '\\"verdicts\\"' in r)
        assert all(v["interviewer"] in seen for v in mine), seen
    hist = [json.loads(x) for x in (ws / "best" / "history.jsonl").read_text().splitlines()]
    # every agent promoted; chatbot-v1's gate accepts ties ("at least as well"), so an agent
    # whose phase-2 round ran twice (a repeated loop prompt) is recorded twice: count agents
    assert {h["agent"] for h in hist} == set(AGENTS)
    assert all(h["score"] == h["of"] for h in hist), hist

    # the mirror: best/ and the phase in the bucket (a local dir here)
    assert (bucket / "best" / "bot.py").read_text() == (ws / "best" / "bot.py").read_text()
    assert (bucket / "best" / "history.jsonl").read_text() == (
        ws / "best" / "history.jsonl").read_text()
    assert json.loads((bucket / "phase.json").read_text())["phase"] == "merged"
    assert not any(x["action"] == "mirror_error" for x in wlog)

    # the agents saw phase 1, then the merged world and the phase-2 prompts
    reqs = [json.loads(x)["body"] for x in (tmp_path / "requests.jsonl").read_text().splitlines()]
    systems = [r["messages"][0]["content"] for r in reqs]
    assert any("## World: chatbot (version 0.1.0-teams)" in s and "You are on TEAM A" in s
               for s in systems)
    assert any("TEAM B" in s and "Your team chooses its own direction" in s for s in systems)
    assert not any("persona + small" in s or "intent-driven" in s for s in systems)
    assert any("## World: chatbot (version 0.1.0-merged)" in s and "PHASE 2" in s
               and "Merged team rules" in s for s in systems)
    users = [json.dumps(r["messages"]) for r in reqs]
    assert any("the world changed since your last turn" in u for u in users)
    offered = {frozenset(t["function"]["name"] for t in r["tools"]) for r in reqs}
    assert offered == {frozenset({
        "read_file", "write_file", "copy_file", "edit_file", "insert_after", "list_files",
        "run_bot", "promote_best", "best_status", "qualify", "verdicts", "spawn_interviewer",
        "post", "read", "search", "done"}), frozenset({"chat", "submit_verdict", "post", "done"})}
    assert len([m for m in board.msgs if "VERDICT GREEN" in m["body"]]) == len(AGENTS)

    # the bundle: run.json with each type's loop (stopped by done, then the final round)
    doc = json.loads((run / "run.json").read_text())
    for t in ("ds-a", "glm-a", "ds-b", "glm-b"):
        lp = doc["loops"][t]
        assert (lp["state"], lp["final_sent"], lp["final_done"]) == ("done", 2, 2), lp
    assert doc["budget"]["budget_usd"] == 50 and doc["budget"]["stop"] is None
    assert (run / "events.jsonl").is_file()
    for secret in ("w0rld-t0ken", board.invite, "fake-key"):  # no secret in the bundle
        assert not any(secret in p.read_text(errors="ignore") for p in run.rglob("*")
                       if p.is_file() and p.suffix in (".json", ".jsonl", ".log", ".md", ""))


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
    assert (lp.every, lp.stagger, lp.while_.until) == ("90s", "4s", "4h")
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


def _job(tmp_path, model_url, board, board_url, variant: str = "one", **extra: str) -> tuple[
        subprocess.Popen, dict, Path]:
    """The job's bootstrap.sh, run locally: CHATBOT_PREINSTALLED (this flock), a directory as
    the bucket, its payload (repo.tar.gz) made from this checkout with the fakes in the specs.
    The process leads its own group: a failing test kills all of it (_kill)."""
    import tarfile

    src, bucket = tmp_path / "src", tmp_path / "bucket"
    shutil.copytree(EXP, src / "experiments" / "chatbot-v1",
                    ignore=shutil.ignore_patterns(".flock*", "*.local.yaml"))
    for w in ("chatbot", "eliza"):
        shutil.copytree(ROOT / "worlds" / w, src / "worlds" / w,
                        ignore=shutil.ignore_patterns("ws", "__py*"))
    for f, phase in (("spec.yaml", 1), ("phase2/spec.yaml", 2),
                     ("variant-two-channels/spec-a.yaml", 1),
                     ("variant-two-channels/spec-b.yaml", 1),
                     ("variant-two-channels/phase2/spec-a.yaml", 2),
                     ("variant-two-channels/phase2/spec-b.yaml", 2)):
        # the variant's phase 2 waits a round (8 s) so a moved channel reaches every worker
        # first (comms changes travel with the heartbeat, every 5 s locally)
        slow = variant != "one" and phase == 2
        _offline(src / "experiments" / "chatbot-v1" / f, model_url, None, phase, False,
                 every="8s" if slow else "1s")
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
           "CHATBOT_FLOCK_ARGS": "--skip-preflight", "CHATBOT_VARIANT": variant,
           "CHATBOT_CHANNEL_B": "chatbot-b", "CHATBOT_WORLD_PORT": str(_free_port()), **extra}
    out = open(tmp_path / "job.out", "w")  # noqa: SIM115 (the process holds it)
    proc = subprocess.Popen(["bash", "-c", (EXP / "job" / "bootstrap.sh").read_text()], env=env,
                            stdout=out, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)
    deadline, ws = time.monotonic() + 120, tmp_path / "job" / "ws"
    try:
        while sum(x.get("tool") == "promote_best" for x in _wlog(ws)) < len(AGENTS):
            assert proc.poll() is None and time.monotonic() < deadline, (
                (tmp_path / "job.out").read_text()[-3000:])
            time.sleep(0.5)
    except BaseException:
        _kill(proc)
        raise
    return proc, env, bucket


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


def test_job_bootstrap_rehearsal_with_the_switch_marker(tmp_path, fakes):
    """bootstrap.sh end to end without a job: payload in, flock run, the operator's control
    marker triggers job/switch.sh (merge + flock update), periodic and final syncs out."""
    model_url, board, board_url = fakes
    proc, _, bucket = _job(tmp_path, model_url, board, board_url)
    (bucket / "control").mkdir()
    (bucket / "control" / "switch").write_text("go\n")  # what the operator uploads
    assert _wait(proc, 180) == 0, (tmp_path / "job.out").read_text()[-3000:]
    out = (tmp_path / "job.out").read_text()
    assert "phase switch ok" in out and "final sync" in out, out
    switch = (bucket / "control" / "switch.log").read_text()
    assert '"phase":"merged"' in switch and "prompts changed" in switch, switch
    (run,) = (bucket / "flock" / ".flock").iterdir()
    doc = json.loads((run / "run.json").read_text())
    assert doc["stop_reason"] == "idle" and doc["loops"]["glm-b"]["state"] == "done"
    assert (bucket / "ws" / "best" / "bot.py").read_text() == (bucket / "best" / "bot.py").read_text()
    assert json.loads((bucket / "ws" / "phase.json").read_text())["phase"] == "merged"
    assert "stopping (idle)" in (bucket / "logs" / "run.out").read_text()
    _no_secrets(bucket, board)


def test_job_bootstrap_cancel_still_writes_the_bundle(tmp_path, fakes):
    """`hf jobs cancel` sends SIGTERM: flock drains and writes its bundle, then the last sync."""
    model_url, board, board_url = fakes
    proc, _, bucket = _job(tmp_path, model_url, board, board_url)
    proc.send_signal(signal.SIGTERM)
    _wait(proc, 90)
    out = (tmp_path / "job.out").read_text()
    assert "stop requested" in out and "final sync" in out, out
    (run,) = (bucket / "flock" / ".flock").iterdir()
    assert json.loads((run / "run.json").read_text())["stop_reason"] == "interrupted"
    assert (bucket / "ws" / "log.jsonl").is_file() and (bucket / "logs" / "run.out").is_file()
    _no_secrets(bucket, board)


def test_job_bootstrap_two_channel_variant(tmp_path, fakes):
    """CHATBOT_VARIANT=two-channels: one world started by the bootstrap, two swarms; team B
    posts on its own channel in phase 1 and on the main one after the switch."""
    model_url, _, _ = fakes
    board, bsrv, board_url = fake_msgboard.serve(
        fake_msgboard.Board(channels={12: "chatbot-test", 13: "chatbot-b"}))
    try:
        proc, _, bucket = _job(tmp_path, model_url, board, board_url, "two-channels")
        b_phase1 = {m["channel_id"] for m in board.msgs if "-b-" in m["sender"]["external_id"]}
        (bucket / "control").mkdir()
        (bucket / "control" / "switch").write_text("go\n")
        assert _wait(proc, 180) == 0, (tmp_path / "job.out").read_text()[-3000:]
    finally:
        bsrv.shutdown()
    assert b_phase1 == {13}, b_phase1  # the B host's kickoff
    verdicts = {(m["channel_id"], m["sender"]["external_id"]) for m in board.msgs
                if m["body"].startswith("VERDICT")}
    assert verdicts == {(12, a) for a in AGENTS}, verdicts
    assert any(m["channel_id"] == 12 and m["body"].startswith("Phase 2 (chatbot-v1): team B")
               for m in board.msgs)
    for t in "ab":
        (run,) = (bucket / "flock" / f".flock-{t}").iterdir()
        doc = json.loads((run / "run.json").read_text())
        assert doc["swarm"] == f"chatbot-v1-{t}" and doc["stop_reason"] == "idle"
    switch = (bucket / "control" / "switch.log").read_text()
    assert switch.count("prompts changed") == 2 and '"phase":"merged"' in switch
    _no_secrets(bucket, board)


def test_job_bootstrap_stop_marker_writes_the_bundle(tmp_path, fakes):
    """<bucket>/control/stop: the graceful stop (a cancel leaves the job only ~3 s)."""
    model_url, board, board_url = fakes
    proc, _, bucket = _job(tmp_path, model_url, board, board_url)
    (bucket / "control").mkdir()
    (bucket / "control" / "stop").write_text("stop\n")
    _wait(proc, 90)
    out = (tmp_path / "job.out").read_text()
    assert "stop marker found" in out and "final sync" in out and "done" in out, out
    (run,) = (bucket / "flock" / ".flock").iterdir()
    assert json.loads((run / "run.json").read_text())["stop_reason"] == "interrupted"
    assert "stop marker found" in (bucket / "logs" / "bootstrap.log").read_text()
    _no_secrets(bucket, board)


def _until(proc, tmp_path: Path, text: str, timeout: float = 120) -> str:
    deadline = time.monotonic() + timeout
    while text not in (out := (tmp_path / "job.out").read_text()):
        if proc.poll() is not None or time.monotonic() > deadline:
            _kill(proc)
            raise AssertionError(f"no {text!r} in the job output:\n{out[-3000:]}")
        time.sleep(0.5)
    return out


def test_job_bootstrap_live_update_switch_and_stop_in_smoke_mode(tmp_path, fakes):
    """The operator's markers in a smoke-mode run: control/update re-syncs <bucket>/experiment/
    and runs `flock update` (marker deleted after), the switch re-syncs it too (phase-2 edits
    count), control/stop ends with a bundle. CHATBOT_RUN names a fresh swarm and kickoff."""
    model_url, board, board_url = fakes
    proc, _, bucket = _job(tmp_path, model_url, board, board_url, CHATBOT_SMOKE="1",
                           CHATBOT_RUN="run2")
    edited = bucket / "experiment"
    shutil.copytree(tmp_path / "src" / "experiments" / "chatbot-v1", edited)
    for f, gap in (("spec.yaml", "5s"), ("phase2/spec.yaml", "5s")):
        (edited / f).write_text((edited / f).read_text().replace("min_post_gap: 10s",
                                                                 f"min_post_gap: {gap}"))
    (bucket / "control").mkdir()
    (bucket / "control" / "update").write_text("go\n")  # 1. live update
    _until(proc, tmp_path, "update 1 ok")
    log1 = (bucket / "control" / "update-1.log").read_text()
    assert "comms.rules.min_post_gap: 10s -> 5s" in log1, log1
    assert not (bucket / "control" / "update").exists()  # the marker is consumed
    p2 = edited / "phase2" / "spec.yaml"  # 2. a phase-2 edit, picked up by the switch
    p2.write_text(p2.read_text().replace("min_post_gap: 5s", "min_post_gap: 7s"))
    (bucket / "control" / "switch").write_text("go\n")
    _until(proc, tmp_path, "phase switch ok")
    switch = (bucket / "control" / "switch.log").read_text()
    assert '"phase":"merged"' in switch and "min_post_gap: 5s -> 7s" in switch, switch
    (bucket / "control" / "stop").write_text("stop\n")  # 3. the graceful stop
    _wait(proc, 90)
    out = (tmp_path / "job.out").read_text()
    assert "smoke settings" in out and "stop marker found" in out and "done" in out, out
    (run,) = (bucket / "flock" / ".flock").iterdir()
    doc = json.loads((run / "run.json").read_text())
    assert doc["swarm"] == "chatbot-v1-run2" and doc["stop_reason"] == "interrupted"
    assert doc["budget"]["budget_usd"] == 1.0  # smoke caps survive the re-renders
    assert board.msgs[0]["body"].startswith("Kickoff (chatbot-v1 run2)")
    _no_secrets(bucket, board)
