"""worlds/chatbot in process: team homes, the phase rules (cross-team reads and promote_best
refused before the merge, open after), the operator's /operator/phase route behind the world
token, the bucket mirror (huggingface_hub faked) and the `!` probe keywords."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import time
import types
import typing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from flock.world import Agent, ToolError

from .test_world import ROOT

BOT = ("import sys\nfor line in sys.stdin:\n    print('Hi Priya! Japan sounds great, the food,"
       " the guitar, the dog Biscuit. I am Bo, nice to meet you, how about pasta? I am good,"
       " bye: ' + line.strip(), flush=True)\n")
A, A2, B = Agent("ds-a-0000", "ds-a", turn="t-1"), Agent("glm-a-0001"), Agent("ds-b-0001")
PA, PB = "teams/a/agents/ds-a-0000/bot.py", "teams/b/agents/ds-b-0001/bot.py"


def _load():
    s = importlib.util.spec_from_file_location("chatbot", ROOT / "worlds/chatbot/chatbot.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load()


@pytest.fixture
def world(mod, tmp_path, monkeypatch):
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.ChatbotWorld(str(tmp_path / "ws"))
    w.write_file(A, PA, BOT)
    w.write_file(B, PB, BOT)
    return w


def _log(w) -> list[dict]:
    return [json.loads(x) for x in (w.ws / "log.jsonl").read_text().splitlines()]


def test_manifest_carries_the_phase(world):
    m = world.manifest()
    assert (m["name"], m["version"]) == ("chatbot", "0.1.0-teams")
    assert "Phase 1 (teams)" in m["description"] and "Not a therapist" in m["description"]
    assert [t["name"] for t in m["tools"]] == [
        "read_file", "write_file", "copy_file", "edit_file", "insert_after", "list_files",
        "run_bot", "promote_best", "best_status", "qualify", "chat", "submit_verdict", "verdicts"]
    assert m["tools"][-2]["parameters"]["properties"]["verdict"]["enum"] == ["agent", "script"]
    assert all("agent" not in t["parameters"]["properties"] for t in m["tools"])
    assert sorted(p.name for p in (world.ws / "probes").glob("*.txt")) == [
        "greeting.txt", "memory.txt", "robust.txt", "smalltalk.txt"]


def test_writes_go_to_the_team_home_only(world):
    for agent, path in ((A, PB), (A2, PA), (A, "best/bot.py"), (A, "agents/ds-a-0000/bot.py")):
        with pytest.raises(ToolError, match="may write only under teams/") as e:
            world.write_file(agent, path, BOT)
        assert e.value.code == "not_yours"
    assert world.write_file(A2, "teams/a/agents/glm-a-0001/x.md", "hi")["written"]
    with pytest.raises(ToolError, match="has no team"):
        world.write_file(Agent("ds-0000"), "teams/a/agents/ds-0000/bot.py", BOT)


def test_phase_teams_refuses_the_other_team_and_promotion(world):
    calls = {"read_file": lambda: world.read_file(PB, agent=A),
             "copy_file": lambda: world.copy_file(A, PB, "teams/a/agents/ds-a-0000/b.py"),
             "run_bot": lambda: asyncio.run(world.run_bot(PB, ["hi"], agent=A))}
    for tool, call in calls.items():
        with pytest.raises(ToolError, match="teams/b/ belongs to team B until the merge") as e:
            call()
        assert e.value.code == "refused"
        assert _log(world)[-1] | {"tool": tool, "path": PB} == _log(world)[-1]
    with pytest.raises(ToolError, match="teams/a/ belongs to team A"):
        world.read_file("teams/a/../a/agents/ds-a-0000/bot.py", agent=B)
    # own team, best/ and probes/ stay readable; the listing hides the other team
    assert world.read_file(PA, agent=A2).startswith(f"# {PA}: lines 1-")
    assert world.read_file("probes/memory.txt", agent=B)
    assert asyncio.run(world.run_bot(PA, ["hi"], agent=A2))["status"] == "ok"
    seen = {f["path"] for f in world.list_files(agent=A)["files"]}
    assert PA in seen and PB not in seen and "probes/greeting.txt" in seen
    assert not world.list_files("teams/b/", agent=A)["files"]
    # promotion is closed, with the reason the agent reads
    with pytest.raises(ToolError, match="promotion opens after the merge"):
        asyncio.run(world.promote_best(A, PA, "too early"))
    assert not (world.ws / "best" / "bot.py").exists()
    assert (_log(world)[-1]["action"], _log(world)[-1]["tool"]) == ("refused", "promote_best")


def test_merge_opens_everything_and_persists(mod, world):
    r = world.set_phase("merged")
    assert (r["phase"], r["before"], r["version"]) == ("merged", "teams", "0.1.0-merged")
    assert "Phase 2 (merged)" in world.manifest()["description"]
    assert world.read_file(PB, agent=A).startswith(f"# {PB}")
    assert world.copy_file(A, PB, "teams/a/agents/ds-a-0000/from-b.py")["written"]
    assert {PA, PB} <= {f["path"] for f in world.list_files(agent=A)["files"]}
    p = asyncio.run(world.promote_best(B, PB, "GREEN on run_bot"))
    assert p["promoted"] and p["best"]["score"] == p["best"]["of"], p
    assert (world.ws / "best" / "bot.py").read_text() == BOT
    with pytest.raises(ToolError, match="may write only under teams/b/agents/ds-b-0001/"):
        world.write_file(B, PA, BOT)  # writes stay in the own dir after the merge
    assert mod.ChatbotWorld(str(world.ws)).phase == "merged"  # a restarted world keeps it
    assert [x["action"] for x in _log(world) if x["agent"] == "operator"] == ["phase"]
    with pytest.raises(ToolError, match="phase must be one of"):
        world.set_phase("chaos")


def test_operator_route_needs_the_token_and_no_agent(mod, world):
    c = TestClient(mod.make_app(world, "s3cret"))
    assert c.post("/operator/phase", json={"phase": "merged"}).status_code == 401
    head = {"X-World-Token": "s3cret"}
    r = c.post("/operator/phase", json={"phase": "merged"},
               headers={**head, "X-Flock-Agent": "ds-a-0000"})
    assert r.status_code == 403 and world.phase == "teams"
    assert c.post("/operator/phase", json={"phase": "x"}, headers=head).status_code == 400
    assert c.post("/tools/promote_best", json={"args": {"path": PA, "reason": "r"}},
                  headers={**head, "X-Flock-Agent": "ds-a-0000"}).json()["error"]["code"] == (
        "refused")
    r = c.post("/operator/phase", json={"phase": "merged"}, headers=head)
    assert r.status_code == 200 and r.json()["phase"] == "merged", r.text
    assert c.get("/operator/phase", headers=head).json() == {
        "phase": "merged", "version": "0.1.0-merged"}
    assert c.get("/manifest", headers=head).json()["version"] == "0.1.0-merged"
    r = c.post("/tools/promote_best", json={"args": {"path": PA, "reason": "r"}},
               headers={**head, "X-Flock-Agent": "ds-a-0000"}).json()
    assert r["ok"] and r["result"]["promoted"], r


class FakeApi:
    calls: typing.ClassVar[list] = []
    fail = False

    def batch_bucket_files(self, bucket_id, *, add=None, **_):
        if FakeApi.fail:
            raise RuntimeError("bucket down")
        FakeApi.calls.append((bucket_id, [(local, remote) for local, remote in add]))


@pytest.fixture
def hub(monkeypatch):
    FakeApi.calls, FakeApi.fail = [], False
    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        types.SimpleNamespace(HfApi=FakeApi))
    return FakeApi


def test_mirror_to_the_bucket_after_each_promotion_and_the_merge(world, hub, monkeypatch):
    world.set_phase("merged")
    assert hub.calls == []  # CHATBOT_BUCKET unset: no-op
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/me/chatbots/run-1/")
    assert asyncio.run(world.promote_best(A, PA, "seed"))["promoted"]
    bucket, files = hub.calls[-1]
    assert bucket == "me/chatbots"
    assert [r for _, r in files] == ["run-1/best/bot.py", "run-1/best/v1.py",
                                     "run-1/best/history.jsonl", "run-1/log.jsonl"]
    assert all(Path(local).read_text() == BOT for local, r in files if r.endswith(".py"))
    world.set_phase("teams")
    assert [r for _, r in hub.calls[-1][1]] == ["run-1/phase.json", "run-1/log.jsonl"]
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/me/chatbots")  # no prefix
    world.set_phase("merged")
    assert [r for _, r in hub.calls[-1][1]] == ["phase.json", "log.jsonl"]
    hub.fail = True  # a failing bucket never fails the promotion
    assert asyncio.run(world.promote_best(B, PB, "tie"))["promoted"]
    assert _log(world)[-1]["action"] == "mirror_error" and "bucket down" in _log(world)[-1]["error"]


def test_mirror_to_a_local_dir(world, tmp_path, monkeypatch):
    monkeypatch.setenv("CHATBOT_BUCKET", str(tmp_path / "bucket"))
    world.set_phase("merged")
    asyncio.run(world.promote_best(A, PA, "seed"))
    assert (tmp_path / "bucket" / "best" / "v1.py").read_text() == BOT
    assert json.loads((tmp_path / "bucket" / "phase.json").read_text())["phase"] == "merged"


def test_bang_keywords_refuse_therapist_replies(world):
    world.set_phase("merged")
    therapist = BOT.replace("bye: ", "bye. How does that make you feel? ")
    world.write_file(A, "teams/a/agents/ds-a-0000/t.py", therapist)
    s = world._score(world.ws / "teams/a/agents/ds-a-0000/t.py")
    assert s["probes"]["greeting.txt"].startswith("RED: ")
    assert "says 'how does that make you feel'" in s["probes"]["greeting.txt"]
    assert s["score"] < world._score(world.ws / PA)["score"] == s["of"]


IA, IB = Agent("ds-a-0000-interviewer-mid-0003"), Agent("glm-b-0001-interviewer-strong-0004")


def _chat(w, agent, path, msg="hi"):
    return asyncio.run(w.chat(agent, path, msg))


def test_qualify_gates_the_interview_and_a_change_ends_it(world):
    with pytest.raises(ToolError, match="not qualified or changed") as e:
        _chat(world, IA, PA)  # before qualify
    assert e.value.code == "not_qualified"
    q = asyncio.run(world.qualify(A, PA))
    assert q["qualified"] and q["fails"] == [] and "score" not in q
    assert q["quota"] == {"team": "a", "left": 3, "per_hour": 3, "strong_counts": 2,
                          "next_unit_in_s": 0}
    r = _chat(world, IA, PA, "hello there")
    assert r == {"turn": 1, "reply": BOT.split("'")[1] + "hello there"}
    assert _chat(world, IA, PA, "and you?")["turn"] == 2  # the conversation goes on
    with pytest.raises(ToolError, match="talk first: 2 of 6"):
        world.submit_verdict(IA, PA, "script", 0.9, ["a", "b", "c"])
    for i in range(4):
        _chat(world, IA, PA, f"line {i}")
    v = world.submit_verdict(IA, PA, "agent", 1.5, ["warm", "remembers", "jokes", "extra"])
    assert (v["verdict"], v["confidence"], v["reasons"], v["turns"], v["level"]) == (
        "agent", 1.0, ["warm", "remembers", "jokes"], 6, "mid")
    seen = world.verdicts(A, PA)
    assert [x["interviewer"] for x in seen["verdicts"]] == [IA.name]
    assert seen["verdicts"][0]["last_turns"][-1]["user"] == "line 3" and seen["quota"]["left"] == 2
    world.edit_file(A, PA, "Hi Priya!", "Hey Priya!")  # changed since qualification
    with pytest.raises(ToolError, match="changed since qualification"):
        _chat(world, IA, PA)
    with pytest.raises(ToolError, match="changed since qualification"):
        world.submit_verdict(IA, PA, "agent", 0.5, ["x"])


def test_unqualified_and_non_interviewers_are_refused(world):
    bad = BOT.replace("print(", "print('' if '?' not in line else ")
    world.write_file(A2, "teams/a/agents/glm-a-0001/bad.py", bad)
    q = asyncio.run(world.qualify(A2, "teams/a/agents/glm-a-0001/bad.py"))
    assert not q["qualified"] and "greeting.txt: no reply to 'hi there!'" in q["fails"]
    # the bar is mechanical: a repeated reply or a timeout fails, keywords do not matter
    world.write_file(A2, "teams/a/agents/glm-a-0001/same.py",
                     "import sys\nfor line in sys.stdin:\n    print('Sure.', flush=True)\n")
    q = asyncio.run(world.qualify(A2, "teams/a/agents/glm-a-0001/same.py"))
    assert q["fails"][0] == "greeting.txt: repeated reply 'sure.'" and len(q["fails"]) == 4
    world.write_file(A2, "teams/a/agents/glm-a-0001/slow.py", "while True:\n    pass\n")
    q = asyncio.run(world.qualify(A2, "teams/a/agents/glm-a-0001/slow.py"))
    assert q["fails"][0].startswith("greeting.txt: timeout after 5 s")
    world.write_file(A2, "teams/a/agents/glm-a-0001/plain.py",
                     "import sys\nfor line in sys.stdin:\n    print('You said ' + line.strip()"
                     " + '!', flush=True)\n")
    assert asyncio.run(world.qualify(A2, "teams/a/agents/glm-a-0001/plain.py"))["qualified"]
    asyncio.run(world.qualify(A, PA))
    with pytest.raises(ToolError, match="only interviewer agents"):
        _chat(world, A, PA)  # a team agent cannot chat or vote itself
    with pytest.raises(ToolError, match="only interviewer agents"):
        world.submit_verdict(A2, PA, "agent", 1.0, ["x"])
    with pytest.raises(ToolError, match="teams/a/ belongs to team A"):
        _chat(world, IB, PA)  # phase 1: a team B interviewer cannot reach team A
    with pytest.raises(ToolError, match="teams/b/ belongs to team B"):
        asyncio.run(world.qualify(A, PB))
    world.set_phase("merged")
    assert _chat(world, IB, PA)["turn"] == 1  # after the merge it can


def test_quota_per_team_and_hour_strong_counts_double(world, monkeypatch, mod):
    asyncio.run(world.qualify(A, PA))
    asyncio.run(world.qualify(B, PB))
    _chat(world, Agent("ds-a-0000-interviewer-strong-0001"), PA)  # 2 units
    _chat(world, Agent("ds-a-0001-interviewer-mid-0002"), PA)  # 1 unit: 3 used
    with pytest.raises(ToolError, match="quota is used up") as e:
        _chat(world, Agent("glm-a-0000-interviewer-mid-0005"), PA)
    assert e.value.code == "quota"
    assert _chat(world, IB, PB)["turn"] == 1  # team B has its own quota
    assert asyncio.run(world.qualify(A, PA))["quota"]["left"] == 0
    now = time.time()
    monkeypatch.setattr(mod.time, "time", lambda: now + 3601)  # an hour later
    assert _chat(world, Agent("glm-a-0000-interviewer-mid-0005"), PA)["turn"] == 1


def test_spawn_interviewer_tool_spawns_then_prompts_once_the_child_is_up(tmp_path):
    import httpx
    import respx

    s = importlib.util.spec_from_file_location(
        "interview", ROOT / "experiments/chatbot-v1/tools/interview.py")
    tool = importlib.util.module_from_spec(s)
    s.loader.exec_module(tool)
    ctx = types.SimpleNamespace(agent="ds-a-0001", data_dir=tmp_path,
                                env={"FLOCK_URL": "http://ctl", "FLOCK_AGENT_TOKEN": "tok"})
    with respx.mock(base_url="http://ctl") as api:
        spawn = api.post("/agents/ds-a-0001/spawn").respond(
            201, json={"name": "ds-a-0001-interviewer-strong-0003"})
        api.get("/agents/ds-a-0001-interviewer-strong-0003").mock(side_effect=[
            httpx.Response(200, json={"last_heartbeat": None}),
            httpx.Response(200, json={"last_heartbeat": "2026-10-22T10:00:00Z"})])
        prompt = api.post("/agents/ds-a-0001-interviewer-strong-0003/prompt").respond(json={})
        r = tool.spawn_interviewer("strong", "teams/a/agents/ds-a-0001/bot.py", "4711", ctx)
        assert r == {"spawned": "ds-a-0001-interviewer-strong-0003", "level": "strong",
                     "path": "teams/a/agents/ds-a-0001/bot.py", "left_this_run": 7}
        assert spawn.calls[0].request.headers["X-Flock-Token"] == "tok"
        assert json.loads(spawn.calls[0].request.content) == {"type": "interviewer-strong"}
        text = json.loads(prompt.calls[0].request.content)["text"]  # once the child is up
        assert "teams/a/agents/ds-a-0001/bot.py for ds-a-0001 (team A)" in text
        assert "thread 4711" in text
        spawn.respond(409, json={"reason": "max_children", "detail": "2 live"})
        with pytest.raises(Exception, match="spawn refused \\(409\\).*max_children"):
            tool.spawn_interviewer("mid", "x.py", "1", ctx)
    (tmp_path / "interviewers-ds-a-0001.json").write_text("8")
    with pytest.raises(Exception, match="you spawned 8 interviewers"):
        tool.spawn_interviewer("mid", "x.py", "1", ctx)
