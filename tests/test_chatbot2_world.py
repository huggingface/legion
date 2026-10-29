"""worlds/chatbot2 in process: bots as packages (run, copy, grep, promote, best/ mirroring), the
2 MB write cap and the read window, swarm-written probes (add_probe with its verification and
refusals) and the strict promotion gate, the compare round after the merge, and interviews by
booking (separate quotas, the frozen copy, charged on the verdict, released when cut off)."""

import asyncio
import importlib.util
import json
import sys
import time
import types
import typing
from pathlib import Path

import pytest

from flock.world import Agent, ToolError

from .test_world import ROOT

A, A2, B = Agent("ds-a-0000", "ds-a", turn="t-1"), Agent("glm-a-0001"), Agent("ds-b-0001")
HOME_A, HOME_B = "teams/a/agents/ds-a-0000", "teams/b/agents/ds-b-0001"
MAIN = ("import sys\nfrom . import words\n\nfor line in sys.stdin:\n"
        "    print(words.reply(line.strip()), flush=True)\n")
WORDS = ("SAID = []\n\n\ndef reply(line):\n    SAID.append(line)\n    return ('Hi Priya! Japan"
         " sounds great, the food, the guitar, the dog Biscuit. I am Bo, nice to meet you, how"
         " about pasta? I am good, bye: ' + {tail})\n")
FORGETS, REMEMBERS = WORDS.format(tail="line"), WORDS.format(tail="' | '.join(SAID)")
TEACHER = ["I'm a teacher", "what's my job again?"]


def _load():
    s = importlib.util.spec_from_file_location("chatbot2", ROOT / "worlds/chatbot2/chatbot2.py")
    mod = importlib.util.module_from_spec(s)
    s.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load()


def _pkg(w, agent, home, words=FORGETS, name="bot"):
    w.write_file(agent, f"{home}/{name}/__main__.py", MAIN)
    w.write_file(agent, f"{home}/{name}/words.py", words)
    return f"{home}/{name}"


@pytest.fixture
def world(mod, tmp_path, monkeypatch):
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    monkeypatch.delenv("CHATBOT_CONFIG", raising=False)
    w = mod.Chatbot2World(str(tmp_path / "ws"))
    _pkg(w, A, HOME_A)
    _pkg(w, B, HOME_B)
    return w


def run(coro):
    return asyncio.run(coro)


def _log(w) -> list[dict]:
    return [json.loads(x) for x in (w.ws / "log.jsonl").read_text().splitlines()]


def _merged(w, tmp_path, monkeypatch, compare: int = 0) -> None:
    cfg = tmp_path / "world.yaml"
    cfg.write_text(f"merge: {{compare_minutes: {compare}}}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    w.set_phase("merged")


def test_manifest_tools_hooks_and_version(world):
    m = world.manifest()
    assert (m["name"], m["version"]) == ("chatbot", "0.2.0-teams")
    assert m["hooks"] == ["agent_done", "turn"]
    assert [t["name"] for t in m["tools"]] == [
        "read_file", "write_file", "copy_file", "edit_file", "insert_after", "list_files",
        "run_bot", "promote_best", "best_status", "qualify", "chat", "submit_verdict", "verdicts",
        "grep_file", "add_probe", "probes", "book_interview"]
    tools = {t["name"]: t["parameters"] for t in m["tools"]}
    assert list(tools["submit_verdict"]["properties"]) == [
        "booking", "evidence_script", "evidence_agent", "verdict", "confidence"]
    assert "package" in m["description"].lower() and "Phase 1 (teams)" in m["description"]


def test_a_package_bot_runs_like_python_m_bot(world):
    r = run(world.run_bot(f"{HOME_A}/bot/", ["hi", "I play the guitar"], agent=A))
    assert r["status"] == "ok" and r["turns"][1]["bot"].endswith("bye: I play the guitar"), r
    world.write_file(A, f"{HOME_A}/abs/__main__.py",
                     "from bot import words\nprint(words.reply('x'))\n")
    world.write_file(A, f"{HOME_A}/abs/words.py", FORGETS)
    assert run(world.run_bot(f"{HOME_A}/abs", ["x"], agent=A))["turns"][0]["bot"].endswith(": x")
    world.write_file(A, f"{HOME_A}/one.py",
                     "import sys\nfor l in sys.stdin: print('ok', flush=True)\n")
    assert run(world.run_bot(f"{HOME_A}/one.py", ["a"], agent=A))["turns"][0]["bot"] == "ok"
    world.write_file(A, f"{HOME_A}/nomain/x.py", "x = 1\n")
    for bad in (f"{HOME_A}/nomain", f"{HOME_A}/bot/words", f"{HOME_B}/bot"):
        with pytest.raises(ToolError, match="not a bot|belongs to team B|no file"):
            run(world.run_bot(bad, ["a"], agent=A))
    assert not list((world.ws / "teams").rglob("__pycache__"))  # runs in a temp copy


def test_copy_file_copies_a_package_and_replaces_stale_files(world):
    r = world.copy_file(A2, f"{HOME_A}/bot", "teams/a/agents/glm-a-0001/bot")
    assert (r["files"], r["removed"]) == (2, [])
    world.write_file(A2, "teams/a/agents/glm-a-0001/bot/old.py", "x = 1\n")
    r = world.copy_file(A2, f"{HOME_A}/bot", "teams/a/agents/glm-a-0001/bot")
    assert r["removed"] == ["teams/a/agents/glm-a-0001/bot/old.py"]
    for dst in ("teams/a/agents/glm-a-0001", "teams/a/agents/glm-a-0001/bot/words.py"):
        with pytest.raises(ToolError, match="may write only under|new subdirectory"):
            world.copy_file(A2, f"{HOME_A}/bot", dst)
    with pytest.raises(ToolError, match="belongs to team B"):
        world.copy_file(A2, f"{HOME_B}/bot", "teams/a/agents/glm-a-0001/theirs")
    with pytest.raises(ToolError, match="may write only under"):
        world.copy_file(A2, f"{HOME_A}/bot", f"{HOME_A}/mine")


def test_grep_file_and_the_read_window(world):
    hits = world.grep_file(f"{HOME_A}/bot", "def reply|from \\.", agent=A)
    assert hits == {"hits": [f"{HOME_A}/bot/__main__.py:2:from . import words",
                             f"{HOME_A}/bot/words.py:4:def reply(line):"], "truncated": False}
    assert world.grep_file("teams", "def reply", agent=A)["hits"] == [
        f"{HOME_A}/bot/words.py:4:def reply(line):"]  # phase 1: not team B's files
    assert world.grep_file(f"{HOME_A}/bot", "(", 1, agent=A) == {  # not a regex: plain text
        "hits": [f"{HOME_A}/bot/__main__.py:5:print(words.reply(line.strip()), flush=True)"],
        "truncated": True}
    big = "".join(f"x{i} = {i}  # {'pad' * 20}\n" for i in range(1500))  # ~100 KB
    world.write_file(A, f"{HOME_A}/big.py", big)
    with pytest.raises(ToolError, match="over the 60 KB read window: give start and end") as e:
        world.read_file(f"{HOME_A}/big.py", agent=A)
    assert "1500 lines" in e.value.message
    assert world.read_file(f"{HOME_A}/big.py", 10, 11, agent=A).startswith(
        f"# {HOME_A}/big.py: lines 10-11 of 1500\nx9 = 9")
    with pytest.raises(ToolError, match="is a directory"):
        world.read_file(f"{HOME_A}/bot", agent=A)
    world.write_file(A, f"{HOME_A}/notes.md", "y" * 1_900_000)  # the 2 MB cap
    with pytest.raises(ToolError, match="the limit is 2000000"):
        world.write_file(A, f"{HOME_A}/notes.md", "y" * 2_000_001)


def test_passes_each_check(mod):
    def r(*bots, status="ok"):
        return {"status": status, "turns": [{"user": "u", "bot": b} for b in bots]}

    assert mod.passes("mentions:Biscuit", r("x", "Your dog Biscuit!"))
    assert not mod.passes("mentions:cat", r("x", "category"))  # whole words
    assert mod.passes("mentions_fact:a teacher", r("x", "You're a Teacher."))
    assert not mod.passes("not_phrase:tell me more", r("Tell me  more!", "ok"))
    assert not mod.passes("not_repeat", r("Same.", "same.", "other"))
    assert mod.passes("not_repeat", r("", "", "a"))
    assert not mod.passes("not_empty", r("a", None))
    assert mod.passes("answers_question", r("x", "I love jazz. You?"))
    assert not mod.passes("answers_question", r("x", "Why do you ask?"))
    assert not mod.passes("not_empty", r("a", "b", status="timeout after 5 s"))


def test_add_probe_verifies_against_the_bot_and_refuses_otherwise(world):
    pa = f"{HOME_A}/bot"
    with pytest.raises(ToolError, match="no best yet: give `path`"):
        run(world.add_probe(A, TEACHER, "mentions_fact:teacher", "it forgot"))
    for lines, check, ev, msg in (
            (TEACHER, "mentions_fact:teacher", " ", "evidence is required"),
            (TEACHER, "sounds_human", "x", "check must be one of"),
            (TEACHER, "mentions_fact:nurse", "x", "needs an earlier line that states 'nurse'"),
            (["hello"], "answers_question", "x", "needs a question"),
            (["l"] * 13, "not_empty", "x", "give 1 to 12 lines")):
        with pytest.raises(ToolError, match=msg):
            run(world.add_probe(A, lines, check, ev, pa))
    with pytest.raises(ToolError, match="already passes this probe") as e:
        run(world.add_probe(A, ["hi"], "mentions:priya", "x", pa))
    assert e.value.code == "passes"
    with pytest.raises(ToolError, match="belongs to team B"):
        run(world.add_probe(A, TEACHER, "mentions_fact:teacher", "x", f"{HOME_B}/bot"))
    r = run(world.add_probe(A, TEACHER, "mentions_fact:teacher", "bye: what's my job again?", pa))
    assert r["added"] == "p1" and r["post"].startswith("PROBE p1 (ds-a-0000) mentions_fact:teacher")
    rec = json.loads((world.ws / "probes/live.jsonl").read_text())
    assert rec["evidence_matches"] and rec["path"] == pa and rec["lines"] == TEACHER
    for i in range(3):  # 4 per agent per hour (the default)
        run(world.add_probe(A, ["x", f"I'm a cook {i}", "what do I do?"], f"mentions_fact:cook {i}",
                            "no", pa))
    with pytest.raises(ToolError, match="you added 4 probes in the last hour"):
        run(world.add_probe(A, TEACHER, "mentions_fact:teacher", "x", pa))
    assert run(world.probes())["total"] == 4  # visible to all


class FakeApi:
    calls: typing.ClassVar[list] = []

    def batch_bucket_files(self, bucket_id, *, add=None, delete=None, **_):
        FakeApi.calls.append((bucket_id, [r for _, r in add], delete or []))


def test_promotion_is_strict_on_the_live_set_and_mirrors_best(world, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=FakeApi))
    monkeypatch.setenv("CHATBOT_BUCKET", "hf://buckets/me/chatbots/run-3/")
    FakeApi.calls = []
    world.write_file(A, f"{HOME_A}/single.py", "import sys\nfor n, l in enumerate(sys.stdin): "
                     "print('Hi Priya! Japan food guitar Biscuit pasta, I am Bo, nice, good, bye',"
                     " n, flush=True)\n")
    with pytest.raises(ToolError, match="promotion opens after the merge"):
        run(world.promote_best(A, f"{HOME_A}/single.py", "early"))
    _merged(world, tmp_path, monkeypatch)
    assert run(world.promote_best(A, f"{HOME_A}/single.py", "first"))["promoted"]
    assert (world.ws / "best/bot.py").is_file()
    r = run(world.promote_best(B, f"{HOME_B}/bot", "a package"))  # seed: higher (echo varies)
    assert r["promoted"], r
    best = world.ws / "best"
    assert not (best / "bot.py").exists() and (best / "bot/words.py").read_text() == FORGETS
    assert (best / "v1.py").is_file() and (best / "v2/__main__.py").is_file()
    cur = json.loads((best / "current.json").read_text())
    assert cur == {"n": 2, "kind": "package", "entry": "best/bot",
                   "files": ["best/bot/__main__.py", "best/bot/words.py"]}
    _, added, deleted = FakeApi.calls[-1]
    assert deleted == ["run-3/best/bot.py"]
    assert {"run-3/best/bot/words.py", "run-3/best/v2/words.py",
            "run-3/best/current.json"} <= set(added)
    # a tie does not promote; a probe best fails, fixed in a copy, does
    world.copy_file(A, "best/bot", f"{HOME_A}/fix")
    r = run(world.promote_best(A, f"{HOME_A}/fix", "same bot"))
    assert not r["promoted"] and "a tie does not promote" in r["why"], r
    run(world.add_probe(B, TEACHER, "mentions_fact:teacher", "bye: what's my job again?"))
    st = run(world.best_status())
    assert st["path"] == "best/bot" and st["score"]["live_fails"] == ["p1"]
    assert st["size"]["files"] == 2 and st["size"]["bytes"] > 0
    world.edit_file(A, f"{HOME_A}/fix/words.py", "+ line)", "+ ' | '.join(SAID))")
    r = run(world.promote_best(A, f"{HOME_A}/fix", "remembers the job"))
    assert r["promoted"] and r["best"]["live"] == "1/1" and r["live_fails"] == [], r
    assert run(world.probes())["live"][0]["best"] == "pass"
    # the seed probes are the floor: more swarm probes cannot buy a seed regression
    world.write_file(A, f"{HOME_A}/mute/__main__.py", "import sys\nfor n, l in enumerate("
                     "sys.stdin): print('teacher', n, flush=True)\n")
    r = run(world.promote_best(A, f"{HOME_A}/mute", "x"))
    assert not r["promoted"] and "(the floor)" in r["why"]
    why = [x["why"] for x in _log(world) if x["action"] == "promote_refused"]
    assert len(why) == 2 and "the current best" in why[0] and "on the seed probes" in why[1]


def test_compare_round_locks_promotion_until_the_knob_says_so(world, tmp_path, monkeypatch):
    _merged(world, tmp_path, monkeypatch, compare=15)
    with pytest.raises(ToolError, match="compare round, promotion opens in 15 min") as e:
        run(world.promote_best(A, f"{HOME_A}/bot", "x"))
    assert e.value.code == "compare_round"
    assert "promotion opens in 15 min" in run(world.best_status())["compare_round"]
    cfg = tmp_path / "world.yaml"  # the operator skips it: a live update of world.yaml
    cfg.write_text("merge: {compare_minutes: 0}\n")
    import os
    os.utime(cfg, (time.time() + 5, time.time() + 5))
    assert run(world.promote_best(A, f"{HOME_A}/bot", "x"))["promoted"]
    assert "compare_round" not in run(world.best_status())


def _iv(name):
    return Agent(name)


def test_interviews_by_booking_with_separate_quotas(world, mod, monkeypatch):
    pa = f"{HOME_A}/bot"
    s1 = run(world.book_interview(A, "strong", pa))
    assert s1["booking"] == "ds-a-0000-iv1-strong" and s1["quota"]["strong"]["left"] == 1
    assert s1["snapshot"] == "teams/a/interviews/ds-a-0000-iv1-strong/bot"
    s2 = run(world.book_interview(A2, "strong", pa))
    with pytest.raises(ToolError, match="no strong interview left this hour .*do not spawn") as e:
        run(world.book_interview(A, "strong", pa))
    assert e.value.code == "quota"
    m1 = run(world.book_interview(A, "mid", pa))  # a separate quota
    assert m1["quota"]["mid"]["left"] == 2 and m1["quota"]["strong"]["left"] == 0
    b1 = run(world.book_interview(B, "strong", f"{HOME_B}/bot"))
    assert b1["booking"] == "ds-b-0001-iv1-strong"
    # the frozen copy: an edit after the booking does not reach the interview
    world.edit_file(A, f"{pa}/words.py", "I am good, bye: ", "CHANGED: ")
    iv = _iv("ds-a-0000-interviewer-strong-0007")
    for i in range(6):
        r = run(world.chat(iv, "ds-a-0000-iv1-strong", f"hello {i}"))
    assert r["turn"] == 6 and r["reply"].endswith("bye: hello 5")
    with pytest.raises(ToolError, match="not open for you"):  # not its parent's booking
        run(world.chat(iv, "glm-a-0001-iv1-strong", "hi"))
    with pytest.raises(ToolError, match="not open for you"):  # the level must match
        run(world.chat(_iv("ds-a-0000-interviewer-mid-0008"), "ds-a-0000-iv1-strong", "hi"))
    with pytest.raises(ToolError, match="only interviewer agents"):
        run(world.chat(A, "ds-a-0000-iv1-strong", "hi"))
    v = world.submit_verdict(iv, "ds-a-0000-iv1-strong", ["repeats 'bye:'"], ["warm"], "script",
                             0.8)
    assert (v["verdict"], v["counts"], v["snapshot"]) == ("script", True, s1["snapshot"])
    with pytest.raises(ToolError, match=r"not open for you \(charged\)"):
        run(world.chat(iv, "ds-a-0000-iv1-strong", "again"))
    # cut off: the interviewer's turn ends without a verdict -> the unit comes back
    iv2 = _iv("glm-a-0001-interviewer-strong-0009")
    run(world.chat(iv2, s2["booking"], "hi"))
    world.on_turn(iv2, {"status": "error"})
    q = world.verdicts(A)["quota"]["strong"]
    assert (q["left"], q["pending"]) == (1, 0)
    assert world._bookings()["glm-a-0001-iv1-strong"]["why"] == "cut off"
    # a booking nobody uses is released after interviews.start_minutes
    now = time.time()
    monkeypatch.setattr(mod.time, "time", lambda: now + 301)
    assert world.verdicts(A)["quota"]["mid"]["left"] == 3
    vs = world.verdicts(A, "teams/a")["verdicts"]
    assert [x["booking"] for x in vs] == ["ds-a-0000-iv1-strong"] and vs[0]["evidence_script"]
    assert world.verdicts(B)["verdicts"] == []  # phase 1: team A's verdicts are not B's
    assert world.verdicts(B)["quota"]["strong"]["left"] == 2  # its unused booking expired


def test_booking_refuses_an_unqualified_bot_without_reserving(world):
    world.write_file(A, f"{HOME_A}/same/__main__.py", "import sys\nfor l in sys.stdin: print("
                     "'same', flush=True)\n")
    with pytest.raises(ToolError, match="does not qualify.*Nothing was reserved"):
        run(world.book_interview(A, "strong", f"{HOME_A}/same"))
    assert world._bookings() == {} and not (world.ws / "teams/a/interviews").exists() or not list(
        (world.ws / "teams/a/interviews").iterdir())
    q = run(world.qualify(A, f"{HOME_A}/bot"))
    assert q["qualified"] and q["quota"]["strong"]["left"] == 2 and q["score"]["live_of"] == 0


def test_one_qualify_bar_for_qualify_booking_and_promotion(world, tmp_path, monkeypatch):
    # passes the probes (it beats having no best) but gives "ok" and "lol" the same reply
    rep = f"{HOME_A}/rep"
    _pkg(world, A, HOME_A, WORDS.format(tail="('fine' if line in ('ok', 'lol') else line)"),
         "rep")
    _merged(world, tmp_path, monkeypatch)
    q = run(world.qualify(A, rep))
    assert not q["qualified"] and q["score"]["seed"] > 0
    line = "robust.txt line 4 'lol': repeated reply"
    assert q["fails"][0].startswith(line), q["fails"]
    r = run(world.promote_best(A, rep, "passes the probes"))
    assert not r["promoted"] and "fails the qualify bar" in r["why"] and line in r["why"], r
    assert not (world.ws / "best").exists() or world._best_path() is None
    with pytest.raises(ToolError, match="does not qualify") as e:
        run(world.book_interview(A, "strong", rep))
    assert str(e.value).split("does not qualify: ")[1].removesuffix(" Nothing was reserved.") \
        .removesuffix(".") == r["why"]  # the same reason, word for word
    assert world._bookings() == {}
    assert [x["why"] for x in _log(world) if x["action"] == "promote_refused"] == [r["why"]]
    assert run(world.promote_best(A, f"{HOME_A}/bot", "qualifies"))["promoted"]


def test_an_unverified_probe_is_kept_but_does_not_count(world, tmp_path, monkeypatch):
    _merged(world, tmp_path, monkeypatch)
    assert run(world.promote_best(A, f"{HOME_A}/bot", "first"))["promoted"]
    r = run(world.add_probe(B, TEACHER, "mentions_fact:teacher", "I never said that"))
    assert r["verified"] is False and "does NOT count" in r["note"]
    assert r["post"].startswith("PROBE p1 (unverified)")
    rec = json.loads((world.ws / "probes/live.jsonl").read_text())
    assert (rec["verified"], rec["evidence_matches"]) == (False, False)
    p = run(world.probes())
    assert p["best_fails"] == [] and p["live"][0] | {"lines": 0} == {
        "id": "p1", "author": "ds-b-0001", "check": "mentions_fact:teacher", "lines": 0,
        "verified": False, "best": "-"}
    st = run(world.best_status())["score"]
    assert (st["live"], st["live_of"]) == (0, 0)
    # a fix of the unverified failure does not beat best: it scores the same
    world.copy_file(A, "best/bot", f"{HOME_A}/fix")
    world.edit_file(A, f"{HOME_A}/fix/words.py", "+ line)", "+ ' | '.join(SAID))")
    r = run(world.promote_best(A, f"{HOME_A}/fix", "remembers"))
    assert not r["promoted"] and "a tie does not promote" in r["why"], r
    v = run(world.add_probe(B, TEACHER, "mentions_fact:teacher", "bye: what's my job again?"))
    assert v["verified"] and "note" not in v
    assert run(world.promote_best(A, f"{HOME_A}/fix", "remembers"))["promoted"]


def test_spawn_interviewer_tool_spawns_for_a_booking(tmp_path):
    import httpx
    import respx

    s = importlib.util.spec_from_file_location(
        "interview2", ROOT / "experiments/chatbot-v2/tools/interview.py")
    tool = importlib.util.module_from_spec(s)
    s.loader.exec_module(tool)
    ctx = types.SimpleNamespace(agent="ds-a-0001", data_dir=tmp_path,
                                env={"FLOCK_URL": "http://ctl", "FLOCK_AGENT_TOKEN": "tok"})
    with pytest.raises(Exception, match="not a booking id; call book_interview first"):
        tool.spawn_interviewer("strong", "1", ctx)
    with respx.mock(base_url="http://ctl") as api:
        spawn = api.post("/agents/ds-a-0001/spawn").respond(
            201, json={"name": "ds-a-0001-interviewer-strong-0003"})
        api.get("/agents/ds-a-0001-interviewer-strong-0003").mock(side_effect=[
            httpx.Response(200, json={"last_heartbeat": None}),
            httpx.Response(200, json={"last_heartbeat": "2026-10-23T10:00:00Z"})])
        prompt = api.post("/agents/ds-a-0001-interviewer-strong-0003/prompt").respond(json={})
        r = tool.spawn_interviewer("ds-a-0001-iv7-strong", "4711", ctx)
        assert r == {"spawned": "ds-a-0001-interviewer-strong-0003",
                     "booking": "ds-a-0001-iv7-strong", "left_this_run": 11}
        assert json.loads(spawn.calls[0].request.content) == {"type": "interviewer-strong"}
        text = json.loads(prompt.calls[0].request.content)["text"]
        assert text == ("Interview booking ds-a-0001-iv7-strong for ds-a-0001 (team A). When "
                        "you have filed your verdict, post it in thread 4711.")


def test_world_restart_keeps_bookings_and_probes(world, mod, tmp_path, monkeypatch):
    _merged(world, tmp_path, monkeypatch)
    run(world.promote_best(A, f"{HOME_A}/bot", "first"))
    run(world.add_probe(B, TEACHER, "mentions_fact:teacher", "bye: what's my job again?"))
    b = run(world.book_interview(A, "strong", "best/bot"))
    assert world._bookings()[b["booking"]]["best_n"] == 1
    again = mod.Chatbot2World(str(Path(world.ws)))
    assert again.phase == "merged" and run(again.probes())["best_fails"] == ["p1"]
    assert run(again.book_interview(A, "mid", "best/bot"))["booking"] == "ds-a-0000-iv2-mid"
