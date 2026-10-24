"""worlds/chatbot3 in process: one flat workspace (own-dir writes, everything readable), the
agent-facing tool descriptions, promotion open from the start behind the qualify bar and the
beat-best gate (ties refused), interviews by booking without a quota (frozen copy, charged on
the verdict, released when cut off), the bucket mirror of best/ and the bot-run pool."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import threading
import time

import pytest

from flock.world import Agent, ToolError

from .test_world import ROOT

A, B = Agent("agent-0000", "agent", turn="t-1"), Agent("agent-0001", "agent")
HOME_A, HOME_B = "agents/agent-0000", "agents/agent-0001"
MAIN = ("import sys\nfrom . import words\n\nfor line in sys.stdin:\n"
        "    print(words.reply(line.strip()), flush=True)\n")
WORDS = ("SAID = []\n\n\ndef reply(line):\n    SAID.append(line)\n    return ('Hi Priya! Japan"
         " sounds great, the food, the guitar, the dog Biscuit. I am Bo, nice to meet you, how"
         " about pasta? I am good, bye: ' + {tail})\n")
FORGETS, REMEMBERS = WORDS.format(tail="line"), WORDS.format(tail="' | '.join(SAID)")
TEACHER = ["I'm a teacher", "what's my job again?"]
EMPTY = "import sys\nfor line in sys.stdin:\n    print(' ', flush=True)\n"


def _load():
    s = importlib.util.spec_from_file_location("chatbot3", ROOT / "worlds/chatbot3/chatbot3.py")
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
    monkeypatch.delenv("CHATBOT_CONFIG", raising=False)
    monkeypatch.setenv("CHATBOT_BUCKET", str(tmp_path / "bucket"))
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    _pkg(w, A, HOME_A)
    _pkg(w, B, HOME_B)
    return w


def run(coro):
    return asyncio.run(coro)


def _log(w) -> list[dict]:
    return [json.loads(x) for x in (w.ws / "log.jsonl").read_text().splitlines()]


def test_manifest_flat_workspace_and_descriptions(world):
    m = world.manifest()
    assert (m["name"], m["version"]) == ("chatbot3", "0.3.0")
    assert m["hooks"] == ["agent_done", "turn"]
    tools = {t["name"]: t for t in m["tools"]}
    assert {"read_file", "grep_file", "write_file", "copy_file", "edit_file", "insert_after",
            "list_files", "run_bot", "add_probe", "probes", "promote_best", "best_status",
            "qualify", "book_interview", "chat", "submit_verdict", "verdicts"} <= set(tools)
    text = json.dumps(m)
    assert "teams/" not in text and "phase" not in text.lower() and "compare round" not in text
    assert "Open from the start" in tools["promote_best"]["description"]
    assert tools["book_interview"]["parameters"]["properties"]["level"]["enum"] == ["strong"]
    assert tools["book_interview"]["parameters"]["required"] == ["path"]
    # own dir only; everyone reads and runs everything
    with pytest.raises(ToolError, match="may write only under agents/agent-0000/"):
        world.write_file(A, f"{HOME_B}/x.py", "x = 1\n")
    assert "def reply" in world.read_file(f"{HOME_B}/bot/words.py", agent=A)
    assert run(world.run_bot(f"{HOME_B}/bot", ["hi"], agent=A))["status"] == "ok"
    hits = world.grep_file("agents", "def reply", agent=A)["hits"]
    assert sorted(h.split(":")[0] for h in hits) == [f"{HOME_A}/bot/words.py",
                                                     f"{HOME_B}/bot/words.py"]
    assert world.copy_file(A, f"{HOME_B}/bot", f"{HOME_A}/theirs")["files"] == 2
    assert world.list_files("agents/", agent=A)["total"] == 6


def test_promotion_from_the_start_bar_and_beat_best(world, tmp_path):
    r = run(world.promote_best(A, f"{HOME_A}/bot", "first"))  # no phase gate: open at once
    assert r["promoted"] and r["best"]["n"] == 1 and r["best"]["kind"] == "package"
    tie = run(world.promote_best(B, f"{HOME_B}/bot", "same"))
    assert not tie["promoted"] and "a tie does not promote" in tie["why"]
    world.write_file(B, f"{HOME_B}/empty.py", EMPTY)  # fails the bar: refused before scoring
    bad = run(world.promote_best(B, f"{HOME_B}/empty.py", "x"))
    assert not bad["promoted"] and "fails the qualify bar" in bad["why"]
    p = run(world.add_probe(B, TEACHER, "mentions_fact:teacher",
                             "I am good, bye: what's my job again?"))  # vs best (verified)
    assert p["added"] == "p1" and p["post"].startswith("PROBE p1 (agent-0001)")
    world.edit_file(B, f"{HOME_B}/bot/words.py", "+ line)", "+ ' | '.join(SAID))")
    win = run(world.promote_best(B, f"{HOME_B}/bot", "remembers"))
    assert win["promoted"] and win["best"]["n"] == 2 and win["best"]["live"] == "1/1"
    st = run(world.best_status())
    assert st["best"]["agent"] == "agent-0001" and "compare_round" not in st
    cur = json.loads((tmp_path / "bucket" / "best" / "current.json").read_text())
    assert cur["n"] == 2 and cur["entry"] == "best/bot"  # the mirror
    assert (tmp_path / "bucket" / "best" / "v1" / "words.py").is_file()
    q = run(world.qualify(A, "best/bot"))
    assert q["qualified"] and q["quota"]["strong"]["limit"] is None


def test_interviews_without_quota(world, mod, monkeypatch):
    pa = f"{HOME_A}/bot"
    books = [run(world.book_interview(A, pa)) for _ in range(5)]  # never refused for quota
    assert [b["booking"] for b in books] == [f"agent-0000-iv{i}-strong" for i in range(1, 6)]
    assert books[0]["snapshot"] == "interviews/agent-0000-iv1-strong/bot"
    world.write_file(A, f"{HOME_A}/empty.py", EMPTY)
    with pytest.raises(ToolError, match="does not qualify") as e:
        run(world.book_interview(A, f"{HOME_A}/empty.py"))
    assert e.value.code == "not_qualified"
    assert world._bookings()["agent-0000-iv6-strong"]["state"] == "released"
    world.edit_file(A, f"{pa}/words.py", "I am good, bye: ", "CHANGED: ")  # after the booking
    iv = Agent("agent-0000-interviewer-0003")
    for i in range(6):
        r = run(world.chat(iv, "agent-0000-iv1-strong", f"hello {i}"))
    assert r["turn"] == 6 and r["reply"].endswith("bye: hello 5")  # the frozen copy
    with pytest.raises(ToolError, match="not open for you"):  # not its parent's booking
        run(world.chat(Agent("agent-0001-interviewer-0004"), "agent-0000-iv2-strong", "hi"))
    with pytest.raises(ToolError, match="only interviewer agents"):
        run(world.chat(A, "agent-0000-iv1-strong", "hi"))
    v = world.submit_verdict(iv, "agent-0000-iv1-strong", ["repeats 'bye:'"], ["warm"],
                             "script", 0.8)
    assert (v["verdict"], v["counts"], v["level"]) == ("script", True, "strong")
    assert world._bookings()["agent-0000-iv1-strong"]["state"] == "charged"
    iv2 = Agent("agent-0000-interviewer-0005")  # cut off: released (for the records)
    run(world.chat(iv2, "agent-0000-iv2-strong", "hi"))
    world.on_turn(iv2, {"status": "error"})
    assert world._bookings()["agent-0000-iv2-strong"]["why"] == "cut off"
    vs = world.verdicts(B)
    assert [x["booking"] for x in vs["verdicts"]] == ["agent-0000-iv1-strong"]
    assert sorted(x["id"] for x in vs["open_bookings"]) == [
        f"agent-0000-iv{i}-strong" for i in (3, 4, 5)]
    now = time.time()  # unused bookings expire after interviews.start_minutes
    monkeypatch.setattr(mod.time, "time", lambda: now + 301)
    assert world.verdicts(B)["open_bookings"] == []
    assert {x["action"] for x in _log(world)} >= {"book_interview", "interview_start", "verdict",
                                                  "interview_released"}


def test_bot_runs_share_the_pool(mod, tmp_path, monkeypatch):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("runs: {parallel: 2}\nprobes: {max_total: 5}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    assert w.cfg("probes", "max_total") == 5 and w.cfg("probes", "max_lines") == 12
    live, peak, lock = [0], [0], threading.Lock()
    real = mod.ElizaWorld._run

    def counted(self, path, lines):
        with lock:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        time.sleep(0.05)
        try:
            return real(self, path, lines)
        finally:
            with lock:
                live[0] -= 1

    monkeypatch.setattr(mod.ElizaWorld, "_run", counted)
    path = _pkg(w, A, HOME_A)

    async def many():
        await asyncio.gather(*(w.run_bot(path, ["hi"]) for _ in range(6)))

    run(many())
    assert peak[0] == 2


def test_words_named_interviewer(world):
    """`naming: words`: the interviewer is known by its type (iv-teal-heron says nothing about
    its parent); the booking binds the first interviewer that chats on it."""
    pa = f"{HOME_A}/bot"
    bid = run(world.book_interview(A, pa))["booking"]
    iv, other = Agent("iv-teal-heron", "interviewer"), Agent("iv-amber-otter", "interviewer")
    assert run(world.chat(iv, bid, "hi"))["turn"] == 1
    with pytest.raises(ToolError, match="not open for you"):  # bound to the first interviewer
        run(world.chat(other, bid, "hi"))
    with pytest.raises(ToolError, match="only interviewer agents"):
        run(world.chat(Agent("teal-heron", "agent"), bid, "hi"))
    world.on_turn(iv, {"status": "error"})  # cut off: released (hook by type)
    assert world._bookings()[bid]["why"] == "cut off"


# ------------------------------------------------------------------ the promotion gate is atomic
PIZZA = ["I love pizza", "what food do I love?"]
TEACHER_ONLY = WORDS.format(tail="' | '.join(x for x in SAID if 'teacher' in x)")


async def _probe(w, agent, lines, check):
    """Add a probe best fails, with the evidence copied from its own reply (verified)."""
    r = await w.run_bot("best/bot", lines)
    return await w.add_probe(agent, lines, check, r["turns"][-1]["bot"])


def _agents(n):
    return [Agent(f"agent-{i:04d}", "agent") for i in range(n)]


def test_concurrent_equal_promotions_one_wins(world):
    """6 equal candidates against a lower best, at once: exactly 1 promotes, 5 are refused as
    ties against the new best (the critical section re-reads best)."""
    async def go():
        assert (await world.promote_best(A, f"{HOME_A}/bot", "base"))["promoted"]
        assert (await _probe(world, A, TEACHER, "mentions_fact:teacher"))["verified"]
        ags = _agents(6)
        paths = [_pkg(world, a, f"agents/{a.name}", REMEMBERS) for a in ags]
        return await asyncio.gather(*(world.promote_best(a, p, "x") for a, p in zip(ags, paths)))

    res = run(go())
    assert sum(r["promoted"] for r in res) == 1
    assert sum("a tie does not promote" in r.get("why", "") for r in res) == 5
    hist = world._history()
    assert len(hist) == 2 and hist[1]["best_before"] == hist[0]["best_after"]
    assert hist[1]["best_after"] != hist[0]["best_after"]


@pytest.mark.parametrize("order", ["better_first", "better_last"])
def test_concurrent_different_scores_the_higher_ends_best(world, order):
    """Two strictly better candidates with different scores, at once: promotions are
    serialized in arrival order and each is compared with the best at its turn, so the higher
    one ends as best in either order (the higher first: the lower is refused; the lower
    first: the higher beats it and promotes after it)."""
    async def go():
        assert (await world.promote_best(A, f"{HOME_A}/bot", "base"))["promoted"]
        assert (await _probe(world, A, TEACHER, "mentions_fact:teacher"))["verified"]
        assert (await _probe(world, A, PIZZA, "mentions_fact:pizza"))["verified"]
        hi, lo = _agents(2)
        p_hi = _pkg(world, hi, f"agents/{hi.name}", REMEMBERS)  # passes both probes
        p_lo = _pkg(world, lo, f"agents/{lo.name}", TEACHER_ONLY)  # passes one
        pair = [(hi, p_hi), (lo, p_lo)] if order == "better_first" else [(lo, p_lo), (hi, p_hi)]
        return await asyncio.gather(*(world.promote_best(a, p, "x") for a, p in pair))

    run(go())
    hist = world._history()
    assert hist[-1]["agent"] == "agent-0000" and hist[-1]["live"] == "2/2"  # the higher
    assert [h["agent"] for h in hist[1:]] in (["agent-0000"], ["agent-0001", "agent-0000"])


def test_two_world_processes_share_the_gate(mod, tmp_path, monkeypatch):
    """A replaced world runs next to the old one for a moment: two instances on the same
    workspace still promote one at a time (the file lock), so equal candidates give one."""
    import threading as th

    monkeypatch.delenv("CHATBOT_CONFIG", raising=False)
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w1 = mod.Chatbot3World(str(tmp_path / "ws"))  # one instance per thread: each has its own
    worlds = [mod.Chatbot3World(str(tmp_path / "ws")) for _ in range(4)]  # in-process lock
    run(w1.promote_best(A, _pkg(w1, A, HOME_A), "base"))
    run(_probe(w1, A, TEACHER, "mentions_fact:teacher"))
    ags = _agents(4)
    paths = [_pkg(w1, a, f"agents/{a.name}", REMEMBERS) for a in ags]
    out: list[dict] = []

    def one(w, a, p):
        out.append(run(w.promote_best(a, p, "x")))

    ts = [th.Thread(target=one, args=(w, a, p)) for w, a, p in zip(worlds, ags, paths)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert sum(r["promoted"] for r in out) == 1 and len(w1._history()) == 2


def test_copy_of_best_survives_a_concurrent_promotion(world):
    """copy_file of best/bot while promotions replace it: the copy comes from the immutable
    snapshot best/v<n>/ named by current.json, so it never sees a half-replaced directory and
    always gets one whole version."""
    import json as js
    import shutil
    import threading as th

    run(world.promote_best(A, f"{HOME_A}/bot", "base"))
    b = world.ws / "best"
    stop, errors, n = th.Event(), [], [1]

    def promoter():  # what promote_best does to best/, as fast as possible
        while not stop.is_set():
            n[0] += 1
            src = b / f".src{n[0]}"
            src.mkdir()
            for i in range(30):
                (src / f"m{i}.py").write_text(f"V = {n[0]}\n")
            (src / "__main__.py").write_text(f"V = {n[0]}\n")
            shutil.copytree(src, b / f"v{n[0]}")
            shutil.rmtree(b / "bot")
            os.replace(src, b / "bot")
            (b / "current.json").write_text(js.dumps({"n": n[0], "kind": "package"}))

    t = th.Thread(target=promoter)
    t.start()
    while n[0] < 3:  # past the base bot: every version from here is 31 identical files
        time.sleep(0.01)
    try:
        for k in range(40):
            try:
                world.copy_file(B, "best/bot", f"{HOME_B}/c{k}")
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))
    finally:
        stop.set()
        t.join()
    assert not errors, errors[:3]
    for k in range(40):  # one whole version per copy
        got = {p.read_text() for p in (world.ws / HOME_B / f"c{k}").glob("*.py")}
        assert len(got) == 1, (k, got)


def test_gate_size_parallelism_and_busy_refusal(mod, tmp_path, monkeypatch):
    """probes.gate_size (the newest N verified probes), runs.per_promotion and
    promote.max_waiting, all re-read from $CHATBOT_CONFIG at each call."""
    cfg = tmp_path / "world.yaml"
    cfg.write_text("probes: {gate_size: 1}\nruns: {per_promotion: 3}\npromote: {max_waiting: 2}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    run(_probe(w, A, TEACHER, "mentions_fact:teacher"))
    run(_probe(w, A, PIZZA, "mentions_fact:pizza"))
    s = run(w.best_status())["score"]
    assert s["live_of"] == 1 and s["live_fails"] == ["p2"]  # only the newest probe gates
    ags = _agents(5)
    paths = [_pkg(w, a, f"agents/{a.name}", REMEMBERS) for a in ags]

    async def burst():
        return await asyncio.gather(*(w.promote_best(a, p, "x") for a, p in zip(ags, paths)))

    res = run(burst())
    busy = [r for r in res if not r["promoted"] and r.get("why", "").startswith("busy")]
    assert sum(r["promoted"] for r in res) == 1 and len(busy) == 3  # 2 in flight, 3 refused
    cfg.write_text("probes: {gate_size: 0}\n")  # live change: the whole set again
    assert run(w.best_status())["score"]["live_of"] == 2
