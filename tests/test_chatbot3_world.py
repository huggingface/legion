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


def _until(ok, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not ok():
        assert time.monotonic() < deadline
        time.sleep(0.02)


def run(coro):
    return asyncio.run(coro)


def _log(w) -> list[dict]:
    return [json.loads(x) for x in (w.ws / "log.jsonl").read_text().splitlines()]


def test_manifest_flat_workspace_and_descriptions(world):
    m = world.manifest()
    assert (m["name"], m["version"]) == ("chatbot3", "0.3.4")
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
    world.on_turn(iv2, {"status": "error"})  # released off the request path
    _until(lambda: world._bookings()["agent-0000-iv2-strong"].get("why") == "cut off")
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
    _until(lambda: world._bookings()[bid].get("why") == "cut off")


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


def _many_probes(w, n):
    for i in range(n):
        assert run(_probe(w, A, [f"my word is w{i}", "what is my word?"],
                          f"mentions_fact:w{i}"))["verified"]


def test_sampled_gate_is_deterministic_and_full_every_kth(mod, tmp_path, monkeypatch):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("probes: {per_agent_per_hour: 100, gate_sample: 4, gate_full_every: 3}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))  # promotion 1
    _many_probes(w, 10)
    one, label = w._gate("a" * 64)
    assert len(one) == 4 and label.startswith("sample 4/10")
    assert [x["id"] for x in w._gate("a" * 64)[0]] == [x["id"] for x in one]  # same bot, same
    assert any([x["id"] for x in w._gate(f"{k:064x}")[0]] != [x["id"] for x in one]
               for k in range(5))  # another bot, another sample
    r = run(w.promote_best(B, _pkg(w, B, HOME_B, REMEMBERS), "remembers"))  # promotion 2
    assert r["promoted"] and w._history()[-1]["gate"].startswith("sample 4/10")
    assert w._gate("a" * 64)[1] == "full 10"  # the 3rd promotion: the full set
    cfg.write_text("probes: {gate_sample: 0}\n")  # live: the full set always
    assert w._gate("b" * 64)[1] == "full 10"


def test_pool_bounds_parallel_runs_and_is_re_read(mod, tmp_path, monkeypatch):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("runs: {parallel: 3}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
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

    async def many(k):
        await asyncio.gather(*(w.run_bot(path, ["hi"]) for _ in range(k)))

    run(many(10))
    assert peak[0] == 3
    peak[0] = 0
    time.sleep(0.02)
    cfg.write_text("runs: {parallel: 6}\n")  # live: a bigger pool, no restart
    run(many(12))
    assert peak[0] == 6


def test_slow_promotion_answers_queued_with_a_ticket(mod, tmp_path, monkeypatch):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("promote: {wait_s: 0.05}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    real = mod.ElizaWorld._run

    def slow(self, path, lines):
        time.sleep(0.05)
        return real(self, path, lines)

    monkeypatch.setattr(mod.ElizaWorld, "_run", slow)
    path = _pkg(w, A, HOME_A)

    async def go():
        r = await w.promote_best(A, path, "base")
        assert r["queued"] and r["promoted"] is None and r["ticket"] == "agent-0000-p1"
        assert (await w.best_status(agent=A))["pending"] == 1
        while getattr(w, "_waiting", 0):
            await asyncio.sleep(0.05)
        return await w.best_status(agent=A)

    st = run(go())
    assert st["pending"] == 0 and st["your_promotions"][0]["result"]["promoted"] is True
    assert {x["action"] for x in _log(w)} >= {"promote_queued", "promote"}


def test_busy_refusal_when_max_waiting(mod, tmp_path, monkeypatch):
    cfg = tmp_path / "world.yaml"
    cfg.write_text("promote: {max_waiting: 2, wait_s: 0}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    run(_probe(w, A, TEACHER, "mentions_fact:teacher"))
    ags = _agents(5)
    paths = [_pkg(w, a, f"agents/{a.name}", REMEMBERS) for a in ags]

    async def burst():
        return await asyncio.gather(*(w.promote_best(a, p, "x") for a, p in zip(ags, paths)))

    res = run(burst())
    busy = [r for r in res if not r["promoted"] and r.get("why", "").startswith("busy")]
    assert sum(bool(r["promoted"]) for r in res) == 1 and len(busy) == 3  # 2 in flight


def test_hooks_and_reads_stay_fast_during_a_promotion_and_slow_tools(mod, tmp_path, monkeypatch):
    """Served over HTTP (flock.world.server): while a long promotion is scored and slow sync
    tools (edit_file with a slow compile check) run, 50 hooks answer with p95 < 100 ms and a
    read_file within 200 ms (the world is `concurrent`: no global lock; hooks do their
    bookkeeping off the request path)."""
    import socket
    import statistics

    import httpx
    import uvicorn

    from flock.world.server import create_app

    monkeypatch.delenv("CHATBOT_CONFIG", raising=False)
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    path = _pkg(w, A, HOME_A)
    real_run, real_save = mod.ElizaWorld._run, mod.ElizaWorld._save

    def slow_run(self, p, lines):
        time.sleep(0.1)
        return real_run(self, p, lines)

    def slow_save(self, *a, **k):
        time.sleep(2.0)  # a slow compile check
        return real_save(self, *a, **k)

    monkeypatch.setattr(mod.ElizaWorld, "_run", slow_run)
    monkeypatch.setattr(mod.ElizaWorld, "_save", slow_save)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(w, "tok"), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    base, head = f"http://127.0.0.1:{port}", {"X-World-Token": "tok"}
    while not server.started:
        time.sleep(0.05)

    def tool(name, agent, args, timeout=60):
        return httpx.post(f"{base}/tools/{name}", json={"args": args}, timeout=timeout,
                          headers={**head, "X-Flock-Agent": agent.name,
                                   "X-Flock-Type": agent.type})

    try:
        busy = [threading.Thread(target=tool, args=("promote_best", A, {
            "path": path, "reason": "x"}))] + [threading.Thread(target=tool, args=(
                "edit_file", A, {"path": f"{HOME_A}/bot/words.py", "old": "SAID = []",
                                 "new": "SAID = []"})) for _ in range(10)]
        for b in busy:
            b.start()
        time.sleep(0.3)  # the promotion is scoring, the edits hold their compile checks
        lat = []
        for i in range(50):
            t0 = time.monotonic()
            r = httpx.post(f"{base}/hooks", headers=head, timeout=10, json={
                "event": "turn", "agent": f"iv-a{i}" if i % 2 else "agent-0001",
                "type": "interviewer" if i % 2 else "agent", "swarm": "s", "turn": "t-1",
                "data": {"status": "completed"}})
            lat.append(time.monotonic() - t0)
            assert r.json()["ok"]
        t0 = time.monotonic()
        r = tool("read_file", A, {"path": f"{HOME_A}/bot/words.py"}, timeout=10)
        read_s = time.monotonic() - t0
        assert r.json()["ok"] and any(b.is_alive() for b in busy)  # still busy meanwhile
        p95 = statistics.quantiles(lat, n=20)[-1]
        assert p95 < 0.1, (p95, max(lat))
        assert read_s < 0.2, read_s
        for b in busy:
            b.join()
    finally:
        server.should_exit = True
        t.join(5)


def _retire_world(mod, tmp_path, monkeypatch, knobs: str):
    cfg = tmp_path / "world.yaml"
    cfg.write_text(f"probes: {{per_agent_per_hour: 100, {knobs}}}\n")
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    w = mod.Chatbot3World(str(tmp_path / "ws"))
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    return w, cfg


def test_retire_off_by_default_but_streaks_are_kept(world):
    """retire_after 0 (the default): the gate and the probe limit are chatbot2's; the gate's
    results for best are still counted (one per version), ready if the knob is turned on."""
    run(world.promote_best(A, f"{HOME_A}/bot", "base"))
    run(_probe(world, A, TEACHER, "mentions_fact:teacher"))
    _pkg(world, B, HOME_B, REMEMBERS)
    assert run(world.promote_best(B, f"{HOME_B}/bot", "remembers"))["promoted"]
    sha = world._sha(world._best_path())[:12]
    assert world._streaks() == {"p1": {"streak": 1, "sha": sha}}
    assert not run(world.promote_best(A, f"{HOME_B}/bot", "tie"))["promoted"]  # re-scores best
    assert world._streaks()["p1"]["streak"] == 1  # the same version counts once
    for k in range(5):
        world._gate_scored(f"{k:012x}", {"p1": True})
    assert world._retired() == set() and world._gate("a" * 64)[1] == "full 1"
    st = run(world.best_status())
    assert st["probes"] == {"live": 1, "retired": 0, "max_total": 400, "retire_after": 0}


def test_retire_after_n_passes_reset_on_fail(mod, tmp_path, monkeypatch):
    w, _ = _retire_world(mod, tmp_path, monkeypatch, "retire_after: 3")
    _many_probes(w, 2)
    w._gate_scored("v1", {"p1": True, "p2": True})
    w._gate_scored("v1", {"p1": True, "p2": True})  # re-scored: no new version
    w._gate_scored("v2", {"p1": True, "p2": False})  # a fail resets
    w._gate_scored("v3", {"p1": True, "p2": True})
    assert w._retired() == {"p1"} and w._streaks()["p2"]["streak"] == 1
    assert [x["probe"] for x in _log(w) if x["action"] == "probe_retired"] == ["p1"]
    w._gate_scored("v4", {"p1": False})  # failed in a full gate: it counts again
    assert w._retired() == set()


def test_retired_left_out_of_sample_and_limit_kept_in_full_gate(mod, tmp_path, monkeypatch):
    w, cfg = _retire_world(mod, tmp_path, monkeypatch,
                           "retire_after: 2, gate_sample: 4, gate_full_every: 0, max_total: 10")
    _many_probes(w, 10)
    with pytest.raises(ToolError, match=r"the live set is full: 10 live probes, the limit is "
                                        r"10 \(0 retired probes do not count"):
        run(_probe(w, A, ["my word is x", "what is my word?"], "mentions_fact:x"))
    for v in ("v1", "v2"):
        w._gate_scored(v, {"p1": True, "p2": True, "p3": True})
    assert w._retired() == {"p1", "p2", "p3"}
    for k in range(30):  # every sample: 4 of the 7 not retired
        got, label = w._gate(f"{k:064x}")
        assert len(got) == 4 and not {x["id"] for x in got} & {"p1", "p2", "p3"}
        assert label.startswith("sample 4/7 ") and label.endswith("(3 retired)")
    st = run(w.best_status())
    assert st["probes"] == {"live": 7, "retired": 3, "max_total": 10, "retire_after": 2}
    pr = run(w.probes())
    assert pr["counts"]["retired"] == 3 and [x["id"] for x in pr["live"] if x.get("retired")] \
        == ["p1", "p2", "p3"]
    assert run(_probe(w, A, ["my word is x", "what is my word?"], "mentions_fact:x"))[
        "added"] == "p11"  # 8 live of 10: room again
    cfg.write_text("probes: {retire_after: 2, gate_sample: 0, max_total: 10}\n")
    full, label = w._gate("a" * 64)  # the final round: everything, the retired included
    assert label == "full 11" and {"p1", "p2", "p3"} <= {x["id"] for x in full}
    cfg.write_text("probes: {retire_after: 2, gate_sample: 60, max_total: 10}\n")
    os.utime(cfg, (time.time() + 5, time.time() + 5))  # a new mtime: re-read
    assert w._gate("a" * 64)[1] == "active 8 (3 retired)"  # the sample would cover the rest


def test_retired_survive_a_restart(mod, tmp_path, monkeypatch):
    w, _ = _retire_world(mod, tmp_path, monkeypatch, "retire_after: 2, gate_sample: 2")
    _many_probes(w, 4)
    for v in ("v1", "v2"):
        w._gate_scored(v, {"p1": True, "p2": True})
    w2 = mod.Chatbot3World(str(tmp_path / "ws"))  # the requeued world on the restored ws/
    assert w2._retired() == {"p1", "p2"} and w2._streaks() == w._streaks()
    assert {x["id"] for x in w2._gate("a" * 64)[0]} == {"p3", "p4"}
    assert not list((w.ws / "probes").glob(".streaks*"))  # written atomically, no leftovers


# ------------------------------------------------------------------ best's score without bot runs
def _counting(mod, monkeypatch, delay=0.0):
    """Patch the bot run: count every run (by bot path and lines) and the peak at once."""
    real, lock = mod.ElizaWorld._run, threading.Lock()
    st = {"runs": [], "live": 0, "peak": 0, "best_peak": 0, "best_live": 0}

    def counted(self, path, lines):
        best = "best" in path.parts
        with lock:
            st["runs"].append((str(path), tuple(lines)))
            st["live"] += 1
            st["best_live"] += best
            st["peak"] = max(st["peak"], st["live"])
            st["best_peak"] = max(st["best_peak"], st["best_live"])
        try:
            time.sleep(delay)
            return real(self, path, lines)
        finally:
            with lock:
                st["live"] -= 1
                st["best_live"] -= best

    monkeypatch.setattr(mod.ElizaWorld, "_run", counted)
    return st


def _cfg_world(mod, tmp_path, monkeypatch, text):
    cfg = tmp_path / "world.yaml"
    cfg.write_text(text)
    monkeypatch.setenv("CHATBOT_CONFIG", str(cfg))
    monkeypatch.delenv("CHATBOT_BUCKET", raising=False)
    return mod.Chatbot3World(str(tmp_path / "ws"))


async def _until_async(ok, timeout=20.0):
    deadline = time.monotonic() + timeout
    while not ok():
        assert time.monotonic() < deadline
        await asyncio.sleep(0.02)


def test_best_status_runs_no_bot_gate_then_background_full(mod, tmp_path, monkeypatch):
    w = _cfg_world(mod, tmp_path, monkeypatch,
                   "probes: {per_agent_per_hour: 100, gate_sample: 3, gate_full_every: 0}\n")
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    _many_probes(w, 8)
    assert run(w.promote_best(B, _pkg(w, B, HOME_B, REMEMBERS), "remembers"))["promoted"]
    sha = w._sha(w._best_path())[:12]
    st = _counting(mod, monkeypatch)

    async def go():
        first = await w.best_status(agent=A)
        ran = len(st["runs"])  # the call itself: no bot run (the background starts after it)
        await _until_async(lambda: (w._fullrec or {}).get("sha") == sha)
        return first, ran, await w.best_status(agent=A)

    first, ran, second = run(go())
    assert ran == 0
    g = first["score"]["gate"]
    assert g["n"] == 2 and g["gate"].startswith("sample 3/8") and g["live"] == "3/3"
    assert len(g["probes"]) == 3 and g["fails"] == []  # the gate's probes, from memory
    assert first["score"]["full"] is None and "full_pending" in first["score"]
    full = second["score"]["full"]
    assert (full["n"], full["sha"], full["current"]) == (2, sha, True) and full["age_s"] >= 0
    assert (full["live"], full["live_of"]) == (8, 8) and "full_pending" not in second["score"]
    # the background ran the 5 probes the gate had not scored, and the 4 seed files, once
    assert len(st["runs"]) == 5 + 4 and len(set(st["runs"])) == 9
    assert all("/best/v2" in p for p, _ in st["runs"])
    n = len(st["runs"])
    run(w.best_status())  # nothing new to score: no run
    pr = run(w.probes())  # probes: best's results from the cache, no run either
    assert len(st["runs"]) == n and {x["best"] for x in pr["live"]} == {"pass"}
    assert pr["marks"]["best_n"] == 2 and pr["marks"]["best_sha"] == sha
    assert pr["marks"]["scored"] == 8
    assert any(x["action"] == "best_full" and x["n"] == 2 for x in _log(w))


def test_best_score_is_single_flight_across_callers(mod, tmp_path, monkeypatch):
    """After a best change, 6 best_status calls and 4 gate re-scores of best at once: each of
    best's probes (and the seed score) runs once in all."""
    w = _cfg_world(mod, tmp_path, monkeypatch,
                   "probes: {per_agent_per_hour: 100, gate_sample: 0}\nruns: {parallel: 4}\n")
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    _many_probes(w, 6)
    w._best, w._seed = ("", {}), ("", None)  # a restarted world: nothing cached for best
    st = _counting(mod, monkeypatch, delay=0.05)

    async def go():
        live = w._gate("a" * 64)[0]
        out = await asyncio.gather(*(w.best_status() for _ in range(6)),
                                   *(w._best_score(live) for _ in range(4)))
        await _until_async(lambda: w._fullrec is not None)
        return out

    out = run(go())
    assert all(s["score"] == out[-1]["score"] and s["live_of"] == 6 for s in out[6:])
    assert len(st["runs"]) == 6 + 4 and len(set(st["runs"])) == 10, " ".join(l[0][-2:] for p, l in st["runs"])  # once each
    assert w._fullrec["s"]["score"] == out[-1]["score"]


def test_background_full_score_is_capped_and_low_priority(mod, tmp_path, monkeypatch):
    w = _cfg_world(mod, tmp_path, monkeypatch,
                   "probes: {per_agent_per_hour: 100}\nruns: {parallel: 8, background: 2}\n")
    run(w.promote_best(A, _pkg(w, A, HOME_A), "base"))
    _many_probes(w, 12)
    w._best, w._seed = ("", {}), ("", None)
    st = _counting(mod, monkeypatch, delay=0.05)

    async def go():
        await w.best_status()
        await _until_async(lambda: w._fullrec is not None)

    run(go())
    assert st["best_peak"] == 2 and len(st["runs"]) == 12 + 4


def test_pool_low_priority_waits_for_normal_runs(mod):
    pool, order = mod._Pool(lambda: 1), []
    pool.acquire()  # the one slot is busy

    def take(name, prio):
        pool.acquire(prio=prio)
        order.append(name)
        time.sleep(0.05)
        pool.release()

    low = threading.Thread(target=take, args=("low", mod._Prio(True)))
    low.start()
    time.sleep(0.1)
    normal = threading.Thread(target=take, args=("normal", None))
    normal.start()
    time.sleep(0.1)
    pool.release()
    low.join(5)
    normal.join(5)
    assert order == ["normal", "low"]
    raised = mod._Prio(True)  # a low run a normal caller joined: no longer waits behind
    pool.acquire()
    t = threading.Thread(target=take, args=("raised", raised))
    t.start()
    time.sleep(0.1)
    raised.low = False
    pool.release()
    t.join(5)
    assert order[-1] == "raised" and pool.active == 0


def test_deadline_drops_the_calls_runs_and_frees_the_slots(mod, tmp_path, monkeypatch):
    w = _cfg_world(mod, tmp_path, monkeypatch,
                   "probes: {per_agent_per_hour: 100}\nruns: {parallel: 1, call_deadline_s: 0.3}\n")
    path = _pkg(w, A, HOME_A)
    run(w.promote_best(A, path, "base"))
    _many_probes(w, 6)
    st = _counting(mod, monkeypatch, delay=0.2)

    async def go():
        with pytest.raises(ToolError, match="gave up after 0.3 s") as e:
            await w.call("qualify", {"path": path}, A)  # 4 bar + 4 seed + 6 probe runs
        assert e.value.code == "busy"
        started = len(st["runs"])
        await asyncio.sleep(0.6)  # the run in progress ends; nothing queued starts
        return started

    started = run(go())
    assert started <= 3 and len(st["runs"]) == started
    assert w._slots.active == 0 and w._slots.waiting == []
    t0 = time.monotonic()  # the pool is free: the next call runs at once
    assert run(w.call("run_bot", {"path": path, "lines": ["hi"]}, A))["status"] == "ok"
    assert time.monotonic() - t0 < 0.5
    assert [x["why"] for x in _log(w) if x["action"] == "call_dropped"] == ["deadline"]
    # promote_best is not bounded: its gate (14 runs of 0.2 s, one slot) outlives the deadline
    cfg = tmp_path / "world.yaml"
    cfg.write_text("probes: {per_agent_per_hour: 100}\nruns: {parallel: 1, call_deadline_s: "
                   "0.3}\npromote: {wait_s: 0}\n")
    os.utime(cfg, (time.time() + 5, time.time() + 5))
    _pkg(w, B, HOME_B, REMEMBERS)
    r = run(w.call("promote_best", {"path": f"{HOME_B}/bot", "reason": "x"}, B))
    assert r["promoted"] is True


def test_disconnected_caller_cancels_its_runs(mod, tmp_path, monkeypatch):
    """Over HTTP: the client gives up (its own timeout) on qualify; the world notices the
    disconnect, cancels the call and drops its runs still waiting for a slot."""
    import socket

    import httpx
    import uvicorn

    from flock.world import server as srv

    monkeypatch.setattr(srv, "DISCONNECT_POLL_S", 0.1)
    w = _cfg_world(mod, tmp_path, monkeypatch,
                   "probes: {per_agent_per_hour: 100}\nruns: {parallel: 1, call_deadline_s: 0}\n")
    path = _pkg(w, A, HOME_A)
    run(w.promote_best(A, path, "base"))
    _many_probes(w, 6)
    st = _counting(mod, monkeypatch, delay=0.2)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(srv.create_app(w, "tok"), host="127.0.0.1", port=port,
                                           log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    while not server.started:
        time.sleep(0.05)
    try:
        with pytest.raises(httpx.TimeoutException):
            httpx.post(f"http://127.0.0.1:{port}/tools/qualify", json={"args": {"path": path}},
                       timeout=0.5, headers={"X-World-Token": "tok", "X-Flock-Agent": A.name})
        _until(lambda: any(x["action"] == "call_dropped" for x in _log(w)))
        time.sleep(0.5)
        started = len(st["runs"])
        time.sleep(0.6)
        assert len(st["runs"]) == started and started < 8  # 14 runs without the cancel
        assert w._slots.active == 0
        assert [x["why"] for x in _log(w) if x["action"] == "call_dropped"] == ["cancelled"]
    finally:
        server.should_exit = True
        t.join(5)
