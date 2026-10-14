"""Adapter protocol parsing against recorded fixtures, and harness session-file lookup."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from swarm_harness.models import TokenUsage, TypeSpec
from swarm_harness.worker.adapters.base import ResumeError
from swarm_harness.worker.adapters.claude import (
    ClaudeAdapter,
    ClaudeMapper,
    encode_cwd,
    project_dir,
)
from swarm_harness.worker.adapters.codex import CodexAdapter, CodexMapper, find_rollout

from .worker_fakes import runtime

FIX = Path(__file__).parent / "fixtures"


def load(name):
    return [json.loads(x) for x in (FIX / name).read_text().splitlines()]


# --------------------------------------------------------------------------- claude paths


@pytest.mark.parametrize(
    ("cwd", "enc"),
    [  # real directories under $CLAUDE_CONFIG_DIR/projects on the dev Space and in the resume gate
        ("/data/workspaces/prj-agent-swarms", "-data-workspaces-prj-agent-swarms"),
        ("/home/node/data/workspaces", "-home-node-data-workspaces"),
        ("/tmp/claude-1000/-data-workspaces-prj-agent-swarms/31b897b7-97ff-4873-a155-ee796c62593f/scratchpad/bench",
         "-tmp-claude-1000--data-workspaces-prj-agent-swarms-31b897b7-97ff-4873-a155-ee796c62593f-scratchpad-bench"),
        ("/work/agents/probe-0001", "-work-agents-probe-0001"),
        ("/work/agents/my_agent.v2", "-work-agents-my-agent-v2"),
    ],
)
def test_encode_cwd_real_examples(cwd, enc):
    assert encode_cwd(cwd) == enc


def test_encode_cwd_long_paths_match_sdk():
    from claude_agent_sdk._internal.sessions import _sanitize_path

    long = "/work/" + "/".join(["very-long-directory-name"] * 12)
    assert len(encode_cwd(long)) > 200
    assert encode_cwd(long) == _sanitize_path(long)


def test_project_dir_tolerates_hash_drift(tmp_path):
    long = str(tmp_path / ("x" * 230))
    enc = encode_cwd(long)
    drifted = tmp_path / "home" / "projects" / (enc[:200] + "-bunhash")
    drifted.mkdir(parents=True)
    assert project_dir(tmp_path / "home", long) == drifted


def test_claude_session_files(tmp_path):
    home = tmp_path / "claude"
    wd = tmp_path / "agents" / "a-0001"
    wd.mkdir(parents=True)
    proj = home / "projects" / encode_cwd(wd)
    (proj / "sid-1" / "tool-results").mkdir(parents=True)
    (proj / "sid-1.jsonl").write_text("{}")
    (proj / "sid-1" / "tool-results" / "r.txt").write_text("x")
    (proj / "sid-0.jsonl").write_text("{}")  # an older session: not part of sid-1's set
    a = ClaudeAdapter(wd, home=home)
    assert a.session_files("sid-1") == [proj / "sid-1.jsonl", proj / "sid-1" / "tool-results" / "r.txt"]
    assert a.session_files("nope") == []


# --------------------------------------------------------------------------- claude stream


def claude_events(name):
    from claude_agent_sdk._internal.message_parser import parse_message

    mapper = ClaudeMapper("claude-haiku-5-5")
    out = mapper.on_send("first prompt")
    for rec in load(name):
        if rec.get("sent") == "second":
            out += mapper.on_send("second prompt")
            continue
        msg = parse_message(rec["msg"])
        if msg is not None:
            out += mapper.on_message(msg)
    return out


def test_claude_stream_fixture_maps_to_one_turn():
    evs = claude_events("claude_stream.jsonl")
    kinds = [e.kind for e in evs]
    # 2.1.296 folds the mid-turn prompt into the running turn: one turn, one result
    assert kinds.count("turn_started") == 1 and kinds.count("turn_ended") == 1
    msgs = [e.message for e in evs if e.kind == "message"]
    roles = [m["role"] for m in msgs]
    assert roles[:2] == ["user", "assistant"] and "toolResult" in roles
    call = msgs[1]["content"][0]
    assert call["type"] == "toolCall" and call["name"] == "Bash"
    result = next(m for m in msgs if m["role"] == "toolResult")
    assert result["toolCallId"] == call["id"] and result["content"][0]["text"] == "swarm-fixture"
    assert msgs[-1]["content"][-1]["text"] == "done-1\ndone-2"
    usage = next(e for e in evs if e.kind == "usage")
    assert usage.cumulative and usage.model == "claude-haiku-5-5"
    assert usage.usage == TokenUsage(input=1201, output=254, cache_read=19515, cache_write=21904, reasoning=137)
    assert usage.cost_usd == pytest.approx(0.00482305)
    assert evs[-1].kind == "turn_ended" and evs[-1].status == "completed"


async def test_claude_cumulative_usage_across_resume(tmp_path):
    rt, _ = runtime(tmp_path)
    for name in ("claude_stream.jsonl", "claude_resume.jsonl"):
        for ev in claude_events(name):
            rt.on_adapter_event(ev)
    # modelUsage is a session total across resume: the sum of per-turn deltas equals the last total
    assert rt.state.tokens["claude-haiku-5-5"] == TokenUsage(
        input=1203, output=263, cache_read=41419, cache_write=23038, reasoning=137)
    assert rt.state.turns == 2


class FakeSDKClient:
    """Stands in for ClaudeSDKClient: replays the resume fixture after the first query."""

    def __init__(self, options):
        self.options = options
        self.queries = []
        import asyncio

        self._q = asyncio.Queue()

    async def connect(self):
        pass

    async def query(self, text):
        from claude_agent_sdk._internal.message_parser import parse_message

        self.queries.append(text)
        for rec in load("claude_resume.jsonl"):
            m = parse_message(rec["msg"])
            if m is not None:
                self._q.put_nowait(m)

    async def receive_messages(self):
        while True:
            m = await self._q.get()
            if m is None:
                return
            yield m

    async def disconnect(self):
        self._q.put_nowait(None)


async def test_claude_adapter_plumbing(tmp_path):
    home = tmp_path / "claude"
    wd = tmp_path / "a-0001"
    wd.mkdir()
    clients = []

    def factory(opts):
        clients.append(FakeSDKClient(opts))
        return clients[-1]

    a = ClaudeAdapter(wd, home=home, cli_path="/bin/true", client_factory=factory)
    spec = TypeSpec(harness="claude", model="claude-haiku-5-5", max_turns=5, max_budget_usd=0.5)
    with pytest.raises(ResumeError):
        await a.start(wd, "78b98833-b44f-4bb5-bf10-dd76025ff872", spec)
    sid = await a.start(wd, None, spec, env={"X": "1"})
    o = clients[-1].options
    assert o.session_id == sid and o.resume is None and o.permission_mode == "bypassPermissions"
    assert o.max_turns == 5 and o.max_budget_usd == 0.5 and o.cwd == str(wd) and o.env["X"] == "1"
    await a.send("what word?")
    got = []
    async for ev in a.events():
        got.append(ev)
        if ev.kind == "turn_ended":
            await a.stop()
    assert got[-1].kind == "exited"
    assert [e.kind for e in got].count("turn_started") == 1
    assert next(e for e in got if e.kind == "message" and e.message["role"] == "assistant").message[
        "content"][0]["text"] == "swarm-fixture"
    # resume path passes resume= once a transcript exists
    proj = home / "projects" / encode_cwd(wd)
    proj.mkdir(parents=True)
    (proj / f"{sid}.jsonl").write_text("{}")
    assert await a.start(wd, sid, spec) == sid
    assert clients[-1].options.resume == sid and clients[-1].options.session_id is None
    await a.stop()


# --------------------------------------------------------------------------- codex


def codex_events(name):
    m = CodexMapper("gpt-6-luna")
    out = []
    for rec in load(name):
        msg = rec["msg"]
        if rec["dir"] != "in":
            continue
        if "method" in msg and "id" not in msg:
            out += m.on_notification(msg["method"], msg.get("params") or {})
        elif "result" in msg and isinstance(msg["result"].get("turn"), dict):
            out += m.on_turn_id(msg["result"]["turn"]["id"])
    return out


def test_codex_fixture_maps_steer_into_one_turn():
    evs = codex_events("codex_appserver.jsonl")
    kinds = [e.kind for e in evs]
    assert kinds.count("turn_started") == 1 and kinds.count("turn_ended") == 1
    msgs = [e.message for e in evs if e.kind == "message"]
    users = [m["content"] for m in msgs if m["role"] == "user"]
    assert users == ["Run the shell command `echo codex-fixture`, then reply with exactly: done-1",
                     "Also reply with exactly: done-2"]
    call = next(m for m in msgs if m["role"] == "assistant" and m["content"][0]["type"] == "toolCall")
    assert call["content"][0]["name"] == "shell"
    res = next(m for m in msgs if m["role"] == "toolResult")
    assert res["content"][0]["text"] == "codex-fixture\n" and res["isError"] is False
    assert msgs[-1]["content"][0]["text"] == "done-1"
    last_usage = [e for e in evs if e.kind == "usage"][-1]
    assert last_usage.cumulative and not last_usage.baseline
    assert last_usage.usage == TokenUsage(input=50494 - 28928, output=38, cache_read=28928)


async def test_codex_resume_replay_is_baseline(tmp_path):
    first = codex_events("codex_appserver.jsonl")
    resumed = codex_events("codex_resume.jsonl")
    replay = next(e for e in resumed if e.kind == "usage")
    assert replay.baseline  # the previous total, replayed on resume, tagged with the old turn
    rt, _ = runtime(tmp_path, harness="codex", model="gpt-6-luna")
    for ev in resumed:  # even with no persisted snapshot, nothing is double counted
        rt.on_adapter_event(ev)
    assert rt.state.tokens["gpt-6-luna"] == TokenUsage(
        input=(71121 - 39936) - (50494 - 28928), output=8, cache_read=39936 - 28928)
    rt2, _ = runtime(tmp_path / "b", harness="codex", model="gpt-6-luna")
    for ev in first + resumed:
        rt2.on_adapter_event(ev)
    assert rt2.state.tokens["gpt-6-luna"] == TokenUsage(input=71121 - 39936, output=46, cache_read=39936)


def test_find_rollout(tmp_path):
    home = tmp_path / "codex"
    tid = "01a139ae-ec51-7bc3-9213-ab58f8d13e47"
    d = home / "sessions" / "2026" / "10" / "14"
    d.mkdir(parents=True)
    r = d / f"rollout-2026-10-14T09-12-07-{tid}.jsonl"
    r.write_text("{}")
    (d / "rollout-2026-10-14T09-12-07-01a139ae-0000-0000-0000-000000000000.jsonl").write_text("{}")
    assert find_rollout(home, tid) == r
    assert find_rollout(home, "missing") is None
    a = CodexAdapter(tmp_path / "wd", home=home)
    assert a.session_files(tid) == [r]
    (home / "state_5.sqlite").write_text("")
    assert CodexAdapter(tmp_path / "wd", home=home, extra_globs=("*.sqlite*",)).session_files(tid) == [
        r, home / "state_5.sqlite"]


FAKE_APP_SERVER = r'''
import json, sys, threading, time
active = None
def send(m):
    sys.stdout.write(json.dumps(m) + "\n"); sys.stdout.flush()
def finish(turn):
    global active
    time.sleep(0.2)
    send({"method": "item/completed", "params": {"turnId": turn, "item": {"type": "agentMessage", "id": "m1", "text": "done"}}})
    send({"method": "thread/tokenUsage/updated", "params": {"turnId": turn, "tokenUsage": {"total": {"inputTokens": 100, "cachedInputTokens": 40, "outputTokens": 7, "reasoningOutputTokens": 0, "totalTokens": 107}}}})
    active = None
    send({"method": "turn/completed", "params": {"turn": {"id": turn, "status": "completed"}}})
n = 0
for line in sys.stdin:
    m = json.loads(line)
    meth, rid, p = m.get("method"), m.get("id"), m.get("params") or {}
    if meth == "initialize":
        send({"id": rid, "result": {"codexHome": "x"}})
    elif meth == "thread/start":
        send({"id": rid, "result": {"model": p["model"], "thread": {"id": "thr-1", "path": None}}})
    elif meth == "thread/resume":
        send({"id": rid, "result": {"model": p["model"], "thread": {"id": p["threadId"], "path": None}}})
    elif meth == "turn/start":
        n += 1; active = "turn-%d" % n
        send({"id": rid, "result": {"turn": {"id": active, "status": "inProgress"}}})
        send({"method": "turn/started", "params": {"turn": {"id": active}}})
        for c in p["input"]:
            send({"method": "item/completed", "params": {"item": {"type": "userMessage", "id": "u", "content": [c]}}})
        threading.Thread(target=finish, args=(active,)).start()
    elif meth == "turn/steer":
        if active and p["expectedTurnId"] == active:
            send({"id": rid, "result": {"turnId": active}})
            for c in p["input"]:
                send({"method": "item/completed", "params": {"item": {"type": "userMessage", "id": "u", "content": [c]}}})
        else:
            send({"id": rid, "error": {"code": -32600, "message": "no active turn"}})
'''


async def test_codex_adapter_plumbing(tmp_path):
    script = tmp_path / "fake-codex"
    script.write_text(f"#!{sys.executable}\n" + FAKE_APP_SERVER)
    script.chmod(0o755)
    wd = tmp_path / "wd"
    wd.mkdir()
    home = tmp_path / "codex"
    a = CodexAdapter(wd, home=home, codex_bin=str(script))
    spec = TypeSpec(harness="codex", model="gpt-6-luna")
    with pytest.raises(ResumeError):  # no rollout under CODEX_HOME
        await a.start(wd, "thr-0", spec)
    assert await a.start(wd, None, spec) == "thr-1"
    await a.send("first")
    await a.send("steered")  # active turn -> turn/steer
    got = []
    async for ev in a.events():
        got.append(ev)
        if ev.kind == "turn_ended":
            break
    users = [e.message["content"] for e in got if e.kind == "message" and e.message["role"] == "user"]
    assert users == ["first", "steered"]
    assert [e.kind for e in got].count("turn_started") == 1
    await a.send("next")  # no active turn -> turn/start
    async for ev in a.events():
        if ev.kind == "turn_started":
            assert ev.data["harness_turn_id"] == "turn-2"
            break
    await a.stop()
    kinds = [ev.kind async for ev in a.events()]
    assert kinds[-1] == "exited"
