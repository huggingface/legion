"""comms (docs/comms.md) against tests/fake_msgboard.py: spec validation, rules, search cut 1,
join at the first prompt, one poller per worker with fan-out, the gateway, the controller's
channel resolution and kickoff. No real board is called."""

from __future__ import annotations

import asyncio
import json
import re
from unittest.mock import patch

import httpx
import pytest
import yaml
from pydantic import ValidationError

from flock import cli
from flock.comms.board import AsyncBoard, BoardError, SyncBoard, same_board
from flock.comms.tools import Refused, check_content, check_rate, query_terms, relaxed_search
from flock.controller.comms import CommsError, resolve
from flock.models import (
    COMMS_TOOLS,
    CommsRules,
    CommsSpec,
    SwarmSpec,
    TypeSpec,
    comms_tools,
    normalize_board_url,
)
from flock.tools.toolset import PIN_ENV, Toolset, read_pin
from flock.worker.agents import AgentRuntime
from flock.worker.comms import CommsHub, Source
from flock.worker.events import Outbox
from flock.worker.identity import Identities, MsgboardIdentity

from . import fake_msgboard
from .worker_fakes import ScriptedAdapter, spec, until

INVITE_ENV = "T_INVITE"
CH = 12


@pytest.fixture
def board():
    b, srv, url = fake_msgboard.serve()
    b.url = url
    yield b
    srv.shutdown()


def swarm_raw(**comms) -> dict:
    return {"swarm": "sw", "namespace": "ns", "bucket": "hf://buckets/ns/b",
            "comms": {"url": "http://board", "invite_env": INVITE_ENV, "channel": "#flock-test",
                      **comms},
            "types": {"dev": {"harness": "claude", "model": "m", "comms": True}}}


def custom(rt: AgentRuntime, kind: str) -> list[dict]:
    return [e.data["data"] for e in list(rt.outbox._q) if e.type == kind]


def posted(board) -> int:
    return sum(1 for m, p, _, _ in board.requests if m == "POST" and p.endswith("/messages"))


class Team:
    """N agents of one worker sharing one CommsHub, against the fake board."""

    def __init__(self, tmp_path, board, names, *, harness="claude", protocols=None, env=None,
                 **rules):
        self.board = board
        self.agents: dict[str, AgentRuntime] = {}
        self.spec = CommsSpec(url=board.url, invite_env=INVITE_ENV, channel="flock-test",
                              channel_id=CH, rules=CommsRules(**rules),
                              protocols=list(protocols or {}), protocols_text=list((protocols or {}).values()))
        self.hub = CommsHub(swarm="sw", runtime=self.agents.get,
                            env=env if env is not None else {INVITE_ENV: board.invite},
                            poll_wait=0.3, backoff_s=0.01)
        self.hub.configure(self.spec, list(COMMS_TOOLS))
        base = Toolset(version=None, dir=None, swarm="sw", type="dev", data_dir=tmp_path / "data")
        for n in names:
            self.agents[n] = AgentRuntime(
                n, "dev", spec(harness=harness), swarm="sw", work_root=tmp_path / "agents",
                adapter=ScriptedAdapter(tmp_path / "home" / n), outbox=Outbox(),
                toolbox=lambda n=n: self.hub.toolset_for(n, base), comms=self.hub)

    async def start(self, *names):
        for n in names or self.agents:
            rt = self.agents[n]
            await rt.setup(restore=False)
            await rt.prompt("hello")
        # comms_joined comes after the whole join, including the extended mode's "now" cursor:
        # a message posted before that is history, not news, and is never delivered.
        await until(lambda: all(self.hub.members.get(n) and self.hub.members[n].ident
                                and custom(self.agents[n], "comms_joined") for n in names or self.agents))

    async def call(self, agent, tool, **args):
        r = await self.hub.call(agent, f"comms:{tool}", args)
        return json.loads(r.output), r.error

    async def close(self):
        for rt in self.agents.values():
            await rt.stop()
        await self.hub.aclose()


# --------------------------------------------------------------------------- spec


def test_comms_block_validates():
    s = SwarmSpec.model_validate(swarm_raw(rules={"min_post_gap": "20s", "max_posts": 80,
                                                  "threads": "per_topic", "no_group_mentions": True}))
    assert s.comms.channel == "flock-test" and s.comms.wake == ["mention", "thread"]
    assert s.comms.rules.gap_s == 20 and s.comms.search == "board"
    assert comms_tools(s.types["dev"], s.comms) == list(COMMS_TOOLS)
    with pytest.raises(ValidationError, match="not built yet"):
        SwarmSpec.model_validate(swarm_raw(search="index"))
    with pytest.raises(ValidationError, match="extra"):
        SwarmSpec.model_validate(swarm_raw(rules={"min_gap": "1s"}))
    raw = swarm_raw()
    raw["types"]["dev"] = {"harness": "llm", "model": "m", "tools": ["comms:shout"]}
    with pytest.raises(ValidationError, match="unknown comms tool"):
        SwarmSpec.model_validate(raw)
    raw = swarm_raw()
    del raw["comms"]
    with pytest.raises(ValidationError, match="swarm-level `comms:` block"):
        SwarmSpec.model_validate(raw)
    raw = swarm_raw()
    raw["types"]["dev"] = {"harness": "claude", "model": "m", "comms": False, "tools": ["comms:post"]}
    with pytest.raises(ValidationError, match="comms: false"):
        SwarmSpec.model_validate(raw)


def test_listed_comms_tools_and_folder_tools():
    raw = swarm_raw(search=None)
    raw["types"]["dev"] = {"harness": "claude", "model": "m", "tools": ["ws_write", "comms:post"]}
    s = SwarmSpec.model_validate(raw)
    ts = s.types["dev"]
    assert ts.folder_tools == ["ws_write"] and comms_tools(ts, s.comms) == ["comms:post"]
    ts.tools = ["comms:read"]
    assert ts.folder_tools is None and comms_tools(ts, s.comms) == ["comms:read"]
    ts.tools, ts.comms = None, True
    assert comms_tools(ts, s.comms) == ["comms:post", "comms:read"]  # search: off
    raw["types"]["dev"]["tools"] = ["comms:search"]
    with pytest.raises(ValidationError, match="comms.search"):
        SwarmSpec.model_validate(raw)


def test_cli_inlines_kickoff_and_protocols(tmp_path):
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "kickoff.md").write_text("Team: build Eliza.")
    (tmp_path / "lead-agree.md").write_text("One leads, others agree.")
    raw = swarm_raw(kickoff_file="prompts/kickoff.md", protocols=["lead-agree.md"])
    f = tmp_path / "spec.yaml"
    f.write_text(yaml.safe_dump(raw))
    s = cli.load_spec(str(f))
    assert s.comms.kickoff == "Team: build Eliza." and s.comms.protocols_text == ["One leads, others agree."]


# --------------------------------------------------------------------------- rules and search


def test_content_rules():
    r = CommsRules(max_len=10, no_group_mentions=True, threads="per_topic")
    for text, thread, rule in [("x" * 11, "new", "max_len"), ("hi @agents", "new", "no_group_mentions"),
                               ("hi @All", "new", "no_group_mentions"), ("hi", None, "threads"),
                               ("  ", "new", "empty")]:
        with pytest.raises(Refused) as e:
            check_content(r, text, thread)
        assert e.value.rule == rule and str(e.value).startswith("refused: ")
    check_content(r, "a@agents", "new")  # an address, not a group mention
    check_content(CommsRules(), "hi @agents", None)  # no rules: anything goes


def test_rate_rules():
    r = CommsRules(min_post_gap="20s", max_posts=3)
    check_rate(r, 0, None, 100.0)
    with pytest.raises(Refused, match=r"refused: wait 15 s \(min_post_gap 20s"):
        check_rate(r, 1, 95.0, 100.0)
    check_rate(r, 1, 79.0, 100.0)
    with pytest.raises(Refused, match="all 3 posts"):
        check_rate(r, 3, None, 100.0)


async def test_search_relaxation_drops_trailing_words():
    asked = []

    async def search(q, limit):
        asked.append((q, limit))
        if "zebra" in q:
            return {"messages": [], "total": 0}
        return {"messages": [{"id": 7, "thread_id": 5, "sender": {"external_id": "ds-1"},
                              "created_at": 0, "snippet": "the \x02bot\x03 passes"}], "total": 1}

    out = await relaxed_search(search, query='bot "fallback ladder" zebra in:#other', channel="eliza",
                               author_="@ds-1", since="12h", limit=5)
    assert [q for q, _ in asked] == ['bot "fallback ladder" zebra in:#eliza from:@ds-1 after:12h',
                                     'bot "fallback ladder" in:#eliza from:@ds-1 after:12h']
    assert out["query"] == 'bot "fallback ladder"' and out["dropped"] == ["zebra"]
    assert out["results"] == [{"id": 7, "thread": 5, "author": "ds-1", "time": "1970-01-01T00:00:00Z",
                               "snippet": "the **bot** passes"}]
    assert query_terms("in:#x from:@y") == []
    with pytest.raises(ValueError, match="since"):
        await relaxed_search(search, query="bot", channel="c", since="last week")


# --------------------------------------------------------------------------- worker hub


async def test_join_at_first_prompt_token_stays_in_worker(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"], protocols={"protocols/lead-agree.md": "One leads; others say agree."})
    rt = t.agents["dev-0000"]
    try:
        await rt.setup(restore=False)
        assert board.joins == [] and not t.hub.members  # assigned, not joined
        await rt.prompt("hello")
        await until(lambda: t.hub.members.get("dev-0000") and t.hub.members["dev-0000"].ident)
        assert board.joins == ["dev-0000"]
        token = t.hub.members["dev-0000"].ident.token
        assert custom(rt, "comms_joined")[0]["channel_id"] == CH
        # never in the harness env, the workdir or the pin file
        env = rt.adapter.envs[0]
        assert token not in json.dumps(env) and "MSGBOARD_TOKEN" not in env
        files = [p for p in (tmp_path / "agents").rglob("*") if p.is_file()]
        assert files and not any(token in p.read_text(errors="replace") for p in files)
        pin = json.loads((tmp_path / "agents" / "_tools_pins" / "dev-0000.json").read_text())
        assert pin["toolset"]["gateway"] == t.hub.address and pin["toolset"]["gateway_key"]
        assert env[PIN_ENV].endswith("dev-0000.json")
        # the instructions: comms section with the protocol, and the tools section
        md = (rt.workdir / "CLAUDE.md").read_text()
        assert "You are @dev-0000 on the message board channel #flock-test" in md
        assert "### Protocol: lead-agree\n\nOne leads; others say agree." in md
        assert "`flock-tool post --json" in md
    finally:
        await t.close()


async def test_relaunch_rejoins_by_name(tmp_path, board):
    a = Team(tmp_path / "w1", board, ["dev-0000"])
    b = Team(tmp_path / "w2", board, ["dev-0000"])
    try:
        await a.start()
        old = a.hub.members["dev-0000"].ident
        await b.start()  # the agent moved to another worker
        new = b.hub.members["dev-0000"].ident
        assert new.id == old.id and new.token != old.token
        out, err = await b.call("dev-0000", "post", text="hi from the new worker", thread="new")
        assert not err and out["posted"]
    finally:
        await a.close()
        await b.close()


async def test_one_poller_fans_out_to_the_agents_it_concerns(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000", "dev-0001", "dev-0002"])
    a, b, c = (t.agents[n] for n in ("dev-0000", "dev-0001", "dev-0002"))
    try:
        await t.start()
        out, err = await t.call("dev-0001", "post", text="topic: fallback ladder", thread="new")
        assert not err
        root = out["thread"]
        await until(lambda: t.hub._cursor is not None)
        board.post("carlos", CH, "@dev-0000 can you look at the probes?")
        board.post("carlos", CH, "I agree with the ladder", reply_to=root)
        board.post("carlos", CH, "@agents status please")  # a group mention wakes nobody
        await until(lambda: custom(a, "comms_mention") and custom(b, "comms_thread_reply"))
        [m] = custom(a, "comms_mention")
        assert m["from"] == "carlos" and m["thread"] == m["msg_id"]
        [r] = custom(b, "comms_thread_reply")
        assert r == {"msg_id": root + 2, "thread": root, "from": "carlos"}
        # dev-0001's own post through the hub fanned out too: a reply by dev-0000 wakes it
        await t.call("dev-0000", "post", text="on it", thread=root)
        await until(lambda: len(custom(b, "comms_thread_reply")) == 2)
        assert not custom(c, "comms_mention") and not custom(c, "comms_thread_reply")
        assert not custom(b, "comms_mention")
        pollers = {who for m_, p, q, who in board.requests if p == "/v1/activity" and "wait" in q}
        assert len(pollers) == 1  # one long-poll for the worker, not one per agent
    finally:
        await t.close()


async def test_each_rule_refuses_before_the_board(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"], min_post_gap="30s", max_posts=2, max_len=40,
             threads="per_topic", no_group_mentions=True)
    try:
        await t.start()
        before = posted(board)
        cases = [({"text": "x" * 41, "thread": "new"}, "max_len is 40"),
                 ({"text": "hey @all", "thread": "new"}, "@all is not allowed"),
                 ({"text": "no thread"}, 'thread: \\"new\\"'),
                 ({"text": "bad thread", "thread": "abc"}, "must be a message id")]
        for args, why in cases:
            out, err = await t.call("dev-0000", "post", **args)
            assert err and why.replace("\\", "") in out["error"], out
        assert posted(board) == before
        out, err = await t.call("dev-0000", "post", text="first", thread="new")
        assert not err and out["posts_left"] == 1
        out, err = await t.call("dev-0000", "post", text="second", thread="new")
        assert err and re.match(r"refused: wait (29|30) s", out["error"]), out
        assert posted(board) == before + 1
        reply = board.post("carlos", CH, "a reply", reply_to=out_root(board))
        rt = t.agents["dev-0000"]
        rt.state.comms_last_post = 0.0  # as if 30 s had passed
        out, err = await t.call("dev-0000", "post", text="in a reply's thread", thread=reply["id"])
        assert err and f"use thread: {reply['thread_id']}" in out["error"]
        out, err = await t.call("dev-0000", "post", text="ok", thread=reply["thread_id"], reply_to=reply["id"])
        assert not err and out["thread"] == reply["thread_id"] and out["posts_left"] == 0
        rt.state.comms_last_post = 0.0
        out, err = await t.call("dev-0000", "post", text="third", thread="new")
        assert err and "all 2 posts" in out["error"]
        assert [x["rule"] for x in custom(rt, "comms_refused")] == [
            "max_len", "no_group_mentions", "threads", "min_post_gap", "threads", "max_posts"]
        assert rt.state.comms_posts == 2  # kept in the synced state
    finally:
        await t.close()


def out_root(board) -> int:
    return next(m["id"] for m in board.msgs if m["thread_id"] is None)


async def test_read_and_search(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"])
    try:
        await t.start()
        root = board.post("carlos", CH, "kickoff: build the fallback ladder")["id"]
        board.post("ds-1", CH, "the ladder breaks on ok", reply_to=root)
        for i in range(3):
            board.post("carlos", CH, f"noise {i}")
        out, err = await t.call("dev-0000", "read", n=2)
        assert not err and [m["text"] for m in out["messages"]] == ["noise 1", "noise 2"]
        out, err = await t.call("dev-0000", "read", thread=str(root + 1))  # a reply: its thread
        assert not err and out["thread"] == root and len(out["messages"]) == 2
        out, err = await t.call("dev-0000", "search", query="ladder breaks zebra quokka")
        assert not err and out["dropped"] == ["zebra", "quokka"]
        assert [(h["id"], h["thread"], h["author"]) for h in out["results"]] == [(root + 1, root, "ds-1")]
        assert "**ladder**" in out["results"][0]["snippet"]
        out, err = await t.call("dev-0000", "search", query="ladder", thread=root, limit=1)
        assert not err and len(out["results"]) == 1
        q = [q["q"] for m, p, q, _ in board.requests if p == "/v1/search"]
        assert q[0] == "ladder breaks zebra quokka in:#flock-test"
        assert all(x["sort"] == "relevance" for m, p, x, _ in board.requests if p == "/v1/search")
        out, err = await t.call("dev-0000", "search", query="")
        assert err and "no words" in out["error"]
    finally:
        await t.close()


async def test_read_defaults_to_the_run_since_the_kickoff(tmp_path, board):
    """Defect 8: on a reused channel `read` leaves out history from before the kickoff unless
    asked with `since`; a thread read is not cut by default."""
    old = [board.post("carlos", CH, f"old {i}")["id"] for i in range(3)]
    kickoff = board.post("sw-host", CH, "kickoff")["id"]
    t = Team(tmp_path, board, ["dev-0000"])
    t.spec.kickoff_id = kickoff  # set at apply, carried in the spec the worker gets
    try:
        await t.start()
        board.post("ds-1", CH, "new 1", reply_to=kickoff)
        out, err = await t.call("dev-0000", "read")
        assert not err and [m["text"] for m in out["messages"]] == ["kickoff", "new 1"]
        assert out["since"] == kickoff and "truncated" not in out and "since:" in out["note"]
        out, err = await t.call("dev-0000", "read", since=f"#{old[0]}")  # strictly newer
        assert not err and [m["text"] for m in out["messages"]] == ["old 1", "old 2", "kickoff", "new 1"]
        assert "truncated" not in out and "note" not in out
        out, err = await t.call("dev-0000", "read", n=1)
        assert [m["text"] for m in out["messages"]] == ["new 1"] and out["truncated"] is True
        out, err = await t.call("dev-0000", "read", thread=old[0])  # an explicit thread: as asked
        assert not err and [m["text"] for m in out["messages"]] == ["old 0"]
        out, err = await t.call("dev-0000", "read", since="yesterday")
        assert err and "since must be a message id" in out["error"]
    finally:
        await t.close()


async def test_flock_tool_reaches_comms_through_the_gateway(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"])
    try:
        await t.start()
        ts, agent, workdir = read_pin(tmp_path / "agents" / "_tools_pins" / "dev-0000.json")
        assert ts.builtin_call is None and ts.get("post").builtin == "comms:post"
        r = await ts.call("post", {"text": "via flock-tool", "thread": "new"}, agent=agent,
                          workdir=workdir, env={})
        assert not r.error and json.loads(r.output)["posted"]
        assert board.msgs[-1]["sender"]["external_id"] == "dev-0000"
        r = await ts.call("post", {"text": 3}, agent=agent, workdir=workdir, env={})
        assert r.error and "text must be string" in r.output
        bad = Toolset(**{**ts.__dict__, "gateway_key": "forged"})
        r = await bad.call("read", {}, agent=agent, workdir=workdir, env={})
        assert r.error and "unknown agent or key" in r.output
    finally:
        await t.close()


# --------------------------------------------------------------------------- controller


def test_controller_resolves_channel_and_posts_kickoff_once(board):
    raw = swarm_raw(url=board.url, kickoff="Kickoff: build Eliza. Reply in this thread.")
    s1 = SwarmSpec.model_validate(raw)
    resolve(s1, None, {INVITE_ENV: board.invite})
    assert s1.comms.channel_id == CH and s1.comms.kickoff_id == board.msgs[0]["id"]
    assert board.msgs[0]["sender"]["external_id"] == "sw-host" and len(board.msgs) == 1
    s2 = SwarmSpec.model_validate({**raw, "comms": {**raw["comms"], "rules": {"max_posts": 5},
                                                    "channel_id": 999}})
    n = len(board.requests)
    resolve(s2, s1, {})  # same board and channel: copied, no board call, no second kickoff
    assert (s2.comms.channel_id, s2.comms.kickoff_id) == (CH, s1.comms.kickoff_id)
    assert len(board.requests) == n and len(board.msgs) == 1
    s3 = SwarmSpec.model_validate(raw)
    resolve(s3, None, {INVITE_ENV: board.invite})  # a lost record: the Idempotency-Key holds
    assert s3.comms.kickoff_id == s1.comms.kickoff_id and len(board.msgs) == 1
    with pytest.raises(CommsError, match="not set on the controller"):
        resolve(SwarmSpec.model_validate(raw), None, {})
    with pytest.raises(CommsError, match="no channel #nope"):
        resolve(SwarmSpec.model_validate(swarm_raw(url=board.url, channel="nope")), None,
                {INVITE_ENV: board.invite})
    with pytest.raises(CommsError, match="not inlined"):
        resolve(SwarmSpec.model_validate(swarm_raw(url=board.url, kickoff_file="k.md")), None,
                {INVITE_ENV: board.invite})
    with pytest.raises(CommsError, match="invite link is not valid"):
        resolve(SwarmSpec.model_validate(swarm_raw(url=board.url)), None, {INVITE_ENV: "wrong"})


def test_controller_refuses_an_archived_channel_before_any_post_or_join():
    """Defect 7: an archived channel is refused at resolve, not at the kickoff post (409)."""
    b, srv, url = fake_msgboard.serve(fake_msgboard.Board(archived={CH}))
    try:
        with pytest.raises(CommsError, match=r"channel flock-test \(id 12\) is archived"):
            resolve(SwarmSpec.model_validate(swarm_raw(url=url, kickoff="Kickoff.")), None,
                    {INVITE_ENV: b.invite})
        assert [(m, p) for m, p, _, _ in b.requests] == [("POST", "/v1/join"), ("GET", "/v1/channels")]
        assert b.msgs == []
    finally:
        srv.shutdown()


# --------------------------------------------------------------------------- review of #16


def polling(hub) -> bool:
    return hub._poller is not None and not hub._poller.done()


async def test_tools_off_then_on_restarts_the_poller(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"])
    rt = t.agents["dev-0000"]
    try:
        await t.start()
        await until(lambda: t.hub._cursor is not None)
        t.hub.configure(t.spec, [])  # a re-apply that turns the type's comms tools off
        await until(lambda: not polling(t.hub))
        t.hub.configure(t.spec, list(COMMS_TOOLS))  # and on again, inside the same burst
        await until(lambda: polling(t.hub))
        await until(lambda: t.hub._cursor is not None)
        board.post("carlos", CH, "@dev-0000 still there?")
        await until(lambda: custom(rt, "comms_mention"))
        assert board.joins == ["dev-0000"]  # the cached identity was enough
    finally:
        await t.close()


async def test_channel_move_rejoins_active_agents_without_a_new_burst(tmp_path):
    b, srv, url = fake_msgboard.serve(fake_msgboard.Board(channels={12: "flock-test", 13: "new-team"}))
    b.url = url
    t = Team(tmp_path, b, ["dev-0000", "dev-0001"])
    t.agents["dev-0001"].spec.idle_grace = "30s"
    try:
        await t.start("dev-0000")  # dev-0001 has had no prompt: it joins only at its first
        old = t.hub.members["dev-0000"].ident.token
        await until(lambda: t.hub._cursor is not None)  # the poller is running
        t.hub.configure(t.spec.model_copy(update={"channel": "new-team", "channel_id": 13}),
                        list(COMMS_TOOLS))
        b.post("carlos", 13, "@dev-0000 over here now")  # before the rejoin is done: not missed
        await until(lambda: t.hub.members["dev-0000"].ident is not None and polling(t.hub))
        assert t.hub.members["dev-0000"].ident.token != old  # joined again, not the cached token
        assert b.joins == ["dev-0000", "dev-0000"] and "dev-0001" not in t.hub.members
        await until(lambda: custom(t.agents["dev-0000"], "comms_mention"))
        out, err = await t.call("dev-0000", "read")
        assert not err and out["channel"] == "#new-team"
    finally:
        await t.close()
        srv.shutdown()


async def test_invite_change_redeems_the_new_invite_for_a_private_channel(tmp_path):
    b, srv, url = fake_msgboard.serve(fake_msgboard.Board(
        channels={12: "old-private", 13: "new-private"}, private={12, 13},
        invites={"inv-old": [12], "inv-new": [13]}))
    b.url = url
    env = {"OLD_INVITE": "inv-old", "NEW_INVITE": "inv-new"}
    t = Team(tmp_path, b, ["dev-0000"], env=env)
    t.spec = t.spec.model_copy(update={"invite_env": "OLD_INVITE", "channel": "old-private"})
    t.hub.configure(t.spec, list(COMMS_TOOLS))
    rt = t.agents["dev-0000"]
    try:
        await t.start()
        out, err = await t.call("dev-0000", "read")
        assert not err, out
        # the channel moves, but the invite does not cover it: a failed join, never "joined"
        t.hub.configure(t.spec.model_copy(update={"channel": "new-private", "channel_id": 13}),
                        list(COMMS_TOOLS))
        await until(lambda: custom(rt, "comms_join_failed"))
        assert t.hub.members["dev-0000"].ident is None
        assert "not visible to dev-0000" in custom(rt, "comms_join_failed")[-1]["error"]
        out, err = await t.call("dev-0000", "read")
        assert err and "not on the board yet" in out["error"]
        # with the invite that is scoped to it, the agent redeems it and reads
        t.hub.configure(t.spec.model_copy(update={"channel": "new-private", "channel_id": 13,
                                                  "invite_env": "NEW_INVITE"}), list(COMMS_TOOLS))
        await until(lambda: t.hub.members["dev-0000"].ident is not None)
        b.post("carlos", 13, "hello new team")
        out, err = await t.call("dev-0000", "read")
        assert not err and out["channel"] == "#new-private"
        assert [m["text"] for m in out["messages"]] == ["hello new team"]
        assert len(custom(rt, "comms_joined")) == 2
    finally:
        await t.close()
        srv.shutdown()


async def test_thread_read_gives_the_newest_messages_and_says_when_truncated(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"])
    try:
        await t.start()
        root = board.post("carlos", CH, "root")["id"]
        ids = [board.post("carlos", CH, f"reply {i}", reply_to=root)["id"] for i in range(105)]
        out, err = await t.call("dev-0000", "read", thread=root, n=5)
        assert not err and [m["id"] for m in out["messages"]] == ids[-5:]
        assert out["truncated"] is True and "older ones exist" in out["note"]
        out, err = await t.call("dev-0000", "read", thread=root)  # default: the newest 100
        got = [m["id"] for m in out["messages"]]
        assert len(got) == 100 and got[-1] == ids[-1] and out["truncated"] is True
        out, err = await t.call("dev-0000", "read", thread=ids[50], n=200)  # a reply: its thread
        assert out["thread"] == root and len(out["messages"]) == 100
        short = board.post("carlos", CH, "short topic")["id"]
        board.post("carlos", CH, "one reply", reply_to=short)
        out, err = await t.call("dev-0000", "read", thread=short)
        assert not err and len(out["messages"]) == 2 and "truncated" not in out
        out, err = await t.call("dev-0000", "read", n=1)
        assert [m["text"] for m in out["messages"]] == ["one reply"] and out["truncated"] is True
    finally:
        await t.close()


async def test_gateway_replies_with_an_error_when_a_tool_raises(tmp_path, board):
    t = Team(tmp_path, board, ["dev-0000"])
    try:
        await t.start()

        async def boom(*a):
            raise RuntimeError("kaput")

        t.hub.call = boom
        ts, agent, workdir = read_pin(tmp_path / "agents" / "_tools_pins" / "dev-0000.json")
        r = await ts.call("post", {"text": "x", "thread": "new"}, agent=agent,
                          workdir=workdir, env={})
        assert r.error and "RuntimeError: kaput" in r.output
    finally:
        await t.close()


async def test_concurrent_gateway_startup_opens_one_listener():
    hub = CommsHub(swarm="sw", env={})
    started = []
    real = asyncio.start_server

    async def capture(*a, **kw):
        srv = await real(*a, **kw)
        started.append(srv)
        return srv

    try:
        with patch("flock.worker.comms.asyncio.start_server", capture):
            addresses = await asyncio.gather(*(hub.start_gateway() for _ in range(8)))
        assert len(started) == 1 and len(set(addresses)) == 1
        await hub.aclose()
        assert not started[0].is_serving()
    finally:
        for srv in started:
            srv.close()
        await hub.identities._client.aclose()


# --------------------------------------------------------------------------- re-review of #16


async def test_a_join_in_flight_during_reconfiguration_redeems_the_new_invite(tmp_path):
    b, srv, url = fake_msgboard.serve(fake_msgboard.Board(
        channels={12: "old-private", 13: "new-private"}, private={12, 13},
        invites={"inv-old": [12], "inv-new": [13]}))
    b.url = url
    t = Team(tmp_path, b, ["dev-0000"], env={"OLD_INVITE": "inv-old", "NEW_INVITE": "inv-new"})
    old = t.spec.model_copy(update={"invite_env": "OLD_INVITE", "channel": "old-private"})
    hub = t.hub
    hub.configure(old, list(COMMS_TOOLS))
    started, release = asyncio.Event(), asyncio.Event()
    real = hub.identities.join
    codes = []

    async def gated(agent, url, code, *, again=False):
        codes.append(code)
        if code == "inv-old":
            started.set()
            await release.wait()  # the old join is still in flight when the settings change
        return await real(agent, url, code, again=again)

    hub.identities.join = gated
    try:
        first = asyncio.create_task(hub.on_prompt("dev-0000", "llm"))
        await asyncio.wait_for(started.wait(), 5)
        hub.configure(old.model_copy(update={"invite_env": "NEW_INVITE", "channel": "new-private",
                                             "channel_id": 13}), list(COMMS_TOOLS))
        release.set()
        await first
        await hub._restore
        assert codes == ["inv-old", "inv-new"]  # the stale result was dropped and joined again
        b.post("carlos", 13, "welcome")
        for _ in range(2):
            out, err = await t.call("dev-0000", "read")
            assert not err and [m["text"] for m in out["messages"]] == ["welcome"], out
    finally:
        release.set()
        await t.close()
        srv.shutdown()


async def test_mentions_for_an_agent_still_rejoining_are_delivered_once_it_is_back(tmp_path):
    b, srv, url = fake_msgboard.serve(fake_msgboard.Board(channels={12: "flock-test", 13: "new-team"}))
    b.url = url
    t = Team(tmp_path, b, ["dev-0000", "dev-0001"])
    hub = t.hub
    entered, release = asyncio.Event(), asyncio.Event()
    real = hub.identities.join

    async def gated(agent, url, code, *, again=False):
        if agent == "dev-0001" and again:
            entered.set()
            await release.wait()
        return await real(agent, url, code, again=again)

    try:
        await t.start()
        await until(lambda: hub._cursor is not None)
        hub.identities.join = gated
        hub.configure(t.spec.model_copy(update={"channel": "new-team", "channel_id": 13}),
                      list(COMMS_TOOLS))
        await asyncio.wait_for(entered.wait(), 5)
        assert hub.members["dev-0000"].ident is not None and hub.members["dev-0001"].ident is None
        during = b.post("carlos", 13, "@dev-0001 while you rejoin")["id"]
        b.post("dev-0001", 13, "my own words")  # never a wake for its sender
        await until(lambda: hub._cursor >= during + 1)  # the poller has consumed both
        release.set()
        await hub._restore
        after = b.post("carlos", 13, "@dev-0001 after the rejoin")["id"]
        rt = t.agents["dev-0001"]
        await until(lambda: len(custom(rt, "comms_mention")) == 2)
        assert [x["msg_id"] for x in custom(rt, "comms_mention")] == [during, after]
        assert not hub.members["dev-0001"].pending and not hub.members["dev-0001"].backlog
    finally:
        release.set()
        await t.close()
        srv.shutdown()


# --------------------------------------------------------------------------- third review of #16


async def test_a_board_client_refuses_a_token_from_another_board():
    seen = []
    board = AsyncBoard("http://b.invalid", transport=httpx.MockTransport(
        lambda r: seen.append(r) or httpx.Response(200, json={"channels": []})))
    try:
        with pytest.raises(BoardError) as e:
            await board.channels(MsgboardIdentity(url="http://a.invalid", token="tok-a", principal="x"))
        assert e.value.code == "wrong_board" and "tok-a" not in str(e.value)
        assert seen == []  # nothing was sent
        await board.channels(MsgboardIdentity(url="http://b.invalid/", token="tok-b", principal="x"))
        assert [r.headers["authorization"] for r in seen] == ["Bearer tok-b"]
    finally:
        await board.aclose()


async def test_a_join_in_flight_when_the_board_url_changes_never_sends_its_token_to_the_new_board():
    blocked, release = asyncio.Event(), asyncio.Event()
    requests: list[tuple[str, str, str]] = []  # (host, path, bearer)

    async def handler(request):
        host, path = request.url.host, request.url.path
        bearer = request.headers.get("authorization", "").removeprefix("Bearer ")
        requests.append((host, path, bearer))
        if path == "/v1/join":
            code = json.loads(request.content)["code"]
            if code == "old-invite":
                blocked.set()
                await release.wait()
            return httpx.Response(200, json={"token": f"tok-{host}",
                                             "principal": {"id": 1, "external_id": "dev-0000"}})
        if path == "/v1/channels":
            return httpx.Response(200, json={"channels": [{"id": 12 if host == "old.invalid" else 13,
                                                           "member": True}]})
        if path == "/v1/activity":
            await asyncio.Event().wait()
        if path.endswith("/messages"):
            return httpx.Response(200, json={"messages": [], "has_more": False})
        return httpx.Response(200, json={})

    hub = CommsHub(swarm="sw", env={"OLD": "old-invite", "NEW": "new-invite"},
                   transport=httpx.MockTransport(handler), poll_wait=0.1)
    old = CommsSpec(url="http://old.invalid", invite_env="OLD", channel="team", channel_id=12)
    try:
        hub.configure(old, list(COMMS_TOOLS))
        first = asyncio.create_task(hub.on_prompt("dev-0000", "llm"))
        await asyncio.wait_for(blocked.wait(), 5)
        hub.configure(old.model_copy(update={"url": "http://new.invalid", "invite_env": "NEW",
                                             "channel_id": 13}), list(COMMS_TOOLS))
        release.set()
        await first
        await hub._restore
        r = await hub.call("dev-0000", "comms:read", {})
        assert not r.error, r.output
        assert hub.members["dev-0000"].ident.token == "tok-new.invalid"
        # the old board's token went to the old board's /v1/join answer only, never anywhere
        assert not [x for x in requests if x[2] == "tok-old.invalid"]
        assert all(b in ("", f"tok-{h}") for h, _, b in requests)
        assert [(h, p) for h, p, _ in requests if p == "/v1/join"] == [
            ("old.invalid", "/v1/join"), ("new.invalid", "/v1/join")]
    finally:
        release.set()
        await hub.aclose()
        await hub.identities._client.aclose()


# --------------------------------------------------------------------------- fourth review of #16


async def test_a_post_finishing_on_the_old_board_never_hides_a_mention_on_the_new_one(tmp_path):
    a, sa, ua = fake_msgboard.serve()
    a.url = ua
    b, sb, ub = fake_msgboard.serve()  # the same ids and channel id as board A
    b.url = ub
    t = Team(tmp_path, a, ["dev-0000", "dev-0001"])
    hub = t.hub
    entered, release = asyncio.Event(), asyncio.Event()
    try:
        await t.start()
        await until(lambda: hub._cursor is not None)
        old = hub._board(ua)
        real = old.post

        async def gated(*args, **kw):
            entered.set()
            await release.wait()
            return await real(*args, **kw)

        old.post = gated
        pending = asyncio.create_task(t.call("dev-0000", "post", text="@dev-0001 from board A",
                                             thread="new"))
        await asyncio.wait_for(entered.wait(), 5)
        hub.configure(t.spec.model_copy(update={"url": ub}), list(COMMS_TOOLS))
        release.set()
        out, err = await pending
        assert not err and a.msgs[-1]["id"] == out["posted"]  # it did post, on board A
        await hub._restore
        await until(lambda: hub._cursor is not None)
        mid = b.post("carlos", CH, "@dev-0001 a real mention on board B")["id"]
        assert mid == out["posted"]  # the same id on both boards
        rt = t.agents["dev-0001"]
        await until(lambda: custom(rt, "comms_mention"))
        assert custom(rt, "comms_mention") == [{"msg_id": mid, "thread": mid, "from": "carlos"}]
        assert hub._cache[(ub, CH, mid)]["body"] == "@dev-0001 a real mention on board B"
        assert (ua, CH, mid) not in hub._cache  # the superseded post fed nothing
    finally:
        release.set()
        await t.close()
        sa.shutdown()
        sb.shutdown()


# --------------------------------------------------------------------------- fifth review of #16


def test_board_urls_have_one_spelling():
    for url in ("http://board.invalid/", "HTTP://Board.Invalid", "http://board.invalid:80//"):
        assert normalize_board_url(url) == "http://board.invalid"
    assert normalize_board_url("https://b.invalid:8443/x/") == "https://b.invalid:8443/x"
    assert same_board("http://board.invalid/", "http://BOARD.invalid")
    assert not same_board("http://board.invalid", "http://board.invalid:8080")
    assert MsgboardIdentity(url="http://board.invalid/", token="t", principal="p").url == "http://board.invalid"
    assert Source("http://board.invalid/", 12, 0) == Source("http://board.invalid", 12, 0)


async def test_legacy_identities_with_a_trailing_slash_url_still_fan_out():
    async def handler(request):
        path = request.url.path
        if path == "/v1/join":
            name = json.loads(request.content)["name"]
            return httpx.Response(200, json={"token": f"tok-{name}", "principal": {
                "id": 1 if name == "dev-0000" else 2, "external_id": name}})
        if path == "/v1/channels":
            return httpx.Response(200, json={"channels": [{"id": 12, "member": True}]})
        if path == "/v1/activity":
            await asyncio.Event().wait()
        if path.endswith("/messages"):
            return httpx.Response(201, json={"id": 42, "channel_id": 12, "body": "@dev-0001 hello",
                                             "sender": {"external_id": "dev-0000"},
                                             "mentions": [{"id": 2, "external_id": "dev-0001",
                                                           "as": "dev-0001"}]})
        return httpx.Response(200, json={})

    transport = httpx.MockTransport(handler)
    ids = Identities(env={"INV": "inv"}, client=httpx.AsyncClient(transport=transport))
    legacy = TypeSpec(harness="claude", model="m", msgboard_invite="INV",
                      env={"MSGBOARD_URL": "http://board.invalid/"})
    for name in ("dev-0000", "dev-0001"):  # the older msgboard_invite path joins first
        assert (await ids.env_for(name, legacy))["MSGBOARD_URL"] == "http://board.invalid"
    spec = CommsSpec(url="http://board.invalid/", invite_env="INV", channel="team", channel_id=12)
    hub = CommsHub(swarm="sw", identities=ids, env={"INV": "inv"}, transport=transport)
    hub.configure(spec, list(COMMS_TOOLS))
    try:
        for name in ("dev-0000", "dev-0001"):
            await hub.on_prompt(name, "llm")
            assert hub.members[name].ident is ids.get(name, "http://board.invalid")  # reused
        r = await hub.call("dev-0000", "comms:post", {"text": "@dev-0001 hello", "thread": "new"})
        assert not r.error, r.output
        assert ("http://board.invalid", 12, 42) in hub._cache
        assert hub.stats["mentions"] == 1 and 42 in hub.members["dev-0000"].threads
    finally:
        await hub.aclose()
        await ids._client.aclose()


# --------------------------------------------------------------------------- sixth review of #16

URLS = [("http://[::1]:8080/", "http://[::1]:8080"), ("https://[2001:db8::1]/", "https://[2001:db8::1]"),
        ("http://[2001:DB8::1]:80/", "http://[2001:db8::1]"), ("http://Board.invalid:80/", "http://board.invalid"),
        ("https://x.y:443", "https://x.y"), ("http://x.y:8080/path/", "http://x.y:8080/path"),
        ("http://user@x.y/", "http://user@x.y")]


@pytest.mark.parametrize(("url", "want"), URLS)
async def test_normalised_board_urls_are_idempotent_and_build_clients(url, want):
    once = normalize_board_url(url)
    assert once == want and normalize_board_url(once) == once
    spec = CommsSpec(url=url, invite_env="INV", channel="team")
    assert spec.url == want
    sync = SyncBoard(spec.url)
    sync.close()
    board = AsyncBoard(spec.url)
    try:
        assert board.url == want and same_board(board.url, url)
    finally:
        await board.aclose()


async def test_read_since_is_exclusive_and_says_caught_up(tmp_path, board):
    """Polling with `since: <last id seen>` returns only newer messages, and an explicit
    "caught up" note when there are none (it used to return the `since` message itself, so a
    poll was never empty)."""
    kickoff = board.post("sw-host", CH, "kickoff")["id"]
    t = Team(tmp_path, board, ["dev-0000"])
    t.spec.kickoff_id = kickoff
    try:
        await t.start()
        first = board.post("ds-1", CH, "one", reply_to=kickoff)["id"]
        out, err = await t.call("dev-0000", "read", since=kickoff)
        assert not err and [m["text"] for m in out["messages"]] == ["one"]
        out, err = await t.call("dev-0000", "read", since=first)
        assert not err and out["messages"] == []
        assert out["note"] == f"caught up: nothing newer than {first}"
        board.post("ds-1", CH, "two")
        out, err = await t.call("dev-0000", "read", since=first)
        assert not err and [m["text"] for m in out["messages"]] == ["two"] and "note" not in out
    finally:
        await t.close()


def test_full_jitter_envelope_and_retry_after(monkeypatch):
    from flock.worker import comms as comms_mod

    monkeypatch.setattr(comms_mod.random, "uniform", lambda a, b: b)
    assert [comms_mod.full_jitter(n, 30.0, 600.0) for n in (1, 2, 3, 6, 9)] == \
        [30.0, 60.0, 120.0, 600.0, 600.0]
    monkeypatch.setattr(comms_mod.random, "uniform", lambda a, b: a)
    assert comms_mod.full_jitter(4, 30.0, 600.0) == 0.0
    assert comms_mod.full_jitter(4, 30.0, 600.0, retry_after=45.0) == 45.0
    assert comms_mod.full_jitter(4, 30.0, 600.0, floor=1.0) == 1.0


async def test_a_rate_limited_join_waits_at_least_retry_after_and_backs_off(monkeypatch):
    from flock.worker import comms as comms_mod

    now = [1000.0]
    monkeypatch.setattr(comms_mod.random, "uniform", lambda a, b: b)

    def handler(request):
        if request.url.path == "/v1/join":
            return httpx.Response(429, headers={"retry-after": "90"})
        return httpx.Response(200, json={})

    hub = CommsHub(swarm="sw", env={"INV": "code"}, transport=httpx.MockTransport(handler),
                   clock=lambda: now[0])
    hub.configure(CommsSpec(url="http://b.invalid", invite_env="INV", channel="team", channel_id=12),
                  list(COMMS_TOOLS))
    try:
        await hub.on_prompt("dev-0000", "llm")
        m = hub.members["dev-0000"]
        assert m.join_failures == 1 and m.retry_at - now[0] == 90.0  # Retry-After beats 30 s
        with pytest.raises(BoardError, match="not on the board yet"):
            await hub._member("dev-0000")
        now[0] = m.retry_at + 1
        await hub.on_prompt("dev-0000", "llm")
        assert m.join_failures == 2 and m.retry_at - now[0] == 90.0  # 60 s envelope < 90 s
        now[0] = m.retry_at + 1
        await hub.on_prompt("dev-0000", "llm")
        assert m.join_failures == 3 and m.retry_at - now[0] == 120.0  # the envelope grew
    finally:
        await hub.aclose()


@pytest.mark.parametrize("reply, why", [
    (httpx.Response(404, text="<html>Space not found</html>"),
     "HTTP 404: non-JSON reply (an HF Space gate or proxy?)"),
    (httpx.Response(403, json={"error": {"code": "invite_exhausted", "message": "x" * 300,
                                         "details": {"code": "INVITE-SECRET"}}}),
     "HTTP 403: invite_exhausted: " + "x" * (200 - len("invite_exhausted: "))),
])
async def test_a_refused_join_names_the_boards_reason_never_the_request(reply, why):
    hub = CommsHub(swarm="sw", env={"INV": "INVITE-SECRET"},
                   transport=httpx.MockTransport(lambda request: reply))
    hub.configure(CommsSpec(url="http://b.invalid", invite_env="INV", channel="team", channel_id=12),
                  list(COMMS_TOOLS))
    try:
        await hub.on_prompt("dev-0000", "llm")
        err = hub.members["dev-0000"].error
        assert err == f"join refused: {why}" and "INVITE-SECRET" not in err
    finally:
        await hub.aclose()

