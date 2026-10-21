"""comms (docs/comms.md) against tests/fake_msgboard.py: spec validation, rules, search cut 1,
join at the first prompt, one poller per worker with fan-out, the gateway, the controller's
channel resolution and kickoff. No real board is called."""

from __future__ import annotations

import json
import re

import pytest
import yaml
from pydantic import ValidationError

from flock import cli
from flock.comms.tools import Refused, check_content, check_rate, query_terms, relaxed_search
from flock.controller.comms import CommsError, resolve
from flock.models import COMMS_TOOLS, CommsRules, CommsSpec, SwarmSpec, comms_tools
from flock.tools.toolset import PIN_ENV, Toolset, read_pin
from flock.worker.agents import AgentRuntime
from flock.worker.comms import CommsHub
from flock.worker.events import Outbox

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

    def __init__(self, tmp_path, board, names, *, harness="claude", protocols=None, **rules):
        self.board = board
        self.agents: dict[str, AgentRuntime] = {}
        self.spec = CommsSpec(url=board.url, invite_env=INVITE_ENV, channel="flock-test",
                              channel_id=CH, rules=CommsRules(**rules),
                              protocols=list(protocols or {}), protocols_text=list((protocols or {}).values()))
        self.hub = CommsHub(swarm="sw", runtime=self.agents.get, env={INVITE_ENV: board.invite},
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
        await until(lambda: all(self.hub.members.get(n) and self.hub.members[n].ident
                                for n in names or self.agents))

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
    check_content(r, "mail a@agents", "new")  # not a group mention
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
        assert [x for x in custom(rt, "comms_joined")][0]["channel_id"] == CH
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
        t.hub.members["dev-0000"].ident  # still joined
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
