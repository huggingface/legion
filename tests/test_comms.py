"""comms (docs/comms.md) against tests/fake_msgboard.py: spec validation, the board rules (the
Harbor board resource applies them), the controller's channel resolution and kickoff. No real
board is called."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flock.comms.board import SyncBoard
from flock.comms.tools import Refused, check_content, check_rate
from flock.controller.comms import CommsError, resolve
from flock.models import (
    COMMS_TOOLS,
    CommsRules,
    CommsSpec,
    SwarmSpec,
    comms_tools,
    normalize_board_url,
)

from . import fake_msgboard
from .conftest import FAKE_PROVIDERS, HB

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
            "providers": FAKE_PROVIDERS, "types": {"dev": {**HB, "model": "fake:m", "comms": True}}}


# --------------------------------------------------------------------------- spec


def test_comms_block_validates():
    s = SwarmSpec.model_validate(swarm_raw(rules={"min_post_gap": "20s", "max_posts": 80,
                                                  "threads": "per_topic", "no_group_mentions": True}))
    assert s.comms.channel == "flock-test" and s.comms.wake == ["mention", "thread"]
    assert s.comms.rules.gap_s == 20 and s.comms.search == "board"
    assert comms_tools(s.types["dev"], s.comms) == list(COMMS_TOOLS)
    with pytest.raises(ValidationError, match="not supported"):
        SwarmSpec.model_validate(swarm_raw(search="index"))
    with pytest.raises(ValidationError, match="extra"):
        SwarmSpec.model_validate(swarm_raw(rules={"min_gap": "1s"}))
    raw = swarm_raw()
    raw["types"]["dev"] = {**HB, "model": "m", "tools": ["comms:shout"]}
    with pytest.raises(ValidationError, match="unknown comms tool"):
        SwarmSpec.model_validate(raw)
    raw = swarm_raw()
    del raw["comms"]
    with pytest.raises(ValidationError, match="swarm-level `comms:` block"):
        SwarmSpec.model_validate(raw)
    raw = swarm_raw()
    raw["types"]["dev"] = {**HB, "model": "fake:m", "comms": False, "tools": ["comms:post"]}
    with pytest.raises(ValidationError, match="comms: false"):
        SwarmSpec.model_validate(raw)


def test_listed_comms_tools_and_folder_tools():
    raw = swarm_raw(search=None)
    raw["types"]["dev"] = {**HB, "model": "fake:m", "tools": ["ws_write", "comms:post"]}
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


# --------------------------------------------------------------------------- worker hub


# --------------------------------------------------------------------------- controller


def test_controller_resolves_channel_and_posts_kickoff_once(board):
    raw = swarm_raw(url=board.url, kickoff="Kickoff: build the bot. Reply in this thread.")
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


# --------------------------------------------------------------------------- board urls


def test_board_urls_have_one_spelling():
    for url in ("http://board.invalid/", "HTTP://Board.Invalid", "http://board.invalid:80//"):
        assert normalize_board_url(url) == "http://board.invalid"
    assert normalize_board_url("https://b.invalid:8443/x/") == "https://b.invalid:8443/x"



URLS = [("http://[::1]:8080/", "http://[::1]:8080"), ("https://[2001:db8::1]/", "https://[2001:db8::1]"),
        ("http://[2001:DB8::1]:80/", "http://[2001:db8::1]"), ("http://Board.invalid:80/", "http://board.invalid"),
        ("https://x.y:443", "https://x.y"), ("http://x.y:8080/path/", "http://x.y:8080/path"),
        ("http://user@x.y/", "http://user@x.y")]


@pytest.mark.parametrize(("url", "want"), URLS)
def test_normalised_board_urls_are_idempotent_and_build_clients(url, want):
    once = normalize_board_url(url)
    assert once == want and normalize_board_url(once) == once
    spec = CommsSpec(url=url, invite_env="INV", channel="team")
    assert spec.url == want
    sync = SyncBoard(spec.url)
    assert str(sync.http.base_url).rstrip("/") == want
    sync.close()


