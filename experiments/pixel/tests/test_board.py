"""run/board.py against a fake msgboard (join, post, reply, read, react, transcript)."""

import importlib.util
import json
import os
import socket
import stat
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse

BOARD = Path(__file__).resolve().parent.parent / "run" / "board.py"
spec = importlib.util.spec_from_file_location("board", BOARD)
board = importlib.util.module_from_spec(spec)
spec.loader.exec_module(board)


def fake_board():
    app = FastAPI()
    st = {
        "tokens": {},
        "msgs": [],
        "keys": {},
        "fail_next_post": 0,
        "seen_keys": [],
        "reactions": [],
    }

    def who(auth: str | None):
        tok = (auth or "").removeprefix("Bearer ")
        return st["tokens"].get(tok)

    @app.post("/v1/join")
    async def join(req: Request):
        b = await req.json()
        if b.get("code") != "INV":
            return JSONResponse({"error": {"code": "not_found", "message": "bad invite"}}, 404)
        name = b.get("name")
        for t in [t for t, n in st["tokens"].items() if n == name]:
            del st["tokens"][t]  # rejoin rotates the token, keeps the identity
        tok = f"fbt_{name}_{len(st['tokens'])}_{time.time_ns()}"
        st["tokens"][tok] = name
        return {"token": tok, "principal": {"id": 7, "external_id": name}, "created": True}

    @app.post("/v1/channels/{ch}/messages")
    async def post(
        ch: int,
        req: Request,
        authorization: str = Header(None),
        idempotency_key: str = Header(None),
    ):
        name = who(authorization)
        if not name:
            return JSONResponse({"error": {"code": "unauthorized", "message": "no"}}, 401)
        if not idempotency_key:
            return JSONResponse({"error": {"code": "bad_request", "message": "key"}}, 400)
        st["seen_keys"].append(idempotency_key)
        if st["fail_next_post"]:
            st["fail_next_post"] -= 1
            return JSONResponse(
                {"error": {"code": "rate", "message": "slow"}}, 429, headers={"Retry-After": "0"}
            )
        if idempotency_key in st["keys"]:
            return st["keys"][idempotency_key]
        b = await req.json()
        m = {
            "id": len(st["msgs"]) + 1,
            "channel_id": ch,
            "seq": len(st["msgs"]) + 1,
            "sender": {"external_id": name, "display_name": name},
            "body": b["body"],
            "reply_to": b.get("reply_to"),
            "thread_id": b.get("reply_to"),
            "created_at": time.time(),
        }
        st["msgs"].append(m)
        st["keys"][idempotency_key] = m
        return JSONResponse(m, 201)

    @app.get("/v1/channels")
    def channels(name: str | None = None, authorization: str = Header(None)):
        if not who(authorization):
            return JSONResponse({"error": {"code": "unauthorized", "message": "no"}}, 401)
        chans = [{"id": 5, "name": "flock-pixel-test", "kind": "channel"}]
        return {"channels": [c for c in chans if name in (None, c["name"])]}

    @app.get("/v1/activity")
    def activity(
        channels: str, after: int | None = None, limit: int = 50, authorization: str = Header(None)
    ):
        if not who(authorization):
            return JSONResponse({"error": {"code": "unauthorized", "message": "no"}}, 401)
        start = after or 0
        page = st["msgs"][start : start + 2]  # small pages: exercises has_more
        nxt = start + len(page)
        return {
            "messages": [{**m, "reason": "channel"} for m in page],
            "next_cursor": nxt,
            "has_more": nxt < len(st["msgs"]),
        }

    @app.put("/v1/channels/{ch}/messages/{mid}/reactions/{emoji}")
    def react(ch: int, mid: int, emoji: str, authorization: str = Header(None)):
        st["reactions"].append((who(authorization), mid, emoji))
        return {"id": mid}

    @app.get("/v1/channels/{ch}/messages")
    def history(ch: int, after: int = 0, limit: int = 50, authorization: str = Header(None)):
        msgs = [m for m in st["msgs"] if m["channel_id"] == ch and m["seq"] > after][:2]
        return {"messages": msgs, "next_cursor": msgs[-1]["seq"] if msgs else after}

    return app, st


@pytest.fixture
def board_env(tmp_path, monkeypatch):
    app, st = fake_board()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("BOARD_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("BOARD_INVITE", "INV")
    monkeypatch.delenv("BOARD_CHANNEL", raising=False)
    monkeypatch.setenv("FLOCK_AGENT", "leader-0003")
    monkeypatch.delenv("BOARD_TOKEN", raising=False)
    yield st
    srv.should_exit = True


def test_join_post_read_react_transcript(board_env, capsys):
    st = board_env
    assert board.main(["read"]) == 1
    assert "not joined yet" in capsys.readouterr().err
    assert board.main(["join"]) == 0
    out = capsys.readouterr().out
    assert "joined as leader-0003" in out and "fbt_" not in out  # token never printed
    assert "channel #5" in out and Path(".board_channel").read_text() == "5"  # found by name
    tok = Path(".board_token").read_text()
    assert stat.S_IMODE(os.stat(".board_token").st_mode) == 0o600
    assert board.main(["join"]) == 0  # rejoin rotates the token
    capsys.readouterr()
    assert Path(".board_token").read_text() != tok and len(st["tokens"]) == 1

    st["fail_next_post"] = 1  # a 429 is retried with the SAME idempotency key
    assert board.main(["post", "leader-0003: leasing region 0,0,1024,103"]) == 0
    assert st["seen_keys"][0] == st["seen_keys"][1] and len(st["msgs"]) == 1
    assert board.main(["reply", "1", "@leader-0007 can your painters help?"]) == 0
    assert board.main(["post", "third"]) == 0
    capsys.readouterr()
    assert board.main(["read"]) == 0  # pages through has_more
    out = capsys.readouterr().out.splitlines()
    assert out == [
        "#1 leader-0003: leader-0003: leasing region 0,0,1024,103",
        "#2 leader-0003 (re #1): @leader-0007 can your painters help?",
        "#3 leader-0003: third",
    ]
    assert board.main(["read"]) == 0
    assert capsys.readouterr().out.strip() == "(no new messages)"  # cursor advanced
    assert board.main(["react", "2", "👍"]) == 0
    assert st["reactions"] == [("leader-0003", 2, "👍")]
    capsys.readouterr()
    assert board.main(["transcript"]) == 0
    lines = [json.loads(x) for x in capsys.readouterr().out.splitlines()]
    assert [x["id"] for x in lines] == [1, 2, 3] and lines[1]["reply_to"] == 1
    assert board.main(["post"]) == 2


def test_explicit_channel_id_and_multi_use_invite(board_env, capsys, monkeypatch):
    monkeypatch.setenv("BOARD_CHANNEL", "5")
    for name in ("leader-0001", "leader-0002"):  # one code, many leaders
        monkeypatch.setenv("FLOCK_AGENT", name)
        assert board.main(["join"]) == 0
        assert f"joined as {name}" in capsys.readouterr().out
    assert len(board_env["tokens"]) == 2
