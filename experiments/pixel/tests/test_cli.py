"""The ``canvas`` CLI against a real uvicorn server (stdlib HTTP client)."""

import json
import socket
import threading
import time

import canvas_cli
import pytest
import uvicorn
from conftest import make_target

from canvas.app import create_app
from canvas.config import Settings


@pytest.fixture
def server(tmp_path, monkeypatch):
    p = tmp_path / "t.png"
    t = make_target(p)
    app = create_app(Settings(target=p, rate_limit=0, token="tok", max_pixels_per_call=4))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    th = threading.Thread(target=srv.run, daemon=True)
    th.start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    monkeypatch.setenv("CANVAS_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("CANVAS_TOKEN", "tok")
    monkeypatch.setenv("FLOCK_AGENT", "cli-agent")
    monkeypatch.setenv("FLOCK_TURN", "turn-7")
    yield app.state.world, t
    srv.should_exit = True
    th.join(5)


def test_claim_paint_flow(server, capsys):
    world, _t = server
    assert canvas_cli.main(["claim"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("claimed 4 px")
    cmd = out.strip().splitlines()[-1]
    assert cmd.startswith("canvas paint ")
    assert canvas_cli.main(["paint", *cmd.split()[2:]]) == 0
    out = capsys.readouterr().out
    assert "4 useful" in out and "px to go" in out
    e = json.loads(world.read_log(0, 1)[0])
    assert e["agent"] == "cli-agent" and e["turn"] == "turn-7"


def test_json_and_errors(server, capsys):
    _world, t = server
    assert canvas_cli.main(["get", "2", "3", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["hex"] == "{:02x}{:02x}{:02x}".format(
        *tuple(t[3, 2])
    )
    assert canvas_cli.main(["get", "1", "1", "2", "2"]) == 0
    assert "y=2:" in capsys.readouterr().out
    pix = [f"{x},1,ffffff" for x in range(1, 6)]
    assert canvas_cli.main(["paint", *pix]) == 1
    err = capsys.readouterr().err
    assert "too many pixels: 5 > 4" in err and "split into calls of 4" in err
    assert canvas_cli.main(["paint", "1,1,red"]) == 1
    assert "want 6 hex digits" in capsys.readouterr().err
    assert canvas_cli.main(["paint", "1,1,010203"]) == 0
    assert "wrong: target is" in capsys.readouterr().out
    for cmd in (["score"], ["status"], ["info"], ["release"]):
        assert canvas_cli.main(cmd) == 0, cmd
    out = capsys.readouterr().out
    assert "exact match" in out and "limits: 4 px per paint" in out and "released" in out


def test_auth_error_is_actionable(server, capsys, monkeypatch):
    monkeypatch.setenv("CANVAS_TOKEN", "wrong")
    assert canvas_cli.main(["claim"]) == 2
    assert "unknown token" in capsys.readouterr().err


def test_unreachable(monkeypatch, capsys):
    monkeypatch.setenv("CANVAS_URL", "http://127.0.0.1:9")
    assert canvas_cli.main(["status"]) == 3
    assert "check CANVAS_URL" in capsys.readouterr().err


def test_claim_pipe_paint(server, capsys, monkeypatch):
    import io

    _world, _ = server
    assert canvas_cli.main(["claim", "3", "--json"]) == 0
    claim = capsys.readouterr().out
    assert len(json.loads(claim)["pixels"]) == 3
    monkeypatch.setattr("sys.stdin", io.StringIO(claim))
    assert canvas_cli.main(["paint", "-", "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["counts"] == {"useful": 3} and out["seq"] == 1
    # An empty claim (finished canvas) is a clean no-op with a clear message.
    empty = json.dumps({"pixels": [], "runs": [], "remaining": 0, "done": True, "lease_s": 120})
    monkeypatch.setattr("sys.stdin", io.StringIO(empty))
    assert canvas_cli.main(["paint", "-"]) == 0
    assert "You are done" in capsys.readouterr().out
    monkeypatch.setattr("sys.stdin", io.StringIO('{"error": "rate limited"}'))
    assert canvas_cli.main(["paint", "-"]) == 1
    assert canvas_cli.main(["claim", "x"]) == 1


def test_cli_served_by_canvas(server):
    import os
    import urllib.request

    src = urllib.request.urlopen(os.environ["CANVAS_URL"] + "/cli/canvas").read().decode()
    assert src.startswith("#!/usr/bin/env python3") and "def main(" in src


def test_region_verbs(server, capsys, monkeypatch):
    import io

    monkeypatch.setenv("FLOCK_AGENT", "leader-0000")
    assert canvas_cli.main(["claim-region", "1", "1", "4", "4", "--ttl", "600"]) == 0
    assert "leased region #" in capsys.readouterr().out
    assert canvas_cli.main(["claim-region", "1", "1", "4", "4"]) == 0
    assert "renewed" in capsys.readouterr().out
    monkeypatch.setattr(
        "sys.stdin", io.StringIO("leader-0000-painter-a-0001\nleader-0000-painter-a-0002\n")
    )
    assert canvas_cli.main(["grant-region", "-"]) == 0
    assert "granted 2 agents" in capsys.readouterr().out
    assert canvas_cli.main(["regions"]) == 0
    out = capsys.readouterr().out
    assert "leader-0000" in out and "2 granted" in out and "% done" in out
    assert canvas_cli.main(["score", "--region", "1,1,4,4"]) == 0
    assert "region 1,1,4,4" in capsys.readouterr().out
    assert canvas_cli.main(["target-info"]) == 0
    capsys.readouterr()
    assert canvas_cli.main(["bands", "2"]) == 0
    assert capsys.readouterr().out.startswith("band 0: claim-region 0 0 8 ")
    assert canvas_cli.main(["release-region"]) == 0
    assert "released 1 region" in capsys.readouterr().out
    assert canvas_cli.main(["claim-region", "1", "x", "4", "4"]) == 1
