"""The ``canvas`` CLI against a real uvicorn server (stdlib HTTP client)."""

from __future__ import annotations

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
