"""run/driver.py end to end against a real canvas and a fake flock controller whose agents
execute each prompt's command in their own workdir (like the llm harness with a forced tool)."""

from __future__ import annotations

import gzip
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import uvicorn
from conftest import make_target
from fastapi import FastAPI

from canvas.app import create_app
from canvas.config import Settings

HERE = Path(__file__).resolve().parent.parent


def _serve(app):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    return srv, f"http://127.0.0.1:{port}"


def fake_controller(canvas_url: str, root: Path, n: int):
    app = FastAPI()
    names = [f"painter-{i:04d}" for i in range(n)]
    ran: list[tuple[str, str]] = []

    @app.get("/swarms/{swarm}/agents")
    def agents(swarm: str, type: str | None = None, cursor: str | None = None):
        return {
            "agents": [{"name": x, "state": "idle", "pending": 0} for x in names],
            "cursor": None,
        }

    @app.get("/swarms/{swarm}")
    def swarm(swarm: str):
        return {"status": {"cost_usd": 0.0}}

    @app.post("/agents/{name}/prompt")
    def prompt(name: str, body: dict):
        cmd = re.search(r"command:\n(.*)$", body["text"], re.DOTALL).group(1)
        wd = root / name
        wd.mkdir(exist_ok=True)
        env = {
            "PATH": os.environ["PATH"],
            "CANVAS_URL": canvas_url,
            "CANVAS_TOKEN": "tok",
            "FLOCK_AGENT": name,
        }
        subprocess.run(
            ["bash", "-c", cmd], cwd=wd, env=env, capture_output=True, timeout=30, check=False
        )
        ran.append((name, cmd))
        return {"prompt_id": f"p{len(ran)}", "agent": name, "status": "queued"}

    return app, ran


def test_driver_paints_to_completion(tmp_path):
    p = tmp_path / "t.png"
    make_target(p)
    canvas_srv, canvas_url = _serve(
        create_app(Settings(target=p, rate_limit=0, token="tok", data_dir=tmp_path / "data"))
    )
    app, ran = fake_controller(canvas_url, tmp_path, 3)
    ctl_srv, ctl_url = _serve(app)
    try:
        r = subprocess.run(
            [
                sys.executable,
                str(HERE / "run" / "driver.py"),
                "--swarm",
                "s",
                "--canvas-url",
                canvas_url,
                "--flock-url",
                ctl_url,
                "--interval",
                "0.05",
                "--max-minutes",
                "1",
                "--out",
                str(tmp_path / "progress.jsonl"),
                "--prompts-log",
                str(tmp_path / "prompts.jsonl"),
                "--no-bulk",
                "--publish-dir",
                str(tmp_path / "space"),
                "--run-id",
                "r1",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=90,
            env={**os.environ, "FLOCK_TOKEN": "x"},
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "canvas complete" in r.stdout
        boots = [c for _, c in ran if "urlretrieve" in c]
        assert len(boots) == 3  # one install per agent, then claim|paint steps
        assert (tmp_path / "progress.jsonl").read_text().count("\n") >= 2
        assert (tmp_path / "prompts.jsonl").read_text().count('"kind": "step"') >= 5
        # The progress Space files: runs.json, the progress series and the final artefacts.
        space = tmp_path / "space"
        runs = json.loads((space / "runs.json").read_text())
        assert runs[-1]["id"] == "r1" and runs[-1]["status"] == "finished"
        assert runs[-1]["canvas_url"] == canvas_url and runs[-1]["final"] == "final/r1"
        row = json.loads((space / "runs/r1/progress.jsonl").read_text().splitlines()[-1])
        assert row["done_pct"] == 100.0 and "agents_by_state" in row and "turns" in row
        score = json.loads((space / "final/r1/score.json").read_text())
        assert score["coverage_pct"] == 100.0 and score["target"]["width"] == 8
        log = gzip.decompress((space / "final/r1/paint-log.jsonl.gz").read_bytes()).decode()
        assert len(log.splitlines()) == score["seq"]
        assert (space / "final/r1/canvas.png").read_bytes()[:4] == b"\x89PNG"
    finally:
        canvas_srv.should_exit = ctl_srv.should_exit = True
