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
from fastapi.responses import JSONResponse

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


def fake_controller(
    canvas_url: str, root: Path, n: int, types=("painter",), bulk=True, fail_first=0
):
    """Agents run each prompt's command synchronously (forced tool, one step per prompt)."""
    app = FastAPI()
    names = {t: [f"{t}-{i:04d}" for i in range(n)] for t in types}
    ran: list[tuple[str, str]] = []
    calls = {"bulk": 0, "429": 0}

    def run(name: str, text: str):
        cmd = re.search(r"command:\n(.*)$", text, re.DOTALL).group(1)
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

    if bulk:

        @app.post("/swarms/{swarm}/prompts")
        def bulk_prompt(swarm: str, body: dict):
            if calls["429"] < fail_first:
                calls["429"] += 1
                return JSONResponse({"detail": "slow down"}, 429, headers={"Retry-After": "0"})
            calls["bulk"] += 1
            for name in names[body["type"]]:
                run(name, body["text"])
            k = len(names[body["type"]])
            return {"swarm": swarm, "type": body["type"], "matched": k, "queued": k, "pending": 0}

    @app.get("/swarms/{swarm}/agents")
    def agents(
        swarm: str, type: str | None = None, cursor: str | None = None, parent: str | None = None
    ):
        if parent:  # three children per leader (experiment 2 observer)
            return {
                "agents": [
                    {"name": f"{parent}-painter-a-{i:04d}", "state": "idle", "pending": 0}
                    for i in range(3)
                ],
                "cursor": None,
            }
        sel = names[type] if type else [x for v in names.values() for x in v]
        return {"agents": [{"name": x, "state": "idle", "pending": 0} for x in sel], "cursor": None}

    @app.get("/swarms/{swarm}")
    def swarm(swarm: str):
        return {
            "status": {
                "cost_usd": 0.0,
                "pending_prompts": 0,
                "types": {t: {"agents_by_state": {"idle": len(v)}} for t, v in names.items()},
            }
        }

    @app.post("/agents/{name}/prompt")
    def prompt(name: str, body: dict):
        run(name, body["text"])
        return {"prompt_id": f"p{len(ran)}", "agent": name, "status": "queued"}

    return app, ran, calls


def test_driver_paints_to_completion(tmp_path):
    p = tmp_path / "t.png"
    make_target(p)
    canvas_srv, canvas_url = _serve(
        create_app(Settings(target=p, rate_limit=0, token="tok", data_dir=tmp_path / "data"))
    )
    app, ran, _ = fake_controller(canvas_url, tmp_path, 3, bulk=False)
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
            env={**{k: v for k, v in os.environ.items() if k != "SPACE_ID"}, "FLOCK_TOKEN": "x"},
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
        assert row["done_pct"] == 100.0 and "agents_by_state" in row and "idle_pct" in row
        score = json.loads((space / "final/r1/score.json").read_text())
        assert score["coverage_pct"] == 100.0 and score["target"]["width"] == 8
        log = gzip.decompress((space / "final/r1/paint-log.jsonl.gz").read_bytes()).decode()
        assert len(log.splitlines()) == score["seq"]
        assert (space / "final/r1/canvas.png").read_bytes()[:4] == b"\x89PNG"
    finally:
        canvas_srv.should_exit = ctl_srv.should_exit = True


def test_bulk_mode_two_types_with_backoff(tmp_path):
    """Default mode: one bulk request per type per tick (install first), 429s retried."""
    p = tmp_path / "t.png"
    make_target(p)
    canvas_srv, canvas_url = _serve(
        create_app(Settings(target=p, rate_limit=0, token="tok", data_dir=tmp_path / "data"))
    )
    app, ran, calls = fake_controller(canvas_url, tmp_path, 2, types=("pa", "pb"), fail_first=2)
    ctl_srv, ctl_url = _serve(app)
    try:
        r = subprocess.run(
            [
                sys.executable,
                str(HERE / "run" / "driver.py"),
                "--swarm",
                "s",
                "--types",
                "pa,pb",
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
                "--publish-dir",
                str(tmp_path / "space"),
                "--run-id",
                "r2",
                "--gpus",
                "3",
            ],
            capture_output=True,
            text=True,
            timeout=90,
            check=False,
            env={**{k: v for k, v in os.environ.items() if k != "SPACE_ID"}, "FLOCK_TOKEN": "x"},
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "canvas complete" in r.stdout and "bulk POST" in r.stdout
        boots = [n for n, c in ran if "urlretrieve" in c]
        assert sorted(boots) == ["pa-0000", "pa-0001", "pb-0000", "pb-0001"]  # one bulk install
        assert calls["429"] == 2
        rows = [json.loads(x) for x in (tmp_path / "progress.jsonl").read_text().splitlines()]
        assert rows[-1]["http_retries"] >= 2 and rows[-1]["idle_pct"] == 100.0
        assert rows[-1]["agents"] == 4 and "active_agents" in rows[-1]
        space = tmp_path / "space"
        run = json.loads((space / "runs.json").read_text())[-1]
        assert run["gpus"] == 3 and run["status"] == "finished"
        assert (space / "final/r2/agents.png").read_bytes()[:4] == b"\x89PNG"
    finally:
        canvas_srv.should_exit = ctl_srv.should_exit = True


def test_observe_mode_records_leaders_and_regions(tmp_path):
    """Experiment 2: the driver only observes (leaders prompt their children)."""
    p = tmp_path / "t.png"
    make_target(p)
    app_c = create_app(Settings(target=p, rate_limit=0, token="tok", region_enforce=True))
    app_c.state.world.claim_region("leader-0000", 0, 0, 8, 4)
    canvas_srv, canvas_url = _serve(app_c)
    app, ran, calls = fake_controller(canvas_url, tmp_path, 2, types=("leader",))
    ctl_srv, ctl_url = _serve(app)
    try:
        r = subprocess.run(
            [
                sys.executable,
                str(HERE / "run" / "driver.py"),
                "--swarm",
                "s",
                "--observe",
                "--types",
                "leader",
                "--canvas-url",
                canvas_url,
                "--flock-url",
                ctl_url,
                "--interval",
                "0.05",
                "--leader-every",
                "0",
                "--children-per-leader",
                "3",
                "--max-minutes",
                "0.01",
                "--out",
                str(tmp_path / "progress.jsonl"),
                "--prompts-log",
                str(tmp_path / "prompts.jsonl"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**{k: v for k, v in os.environ.items() if k != "SPACE_ID"}, "FLOCK_TOKEN": "x"},
        )
        assert r.returncode == 0, r.stdout + r.stderr
        assert "observe (leaders drive)" in r.stdout
        assert not ran and calls["bulk"] == 0  # no prompts at all
        row = json.loads((tmp_path / "progress.jsonl").read_text().splitlines()[-1])
        ld = row["leaders"]["leader-0000"]
        assert ld["children"] == 3 and ld["children_by_state"] == {"idle": 3}
        assert ld["first_child_s"] is not None and ld["all_children_s"] is not None
        assert row["regions"][0]["owner"] == "leader-0000" and row["regions"][0]["done_pct"] == 0.0
    finally:
        canvas_srv.should_exit = ctl_srv.should_exit = True
