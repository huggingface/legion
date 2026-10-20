"""run/leader.md, executed: its literal commands against a real canvas (REGION_ENFORCE=1) and a
fake `flock` CLI whose spawned painters run the prompt's commands like a forced-tool agent."""

from __future__ import annotations

import os
import re
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest
import uvicorn
from conftest import make_target

from canvas.app import create_app
from canvas.config import Settings

HERE = Path(__file__).resolve().parent.parent
LEADER_MD = (HERE / "run" / "leader.md").read_text()

FAKE_FLOCK = r"""#!/usr/bin/env bash
# fake flock: spawn prints a child name; prompt runs the painter's commands in its workdir
S="$FAKE_STATE"; mkdir -p "$S"
case "$1" in
  spawn)
    t="$2"; n=$(cat "$S/n" 2>/dev/null || echo 0); n=$((n+1)); echo $n > "$S/n"
    printf '%s-%s-%04d\n' "$FLOCK_AGENT" "$t" "$n"; echo "$FLOCK_AGENT-$t-$(printf %04d $n)" >> "$S/kids" ;;
  prompt)
    k="$2"; text="$3"; echo "$k" >> "$S/prompts"; mkdir -p "$S/wd/$k"
    install=$(printf '%s\n' "$text" | sed -n 's/^Run this once: //p')
    step=$(printf '%s\n' "$text" | tail -n 1)
    ( cd "$S/wd/$k" && export FLOCK_AGENT="$k" && bash -c "$install" > /dev/null 2>&1
      for i in $(seq 1 3); do out=$(bash -c "$step" 2>&1); echo "$out" | grep -q "You are done" && break; done ) ;;
  agents)
    echo "name state provider worker turns pending cost last_heartbeat"
    while read k; do echo "$k idle - w 1 0 \$0 -"; done < "$S/kids" ;;
  *) echo "fake flock: $*" >&2; exit 2 ;;
esac
"""


def _cmd(prefix: str) -> str:
    """The command block in leader.md that starts with `prefix` (multi-line for P=...)."""
    lines = LEADER_MD.splitlines()
    i = next(j for j, ln in enumerate(lines) if ln.startswith(prefix))
    out = [lines[i]]
    if prefix.startswith("P="):
        while not out[-1].endswith("echo started"):
            i += 1
            out.append(lines[i])
    return "\n".join(out)


@pytest.fixture
def canvas(tmp_path):
    p = tmp_path / "t.png"
    make_target(p, w=40, h=40, seed=9)
    app = create_app(Settings(target=p, rate_limit=0, token="tok", region_enforce=True))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=srv.run, daemon=True).start()
    for _ in range(200):
        if srv.started:
            break
        time.sleep(0.02)
    yield app.state.world, f"http://127.0.0.1:{port}"
    srv.should_exit = True


def test_leader_md_commands_run_end_to_end(tmp_path, canvas):
    world, url = canvas
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "flock").write_text(FAKE_FLOCK)
    (bin_dir / "flock").chmod(0o755)
    wd = tmp_path / "leader"
    wd.mkdir()
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CANVAS_URL": url,
        "CANVAS_TOKEN": "tok",
        "FLOCK_AGENT": "leader-0003",
        "SWARM": "pixel-spawn",
        "FAKE_STATE": str(tmp_path / "st"),
        "HOME": str(tmp_path),
    }

    def run(cmd, extra=""):
        r = subprocess.run(
            ["bash", "-c", extra + cmd],
            cwd=wd,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        assert r.returncode == 0, r.stdout + r.stderr
        return r.stdout

    assert "band 9:" in run(_cmd("python3 -c"))
    out = run(_cmd("B="))
    assert "BAND 3" in out and (wd / "band.txt").exists()
    # another leader wanting band 3 gets the next free band
    env2 = {**env, "FLOCK_AGENT": "leader-0013"}
    (tmp_path / "leader2").mkdir()
    (tmp_path / "leader2" / "canvas").write_bytes((wd / "canvas").read_bytes())
    r = subprocess.run(
        ["bash", "-c", _cmd("B=")],
        cwd=tmp_path / "leader2",
        env=env2,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert "BAND 4" in r.stdout
    out = run(_cmd(": >"))
    kids = (wd / "kids.txt").read_text().split()
    assert len(kids) == 100 and kids[0] == "leader-0003-painter-a-0001"
    assert "granted 100 agents" in out
    assert re.search(r"started", run(_cmd("P=")))
    out = run(_cmd("RC="), extra="sleep(){ :; }; ")
    assert "BAND DONE" in out
    band = [int(v) for v in (wd / "band.txt").read_text().split()]
    s = world.region_score(*band)
    assert s["coverage_pct"] == 100.0
    # only this leader's children painted inside its band
    regs = {r["owner"]: r for r in world.list_regions()}
    assert regs["leader-0003"]["done_pct"] == 100.0
    painters = {n for n, st in world.stats.items() if st.calls}
    assert painters and all(n.startswith("leader-0003-painter-") for n in painters)
