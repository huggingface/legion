"""Re-record tests/fixtures/pi_rpc.jsonl and pi_rpc_server_down.jsonl against the installed `pi`
and tests/fake_openai.py.

    python tests/fixtures/record_pi_rpc.py [path/to/pi]

Scenario (one `pi --mode rpc` process, no real model calls):
  set_steering_mode all, get_state; prompt "hello" (text reply);
  prompt "bash: sleep 1; echo slept" (tool call), two more prompts while the tool runs (queued);
  prompt "please fail!" (endpoint answers 400); prompt "bash: sleep 30" then abort mid-tool;
  get_session_stats. Absolute paths are rewritten to /tmp/pi-fixture.
Server down (pi_rpc_server_down.jsonl): prompt "Continue your work." with nothing listening on
  the endpoint's port (Pi retries on its own, ~15 s), then the server is back and "hello" works.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import fake_openai

FIXTURE_ROOT = "/tmp/pi-fixture"


async def record(pi: str, out: Path, down: bool = False) -> None:
    root = Path(tempfile.mkdtemp(prefix="pi-rec-"))
    srv = fake_openai.serve(0, str(root / "requests.jsonl"))
    port = srv.server_port
    if down:  # the model server went away: nothing listens on its port
        srv.shutdown()
        srv.server_close()
    home, wd = root / "pi", root / "work" / "agents" / "probe-0001"
    wd.mkdir(parents=True)
    home.mkdir()
    (wd / "AGENTS.md").write_text("You are probe-0001.\n")
    (home / "models.json").write_text(json.dumps({"providers": {"fake": {
        "baseUrl": f"http://127.0.0.1:{port}/v1", "api": "openai-completions",
        "apiKey": "$FAKE_KEY", "models": [{"id": "fake-model", "reasoning": False}]}}}))
    env = {"PATH": os.environ["PATH"], "HOME": str(root), "PI_CODING_AGENT_DIR": str(home),
           "PI_OFFLINE": "1", "PI_SKIP_VERSION_CHECK": "1", "PI_TELEMETRY": "0", "FAKE_KEY": "sk-fake"}
    p = await asyncio.create_subprocess_exec(
        pi, "--mode", "rpc", "--session-dir", str(home / "sessions" / "probe-0001"),
        "--session-id", "00000000-0000-4000-8000-000000000001", "--model", "fake/fake-model",
        "--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-approve",
        cwd=str(wd), env=env, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, limit=64 * 1024 * 1024)
    lines: list[str] = []
    settled = asyncio.Event()

    async def read() -> None:
        assert p.stdout
        while line := await p.stdout.readline():
            lines.append(line.decode())
            if json.loads(line).get("type") == "agent_settled":
                settled.set()

    reader = asyncio.create_task(read())

    async def send(cmd: dict) -> None:
        assert p.stdin
        p.stdin.write((json.dumps(cmd) + "\n").encode())
        await p.stdin.drain()

    async def run(cmd: dict, then: list[tuple[float, dict]] = ()) -> None:
        settled.clear()
        await send(cmd)
        for delay, c in then:
            await asyncio.sleep(delay)
            await send(c)
        await asyncio.wait_for(settled.wait(), 30)

    await send({"type": "set_steering_mode", "id": "r1", "mode": "all"})
    await send({"type": "get_state", "id": "r2"})
    if down:
        await run({"type": "prompt", "id": "r3", "message": "Continue your work.",
                   "streamingBehavior": "steer"})
        srv = fake_openai.serve(port, str(root / "requests.jsonl"))  # back on the same port
        await run({"type": "prompt", "id": "r4", "message": "hello", "streamingBehavior": "steer"})
    else:
        await run({"type": "prompt", "id": "r3", "message": "hello", "streamingBehavior": "steer"})
        await run({"type": "prompt", "id": "r4", "message": "bash: sleep 1; echo slept", "streamingBehavior": "steer"},
                  [(0.4, {"type": "prompt", "id": "r5", "message": "also this", "streamingBehavior": "steer"}),
                   (0.0, {"type": "prompt", "id": "r6", "message": "and this", "streamingBehavior": "steer"})])
        await run({"type": "prompt", "id": "r7", "message": "please fail!", "streamingBehavior": "steer"})
        await run({"type": "prompt", "id": "r8", "message": "bash: sleep 30", "streamingBehavior": "steer"},
                  [(0.8, {"type": "abort", "id": "r9"})])
        await send({"type": "get_session_stats", "id": "r10"})
    await asyncio.sleep(0.5)
    assert p.stdin
    p.stdin.close()
    await p.wait()
    await reader
    srv.shutdown()
    text = "".join(lines).replace(str(root), FIXTURE_ROOT).replace(str(port), "8000")
    out.write_text(text)
    print(f"{len(lines)} records -> {out} ({time.strftime('%Y-%m-%d')})")


if __name__ == "__main__":
    pi = sys.argv[1] if len(sys.argv) > 1 else "pi"
    asyncio.run(record(pi, HERE / "pi_rpc.jsonl"))
    asyncio.run(record(pi, HERE / "pi_rpc_server_down.jsonl", down=True))
