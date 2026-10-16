"""Pi resume gate driver: talks to `pi --mode rpc` directly (as the claude/codex gates use their
CLIs) against stub_openai.py in the same container, so there is no inference spend.

    gate.py A   two sessions in WORKDIR:
                  main: "the codeword is PELICAN-7" (text) + "bash: echo PELICAN-7 > notes.txt" (tool)
                  cut:  "bash: sleep 60; echo late", Pi SIGKILLed mid-tool
                writes OUT/A.json (session files, get_session_stats, timings, memory)
    gate.py B   after restore: resume both ids from a fresh PI_CODING_AGENT_DIR holding only the
                restored session files + a new models.json; writes OUT/B.json with the verdict

Env: PI_CODING_AGENT_DIR (required), WORKDIR (default /work/agents/probe-0001), OUT (default ./out),
     SID_MAIN, SID_CUT (required), PI (default: pi on PATH).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stub_openai

HOME = Path(os.environ["PI_CODING_AGENT_DIR"])
WORKDIR = Path(os.environ.get("WORKDIR", "/work/agents/probe-0001"))
AGENT = WORKDIR.name
SESS = HOME / "sessions" / AGENT
OUT = Path(os.environ.get("OUT", "out"))
PI = os.environ.get("PI") or shutil.which("pi") or "pi"
FLAGS = ["--no-extensions", "--no-mcp", "--no-skills", "--no-prompt-templates", "--no-approve"]


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()


def cgroup_mem() -> int | None:
    for p in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            return int(Path(p).read_text())
        except (OSError, ValueError):
            continue
    return None


def vmrss_kb(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def descendants(pid: int) -> list[int]:
    kids: dict[int, list[int]] = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            with contextlib.suppress(OSError):
                st = Path(f"/proc/{d}/stat").read_text()
                kids.setdefault(int(st[st.rfind(")") + 2:].split()[1]), []).append(int(d))
    out, todo = [], [pid]
    while todo:
        for c in kids.get(todo.pop(), []):
            out.append(c)
            todo.append(c)
    return out


class Pi:
    def __init__(self, sid: str, port: int) -> None:
        self.sid, self.port = sid, port
        self.records: list[dict] = []
        self.settled = asyncio.Event()
        self.pending: dict[str, asyncio.Future] = {}
        self.n = 0
        self.stderr = b""

    async def start(self) -> float:
        HOME.mkdir(parents=True, exist_ok=True)
        (HOME / "models.json").write_text(json.dumps({"providers": {"stub": {
            "baseUrl": f"http://127.0.0.1:{self.port}/v1", "api": "openai-completions", "apiKey": "$STUB_KEY",
            "models": [{"id": "stub-model", "reasoning": False}]}}}))
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG") if k in os.environ}
        env.update(PI_CODING_AGENT_DIR=str(HOME), PI_OFFLINE="1", PI_SKIP_VERSION_CHECK="1",
                   PI_TELEMETRY="0", STUB_KEY="stub-key")
        t0 = time.monotonic()
        self.proc = await asyncio.create_subprocess_exec(
            PI, "--mode", "rpc", "--session-dir", str(SESS), "--session-id", self.sid,
            "--model", "stub/stub-model", *FLAGS, cwd=str(WORKDIR), env=env, limit=64 << 20,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        self.reader = asyncio.create_task(self._read())
        self.err = asyncio.create_task(self._err())
        await self.cmd({"type": "set_steering_mode", "mode": "all"})
        return time.monotonic() - t0

    async def _read(self) -> None:
        while line := await self.proc.stdout.readline():
            rec = json.loads(line)
            self.records.append(rec)
            if rec.get("type") == "response" and rec.get("id") in self.pending:
                self.pending.pop(rec["id"]).set_result(rec)
            if rec.get("type") == "agent_settled":
                self.settled.set()

    async def _err(self) -> None:
        self.stderr = await self.proc.stderr.read()

    async def cmd(self, c: dict) -> dict:
        self.n += 1
        rid = f"g{self.n}"
        fut = asyncio.get_running_loop().create_future()
        self.pending[rid] = fut
        self.proc.stdin.write((json.dumps({**c, "id": rid}) + "\n").encode())
        await self.proc.stdin.drain()
        return await asyncio.wait_for(fut, 60)

    async def prompt(self, text: str, wait: bool = True) -> dict:
        self.settled.clear()
        r = await self.cmd({"type": "prompt", "message": text, "streamingBehavior": "steer"})
        if wait:
            await asyncio.wait_for(self.settled.wait(), 60)
        return r

    async def close(self) -> int:
        self.proc.stdin.close()
        rc = await asyncio.wait_for(self.proc.wait(), 10)
        await self.reader
        await self.err
        return rc

    def last_assistant(self) -> dict:
        ends = [r["message"] for r in self.records if r.get("type") == "message_end"
                and r["message"].get("role") == "assistant"]
        return ends[-1] if ends else {}


def requests(log: Path) -> list[dict]:
    return [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []


async def phase_a(port: int, log: Path) -> dict:
    WORKDIR.mkdir(parents=True, exist_ok=True)
    (WORKDIR / "AGENTS.md").write_text("You are probe-0001, a resume-gate agent.\n")
    res: dict = {"user": sh(["id", "-un"]), "pi": sh([PI, "--version"]), "node": sh(["node", "--version"]),
                 "nproc": os.cpu_count()}
    mem0 = cgroup_mem()
    p = Pi(os.environ["SID_MAIN"], port)
    res["startup_s"] = round(await p.start(), 3)
    await asyncio.sleep(1)
    res["idle_vmrss_kb"] = vmrss_kb(p.proc.pid)
    res["idle_cgroup_delta_bytes"] = (cgroup_mem() or 0) - (mem0 or 0) if mem0 else None
    await p.prompt("The codeword is PELICAN-7. Remember it.")
    await p.prompt("bash: echo PELICAN-7 > notes.txt")
    res["main_stats"] = (await p.cmd({"type": "get_session_stats"}))["data"]
    res["main_rc"] = await p.close()
    res["main_stderr"] = p.stderr.decode(errors="replace")[-500:]
    c = Pi(os.environ["SID_CUT"], port)
    await c.start()
    await c.prompt("bash: sleep 60; echo late", wait=False)
    for _ in range(100):  # wait for the tool to start, then cut the process off
        if any(r.get("type") == "tool_execution_start" for r in c.records):
            break
        await asyncio.sleep(0.05)
    await asyncio.sleep(0.5)
    kids = descendants(c.proc.pid)
    os.kill(c.proc.pid, signal.SIGKILL)
    res["cut_rc"] = await asyncio.wait_for(c.proc.wait(), 10)
    await asyncio.sleep(0.3)
    # the tool's bash runs in its own session and outlives a SIGKILLed Pi (the adapter kills it)
    res["cut_orphans"] = [pid for pid in kids if Path(f"/proc/{pid}").exists()]
    for pid in res["cut_orphans"]:
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)
    res["session_files"] = sorted(str(x.relative_to(HOME)) for x in SESS.glob("*.jsonl"))
    res["requests"] = len(requests(log))
    return res


async def phase_b(port: int, log: Path) -> dict:
    a = json.loads((OUT / "A.json").read_text())
    before = sorted(str(x.relative_to(HOME)) for x in SESS.glob("*.jsonl"))
    res: dict = {"user": sh(["id", "-un"]), "restored_session_files": before,
                 "home_files_before": sorted(str(x.relative_to(HOME)) for x in HOME.rglob("*") if x.is_file())}
    p = Pi(os.environ["SID_MAIN"], port)
    res["startup_s"] = round(await p.start(), 3)
    state = (await p.cmd({"type": "get_state"}))["data"]
    res["session_file"] = state.get("sessionFile")
    await p.prompt("What is the codeword? Answer from memory.")
    res["main_stats"] = (await p.cmd({"type": "get_session_stats"}))["data"]
    await p.close()
    res["main_stderr"] = p.stderr.decode(errors="replace")[-500:]
    reqs = requests(log)
    hist = json.dumps(reqs[-1]["body"]["messages"]) if reqs else ""
    c = Pi(os.environ["SID_CUT"], port)
    await c.start()
    await c.prompt("Carry on.")
    cut_hist = json.dumps(requests(log)[-1]["body"]["messages"])
    await c.close()
    res["cut_last_assistant"] = c.last_assistant()
    after = sorted(str(x.relative_to(HOME)) for x in SESS.glob("*.jsonl"))
    old, new = a["main_stats"]["tokens"], res["main_stats"]["tokens"]
    v = {
        "history_has_codeword": "PELICAN-7" in hist and "notes.txt" in hist,
        "history_has_tool_result": '"role": "tool"' in hist,
        "same_session_file": bool(res["session_file"]) and res["session_file"].endswith(
            next((f for f in before if os.environ["SID_MAIN"] in f), "?")),
        "no_new_session_files": after == before,
        "no_stderr_warning": "No project session found" not in res["main_stderr"] + c.stderr.decode(errors="replace"),
        "stats_continue": new["total"] > old["total"] and res["main_stats"]["userMessages"] == a["main_stats"]["userMessages"] + 1,
        "cut_session_resumed": "sleep 60" in cut_hist and c.last_assistant().get("stopReason") == "stop",
        "notes_txt": (WORKDIR / "notes.txt").read_text().strip() if (WORKDIR / "notes.txt").exists() else None,
    }
    v["pass"] = all(v[k] for k in v if k != "notes_txt") and v["notes_txt"] == "PELICAN-7"
    res["verdict"] = v
    return res


async def main() -> None:
    phase = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f"stub-{phase}.jsonl"
    srv = stub_openai.serve(0, str(log))
    res = await (phase_a if phase == "A" else phase_b)(srv.server_port, log)
    (OUT / f"{phase}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res.get("verdict", res)))
    srv.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
