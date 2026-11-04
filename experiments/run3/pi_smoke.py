"""Local smoke of run3's agent-pi type: `flock run` (compute local) with the chatbot3 world, the
fake OpenAI server scripted so each Pi agent calls flock-tool write_file, run_bot, add_probe and
post from its bash tool, then ends the turn in plain text; the fake msgboard as the board.

    PI_NODE_BIN=<node 22 bin dir> PI_PREFIX_BIN=<dir with pi> \
      uv run python experiments/run3/pi_smoke.py <repo> <outdir> [n_agents] [seconds]

Each agent's first prompt is the real agent-pi first prompt plus the script; every round
(PI_SMOKE_ROUNDS, default 20, back to back) runs best_status and run_bot through flock-tool.
Prints a summary (also <outdir>/summary.json): the world log's actions by agent, the board's
messages by sender, the tools Pi offered the model, the reasoning_effort it sent, the Pi
processes' VmRSS and the worker's whole process tree RSS (worker + Pi + tool calls), sampled
every second.

2026-10-25, Pi 1.2.0 on Node 22.23.3: 2 agents (3 rounds): write_file + add_probe (verified)
by each agent in the world log, one post each on the board under its own name, run_bot's turns
in the tool result, reasoning_effort high, the offered tools Pi's own (bash, read, edit, write).
5 agents x 15 rounds, 150 s: 85 turns, all completed; one Pi process per agent for the whole
run, VmRSS max 88 MB; worker tree max 786 MB (about 150 MB per agent with tool calls running).
"""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO, OUT = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
N = int(sys.argv[3]) if len(sys.argv) > 3 else 2
SECS = float(sys.argv[4]) if len(sys.argv) > 4 else 150
sys.path.insert(0, str(REPO))
import yaml

from flock.models import load_spec_yaml
from tests import fake_msgboard, fake_openai

if OUT.exists():
    shutil.rmtree(OUT)
src = OUT / "src"
exp = src / "experiments" / "run3"
shutil.copytree(REPO / "experiments" / "run3", exp,
                ignore=shutil.ignore_patterns("*.local.yaml*", "__py*", "ws"))
for w in ("chatbot3", "chatbot2", "chatbot", "eliza"):
    shutil.copytree(REPO / "worlds" / w, src / "worlds" / w, ignore=shutil.ignore_patterns("ws", "__py*"))

srv = fake_openai.serve(0, str(OUT / "requests.jsonl"))
board, bsrv, burl = fake_msgboard.serve(fake_msgboard.Board(channels={12: "run3-pi-smoke"}))
model_url = f"http://127.0.0.1:{srv.server_port}/v1"

ME = 'ME=$(basename "$PWD"); '
BOT = 'import sys\\nfor line in sys.stdin:\\n    print(\\"hello there\\", flush=True)\\n'
steps = [
    ["bash", {"command": ME + "flock-tool write_file --json - <<EOF\n"
              '{"path": "agents/$ME/bot.py", "content": "' + BOT + '"}\nEOF'}],
    ["bash", {"command": ME + "flock-tool run_bot --json - <<EOF\n"
              '{"path": "agents/$ME/bot.py", "lines": ["hi", "how are you?"]}\nEOF'}],
    ["bash", {"command": ME + "flock-tool add_probe --json - <<EOF\n"
              '{"lines": ["hi", "how are you?"], "check": "not_repeat", '
              '"evidence": "hello there", "path": "agents/$ME/bot.py"}\nEOF'}],
    ["bash", {"command": ME + "flock-tool post --json - <<EOF\n"
              '{"text": "[pi-smoke] $ME here: my bot repeats itself, anyone on greetings?", '
              '"thread": "new"}\nEOF'}],
]
SCRIPT = "script: " + json.dumps(steps)
ROUNDS = int(os.environ.get("PI_SMOKE_ROUNDS", "20"))
ROUND = "script: " + json.dumps([  # every round: two world calls, then a plain-text end
    ["bash", {"command": "flock-tool best_status --json '{}'"}],
    ["bash", {"command": ME + "flock-tool run_bot --json - <<EOF\n"
              '{"path": "agents/$ME/bot.py", "lines": ["hi", "what is your name?"]}\nEOF'}]])

# the rendered smoke variant (the real prompts), then the fakes
env0 = {**os.environ, "OPENAI_BASE_URL": model_url, "MODEL_WORKER": "fake-model",
        "CHATBOT_BOARD_URL": burl, "CHATBOT_CHANNEL": "run3-pi-smoke", "CHATBOT_RUN": "pismoke"}
subprocess.run([sys.executable, str(exp / "render.py"), "--variant", "smoke", "--out",
                str(exp / "spec.pi.yaml")], env=env0, check=True)
d = load_spec_yaml((exp / "spec.pi.yaml").read_text())
d["comms"]["invite_env"] = "FAKE_INVITE"
for name, t in d["types"].items():
    if name in ("agent", "agent-think"):
        t["replicas"] = 0
    if name == "agent-pi":
        t["replicas"] = N
        lp = t["loop"]
        lp.pop("prompt_file", None)
        # the real Pi first prompt, then the fake model's script
        t["loop"] = {**lp, "first": lp["first"] + "\n" + SCRIPT, "prompt": "Round {round}. " + ROUND,
                     "every": "turn_end", "stagger": "1s",
                     "while": {"until": "10m", "max_rounds": ROUNDS}, "final": {"prompt": "bye"}}
(exp / "spec.pi.yaml").write_text(yaml.safe_dump(d, sort_keys=False, width=100))

node = os.environ["PI_NODE_BIN"]  # Node 22 + pi
env = {k: v for k, v in os.environ.items() if not k.startswith("FLOCK_")}
env.update(HOME=str(OUT / "home"), PATH=f"{node}:{os.environ['PI_PREFIX_BIN']}:{Path(sys.executable).parent}:{env['PATH']}",
           FLOCK_PI_CLI=str(Path(os.environ["PI_PREFIX_BIN"]) / "pi"), OPENAI_API_KEY="fake-key",
           FAKE_INVITE=board.invite, CHATBOT_WORLD_TOKEN="w0rld-t0ken",
           CHATBOT_WS=str(OUT / "ws"))
(OUT / "home").mkdir()
out = open(OUT / "flock-run.out", "w")  # noqa: SIM115 (the process holds it)
t0 = time.time()
proc = subprocess.Popen([sys.executable, "-m", "flock.cli", "run", "spec.pi.yaml", "--until",
                         f"{int(SECS)}s", "--skip-preflight"], cwd=exp, env=env, stdout=out,
                        stderr=subprocess.STDOUT, start_new_session=True)
rss: dict[str, list[int]] = {}  # Pi pid -> VmRSS samples (kB)
tree: list[int] = []  # the worker's whole process tree (worker + Pi + tool calls), kB


def _procs() -> dict[int, tuple[int, list[bytes], int]]:
    """pid -> (ppid, argv, VmRSS kB) for every process we can read."""
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            st = (p / "stat").read_text()
            ppid = int(st[st.rfind(")") + 2:].split()[1])
            argv = (p / "cmdline").read_bytes().split(b"\0")
            kb = next((int(x.split()[1]) for x in (p / "status").read_text().splitlines()
                       if x.startswith("VmRSS:")), 0)
            out[int(p.name)] = (ppid, argv, kb)
        except (OSError, ValueError):
            pass
    return out


def sample_rss() -> None:
    ps = _procs()
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _, _) in ps.items():
        kids.setdefault(ppid, []).append(pid)
    roots = [pid for pid, (_, argv, _) in ps.items() if b"flock.worker.main" in argv
             and any(str(OUT).encode() in a for a in argv) or _under(pid, ps)]
    total, todo, seen = 0, list(set(roots)), set()
    while todo:
        pid = todo.pop()
        if pid in seen or pid not in ps:
            continue
        seen.add(pid)
        _, argv, kb = ps[pid]
        total += kb
        if b"--mode" in argv and b"rpc" in argv:
            rss.setdefault(str(pid), []).append(kb)
        todo += kids.get(pid, [])
    if total:
        tree.append(total)


def _under(pid: int, ps: dict) -> bool:
    """A flock worker started by our `flock run` (its parent chain reaches proc.pid)."""
    if b"flock.worker.main" not in ps[pid][1]:
        return False
    while pid in ps and pid > 1:
        if pid == proc.pid:
            return True
        pid = ps[pid][0]
    return False


while proc.poll() is None and time.time() - t0 < SECS + 60:
    sample_rss()
    time.sleep(1)
if proc.poll() is None:
    proc.terminate()
    proc.wait(30)
out.close()

log = [json.loads(x) for x in (OUT / "ws" / "log.jsonl").read_text().splitlines()] \
    if (OUT / "ws" / "log.jsonl").is_file() else []
reqs = [json.loads(x) for x in (OUT / "requests.jsonl").read_text().splitlines()] if (OUT / "requests.jsonl").is_file() else []
summary = {
    "exit": proc.returncode,
    "world_log": [(x.get("agent"), x.get("action")) for x in log],
    "board": [(m["sender"].get("external_id") or m["sender"].get("name"), m["body"][:90])
              for m in board.msgs],
    "model_requests": len(reqs),
    "tools_offered_to_pi": sorted({t["function"]["name"] for r in reqs
                                   for t in r["body"].get("tools") or []}),
    "reasoning_effort": sorted({str(r["body"].get("reasoning_effort")) for r in reqs}),
    "pi_processes_seen": len(rss),
    "pi_rss_mb_max": round(max((max(v) for v in rss.values()), default=0) / 1024, 1),
    "pi_rss_mb_median_of_last": round(sorted(v[-1] for v in rss.values())[len(rss) // 2] / 1024, 1)
    if rss else None,
    "worker_tree_rss_mb_max": round(max(tree, default=0) / 1024, 1),
}
(OUT / "summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
srv.shutdown()
bsrv.shutdown()
