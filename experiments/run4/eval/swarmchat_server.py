"""Serve a checkout of a swarm's chatbot as SwarmChat, an OpenAI-compatible endpoint for evals.

    python experiments/run4/eval/swarmchat_server.py <chatbot checkout> --port 8000

Loading the bot takes seconds (hundreds of modules), answering takes milliseconds. So one
process loads the bot and forks a worker per conversation, and every conversation starts from
the same just-loaded state. The same lines get the same replies whatever the order or the
concurrency of the requests, in this run or the next: the hash seed, the global `random` state
(restored in each worker) and the seeds of the bot's own random.Random() instances are fixed.
Replies that read the clock or seed from an object id (a few modules use id(memory)) can still
change: ids move with address randomisation, which HF Jobs do not let us turn off.

POST /v1/chat/completions feeds the request's user messages to the bot in order (system and
assistant messages are ignored: the bot only reads user lines) and returns its last reply.
There is no sandbox: the bot runs as the server's user, with an empty environment, so run it
somewhere disposable such as an HF Job.
"""

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOADER = r'''
import io, json, os, random, resource, runpy, signal, socket, sys, time
root, seed, timeout, path, workers = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], int(sys.argv[5])
sys.path.insert(0, root)
status = os.fdopen(os.dup(1), "w")
os.dup2(2, 1)  # stray prints from the bot go to the log
listener = socket.socket(socket.AF_UNIX)
listener.bind(path)
listener.listen(256)
# A worker writes a byte here when it takes a conversation; the loader forks one per byte.
spawn_r, spawn_w = os.pipe()
os.write(spawn_w, bytes(workers))


def converse():
    """A worker: take one connection, hold that conversation, send the result and exit."""
    conn = listener.accept()[0]
    os.write(spawn_w, b"\0")
    random.setstate(loaded)  # random reseeds itself from the OS in every forked child
    started = time.monotonic()
    lines = json.loads(conn.makefile("rb").readline())
    limit = int(timeout) + 1
    signal.alarm(limit)
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    sys.stdin = io.StringIO("".join(line + "\n" for line in lines))
    sys.stdout = buffer = io.StringIO()
    result = "ok"
    try:
        runpy.run_module("bot", run_name="__main__")
    except BaseException as exc:  # the fixed __main__ catches reply errors; this is the rest
        result = f"error {type(exc).__name__}"
    conn.sendall((json.dumps({"replies": buffer.getvalue().splitlines(), "status": result,
                              "seconds": round(time.monotonic() - started, 3)}) + "\n").encode())
    os._exit(0)


# Modules make their own random.Random(), which seeds itself from the OS: derive those seeds
# from the run's seed and a counter instead, so every run starts in the same state.
os_seeded, created = random.Random.seed, iter(range(1 << 62))
def derived_seed(self, a=None, version=2):
    os_seeded(self, f"{seed}:{next(created)}" if a is None else a, version)
random.Random.seed = derived_seed
random.seed = random._inst.seed
random.seed(seed)
started = time.monotonic()
import bot.runtime
loaded = random.getstate()
print(json.dumps({"load_s": round(time.monotonic() - started, 2),
                  "handlers": len(bot.runtime._HANDLERS)}), file=status, flush=True)
# Keep `workers` forks waiting for a connection. The workers read the requests, never the
# loader, and the kernel reaps them.
signal.signal(signal.SIGCHLD, signal.SIG_IGN)
while True:
    os.read(spawn_r, 1)
    if os.fork() == 0:
        converse()
'''


class Bot:
    """The loader process; ask() holds one conversation in a fresh worker."""

    def __init__(self, tree: Path, *, seed: int, timeout: float, workers: int, log):
        self.timeout = timeout
        self.tmp = tempfile.TemporaryDirectory(prefix="swarmchat-")
        self.path = os.path.join(self.tmp.name, "bot.sock")
        # No -I: PYTHONHASHSEED must apply. The environment holds nothing else.
        self.proc = subprocess.Popen(
            [sys.executable, "-s", "-S", "-P", "-c", LOADER, str(tree), str(seed), str(timeout),
             self.path, str(workers)],
            cwd=self.tmp.name, env={"PYTHONHASHSEED": str(seed)}, stdout=subprocess.PIPE,
            stderr=log, text=True, start_new_session=True)
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"the bot did not load (exit {self.proc.wait()}); see the log")
        self.ready = json.loads(line)

    def ask(self, lines: list[str]) -> dict:
        with socket.socket(socket.AF_UNIX) as conn:
            conn.settimeout(self.timeout + 5)
            conn.connect(self.path)
            conn.sendall((json.dumps(lines) + "\n").encode())
            try:
                raw = conn.makefile("rb").readline()
            except TimeoutError:
                return {"replies": [], "status": "timeout"}
        # The worker died without replying, e.g. on its CPU or wall-clock limit.
        return json.loads(raw) if raw else {"replies": [], "status": "worker exited"}

    def alive(self) -> bool:
        return self.proc.poll() is None

    def close(self):
        if self.alive():
            os.killpg(self.proc.pid, signal.SIGKILL)
        self.proc.wait()
        self.proc.stdout.close()
        self.tmp.cleanup()


def user_lines(messages) -> list[str]:
    """User message texts as single bot input lines (the bot reads one line per turn)."""
    lines = []
    for m in messages:
        if m.get("role") != "user":
            continue
        content = m.get("content") or ""
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if p.get("type") == "text")
        lines.append(" ".join(str(content).split()))
    return lines


def handler(bot: Bot, model: str):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                return self._send(200, {"object": "list", "data": [
                    {"id": model, "object": "model", "owned_by": "chatbot-swarm"}]})
            if self.path == "/health":
                return self._send(200 if bot.alive() else 503,
                                  {"ok": bot.alive(), "model": model, **bot.ready})
            self._send(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._send(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                lines = user_lines(body.get("messages", []))
            except (ValueError, AttributeError, TypeError):
                return self._send(400, {"error": {"message": "invalid JSON body"}})
            if not lines:
                return self._send(400, {"error": {"message": "no user message"}})
            if body.get("stream"):
                return self._send(400, {"error": {"message": "streaming is not supported"}})
            result = bot.ask(lines)
            replies = result.get("replies", [])
            reply = replies[-1] if len(replies) == len(lines) else ""
            self._send(200, {
                "id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                "created": int(time.time()), "model": model,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": reply}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "bot": result})

        def log_message(self, fmt, *args):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tree", type=Path, help="a checkout of a swarm's chatbot")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="SwarmChat", help="the model id the endpoint serves")
    parser.add_argument("--workers", type=int, default=4,
                        help="forks kept ready for new conversations; the client caps how many "
                             "run at once")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=60, help="seconds per conversation")
    parser.add_argument("--log", type=Path, default=Path("swarmchat_server.log"),
                        help="the bot's stderr")
    args = parser.parse_args()
    with open(args.log, "a") as log:
        bot = Bot(args.tree.resolve(), seed=args.seed, timeout=args.timeout,
                  workers=args.workers, log=log)
        print(f"{args.model}: loaded in {bot.ready['load_s']}s ({bot.ready['handlers']} "
              f"handlers), {args.workers} workers; http://{args.host}:{args.port}/v1", flush=True)
        server = ThreadingHTTPServer((args.host, args.port), handler(bot, args.model))
        # hf_job.py stops the server with SIGTERM: leave through `finally`, which stops the bot
        signal.signal(signal.SIGTERM, lambda *_: sys.exit())
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            bot.close()


if __name__ == "__main__":
    main()
