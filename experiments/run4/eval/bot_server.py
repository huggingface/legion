"""Serve a chatbot-swarm checkout as an OpenAI-compatible chat endpoint for evals.

    python experiments/run4/eval/bot_server.py <clone of cmpatino/chatbot-swarm> --port 8000

Loading the bot takes seconds (hundreds of modules), answering takes milliseconds. So each of
--hosts sandboxed processes loads the bot once and forks a child per request, and every
conversation starts from the same just-loaded state. The same lines get the same replies, on
any host and in any order: the hash seed, the global `random` state (restored in each child),
the seeds of the bot's own random.Random() instances and its object ids (no address
randomisation, no request data in the parent) are all fixed. Replies that read the clock can
still change. POST /v1/chat/completions feeds the request's user
messages to the bot in order (system and assistant messages are ignored: the bot only reads
user lines) and returns its last reply. The hosts run under flock.isolation, like the world's
contract runner (worlds/chatbot4/sandbox.py), with the same rlimits and audit hook.
"""
from __future__ import annotations

import argparse
import json
import queue
import select
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from flock.isolation import command as isolate_command

# The world's SHIM, plus a loop that forks one child per conversation. Only that loop may fork
# (the `trusted` flag); children lose it before running any bot code.
HOST = r'''
import io, json, os, random, resource, runpy, sys, time
root, seed, timeout = os.path.realpath(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3])
resource.setrlimit(resource.RLIMIT_AS, (512 << 20, 512 << 20))
resource.setrlimit(resource.RLIMIT_FSIZE, (1 << 20, 1 << 20))
resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
sys.dont_write_bytecode = True
sys.path.insert(0, root)
requests = os.fdopen(os.dup(0), "r")
reply_fd = os.dup(1)
os.dup2(2, 1)  # stray prints from the bot go to the log, not the protocol
allowed = set(sys.stdlib_module_names) | {'bot'}
blocked = {'ctypes', '_ctypes', '_posixsubprocess', 'subprocess', 'socket', '_socket', 'multiprocessing'}
trusted = [False]
def audit(event, args):
    if trusted[0] and event == 'os.fork':
        return
    if event == 'import':
        name = args[0].split('.')[0]
        if name not in allowed or name in blocked:
            raise PermissionError('only standard-library bot imports are allowed')
    if event == 'open':
        path, mode, flags = args
        if not isinstance(path, (str, bytes, os.PathLike)):
            raise PermissionError('file access is disabled')
        path = os.path.realpath(os.fsdecode(path))
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise PermissionError('file writes are disabled')
        roots = (root + os.sep, os.path.realpath(sys.base_prefix) + os.sep)
        if not path.startswith(roots) or not path.endswith(('.py', '.pyc', '.so')):
            raise PermissionError('file access is disabled except module loading')
    if event.startswith(('socket.', 'subprocess.', 'ctypes.', 'os.kill', 'signal.')) or event in {
        'os.system', 'os.exec', 'os.fork', 'os.forkpty', 'os.posix_spawn', 'os.kill', 'os.killpg',
        'os.remove', 'os.rename', 'os.mkdir', 'os.rmdir', 'os.chmod', 'os.link',
        'os.symlink', 'os.truncate', 'os.putenv', 'os.unsetenv'}:
        raise PermissionError('bot side effects are disabled')
import signal, _signal
for module in (os, sys.modules.get('posix'), signal, _signal):
    for name in ('kill', 'killpg', 'pthread_kill', 'pidfd_send_signal'):
        if module is not None and hasattr(module, name):
            delattr(module, name)
sys.addaudithook(audit)


def send(obj):
    data = (json.dumps(obj) + "\n").encode()
    while data:
        data = data[os.write(reply_fd, data):]


def converse():
    """A child: read one request, hold the conversation, send the result."""
    raw = requests.readline()
    if not raw:
        os._exit(EOF)
    random.setstate(loaded)  # random reseeds itself from the OS in every forked child
    lines = json.loads(raw)
    started = time.monotonic()
    limit = int(timeout) + 1
    resource.setrlimit(resource.RLIMIT_CPU, (limit, limit + 1))
    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
    sys.stdin = io.StringIO("".join(line + "\n" for line in lines))
    sys.stdout = buffer = io.StringIO()
    status = "ok"
    try:
        runpy.run_module("bot", run_name="__main__")
    except BaseException as exc:  # the fixed __main__ catches reply errors; this is the rest
        status = f"error {type(exc).__name__}"
    send({"replies": buffer.getvalue().splitlines(), "status": status,
          "seconds": round(time.monotonic() - started, 3)})
    os._exit(0)


# Modules make their own random.Random(), which seeds itself from the OS: derive those seeds
# from the run's seed and a counter instead, so every host is in the same state.
os_seeded, created = random.Random.seed, iter(range(1 << 62))
def derived_seed(self, a=None, version=2):
    os_seeded(self, f"{seed}:{next(created)}" if a is None else a, version)
random.Random.seed = derived_seed
random.seed = random._inst.seed
random.seed(seed)
started = time.monotonic()
import bot.runtime
loaded = random.getstate()
send({"ready": True, "load_s": round(time.monotonic() - started, 2),
      "handlers": len(bot.runtime._HANDLERS)})
# Fork first, then the child reads its request: the parent never touches a request, so every
# child starts from the same heap and a seed made from id(memory) repeats too. The server
# sends one request at a time and enforces the wall-clock timeout (by replacing this host).
EOF = 77
while True:
    trusted[0] = True
    pid = os.fork()
    if pid == 0:
        trusted[0] = False
        converse()
    trusted[0] = False
    code = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
    if code == EOF:
        break
    if code:
        send({"replies": [], "status": f"exit {code}"})
'''


class Host:
    """One sandboxed process with the bot loaded; ask() runs one conversation in a fork."""

    def __init__(self, tree: Path, seed: int, timeout: float, log):
        self.timeout = timeout
        self.tmp = Path(tempfile.mkdtemp(prefix="bot-host-"))
        argv = isolate_command([sys.executable, "-s", "-S", "-P", "-B", "-c", HOST, str(tree),
                                str(seed), str(timeout)], self.tmp / "root", cwd=tree,
                               readonly=[tree], python_only=True)
        # setarch -R: the same addresses in every host, so id()-based choices agree across
        # hosts. No -I: PYTHONHASHSEED must apply. The environment holds nothing else.
        self.proc = subprocess.Popen(["setarch", "-R", *argv], cwd=self.tmp,
                                     env={"PYTHONHASHSEED": str(seed)}, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1,
                                     start_new_session=True)
        self.ready = self._read(None)
        if not self.ready.get("ready"):
            raise RuntimeError(f"bot host did not start: {self.ready}")

    def _read(self, timeout):
        if timeout is not None and not select.select([self.proc.stdout], [], [], timeout)[0]:
            raise TimeoutError
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError(f"bot host exited ({self.proc.poll()})")
        return json.loads(line)

    def ask(self, lines: list[str]) -> dict:
        self.proc.stdin.write(json.dumps(lines) + "\n")
        self.proc.stdin.flush()
        return self._read(self.timeout + 5)

    def close(self):
        self.proc.kill()
        self.proc.wait()
        shutil.rmtree(self.tmp, ignore_errors=True)


class Pool:
    def __init__(self, n: int, start):
        self.start, self.idle = start, queue.Queue()
        threads = [threading.Thread(target=lambda: self.idle.put(start())) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        if self.idle.qsize() != n:
            raise RuntimeError("some bot hosts failed to start; see the log")
        self.hosts = list(self.idle.queue)

    def _replace(self, host):
        host.close()
        self.hosts.remove(host)
        host = self.start()
        self.hosts.append(host)
        return host

    def ask(self, lines):
        host = self.idle.get()
        try:
            return host.ask(lines)
        except TimeoutError:  # the conversation is stuck in its child: replace the host
            host = self._replace(host)
            return {"replies": [], "status": "timeout"}
        except (OSError, RuntimeError, ValueError):  # a dead host: replace it, retry once
            host = self._replace(host)
            return host.ask(lines)
        finally:
            self.idle.put(host)


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


def handler(pool: Pool, model: str):
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
                return self._send(200, {"ok": True, "model": model, "hosts": len(pool.hosts)})
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
            result = pool.ask(lines)
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
    parser.add_argument("tree", type=Path, help="a checkout of cmpatino/chatbot-swarm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hosts", type=int, default=4, help="sandboxed bot processes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300, help="seconds per conversation")
    parser.add_argument("--log", type=Path, default=Path("bot_server.log"),
                        help="the bot hosts' stderr")
    args = parser.parse_args()
    tree = args.tree.resolve()
    sha = subprocess.run(["git", "-C", str(tree), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    model = f"chatbot-swarm@{sha[:12]}"
    with open(args.log, "a") as log:
        started = time.monotonic()
        pool = Pool(args.hosts, lambda: Host(tree, args.seed, args.timeout, log))
        ready = pool.hosts[0].ready
        print(f"{model}: {args.hosts} hosts ready in {time.monotonic() - started:.1f}s "
              f"({ready['handlers']} handlers, {ready['load_s']}s load); "
              f"http://{args.host}:{args.port}/v1", flush=True)
        server = ThreadingHTTPServer((args.host, args.port), handler(pool, model))
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            for host in pool.hosts:
                host.close()


if __name__ == "__main__":
    main()
