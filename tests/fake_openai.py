"""A tiny OpenAI-compatible chat-completions server for harness tests (no real model calls).

Run: python fake_openai.py <port|0> <log.jsonl>   (prints "PORT <n>" once listening)

Replies, decided from the last message of each request:
- last message is a tool result     -> a final text "done: <first 40 chars of the result>"
- last user text contains `bash: X` -> one `bash` tool call with command X
- last user text contains `json: X` -> the call written as text: `{"tool": "bash", "command": X}`
- last user text contains `py: X`   -> the call written as text, pythonic: `[bash(command=X)]`
- `tool_choice` forces a call (`required` or a named function) -> a `bash` call every time,
  command from `bash: X` in the last user text, else `echo forced`; with a `done` tool offered
  and `done after N` in that text, `done {summary: "painted N"}` once N tool results followed it
- `response_format` set             -> its JSON as text: `{"tool": "bash", "command": "echo rf"}`
- last user text contains `think!`  -> inline reasoning without the opening tag (LFM2 style):
                                       "pondering</think>echo: <text>"
- last user text contains `think2!` -> `reasoning_content: "pondering"` + the echo
- last user text contains `fail!`   -> HTTP 400 with an OpenAI-style error body
- last user text contains `429!`    -> HTTP 429 (Retry-After: 0) the first time a request with
                                       these exact messages is seen, then the normal reply
- otherwise                         -> text "echo: <last user text>"
Streams SSE when the request asks for `stream: true` (Pi does), with a usage chunk at the end:
prompt_tokens 100 (+1 per message), 40 of them cached, completion_tokens 7.
Every request is appended to the log as {"auth": <Authorization header>, "body": <request>}.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

LOCK = threading.Lock()
SEEN_429: set[str] = set()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


def reply_for(body: dict[str, Any]) -> dict[str, Any]:
    """{'content': str|None, 'tool_calls': list|None} for this request."""
    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else {}
    tc = body.get("tool_choice")
    if tc == "required" or isinstance(tc, dict):
        user = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
        cmd = user.split("bash: ", 1)[1].splitlines()[0].strip() if "bash: " in user else "echo forced"
        n = sum(1 for m in msgs if m.get("role") == "assistant")
        offered = {t.get("function", {}).get("name") for t in body.get("tools") or []}
        if "done" in offered and "done after " in user:
            want = int(user.split("done after ", 1)[1].split()[0])
            since = next(i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user")
            if sum(1 for m in msgs[since:] if m.get("role") == "tool") >= want:
                return {"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {
                    "name": "done", "arguments": json.dumps({"summary": f"painted {want}"})}}]}
        return {"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {
            "name": "bash", "arguments": json.dumps({"command": cmd})}}]}
    if body.get("response_format") and last.get("role") != "tool":
        return {"content": json.dumps({"tool": "bash", "command": "echo rf"}), "tool_calls": None}
    if last.get("role") == "tool":
        return {"content": "done: " + _text(last.get("content")).strip()[:40], "tool_calls": None}
    user = _text(last.get("content")) if last.get("role") == "user" else ""
    if "json: " in user:
        cmd = user.split("json: ", 1)[1].splitlines()[0].strip()
        return {"content": json.dumps({"tool": "bash", "command": cmd}), "tool_calls": None}
    if "py: " in user:
        cmd = user.split("py: ", 1)[1].splitlines()[0].strip()
        return {"content": f"[bash(command={cmd!r})]", "tool_calls": None}
    if "bash: " in user:
        cmd = user.split("bash: ", 1)[1].splitlines()[0].strip()
        n = sum(1 for m in msgs if m.get("role") == "assistant")
        call = {"id": f"call_{n}", "type": "function",
                "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}
        return {"content": None, "tool_calls": [call]}
    if "think2!" in user:
        return {"content": "echo: " + user.strip(), "tool_calls": None, "reasoning_content": "pondering"}
    if "think!" in user:
        return {"content": "pondering</think>\n\necho: " + user.strip(), "tool_calls": None}
    return {"content": "echo: " + user.strip(), "tool_calls": None}


def usage_for(body: dict[str, Any]) -> dict[str, Any]:
    n = 100 + len(body.get("messages") or [])
    return {"prompt_tokens": n, "completion_tokens": 7, "total_tokens": n + 7,
            "prompt_tokens_details": {"cached_tokens": 40}}


class Handler(BaseHTTPRequestHandler):
    log_path = ""

    def log_message(self, *a: Any) -> None:
        pass

    def do_GET(self) -> None:  # /v1/models, health
        self._json(200, {"object": "list", "data": [{"id": "fake-model", "object": "model"}]})

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}")
        with LOCK, open(self.log_path, "a") as f:
            f.write(json.dumps({"auth": self.headers.get("authorization"), "body": body}) + "\n")
        if "fail!" in _text((body.get("messages") or [{}])[-1].get("content")):
            self._json(400, {"error": {"message": "fake failure", "type": "invalid_request_error"}})
            return
        if "429!" in _text((body.get("messages") or [{}])[-1].get("content")):
            key = json.dumps(body.get("messages"), sort_keys=True)
            with LOCK:
                first = key not in SEEN_429
                SEEN_429.add(key)
            if first:
                self._json(429, {"error": {"message": "rate limited", "type": "rate_limit"}},
                           headers={"retry-after": "0"})
                return
        r = reply_for(body)
        model = body.get("model", "fake-model")
        finish = "tool_calls" if r["tool_calls"] else "stop"
        if not body.get("stream"):
            msg = {"role": "assistant", "content": r["content"]}
            if r.get("reasoning_content"):
                msg["reasoning_content"] = r["reasoning_content"]
            if r["tool_calls"]:
                msg["tool_calls"] = r["tool_calls"]
            self._json(200, {"id": "c1", "object": "chat.completion", "created": int(time.time()),
                             "model": model, "usage": usage_for(body),
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}]})
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        base = {"id": "c1", "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        delta: dict[str, Any] = {"role": "assistant"}
        if r["content"] is not None:
            delta["content"] = r["content"]
        if r["tool_calls"]:
            delta["tool_calls"] = [{**c, "index": i} for i, c in enumerate(r["tool_calls"])]
        chunks = [{**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                  {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]},
                  {**base, "choices": [], "usage": usage_for(body)}]
        for c in chunks:
            self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _json(self, code: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        data = json.dumps(obj).encode()
        self.send_response(code)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def serve(port: int, log_path: str) -> ThreadingHTTPServer:
    """Start in a daemon thread; returns the server (`server.server_port`)."""
    Handler.log_path = log_path
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    srv = serve(int(sys.argv[1]), sys.argv[2])
    print(f"PORT {srv.server_port}", flush=True)
    threading.Event().wait()
