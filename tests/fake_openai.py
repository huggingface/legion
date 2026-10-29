"""A tiny OpenAI-compatible chat-completions server for harness tests (no real model calls).

Run: python fake_openai.py <port|0> <log.jsonl>   (prints "PORT <n>" once listening)

Replies, decided from the last message of each request:
- last message is a tool result     -> a final text "done: <first 40 chars of the result>"
- last user text contains `bash: X` -> one `bash` tool call with command X
- last user text contains `call: NAME {json}` and NAME is an offered tool -> one call of NAME
  with those arguments (also under a forcing `tool_choice`, until a tool result follows)
- the latest user message contains `script: [[NAME, {args}], ...]` (JSON on that line) -> the
  k-th of those calls, k = tool results since that message (also under a forcing `tool_choice`);
  `{me}` in it is the agent's name from the system message ("You are <name>, ..."); after the
  last one `done {summary: "script done"}` when `done` is offered, else a text reply
- last user text contains `astext: NAME {json}` -> that call written as text:
  `{"name": NAME, "arguments": {json}}`
- last user text contains `json: X` -> the call written as text: `{"tool": "bash", "command": X}`
- last user text contains `py: X`   -> the call written as text, pythonic: `[bash(command=X)]`
- `tool_choice` forces a call (`required` or a named function) -> a `bash` call every time,
  command from `bash: X` in the last user text, else `echo forced`; with a `done` tool offered
  and `done after N` in that text, `done {summary: "painted N"}` once N tool results followed it
- `response_format` set             -> its JSON as text: `{"tool": "bash", "command": "echo rf"}`
- last user text contains `think!`  -> inline reasoning without the opening tag (LFM2 style):
                                       "pondering</think>echo: <text>"
- last user text contains `think2!` -> `reasoning_content: "pondering"` + the echo
- last user text contains `fail!`   -> HTTP 400 with an OpenAI-style error body (long, and
                                       echoing the Authorization header)
- `realtok!` in any user message: prompt_tokens = characters of the messages / 2 (else 100 +
  one per message); `ctxlimit=N` there: HTTP 400 "maximum context length is N tokens" (vLLM's
  wording) when prompt_tokens > N
- forced `tool_choice` with `done` offered: `done` once the last tool result contains DONE-NOW
- last user text contains `503!`    -> HTTP 503 every time
- last user text contains `429!`    -> HTTP 429 (Retry-After: 0) the first time a request with
                                       these exact messages is seen, then the normal reply
- last user text contains `empty!`  -> an empty assistant message (no content, no tool call),
                                       also under a forcing `tool_choice`; `empty-always!` in
                                       any user message: every reply is empty
- a `script:` step named EMPTY      -> an empty assistant message at that point
- last user text is the llm harness's NUDGE ("Reply with exactly one tool call.") and `done`
  is offered                        -> `done {summary: "nudged"}`
- otherwise                         -> text "echo: <last user text>"
Streams SSE when the request asks for `stream: true` (Pi does), with a usage chunk at the end:
prompt_tokens 100 (+1 per message), 40 of them cached, completion_tokens 7.
Every request is appended to the log as {"auth": <Authorization header>, "body": <request>}.
"""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

NUDGE = "Reply with exactly one tool call."  # flock.worker.adapters.llm.NUDGE (kept import-free)

LOCK = threading.Lock()
SEEN_429: set[str] = set()


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(str(c.get("text", "")) for c in content if isinstance(c, dict))
    return ""


def _named_call(body: dict[str, Any], user: str) -> dict[str, Any] | None:
    """`call: NAME {json args}` -> a native call of NAME, when the request offers that tool."""
    if "call: " not in user:
        return None
    name, _, args = user.split("call: ", 1)[1].splitlines()[0].strip().partition(" ")
    offered = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    if name not in offered:
        return None
    n = sum(1 for m in body.get("messages") or [] if m.get("role") == "assistant")
    return {"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {
        "name": name, "arguments": args.strip() or "{}"}}]}


def _scripted(body: dict[str, Any]) -> dict[str, Any] | None:
    """`script: [[NAME, {args}], ...]` in the latest user message: one call per model request."""
    msgs = body.get("messages") or []
    since = next((i for i in range(len(msgs) - 1, -1, -1) if msgs[i].get("role") == "user"), None)
    if since is None or "script: " not in (user := _text(msgs[since].get("content"))):
        return None
    system = _text(msgs[0].get("content")) if msgs[0].get("role") == "system" else ""
    me = m.group(1) if (m := re.search(r"You are ([^\s,]+),", system)) else "?"
    steps = json.loads(user.split("script: ", 1)[1].splitlines()[0].replace("{me}", me))
    k = sum(1 for m in msgs[since:] if m.get("role") == "tool")
    offered = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    if k < len(steps):
        name, args = steps[k]
        if name == "EMPTY":  # a dropped call: an empty assistant message
            return {"content": "", "tool_calls": None}
    elif "done" in offered:
        name, args = "done", {"summary": "script done"}
    else:
        return {"content": "script done", "tool_calls": None}
    n = sum(1 for m in msgs if m.get("role") == "assistant")
    return {"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {
        "name": name, "arguments": json.dumps(args)}}]}


def reply_for(body: dict[str, Any]) -> dict[str, Any]:
    """{'content': str|None, 'tool_calls': list|None} for this request."""
    msgs = body.get("messages") or []
    last = msgs[-1] if msgs else {}
    tc = body.get("tool_choice")
    offered = {t.get("function", {}).get("name") for t in body.get("tools") or []}
    users = [_text(m.get("content")) for m in msgs if m.get("role") == "user"]
    if any("empty-always!" in u for u in users) or (
            last.get("role") == "user" and "empty!" in _text(last.get("content"))):
        return {"content": "", "tool_calls": None}
    if last.get("role") == "user" and _text(last.get("content")) == NUDGE and "done" in offered:
        return {"content": None, "tool_calls": [{"id": f"call_nudged_{len(msgs)}", "type": "function",
                "function": {"name": "done", "arguments": json.dumps({"summary": "nudged"})}}]}
    if scripted := _scripted(body):
        return scripted
    if last.get("role") == "user" and (named := _named_call(body, _text(last.get("content")))):
        return named
    if last.get("role") == "user" and "astext: " in _text(last.get("content")):
        name, _, args = _text(last.get("content")).split("astext: ", 1)[1].strip().partition(" ")
        return {"content": json.dumps({"name": name, "arguments": json.loads(args or "{}")}),
                "tool_calls": None}
    if tc == "required" or isinstance(tc, dict):
        user = next((_text(m.get("content")) for m in reversed(msgs) if m.get("role") == "user"), "")
        cmd = user.split("bash: ", 1)[1].splitlines()[0].strip() if "bash: " in user else "echo forced"
        n = sum(1 for m in msgs if m.get("role") == "assistant")
        offered = {t.get("function", {}).get("name") for t in body.get("tools") or []}
        if "done" in offered and last.get("role") == "tool" and "DONE-NOW" in _text(last.get("content")):
            return {"content": None, "tool_calls": [{"id": f"call_{n}", "type": "function", "function": {
                "name": "done", "arguments": json.dumps({"summary": "finished"})}}]}
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


def _users(body: dict[str, Any]) -> str:
    return "\n".join(_text(m.get("content")) for m in body.get("messages") or [] if m.get("role") == "user")


def prompt_tokens(body: dict[str, Any]) -> int:
    """100 + one per message; with `realtok!` in a user message, a realistic count for JSON-ish
    text: characters of the messages / 2 (pixel run 3 measured ~2 chars/token)."""
    if "realtok!" in _users(body):
        return len(json.dumps(body.get("messages") or [])) // 2
    return 100 + len(body.get("messages") or [])


def usage_for(body: dict[str, Any]) -> dict[str, Any]:
    n = prompt_tokens(body)
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
            f.write(json.dumps({"auth": self.headers.get("authorization"), "body": body,
                                "session_id": self.headers.get("x-session-id")}) + "\n")
        last_text = _text((body.get("messages") or [{}])[-1].get("content"))
        if "fail!" in last_text:
            # echoes the caller's Authorization header, as a careless server might
            self._json(400, {"error": {"auth_seen": self.headers.get("authorization"),
                                       "message": "fake failure: bad request " + "x" * 600,
                                       "type": "invalid_request_error"}})
            return
        if m := re.search(r"ctxlimit=(\d+)", _users(body)):
            limit, n = int(m.group(1)), prompt_tokens(body)
            if n > limit:  # vLLM's wording
                self._json(400, {"object": "error", "type": "BadRequestError", "code": 400, "message": (
                    f"This model's maximum context length is {limit} tokens. However, your request has "
                    f"{n} input tokens. Please reduce the length of the input messages.")})
                return
        if "503!" in last_text:
            self._json(503, {"error": {"message": "engine overloaded"}})
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
