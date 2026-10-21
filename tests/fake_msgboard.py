"""An in-memory msgboard for comms tests (no real board is ever called from tests).

Follows the real board's API (`GET /v1/guide`) for the routes flock uses:
- POST /v1/join {code, name}: the invite must match; joining again with the same name gives
  the same principal and a new token, and the old token stops working (401). Joining adds the
  principal to every channel (the invite is scoped to them).
- GET /v1/channels?name=, POST /v1/channels/{id}/members, PATCH /v1/principals/{id}
- POST /v1/channels/{id}/messages {body, reply_to}: Idempotency-Key required; a repeated key from
  the same sender returns the first message (200). A reply joins its parent's thread. `@name`
  mentions a principal; `@all` / `@agents` / `@humans` mention every principal of that kind
  (`as` is the group name), like the real board.
- GET /v1/channels/{id}/messages?after=|before=&limit=|thread=, GET /v1/messages/{id}
- GET /v1/activity?after=&wait=&channels=: the caller's own messages are left out; without
  `after`, `next_cursor` means "now"; `wait` long-polls.
- GET /v1/search?q=&sort=&limit=: every word must appear as a prefix, "phrases" as written,
  `in:#channel`, `from:@name`, `after:` (date or 12h / 7d); sort newest | oldest | relevance
  (term hits); snippets mark hits with \\x02 / \\x03 as FTS5 does.
- PUT /v1/channels/{id}/messages/{mid}/reactions/{emoji}

`requests` records (method, path, query, principal name) for assertions. Not thread-safe
beyond one lock; fine for tests.
"""

from __future__ import annotations

import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

GROUPS = {"all": ("agent", "user"), "agents": ("agent",), "humans": ("user",),
          "everyone": ("agent", "user"), "people": ("user",)}


class Board:
    def __init__(self, invite: str = "inv-code", channels: dict[int, str] | None = None) -> None:
        self.invite = invite
        self.channels = dict(channels or {12: "flock-test"})
        self.lock = threading.Condition()
        self.principals: dict[int, dict[str, Any]] = {}
        self.tokens: dict[str, int] = {}
        self.members: dict[int, set[int]] = {c: set() for c in self.channels}
        self.msgs: list[dict[str, Any]] = []
        self.seq: dict[int, int] = {c: 0 for c in self.channels}
        self.idem: dict[tuple[int, str], dict] = {}
        self.requests: list[tuple[str, str, dict, str | None]] = []
        self.joins: list[str] = []
        self._n = 0

    # ------------------------------------------------------------------ state helpers
    def principal(self, name: str, kind: str = "agent") -> dict[str, Any]:
        for p in self.principals.values():
            if p["external_id"] == name:
                return p
        pid = 100 + len(self.principals)
        p = {"id": pid, "external_id": name, "display_name": name, "identity_type": kind}
        self.principals[pid] = p
        return p

    def token_for(self, name: str, kind: str = "agent") -> str:
        """A token for a test-side principal (a human posting from outside the swarm)."""
        with self.lock:
            p = self.principal(name, kind)
            self._n += 1
            tok = f"fbt_{name}_{self._n}"
            self.tokens = {t: i for t, i in self.tokens.items() if i != p["id"]}
            self.tokens[tok] = p["id"]
            for c in self.channels:
                self.members[c].add(p["id"])
            return tok

    def names(self) -> set[str]:
        return {p["external_id"] for p in self.principals.values()}

    def post(self, sender: str, channel: int, body: str, reply_to: int | None = None) -> dict:
        """Post directly (as a test-side principal)."""
        with self.lock:
            return self._post(self.principal(sender, "user")["id"], channel, body, reply_to)

    def _post(self, pid: int, channel: int, body: str, reply_to: int | None) -> dict:
        parent = next((m for m in self.msgs if m["id"] == reply_to), None) if reply_to else None
        if reply_to and (parent is None or parent["channel_id"] != channel):
            raise LookupError("reply_to not found")
        mentions = []
        for p in self.principals.values():
            if p["id"] != pid and re.search(rf"(?<![\w@])@{re.escape(p['external_id'])}(?![\w-])", body):
                mentions.append({**p, "as": p["external_id"]})
        for g in re.findall(r"(?<![\w@/])@(all|agents|humans|everyone|people)(?![\w-])", body, re.I):
            for p in self.principals.values():
                if p["id"] != pid and p["identity_type"] in GROUPS[g.lower()] and p not in mentions:
                    mentions.append({**p, "as": g.lower()})
        self.seq[channel] += 1
        mid = 1000 + len(self.msgs) + 1
        msg = {"id": mid, "channel_id": channel, "seq": self.seq[channel],
               "sender": dict(self.principals[pid]), "body": body, "reply_to": reply_to,
               "thread_id": (parent["thread_id"] or parent["id"]) if parent else None,
               "attachments": [], "mentions": mentions, "reactions": [],
               "created_at": time.time(), "edited_at": None}
        self.msgs.append(msg)
        self.lock.notify_all()
        return msg

    def _view(self, m: dict, me: int | None = None, reason: bool = False) -> dict:
        out = dict(m)
        out["mentions"] = [{k: v for k, v in x.items()} for x in m["mentions"]]
        if reason:
            mine = {x["id"] for x in self.msgs if x["sender"]["id"] == me and x["thread_id"] is None} | {
                x["thread_id"] for x in self.msgs if x["sender"]["id"] == me and x["thread_id"]}
            out["reason"] = ("mention" if any(x["id"] == me for x in m["mentions"])
                             else "reply" if m["thread_id"] in mine else "channel")
            out["channel"] = {"id": m["channel_id"], "name": self.channels[m["channel_id"]],
                              "kind": "public"}
        return out

    # ------------------------------------------------------------------ search
    def search(self, q: str, sort: str, limit: int) -> dict:
        tokens = re.findall(r'"[^"]*"|\S+', q)
        words, phrases, where = [], [], []
        for t in tokens:
            m = re.match(r"^(in|from|after):(.+)$", t)
            if m:
                where.append((m.group(1), m.group(2)))
            elif t.startswith('"'):
                phrases.append(t.strip('"').lower())
            else:
                words.append(t.lower())
        out = []
        for msg in self.msgs:
            body = msg["body"].lower()
            toks = re.findall(r"\w+", body)
            ok = all(any(x.startswith(w2) for x in toks) for w in words for w2 in re.findall(r"\w+", w))
            ok = ok and all(p in body for p in phrases)
            for op, val in where:
                if op == "in":
                    ok = ok and self.channels.get(msg["channel_id"]) == val.lstrip("#")
                elif op == "from":
                    ok = ok and msg["sender"]["external_id"] == val.lstrip("@")
                elif op == "after":
                    rel = re.fullmatch(r"(\d+)([hdw])", val)
                    if rel:
                        since = time.time() - int(rel.group(1)) * {"h": 3600, "d": 86400, "w": 604800}[rel.group(2)]
                    else:
                        since = time.mktime(time.strptime(val, "%Y-%m-%d"))
                    ok = ok and msg["created_at"] >= since
            if ok:
                score = sum(sum(1 for x in toks if x.startswith(w)) for w in words)
                snip = msg["body"]
                for w in words:
                    snip = re.sub(rf"(?i)\b({re.escape(w)}\w*)", "\x02\\1\x03", snip)
                out.append((score, {**self._view(msg), "snippet": snip[:200],
                                    "channel": {"id": msg["channel_id"],
                                                "name": self.channels[msg["channel_id"]]}}))
        if sort == "relevance":
            out.sort(key=lambda x: (-x[0], -x[1]["id"]))
        elif sort == "oldest":
            out.sort(key=lambda x: x[1]["id"])
        else:
            out.sort(key=lambda x: -x[1]["id"])
        return {"messages": [m for _, m in out[:limit]], "channels": [], "people": [],
                "total": len(out), "next_offset": None}


def make_handler(board: Board) -> type[BaseHTTPRequestHandler]:
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a: Any) -> None:
            pass

        def send(self, code: int, obj: Any, headers: dict | None = None) -> None:
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(b)))
            for k, v in (headers or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(b)

        def fail(self, code: int, err: str, msg: str = "") -> None:
            self.send(code, {"error": {"code": err, "message": msg or err}})

        def body(self) -> dict:
            n = int(self.headers.get("content-length") or 0)
            return json.loads(self.rfile.read(n) or b"{}") if n else {}

        def who(self) -> int | None:
            auth = self.headers.get("Authorization", "")
            return board.tokens.get(auth[7:]) if auth.startswith("Bearer ") else None

        def route(self, method: str) -> None:
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            body = self.body() if method in ("POST", "PUT", "PATCH") else {}
            me = self.who()
            board.requests.append((method, u.path, q, board.principals[me]["external_id"] if me else None))
            if method == "POST" and u.path == "/v1/join":
                if body.get("code") != board.invite:
                    return self.fail(404, "not_found", "this invite link is not valid")
                name = body.get("name") or ""
                with board.lock:
                    p = board.principal(name)
                    created = name not in board.joins
                    board.joins.append(name)
                tok = board.token_for(name)
                return self.send(201 if created else 200, {"token": tok, "principal": p,
                                                           "created": created})
            if me is None:
                return self.fail(401, "unauthorized")
            if method == "GET" and u.path == "/v1/channels":
                out = [{"id": c, "name": n, "kind": "public", "member": me in board.members[c],
                        "last_seq": board.seq[c]} for c, n in board.channels.items()
                       if q.get("name") in (None, n)]
                return self.send(200, {"channels": out})
            if m := re.fullmatch(r"/v1/channels/(\d+)/members", u.path):
                board.members[int(m.group(1))].add(int(body.get("principal_id")))
                return self.send(200, {"members": []})
            if re.fullmatch(r"/v1/principals/\d+", u.path):
                return self.send(200, {})
            if method == "GET" and (m := re.fullmatch(r"/v1/messages/(\d+)", u.path)):
                msg = next((x for x in board.msgs if x["id"] == int(m.group(1))), None)
                return self.send(200, board._view(msg)) if msg else self.fail(404, "not_found")
            if m := re.fullmatch(r"/v1/channels/(\d+)/messages", u.path):
                ch = int(m.group(1))
                if ch not in board.channels:
                    return self.fail(404, "not_found")
                if method == "POST":
                    key = self.headers.get("Idempotency-Key")
                    if not key:
                        return self.fail(400, "bad_request", "Idempotency-Key header is required")
                    with board.lock:
                        if (me, key) in board.idem:
                            return self.send(200, board.idem[(me, key)], {"Idempotent-Replayed": "true"})
                        try:
                            msg = board._post(me, ch, body.get("body") or "", body.get("reply_to"))
                        except LookupError:
                            return self.fail(404, "not_found", "reply_to not found")
                        board.idem[(me, key)] = msg
                    return self.send(201, board._view(msg))
                limit = int(q.get("limit") or 50)
                with board.lock:
                    rows = [x for x in board.msgs if x["channel_id"] == ch]
                    if "thread" in q:
                        r = int(q["thread"])
                        rows = [x for x in rows if x["id"] == r or x["thread_id"] == r][:limit]
                    elif "before" in q:
                        rows = [x for x in rows if x["seq"] < int(q["before"])][-limit:]
                    else:
                        rows = [x for x in rows if x["seq"] > int(q.get("after") or 0)][:limit]
                    out = [board._view(x) for x in rows]
                return self.send(200, {"messages": out, "next_cursor": out[-1]["seq"] if out else 0,
                                       "has_more": False})
            if method == "GET" and u.path == "/v1/activity":
                follow = [int(x) for x in (q.get("channels") or "").split(",") if x.strip().isdigit()]
                wait = float(q.get("wait") or 0)
                with board.lock:
                    if "after" not in q:
                        return self.send(200, {"messages": [], "next_cursor": board.msgs[-1]["id"]
                                               if board.msgs else 0, "has_more": False})
                    after = int(q["after"])

                    def mine() -> list[dict]:
                        return [board._view(x, me, reason=True) for x in board.msgs
                                if x["id"] > after and x["sender"]["id"] != me
                                and (x["channel_id"] in follow
                                     or any(y["id"] == me for y in x["mentions"]))]

                    end = time.monotonic() + wait
                    out = mine()
                    while not out and time.monotonic() < end:
                        board.lock.wait(max(0.0, end - time.monotonic()))
                        out = mine()
                return self.send(200, {"messages": out, "next_cursor": out[-1]["id"] if out else after,
                                       "has_more": False})
            if method == "GET" and u.path == "/v1/search":
                with board.lock:
                    return self.send(200, board.search(q.get("q") or "", q.get("sort") or "newest",
                                                       int(q.get("limit") or 20)))
            if method == "PUT" and (m := re.fullmatch(r"/v1/channels/(\d+)/messages/(\d+)/reactions/(.+)", u.path)):
                with board.lock:
                    msg = next((x for x in board.msgs if x["id"] == int(m.group(2))), None)
                    if msg is None:
                        return self.fail(404, "not_found")
                    msg["reactions"].append({"emoji": unquote(m.group(3)),
                                             "by": board.principals[me]["external_id"]})
                return self.send(200, board._view(msg))
            return self.fail(404, "not_found")

        def do_GET(self) -> None:
            self.route("GET")

        def do_POST(self) -> None:
            self.route("POST")

        def do_PUT(self) -> None:
            self.route("PUT")

        def do_PATCH(self) -> None:
            self.route("PATCH")

    return H


def serve(board: Board | None = None) -> tuple[Board, ThreadingHTTPServer, str]:
    """Start on a free port in a daemon thread: (board, server, base url)."""
    board = board or Board()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(board))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return board, srv, f"http://127.0.0.1:{srv.server_port}"
