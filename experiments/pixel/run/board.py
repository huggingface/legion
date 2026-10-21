#!/usr/bin/env python3
"""board.py: the msgboard from a leader's shell (experiment 2). Stdlib only.

    python3 board.py join                 # redeem $BOARD_INVITE as $FLOCK_AGENT; token -> .board_token
    python3 board.py post "text"          # to the channel (mention with @leader-0003)
    python3 board.py reply MSG_ID "text"  # in that message's thread
    python3 board.py read                 # new messages in the channel since the last read
    python3 board.py react MSG_ID EMOJI   # e.g. 👍
    python3 board.py transcript           # the whole channel as JSONL (observer)

Env: BOARD_URL, FLOCK_AGENT, BOARD_INVITE (one multi-use code; each leader joins under its own
flock name, and rejoining with the same name rotates the token, same identity). The channel is
found after joining: BOARD_CHANNEL if it is a numeric id, else by name (BOARD_CHANNEL_NAME,
default flock-pixel-test) from GET /v1/channels; the id is cached in .board_channel.
The token lives in .board_token (mode 600, in the working directory) and is never printed;
BOARD_TOKEN in the env is used instead when set (the observer). Writes send an Idempotency-Key and
retry 429 / 5xx with the same key, so a retried post never duplicates.
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

TOKEN_FILE = ".board_token"
CURSOR_FILE = ".board_cursor"
CHANNEL_FILE = ".board_channel"


def channel_id() -> int:
    v = os.environ.get("BOARD_CHANNEL", "").strip()
    if v.isdigit():
        return int(v)
    if os.path.exists(CHANNEL_FILE):
        with open(CHANNEL_FILE) as fh:
            return int(fh.read().strip())
    want = v or os.environ.get("BOARD_CHANNEL_NAME", "").strip() or "flock-pixel-test"
    want = want.lstrip("#")
    chans = call("GET", "/v1/channels?" + urllib.parse.urlencode({"name": want})).get(
        "channels", []
    )
    hit = [c for c in chans if c.get("name") == want] or (chans if len(chans) == 1 else [])
    if not hit:
        raise BoardError(f"channel {want!r} not visible to this agent (is the invite for it?)")
    with open(CHANNEL_FILE, "w") as fh:
        fh.write(str(hit[0]["id"]))
    return int(hit[0]["id"])


class BoardError(Exception):
    pass


def env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise BoardError(f"{name} is not set")
    return v


def token() -> str:
    if os.environ.get("BOARD_TOKEN"):
        return os.environ["BOARD_TOKEN"].strip()
    try:
        with open(TOKEN_FILE) as fh:
            return fh.read().strip()
    except FileNotFoundError:
        raise BoardError("not joined yet: run `python3 board.py join` first") from None


def call(
    method: str, path: str, body: dict | None = None, auth: bool = True, idem: bool = False
) -> dict:
    url = env("BOARD_URL").rstrip("/") + path
    headers = {"Content-Type": "application/json"}
    if auth:
        headers["Authorization"] = f"Bearer {token()}"
    if idem:
        headers["Idempotency-Key"] = uuid.uuid4().hex  # same key on every retry below
    data = json.dumps(body).encode() if body is not None else None
    for attempt in range(5):
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read()).get("error", {})
            except Exception:  # noqa: BLE001
                err = {}
            if (e.code == 429 or e.code >= 500) and attempt < 4:
                time.sleep(min(10.0, float(e.headers.get("Retry-After") or 2**attempt)))
                continue
            raise BoardError(
                f"HTTP {e.code} {err.get('code', '')}: {err.get('message', e.reason)}"
            ) from None
        except urllib.error.URLError as e:
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            raise BoardError(f"cannot reach the board at {url}: {e.reason}") from None
    raise BoardError("unreachable")


def join() -> str:
    name = env("FLOCK_AGENT")
    body = {"code": env("BOARD_INVITE"), "name": name, "display_name": name}
    try:
        out = call("POST", "/v1/join", body, auth=False)
    except BoardError as e:
        if "this invite is for the agent" not in str(e):
            raise
        out = call("POST", "/v1/join", {"code": body["code"]}, auth=False)  # a named invite
    fd = os.open(TOKEN_FILE, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(out["token"])
    os.chmod(TOKEN_FILE, 0o600)
    p = out.get("principal", {})
    if os.path.exists(CHANNEL_FILE):
        os.remove(CHANNEL_FILE)
    return (
        f"joined as {p.get('external_id', name)} (principal {p.get('id')}), channel #{channel_id()}"
    )


def one_line(m: dict) -> str:
    s = m.get("sender") or {}
    who = s.get("display_name") or s.get("external_id") or "?"
    body = " ".join(str(m.get("body", "")).split())
    if len(body) > 300:
        body = body[:299] + "…"
    thread = f" (re #{m['thread_id']})" if m.get("thread_id") and m["thread_id"] != m["id"] else ""
    return f"#{m['id']} {who}{thread}: {body}"


def channel_of(m: dict):
    return m.get("channel_id") or (m.get("channel") or {}).get("id")


def read() -> str:
    ch = channel_id()
    cursor = None
    if os.path.exists(CURSOR_FILE):
        with open(CURSOR_FILE) as fh:
            cursor = fh.read().strip() or None
    lines = []
    while True:
        q = {"channels": "joined", "limit": 200, **({"after": cursor} if cursor else {})}
        page = call("GET", "/v1/activity?" + urllib.parse.urlencode(q))
        lines += [one_line(m) for m in page.get("messages", []) if channel_of(m) == ch]
        if page.get("next_cursor") is not None:
            cursor = str(page["next_cursor"])
        if not page.get("has_more"):
            break
    if cursor:
        with open(CURSOR_FILE, "w") as fh:
            fh.write(cursor)
    return "\n".join(lines) if lines else "(no new messages)"


def transcript() -> str:
    ch = channel_id()
    out, after = [], 0
    while True:
        page = call("GET", f"/v1/channels/{ch}/messages?after={after}&limit=200")
        msgs = page.get("messages", [])
        out += [
            json.dumps(
                {
                    "id": m["id"],
                    "seq": m.get("seq"),
                    "ts": m.get("created_at"),
                    "author": (m.get("sender") or {}).get("display_name")
                    or (m.get("sender") or {}).get("external_id"),
                    "reply_to": m.get("reply_to"),
                    "thread_id": m.get("thread_id"),
                    "body": m.get("body"),
                }
            )
            for m in msgs
        ]
        nxt = page.get("next_cursor")
        if not msgs or nxt is None or nxt == after:
            return "\n".join(out)
        after = nxt


def main(argv: list[str]) -> int:
    try:
        cmd, args = (argv[0] if argv else ""), argv[1:]
        if cmd == "join":
            print(join())
        elif cmd == "post" and len(args) == 1:
            m = call(
                "POST",
                f"/v1/channels/{channel_id()}/messages",
                {"body": args[0]},
                idem=True,
            )
            print(f"posted #{m['id']}")
        elif cmd == "reply" and len(args) == 2 and args[0].isdigit():
            m = call(
                "POST",
                f"/v1/channels/{channel_id()}/messages",
                {"body": args[1], "reply_to": int(args[0])},
                idem=True,
            )
            print(f"replied #{m['id']} in #{args[0]}")
        elif cmd == "read" and not args:
            print(read())
        elif cmd == "react" and len(args) == 2 and args[0].isdigit():
            em = urllib.parse.quote(args[1], safe="")
            call("PUT", f"/v1/channels/{channel_id()}/messages/{args[0]}/reactions/{em}")
            print(f"reacted {args[1]} on #{args[0]}")
        elif cmd == "transcript" and not args:
            print(transcript())
        else:
            print(
                'usage: board.py join | post "text" | reply ID "text" | read | react ID EMOJI '
                "| transcript",
                file=sys.stderr,
            )
            return 2
        return 0
    except BoardError as e:
        print(f"board error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
