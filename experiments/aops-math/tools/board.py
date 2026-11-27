"""Team board for small models: compact, quote-proof board tools (#general only).

board.post  posts a message from a FILE (or short inline text), so math, quotes and $ never pass through the shell.
board.read  returns only messages you have not seen yet, as compact lines "#7 agent-002: ...", own posts marked.
board.wait  waits (up to `timeout` s) until a teammate posts something new, then returns it like board.read.
The read cursor lives in a small file in the agent's container, so agents never handle sequence numbers.
"""

import os
import re
import time

import httpx
from msgboard.client import Client

DEPENDENCIES = ["httpx"]
REQUIRES = ["message_board"]
BUNDLE_FILES = {"msgboard": ["msgboard/__init__.py", "msgboard/client.py"]}

CURSOR = "/tmp/.board_cursor"
MAX_LEN = int(os.environ.get("BOARD_MAX_LEN", "2000"))
GATE = False          # solo-first variants: reading stays closed until the agent wrote /app/answer.txt and posted once
POSTED_MARK = "/tmp/.board_posted"
ANSWER_PATH = "/app/answer.txt"
GATE_NOTE = "(the board opens once you have written your own answer to /app/answer.txt and posted it)"


def _gate_closed() -> bool:
    if not GATE:
        return False
    try:
        has_answer = os.path.isfile(ANSWER_PATH) and open(ANSWER_PATH).read().strip() != ""
    except OSError:
        has_answer = False
    return not (has_answer and os.path.exists(POSTED_MARK))   # per-variant board limit (tool and spec must agree)


def _client(max_retries=10):
    socket_path = os.environ.get("MSGBOARD_SOCKET")
    transport = httpx.HTTPTransport(uds=socket_path) if socket_path else None
    return Client.from_env(transport=transport, timeout=40, max_retries=max_retries)


def _cursor() -> int:
    try:
        return int(open(CURSOR).read().strip() or 0)
    except (OSError, ValueError):
        return 0


def _set_cursor(seq: int) -> None:
    with open(CURSOR, "w") as f:
        f.write(str(seq))


def _me(board) -> str:
    me = board.me()
    who = me.get("principal", me)
    return who.get("external_id") or who.get("display_name") or ""


def _lines(messages, me):
    out = []
    for m in messages:
        sender = (m.get("sender") or {}).get("external_id") or (m.get("sender") or {}).get("display_name") or "?"
        tag = f"#{m.get('seq')} {sender}{' (you)' if sender == me else ''}"
        if m.get("reply_to"):
            tag += f" (reply to #{m['reply_to']})"
        out.append(f"{tag}: {m.get('body') or m.get('text') or ''}")
    return out


def _fetch(board, wait: float = 0):
    channel = board.channel_id("general")
    page = board.read(channel, after=_cursor(), limit=100, wait=wait)
    msgs = page.get("messages", []) if isinstance(page, dict) else page
    if msgs:
        _set_cursor(max(m.get("seq", 0) for m in msgs))
    return msgs


def post(path: str = "", text: str = "", reply_to: int | None = None, content: str = "", message: str = "",
         body: str = "", file: str = "") -> dict:
    """Post a message to the team board. Write it to /tmp/msg.txt first, then call with {} (or {"path": "<file>"}); add "reply_to": N to answer message #N. Short plain text also works as {"text": "..."}."""
    raw = next((v for v in (path, file, text, content, message, body) if v), "")
    if not raw and os.path.isfile("/tmp/msg.txt"):          # default: the message file from the heredoc
        raw = "/tmp/msg.txt"
    if raw and len(raw) < 300 and "\n" not in raw and os.path.isfile(raw.strip()):
        raw = open(raw.strip()).read()
    raw = (raw or "").strip()
    if not raw:
        raise ValueError('empty message: write it to /tmp/msg.txt first, then post {"path": "/tmp/msg.txt"}')
    if len(raw) < 15 or raw.lower() in {"file msg", "file post", "message", "msg", "post", "test"}:
        raise ValueError('message too short: write your full message to /tmp/msg.txt, then post {"path": "/tmp/msg.txt"}')
    if len(raw) > MAX_LEN:                     # never cut a message: reject and let the agent shorten it
        raise ValueError(f"message is {len(raw)} characters; the board allows {MAX_LEN}. Post your result and the step "
                         "that matters, not your working; if you really need more, split it into two posts.")
    with _client() as board:
        for attempt in range(4):
            try:
                msg = board.post(board.channel_id("general"), raw, reply_to=reply_to)
                if GATE:
                    open(POSTED_MARK, "w").close()
                return {"posted": f"#{msg.get('seq')}"}
            except Exception as e:            # the board's min_post_gap: wait it out and retry
                m = re.search(r"wait (\d+) s", str(e))
                if not m or attempt == 3:
                    raise
                time.sleep(int(m.group(1)) + 1)


def read() -> dict:
    """Show the board messages you have not seen yet (compact; your own posts are marked)."""
    if _gate_closed():
        return {"new_messages": [GATE_NOTE]}
    with _client() as board:
        msgs = _fetch(board)
        return {"new_messages": _lines(msgs, _me(board)) or ["(no new messages)"]}


def wait(timeout: int = 30) -> dict:
    """Wait up to `timeout` seconds (max 60) until a teammate posts something new, then show the new messages."""
    timeout = max(1, min(int(timeout), 60))
    if _gate_closed():
        return {"new_messages": [GATE_NOTE]}
    with _client(max_retries=5) as board:
        me = _me(board)
        msgs = _fetch(board, wait=timeout)
        lines = _lines(msgs, me)
        if msgs and all(l.split(" ", 2)[1] == me for l in lines):   # only my own posts: wait once more
            msgs = _fetch(board, wait=timeout)
            lines += _lines(msgs, me)
        return {"new_messages": lines or [f"(nothing new in {timeout} s)"]}


TOOLS = {"post": post, "read": read, "wait": wait}
