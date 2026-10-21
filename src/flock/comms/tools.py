"""The built-in comms tools (docs/comms.md): definitions, rule checks, result shapes, search cut 1.

The model sees the bare names `post`, `read`, `search`; the spec names them `comms:post`, ... .
Everything here is pure (no board, no clock) except `relaxed_search`, which takes the board call
as a function, so the rules and the relaxation are tested without a board.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

from flock.models import CommsRules, comms_name
from flock.tools.loader import ToolDef

BUILTIN_MODULE = "flock:comms"
GROUP_RE = re.compile(r"(?<![\w@/])@(all|agents|humans|everyone|people)(?![\w-])", re.IGNORECASE)
GROUP_NAMES = frozenset({"all", "agents", "humans", "everyone", "people"})
READ_DEFAULT, READ_MAX = 20, 100
SEARCH_DEFAULT, SEARCH_MAX = 10, 50
TEXT_CHARS = 2000  # per message in a read result
_ID = {"type": ["integer", "string"]}

DEFS: dict[str, dict[str, Any]] = {
    "comms:post": {
        "description": "Post a message to your team's board channel. Start a topic with "
                       "thread: \"new\", or reply inside an existing topic with thread: <its root "
                       "message id>. Returns the message id and its thread.",
        "properties": {
            "text": {"type": "string", "description": "the message (markdown); mention someone "
                                                      "with @name"},
            "thread": {**_ID, "description": "\"new\" for a new topic, or the root message id of "
                                             "an existing thread, e.g. 5012"},
            "reply_to": {**_ID, "description": "optional: the id of the message you answer "
                                               "(inside that thread)"},
        },
        "required": ["text"],
    },
    "comms:read": {
        "description": "Read your team's board channel: the newest messages, or one whole "
                       "thread. Each message has id, thread, from, time and text.",
        "properties": {
            "n": {"type": "integer", "description": f"how many of the newest messages (default "
                                                    f"{READ_DEFAULT}, at most {READ_MAX})"},
            "thread": {**_ID, "description": "optional: a thread's root message id, to read "
                                             "that thread instead"},
        },
        "required": [],
    },
    "comms:search": {
        "description": "Search the messages of your team's board channel, best matches first. "
                       "Returns id, thread, author, time and a snippet for each hit; if no "
                       "message has every word, trailing words are dropped until something "
                       "matches.",
        "properties": {
            "query": {"type": "string", "description": "words to look for (prefixes match: "
                                                       "deplo finds deployed); \"exact phrase\" ok"},
            "thread": {**_ID, "description": "optional: only this thread (its root id)"},
            "from": {"type": "string", "description": "optional: only messages by this name, "
                                                      "e.g. ds-1"},
            "since": {"type": "string", "description": "optional: only newer messages: a date "
                                                       "(2026-10-21), today, or an age (12h, 7d)"},
            "limit": {"type": "integer", "description": f"how many results (default "
                                                        f"{SEARCH_DEFAULT}, at most {SEARCH_MAX})"},
        },
        "required": ["query"],
    },
}


def tool_def(entry: str) -> ToolDef:
    d = DEFS[entry]
    return ToolDef(name=comms_name(entry), description=d["description"],
                   parameters={"type": "object", "additionalProperties": False,
                               "properties": d["properties"], "required": d["required"]},
                   module=BUILTIN_MODULE, function=comms_name(entry), builtin=entry)


class Refused(Exception):
    """A rule said no; the message is what the model reads (it starts with "refused:")."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"refused: {message}")
        self.rule = rule


def as_id(value: Any, what: str) -> int | None:
    """An id the model may give as 5012, "5012" or "#5012"; None for null / ""."""
    if value is None or value == "":
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    s = str(value).strip().lstrip("#")
    if s.isdigit():
        return int(s)
    raise ValueError(f"{what} must be a message id (a number) or \"new\", got {value!r}")


def check_content(rules: CommsRules, text: str, thread: Any) -> None:
    """Rules that look only at the post itself (no state): raises Refused."""
    if not isinstance(text, str) or not text.strip():
        raise Refused("empty", "text is empty")
    if rules.max_len is not None and len(text) > rules.max_len:
        raise Refused("max_len", f"text is {len(text)} chars; max_len is {rules.max_len}. "
                                 "Shorten it and post again.")
    if rules.no_group_mentions and (m := GROUP_RE.search(text)):
        raise Refused("no_group_mentions", f"@{m.group(1)} is not allowed here (it reaches "
                                           "everyone on the board); mention people by name")
    if rules.threads == "per_topic" and (thread is None or thread == ""):
        raise Refused("threads", "say which thread: thread: \"new\" for a new topic, or the "
                                 "root id of the topic's existing thread")


def check_rate(rules: CommsRules, posts: int, last_post: float | None, now: float) -> None:
    """Per-agent `max_posts` and `min_post_gap` (wall clock): raises Refused."""
    if rules.max_posts is not None and posts >= rules.max_posts:
        raise Refused("max_posts", f"you have used all {rules.max_posts} posts (max_posts)")
    gap = rules.gap_s
    if gap and last_post is not None and now - last_post < gap:
        wait = max(1, math.ceil(gap - (now - last_post)))
        raise Refused("min_post_gap", f"wait {wait} s (min_post_gap {rules.min_post_gap} "
                                      "between your posts)")


def thread_of(m: dict[str, Any]) -> int:
    return m.get("thread_id") or m["id"]


def _time(ts: Any) -> str | None:
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    return str(ts) if ts else None


def author(m: dict[str, Any]) -> str:
    s = m.get("sender") or {}
    return str(s.get("external_id") or s.get("display_name") or s.get("id") or "?")


def brief(m: dict[str, Any], chars: int = TEXT_CHARS) -> dict[str, Any]:
    """A message as a read result shows it."""
    body = m.get("body") or ""
    if len(body) > chars:
        body = body[:chars] + f" [... {len(body) - chars} more chars]"
    out = {"id": m["id"], "thread": thread_of(m), "from": author(m), "time": _time(m.get("created_at")),
           "text": body}
    if m.get("reply_to") and m.get("reply_to") != thread_of(m):
        out["reply_to"] = m["reply_to"]
    return out


def snippet(m: dict[str, Any]) -> str:
    s = m.get("snippet")
    if not isinstance(s, str) or not s:
        s = (m.get("body") or "")[:160]
    return s.replace("\x02", "**").replace("\x03", "**")


def hit(m: dict[str, Any]) -> dict[str, Any]:
    return {"id": m["id"], "thread": thread_of(m), "author": author(m),
            "time": _time(m.get("created_at")), "snippet": snippet(m)}


# --------------------------------------------------------------------------- search, cut 1

_TOKEN = re.compile(r'"[^"]*"|\S+')
_SINCE = re.compile(r"^(\d{4}-\d{1,2}-\d{1,2}|today|yesterday|\d+[hdw])$", re.IGNORECASE)
_FILTER = re.compile(r"^[a-z]+:", re.IGNORECASE)


def query_terms(query: str) -> list[str]:
    """Words and "phrases" of the model's query; board filters it may have typed (in:, from:,
    ...) are dropped, the tool sets those itself."""
    return [t for t in _TOKEN.findall(query or "") if t.strip('"').strip() and not _FILTER.match(t)]


def board_query(terms: list[str], channel: str, *, author_: str | None, since: str | None) -> str:
    q = [*terms, f"in:#{channel}"]
    if author_:
        q.append(f"from:@{author_.lstrip('@')}")
    if since:
        q.append(f"after:{since}")
    return " ".join(q)


Search = Callable[[str, int], Awaitable[dict[str, Any]]]


async def relaxed_search(search: Search, *, query: str, channel: str, thread: int | None = None,
                         author_: str | None = None, since: str | None = None,
                         limit: int = SEARCH_DEFAULT) -> dict[str, Any]:
    """Cut 1: the board's `GET /v1/search` (`in:#<channel>`, `sort=relevance`, every word
    required, prefixes match). With no hit, drop the last word and try again, down to one word.
    `search(q, limit)` is the board call."""
    terms = query_terms(query)
    if not terms:
        raise ValueError("query has no words to search for")
    if since is not None and not _SINCE.match(since.strip()):
        raise ValueError(f"since must be a date (2026-10-21), today, yesterday or an age like 12h "
                         f"or 7d, got {since!r}")
    limit = max(1, min(int(limit or SEARCH_DEFAULT), SEARCH_MAX))
    ask = min(100, limit * 5) if thread is not None else limit
    dropped: list[str] = []
    tries = 0
    for n in range(len(terms), 0, -1):
        tries += 1
        q = board_query(terms[:n], channel, author_=author_, since=since.strip() if since else None)
        res = await search(q, ask)
        msgs = [m for m in res.get("messages") or [] if isinstance(m, dict)]
        if thread is not None:
            msgs = [m for m in msgs if thread_of(m) == thread]
        if msgs:
            out: dict[str, Any] = {"query": " ".join(terms[:n]), "results": [hit(m) for m in msgs[:limit]]}
            if dropped:
                out["dropped"] = dropped
                out["note"] = f"no message had every word; dropped: {' '.join(dropped)}"
            if thread is None:
                out["total"] = res.get("total", len(msgs))
            return out
        dropped.insert(0, terms[n - 1])
    return {"query": " ".join(terms), "results": [], "tries": tries,
            "note": "no message in the channel matches even the first word"}
