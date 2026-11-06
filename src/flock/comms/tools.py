"""The board rules (docs/comms.md, "Rules") the Harbor board resource applies to every post.
Pure: no board, no clock."""

from __future__ import annotations

import math
import re
from typing import Any

from flock.models import CommsRules

GROUP_RE = re.compile(r"(?<![\w@/])@(all|agents|humans|everyone|people)(?![\w-])", re.IGNORECASE)
GROUP_NAMES = frozenset({"all", "agents", "humans", "everyone", "people"})


class Refused(Exception):
    """A rule said no; the message is what the model reads (it starts with "refused:")."""

    def __init__(self, rule: str, message: str) -> None:
        super().__init__(f"refused: {message}")
        self.rule = rule


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
