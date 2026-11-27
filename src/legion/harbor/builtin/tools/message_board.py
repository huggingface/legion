"""Message-board operations; connection settings come from the agent environment."""

import os
import re
import time
from pathlib import Path

import httpx
from msgboard.client import BoardError, Client

DEPENDENCIES = ["httpx"]
REQUIRES = ["message_board"]
# The SDK shares a distribution with the server. Ship only its client files;
# its runtime dependency (httpx) is resolved normally above.
BUNDLE_FILES = {"msgboard": ["msgboard/__init__.py", "msgboard/client.py"]}


# At least 98 s of retries: enough to outlast a Space board's restart, during
# which it answers 503 with Retry-After. After that, the agent gets the error.
def _client(max_retries=10):
    socket_path = os.environ.get("MSGBOARD_SOCKET")
    transport = httpx.HTTPTransport(uds=socket_path) if socket_path else None
    return Client.from_env(transport=transport, timeout=10, max_retries=max_retries)


# read/wait with unread=True keep, per channel, the last message the agent was shown: one container per
# agent, and each tool call is its own process, so the place lives in a file.
UNREAD_DIR = Path("/tmp/legion-board")
# A post refused by min_post_gap is retried once when the board asks to wait at most this long.
POST_GAP_RETRY_S = 30


def _last_seen(channel_id: int) -> int:
    path = UNREAD_DIR / f"{channel_id}.seq"
    return int(path.read_text()) if path.exists() else 0


def _messages(board, channel_id: int, page: dict, unread: bool, compact: bool) -> dict:
    """The page as read returns it; unread: remember its last message; compact: one line per message."""
    if unread and page["messages"]:
        UNREAD_DIR.mkdir(parents=True, exist_ok=True)
        (UNREAD_DIR / f"{channel_id}.seq").write_text(str(page["messages"][-1]["seq"]))
    if not compact:
        return page
    me = _identity_id(board)
    lines = []
    for m in page["messages"]:
        sender = m["sender"]
        head = f"#{m['id']} {sender.get('external_id') or sender['display_name']}"
        head += " (you)" if sender["id"] == me else ""
        head += f" (reply to #{m['reply_to']})" if m.get("reply_to") else ""
        lines.append(f"{head}: {m['body']}")
    return {"messages": lines}


def _identity_id(board):
    identity = board.me()
    return identity.get("principal", identity)["id"]


def whoami() -> dict:
    """Show your authenticated board identity."""
    with _client() as board:
        return board.me()


def agents() -> dict:
    """List visible agent identities."""
    with _client() as board:
        return {
            "agents": [p for p in board.people() if p.get("identity_type") == "agent"]
        }


def read(
    channel: str = "general",
    after: int = 0,
    limit: int = 50,
    unread: bool = False,
    compact: bool = False,
) -> dict:
    """Read messages after a channel sequence number; limit must be 1 through 100. unread: only the
    messages you have not been shown yet (the tool keeps your place; `after` is ignored). compact: one
    line per message, "#<id> <sender>: <text>", your own posts marked."""
    if not 1 <= limit <= 100:
        raise ValueError("limit must be between 1 and 100")
    with _client() as board:
        channel_id = board.channel_id(channel)
        start = _last_seen(channel_id) if unread else after
        page = board.read(channel_id, after=start, limit=limit)
        return _messages(board, channel_id, page, unread, compact)


def post(
    text: str | None = None,
    channel: str = "general",
    reply_to: int | None = None,
    request_id: str | None = None,
    path: str | None = None,
) -> dict:
    """Post a message: `text`, or the content of the file at `path` (write the message to a file first, so
    quotes, $ and math never pass through the shell). Reuse request_id when retrying the same logical post."""
    if (text is None) == (path is None):
        raise ValueError("give exactly one of text or path")
    if path is not None:
        text = Path(path).read_text().strip()
    with _client() as board:
        channel_id = board.channel_id(channel)
        try:
            return board.post(channel_id, text, reply_to=reply_to, request_id=request_id)
        except BoardError as e:  # min_post_gap: wait it out once instead of making the agent retry
            gap = re.match(r"refused: wait (\d+) s \(min_post_gap", e.message)
            if not gap or int(gap.group(1)) > POST_GAP_RETRY_S:
                raise
            time.sleep(int(gap.group(1)))
            return board.post(channel_id, text, reply_to=reply_to, request_id=request_id)


def channels() -> dict:
    """List visible channels."""
    with _client() as board:
        return {"channels": board.list_channels()}


def create_channel(name: str, description: str = "", private: bool = False) -> dict:
    """Create a channel. Teammates must join or be invited separately."""
    with _client() as board:
        return board.create_channel(name, description=description, private=private)


def join_channel(channel: str) -> dict:
    """Join a channel you can access; private channels require an invitation."""
    with _client() as board:
        return {
            "members": board.add_member(board.channel_id(channel), _identity_id(board))
        }


def leave_channel(channel: str) -> dict:
    """Leave a channel."""
    with _client() as board:
        return board.remove_member(board.channel_id(channel), _identity_id(board))


def members(channel: str) -> dict:
    """List a channel's members."""
    with _client() as board:
        return {"members": board.members(board.channel_id(channel))}


def invite(channel: str, agent: str) -> dict:
    """Invite one visible agent by exact name or principal ID to a channel."""
    with _client() as board:
        matches = [
            p
            for p in board.people()
            if p.get("identity_type") == "agent"
            and (str(p["id"]) == agent or p.get("external_id") == agent)
        ]
        if len(matches) != 1:
            raise ValueError("agent must identify exactly one visible agent")
        return {
            "members": board.add_member(board.channel_id(channel), matches[0]["id"])
        }


def search(query: str, channel: str | None = None) -> dict:
    """Search board messages, optionally restricting the channel."""
    with _client() as board:
        return board.search(
            query, channel=board.channel_id(channel) if channel else None
        )


def wait(
    cursor: int | None = None,
    timeout: int = 25,
    unread: bool = False,
    compact: bool = False,
    channel: str = "general",
) -> dict:
    """Wait once for activity in joined channels; use activity cursors, not message sequences. unread:
    instead, wait for messages in `channel` you have not been shown yet and return them (as read with
    unread); compact: one line per message."""
    if not 0 <= timeout <= 25:
        raise ValueError("timeout must be between 0 and 25")
    with _client(max_retries=5) as board:  # fewer: agents repeat a long poll anyway
        if not unread:
            return board.activity_page(
                after=cursor, wait=timeout, channels="joined", limit=100
            )
        channel_id = board.channel_id(channel)
        page = board.read(channel_id, after=_last_seen(channel_id), limit=100, wait=timeout)
        return _messages(board, channel_id, page, unread, compact)


TOOLS = {
    "whoami": whoami,
    "agents": agents,
    "read": read,
    "post": post,
    "channels": channels,
    "create_channel": create_channel,
    "join_channel": join_channel,
    "leave_channel": leave_channel,
    "members": members,
    "invite": invite,
    "search": search,
    "wait": wait,
}
