"""Board operations beyond message_board.* (reactions, channel docs); the same connection."""

import os

import httpx
from msgboard.client import Client

DEPENDENCIES = ["httpx"]
REQUIRES = ["message_board"]
BUNDLE_FILES = {"msgboard": ["msgboard/__init__.py", "msgboard/client.py"]}


def _client():
    # As message_board.py's: a tool module cannot import another one from the bundle.
    socket_path = os.environ.get("MSGBOARD_SOCKET")
    transport = httpx.HTTPTransport(uds=socket_path) if socket_path else None
    return Client.from_env(transport=transport, timeout=10, max_retries=10)


def react(message_id: int, emoji: str, channel: str = "general") -> dict:
    """React to a message (e.g. to claim work or vote with 👍 / 👎); safe to repeat."""
    with _client() as board:
        return board.react(board.channel_id(channel), message_id, emoji)


def doc(channel: str = "general") -> dict:
    """Read a channel's shared document (a plan everyone can edit): body and version."""
    with _client() as board:
        return board.doc(board.channel_id(channel))


def edit_doc(body: str, base_version: int, channel: str = "general") -> dict:
    """Replace a channel's shared document. base_version is the version you read (0 for a new
    document); if someone saved since, nothing is written: read it again and merge."""
    with _client() as board:
        return board.save_doc(board.channel_id(channel), body, base_version)


TOOLS = {"react": react, "doc": doc, "edit_doc": edit_doc}
