"""Child agents through flock's controller (docs/spawn.md): spawn one, prompt an agent, read
an agent's status, remove a child. Each call goes through SPAWN_SOCKET (resources.spawn), a
worker proxy that sends it as you, with your own token; the controller decides what you may do.
"""

import json
import os
import socket

REQUIRES = ["spawn"]


def _call(method, path, body=None):
    with socket.socket(socket.AF_UNIX) as s:
        s.settimeout(180)
        s.connect(os.environ["SPAWN_SOCKET"])
        s.sendall(json.dumps({"method": method, "path": path, "json": body}).encode() + b"\n")
        line = s.makefile("rb").readline()
    out = json.loads(line)
    if "error" in out:
        raise RuntimeError(out["error"])
    status, body = out["status"], out["body"]
    if status >= 400:
        if isinstance(body, dict) and "reason" in body:  # a spawn refusal (409)
            raise RuntimeError(f"{body['reason']}: {body.get('detail')}")
        detail = body.get("detail") if isinstance(body, dict) else body
        raise RuntimeError(f"HTTP {status}: {detail}")
    return body


def _me():
    return os.environ["SPAWN_AGENT"]


def spawn(type: str, prompt: str | None = None, key: str | None = None) -> dict:
    """Create a child agent of `type` (your spawn rules say which types and how many); it
    starts on `prompt`. Reusing a `key` returns the same child, so a retry is safe. Returns
    the child's `name`. A refusal names the reason: max_children (remove a finished child
    first), depth, type_not_allowed, or a swarm limit (alive, budget, workers, spawned_total:
    stop spawning)."""
    return _call("POST", f"/agents/{_me()}/spawn", {"type": type, "prompt": prompt, "key": key})


def prompt(agent: str, text: str) -> dict:
    """Send `text` to one of your children (or theirs) or to your parent; it arrives as their
    next prompt. Children report back to their parent this way."""
    return _call("POST", f"/agents/{agent}/prompt", {"text": text})


def status(agent: str | None = None) -> dict:
    """An agent's status: yours by default (`parent`, `children`, `depth`), or a child's
    (`state`, turns, cost)."""
    return _call("GET", f"/agents/{agent or _me()}")


def remove(agent: str) -> dict:
    """Remove one of your children and everything it spawned; frees its place under your cap."""
    return _call("DELETE", f"/agents/{agent}")


TOOLS = {"spawn": spawn, "prompt": prompt, "status": status, "remove": remove}
