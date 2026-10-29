"""Who is calling. Agent identity comes from the token.

Token sources (first match wins): ``X-Canvas-Token`` header, then ``Authorization: Bearer``.
``X-Canvas-Token`` exists because behind the HF jobs proxy ``Authorization`` carries the
caller's HF token, which the proxy checks; the canvas token then rides in its own header.

Modes (both can be on at once):
  A  shared secret  CANVAS_TOKEN; the agent name comes from the ``X-Flock-Agent`` header.
  B  per-agent      tokens minted by ``POST /agents/register {name}`` with CANVAS_ADMIN_TOKEN;
                    the token *is* the identity (``X-Flock-Agent`` is ignored).
If neither CANVAS_TOKEN nor CANVAS_ADMIN_TOKEN is set the service is open (dev only):
the name comes from ``X-Flock-Agent``.
"""

import hashlib
import hmac
import json
import re
import secrets
import threading
from pathlib import Path

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,63}$")
AGENTS_FILE = "agents.json"


class AuthError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status
        self.msg = msg


def _h(tok: str) -> str:
    return hashlib.sha256(tok.encode()).hexdigest()


def token_from_headers(headers) -> str | None:
    tok = headers.get("x-canvas-token")
    if tok:
        return tok.strip()
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


class Auth:
    def __init__(self, shared: str | None, admin: str | None, data_dir: Path | None):
        self.shared = shared
        self.admin = admin
        self.data_dir = data_dir
        self.lock = threading.Lock()
        self.by_hash: dict[str, str] = {}  # sha256(token) -> name (tokens never stored raw)
        if data_dir and (data_dir / AGENTS_FILE).exists():
            self.by_hash = json.loads((data_dir / AGENTS_FILE).read_text())

    @property
    def open(self) -> bool:
        return not self.shared and not self.admin

    def register(self, headers, name: str) -> str:
        tok = token_from_headers(headers)
        if not self.admin:
            raise AuthError(403, "registration disabled: server has no CANVAS_ADMIN_TOKEN")
        if not tok or not hmac.compare_digest(tok, self.admin):
            raise AuthError(401, "admin token required to register agents")
        if not NAME_RE.match(name or ""):
            raise AuthError(400, "bad agent name: use 1-64 chars of [A-Za-z0-9._:@-]")
        new = "cv_" + secrets.token_urlsafe(24)
        with self.lock:
            # Re-registering a name rotates its token.
            self.by_hash = {h: n for h, n in self.by_hash.items() if n != name}
            self.by_hash[_h(new)] = name
            if self.data_dir:
                tmp = self.data_dir / (AGENTS_FILE + ".tmp")
                tmp.write_text(json.dumps(self.by_hash))
                tmp.replace(self.data_dir / AGENTS_FILE)
        return new

    def identify(self, headers) -> str:
        tok = token_from_headers(headers)
        if tok:
            name = self.by_hash.get(_h(tok))
            if name:
                return name
            if self.shared and hmac.compare_digest(tok, self.shared):
                return self._header_name(headers)
            if self.open:
                return self._header_name(headers)
            raise AuthError(401, "unknown token: check CANVAS_TOKEN")
        if self.open:
            return self._header_name(headers)
        raise AuthError(401, "missing token: set CANVAS_TOKEN (sent as Bearer or X-Canvas-Token)")

    @staticmethod
    def _header_name(headers) -> str:
        name = headers.get("x-flock-agent", "").strip()
        if not name:
            raise AuthError(400, "missing agent name: set FLOCK_AGENT (X-Flock-Agent header)")
        if not NAME_RE.match(name):
            raise AuthError(400, "bad agent name: use 1-64 chars of [A-Za-z0-9._:@-]")
        return name

    def check_read(self, headers) -> None:
        """Reads need *some* valid token when CANVAS_READ_AUTH=1."""
        tok = token_from_headers(headers)
        if self.open:
            return
        if tok and (
            _h(tok) in self.by_hash
            or (self.shared and hmac.compare_digest(tok, self.shared))
            or (self.admin and hmac.compare_digest(tok, self.admin))
        ):
            return
        raise AuthError(401, "read auth on: send CANVAS_TOKEN")
