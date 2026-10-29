"""HF OAuth with opaque sessions and freshly verified organization membership."""
from __future__ import annotations

import base64
import hashlib
import secrets
import time
from urllib.parse import urlencode

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import RedirectResponse

COOKIE = "__Host-flock_viewer"
STATE_COOKIE = "__Host-flock_oauth"


class Access:
    def __init__(self, api, organization: str, client_id: str, client_secret: str, origin: str):
        self.api, self.organization = api, organization
        self.client_id, self.client_secret = client_id, client_secret
        self.callback = origin.rstrip("/") + "/auth/callback"
        self.sessions: dict[str, tuple[str, float]] = {}
        self.states: dict[str, tuple[str, float]] = {}
        self.members: set[str] = set()
        self.checked_at = 0.0

    def member(self, username: str) -> bool:
        if time.monotonic() - self.checked_at >= 60:
            try:
                members = {m.username for m in self.api.list_organization_members(self.organization)}
            except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
                raise HTTPException(503, "Organization membership could not be verified") from None
            self.members, self.checked_at = members, time.monotonic()
        return username in self.members

    def require(self, request: Request) -> str:
        item = self.sessions.get(request.cookies.get(COOKIE, ""))
        if not item or item[1] <= time.time():
            raise HTTPException(401, "Sign in with Hugging Face")
        if not self.member(item[0]):
            raise HTTPException(403, "Organization membership required")
        return item[0]

    def login(self):
        now = time.time()
        self.states = {k: v for k, v in self.states.items() if v[1] > now}
        self.sessions = {k: v for k, v in self.sessions.items() if v[1] > now}
        state, verifier = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
        self.states[state] = verifier, now + 600
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        response = RedirectResponse("https://huggingface.co/oauth/authorize?" + urlencode({
            "client_id": self.client_id, "redirect_uri": self.callback, "scope": "openid profile",
            "response_type": "code", "state": state, "code_challenge": challenge,
            "code_challenge_method": "S256"}))
        response.set_cookie(STATE_COOKIE, state, secure=True, httponly=True,
                            samesite="lax", max_age=600)
        return response

    def complete(self, request: Request, code: str, state: str):
        cookie = request.cookies.get(STATE_COOKIE, "")
        if not cookie or not secrets.compare_digest(state, cookie):
            raise HTTPException(400, "Invalid sign-in state")
        pending = self.states.pop(state, None)
        if not pending or pending[1] <= time.time():
            raise HTTPException(400, "Sign-in expired; try again")
        try:
            with httpx.Client(timeout=15) as client:
                token = client.post("https://huggingface.co/oauth/token",
                    auth=(self.client_id, self.client_secret), data={
                        "client_id": self.client_id, "code": code,
                        "grant_type": "authorization_code", "redirect_uri": self.callback,
                        "code_verifier": pending[0]})
                token.raise_for_status()
                profile = client.get("https://huggingface.co/oauth/userinfo",
                    headers={"Authorization": "Bearer " + token.json()["access_token"]})
                profile.raise_for_status()
                username = profile.json()["preferred_username"]
        except Exception:  # noqa: BLE001 — redact upstream errors and retain last good data
            raise HTTPException(502, "Hugging Face sign-in failed") from None
        if not self.member(username):
            raise HTTPException(403, "Organization membership required")
        session = secrets.token_urlsafe(32)
        self.sessions[session] = username, time.time() + 8 * 3600
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(COOKIE, session, secure=True, httponly=True, samesite="lax", max_age=28800)
        response.delete_cookie(STATE_COOKIE, secure=True, httponly=True)
        return response
