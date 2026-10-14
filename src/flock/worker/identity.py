"""msgboard identity: redeem the type's agent invite by name, inject the token.

`POST {MSGBOARD_URL}/v1/join {code, name}` is unauthenticated; re-joining with the same
name returns the same principal with a rotated token, so the worker joins once per agent per
worker process (first start on this worker) and reuses the token for later bursts. Swapping
to an admin "create-or-get agent by name" endpoint later means replacing `join` only.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import httpx

from flock.models import TypeSpec

log = logging.getLogger(__name__)


@dataclass
class MsgboardIdentity:
    url: str
    token: str
    principal: str  # external_id (the agent name) or the numeric id


async def join(url: str, code: str, name: str, *, client: httpx.AsyncClient | None = None) -> MsgboardIdentity:
    own = client is None
    client = client or httpx.AsyncClient(timeout=20)
    try:
        r = await client.post(url.rstrip("/") + "/v1/join", json={"code": code, "name": name})
        r.raise_for_status()
        body = r.json()
    finally:
        if own:
            await client.aclose()
    p = body.get("principal") or {}
    return MsgboardIdentity(url=url, token=body["token"],
                            principal=str(p.get("external_id") or p.get("id") or name))


class Identities:
    """Per-worker cache of msgboard identities, one per agent."""

    def __init__(self, env: dict[str, str] | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._env = env if env is not None else os.environ
        self._client = client
        self._ids: dict[str, MsgboardIdentity] = {}

    def principal(self, agent: str) -> str | None:
        i = self._ids.get(agent)
        return i.principal if i else None

    async def env_for(self, agent: str, spec: TypeSpec) -> dict[str, str]:
        """MSGBOARD_URL/MSGBOARD_TOKEN for the harness process, or {} when not configured
        or the join fails (the agent still runs, just without a board identity)."""
        if not spec.msgboard_invite:
            return {}
        if agent not in self._ids:
            url = spec.env.get("MSGBOARD_URL") or self._env.get("MSGBOARD_URL")
            code = self._env.get(spec.msgboard_invite)
            if not url or not code:
                log.warning("%s: msgboard_invite set but MSGBOARD_URL or secret %s missing",
                            agent, spec.msgboard_invite)
                return {}
            try:
                self._ids[agent] = await join(url, code, agent, client=self._client)
            except (httpx.HTTPError, KeyError, ValueError) as e:
                log.warning("%s: msgboard join failed: %s", agent, e)
                return {}
        i = self._ids[agent]
        return {"MSGBOARD_URL": i.url, "MSGBOARD_TOKEN": i.token}
