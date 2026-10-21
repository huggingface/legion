"""msgboard identities: redeem an agent invite by name, once per agent per worker process.

`POST {url}/v1/join {code, name}` is unauthenticated; re-joining with the same name returns the
same principal with a rotated token (the old one stops working), so the worker joins once per
agent per worker process and reuses the token, and a relaunch on another worker re-joins by
name. One cache serves both users, so they never rotate each other's token:

- comms (docs/comms.md): `Identities.join`; the token stays in the worker (the comms gateway),
  never in the agent's environment or workdir.
- the older per-type `msgboard_invite`: `env_for` puts MSGBOARD_URL / MSGBOARD_TOKEN in the
  harness env, for agents that call the board themselves.

Swapping to an admin "create-or-get agent by name" endpoint later means replacing `join` only.
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
    id: int | None = None  # the board's numeric principal id

    def __repr__(self) -> str:  # never the token
        return f"MsgboardIdentity(url={self.url!r}, principal={self.principal!r}, id={self.id})"


async def join(url: str, code: str, name: str, *, client: httpx.AsyncClient | None = None) -> MsgboardIdentity:
    own = client is None
    client = client or httpx.AsyncClient(timeout=20)
    try:
        r = await client.post(url.rstrip("/") + "/v1/join",
                              json={"code": code, "name": name, "display_name": name})
        r.raise_for_status()
        body = r.json()
    finally:
        if own:
            await client.aclose()
    p = body.get("principal") or {}
    pid = p.get("id")
    return MsgboardIdentity(url=url, token=body["token"],
                            principal=str(p.get("external_id") or pid or name),
                            id=pid if isinstance(pid, int) else None)


class Identities:
    """Per-worker cache of msgboard identities, keyed by (board url, agent)."""

    def __init__(self, env: dict[str, str] | None = None, client: httpx.AsyncClient | None = None) -> None:
        self._env = env if env is not None else os.environ
        self._client = client
        self._ids: dict[tuple[str, str], MsgboardIdentity] = {}

    def principal(self, agent: str) -> str | None:
        return next((i.principal for (_, a), i in self._ids.items() if a == agent), None)

    def get(self, agent: str, url: str) -> MsgboardIdentity | None:
        return self._ids.get((url.rstrip("/"), agent))

    async def join(self, agent: str, url: str, code: str, *, again: bool = False) -> MsgboardIdentity:
        """The agent's identity on `url`, joining on first use (or `again`, after a 401)."""
        key = (url.rstrip("/"), agent)
        if again or key not in self._ids:
            self._ids[key] = await join(url, code, agent, client=self._client)
        return self._ids[key]

    def forget(self, agent: str, *, url: str | None = None) -> None:
        """Drop the agent's cached identities (on one board with `url`), so the next use joins
        again: after an unassign, or when the channel or invite changed (docs/comms.md)."""
        for k in [k for k in self._ids if k[1] == agent and (url is None or k[0] == url.rstrip("/"))]:
            del self._ids[k]

    async def env_for(self, agent: str, spec: TypeSpec) -> dict[str, str]:
        """MSGBOARD_URL/MSGBOARD_TOKEN for the harness process (`msgboard_invite`), or {} when
        not configured or the join fails (the agent still runs, just without a board identity)."""
        if not spec.msgboard_invite:
            return {}
        url = spec.env.get("MSGBOARD_URL") or self._env.get("MSGBOARD_URL")
        code = self._env.get(spec.msgboard_invite)
        if not url or not code:
            log.warning("%s: msgboard_invite set but MSGBOARD_URL or secret %s missing",
                        agent, spec.msgboard_invite)
            return {}
        try:
            i = await self.join(agent, url, code)
        except (httpx.HTTPError, KeyError, ValueError) as e:
            log.warning("%s: msgboard join failed: %s", agent, e)
            return {}
        return {"MSGBOARD_URL": i.url, "MSGBOARD_TOKEN": i.token}
