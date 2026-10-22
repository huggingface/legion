"""Controller side of comms (docs/comms.md): resolve the channel once at apply, post the kickoff
once as the swarm's host identity.

The host `<swarm>-host` joins with the same invite as the agents (the invite secret must be in
the controller's environment, like the provider keys it uses for preflight; it is also what the
controller hands to the workers as a job secret). Its token is used for this apply only and kept
nowhere. A re-apply with the same board and channel copies `channel_id` and `kickoff_id` from
the stored spec and makes no board call; a kickoff is never posted twice (and its
Idempotency-Key, `flock-kickoff-<swarm>-<channel id>`, makes even a lost record safe).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import httpx

from flock.comms.board import BoardError, SyncBoard
from flock.models import SwarmSpec

log = logging.getLogger(__name__)


class CommsError(ValueError):
    """Why the comms block cannot be applied (the message names the field)."""


def host_name(swarm: str) -> str:
    return f"{swarm}-host"


def resolve(spec: SwarmSpec, previous: SwarmSpec | None, secrets: Mapping[str, str], *,
            transport: httpx.BaseTransport | None = None) -> None:
    """Fill `spec.comms.channel_id` (and `kickoff_id` once the kickoff is posted). Raises
    CommsError. Values a client sent for either field are replaced."""
    c = spec.comms
    if c is None:
        return
    c.channel_id = c.kickoff_id = None
    if c.kickoff_file and c.kickoff is None:
        raise CommsError(f"comms.kickoff_file {c.kickoff_file!r} was not inlined: apply the spec with "
                         "`flock apply`, which reads it next to the spec")
    if c.protocols and len(c.protocols_text) != len(c.protocols):
        raise CommsError("comms.protocols were not inlined: apply the spec with `flock apply`, "
                         "which reads them next to the spec")
    p = previous.comms if previous is not None else None
    if p is not None and p.channel_id is not None and (p.url, p.channel) == (c.url, c.channel):
        c.channel_id, c.kickoff_id = p.channel_id, p.kickoff_id
    want_kickoff = bool(c.kickoff and c.kickoff.strip()) and c.kickoff_id is None
    if c.channel_id is not None and not want_kickoff:
        return
    code = secrets.get(c.invite_env)
    if not code:
        raise CommsError(f"comms.invite_env: the secret {c.invite_env} is not set on the controller "
                         "(it joins the host identity and resolves the channel with it)")
    board = SyncBoard(c.url, transport=transport)
    try:
        joined = board.join(code, host_name(spec.swarm))
        token = joined["token"]
        chans = board.channels(token, c.channel)
        if not chans:
            raise CommsError(f"comms.channel: no channel #{c.channel} visible to "
                             f"{host_name(spec.swarm)} on {c.url} (is the invite scoped to it?)")
        ch = chans[0]
        if ch.get("archived"):  # refused before the host joins it or the kickoff is posted
            raise CommsError(f"comms.channel: channel {c.channel} (id {ch['id']}) is archived on "
                             f"{c.url}; unarchive it or use another channel")
        pid = (joined.get("principal") or {}).get("id")
        if not ch.get("member") and isinstance(pid, int):
            try:
                board.add_member(token, int(ch["id"]), pid)
            except BoardError as e:
                log.warning("comms: %s could not join #%s: %s", host_name(spec.swarm), c.channel, e)
        c.channel_id = int(ch["id"])
        if want_kickoff:
            assert c.kickoff is not None
            msg = board.post(token, c.channel_id, c.kickoff,
                             key=f"flock-kickoff-{spec.swarm}-{c.channel_id}")
            c.kickoff_id = int(msg["id"])
            log.info("comms: kickoff posted to #%s as message %s", c.channel, c.kickoff_id)
    except BoardError as e:
        raise CommsError(f"comms: {e}") from None
    except (KeyError, TypeError, ValueError) as e:
        if isinstance(e, CommsError):
            raise
        raise CommsError(f"comms: unexpected reply from the board ({type(e).__name__})") from None
    finally:
        board.close()
