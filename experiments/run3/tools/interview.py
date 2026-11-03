"""spawn_interviewer: an agent spawns a short-lived strong interviewer child (flock-agent spawn,
docs/spawn.md) through the controller API, as itself (FLOCK_URL, FLOCK_AGENT_TOKEN in its env),
for a booking made first with the world's book_interview. A folder tool cannot call the world
(the world secret never reaches tools), so the world freezes the copy and checks the bar at
booking time, and this tool only spawns. There is no quota (run 3): the spec's `spawn` rule
caps the live interviewers per agent. The `interviewer` type is `ephemeral`: flock removes
each one when its session ends. The child is spawned without a prompt and prompted once its
worker reports it (a prompt sent with the spawn can be dropped)."""

from __future__ import annotations

import re
import time

import httpx

from flock.tools import Context, ToolError

READY_S = 45  # wait for the child's worker (the tool call times out at 60 s)
PROMPT = ("Interview booking {booking} for {parent}. When you have filed your verdict, post it "
          "in thread {thread}.")


def spawn_interviewer(booking: str, thread: str, ctx: Context) -> dict:
    """Spawn the strong interviewer for a booking from book_interview (call that first: it
    freezes the bot and checks the bar). The interviewer chats with the frozen bot, files
    "agent" or "script" with its evidence, and posts it in `thread`.

    Args:
        booking: the booking id book_interview returned, e.g. agent-0001-iv2-strong
        thread: the board thread id for the interviewer's verdict (your thread's root id)
    """
    env = ctx.env
    if not re.fullmatch(r"[a-z0-9-]+-iv\d+-strong", booking.strip()):
        raise ToolError(f"refused: {booking!r} is not a booking id; call book_interview first")
    if not env.get("FLOCK_URL") or not env.get("FLOCK_AGENT_TOKEN"):
        raise ToolError("no agent token: this agent cannot spawn")
    with httpx.Client(base_url=env["FLOCK_URL"], timeout=30,
                      headers={"X-Flock-Token": env["FLOCK_AGENT_TOKEN"]}) as http:
        prompt = (PROMPT.replace("{booking}", booking.strip()).replace("{parent}", ctx.agent)
                  .replace("{thread}", str(thread)))
        r = http.post(f"/agents/{ctx.agent}/spawn", json={"type": "interviewer"})
        if r.status_code >= 400:
            detail = r.json() if r.headers.get("content-type", "").startswith(
                "application/json") else r.text
            raise ToolError(f"spawn refused ({r.status_code}): {detail}")
        child, end = r.json()["name"], time.monotonic() + READY_S
        while not http.get(f"/agents/{child}").json().get("last_heartbeat"):
            if time.monotonic() > end:
                http.delete(f"/agents/{child}")
                raise ToolError(f"{child} did not start within {READY_S} s; removed it")
            time.sleep(1)
        http.post(f"/agents/{child}/prompt", json={"text": prompt}).raise_for_status()
    return {"spawned": child, "booking": booking.strip()}
