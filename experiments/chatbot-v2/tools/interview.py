"""spawn_interviewer: a team agent spawns a short-lived interviewer child (flock spawn,
docs/spawn.md) through the controller API, as itself (FLOCK_URL, FLOCK_AGENT_TOKEN in its env),
for a booking made first with the world's book_interview. A folder tool cannot call the world
(the world secret never reaches tools), so the world does the quota check, the reservation and
the frozen copy at booking time, and this tool only spawns: the booking id names the level
(`ds-a-0001-iv2-strong`), and the interviewer's chat and submit_verdict take the booking id. A
booking the interviewer never uses is released by the world. The interviewer types are `ephemeral`:
flock removes each one when its session ends. The child is spawned without a prompt and
prompted once its worker reports it (a prompt sent with the spawn can be dropped)."""

import json
import re
import time

import httpx

from flock.tools import Context, ToolError

READY_S = 45  # wait for the child's worker (the tool call times out at 60 s)
PER_RUN = 12  # interviewers one team agent may spawn in a run (live ones: spec spawn.<type>.max)
PROMPT = ("Interview booking {booking} for {parent} (team {team}). When you have filed your "
          "verdict, post it in thread {thread}.")


def spawn_interviewer(booking: str, thread: str, ctx: Context) -> dict:
    """Spawn the interviewer for a booking from book_interview (call that first: it checks the
    quota, reserves it and freezes the bot). The interviewer chats with the frozen bot, files
    "agent" or "script" with its evidence, and posts it in `thread`.

    Args:
        booking: the booking id book_interview returned, e.g. ds-a-0001-iv2-strong
        thread: the board thread id for the interviewer's verdict (your team thread's root id)
    """
    env, m = ctx.env, re.fullmatch(r"[a-z0-9-]+-iv\d+-(mid|strong)", booking.strip())
    if not m:
        raise ToolError(f"refused: {booking!r} is not a booking id; call book_interview first")
    if not env.get("FLOCK_URL") or not env.get("FLOCK_AGENT_TOKEN"):
        raise ToolError("no agent token: this agent cannot spawn")
    count = ctx.data_dir / f"interviewers-{ctx.agent}.json"
    n = json.loads(count.read_text()) if count.is_file() else 0
    if n >= PER_RUN:
        raise ToolError(f"refused: you spawned {n} interviewers, the limit for this run")
    with httpx.Client(base_url=env["FLOCK_URL"], timeout=30,
                      headers={"X-Flock-Token": env["FLOCK_AGENT_TOKEN"]}) as http:
        prompt = (PROMPT.replace("{booking}", booking.strip()).replace("{parent}", ctx.agent)
                  .replace("{team}", ctx.agent.split("-")[1].upper())
                  .replace("{thread}", str(thread)))
        r = http.post(f"/agents/{ctx.agent}/spawn", json={"type": f"interviewer-{m.group(1)}"})
        if r.status_code >= 400:
            detail = r.json() if r.headers.get("content-type", "").startswith(
                "application/json") else r.text
            raise ToolError(f"spawn refused ({r.status_code}): {detail}")
        count.write_text(json.dumps(n + 1))
        child, end = r.json()["name"], time.monotonic() + READY_S
        while not http.get(f"/agents/{child}").json().get("last_heartbeat"):
            if time.monotonic() > end:
                http.delete(f"/agents/{child}")
                raise ToolError(f"{child} did not start within {READY_S} s; removed it")
            time.sleep(1)
        http.post(f"/agents/{child}/prompt", json={"text": prompt}).raise_for_status()
    return {"spawned": child, "booking": booking.strip(), "left_this_run": PER_RUN - n - 1}
