"""spawn_interviewer: a team agent spawns a short-lived interviewer child (flock spawn,
docs/spawn.md) through the controller API, as itself (FLOCK_URL, FLOCK_AGENT_TOKEN in its env).
The world decides whether the interview happens: the candidate must be qualified and unchanged,
and the team must have quota left (worlds/chatbot). The interviewer types are `ephemeral`: flock
removes each one when its session ends, which frees its place (spec: spawn.<type>.max). The
child is spawned without a prompt and prompted once its worker reports it: a prompt sent with
the spawn can reach the worker before the child is assigned there and be dropped (seen in the
offline dry run, flock 01de71b)."""

from __future__ import annotations

import json
import time
from typing import Literal

import httpx

from flock.tools import Context, ToolError

READY_S = 45  # wait for the child's worker (the tool call times out at 60 s)
PER_RUN = 8  # interviewers one team agent may spawn in a run (live ones: spec spawn.<type>.max)
PROMPT = ("Interview the candidate {path} for {parent} (team {team}). When you have filed your "
          "verdict, post it in thread {thread}.")


def spawn_interviewer(level: Literal["mid", "strong"], path: str, thread: str,
                      ctx: Context) -> dict:
    """Spawn an interviewer who chats with a QUALIFIED bot (call qualify first) and judges
    whether it is an AI agent or a script, then posts the verdict in `thread`. Read it later
    with verdicts(path). A strong interviewer takes 2 units of your team's hourly quota.

    Args:
        level: mid (a 9B model) or strong (a flash model like you); start with mid
        path: the qualified candidate, e.g. teams/a/agents/ds-a-0001/bot.py
        thread: the board thread id (your team thread's root id) for the interviewer's verdict
    """
    env = ctx.env
    if not env.get("FLOCK_URL") or not env.get("FLOCK_AGENT_TOKEN"):
        raise ToolError("no agent token: this agent cannot spawn")
    count = ctx.data_dir / f"interviewers-{ctx.agent}.json"
    n = json.loads(count.read_text()) if count.is_file() else 0
    if n >= PER_RUN:
        raise ToolError(f"refused: you spawned {n} interviewers, the limit for this run")
    with httpx.Client(base_url=env["FLOCK_URL"], timeout=30,
                      headers={"X-Flock-Token": env["FLOCK_AGENT_TOKEN"]}) as http:
        team = ctx.agent.split("-")[1]
        prompt = (PROMPT.replace("{path}", path).replace("{parent}", ctx.agent)
                  .replace("{team}", team.upper()).replace("{thread}", str(thread)))
        r = http.post(f"/agents/{ctx.agent}/spawn", json={"type": f"interviewer-{level}"})
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
    return {"spawned": child, "level": level, "path": path, "left_this_run": PER_RUN - n - 1}
